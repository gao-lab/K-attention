import os
from abc import abstractmethod
from logging import getLogger
from typing import Optional, Literal, Any
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    pass

from .import_utils import is_flash_attn_2_available

# Note
if is_flash_attn_2_available():
    from .utils import apply_rotary_emb_qkv_

logger = getLogger(__name__)


# design choices
# 1. cache or not
# 2. whether support kv cache
# 3. input to relative embedding? q, k, or attention matrix

LEGAL_POSITION_EMBEDDING = Literal[
    "absolute_learned",
    "absolute_sinusoidal",
    "absolute_sinusoidal_learned",

    "relative_RoPE",
    "relative_ALiBi",
    "relative_learned",
    "relative_learned_T5",
]

####################################################
# ========== Relative Position Embedding ==========
####################################################

class RelativePositionEmbedding(nn.Module):
    def __init__(self, apply_on: Literal["qk", "attn_mtx"]):
        super().__init__()
        self.apply_on = apply_on

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class RotaryEmbedding(RelativePositionEmbedding):
    def __init__(
        self,
        head_dim: int,
        max_seqlen: int = 4096,
        base: int = 10_000,
        scaling_factor: float = 1.0,
        device: Optional[torch.device | str] = None,
        use_FA_triton_kernel: bool = False,
        num_posids: Optional[int] = None,
        full_precision: bool = False,
        learnable: bool = False,
        dynamic: bool = False,
        max_warns: int = 5,
        forward_mode: Literal["qk", "qk_posids", "qkv_inplace"] = "qk",
    ):
        r"""
        For general purpose RoPE, num_posids is 1 or None, for GLM like models with 2 position
        indices, num_posids = 2

        When applying RoPE, we need to compute sin/cos vector for each input sequence, (this is
        an Embedding index operation, maybe expensive?), which may vary when using incontinuous
        position ids (like in GLM models).

        RoPE, adapted from:
        - huggingface transformers: models/llama/modeling_llama.py
        - SwissArmyTransformer: sat/model/official/glm130B_model.py
        - flash-attention: flash_attn/layers/rotary.py
        - Megatron-LM: megatron/core/models/common/embeddings/rotary_pos_embedding.py

        Parameters
        --------------
        use_FA_triton_kernel: bool
            Whether to use the FA Triton kernel (fused) for the operation, only applicable for start_pos
            or cu_seqlens.
        num_posids: int
            Number of position indices. 2 for GLM like models
        full_precision: bool
            Whether to apply RoPE on full precision vectors. Note: inplace operation
            is not supported for full precision.
        dynamic: bool
            Whether max_seqlen limit can be dynamically extended.
        """
        super().__init__(apply_on="qk")
        # TODO: learnable, check SwissArmyTransformer/sat/model/position_embedding/rotary_embedding.py
        # TODO: multiple posids, for GLM like models
        # TODO: RoPE that supports context extension
        if num_posids is None:
            num_posids = 1
        self.num_posids = num_posids

        assert head_dim % num_posids == 0, \
            f"head_dim {head_dim} should be divisible by num_posids {num_posids}"
        self.dim = head_dim // self.num_posids
        self.max_seqlen = max_seqlen
        self.base = base
        self.scaling_factor = scaling_factor
        self.device = device
        self.full_precision = full_precision
        self.use_FA_triton_kernel = use_FA_triton_kernel
        self.dynamic = dynamic
        self.max_warns = max_warns

        if self.use_FA_triton_kernel:
            assert is_flash_attn_2_available(), "flash-attn 2 is required for using FA Triton kernel"
        # We still need to define and use forward function, rather than use the
        # customized forward functions directly, as it's needed for forward hooks
        # and thus DDPs, etc.
        if forward_mode == "qk":
            self.forward = self.q_or_k_forward
        elif forward_mode == "qk_posids":
            self.forward = self.q_or_k_forward_by_pos_ids
        elif forward_mode == "qkv_inplace":
            self.forward = self.qkv_packed_forward_

        self._update_core_cache(max_seqlen)
        self.to(self.device)

    def __repr__(self):
        return (
            f"RotaryEmbedding(dim={self.dim}, max_seqlen={self.max_seqlen}, base={self.base}, "
            f"scaling_factor={self.scaling_factor}, num_posids={self.num_posids}, "
            f"full_precision={self.full_precision}, use_FA_triton_kernel={self.use_FA_triton_kernel})"
        )

    def _update_core_cache(self, max_seqlen: Optional[int] = None):
        # Different from original paper, uses a different permutation in order to obtain the same calculation
        # check implementation in GPT-J and GPT-Neox for comparison

        # Note: use float32 to prevent models from using bfloat16 (which will round 1995.0 to 2000.0)
        # dim: [emb_dim // 2]
        if max_seqlen is None or max_seqlen < self.max_seqlen:
            return

        self.max_seqlen = max_seqlen
        wave_num = 1.0 / (
            self.base ** (
                torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32) / self.dim
            )
        )

        t = torch.arange(max_seqlen, device=self.device, dtype=torch.float32)
        t = t / self.scaling_factor
        # dim: [max_seqlen, emb_dim // 2]
        rotation_angles = torch.outer(t, wave_num)

        if self.use_FA_triton_kernel:
            assert is_flash_attn_2_available(), "flash-attn 2 is required for using FA Triton kernel"
            self.register_buffer("backup_cos", rotation_angles.cos(), persistent=False)
            self.register_buffer("backup_sin", rotation_angles.sin(), persistent=False)
        else:
            emb = torch.cat((rotation_angles, rotation_angles), dim=-1)
            # dim: [max_seqlen, emb_dim]
            self.register_buffer("backup_cos", emb.cos(), persistent=False)
            self.register_buffer("backup_sin", emb.sin(), persistent=False)

        # backup_cos/sin is the full precision version, cached is the autocast version
        self.register_buffer("cached_cos", self.backup_cos, persistent=False)
        self.register_buffer("cached_sin", self.backup_sin, persistent=False)

    def _update_cos_sin_cache(
        self, max_seqlen: Optional[int] = None, dtype: Optional[torch.dtype] = None
    ):
        if max_seqlen is not None and self.dynamic:
            self._update_core_cache(self, max_seqlen)

        if dtype is None or self.full_precision:
            self.cached_cos = self.backup_cos
            self.cached_sin = self.backup_sin
        elif self.cached_cos.dtype != dtype:
            self.cached_cos = self.backup_cos.to(dtype)
            self.cached_sin = self.backup_sin.to(dtype)

    @staticmethod
    def rotate_half(x: torch.Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        q_or_k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        inplace: bool = False
    ):
        """Applies Rotary Position Embedding to the query and key tensors.
        """
        # sin/cos: [batch_size, seq_len, emb_dim] or [total_len, emb_dim]
        # q, k: [batch_size, seq_len, num_heads, emb_dim] or [total_len, num_heads, emb_dim]
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)    # add num_heads dim
        if inplace:
            # q_or_k.copy_(q_or_k_embed)
            # in the following way, we can avoid half of mem copies (on view1 and view2)?
            cos = cos[..., q_or_k.shape[-1] // 2 :]
            sin = sin[..., q_or_k.shape[-1] // 2 :]
            v1 = q_or_k[..., : q_or_k.shape[-1] // 2]
            v2 = q_or_k[..., q_or_k.shape[-1] // 2 :]
            v1_c = v1.clone()
            v1.mul_(cos).sub_(v2 * sin)
            v2.mul_(cos).add_(v1_c * sin)

            return q_or_k
        else:
            q_or_k_embed = (q_or_k * cos) + (RotaryEmbedding.rotate_half(q_or_k) * sin)
            return q_or_k_embed

    def q_or_k_forward(
        self,
        q_or_k: torch.Tensor,
        seqlen_offsets: int | torch.Tensor = 0,

        # for flash-attn, q_or_k of shape: [total_seqlen, num_heads, head_dim]
        # pass cu_seqlens
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        precomputed_cos_sin_cache: Optional[tuple[torch.Tensor]] = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        r"""
        Parameters
        --------------
        q_or_k: torch.Tensor
            The query or key tensors. Shape [batch_size, seq_len, num_heads, head_dim] if not using
            flash attention, otherwise [total_seqlen, num_heads, head_dim]
        seqlen_offsets:
            assume the position ids are **contiguous** starting from seqlen_offsets in the batch (for kv cache).
        cu_seqlens: Optional[torch.Tensor]
            Tensor of shape (batch_size + 1,). Used for flash attention
        max_seqlen: Optional[int]
            The maximum sequence length. Used for flash attention
        precomputed_cos_sin_cache: Optional[tuple[torch.Tensor]]
            The precomputed cos and sin vectors.
        """
        # capture cache of current forward
        if precomputed_cos_sin_cache is not None:
            precomputed_cos, precomputed_sin = precomputed_cos_sin_cache
        else:
            precomputed_cos, precomputed_sin = None, None

        # update inner cos/sin cache if necessary
        if cu_seqlens is None:    # not using flash-attn
            if self.dynamic and max_seqlen is None:  # check dynamic here (redundant) to avoid unnecessary computation
                max_seqlen = q_or_k.size(1) + seqlen_offsets.max().item() \
                    if isinstance(seqlen_offsets, torch.Tensor) else \
                    q_or_k.size(1) + seqlen_offsets
            self._update_cos_sin_cache(max_seqlen, dtype=q_or_k.dtype)
        else:                     # flash attn
            assert max_seqlen is not None, "If using flash-attn, max_seqlen should be passed in"
            self._update_cos_sin_cache(max_seqlen, dtype=q_or_k.dtype)

        # apply RoPE
        if self.use_FA_triton_kernel:
            assert not self.full_precision, "FA triton kernel does not support full precision"
            q_or_k_embed = apply_rotary_emb(
                q_or_k, self.cached_cos, self.cached_sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        else:
            assert isinstance(seqlen_offsets, int), "if passing tensor seqlen_offsets, please use FA triton kernel"

            if precomputed_cos_sin_cache is None:
                if cu_seqlens is None:     # not using flash-attn
                    precomputed_cos = self.cached_cos[seqlen_offsets:seqlen_offsets + q_or_k.size(1)]
                    precomputed_sin = self.cached_sin[seqlen_offsets:seqlen_offsets + q_or_k.size(1)]
                else:                      # using flash-attn
                    seqlen_offsets = seqlen_offsets if seqlen_offsets is not None else 0
                    position_ids = torch.cat(
                        [
                            torch.arange(0, e - s - seqlen_offsets, device=q_or_k.device) \
                                for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:])
                        ],
                        dim=0
                    )  # [total_seqlen, ]
                    precomputed_cos = F.embedding(position_ids, self.cached_cos)
                    precomputed_sin = F.embedding(position_ids, self.cached_sin)

            q_or_k_embed = self.apply_rotary_pos_emb(
                q_or_k, precomputed_cos, precomputed_sin
            ).to(q_or_k.dtype)   # project back to original dtype if using full precision

        return q_or_k_embed, (precomputed_cos, precomputed_sin)

    def q_or_k_forward_by_pos_ids(
        self,
        q_or_k: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        precomputed_cos_sin_cache: Optional[torch.Tensor] = None,
    ):
        assert not self.use_FA_triton_kernel, "position_ids based indexing doesn't support FA triton kernel"

        # capture cache of current forward
        if precomputed_cos_sin_cache is not None:
            precomputed_cos, precomputed_sin = precomputed_cos_sin_cache
        else:
            precomputed_cos, precomputed_sin = None, None

        # update inner cos/sin cache if necessary
        max_seqlen = position_ids.max().item() + 1 if self.dynamic else None
        self._update_cos_sin_cache(max_seqlen, dtype=q_or_k.dtype)

        # apply RoPE
        if precomputed_cos_sin_cache is None:
            precomputed_cos = F.embedding(position_ids, self.cached_cos)
            precomputed_sin = F.embedding(position_ids, self.cached_sin)
        q_or_k_embed = self.apply_rotary_pos_emb(
            q_or_k, precomputed_cos, precomputed_sin
        ).to(q_or_k.dtype)   # project back to original dtype if using full precision

        return q_or_k_embed, (precomputed_cos, precomputed_sin)

    def qkv_packed_forward_(
        self,
        qkv: Optional[torch.Tensor] = None,
        seqlen_offsets: int | torch.Tensor = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        precomputed_cos_sin_cache: Optional[tuple[torch.Tensor]] = None,
        force_inplace: bool = False,     # for self implemented inplace (default will rebind qk with v)
    ):
        r"""
        Position ids are used to index the precomputed sin/cos vectors, which are then used to rotate
        the query and key tensors. Should be provided in one of the following ways:

        - cu_seqlens (max_seqlen should be passed in concurrently): Tensor of shape [batch_size + 1,],
            used for flash attention (only when q_or_k is passed as total_seqlen)

        Parameters
        --------------
        qkv: torch.Tensor
            The query or key tensors. Shape [batch_size, seq_len, 3, num_heads, head_dim] if not using
            flash attention, otherwise [total_seqlen, 3, num_heads, head_dim]

        Returns
        --------------
        q_embed, k_embed: torch.Tensor
            The rotated query and key tensors. Shape [batch_size, seq_len, 3, num_heads, head_dim] if not using
            flash attention, otherwise [total_seqlen, 3, num_heads, head_dim]
        precomputed_cos, precomputed_sin: torch.Tensor
            The precomputed cos and sin vectors.
        """
        assert not self.full_precision, "Inplace operation on qkv does not support full precision"
        if not self.use_FA_triton_kernel and self.max_warns > 0:
            logger.warning("Inplace version of RoPE should use_FA_triton_kernel, "
                           "otherwise there won't be any performance gain due to "
                           "memory copys.")
            self.max_warns -= 1

        # capture cache of current forward
        if precomputed_cos_sin_cache is not None:
            precomputed_cos, precomputed_sin = precomputed_cos_sin_cache
        else:
            precomputed_cos, precomputed_sin = None, None

        # update inner cos/sin cache if necessary
        if cu_seqlens is None:    # not using flash-attn
            if self.dynamic and max_seqlen is None:  # check dynamic here (redundant) to avoid unnecessary computation
                max_seqlen = qkv.size(1) + seqlen_offsets.max().item() \
                    if isinstance(seqlen_offsets, torch.Tensor) else \
                    qkv.size(1) + seqlen_offsets
            self._update_cos_sin_cache(max_seqlen, dtype=qkv.dtype)
        else:                     # flash attn
            assert max_seqlen is not None, "If using flash-attn, max_seqlen should be passed in"
            self._update_cos_sin_cache(max_seqlen, dtype=qkv.dtype)

        if self.use_FA_triton_kernel:
            # TODO: check can FA_triton_kernel support full_precision
            # assert not self.full_precision, "FA triton kernel does not support full precision"
            qkv_embed = apply_rotary_emb_qkv_(
                qkv, self.cached_cos, self.cached_sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
        else:
            assert isinstance(seqlen_offsets, int), "if passing tensor seqlen_offsets, please use FA triton kernel"

            if precomputed_cos_sin_cache is None:
                if cu_seqlens is None:     # not using flash-attn
                    precomputed_cos = self.cached_cos[seqlen_offsets:seqlen_offsets + qkv.size(1)]
                    precomputed_sin = self.cached_sin[seqlen_offsets:seqlen_offsets + qkv.size(1)]
                else:                      # using flash-attn
                    seqlen_offsets = seqlen_offsets if seqlen_offsets is not None else 0
                    position_ids = torch.cat(
                        [
                            torch.arange(0, e - s - seqlen_offsets, device=qkv.device) \
                                for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:])
                        ],
                        dim=0
                    )  # [total_seqlen, ]
                    precomputed_cos = F.embedding(position_ids, self.cached_cos)
                    precomputed_sin = F.embedding(position_ids, self.cached_sin)

            if cu_seqlens is None:
                qk = qkv[:, :, :2]
                v = qkv[:, :, 2:]
            else:
                qk = qkv[:, :2]
                v = qkv[:, 2:]
            qk_r = rearrange(qk, "... two h d -> ... (two h) d")
            assert qk_r.data_ptr() == qk.data_ptr()
            if force_inplace:
                qk_r = self.apply_rotary_pos_emb(
                    qk_r, precomputed_cos, precomputed_sin, inplace=True
                )
                qkv_embed = qkv
            else:
                qk_r = self.apply_rotary_pos_emb(
                    qk_r, precomputed_cos, precomputed_sin, inplace=False
                )
                qk_r = rearrange(qk_r, "... (two h) d -> ... two h d", two=2)
                qkv_embed = torch.cat(
                    [qk_r, v], dim=-3
                )

        return qkv_embed, (precomputed_cos, precomputed_sin)

    def qkv_packed_forward_by_pos_ids_(
        self,
        qkv: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        r"""
        For complicate cases, where position ids are not contiguous in input
        sequence, like in GLM. Pass position_ids.

        position_ids: a tensor of shape [batch_size, seq_len] or [seq_len,] containing the position ids.
        precomputed_cos, precomputed_sin: torch.Tensor
            The precomputed cosine and sine vectors.

        """
        assert not self.full_precision, "Inplace operation on qkv does not support full precision"
        assert not self.use_FA_triton_kernel, "position_ids based indexing doesn't support FA triton kernel"

        raise NotImplementedError(
            "As FA triton kernel only support consecutive position ids, complicated "
            "cases like that for GLM models doesn't support packed inplace RoPE, " \
            "as there's no performance gain. Please fall back to q_or_k non-inplace "
            "version."
        )


class AlibiEmbedding(RelativePositionEmbedding):
    def __init__(self, num_heads: int, max_seqlen: int = 4096, alibi_max_exp: int = 8,
                 is_causal: bool = False, device: Optional[torch.device | str] = None):
        r"""
        Parameters
        --------------
        num_heads: int
            The number of attention heads.
        max_seqlen: int
        alibi_max_exp: int
            The maximum exponent for the alibi matrix.
        is_causal: bool
            Whether the model is GPT-like (Causal). If True, we only construct one row of the alibi matrix,
            else we construct the whole matrix.
        """
        super().__init__(apply_on="attn_mtx")
        self.is_causal = is_causal
        alibi, slopes = self.build_alibi_mtx(
            num_heads, max_seqlen, alibi_max_exp, is_causal=is_causal, device=device)
        self.register_buffer("_alibi", alibi, persistent=False)
        self.register_buffer("_slopes", slopes, persistent=False)
        self.to(device)

    @property
    def slopes(self):
        return self._slopes

    @staticmethod
    def build_alibi_mtx(num_heads: int, max_seqlen: int, alibi_max_exp: int = 8,
                        is_causal: bool = True, device: Optional[torch.device | str] = None):
        r"""
        Note: Here we don't use [-(n-1), -(n-2), ..., -1, 0] sequence in the original paper,
        but use [0, 1, ..., n-1] instead, as later softmax operation is translation invariant.
        Also we don't construct the whole matrix but only one row (for causal case), as
        applying [-1, 0] to row is the same as applying [0, 1]

        Returns
        --------------
        alibi_mtx: torch.Tensor
            The alibi matrix. If is_causal: Shape [num_heads, 1, max_seqlen], else
            [num_heads, max_seqlen, max_seqlen]
        """
        slopes = AlibiEmbedding.build_alibi_slopes(num_heads, alibi_max_exp, device)
        if is_causal:
            alibi = torch.arange(max_seqlen, device=device, dtype=torch.float32)

            alibi = alibi[None, None, :] * slopes[:, None, None]
        else:
            alibi = torch.arange(max_seqlen, device=device, dtype=torch.float32)
            alibi = - torch.abs(alibi[None, :] - alibi[:, None])

            alibi = alibi * slopes[:, None, None]

        return alibi, slopes

    @staticmethod
    def build_alibi_slopes(num_heads: int, alibi_max_exp: int = 8, device: Optional[torch.device | str] = None):
        r"""
        Compatible with flash-attn 2

        Returns
        --------------
        slopes: torch.Tensor
            The slopes. Shape [num_heads,]
        """
        num_heads_closest_pow2 = 2 ** math.ceil(math.log2(num_heads))
        exponent = torch.arange(1, num_heads_closest_pow2 + 1, device=device, dtype=torch.float32)
        exponent = (exponent / num_heads_closest_pow2) * alibi_max_exp
        slopes = 1.0 / torch.pow(2, exponent)

        if num_heads_closest_pow2 != num_heads:
            slopes = torch.concat([slopes[1::2], slopes[::2]])[:num_heads]

        return slopes

    def forward(self, attn_mtx: torch.Tensor, query_start_pos: int = 0,
                precomputed_cache: Optional[list[Any]] = None):
        r"""
        Parameters
        --------------
        attn_mtx: torch.Tensor
            The attention matrix. Shape [batch_size, num_heads, q_seq_len, kv_seq_len]
        query_start_pos: int
            The start position of the query sequence (maybe not from 0 if using kv cache)
        precomputed_cache: list[torch.Tensor]
            Never passed, only for API compliance

        # TODO maybe this type of position embedding is naturally unsuitable for GLM like models
        # with multiple indices
        """
        if self.is_causal:
            return attn_mtx + self._alibi[:, :, :attn_mtx.size(3)], None  # None is a placeholder for not used cache (for compatibility)
        else:
            return attn_mtx + self._alibi[
                :, query_start_pos:query_start_pos + attn_mtx.size(2), :attn_mtx.size(3)
            ], None


# adapted from:
class RelativeLearnedEmbedding(RelativePositionEmbedding):
    def __init__(self, num_heads: int, max_seqlen: int = 4096, use_T5_bucket: bool = False,
                 T5_num_buckets: int = 128, T5_bidirectional: bool = True,
                 device: Optional[torch.device | str] = None):
        r"""
        Adapted from huggingface transformers/models/modeling_t5.py

        Note: this T5 bucket implementation may have performance issues, as it need to
        recompute the relative embeddings for each forward pass. Needs caching for every
        layer
        """
        super().__init__(apply_on="attn_mtx")

        self.num_heads = num_heads
        self.max_seqlen = max_seqlen
        self.device = device

        self.use_T5_bucket = use_T5_bucket
        self.T5_num_buckets = T5_num_buckets
        self.T5_bidirectional = T5_bidirectional
        if use_T5_bucket:
            self.num_relative_distances = T5_num_buckets
        else:
            self.num_relative_distances = max_seqlen * 2 - 1
        self.embedding = nn.Embedding(self.num_relative_distances, self.num_heads,
                                      device=device)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, bidirectional: bool = True,
                                  num_buckets: int = 64, max_distance: int = 128):
        r"""
        Translate relative position to a bucket number for relative attention. The relative
        position is defined as key_position - query_position, i.e. the distance in tokens
        from the attending position to the attended-to position. If bidirectional=False,
        then positive relative positions are invalid. We use smaller buckets for small
        absolute relative_position and larger buckets for larger absolute relative_positions.
        All relative positions >=max_distance map to the same bucket. All relative
        positions <=-max_distance map to the same bucket. This should allow for more graceful
        generalization to longer sequences than the model has been trained on
        """
        relative_buckets = 0   # here means torch.zeros_like(relative_position)

        if bidirectional:
            num_buckets //= 2
            # positive half of relative_positions use [num_buckets/2, num_buckets) as
            # their bucket, the negative half use [0, num_buckets/2)
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = torch.max(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_T5_bias(self, query_length: int, key_length: int,
                     query_start_pos: int = 0, key_start_pos: int = 0):
        r"""
        Compute binned relative position bias
        """
        query_position = torch.arange(query_start_pos, query_length, dtype=torch.long, device=self.device)[:, None]
        key_position = torch.arange(key_start_pos, key_length, dtype=torch.long, device=self.device)[None, :]
        relative_position = query_position - key_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.T5_bidirectional,
            num_buckets=self.T5_num_buckets,
            max_distance=self.max_seqlen,
        )
        bias = self.embedding(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        bias = bias.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return bias

    def compute_normal_bias(self, query_length: int, key_length: int,
                            query_start_pos: int = 0, key_start_pos: int = 0):
        query_position = torch.arange(query_start_pos, query_length, dtype=torch.long, device=self.device)[:, None]
        key_position = torch.arange(key_start_pos, key_length, dtype=torch.long, device=self.device)[None, :]
        relative_position = (query_position - key_position) + self.max_seqlen - 1
        relative_position = torch.clamp(relative_position, 0, self.num_relative_distances - 1)
        bias = self.embedding(relative_position)
        bias = bias.permute([2, 0, 1]).unsqueeze(0)
        return bias

    def forward(self, attn_mtx: torch.Tensor, query_start_pos: int = 0,
                precomputed_cache: Optional[list[Any]] = None):
        # return computed cache
        if precomputed_cache is not None:
            return attn_mtx + precomputed_cache
        else:
            if self.use_T5_bucket:
                bias = self.compute_T5_bias(attn_mtx.size(2), attn_mtx.size(3),
                                            query_start_pos=query_start_pos)
                return attn_mtx + bias, bias
            else:
                bias = self.compute_normal_bias(attn_mtx.size(2), attn_mtx.size(3),
                                                query_start_pos=query_start_pos)
                return attn_mtx + bias, bias


####################################################
# ========== Absolute Position Embedding ==========
####################################################
# 1. used once before the first layer, so no cache is needed

class AbsoluteLearnedEmbedding(nn.Embedding):
    def __init__(self, max_seqlen: int, emb_dim: int, device: Optional[torch.device | str] = None,
                 sinusoidal: bool = False, learnable: bool = False, base: int = 10000,
                 pad_token_id: Optional[int] = None):
        super().__init__(max_seqlen, emb_dim, padding_idx=pad_token_id, device=device)

        self.max_seqlen = max_seqlen
        self.emb_dim = emb_dim

        if not learnable:
            self.weight.requires_grad = False

        if sinusoidal:
            self.base = base
            self._init_sinusoidal()

    def _init_sinusoidal(self):
        r"""
        Initialize the embedding table with sinusoidal values.
        """
        assert self.emb_dim % 2 == 0, "The embedding dimension must be even for sinusoidal embedding"
        emb = torch.zeros_like(self.weight, device=self.weight.device)

        wave_num = 1.0 / (self.base ** (torch.arange(0, self.emb_dim, 2, device=self.weight.device,
                                                     dtype=torch.float32) / self.emb_dim))
        t = torch.arange(self.max_seqlen, device=self.weight.device, dtype=torch.float32)

        rotation_angles = torch.outer(t, wave_num)

        emb[:, 0::2] = rotation_angles.sin()
        emb[:, 1::2] = rotation_angles.cos()

        self.weight.data.copy_(emb)

    def forward(self, position_ids: torch.Tensor):
        r"""
        Parameters
        --------------
        position_ids: torch.Tensor
            The position indices. Shape [batch_size, seq_len] or [seq_len,]
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        return super().forward(position_ids)
