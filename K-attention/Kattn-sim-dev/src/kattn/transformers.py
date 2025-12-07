import math
import os
from typing import Optional, Any, Literal
import logging
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .import_utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import (
        flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_func,
    )
    from flash_attn.bert_padding import (
        pad_input, unpad_input
    )

from .utils import (
    expand_key_padding_mask,
    ATTN_METHODS,
    FFN_TYPES,
    LAYERNORM_TYPES,
    GATED_ACT_FUNCS_MAP,
    NORMAL_ACT_FUNCS_MAP,
    LAYERNORM_MAP,
)
from .position_embeddings import (
    AbsoluteLearnedEmbedding,
    RelativePositionEmbedding,
    RelativeLearnedEmbedding,
    RotaryEmbedding,
    AlibiEmbedding,
    LEGAL_POSITION_EMBEDDING
)
from .modules import BaseClassifier, CNNMixer, CNNMixerConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attn_heads: int = 12
    per_head_size: Optional[int] = None
    W_qkv_bias: bool = False                       #TODO, check which bias? FFN bias or W_qkv bias
    proj_bias: bool = True
    FFN_bias: bool = True
    attn_dropout: float = 0.0
    residual_dropout: float = 0.1
    FFN_size: int = None
    FFN_type: FFN_TYPES = "gelu"                  #TODO
    layernorm_eps: float = 1e-12
    layernorm_type: LAYERNORM_TYPES = "normal"    #TODO
    fuse_dropout_add_ln: bool = False
    prenorm: bool = True

    # positional embeddings
    position_emb_type: LEGAL_POSITION_EMBEDDING = "absolute_learned"
    PE_kwargs: dict = None
    max_seqlen: int = 512
    vocab_size: int = 10   # AUCGN, [PAD][CLS][MASK][EOS][UNK]
    pad_token_id: int = 0

    # attention details
    attn_method: ATTN_METHODS = "flash-attn"
    pack_level: Literal["pack_kv", "pack_qkv"] = "pack_qkv"


class QKVAttentionOp(nn.Module):
    def __init__(
        self, attn_dropout: float = 0.0, softmax_scale: Optional[float] = None,
        is_causal: bool = False, position_emb: Optional[RelativeLearnedEmbedding | AlibiEmbedding] = None,
    ):
        #TODO gated head
        super().__init__()
        self.attn_dropout = attn_dropout
        self.softmax_scale = softmax_scale
        self.is_causal = is_causal
        self.attn_drop = nn.Dropout(attn_dropout)
        if isinstance(position_emb, (RelativeLearnedEmbedding, AlibiEmbedding)):
            self.position_emb = position_emb
        else:
            if position_emb is not None:
                logger.warning(
                    "position embedding applied on flash attn Op should be AlibiEmbedding. "
                    f"Got {type(position_emb)} instead. Fall back to None."
                )
            self.position_emb = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[Any] = None,   #TODO, mask certain heads for Interpretability
        is_causal: Optional[bool] = None,
        return_attn_mtx: bool = False,
        average_attn: bool = False,
        pos_emb_cache: Optional[Any] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        r"""
        Parameters
        ------------------------------
        q: torch.Tensor
            shape of [batch, query_seq_len, num_heads, head_dim]
        k: torch.Tensor
            shape of [batch, key/value_seq_len, num_heads, head_dim]
        v: torch.Tensor
            shape of [batch, key/value_seq_len, num_heads, head_dim]
        key_padding_mask: torch.Tensor
            shape of [batch, key/value_seq_len], masked positions filled with 0 or
            shape of [batch, 1, 1, key/value_seq_len], masked positions filled with -inf
        attn_mask: torch.Tensor
            shape of [batch, num_heads, query_seq_len, key/value_seq_len]
        head_mask: torch.Tensor
            Not implemented yet (#TODO)
        is_causal: bool
            whether to apply causal mask
        return_attn_mtx: bool
            whether to return attention matrix
        average_attn: bool
            whether to average attention matrix across heads if return_attn_mtx is True
        pos_emb_cache: Optional[Any]
            cache for position embedding, here for AlibiEmbedding and RelativeLearnedEmbedding

        Returns
        ------------------------------
        output: torch.Tensor
            shape of [batch, query_seq_len, num_heads, head_dim]
        attn_mtx: torch.Tensor
            if return_attn_mtx is True, shape of [batch, (num_heads,) query_seq_len, key_seq_len]
            else None
        pos_emb_cache: Optional[Any]
        """
        batch_size, q_seq_len, num_heads, head_dim = q.shape
        softmax_scale = 1.0 / math.sqrt(head_dim) if self.softmax_scale is None else self.softmax_scale
        dtype = q.dtype
        device = q.device

        logits_matrix = torch.einsum("bqhd,bkhd->bhqk", q, k) * softmax_scale

        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:  # [batch_size, k_seq_len], masked positions filled with 0
                key_padding_mask = expand_key_padding_mask(key_padding_mask, dtype=dtype, device=device)
            else:  # [batch_size, 1, 1, k_seq_len], masked positions filled with -inf (has already been expanded)
                assert key_padding_mask.dim() == 4, "key_padding_mask must be of dim 2 or 4"
            logits_matrix.add_(key_padding_mask)

        if attn_mask is not None:
            assert attn_mask.dim() == 4, "attn_mask must be of dim 4"
            logits_matrix.add_(attn_mask)
        if head_mask is not None:    #TODO check how is head_mask applied
            raise NotImplementedError("head mask not implemented")
            logits_matrix.add_(head_mask)
        is_causal = is_causal if is_causal is not None else self.is_causal
        if is_causal:
            raise NotImplementedError("causal not implemented")
            #TODO causal mask shape needs to be changed
            causal_mask = torch.triu(
                torch.full((q_seq_len, q_seq_len), torch.finfo(dtype).min, dtype=dtype, device=device), 1
            )
            logits_matrix.add_(causal_mask)

        # apply position embedding, including ALiBi (no cache), and RelativeLearned
        if isinstance(self.position_emb, (AlibiEmbedding, RelativeLearnedEmbedding)):
            logits_matrix, pos_emb_cache = self.position_emb(
                logits_matrix, precomputed_cache=pos_emb_cache
            )

        attn_matrix = F.softmax(logits_matrix, dim=-1)
        attn_matrix = self.attn_drop(attn_matrix)
        output = torch.einsum("bhqk,bkhd->bqhd", attn_matrix, v)

        attn_mtx = None
        if return_attn_mtx:
            attn_mtx = torch.mean(attn_matrix, dim=1) if average_attn else attn_matrix
        return output, attn_mtx, pos_emb_cache


class FlashQKVAttentionOp(nn.Module):
    def __init__(
        self, attn_dropout: float = 0.0, softmax_scale: Optional[float] = None,
        is_causal: bool = False, is_cross_attention: bool = False,
        position_emb: Optional[RelativeLearnedEmbedding | AlibiEmbedding] = None,
        deterministic: bool = False
    ):
        super().__init__()
        assert is_flash_attn_2_available(), "FlashAttention2 is not installed"
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention
        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)
        if isinstance(position_emb, AlibiEmbedding):
            self.position_emb = position_emb
        elif isinstance(position_emb, RelativeLearnedEmbedding):
            raise TypeError("RelativeLearnedEmbedding not supported for flash attention Op")
        else:
            if position_emb is not None:
                logger.warning(
                    "position embedding applied on flash attn Op should be AlibiEmbedding. "
                    f"Got {type(position_emb)} instead. Fall back to None."
                )
            self.position_emb = None
        self.deterministic = deterministic

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_kv: Optional[int] = None,
        is_causal: Optional[bool] = None,
        return_attn_mtx: bool = False,
        average_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Parameters
        ------------------------------
        q: torch.Tensor
            shape of [total_seq_len_q, num_heads, head_dim]
        k: torch.Tensor
            shape of [total_seq_len_kv, num_heads, head_dim]
        v: torch.Tensor
            shape of [total_seq_len_kv, num_heads, head_dim]
        cu_seqlens_q: torch.Tensor
            cumulative sequence lengths of q, shape of [batch_size + 1, ]
        max_seqlen_q: int
            maximum sequence length of q
        cu_seqlens_kv: torch.Tensor
            cumulative sequence lengths of kv, shape of [batch_size + 1, ]
        max_seqlen_kv: int
            maximum sequence length of kv
        is_causal: bool
        return_attn_mtx: bool
            whether to return attention matrix
        average_attn: bool
            whether to average attention matrix across heads if return_attn_mtx is True

        Returns
        ------------------------------
        o: torch.Tensor
            shape of [total_seq_len, num_heads, head_dim]
        attn_mtx: torch.Tensor
            if return_attn_mtx is True, shape of [batch, (num_heads,) query_seq_len, key_seq_len]
            else None
        """
        # ALiBi position embedding
        alibi_slopes = None
        if self.position_emb is not None and isinstance(self.position_emb, AlibiEmbedding):
            alibi_slopes = self.position_emb.slopes

        is_causal = is_causal if is_causal is not None else self.is_causal

        flash_attn_kwargs = {
            "dropout_p": self.attn_drop.p if self.training else 0.0,
            "softmax_scale": self.softmax_scale,
            "causal": is_causal,
            "alibi_slopes": alibi_slopes,
            "deterministic": self.deterministic,
            "return_attn_probs": return_attn_mtx,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": max_seqlen_q,
        }
        if self.is_cross_attention:
            flash_attn_kwargs.update({
                "cu_seqlens_k": cu_seqlens_kv,
                "max_seqlen_k": max_seqlen_kv,
            })
            o = flash_attn_varlen_func(
                q, k, v,
                **flash_attn_kwargs
            )
        else:
            flash_attn_kwargs.update({
                "cu_seqlens_k": cu_seqlens_q,
                "max_seqlen_k": max_seqlen_q,
            })
            o = flash_attn_varlen_func(
                q, k, v,
                **flash_attn_kwargs
            )

        if return_attn_mtx:
            o, _, attn_mtx = o
            attn_mtx = torch.mean(attn_mtx, dim=1) if average_attn else attn_mtx
            return o, attn_mtx
        else:
            return o, None


class FlashQKVPackedAttentionOp(nn.Module):
    r"""
    Attention op for packed qkv
    """
    def __init__(
        self, attn_dropout=0.0, softmax_scale=None, is_causal=False, is_cross_attention=False,
        position_emb: Optional[RelativeLearnedEmbedding | AlibiEmbedding] = None,
        deterministic=False
    ):
        super().__init__()
        assert is_flash_attn_2_available(), "FlashAttention2 is not installed"
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention
        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)
        if isinstance(position_emb, AlibiEmbedding):
            self.position_emb = position_emb
        elif isinstance(position_emb, RelativeLearnedEmbedding):
            raise TypeError("RelativeLearnedEmbedding not supported for flash attention Op")
        else:
            if position_emb is not None:
                logger.warning(
                    "position embedding applied on flash attn Op should be AlibiEmbedding. "
                    f"Got {type(position_emb)} instead. Fall back to None."
                )
            self.position_emb = None
        self.deterministic = deterministic

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        is_causal: Optional[bool] = None,
        return_attn_mtx: bool = False,
        average_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Parameters
        ------------------------------
        qkv: torch.Tensor
            shape of [total_seq_len, 3, num_heads, head_dim]
        cu_seqlens_q: torch.Tensor
            cumulative sequence lengths of q (same for kv), shape of [batch_size + 1]
        max_seqlen_q: int
            maximum sequence length of q (same for kv)
        is_causal: Optional[bool]
        return_attn_mtx: bool
            whether to return attention matrix
        average_attn: bool
            whether to average attention matrix across heads if return_attn_mtx is True

        Returns
        ------------------------------
        o: torch.Tensor
            shape of [total_seq_len, num_heads, head_dim]
        attn_mtx: torch.Tensor
            if return_attn_mtx is True, shape of [batch, (num_heads,) query_seq_len, key_seq_len]
            else None
        """
        # ALiBi position embedding
        alibi_slopes = None
        if self.position_emb is not None and isinstance(self.position_emb, AlibiEmbedding):
            alibi_slopes = self.position_emb.slopes

        is_causal = is_causal if is_causal is not None else self.is_causal

        o = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens_q,
            max_seqlen=max_seqlen_q,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
            alibi_slopes=alibi_slopes,
            deterministic=self.deterministic,
            return_attn_probs=return_attn_mtx,
        )

        if return_attn_mtx:
            o, _, attn_mtx = o
            attn_mtx = torch.mean(attn_mtx, dim=1) if average_attn else attn_mtx
            return o, attn_mtx
        else:
            return o, None


class MultiHeadAttention(nn.Module):
    r"""Attention operation. Holding projection parameters.
    Can be used as **self attention** or **cross attention**. If used as cross attention,
    pass in both X and Y.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        per_head_size: Optional[int] = None,
        attn_method: Literal["homemade", "flash-attn"] = "homemade",
        attn_dropout: float = 0.0,
        W_qkv_bias: bool = False,
        proj_bias: bool = True,
        is_causal: bool = False,
        is_cross_attention: bool = False,
        position_emb: Optional[RelativePositionEmbedding] = None,
        gated_head: bool = False,   #TODO gated head
        pack_level: Literal["pack_kv", "pack_qkv"] = "pack_kv"
    ):
        # TODO: whether to include attention score dropout? (not usual)
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.is_cross_attention = is_cross_attention

        if per_head_size is not None:
            self.per_head_size = per_head_size
        else:
            assert hidden_size % num_heads == 0, "hidden_dim should be divisible by num_heads to enable residual connection."
            self.per_head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.per_head_size

        self.pack_level = pack_level
        self.attn_method = attn_method
        if self.pack_level == "pack_kv":
            self.proj_q = nn.Linear(
                self.hidden_size, self.all_head_size, bias=W_qkv_bias
            )
            self.proj_kv = nn.Linear(
                self.hidden_size, self.all_head_size * 2, bias=W_qkv_bias
            )
        elif self.pack_level == "pack_qkv":
            assert not self.is_cross_attention, "pack_qkv mode should not be used for cross attention"
            self.proj_qkv = nn.Linear(
                self.hidden_size, self.all_head_size * 3, bias=W_qkv_bias
            )
        else:
            raise ValueError(f"Unrecognized pack level: {pack_level}")
        
        # deal with RoPE here
        if isinstance(position_emb, RotaryEmbedding):
            self.position_emb = position_emb
            lower_level_pe = None
        else:
            self.position_emb = None
            lower_level_pe = position_emb

        if attn_method == "homemade":
            self.attn_op = QKVAttentionOp(
                attn_dropout=attn_dropout, is_causal=is_causal,
                position_emb=lower_level_pe
            )
        elif attn_method == "flash-attn":
            if self.pack_level == "pack_kv":
                attn_op_cls = FlashQKVAttentionOp
            else:
                attn_op_cls = FlashQKVPackedAttentionOp
            self.attn_op = attn_op_cls(
                attn_dropout=attn_dropout,
                is_causal=is_causal,
                is_cross_attention=is_cross_attention,
                position_emb=lower_level_pe,
            )
        else:
            raise ValueError(f"Unrecognized attention method: {attn_method}")

        self.proj_e = nn.Linear(self.all_head_size, self.hidden_size, bias=proj_bias)

    def forward(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        flash_attn_kwargs: Optional[dict] = None,
        return_attn_mtx: bool = False,
        average_attn: bool = False,
        pos_emb_cache: Optional[Any] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        r"""
        Parameters
        ------------------------------
        X: torch.Tensor
            shape of [batch_size, seq_len_q, hidden_size] when not using flash-attn,
            shape of [total_seq_len_q, hidden_size] when using flash-attn for varlen input
        Y: torch.Tensor
            shape of [batch_size, seq_len, hidden_size] when not using flash-attn,
            shape of [total_seq_len_k, hidden_size] when using flash-attn for varlen input
        key_padding_mask: torch.Tensor
            used for self implemented attention
        attn_mask: torch.Tensor
        head_mask: torch.Tensor
        flash_attn_kwargs: dict
            kwargs for flash attention, including
            - If not cross attention: cu_seqlens_q, max_seqlen_q
            - If cross attention: cu_seqlens_q, max_seqlen_q, cu_seqlens_kv, max_seqlen_kv
        return_attn_mtx: bool
            whether to return attention matrix
        average_attn: bool
            whether to average attention matrix across heads if return_attn_mtx is True
        pos_emb_cache: Optional[Any]
            cache for position embedding, here for RotaryEmbedding, AlibiEmbedding
            and RelativeLearnedEmbedding. Note: flash-attn does not support
            AbsoluteLearnedEmbedding, and need no pos_emb_cache for AlibiEmbedding

        Returns
        ------------------------------
        o: torch.Tensor
        attn_mtx: torch.Tensor
            if return_attn_mtx is True, shape of [batch, (num_heads,) query_seq_len, key_seq_len]
            else None
        pos_emb_cache: Optional[Any]
        """
        if self.pack_level == "pack_kv":
            if self.is_cross_attention:
                assert Y is not None
            else:
                Y = X
            q = self.proj_q(X)
            kv = self.proj_kv(Y)
            q = rearrange(q, "... (h d) -> ... h d", h=self.num_heads)
            kv = rearrange(kv, "... (two h d) -> ... two h d", two=2, h=self.num_heads)
        elif self.pack_level == "pack_qkv":
            assert Y is None and not self.is_cross_attention, \
                "pack_qkv mode should not be used for cross attention"
            qkv = self.proj_qkv(X)
            qkv = rearrange(
                qkv, "... (three h d) -> ... three h d", three=3, h=self.num_heads
            )

        if self.attn_method == "homemade":
            if self.pack_level == "pack_kv":
                k, v = torch.unbind(kv, dim=2)
            elif self.pack_level == "pack_qkv":
                q, k, v = torch.unbind(qkv, dim=2)

            # apply RoPE
            if self.position_emb is None:
                lower_level_pe_cache = None
            elif isinstance(self.position_emb, RotaryEmbedding):
                if pos_emb_cache is not None:
                    q_pe_cache, k_pe_cache = pos_emb_cache
                else:
                    q_pe_cache, k_pe_cache = None, None
                q, q_pe_cache = self.position_emb(
                    q, seqlen_offsets=0, precomputed_cos_sin_cache=q_pe_cache,
                )
                k, k_pe_cache = self.position_emb(
                    k, seqlen_offsets=0, precomputed_cos_sin_cache=k_pe_cache,
                )
                pos_emb_cache = (q_pe_cache, k_pe_cache)
                lower_level_pe_cache = None
            else:
                lower_level_pe_cache = pos_emb_cache

            o, attn_mtx, lower_level_pe_cache = self.attn_op(
                q, k, v,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                head_mask=head_mask,
                return_attn_mtx=return_attn_mtx,
                average_attn=average_attn,
                pos_emb_cache=lower_level_pe_cache
            )
            if not (self.position_emb is None or isinstance(self.position_emb, RotaryEmbedding)):
                pos_emb_cache = lower_level_pe_cache
            o = rearrange(o, "b l h d -> b l (h d)")
        elif self.attn_method == "flash-attn":
            assert "cu_seqlens_q" in flash_attn_kwargs and \
                "max_seqlen_q" in flash_attn_kwargs, \
                "cu_seqlens_q and max_seqlen_q should be included in flash_attn_kwargs"
            assert attn_mask is None and head_mask is None, "flash-attn does not support attn_mask and head_mask"

            # Note: flash-attn does not support AbsoluteLearnedEmbedding, and need
            # no pos_emb_cache for AlibiEmbedding

            if self.pack_level == "pack_kv":
                k, v = torch.unbind(kv, dim=1)

                if isinstance(self.position_emb, RotaryEmbedding):
                    if pos_emb_cache is not None:
                        q_pe_cache, k_pe_cache = pos_emb_cache
                    else:
                        q_pe_cache, k_pe_cache = None, None
                    q, q_pe_cache = self.position_emb(
                        q,
                        cu_seqlens=flash_attn_kwargs["cu_seqlens_q"],
                        max_seqlen=flash_attn_kwargs["max_seqlen_q"],
                        precomputed_cos_sin_cache=q_pe_cache,
                    )
                    k, k_pe_cache = self.position_emb(
                        k,
                        cu_seqlens=flash_attn_kwargs["cu_seqlens_kv"] if self.is_cross_attention else flash_attn_kwargs["cu_seqlens_q"],
                        max_seqlen=flash_attn_kwargs["max_seqlen_kv"] if self.is_cross_attention else flash_attn_kwargs["max_seqlen_q"],
                        precomputed_cos_sin_cache=k_pe_cache,
                    )
                    pos_emb_cache = (q_pe_cache, k_pe_cache)

                if self.is_cross_attention:
                    assert "cu_seqlens_kv" in flash_attn_kwargs and \
                        "max_seqlen_kv" in flash_attn_kwargs, \
                        "cu_seqlens_kv and max_seqlen_kv should be included in flash_attn_kwargs if cross attention"

                o, attn_mtx = self.attn_op(
                    q, k, v,
                    **flash_attn_kwargs,
                    return_attn_mtx=return_attn_mtx,
                    average_attn=average_attn
                )
            elif self.pack_level == "pack_qkv":
                if isinstance(self.position_emb, RotaryEmbedding):
                    qkv, pos_emb_cache = self.position_emb(
                        qkv,
                        cu_seqlens=flash_attn_kwargs["cu_seqlens_q"],
                        max_seqlen=flash_attn_kwargs["max_seqlen_q"],
                        precomputed_cos_sin_cache=pos_emb_cache,
                    )
                o, attn_mtx = self.attn_op(
                    qkv,
                    **flash_attn_kwargs,
                    return_attn_mtx=return_attn_mtx,
                    average_attn=average_attn
                )

            o = rearrange(o, "t h d -> t (h d)")

        o = self.proj_e(o)

        return o, attn_mtx, pos_emb_cache


class GatedFFN(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, act_fn: nn.Module, bias: bool = False
    ):
        super().__init__()
        self.gate_proj = nn.Linear(in_dim, out_dim, bias=bias)
        self.up_proj = nn.Linear(in_dim, out_dim, bias=bias)
        self.down_proj = nn.Linear(out_dim, in_dim, bias=bias)
        self.act_fn = act_fn

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(X)) * self.up_proj(X))


class AttentionBlock(nn.Module):
    r"""
    Attention Block including 2 sublayers: Self attention sublayer and FeedForward sublayer, including
    residual connection and layer normalization. Support pre-norm and post-norm.
    """
    #TODO fuse dropout add layernorm operation from triton, may change input/output structure
    def __init__(
        self,
        # specific to AttentionBlock
        FFN_type: FFN_TYPES = "gelu",
        FFN_bias: bool = True,
        FFN_size: Optional[int] = None,
        prenorm: bool = True,
        layer_norm_type: LAYERNORM_TYPES = "normal",
        layer_norm_eps: float = 1e-12,
        fuse_dropout_add_ln: bool = False,
        residual_dropout: float = 0.1,
        layer_idx: Optional[int] = None,
        layer_idx_scale=False,   #TODO layer_idx_scale

        # Parameters for MultiheadAttention
        hidden_size: int = 768,
        num_heads: int = 12,
        per_head_size: Optional[int] = None,
        is_causal: bool = False,
        W_qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_method: ATTN_METHODS = "homemade",
        attn_dropout: float = 0.,
        pack_level: Literal["pack_qkv", "pack_kv"] = "pack_kv",
        position_emb: Optional[RelativePositionEmbedding] = None,
    ):
        super().__init__()
        self.attn_method = attn_method
        if attn_method in ("homemade", "flash-attn"):
            self.self_attn = MultiHeadAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                per_head_size=per_head_size,
                attn_method=attn_method,
                attn_dropout=attn_dropout,
                W_qkv_bias=W_qkv_bias,
                proj_bias=proj_bias,
                is_causal=is_causal,
                is_cross_attention=False,
                position_emb=position_emb,
                pack_level=pack_level
            )
        elif attn_method == "pytorch":
            assert position_emb is None, "pytorch MultiheadAttention doesn't support hand-craft position embedding"
            self.self_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=attn_dropout, batch_first=True,
                add_bias_kv=W_qkv_bias, bias=proj_bias
            )
        else:
            raise ValueError("attn_method should be one of homemade, flash-attn, pytorch")

        self.layer_idx = layer_idx
        #TODO layer_idx based scale
        self.prenorm = prenorm
        #TODO fuse_dropout_add_ln
        self.fuse_dropout_add_ln = fuse_dropout_add_ln
        self.layernorm1 = LAYERNORM_MAP[layer_norm_type](hidden_size, eps=layer_norm_eps)
        self.layernorm2 = LAYERNORM_MAP[layer_norm_type](hidden_size, eps=layer_norm_eps)
        self.residual_drop1 = nn.Dropout(residual_dropout)
        self.residual_drop2 = nn.Dropout(residual_dropout)

        self.FFN_type = FFN_type
        if self.FFN_type in NORMAL_ACT_FUNCS_MAP:
            FFN_size = FFN_size if FFN_size is not None else hidden_size * 4
            self.FFN = nn.Sequential(
                nn.Linear(hidden_size, FFN_size, bias=FFN_bias),
                NORMAL_ACT_FUNCS_MAP[self.FFN_type](),
                nn.Linear(FFN_size, hidden_size, bias=FFN_bias),
            )
        elif self.FFN_type in GATED_ACT_FUNCS_MAP:
            FFN_size = FFN_size if FFN_size is not None else int(hidden_size * 8 / 3)    # for parameter size match
            act_fn = GATED_ACT_FUNCS_MAP[self.FFN_type]()
            self.FFN = GatedFFN(hidden_size, FFN_size, act_fn, bias=FFN_bias)

        self.position_emb = position_emb

    def forward(
        self,
        X: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        flash_attn_kwargs: Optional[dict] = None,
        return_attn_mtx: bool = False,
        average_attn: bool = False,
        pos_emb_cache: Optional[Any] = None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        r"""
        Parameters
        ------------------------------
        X: torch.Tensor
            shape of [batch_size, seq_len, hidden_size] when not using self-attention,
            shape of [total_seq_len, hidden_size] when using self-attention
        key_padding_mask: Optional[torch.Tensor]
            used for self homemade attention and pytorch (Note the content is different, though)
        flash_attn_kwargs: dict
            kwargs for flash attn, including cu_seqlens, max_seqlen
        attn_mask: Optional[torch.Tensor]
            support by homemade and pytorch
        head_mask:
            support by homemade, not implemented yet
        return_attn_mtx: bool
        average_attn: bool
        pos_emb_cache: Optional[Any]
            cache for position embedding, here for RotaryEmbedding, AlibiEmbedding
            and RelativeLearnedEmbedding.

        Returns
        ------------------------------
        out_hidden_states: torch.Tensor
        attn_mtx: torch.Tensor
            if return_attn_mtx is True, shape of [batch, (num_heads,) query_seq_len, key_seq_len]
            else None
        position_emb_cache: Optional[list[Any]]
            Cache for position embeddings, includes RoPE, RelativeLearned
        """
        # for compatibility with pytorch MultiheadAttention
        output_kwargs = {
            "return_attn_mtx": return_attn_mtx,
            "average_attn": average_attn,
        } if self.attn_method in ("homemade", "flash-attn") else {
            "need_weights": return_attn_mtx,
            "average_attn_weights": average_attn
        }
        varlen_kwargs = {
            "flash_attn_kwargs": flash_attn_kwargs
        } if self.attn_method == "flash-attn" else {
            "key_padding_mask": key_padding_mask
        }
        misc_kwargs = {
            "head_mask": head_mask,
            "attn_mask": attn_mask,
            "pos_emb_cache": pos_emb_cache
        } if self.attn_method in ("homemade", "flash-attn") else {
            "attn_mask": attn_mask
        }

        if self.prenorm:
            if self.fuse_dropout_add_ln:
                raise NotImplementedError("fuse not implemented")
            else:
                residual = X
                X = self.layernorm1(X)
                # !!! Note cannot set before, as we need to pass layernorm
                input_kwargs = {"X": X} \
                    if self.attn_method in ("homemade", "flash-attn") else \
                    {"query": X, "key": X, "value": X}
                X, *misc_outputs = self.self_attn(
                    **input_kwargs,
                    **varlen_kwargs,
                    **misc_kwargs,
                    **output_kwargs,
                )
                X = self.residual_drop1(X) + residual

                residual = X
                X = self.layernorm2(X)
                X = self.FFN(X)
                X = self.residual_drop2(X) + residual
        else:  # post norm
            residual = X
            input_kwargs = {"X": X} if self.attn_method in ("homemade", "flash-attn") else \
                {"query": X, "key": X, "value": X}
            X, *misc_outputs = self.self_attn(
                **input_kwargs,
                **varlen_kwargs,
                **misc_kwargs,
                **output_kwargs,
            )
            if self.fuse_dropout_add_ln:
                raise NotImplementedError("fuse not implemented")
            else:
                X = self.residual_drop1(X)
                X = self.layernorm1(X + residual)

            residual = X
            X = self.FFN(X)
            if self.fuse_dropout_add_ln:
                raise NotImplementedError("fuse not implemented")
            else:
                X = self.residual_drop2(X)
                X = self.layernorm2(X + residual)

        # extract misc_outputs (attn_mtx, pos_emb_cache)
        if self.attn_method == "pytorch":
            attn_mtx, = misc_outputs
        else:
            attn_mtx, pos_emb_cache = misc_outputs

        return X, attn_mtx, pos_emb_cache


class TransformerEncoder(nn.Module):
    def __init__(
        self,

        # Encoder specific parameters
        num_layers: int = 12,
        activation_checkpointing: bool = False,

        # parameters for AttentionBlock
        hidden_size: int = 768,
        num_heads: int = 12,
        per_head_size: Optional[int] = None,
        prenorm: bool = True,
        W_qkv_bias: bool = False,
        proj_bias: bool = True,
        FFN_type: FFN_TYPES = "gelu",
        FFN_bias: bool = True,
        FFN_size: Optional[int] = None,
        attn_method: ATTN_METHODS = "homemade",
        fuse_dropout_add_ln: bool = False,
        layer_norm_type: LAYERNORM_TYPES = "normal",
        layer_norm_eps: float = 1e-12,
        attn_dropout: float = 0.0,
        residual_dropout: float = 0.1,
        layer_idx_scale: bool = False,
        pack_level: Literal["pack_qkv", "pack_kv"] = "pack_kv",
        position_emb: Optional[RelativePositionEmbedding] = None,
    ):
        super().__init__()
        self.attn_method = attn_method
        self.prenorm = prenorm
        #TODO fuse_dropout_add_ln
        self.fuse_dropout_add_ln = fuse_dropout_add_ln
        self.layers = nn.ModuleList([
            AttentionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                per_head_size=per_head_size,
                is_causal=False,
                W_qkv_bias=W_qkv_bias,
                proj_bias=proj_bias,
                FFN_type=FFN_type,
                FFN_bias=FFN_bias,
                FFN_size=FFN_size,
                prenorm=prenorm,
                attn_method=attn_method,
                fuse_dropout_add_ln=fuse_dropout_add_ln,
                layer_norm_type=layer_norm_type,
                layer_norm_eps=layer_norm_eps,
                attn_dropout=attn_dropout,
                residual_dropout=residual_dropout,
                layer_idx=i,
                layer_idx_scale=layer_idx_scale,
                pack_level=pack_level,
                position_emb=position_emb,
            )
            for i in range(num_layers)
        ])
        self.extra_layernorm = LAYERNORM_MAP[layer_norm_type](hidden_size, eps=layer_norm_eps)

        self.activation_checkpointing = activation_checkpointing

    def forward(
        self,
        X: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        return_attn_mtx: bool = False,
        average_attn: bool = False,
        return_mid_hidden: bool = False
    ):
        r"""
        Parameters
        ------------------------------
        X: torch.Tensor
            shape of [batch_size, seq_len, hidden_size]
        key_padding_mask: torch.Tensor
            shape of [batch_size, seq_len], 0 for padding, 1 for real token
        attn_mask: torch.Tensor  #TODO
            shape of [batch_size, nheads, seq_len, seq_len] or [seq_len, seq_len]

        Returns
        ------------------------------
        final_hidden_states: torch.Tensor
        attn_mtxs: tuple
            if return_attn_mtx is True, tuple of torch.Tensor, each of shape [batch, (num_heads,) query_seq_len, key_seq_len]
            else None
        mid_hidden_states: tuple
            if return_mid_hidden is True, tuple of torch.Tensor, each of shape [batch, seq_len, hidden_size]
            else None
        """
        # post norm
        #   default: LN -> (MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN)n-layer
        #   fuse_dropout_add_ln: Dropout_LN -> (MHA -> Dropout_Add_LN -> MLP -> Dropout_Add_LN)n-layer

        # Currently flash attention v2 doesn't support return_attn_mtx
        if self.attn_method == "flash-attn" and return_attn_mtx:
            raise ValueError("flash-attn v2 doesn't support return_attn_mtx for now.")

        if not self.prenorm:
            X = self.extra_layernorm(X)

        attn_mtxs = list() if return_attn_mtx else None
        mid_hidden_states = [X, ] if return_mid_hidden else None

        if self.attn_method == "flash-attn":
            batch_size, seq_len = X.shape[:2]
            # Note: flash attn 2.7 change this api and return 5 values (with an extra used_seqlens)
            X, indices, cu_seqlens, max_seqlen, *_ = unpad_input(X, key_padding_mask)
            flash_attn_kwargs = {
                "cu_seqlens_q": cu_seqlens,
                "max_seqlen_q": max_seqlen
            }

        if self.attn_method == "homemade":
            # expand and transform key_padding_mask
            expanded_key_padding_mask = \
                expand_key_padding_mask(key_padding_mask,
                                        dtype=X.dtype,
                                        device=X.device)
        elif self.attn_method == "pytorch":
            expanded_key_padding_mask = ~key_padding_mask.bool()

        pos_emb_cache = None
        for i, layer in enumerate(self.layers):
            attn_block_kwargs = {
                "X": X,
                "attn_mask": attn_mask,
                "head_mask": head_mask,
                "return_attn_mtx": return_attn_mtx,
                "average_attn": average_attn,
                "pos_emb_cache": pos_emb_cache,
            }
            if self.attn_method == "flash-attn":
                attn_block_kwargs.update({
                    "flash_attn_kwargs": flash_attn_kwargs
                })
            else:
                attn_block_kwargs.update({
                    "key_padding_mask": expanded_key_padding_mask
                })

            if self.activation_checkpointing:
                X, attn_mtx, pos_emb_cache = torch.utils.checkpoint.checkpoint(
                    layer, **attn_block_kwargs,
                    use_reentrant=False,
                )
            else:
                X, attn_mtx, pos_emb_cache = layer(
                    **attn_block_kwargs
                )

            if return_attn_mtx:
                attn_mtxs.append(attn_mtx)
            if return_mid_hidden:
                if self.attn_method == "flash-attn":
                    mid_hidden_states.append(pad_input(X, indices, batch_size, seq_len))
                else:
                    mid_hidden_states.append(X)

        if self.attn_method == "flash-attn":
            X = pad_input(X, indices, batch_size, seq_len)

        # pre norm
        #   default: (LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add)n-layer -> LN
        #   fuse_dropout_add_ln: (Dropout_Add_LN -> MHA -> Dropout_Add_LN -> MLP)n-layer -> Dropout_Add_LN
        if self.prenorm:
            X = self.extra_layernorm(X)

        return {
            "output_hidden_states": X,
            "attn_mtxs": attn_mtxs,
            "mid_hidden_states": mid_hidden_states
        }
        # return X, attn_mtxs, mid_hidden_states

    def freeze_n_layers(self, n: int):
        for layer in self.layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        position_embedding_type: Optional[str] = None,
        emb_dim: int = 512,
        vocab_size: int = 16,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        base: Optional[int] = None
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        self.position_embeddings = None
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=max_seq_len, emb_dim=emb_dim, pad_token_id=pad_token_id,
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000 if base is None else base,
            )
        else:
            self.position_embeddings = None

    def forward(self, input_ids: torch.Tensor):
        r"""
        Parameters
        ----------------------------------------
        input_ids: torch.Tensor
            shape of [batch_size, seq_len]
        """
        word_embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape)
            position_embeddings = self.position_embeddings(position_ids)
            return word_embeddings + position_embeddings
        return word_embeddings


class TransformerModel(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        activation_checkpointing: bool = False
    ):
        super().__init__()
        self.config = config

        # position embedding
        absolute_pe_type, relative_pe_type = None, None
        if config.position_emb_type is None:
            logger.warning("No position embedding is being used.")
        elif isinstance(config.position_emb_type, str):
            if config.position_emb_type.startswith("absolute"):
                absolute_pe_type = config.position_emb_type
            elif config.position_emb_type.startswith("relative"):
                relative_pe_type = config.position_emb_type
            else:
                raise ValueError(f"Unrecognized position embedding type: {config.position_emb_type}")
        elif isinstance(config.position_emb_type, list):
            for pe_type in config.position_emb_type:
                if absolute_pe_type is None and pe_type.startswith("absolute"):
                    absolute_pe_type = pe_type
                elif relative_pe_type is None and pe_type.startswith("relative"):
                    relative_pe_type = pe_type
                else:
                    raise ValueError("Only one for each kind (absolute/relative) of "
                                     "position embedding is allowed.")
        else:
            raise ValueError(f"Unrecognized position embedding type: {config.position_emb_type}")

        absolute_pe_kwargs, relative_pe_kwargs = None, None
        if config.PE_kwargs is not None:
            if absolute_pe_type is not None and relative_pe_type is not None:   # dual position embedding
                assert absolute_pe_type in config.PE_kwargs or relative_pe_type in config.PE_kwargs, \
                    "PE_kwargs for dual position embedding should contain pe type as key."
                absolute_pe_kwargs = config.PE_kwargs.get(absolute_pe_type, None)
                relative_pe_kwargs = config.PE_kwargs.get(relative_pe_type, None)
            else:    # single position embedding
                if absolute_pe_type is not None:
                    if absolute_pe_type in config.PE_kwargs:
                        absolute_pe_kwargs = config.PE_kwargs[absolute_pe_type]
                    else:
                        absolute_pe_kwargs = config.PE_kwargs
                if relative_pe_type is not None:
                    if relative_pe_type in config.PE_kwargs:
                        relative_pe_kwargs = config.PE_kwargs[relative_pe_type]
                    else:
                        relative_pe_kwargs = config.PE_kwargs

        self.embeddings = TransformerEmbedding(
            absolute_pe_type, vocab_size=config.vocab_size,
            max_seq_len=config.max_seqlen, emb_dim=config.hidden_size,
            pad_token_id=config.pad_token_id,
            base=absolute_pe_kwargs.get("base", None) if absolute_pe_kwargs is not None else None
        )

        if relative_pe_type is None:
            self.relative_pe = None
        else:
            relative_pe_kwargs = dict() if relative_pe_kwargs is None else relative_pe_kwargs.copy()
            if relative_pe_type == "relative_RoPE":
                use_FA_triton_kernel = relative_pe_kwargs.pop("use_FA_triton_kernel", None)
                if use_FA_triton_kernel is None:
                    use_FA_triton_kernel = config.attn_method == "flash-attn"
                if use_FA_triton_kernel:
                    assert is_flash_attn_2_available(), "FA triton kernel needs flash-attn"
                if config.pack_level == "pack_qkv" and config.attn_method == "flash-attn":
                    forward_mode = "qkv_inplace"
                else:
                    forward_mode = "qk"
                self.relative_pe = RotaryEmbedding(
                    head_dim=config.hidden_size // config.num_attn_heads \
                        if config.per_head_size is None else config.per_head_size,
                    max_seqlen=config.max_seqlen,
                    use_FA_triton_kernel=use_FA_triton_kernel,
                    forward_mode=forward_mode,
                    **relative_pe_kwargs
                )
            elif config.position_emb_type == "relative_ALiBi":
                self.relative_pe = AlibiEmbedding(
                    num_heads=config.num_attn_heads, max_seqlen=config.max_seqlen,
                    **relative_pe_kwargs
                )
            elif config.position_emb_type is not None and "relative_learned" in config.position_emb_type:
                self.relative_pe = RelativeLearnedEmbedding(
                    num_heads=config.num_attn_heads, max_seqlen=config.max_seqlen,
                    use_T5_bucket="T5" in config.position_emb_type,
                    **relative_pe_kwargs
                )
            else:
                raise ValueError(f"Unrecognized relative position embedding type: {config.position_emb_type}")

        self.encoder = TransformerEncoder(
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attn_heads,
            prenorm=config.prenorm,
            W_qkv_bias=config.W_qkv_bias,
            proj_bias=config.proj_bias,
            FFN_type=config.FFN_type,
            FFN_bias=config.FFN_bias,
            FFN_size=config.FFN_size,
            attn_method=config.attn_method,
            fuse_dropout_add_ln=config.fuse_dropout_add_ln,
            layer_norm_type=config.layernorm_type,
            layer_norm_eps=config.layernorm_eps,
            attn_dropout=config.attn_dropout,
            residual_dropout=config.residual_dropout,
            pack_level=config.pack_level,
            position_emb=self.relative_pe,
            activation_checkpointing=activation_checkpointing
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        return_attn_mtx: bool = False,
        average_attn: bool = False,
        return_mid_hidden: bool = False,
        **kwargs
    ):
        X = self.embeddings(input_ids)
        outputs = self.encoder(
            X,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            head_mask=head_mask,
            return_attn_mtx=return_attn_mtx,
            average_attn=average_attn,
            return_mid_hidden=return_mid_hidden
        )

        # Add embedding output layer to mid hidden states outputs
        mid_hidden_states = outputs["mid_hidden_states"]

        return {
            "output_hidden_states": outputs["output_hidden_states"],
            "attn_mtxs": outputs["attn_mtxs"] if return_attn_mtx else None,
            "mid_hidden_states": mid_hidden_states if return_mid_hidden else None
        }

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def freeze_n_layers(self, n: int):
        self.encoder.freeze_n_layers(n)



#####################################
# Models
#####################################

class MHAModel(nn.Module):
    """
    A transformer model without FFN and all residual connections, only MHA.
    """
    def __init__(
        self,
        config: TransformerConfig,
        cnn_config: Optional[CNNMixerConfig] = None,  # whether to cnn process attn mtxs before max pooling
        cls_mid_channels: int | list[int] = 128,
    ):
        super().__init__()
        self.config = config
        assert self.config.num_hidden_layers == 1, "MHAModel only supports 1 layer"
        assert self.config.attn_method == "homemade", "If attn matrix is needed " \
            "homemade attn should be used."

        # need to deal with position embedding manually
        absolute_pe_type, relative_pe_type = None, None
        if config.position_emb_type is None:
            logger.warning("No position embedding is being used.")
        elif isinstance(config.position_emb_type, str):
            if config.position_emb_type.startswith("absolute"):
                absolute_pe_type = config.position_emb_type
            elif config.position_emb_type.startswith("relative"):
                relative_pe_type = config.position_emb_type
            else:
                raise ValueError(f"Unrecognized position embedding type: {config.position_emb_type}")
        else:
            raise ValueError(f"Unrecognized position embedding type: {config.position_emb_type}")

        absolute_pe_kwargs, relative_pe_kwargs = None, None
        if absolute_pe_type is not None:
            absolute_pe_kwargs = config.PE_kwargs
        if relative_pe_type is not None:
            relative_pe_kwargs = config.PE_kwargs

        self.embeddings = TransformerEmbedding(
            absolute_pe_type, vocab_size=config.vocab_size,
            max_seq_len=config.max_seqlen, emb_dim=config.hidden_size,
            pad_token_id=config.pad_token_id,
            base=absolute_pe_kwargs.get("base", None) if absolute_pe_kwargs is not None else None
        )

        if relative_pe_type is None:
            self.relative_pe = None
        else:
            relative_pe_kwargs = dict() if relative_pe_kwargs is None else relative_pe_kwargs.copy()
            if relative_pe_type == "relative_RoPE":
                use_FA_triton_kernel = relative_pe_kwargs.pop("use_FA_triton_kernel", None)
                if use_FA_triton_kernel is None:
                    use_FA_triton_kernel = config.attn_method == "flash-attn"
                if use_FA_triton_kernel:
                    assert is_flash_attn_2_available(), "FA triton kernel needs flash-attn"
                if config.pack_level == "pack_qkv" and config.attn_method == "flash-attn":
                    forward_mode = "qkv_inplace"
                else:
                    forward_mode = "qk"
                self.relative_pe = RotaryEmbedding(
                    head_dim=config.hidden_size // config.num_attn_heads \
                        if config.per_head_size is None else config.per_head_size,
                    max_seqlen=config.max_seqlen,
                    use_FA_triton_kernel=use_FA_triton_kernel,
                    forward_mode=forward_mode,
                    **relative_pe_kwargs
                )
            elif config.position_emb_type == "relative_ALiBi":
                self.relative_pe = AlibiEmbedding(
                    num_heads=config.num_attn_heads, max_seqlen=config.max_seqlen,
                    **relative_pe_kwargs
                )
            elif config.position_emb_type is not None and "relative_learned" in config.position_emb_type:
                self.relative_pe = RelativeLearnedEmbedding(
                    num_heads=config.num_attn_heads, max_seqlen=config.max_seqlen,
                    use_T5_bucket="T5" in config.position_emb_type,
                    **relative_pe_kwargs
                )
            else:
                raise ValueError(f"Unrecognized relative position embedding type: {config.position_emb_type}")

        self.mha = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attn_heads,
            per_head_size=config.per_head_size,
            attn_method=config.attn_method,
            W_qkv_bias=config.W_qkv_bias,
            proj_bias=config.proj_bias,
            is_causal=False,
            is_cross_attention=False,
            position_emb=self.relative_pe,
            pack_level=config.pack_level
        )

        if cnn_config is not None:
            cnn_config.in_channels = config.num_attn_heads
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = config.num_attn_heads

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_channels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
    ):
        X = self.embeddings(input_ids)
        _, attn_mtx, _ = self.mha(
            X,
            key_padding_mask=key_padding_mask,
            return_attn_mtx=True
        )

        if self.cnnmixer is not None:
            attn_mtx = self.cnnmixer(attn_mtx)

        pooled_attn = self.max_pool(attn_mtx).squeeze(-1).squeeze(-1)   # [batch, nheads]

        return self.classifier(pooled_attn, cls_labels)


class TransformerAttnModel(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        cnn_config: Optional[CNNMixerConfig] = None,   # whether to cnn process attn mtxs before max pooling
        cls_mid_channels: int | list[int] = 128,
    ):
        super().__init__()
        self.config = config
        assert self.config.attn_method == "homemade", "If attn matrix is needed " \
            "homemade attn should be used."
        self.transformer = TransformerModel(config)

        if cnn_config is not None:
            cnn_config.in_channels = config.num_attn_heads * config.num_hidden_layers
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = config.num_attn_heads * config.num_hidden_layers

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_channels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            key_padding_mask=key_padding_mask,
            return_attn_mtx=True,
            average_attn=False,
        )
        attn_mtxs = outputs["attn_mtxs"]   # list of [batch, nheads, seqlen, seqlen]
        attn_mtx = torch.cat(attn_mtxs, dim=1)

        if self.cnnmixer is not None:
            attn_mtx = self.cnnmixer(attn_mtx)

        pooled_attn = self.max_pool(attn_mtx).squeeze(-1).squeeze(-1)   # [batch, nheads]

        return self.classifier(pooled_attn, cls_labels)


class TransformerCLSModel(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        cls_mid_channels: int | list[int] = 128,
    ):
        super().__init__()
        self.config = config
        self.transformer = TransformerModel(config)
        self.classifier = BaseClassifier(
            in_features=config.hidden_size,
            mid_features=cls_mid_channels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            key_padding_mask=key_padding_mask,
        )
        hidden_states = outputs["output_hidden_states"][:, 0, ...]   # [batch, hidden_size]
        return self.classifier(hidden_states, cls_labels)
