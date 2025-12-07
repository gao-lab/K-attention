from typing import Literal, Optional

import torch
from torch import nn
from einops import rearrange

from .import_utils import is_flash_attn_2_available


ATTN_METHODS = Literal["homemade", "flash-attn", "pytorch"]
FFN_TYPES = Literal["gelu", "swiglu", "geglu"]
LAYERNORM_TYPES = Literal["normal", "rmsnorm"]

GATED_ACT_FUNCS_MAP = {
    "swiglu": nn.SiLU,
    "geglu": nn.GELU,
}
NORMAL_ACT_FUNCS_MAP = {
    "gelu": nn.GELU,
}
LAYERNORM_MAP = {
    "normal": nn.LayerNorm,
    # "rmsnorm": nn.RMSNorm,
}


def expand_key_padding_mask(key_padding_mask: torch.Tensor, dtype=None, device=None):
    r"""
    expand mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
    and transform 1 -> 0, 0 -> -inf for direct add on attention logits.
    Adapted from huggingface/transformers
    e.g.:
    [[1, 1, 1, 1, 1, 0, 0, 0],
     [1, 1, 1, 0, 0, 0, 0, 0]]
    ->
    [[0, 0, 0, 0, 0, -inf, -inf, -inf],
     [0, 0, 0, -inf, -inf, -inf, -inf, -inf]]
    """
    if dtype is not None:
        SMALL_VALUE = torch.finfo(dtype).min
    else:
        SMALL_VALUE = -10000.0
    padding_mask = torch.full(
        key_padding_mask.shape, SMALL_VALUE, dtype=dtype, device=device
    )
    padding_mask.masked_fill_(key_padding_mask.bool(), 0.0)
    return padding_mask[:, None, None, :]


"""
apply_rotary_emb_qkv_ in flash-attn only support [batch, seqlen, 3, nheads, headdim]
input. Here we extend to support [total_len, 3, nheads, headdim] for integration
with flash-attn
"""
if is_flash_attn_2_available():
    from flash_attn.ops.triton.rotary import apply_rotary
    class ApplyRotaryEmbQKV_(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            qkv,
            cos,
            sin,
            seqlen_offsets: int | torch.Tensor = 0,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,
            interleaved=False,
        ):
            """
            qkv: (batch_size, seqlen, 3, nheads, headdim) or (total_seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            """
            # need to check qkv is not an expanded tensor
            assert not any(s == 0 for s in qkv.stride()), "qkv should not be an expanded tensor"
            if cu_seqlens is None:
                qk = qkv[:, :, :2]
            else:
                qk = qkv[:, :2]
            qk_r = rearrange(qk, "... two h d -> ... (two h) d")
            # for extremely complicated cases (qkv is a noncontiguous tensor with
            # interleaved dims), this rearrange may create new tensor and thus
            # change data_ptr (and break inplace operation) raise error when such
            # case happens
            apply_rotary(
                qk_r,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=interleaved,
                inplace=True
            )
            if isinstance(seqlen_offsets, int):  # cannot save int with save_for_backward
                ctx.save_for_backward(cos, sin, cu_seqlens)
                ctx.seqlen_offsets = seqlen_offsets
            else:
                ctx.save_for_backward(cos, sin, seqlen_offsets, cu_seqlens)
                ctx.seqlen_offsets = None
            ctx.interleaved = interleaved
            ctx.max_seqlen = max_seqlen
            return qkv

        @staticmethod
        def backward(ctx, dqkv):
            assert not any(s == 0 for s in dqkv.stride()), "dqkv should not be an expanded tensor"
            seqlen_offsets = ctx.seqlen_offsets
            if seqlen_offsets is None:
                cos, sin, seqlen_offsets, cu_seqlens = ctx.saved_tensors
            else:
                cos, sin, cu_seqlens = ctx.saved_tensors

            if cu_seqlens is None:   # [batch_size, seqlen, 3, nheads, headdim]
                dqk = dqkv[:, :, :2]
            else:                    # [total_seqlen, 3, nheads, headdim]
                dqk = dqkv[:, :2]
            dqk_r = rearrange(dqk, "... two h d -> ... (two h) d")
            assert dqk.data_ptr() == dqk_r.data_ptr()
            apply_rotary(
                dqk_r,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
            return dqkv, None, None, None, None, None, None


    def apply_rotary_emb_qkv_(
        qkv,
        cos,
        sin,
        seqlen_offsets: int | torch.Tensor = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        interleaved=False,
    ):
        """
        Arguments:
            qkv: (batch_size, seqlen, 3, nheads, headdim) or (total_seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
                1st half and 2nd half (GPT-NeoX style).
            seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
                Most commonly used in inference when we have KV cache.
            cu_seqlens: (batch_size + 1,), optional. Need for varlen flash_attn.
            max_seqlen: int, optional. Need for varlen flash_attn.
        Return:
            qkv: (batch_size, seqlen, 3, nheads, headdim) or (total_seqlen, 3, nheads, headdim)
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
        """
        return ApplyRotaryEmbQKV_.apply(
            qkv,
            cos,
            sin,
            seqlen_offsets,
            cu_seqlens,
            max_seqlen,
            interleaved,
        )
