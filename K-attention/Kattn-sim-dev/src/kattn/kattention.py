import os
from typing import Optional, Literal
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .utils import expand_key_padding_mask
from .modules import BaseClassifier, CNNMixer, CNNMixerConfig, BaseClassifier_reg
import pdb
from .position_embeddings import AlibiEmbedding
import torch.nn.init as init

class KattentionV3(nn.Module):
    def __init__(self, hidden_dim: int, first_kernel_size: int = 10, kernel_cnts: tuple[int] = (32, 64, 128),
                 num_kernels: int = 8, kernel_dim: Optional[int] = None, bias: bool = True,
                 softmax_scale: Optional[float] = None, attn_dropout: float = 0.):
        super().__init__()
        #TODO conv/linear proj share or not share kernel (grouped conv)
        self.hidden_size = hidden_dim
        self.conv_out_dim = kernel_cnts[-1]

        #? is this right, as the reception field is beyond control
        kernel_cnts = [hidden_dim] + list(kernel_cnts)
        kernel_sizes = [first_kernel_size] + [3] * (len(kernel_cnts) - 1)
        self.Wq_conv = nn.Sequential(
            *[nn.Conv1d(
                c1, c2, kernel_size=k, padding="same"
            ) for k, (c1, c2) in zip(kernel_sizes, zip(kernel_cnts[:-1], kernel_cnts[1:]))]
        )
        self.Wk_conv = nn.Sequential(
            *[nn.Conv1d(
                c1, c2, kernel_size=k, padding="same"
            ) for k, (c1, c2) in zip(kernel_sizes, zip(kernel_cnts[:-1], kernel_cnts[1:]))]
        )

        #TODO ? directly as linear proj input, or just like in DeepSeek, use it as a latent space?
        self.num_kernels = num_kernels
        if kernel_dim is None:
            assert self.conv_out_dim % num_kernels == 0, "kernal_cnt should be divisible by num_kernels"
            kernel_dim = self.conv_out_dim // num_kernels
        self.kernel_dim = kernel_dim
        all_kernel_dim = self.num_kernels * kernel_dim

        self.Wq = nn.Linear(self.conv_out_dim, all_kernel_dim, bias=bias)
        self.Wk = nn.Linear(self.conv_out_dim, all_kernel_dim, bias=bias)

        #? not sure where should V rooted from? if we want to stack multiple layers of Kattention
        self.Wv = nn.Linear(hidden_dim, all_kernel_dim, bias=bias)

        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        r"""
        Parameters
        ----------------
        X: torch.Tensor
            Shape: (batch_size, seq_len, hidden_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        X_ = X.transpose(1, 2)

        Q_conv = self.Wq_conv(X_)                # (batch_size, conv_out_dim, seq_len)
        K_conv = self.Wk_conv(X_)
        Q_conv = Q_conv.transpose(1, 2)
        K_conv = K_conv.transpose(1, 2)

        Q = self.Wq(Q_conv)
        K = self.Wk(K_conv)
        V = self.Wv(X)

        Q = rearrange(Q, "b l (h d) -> b l h d", h=self.num_kernels)    # (batch_size, seq_len, num_kernels, kernel_dim)
        K = rearrange(K, "b l (h d) -> b l h d", h=self.num_kernels)
        V = rearrange(V, "b l (h d) -> b l h d", h=self.num_kernels)

        #TODO residue connection?
        softmax_scale = self.softmax_scale or (1 / math.sqrt(self.kernel_dim))
        attn_logits = torch.einsum("bqhd, bkhd -> bhqk", Q, K) * softmax_scale

        if key_padding_mask is not None:
            if key_padding_mask.dim() == 2:  # [batch_size, k_seq_len], masked positions filled with 0
                key_padding_mask = expand_key_padding_mask(key_padding_mask, dtype=Q.dtype, device=Q.device)
            else:
                assert key_padding_mask.dim() == 4, "key_padding_mask must be of dim 2 or 4"   # [batch_size, 1, 1, k_seq_len], masked positions filled with -inf
        attn_logits.add_(key_padding_mask)

        attn_probs = F.softmax(attn_logits, dim=-1)

        output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, V)

        return {
            "output_hidden_states": output,
            "attn_probs": attn_probs,
            "attn_logits": attn_logits
        }

class KattentionV4(nn.Module):
    def __init__(
        self, channel_size: int, kernel_size: int = 10, num_kernels: int = 32,
        bias: bool = False, softmax_scale: Optional[float] = None, attn_dropout: float = 0.,
        reverse: bool = False
    ):
        super().__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        # Wattn inplace of Wq.T \codt Wk, by group conv
        self.Wattn = nn.Conv1d(
            in_channels=kernel_size * channel_size,
            out_channels=kernel_size * channel_size * num_kernels,
            kernel_size=1,
            groups=kernel_size,
            bias=bias
        )

        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)

        self.reverse = reverse

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        r"""
        Parameters
        ----------------
        X: torch.Tensor
            Shape: (batch_size, seq_len, hidden_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        if self.reverse:
            X_rev = X.flip([1])
            X_rev = rearrange(X_rev, "b l c -> b (l c)")
        X = rearrange(X, "b l c -> b (l c)")

        Q = X.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)    # (batch_size, seq_len - k + 1, kernel_size * channel_size)
        if self.reverse:
            K = X_rev.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        else:
            K = X.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        Q = Q.transpose(1, 2)    # (batch_size, kernel_size * channel_size, seq_len - k + 1)

        # attention operation as depthwise conv
        Q_W = self.Wattn(Q)          # (batch_size, kernel_size * channel_size * num_kernels, seq_len - k + 1)
        Q_W = rearrange(Q_W, "b (k h c) l -> b l h (k c)", k=self.kernel_size, c=self.channel_size)
        attn_logits = torch.einsum("bQhD,bKD->bhQK", Q_W, K)

        # Q_W = rearrange(Q_W, "b Q h D -> b h Q D")
        # K = K.unsqueeze(1)
        # attn_logits = Q_W @ K.transpose(-1, -2)

        #TODO: currently no padding
        #TODO: key_padding_mask, need rethinking, as the padding tokens may not be the same (conv padding and tokenizer padding)
        # if key_padding_mask is not None:
        #     if key_padding_mask.dim() == 2:  # [batch_size, k_seq_len], masked positions filled with 0
        #         key_padding_mask = expand_key_padding_mask(key_padding_mask, dtype=attn_logits.dtype, device=attn_logits.device)
        #     else:
        #         assert key_padding_mask.dim() == 4, "key_padding_mask must be of dim 2 or 4"   # [batch_size, 1, 1, k_seq_len], masked positions filled with -inf
        # attn_logits.add_(key_padding_mask)

        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        return {
            "attn_probs": attn_probs,
            "attn_logits": attn_logits,
            "Q_W": Q_W,
            "K": K
        }

class KattentionV4_uncons(nn.Module):
    def __init__(
        self, channel_size: int, kernel_size: int = 10, num_kernels: int = 32,
        bias: bool = False, softmax_scale: Optional[float] = None, attn_dropout: float = 0.,
        reverse: bool = False
    ):
        super().__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        # Wattn inplace of Wq.T \codt Wk, by group conv
        self.Wattn = nn.Conv1d(
            in_channels=kernel_size * channel_size,
            out_channels=kernel_size * channel_size * num_kernels,
            kernel_size=1,
            bias=bias
        )

        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)

        self.reverse = reverse

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        r"""
        Parameters
        ----------------
        X: torch.Tensor
            Shape: (batch_size, seq_len, hidden_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        if self.reverse:
            X_rev = X.flip([1])
            X_rev = rearrange(X_rev, "b l c -> b (l c)")
        X = rearrange(X, "b l c -> b (l c)")

        Q = X.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)    # (batch_size, seq_len - k + 1, kernel_size * channel_size)
        if self.reverse:
            K = X_rev.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        else:
            K = X.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        Q = Q.transpose(1, 2)    # (batch_size, kernel_size * channel_size, seq_len - k + 1)

        # attention operation as depthwise conv
        Q_W = self.Wattn(Q)          # (batch_size, kernel_size * channel_size * num_kernels, seq_len - k + 1)
        Q_W = rearrange(Q_W, "b (k h c) l -> b l h (k c)", k=self.kernel_size, c=self.channel_size)
        attn_logits = torch.einsum("bQhD,bKD->bhQK", Q_W, K)

        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        return {
            "attn_probs": attn_probs,
            "attn_logits": attn_logits,
            "Q_W": Q_W,
            "K": K
        }

class KattentionV5(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        # input preprocessing
        mixer_channels_per_kernel: int = 16,
        qk_share_in_mixer: bool = False,
        kernel_size: int = 10,
        num_kernels: int = 32,
        bias: bool = False,
        softmax_scale: Optional[float] = None,
        attn_dropout: float = 0.,
        reverse: bool = False,
    ):
        super().__init__()
        self.mixer_channels_per_kernel = mixer_channels_per_kernel
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        # input mixer (also split kernels)
        # TODO: can we introduce other mixers to solve problems like bubbles?
        self.q_in_mixer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=mixer_channels_per_kernel * num_kernels,
                kernel_size=1,
                padding=0,
                bias=bias
            ),
            nn.BatchNorm1d(mixer_channels_per_kernel * num_kernels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=mixer_channels_per_kernel * num_kernels,
                out_channels=mixer_channels_per_kernel * num_kernels,
                kernel_size=1,
                padding=0,
                groups=num_kernels,
                bias=bias
            ),
        )
        if qk_share_in_mixer:
            self.k_in_mixer = self.q_in_mixer
        else:
            self.k_in_mixer = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=mixer_channels_per_kernel * num_kernels,
                    kernel_size=1,
                    padding=0,
                    bias=bias
                ),
                nn.BatchNorm1d(mixer_channels_per_kernel * num_kernels),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    in_channels=mixer_channels_per_kernel * num_kernels,
                    out_channels=mixer_channels_per_kernel * num_kernels,
                    kernel_size=1,
                    padding=0,
                    groups=num_kernels,
                    bias=bias
                ),
            )

        # Wattn inplace of Wq.T \codt Wk, by group conv
        self.Wattn = nn.Conv1d(
            in_channels=kernel_size * num_kernels * mixer_channels_per_kernel,
            out_channels=kernel_size * num_kernels * mixer_channels_per_kernel,
            kernel_size=1,
            groups=kernel_size * num_kernels,
            bias=bias
        )

        # TODO: softmax_scale is not used
        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)

        self.reverse = reverse

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.Wattn.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, X: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        r"""
        Parameters
        ----------------
        X: torch.Tensor
            Shape: (batch_size, seq_len, in_channels)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        X = X.transpose(1, 2)            # (batch_size, in_channels, seq_len)

        # input mixing
        mixed_q = self.q_in_mixer(X)     # (batch_size, num_kernels * mixer_channels_per_kernel, seq_len)
        mixed_k = self.k_in_mixer(X.flip([2])) if self.reverse else self.k_in_mixer(X)

        # Note: we cannot use nn.Unfold as it only support 4d tensor now
        mixed_q = rearrange(mixed_q, "b (h c) l -> b (h c l)", h=self.num_kernels, c=self.mixer_channels_per_kernel)
        mixed_k = rearrange(mixed_k, "b (h c) l -> b (h c l)", h=self.num_kernels, c=self.mixer_channels_per_kernel)

        Q = mixed_q.unfold(
            dimension=1,
            size=self.kernel_size * self.num_kernels * self.mixer_channels_per_kernel,
            step=self.num_kernels * self.mixer_channels_per_kernel
        )    # (batch_size, seq_len - kernel_size + 1, kernel_size * num_kernels * mixer_channels_per_kernel)
        K = mixed_k.unfold(
            dimension=1,
            size=self.kernel_size * self.num_kernels * self.mixer_channels_per_kernel,
            step=self.num_kernels * self.mixer_channels_per_kernel
        )

        # attention operation as depthwise conv
        Q = Q.transpose(1, 2)        # (batch_size, kernel_size * num_kernels * mixer_channels_per_kernel, seq_len - k + 1)
        Q_W = self.Wattn(Q)          # (batch_size, kernel_size * num_kernels * mixer_channels_per_kernel, seq_len - k + 1)
        Q_W = rearrange(Q_W, "b (k h c) l -> b l h (k c)", k=self.kernel_size, c=self.mixer_channels_per_kernel)
        K = rearrange(K, "b l (k h c)-> b l h (k c)", k=self.kernel_size, c=self.mixer_channels_per_kernel)

        # Q_W_K
        attn_logits = torch.einsum("bQhD,bKhD->bhQK", Q_W, K)

        #TODO: currently no padding
        #TODO: key_padding_mask, need rethinking, as the padding tokens may not be the same (conv padding and tokenizer padding)
        # if key_padding_mask is not None:
        #     if key_padding_mask.dim() == 2:  # [batch_size, k_seq_len], masked positions filled with 0
        #         key_padding_mask = expand_key_padding_mask(key_padding_mask, dtype=attn_logits.dtype, device=attn_logits.device)
        #     else:
        #         assert key_padding_mask.dim() == 4, "key_padding_mask must be of dim 2 or 4"   # [batch_size, 1, 1, k_seq_len], masked positions filled with -inf
        # attn_logits.add_(key_padding_mask)

        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        return {
            "attn_probs": attn_probs,
            "attn_logits": attn_logits,
            "Q_W": Q_W,
            "K": K
        }

            # self.kattn = KattentionV6(
            #     in_channels=input_embedding_dim,
            #     mixer_channels_per_kernel=16,
            #     kernel_size=kernel_size,
            #     num_kernels=num_kernels,
            #     reverse="rev" in kattn_version,
            # )

class KattentionV6(nn.Module):
    def __init__(
        self, channel_size: int, kernel_size: int = 10, num_kernels: int = 32,
        bias: bool = False, softmax_scale: Optional[float] = None, attn_dropout: float = 0.,
        reverse: bool = False
    ):
        super().__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels

        # Wattn inplace of Wq.T \codt Wk, by group conv
        self.Wattn = nn.Conv1d(
            in_channels=kernel_size * channel_size,
            out_channels=kernel_size * channel_size * num_kernels,
            kernel_size=1,
            groups=kernel_size,
            bias=bias
        )

        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)

        self.reverse = reverse
    #     # 初始化权重
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     """初始化所有权重"""
    #     # 针对Wattn层的特定初始化
    #     init.xavier_uniform_(self.Wattn.weight)
    #     if self.Wattn.bias is not None:
    #         init.constant_(self.Wattn.bias, 0.0)

        # 如果有其他层，继续初始化...
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0.0)
    def forward(self, X1: torch.Tensor, X2: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        r"""
        Parameters
        ----------------
        X: torch.Tensor
            Shape: (batch_size, seq_len, hidden_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        if self.reverse:
            X_rev = X1.flip([1])
            X_rev = rearrange(X_rev, "b l c -> b (l c)")
        X1 = rearrange(X1, "b l c -> b (l c)")
        X2 = rearrange(X2, "b l c -> b (l c)")
        Q = X1.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)    # (batch_size, seq_len - k + 1, kernel_size * channel_size)
        if self.reverse:
            K = X_rev.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        else:
            K = X2.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        Q = Q.transpose(1, 2)    # (batch_size, kernel_size * channel_size, seq_len - k + 1)

        # attention operation as depthwise conv
        Q_W = self.Wattn(Q)          # (batch_size, kernel_size * channel_size * num_kernels, seq_len - k + 1)
        Q_W = rearrange(Q_W, "b (k h c) l -> b l h (k c)", k=self.kernel_size, c=self.channel_size)
        attn_logits = torch.einsum("bQhD,bKD->bhQK", Q_W, K)

        # Q_W = rearrange(Q_W, "b Q h D -> b h Q D")
        # K = K.unsqueeze(1)
        # attn_logits = Q_W @ K.transpose(-1, -2)

        #TODO: currently no padding
        #TODO: key_padding_mask, need rethinking, as the padding tokens may not be the same (conv padding and tokenizer padding)
        # if key_padding_mask is not None:
        #     if key_padding_mask.dim() == 2:  # [batch_size, k_seq_len], masked positions filled with 0
        #         key_padding_mask = expand_key_padding_mask(key_padding_mask, dtype=attn_logits.dtype, device=attn_logits.device)
        #     else:
        #         assert key_padding_mask.dim() == 4, "key_padding_mask must be of dim 2 or 4"   # [batch_size, 1, 1, k_seq_len], masked positions filled with -inf
        # attn_logits.add_(key_padding_mask)

        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        return {
            "attn_probs": attn_probs,
            "attn_logits": attn_logits,
            "Q_W": Q_W,
            "K": K
        }

class KattentionV6_test(nn.Module):
    def __init__(
        self,
        embedding_method: Literal["onehot", "learned"] = "onehot",
        kattn_version: str = "v6",
        vocab_size: int = 10,
        kernel_size: int = 10,
        num_kernels: int = 32,
        cnn_config: Optional[CNNMixerConfig] = None,
        cls_mid_features: int | list[int] = 128,
    ):
        super().__init__()
        if embedding_method == "learned":
            self.embedding = nn.Embedding(vocab_size, 128)
            input_embedding_dim = 128
        elif embedding_method == "onehot":
            self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
            input_embedding_dim = vocab_size
        else:
            raise ValueError(f"embedding_method {embedding_method} not supported")

        if kattn_version.startswith("v6"):
            self.kattn = KattentionV6(
                channel_size=input_embedding_dim,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        else:
            raise ValueError(f"kattn_version {kattn_version} not supported")


        if cnn_config is not None:
            cnn_config.in_channels = num_kernels
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = num_kernels*3

        self.kattn1 = KattentionV4(
                channel_size=input_embedding_dim,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        self.kattn2 = KattentionV4(
                channel_size=input_embedding_dim,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
            
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_features,
        )
        # self.classifier = nn.Sequential(
        #     torch.nn.Linear(num_kernels, 1)
        # )

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        r"""
        Parameters
        ----------------
        input_ids: torch.Tensor
            Shape: (batch_size, seq_len)
        labels: torch.Tensor
            Shape: (batch_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        #TODO, position encoding
        # X = self.embedding(input_ids)
        X1 = self.embedding(X1.to(dtype=torch.long))
        X2 = self.embedding(X2.to(dtype=torch.long))

        attn_logits0 = self.kattn(X1, X2, key_padding_mask)["attn_logits"]
        attn_logits1 = self.kattn1(X1, key_padding_mask)["attn_logits"]
        attn_logits2 = self.kattn2(X2, key_padding_mask)["attn_logits"]
        # if self.cnnmixer is not None:
        #     attn_logits = self.cnnmixer(attn_logits)

        pooled_attn0 = self.max_pool(attn_logits0).squeeze(-1).squeeze(-1)
        pooled_attn1 = self.max_pool(attn_logits1).squeeze(-1).squeeze(-1)
        pooled_attn2 = self.max_pool(attn_logits2).squeeze(-1).squeeze(-1)
        pooled_attn = torch.cat([pooled_attn0, pooled_attn1, pooled_attn2], dim=1)
        return self.classifier(pooled_attn, cls_labels)
        # return self.classifier(pooled_attn)

class KattentionModel(nn.Module):
    def __init__(
        self,
        embedding_method: Literal["onehot", "learned"] = "onehot",
        kattn_version: str = "v4",
        vocab_size: int = 10,
        kernel_size: int = 10,
        num_kernels: int = 32,
        cnn_config: Optional[CNNMixerConfig] = None,
        cls_mid_features: int | list[int] = 128,
    ):
        super().__init__()
        if embedding_method == "learned":
            self.embedding = nn.Embedding(vocab_size, 128)
            input_embedding_dim = 128
        elif embedding_method == "onehot":
            self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
            input_embedding_dim = vocab_size
        else:
            raise ValueError(f"embedding_method {embedding_method} not supported")

        if kattn_version == "v3":
            self.kattn = KattentionV3(
                hidden_dim=input_embedding_dim,
                first_kernel_size=kernel_size,
                num_kernel=num_kernels
            )
        elif kattn_version.startswith("v4"):
            self.kattn = KattentionV4(
                channel_size=input_embedding_dim,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        elif kattn_version.startswith("v5"):
            self.kattn = KattentionV5(
                in_channels=input_embedding_dim,
                mixer_channels_per_kernel=16,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        else:
            raise ValueError(f"kattn_version {kattn_version} not supported")

        if cnn_config is not None:
            cnn_config.in_channels = num_kernels
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = num_kernels

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_features,
        )
        # self.classifier = nn.Sequential(
        #     torch.nn.Linear(num_kernels, 1)
        # )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        r"""
        Parameters
        ----------------
        input_ids: torch.Tensor
            Shape: (batch_size, seq_len)
        labels: torch.Tensor
            Shape: (batch_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        #TODO, position encoding
        X = self.embedding(input_ids)
        attn_logits = self.kattn(X, key_padding_mask)["attn_logits"]

        if self.cnnmixer is not None:
            attn_logits = self.cnnmixer(attn_logits)

        pooled_attn = self.max_pool(attn_logits).squeeze(-1).squeeze(-1)
        return self.classifier(pooled_attn, cls_labels)
        # return self.classifier(pooled_attn)

class KattentionModel_pos(nn.Module):
    def __init__(
        self,
        embedding_method: Literal["onehot", "learned"] = "onehot",
        kattn_version: str = "v4",
        vocab_size: int = 10,
        kernel_size: int = 10,
        num_kernels: int = 32,
        cnn_config: Optional[CNNMixerConfig] = None,
        cls_mid_features: int | list[int] = 128,
    ):
        super().__init__()
        if embedding_method == "learned":
            self.embedding = nn.Embedding(vocab_size, 128)
            input_embedding_dim = 128
        elif embedding_method == "onehot":
            self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
            input_embedding_dim = vocab_size
        else:
            raise ValueError(f"embedding_method {embedding_method} not supported")

        if kattn_version == "v3":
            self.kattn = KattentionV3(
                hidden_dim=input_embedding_dim,
                first_kernel_size=kernel_size,
                num_kernel=num_kernels
            )
        elif kattn_version.startswith("v4"):
            self.kattn = KattentionV4(
                channel_size=input_embedding_dim,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        elif kattn_version.startswith("v5"):
            self.kattn = KattentionV5(
                in_channels=input_embedding_dim,
                mixer_channels_per_kernel=16,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        else:
            raise ValueError(f"kattn_version {kattn_version} not supported")

        if cnn_config is not None:
            cnn_config.in_channels = num_kernels
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = num_kernels

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_features,
        )
        # self.classifier = nn.Sequential(
        #     torch.nn.Linear(num_kernels, 1)
        # )
        self.relative_pe = AlibiEmbedding(
            num_heads=num_kernels, max_seqlen=100,
        )
    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        r"""
        Parameters
        ----------------
        input_ids: torch.Tensor
            Shape: (batch_size, seq_len)
        labels: torch.Tensor
            Shape: (batch_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        #TODO, position encoding
        X = self.embedding(input_ids)
        attn_logits = self.kattn(X, key_padding_mask)["attn_logits"]
        attn_logits = self.relative_pe(attn_logits)[0]
        if self.cnnmixer is not None:
            attn_logits = self.cnnmixer(attn_logits)
        pooled_attn = self.max_pool(attn_logits).squeeze(-1).squeeze(-1)
        return self.classifier(pooled_attn, cls_labels)
        # return self.classifier(pooled_attn)

class KattentionModel_uncons(nn.Module):
    def __init__(
        self,
        embedding_method: Literal["onehot", "learned"] = "onehot",
        kattn_version: str = "v4",
        vocab_size: int = 10,
        kernel_size: int = 10,
        num_kernels: int = 32,
        cnn_config: Optional[CNNMixerConfig] = None,
        cls_mid_features: int | list[int] = 128,
    ):
        super().__init__()
        if embedding_method == "learned":
            self.embedding = nn.Embedding(vocab_size, 128)
            input_embedding_dim = 128
        elif embedding_method == "onehot":
            self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
            input_embedding_dim = vocab_size
        else:
            raise ValueError(f"embedding_method {embedding_method} not supported")

        self.kattn = KattentionV4_uncons(
            channel_size=input_embedding_dim,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
            reverse="rev" in kattn_version,
        )

        if cnn_config is not None:
            cnn_config.in_channels = num_kernels
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = num_kernels

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_features,
        )
        # self.classifier = nn.Sequential(
        #     torch.nn.Linear(num_kernels, 1)
        # )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        r"""
        Parameters
        ----------------
        input_ids: torch.Tensor
            Shape: (batch_size, seq_len)
        labels: torch.Tensor
            Shape: (batch_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        #TODO, position encoding
        X = self.embedding(input_ids)
        attn_logits = self.kattn(X, key_padding_mask)["attn_logits"]

        if self.cnnmixer is not None:
            attn_logits = self.cnnmixer(attn_logits)

        pooled_attn = self.max_pool(attn_logits).squeeze(-1).squeeze(-1)
        return self.classifier(pooled_attn, cls_labels)

class KattentionModel_mask(nn.Module):
    def __init__(
        self,
        embedding_method: Literal["onehot", "learned"] = "onehot",
        kattn_version: str = "v4",
        vocab_size: int = 10,
        kernel_size: int = 10,
        num_kernels: int = 32,
        cnn_config: Optional[CNNMixerConfig] = None,
        cls_mid_features: int | list[int] = 128,
    ):
        super().__init__()
        if embedding_method == "learned":
            self.embedding = nn.Embedding(vocab_size, 128)
            input_embedding_dim = 128
        elif embedding_method == "onehot":
            self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
            input_embedding_dim = vocab_size
        else:
            raise ValueError(f"embedding_method {embedding_method} not supported")

        if kattn_version == "v3":
            self.kattn = KattentionV3(
                hidden_dim=input_embedding_dim,
                first_kernel_size=kernel_size,
                num_kernel=num_kernels
            )
        elif kattn_version.startswith("v4"):
            self.kattn = KattentionV4(
                channel_size=input_embedding_dim,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        elif kattn_version.startswith("v5"):
            self.kattn = KattentionV5(
                in_channels=input_embedding_dim,
                mixer_channels_per_kernel=16,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        else:
            raise ValueError(f"kattn_version {kattn_version} not supported")

        if cnn_config is not None:
            cnn_config.in_channels = num_kernels
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = num_kernels

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_features,
        )
        # self.classifier = nn.Sequential(
        #     torch.nn.Linear(num_kernels, 1)
        # )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        r"""
        Parameters
        ----------------
        input_ids: torch.Tensor
            Shape: (batch_size, seq_len)
        labels: torch.Tensor
            Shape: (batch_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        #TODO, position encoding
        X = self.embedding(input_ids)
        attn_logits = self.kattn(X, key_padding_mask)["attn_logits"]
        len_ = attn_logits.shape[-1]
        mask = torch.eye(len_, dtype=torch.int32)  # 创建对角线为 1 的矩阵
        mask += torch.diag(torch.ones(len_ - 1, dtype=torch.int32), diagonal=1)  # 上方一位
        mask += torch.diag(torch.ones(len_ - 1, dtype=torch.int32), diagonal=-1)  # 下方一位
        mask += torch.diag(torch.ones(len_ - 2, dtype=torch.int32), diagonal=2)  # 上方一位
        mask += torch.diag(torch.ones(len_ - 2, dtype=torch.int32), diagonal=-2)  # 下方一位
        mask = mask.cuda()
        attn_logits = attn_logits*mask
        if self.cnnmixer is not None:
            attn_logits = self.cnnmixer(attn_logits)

        pooled_attn = self.max_pool(attn_logits).squeeze(-1).squeeze(-1)
        return self.classifier(pooled_attn, cls_labels)
        # return self.classifier(pooled_attn)

class KattentionModel_uncons_mask(nn.Module):
    def __init__(
        self,
        embedding_method: Literal["onehot", "learned"] = "onehot",
        kattn_version: str = "v4",
        vocab_size: int = 10,
        kernel_size: int = 10,
        num_kernels: int = 32,
        cnn_config: Optional[CNNMixerConfig] = None,
        cls_mid_features: int | list[int] = 128,
    ):
        super().__init__()
        if embedding_method == "learned":
            self.embedding = nn.Embedding(vocab_size, 128)
            input_embedding_dim = 128
        elif embedding_method == "onehot":
            self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
            input_embedding_dim = vocab_size
        else:
            raise ValueError(f"embedding_method {embedding_method} not supported")

        self.kattn = KattentionV4_uncons(
            channel_size=input_embedding_dim,
            kernel_size=kernel_size,
            num_kernels=num_kernels,
            reverse="rev" in kattn_version,
        )

        if cnn_config is not None:
            cnn_config.in_channels = num_kernels
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = num_kernels

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = BaseClassifier(
            in_features=cls_in_dim,
            mid_features=cls_mid_features,
        )
        # self.classifier = nn.Sequential(
        #     torch.nn.Linear(num_kernels, 1)
        # )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        r"""
        Parameters
        ----------------
        input_ids: torch.Tensor
            Shape: (batch_size, seq_len)
        labels: torch.Tensor
            Shape: (batch_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        #TODO, position encoding
        X = self.embedding(input_ids)
        attn_logits = self.kattn(X, key_padding_mask)["attn_logits"]
        len_ = attn_logits.shape[-1]
        mask = torch.eye(len_, dtype=torch.int32)  # 创建对角线为 1 的矩阵
        mask += torch.diag(torch.ones(len_ - 1, dtype=torch.int32), diagonal=1)  # 上方一位
        mask += torch.diag(torch.ones(len_ - 1, dtype=torch.int32), diagonal=-1)  # 下方一位
        mask += torch.diag(torch.ones(len_ - 2, dtype=torch.int32), diagonal=2)  # 上方一位
        mask += torch.diag(torch.ones(len_ - 2, dtype=torch.int32), diagonal=-2)  # 下方一位
        mask = mask.cuda()
        attn_logits = attn_logits*mask
        if self.cnnmixer is not None:
            attn_logits = self.cnnmixer(attn_logits)

        pooled_attn = self.max_pool(attn_logits).squeeze(-1).squeeze(-1)
        return self.classifier(pooled_attn, cls_labels)
        # return self.classifier(pooled_attn)

class Conv_layer(nn.Module):
    def __init__(self, filters1=48, filters2=24, kernel_size=3,dilation_rate=1, dropout=0):
        super().__init__()
        self.filters1 = filters1
        self.filters2 = filters2
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.dropout = dropout
        layers = []
        layers.append(
            nn.Sequential(
                nn.BatchNorm2d(self.filters1),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.filters1,
                    out_channels=self.filters1//2,
                    kernel_size=1,
                    padding=0,
                ),
                nn.BatchNorm2d(self.filters1//2),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = self.filters1//2,
                    out_channels = self.filters2,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size//2,
                    # bias=False
                ),
                nn.BatchNorm2d(self.filters2),
                nn.ReLU(),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

# class KNET(nn.Module):
#     def __init__(
#         self,
#         embedding_method: Literal["onehot", "learned"] = "onehot",
#         kattn_version: str = "v4",
#         vocab_size: int = 10,
#         kernel_size: int = 12,
#         num_kernels: int = 64,
#         cnn_config: Optional[CNNMixerConfig] = None,
#         cls_mid_features: int | list[int] = 128,
#     ):
#         super().__init__()
#
#         self.conv = Conv_layer(filters1=num_kernels, filters2=num_kernels,kernel_size=3)
#
#         if embedding_method == "learned":
#             self.embedding = nn.Embedding(vocab_size, 128)
#             input_embedding_dim = 128
#         elif embedding_method == "onehot":
#             self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
#             input_embedding_dim = vocab_size
#         else:
#             raise ValueError(f"embedding_method {embedding_method} not supported")
#
#         if kattn_version == "v3":
#             self.kattn = KattentionV3(
#                 hidden_dim=input_embedding_dim,
#                 first_kernel_size=kernel_size,
#                 num_kernel=num_kernels
#             )
#         elif kattn_version.startswith("v4"):
#             self.kattn = KattentionV4(
#                 channel_size=input_embedding_dim,
#                 kernel_size=kernel_size,
#                 num_kernels=num_kernels,
#                 reverse="rev" in kattn_version,
#             )
#         elif kattn_version.startswith("v5"):
#             self.kattn = KattentionV5(
#                 in_channels=input_embedding_dim,
#                 mixer_channels_per_kernel=16,
#                 kernel_size=kernel_size,
#                 num_kernels=num_kernels,
#                 reverse="rev" in kattn_version,
#             )
#         else:
#             raise ValueError(f"kattn_version {kattn_version} not supported")
#
#         if cnn_config is not None:
#             cnn_config.in_channels = num_kernels
#             self.cnnmixer = CNNMixer(cnn_config)
#             cls_in_dim = self.cnnmixer.out_channels
#         else:
#             self.cnnmixer = None
#             cls_in_dim = num_kernels
#
#         self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
#         self.classifier = BaseClassifier(
#             in_features=num_kernels,
#             mid_features=cls_mid_features,
#         )
#         # self.classifier = nn.Sequential(
#         #     torch.nn.Linear(num_kernels, 1)
#         # )
#
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         cls_labels: torch.Tensor,
#         key_padding_mask: Optional[torch.Tensor] = None
#     ):
#         r"""
#         Parameters
#         ----------------
#         input_ids: torch.Tensor
#             Shape: (batch_size, seq_len)
#         labels: torch.Tensor
#             Shape: (batch_size)
#         key_padding_mask: torch.Tensor, optional
#             Shape: (batch_size, seq_len), padding positions filled with 0
#         """
#         #TODO, position encoding
#         X = self.embedding(input_ids)
#         attn_logits = self.kattn(X, key_padding_mask)["attn_logits"]
#
#         if self.cnnmixer is not None:
#             attn_logits = self.cnnmixer(attn_logits)
#
#         # x1 = self.conv1(attn_logits)
#         # x2 = self.conv2(attn_logits)
#         # x3 = self.conv3(attn_logits)
#         # x4 = self.conv4(attn_logits)
#         # attns = torch.cat((x1, x2, x3, x4), dim=1)
#         attns = self.conv(attn_logits)
#         pooled_attn = self.max_pool(attns).squeeze(-1).squeeze(-1)
#         return self.classifier(pooled_attn, cls_labels)
#         # return self.classifier(pooled_attn)

class KNET(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                 kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.structure = structure
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
        if self.structure:
            self.Kattention = KattentionV4(channel_size=5, kernel_size=kernel_size, num_kernels=number_of_kernel,
                                           reverse=False)
        else:
            self.Kattention = KattentionV4(channel_size=4, kernel_size=kernel_size, num_kernels=number_of_kernel,reverse=False)
        self.n = number_of_kernel

        self.conv1 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=1)
        self.conv2 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=3)
        self.conv3 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=5)
        self.conv4 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=7)

        self.classifier = BaseClassifier(
            in_features=4*number_of_kernel,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        icshape: Optional[torch.Tensor] = None
    ):
        input_ids = input_ids[:,1:-1]
        x_seq = self.embedding(input_ids)[:,:,-4:]
        if self.structure:
            icshape = icshape.unsqueeze(2)
            X = torch.cat((x_seq, icshape), dim=2)
        else:
            X = x_seq
        x = self.Kattention(X)["attn_logits"]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        pooling1 = torch.max(x,dim=2).values
        pooling2 = torch.max(pooling1,dim=2).values

        return self.classifier(pooling2,cls_labels)

class KNET_uncons(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                 kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.structure = structure
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
        if self.structure:
            self.Kattention = KattentionV4_uncons(channel_size=5, kernel_size=kernel_size, num_kernels=number_of_kernel,
                                           reverse=False)
        else:
            self.Kattention = KattentionV4_uncons(channel_size=4, kernel_size=kernel_size, num_kernels=number_of_kernel,reverse=False)
        self.n = number_of_kernel

        self.conv1 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=1)
        self.conv2 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=3)
        self.conv3 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=5)
        self.conv4 = Conv_layer(filters1=self.n, filters2=self.n,kernel_size=7)

        self.classifier = BaseClassifier(
            in_features=4*number_of_kernel,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        icshape: Optional[torch.Tensor] = None
    ):
        input_ids = input_ids[:,1:-1]
        x_seq = self.embedding(input_ids)[:,:,-4:]
        if self.structure:
            icshape = icshape.unsqueeze(2)
            X = torch.cat((x_seq, icshape), dim=2)
        else:
            X = x_seq
        x = self.Kattention(X.transpose(1, 2))["attn_logits"]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        pooling1 = torch.max(x,dim=2).values
        pooling2 = torch.max(pooling1,dim=2).values

        return self.classifier(pooling2,cls_labels)

# class TransformerCLSTest(nn.Module):
#     def __init__(
#         self,
#         config: TransformerConfig,
#         task: str = "test",
#         cls_mid_channels: int | list[int] = 128,
#         structure: Optional[bool] = None,  # 新增参数
#         use_bias: bool = True
#     ):
#         super().__init__()
#         self.config = config
#         self.task = task
#         self.use_bias = use_bias
#         self.transformer = TransformerModel(config,structure)

#         self.output_heads = nn.ModuleDict()
#         # 主轨道
#         self.output_heads[f'{self.task}_profile_target' if use_bias else f'{self.task}_profile'] = ProfileHead(64*2)

#         # 控制轨道（如果使用偏置）
#         if self.use_bias:
#             self.output_heads[f'{self.task}_profile_control'] = ProfileHead(64*2)
#             self.output_heads[f'{self.task}_mixing_coefficient'] = SequenceAdditiveMixingCoefficient(
#                 64*2, name=f'{self.task}_mixing_coefficient')
#             self.output_heads[f'{self.task}_profile'] = AdditiveTargetBias(name=f'{self.task}_profile')
#         # self.classifier = BaseClassifier(
#         #     in_features=config.hidden_size,
#         #     mid_features=cls_mid_channels,
#         # )


#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         icshape: Optional[torch.Tensor] = None  # 新增参数
#         # key_padding_mask: torch.Tensor,
#     ):
#         # key_padding_mask: torch.Tensor
#         # shape of [batch_size, seq_len], 0 for padding, 1 for real token
#         key_padding_mask = torch.ones(input_ids.shape[:2], dtype=torch.long)
#         outputs = self.transformer(
#             input_ids=input_ids,
#             key_padding_mask=key_padding_mask,
#             icshape=icshape
#         )
#         pdb.set_trace()
#         # hidden_states = outputs["output_hidden_states"][:, 0, ...]   # [batch, hidden_size]
#         return outputs

class kattn_module(nn.Module):
    """
    """
    def __init__(
        self,
        k_l: int,
        k_n: int,
        in_c: int =4,
        # reverse: bool = False,
    ):
        super().__init__()
        self.k_n = k_n
        self.k_l = k_l
        self.kattn1 = KattentionV4(channel_size=in_c, kernel_size=k_l, num_kernels=k_n,
                                       reverse=False)
        self.kattn2 = KattentionV4(channel_size=in_c, kernel_size=k_l, num_kernels=k_n,
                                       reverse=True)

    def forward(self, x):

        pad = int((self.k_l - 1) / 2)
        x_padded = F.pad(x.transpose(1, 2), (pad, pad), mode="constant", value=0.25)
        # A = self.kattn(x_padded.transpose(1, 2)) 
        x1 = self.kattn1(x_padded.transpose(1, 2))["attn_logits"]
        x2 = self.kattn2(x_padded.transpose(1, 2))["attn_logits"]
        
        x = torch.cat((x1, x2), dim=1)
        out = F.adaptive_max_pool2d(x, output_size=(x.size(2), 1)).squeeze(-1)
        return out

class KNET_Crispr_test(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module(k_n=number_of_kernel, k_l=3)
        self.Kattention2 = kattn_module(k_n=number_of_kernel, k_l=5)
        self.Kattention3 = kattn_module(k_n=number_of_kernel, k_l=7)
    
        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*6*30, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = BaseClassifier_reg(
            in_features=128,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        x3 = self.Kattention3(X)
        x = torch.cat((x1, x2, x3), dim=1)

        feats = x.reshape(x.size(0), -1) #C*6*30
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class CRISPRon(nn.Module):
    """
    PyTorch 实现的 CRISPRon 主干：
    one-hot(30-mer) → Conv1D(k=3/5/7, padding='same') × 多核 → Flatten
      → FC_collect → Concat(ΔG_B) → FC → FC → 标量输出

    Args:
        k_list: 卷积核大小列表（论文使用 3/5/7）
        c_list: 每个卷积核对应的输出通道数（可统一或分别设置）
        fc_collect: 卷积后收集的全连接层输出维度
        fc_hidden: 融合 ΔG_B 后的隐藏层维度
        dropout: dropout 比例
        seq_len: 30-mer 序列长度（默认 30）
    """
    def __init__(
        self,
        k_list=(3, 5, 7),
        c_list=(100, 70, 40),
        fc_collect=80,
        fc_hidden=60,
        dropout=0.2,
        seq_len=15,
        vocab_size=10,

    ):
        super().__init__()
        assert len(k_list) == len(c_list)

        self.seq_len = seq_len
        self.k_list = k_list
        self.c_list = c_list

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
        # 多尺度卷积块：输入通道=4（A/C/G/T one-hot）
        convs = []
        for k, c in zip(k_list, c_list):
            pad = (k - 1) // 2  # 'same' 长度保持
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=4, out_channels=c, kernel_size=k, padding=pad, bias=True),
                    # nn.ReLU(inplace=True),
                    nn.AvgPool1d(kernel_size=2, stride=2)
                    # 可选加入 BN/Dropout
                    # nn.BatchNorm1d(c),
                    # nn.Dropout(dropout),
                )
            )
        self.convs = nn.ModuleList(convs)

        # Flatten 后的维度：sum(c_list) * seq_len
        # flat_dim = int(sum(c_list) * seq_len/2)
        flat_dim = int(sum(c_list) * 15)
        # 卷积收集层（论文里先用一层 FC 收敛卷积表示，再与 ΔG_B 融合）
        self.fc_collect = nn.Sequential(
            nn.Linear(flat_dim, fc_collect),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 与 ΔG_B 融合：ΔG_B 是一个标量（B, 1），拼接到特征后
        # fused_in = fc_collect + 1

        # 两层全连接回归头
        self.regressor = nn.Sequential(
            nn.Linear(fc_collect, fc_hidden),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # 简单的 Kaiming 初始化（可按需微调）
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _indices_to_onehot(x_idx: torch.Tensor, num_classes=4):
        """
        将 (B, L) 的 0/1/2/3 索引转为 (B, 4, L) 的 one-hot。
        """
        # x_idx: long, 值域 [0, 3]
        oh = F.one_hot(x_idx.long(), num_classes=num_classes).float()  # (B, L, 4)
        return oh.permute(0, 2, 1).contiguous()  # (B, 4, L)

    # def forward(self, x, delta_g):
    def forward(
        self,
        x: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x:  (B, 4, L) 的 one-hot，或 (B, L) 的 0/1/2/3 索引
            delta_g: (B,) 或 (B,1) 的 ΔG_B 标量特征

        Returns:
            {'pred': (B,) 的标量预测}
        """
        # pdb.set_trace()
        X = self.embedding(x)[:,:,-4:].transpose(1, 2)

        # 多尺度卷积
        feats = [conv(X) for conv in self.convs]            # [(B, c_i, L), ...]
        feats = torch.cat(feats, dim=1)                     # (B, sum_c, L)
        # Flatten + FC 收集
        feats = feats.reshape(feats.size(0), -1)            # (B, sum_c * L)
        feats = self.fc_collect(feats)                      # (B, fc_collect)

        # # 处理 ΔG_B，拼接
        # if delta_g.dim() == 1:
        #     delta_g = delta_g.unsqueeze(1)                  # (B,1)
        # fused = torch.cat([feats, delta_g], dim=1)          # (B, fc_collect + 1)

        # 回归输出
        pred = self.regressor(feats).squeeze(-1)            # (B,)
        loss = F.mse_loss(pred, cls_labels, reduction='mean')

        return {"pred": pred,
        "loss": loss}

class kattn_module_(nn.Module):
    """
    """
    def __init__(
        self,
        k_l: int,
        k_n: int,
        in_c: int =4,
        # reverse: bool = False,
    ):
        super().__init__()
        self.k_l = k_l
        self.k_n = k_n
        self.kattn1 = KattentionV4(channel_size=in_c, kernel_size=k_l, num_kernels=k_n//2,
                                       reverse=False)
        self.kattn2 = KattentionV4(channel_size=in_c, kernel_size=k_l, num_kernels=k_n//2,
                                       reverse=True)

    def forward(self, x):

        pad = int((self.k_l - 1) / 2)
        x_padded = F.pad(x.transpose(1, 2), (pad, pad), mode="constant", value=0.25)
        # A = self.kattn(x_padded.transpose(1, 2)) 
        x1 = self.kattn1(x_padded.transpose(1, 2))["attn_logits"]
        x2 = self.kattn2(x_padded.transpose(1, 2))["attn_logits"].flip([3])
        
        out = torch.cat((x1, x2), dim=1)
        # out = F.adaptive_max_pool2d(x, output_size=(x.size(2), 1)).squeeze(-1)
        return out

class KNET_Crispr_test1(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        # self.structure = structure
        self.n = number_of_kernel

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module_(k_n=self.n, k_l=3)

        self.conv2 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=3)

        self.fc_collect = nn.Sequential(
            nn.Linear(self.n*16, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.conv2(x1)

        x3 = F.adaptive_avg_pool2d(x2, (4, 4))
        feats = x3.reshape(x3.size(0), -1) #C*6*30
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class CRISPRon_pt(nn.Module):
    """
    PyTorch 实现的 CRISPRon 主干：
    one-hot(30-mer) → Conv1D(k=3/5/7, padding='same') × 多核 → Flatten
      → FC_collect → Concat(ΔG_B) → FC → FC → 标量输出

    Args:
        k_list: 卷积核大小列表（论文使用 3/5/7）
        c_list: 每个卷积核对应的输出通道数（可统一或分别设置）
        fc_collect: 卷积后收集的全连接层输出维度
        fc_hidden: 融合 ΔG_B 后的隐藏层维度
        dropout: dropout 比例
        seq_len: 30-mer 序列长度（默认 30）
    """
    def __init__(
        self,
        k_list=(3, 5, 7),
        c_list=(64, 64, 64),
        fc_collect=256,
        fc_hidden=128,
        dropout=0.2,
        seq_len=30,
        vocab_size=10,

    ):
        super().__init__()
        assert len(k_list) == len(c_list)

        self.seq_len = seq_len
        self.k_list = k_list
        self.c_list = c_list

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
        # 多尺度卷积块：输入通道=4（A/C/G/T one-hot）
        convs = []
        for k, c in zip(k_list, c_list):
            pad = (k - 1) // 2  # 'same' 长度保持
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=4, out_channels=c, kernel_size=k, padding=pad, bias=True),
                    nn.ReLU(inplace=True),
                    # 可选加入 BN/Dropout
                    # nn.BatchNorm1d(c),
                    # nn.Dropout(dropout),
                )
            )
        self.convs = nn.ModuleList(convs)

        # Flatten 后的维度：sum(c_list) * seq_len
        flat_dim = sum(c_list) * seq_len

        # 卷积收集层（论文里先用一层 FC 收敛卷积表示，再与 ΔG_B 融合）
        self.fc_collect = nn.Sequential(
            nn.Linear(flat_dim, fc_collect),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 与 ΔG_B 融合：ΔG_B 是一个标量（B, 1），拼接到特征后
        # fused_in = fc_collect + 1

        # 两层全连接回归头
        self.regressor = nn.Sequential(
            nn.Linear(fc_collect, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # 简单的 Kaiming 初始化（可按需微调）
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _indices_to_onehot(x_idx: torch.Tensor, num_classes=4):
        """
        将 (B, L) 的 0/1/2/3 索引转为 (B, 4, L) 的 one-hot。
        """
        # x_idx: long, 值域 [0, 3]
        oh = F.one_hot(x_idx.long(), num_classes=num_classes).float()  # (B, L, 4)
        return oh.permute(0, 2, 1).contiguous()  # (B, 4, L)

    # def forward(self, x, delta_g):
    def forward(
        self,
        x: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x:  (B, 4, L) 的 one-hot，或 (B, L) 的 0/1/2/3 索引
            delta_g: (B,) 或 (B,1) 的 ΔG_B 标量特征

        Returns:
            {'pred': (B,) 的标量预测}
        """
        # pdb.set_trace()
        X = self.embedding(x)[:,:,-4:].transpose(1, 2)

        # 多尺度卷积
        feats = [conv(X) for conv in self.convs]            # [(B, c_i, L), ...]
        feats = torch.cat(feats, dim=1)                     # (B, sum_c, L)

        # Flatten + FC 收集
        feats = feats.reshape(feats.size(0), -1)            # (B, sum_c * L)
        feats = self.fc_collect(feats)                      # (B, fc_collect)

        # # 处理 ΔG_B，拼接
        # if delta_g.dim() == 1:
        #     delta_g = delta_g.unsqueeze(1)                  # (B,1)
        # fused = torch.cat([feats, delta_g], dim=1)          # (B, fc_collect + 1)

        # 回归输出
        pred = self.regressor(feats).squeeze(-1)            # (B,)
        loss = F.mse_loss(pred, cls_labels, reduction='mean')

        return {"pred": pred,
        "loss": loss}

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

class kattn_with_V(nn.Module):
    """
    """
    def __init__(
        self,
        k_l: int,
        k_n: int = 64,
        h_d: int = 32,
        in_c: int =4,
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type='absolute_sinusoidal'
    ):
        super().__init__()
        self.k_n = k_n
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=30, emb_dim=4, pad_token_id=0,
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )
        self.k_l = k_l
        self.v = nn.Sequential(
            nn.Conv1d(
            in_channels=in_c,
            out_channels=k_n*h_d,
            kernel_size=k_l,
            padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )
        self.kattn = KattentionV4(channel_size=in_c, kernel_size=k_l, num_kernels=k_n,
                                       reverse=reverse)
        self.ds = nn.Sequential(
            nn.Conv1d(
            in_channels=k_n*h_d,
            out_channels=k_n,
            kernel_size=3,
            padding='same'
            ),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        b,seq_len,_ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.shape[:2])
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        v = self.v(x.transpose(1, 2))                               # (B, C, L)  C=H*D
        v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)
        pad = int((self.k_l - 1) / 2)
        x_padded = F.pad(x.transpose(1, 2), (pad, pad), mode="constant", value=0.25)
        A = self.kattn(x_padded.transpose(1, 2))["attn_logits"]
        attn = torch.softmax(A , dim=-1)     # (B,H,L,L) 或 (B,L,L)

        # **(B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)**
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)

        # x = rearrange(out, "b h q d -> b q (h d)")
        x = rearrange(out, "b h q d -> b (h d) q")
        out = self.ds(x)
        return out
    
class KNET_Crispr_test2(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_with_V(k_l=5,k_n= number_of_kernel,h_d= hiddn_dim)

        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*15, 80),
            nn.Dropout(0.3),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.Dropout(0.3),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)

        feats = x1.reshape(x1.size(0), -1) #C*6*30
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class CRISPRon_base(nn.Module):
    """
    PyTorch 实现的 CRISPRon 主干：
    one-hot(30-mer) → Conv1D(k=3/5/7, padding='same') × 多核 → Flatten
      → FC_collect → Concat(ΔG_B) → FC → FC → 标量输出

    Args:
        k_list: 卷积核大小列表（论文使用 3/5/7）
        c_list: 每个卷积核对应的输出通道数（可统一或分别设置）
        fc_collect: 卷积后收集的全连接层输出维度
        fc_hidden: 融合 ΔG_B 后的隐藏层维度
        dropout: dropout 比例
        seq_len: 30-mer 序列长度（默认 30）
    """
    def __init__(
        self,
        k_list=(3, 5, 7),
        c_list=(512, 512, 512),
        fc_collect=80,
        fc_hidden=60,
        dropout=0.2,
        seq_len=15,
        vocab_size=10,

    ):
        super().__init__()
        assert len(k_list) == len(c_list)

        self.seq_len = seq_len
        self.k_list = k_list
        self.c_list = c_list

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
        # 多尺度卷积块：输入通道=4（A/C/G/T one-hot）
        convs = []
        for k, c in zip(k_list, c_list):
            pad = (k - 1) // 2  # 'same' 长度保持
            convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=4, out_channels=c, kernel_size=k, padding=pad, bias=True),
                    # nn.ReLU(inplace=True),
                    nn.Conv1d(
                        in_channels=512,
                        out_channels=16,
                        kernel_size=3,
                        padding='same'
                        ),
                    nn.AvgPool1d(kernel_size=2, stride=2)
                    # 可选加入 BN/Dropout
                    # nn.BatchNorm1d(c),
                    # nn.Dropout(dropout),
                )
            )
        self.convs = nn.ModuleList(convs)

        # Flatten 后的维度：sum(c_list) * seq_len
        # flat_dim = int(sum(c_list) * seq_len/2)
        flat_dim = 48 * 15
        # 卷积收集层（论文里先用一层 FC 收敛卷积表示，再与 ΔG_B 融合）
        self.fc_collect = nn.Sequential(
            nn.Linear(flat_dim, fc_collect),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 与 ΔG_B 融合：ΔG_B 是一个标量（B, 1），拼接到特征后
        # fused_in = fc_collect + 1

        # 两层全连接回归头
        self.regressor = nn.Sequential(
            nn.Linear(fc_collect, fc_hidden),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # 简单的 Kaiming 初始化（可按需微调）
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _indices_to_onehot(x_idx: torch.Tensor, num_classes=4):
        """
        将 (B, L) 的 0/1/2/3 索引转为 (B, 4, L) 的 one-hot。
        """
        # x_idx: long, 值域 [0, 3]
        oh = F.one_hot(x_idx.long(), num_classes=num_classes).float()  # (B, L, 4)
        return oh.permute(0, 2, 1).contiguous()  # (B, 4, L)

    # def forward(self, x, delta_g):
    def forward(
        self,
        x: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x:  (B, 4, L) 的 one-hot，或 (B, L) 的 0/1/2/3 索引
            delta_g: (B,) 或 (B,1) 的 ΔG_B 标量特征

        Returns:
            {'pred': (B,) 的标量预测}
        """
        # pdb.set_trace()
        X = self.embedding(x)[:,:,-4:].transpose(1, 2)

        # 多尺度卷积
        feats = [conv(X) for conv in self.convs]            # [(B, c_i, L), ...]
        feats = torch.cat(feats, dim=1)                     # (B, sum_c, L)
        # Flatten + FC 收集
        feats = feats.reshape(feats.size(0), -1)            # (B, sum_c * L)
        feats = self.fc_collect(feats)                      # (B, fc_collect)

        # 回归输出
        pred = self.regressor(feats).squeeze(-1)            # (B,)
        loss = F.mse_loss(pred, cls_labels, reduction='mean')

        return {"pred": pred,
        "loss": loss}

class KNET_Crispr_test3(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_with_V(k_l=5,k_n= number_of_kernel,h_d= hiddn_dim)

        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*15, 80),
            nn.Dropout(0.3),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.Dropout(0.3),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)

        feats = x1.reshape(x1.size(0), -1) #C*6*30
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)


class kattn_with_V_nopos(nn.Module):
    """
    """
    def __init__(
        self,
        k_l: int,
        k_n: int = 64,
        h_d: int = 32,
        in_c: int =4,
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type='absolute_sinusoidal'
    ):
        super().__init__()
        self.k_n = k_n

        self.k_l = k_l
        self.v = nn.Sequential(
            nn.Conv1d(
            in_channels=in_c,
            out_channels=k_n*h_d,
            kernel_size=k_l,
            padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )
        self.kattn = KattentionV4(channel_size=in_c, kernel_size=k_l, num_kernels=k_n,
                                       reverse=reverse)
        self.ds = nn.Sequential(
            nn.Conv1d(
            in_channels=k_n*h_d,
            out_channels=k_n,
            kernel_size=3,
            padding='same'
            ),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        b,seq_len,_ = x.shape

        v = self.v(x.transpose(1, 2))                               # (B, C, L)  C=H*D
        v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)
        pad = int((self.k_l - 1) / 2)
        x_padded = F.pad(x.transpose(1, 2), (pad, pad), mode="constant", value=0.25)
        A = self.kattn(x_padded.transpose(1, 2))["attn_logits"]
        attn = torch.softmax(A , dim=-1)     # (B,H,L,L) 或 (B,L,L)

        # **(B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)**
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)

        # x = rearrange(out, "b h q d -> b q (h d)")
        x = rearrange(out, "b h q d -> b (h d) q")
        out = self.ds(x)
        return out

class KNET_Crispr_test4(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_with_V_nopos(k_l=5,k_n= number_of_kernel,h_d= hiddn_dim)

        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*15, 80),
            nn.Dropout(0.3),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.Dropout(0.3),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)

        feats = x1.reshape(x1.size(0), -1) #C*6*30
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class self_kattn_ffn(nn.Module):
    """
    K-attention + Add&Norm + FFN（Post-Norm）
    - 输入/残差/LayerNorm 都在 (B, L, C) 格式
    - V 由 1D Conv 生成并拆成 (H, D)
    - 注意力权重由 KattentionV4 给出 (B, H, L, L)
    - 注意力输出经 O-proj 投回 C 维 -> 残差相加 -> LayerNorm（Post-Norm）
    - FFN：Linear -> GELU -> Dropout -> Linear -> 残差相加 -> LayerNorm（Post-Norm）
    """
    def __init__(
        self,
        k_l: int,
        k_n: int = 64,             # 头数 H
        h_d: int = 12,             # 每头维度 D
        in_c: int = 4,             # 输入通道/特征维 C_in
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type: str = 'absolute_sinusoidal',
        ffn_multiplier: int = 4,   # FFN 扩张倍数
        dropout: float = 0.1,      # Dropout 概率
    ):
        super().__init__()
        self.k_n = k_n
        self.k_l = k_l
        self.in_c = in_c
        self.h_d = h_d
        self.hidden_dim = k_n * h_d  # H * D
        self.reverse = reverse

        # 位置编码（保持你的实现）
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=30, emb_dim=self.hidden_dim, pad_token_id=0,  # 如需与 x 维度匹配，可改为 emb_dim=in_c
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )
        else:
            self.position_embeddings = None

        # 生成 V 的投影：Conv1d (B, C, L) -> (B, H*D, L)
        self.v = nn.Sequential(
            nn.Conv1d(
                in_channels=in_c,
                out_channels=self.hidden_dim,
                kernel_size=k_l,
                padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

        # 你的 K-attention（输出 attn_logits: (B, H, L, L) 或 (B, L, L)）
        self.kattn = KattentionV4(
            channel_size=in_c,
            kernel_size=k_l,
            num_kernels=k_n,
            reverse=reverse
        )

        # LayerNorm（Post-Norm：放在残差之后）
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # FFN：C -> C*ffn_multiplier -> C
        ffn_hidden =self.hidden_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, self.hidden_dim),
        )

        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn  = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, C_in)
        return:
            out: (B, L, C_out)
        """
        B, L, C = x.shape
        assert C == self.in_c, f"Expected in_c={self.in_c}, got {C}"


        # ---------------- Self-Attention 子层（Post-Norm） ----------------
        # 生成 V: 输入要求 (B, C, L)
        v = self.v(x.transpose(1, 2))
        
        # 加位置编码（如果有）
        if self.position_embeddings is not None:
            position_ids = torch.arange(L, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, L)              # (B, L)
            pos = self.position_embeddings(position_ids)                        # (B, L, ?)
            v = v + pos.transpose(1, 2)                                                  
        v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)                # (B, H, L, D)

        # K-attn 权重
        pad = int((self.k_l - 1) / 2)
        x_ch_first = x.transpose(1, 2)                                         # (B, C, L)
        x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)  # (B, C, L+2*pad)
        if self.reverse:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"].flip([-1])
        else:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"]         # (B, H, L, L) 或 (B, L, L)

        attn = torch.softmax(A_logits/ math.sqrt(self.h_d), dim=-1)                                  # (B, H, L, L)

        # 注意力加权： (B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)
        attn_out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)                  # (B, H, L, D)
        attn_out = rearrange(attn_out, "b h q d -> b (h d) q")                  # (B, H*D, L)
        attn_out_c = self.drop_attn(attn_out)

        # 残差相加，随后做 LayerNorm（Post-Norm）
        y1 = (v + attn_out).transpose(1, 2)       # (B, L, H*D)
        h = self.ln1(y1)                                                        # Norm（Post-Norm）

        # ---------------- FFN 子层（Post-Norm） ----------------
        ffn_out = self.ffn(h)                                                   # (B, L, out_c)
        ffn_out = self.drop_ffn(ffn_out)
        y2 = h + self.drop_ffn(self.ffn(h))       # (B, L, H*D)
        out = self.ln2(y2)                                                      # Norm（Post-Norm）

        return out.transpose(1, 2) 
    
class KNET_Crispr_test5(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn_ffn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn_ffn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.merge = nn.Sequential(
            nn.Conv1d(hiddn_dim*number_of_kernel, 512, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),  # 最终想要的通道
        )
        # 再做 stride=2 的深度可分离卷积代替 AvgPool
        self.ds = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, groups=128)

        self.fc_collect = nn.Sequential(
            nn.Linear(128*15, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        pooled_x = torch.cat([x1, x2], dim=1)

        feats1 = self.ds(self.merge(pooled_x))
        feats2 = feats1.reshape(feats1.size(0), -1) #C*6*30

        feats3 = self.fc_collect(feats2) 

        feats4 = self.regressor(feats3)
        return self.classifier(feats4,cls_labels)

class self_kattn(nn.Module):
    """
    K-attention + Add&Norm + FFN（Post-Norm）
    - 输入/残差/LayerNorm 都在 (B, L, C) 格式
    - V 由 1D Conv 生成并拆成 (H, D)
    - 注意力权重由 KattentionV4 给出 (B, H, L, L)
    - 注意力输出经 O-proj 投回 C 维 -> 残差相加 -> LayerNorm（Post-Norm）
    - FFN：Linear -> GELU -> Dropout -> Linear -> 残差相加 -> LayerNorm（Post-Norm）
    """
    def __init__(
        self,
        k_l: int,
        k_n: int = 64,             # 头数 H
        h_d: int = 12,             # 每头维度 D
        in_c: int = 4,             # 输入通道/特征维 C_in
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type: str = 'absolute_sinusoidal',
        ffn_multiplier: int = 4,   # FFN 扩张倍数
        dropout: float = 0.1,      # Dropout 概率
    ):
        super().__init__()
        self.k_n = k_n
        self.k_l = k_l
        self.in_c = in_c
        self.h_d = h_d
        self.hidden_dim = k_n * h_d  # H * D
        self.reverse = reverse

        # 位置编码（保持你的实现）
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=30, emb_dim=self.hidden_dim, pad_token_id=0,  # 如需与 x 维度匹配，可改为 emb_dim=in_c
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )
        else:
            self.position_embeddings = None

        # 生成 V 的投影：Conv1d (B, C, L) -> (B, H*D, L)
        self.v = nn.Sequential(
            nn.Conv1d(
                in_channels=in_c,
                out_channels=self.hidden_dim,
                kernel_size=k_l,
                padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

        # 你的 K-attention（输出 attn_logits: (B, H, L, L) 或 (B, L, L)）
        self.kattn = KattentionV4(
            channel_size=in_c,
            kernel_size=k_l,
            num_kernels=k_n,
            reverse=reverse
        )

        # LayerNorm（Post-Norm：放在残差之后）
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.drop_attn = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, C_in)
        return:
            out: (B, L, C_out)
        """
        B, L, C = x.shape
        assert C == self.in_c, f"Expected in_c={self.in_c}, got {C}"


        # ---------------- Self-Attention 子层（Post-Norm） ----------------
        # 生成 V: 输入要求 (B, C, L)
        v = self.v(x.transpose(1, 2))
        
        # 加位置编码（如果有）
        if self.position_embeddings is not None:
            position_ids = torch.arange(L, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, L)              # (B, L)
            pos = self.position_embeddings(position_ids)                        # (B, L, ?)
            v = v + pos.transpose(1, 2)                                                  
        v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)                # (B, H, L, D)

        # K-attn 权重
        pad = int((self.k_l - 1) / 2)
        x_ch_first = x.transpose(1, 2)                                         # (B, C, L)
        x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)  # (B, C, L+2*pad)
        if self.reverse:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"].flip([-1])
        else:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"]         # (B, H, L, L) 或 (B, L, L)

        attn = torch.softmax(A_logits/ math.sqrt(self.h_d), dim=-1)                                  # (B, H, L, L)

        # 注意力加权： (B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)
        attn_out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)                  # (B, H, L, D)
        attn_out = rearrange(attn_out, "b h q d -> b (h d) q")                  # (B, H*D, L)
        attn_out_c = self.drop_attn(attn_out)

        # 残差相加，随后做 LayerNorm（Post-Norm）
        y1 = (v + attn_out_c).transpose(1, 2)       # (B, L, H*D)
        out = self.ln1(y1)                                                        # Norm（Post-Norm）

        return out.transpose(1, 2) 

class KNET_Crispr_test6(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.ds = nn.Sequential(
            nn.Conv1d(
            in_channels=number_of_kernel*hiddn_dim,
            out_channels=number_of_kernel,
            kernel_size=3,
            padding='same'
            ),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*15, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        pooled_x = torch.cat([x1, x2], dim=1)
        feats1 = self.ds(pooled_x)
        feats2 = feats1.reshape(feats1.size(0), -1) #C*6*30

        feats3 = self.fc_collect(feats2) 

        feats4 = self.regressor(feats3)
        return self.classifier(feats4,cls_labels)


class self_kattn1(nn.Module):
    """
    K-attention no res
    - 输入/残差/LayerNorm 都在 (B, L, C) 格式
    - V 由 1D Conv 生成并拆成 (H, D)
    - 注意力权重由 KattentionV4 给出 (B, H, L, L)
    - 注意力输出经 O-proj 投回 C 维 -> 残差相加 -> LayerNorm（Post-Norm）
    - FFN：Linear -> GELU -> Dropout -> Linear -> 残差相加 -> LayerNorm（Post-Norm）
    """
    def __init__(
        self,
        k_l: int,
        k_n: int = 64,             # 头数 H
        h_d: int = 12,             # 每头维度 D
        in_c: int = 4,             # 输入通道/特征维 C_in
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type: str = 'absolute_sinusoidal',
        ffn_multiplier: int = 4,   # FFN 扩张倍数
        dropout: float = 0.1,      # Dropout 概率
    ):
        super().__init__()
        self.k_n = k_n
        self.k_l = k_l
        self.in_c = in_c
        self.h_d = h_d
        self.hidden_dim = k_n * h_d  # H * D
        self.reverse = reverse

        # 位置编码（保持你的实现）
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=30, emb_dim=self.hidden_dim, pad_token_id=0,  # 如需与 x 维度匹配，可改为 emb_dim=in_c
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )
        else:
            self.position_embeddings = None

        # 生成 V 的投影：Conv1d (B, C, L) -> (B, H*D, L)
        self.v = nn.Sequential(
            nn.Conv1d(
                in_channels=in_c,
                out_channels=self.hidden_dim,
                kernel_size=k_l,
                padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

        # 你的 K-attention（输出 attn_logits: (B, H, L, L) 或 (B, L, L)）
        self.kattn = KattentionV4(
            channel_size=in_c,
            kernel_size=k_l,
            num_kernels=k_n,
            reverse=reverse
        )

        # LayerNorm（Post-Norm：放在残差之后）
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.drop_attn = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, C_in)
        return:
            out: (B, L, C_out)
        """
        B, L, C = x.shape
        assert C == self.in_c, f"Expected in_c={self.in_c}, got {C}"


        # ---------------- Self-Attention 子层（Post-Norm） ----------------
        # 生成 V: 输入要求 (B, C, L)
        v = self.v(x.transpose(1, 2))
        
        # 加位置编码（如果有）
        if self.position_embeddings is not None:
            position_ids = torch.arange(L, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, L)              # (B, L)
            pos = self.position_embeddings(position_ids)                        # (B, L, ?)
            v = v + pos.transpose(1, 2)                                                  
        v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)                # (B, H, L, D)

        # K-attn 权重
        pad = int((self.k_l - 1) / 2)
        x_ch_first = x.transpose(1, 2)                                         # (B, C, L)
        x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)  # (B, C, L+2*pad)
        if self.reverse:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"].flip([-1])
        else:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"]         # (B, H, L, L) 或 (B, L, L)

        attn = torch.softmax(A_logits/ math.sqrt(self.h_d), dim=-1)                                  # (B, H, L, L)

        # 注意力加权： (B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)
        attn_out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)                  # (B, H, L, D)
        attn_out = rearrange(attn_out, "b h q d -> b (h d) q")                  # (B, H*D, L)
        # out = self.drop_attn(attn_out)

        # 残差相加，随后做 LayerNorm（Post-Norm）
        # y1 = (v + attn_out).transpose(1, 2)       # (B, L, H*D)
        # out = self.ln1(attn_out.transpose(1, 2))                                                        # Norm（Post-Norm）

        return attn_out

class KNET_Crispr_test7(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 8,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn1(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        # self.Kattention2 = self_kattn1(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)


        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*hiddn_dim*6, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        x = torch.cat([x1, x2], dim=1)
        pooled_x = F.adaptive_avg_pool1d(x, 6)
        feats2 = pooled_x.reshape(pooled_x.size(0), -1) #C*6*30
        feats3 = self.fc_collect(feats2) 

        feats4 = self.regressor(feats3)
        return self.classifier(feats4,cls_labels)


class KNET_Crispr_test8(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn1(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn1(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.ds = nn.Sequential(
            nn.Conv1d(
            in_channels=number_of_kernel*hiddn_dim,
            out_channels=number_of_kernel,
            kernel_size=3,
            padding='same'
            ),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*15, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        pooled_x = torch.cat([x1, x2], dim=1)

        feats1 = self.ds(pooled_x)
        feats2 = feats1.reshape(feats1.size(0), -1) #C*6*30

        feats3 = self.fc_collect(feats2) 

        feats4 = self.regressor(feats3)
        return self.classifier(feats4,cls_labels)

class KNET_Crispr_test9(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 8,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.fc_collect = nn.Sequential(
            nn.Linear(hiddn_dim*number_of_kernel*6, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        pooled_x = torch.cat([x1, x2], dim=1)
        feats1 = F.adaptive_avg_pool1d(pooled_x, 6)
        feats2 = feats1.reshape(feats1.size(0), -1) #C*6*30
        feats3 = self.fc_collect(feats2) 

        feats4 = self.regressor(feats3)
        return self.classifier(feats4,cls_labels)

class KNET_Crispr_test10(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        # self.structure = structure
        self.n = number_of_kernel

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module_(k_n=self.n, k_l=3)

        # self.conv2 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=3)

        self.fc_collect = nn.Sequential(
            nn.Linear(self.n*16, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        # x2 = self.conv2(x1)

        x3 = F.adaptive_avg_pool2d(x1, (4, 4))
        feats = x3.reshape(x3.size(0), -1) #C*6*30
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class KNET_Crispr_test11(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        # self.structure = structure
        self.n = number_of_kernel

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module_(k_n=self.n, k_l=5)

        self.conv2 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=3)

        self.fc_collect = nn.Sequential(
            nn.Linear(self.n*16, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.conv2(x1)

        x3 = F.adaptive_avg_pool2d(x2, (4, 4))
        feats = x3.reshape(x3.size(0), -1) #C*6*30
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class KNET_Crispr_test12(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        # self.structure = structure
        self.n = number_of_kernel

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module_(k_n=self.n, k_l=5)

        self.fc_collect = nn.Sequential(
            nn.Linear(self.n*30, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        pool = F.adaptive_max_pool2d(x1, output_size=(x1.size(2), 1)).squeeze(-1)
        feats = pool.reshape(pool.size(0), -1)
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class KNET_Crispr_test13(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        # self.structure = structure
        self.n = number_of_kernel

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module_(k_n=self.n, k_l=5)

        self.conv = nn.Sequential(
            nn.Conv1d(
            in_channels=self.n,
            out_channels=self.n,
            kernel_size=3,
            padding='same'
            ),
            nn.GELU(),
            # nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.fc_collect = nn.Sequential(
            nn.Linear(self.n*30, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        pool = F.adaptive_max_pool2d(x1, output_size=(x1.size(2), 1)).squeeze(-1)
        agg = self.conv(pool)
        feats = agg.reshape(agg.size(0), -1)
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class KNET_Crispr_test14(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        # self.structure = structure
        self.n = number_of_kernel

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module_(k_n=self.n, k_l=5)

        self.conv = nn.Conv2d(in_channels=self.n, out_channels=self.n, kernel_size=(1, 6), stride=(1, 1), padding=(0, 0))

        self.pool = nn.Sequential(
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.fc_collect = nn.Sequential(
            nn.Linear(self.n*15, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        pool = F.adaptive_avg_pool2d(x1, (x1.size(2), 6))
        agg = self.conv(pool).squeeze(-1)
        agg = self.pool(agg)
        feats = agg.reshape(agg.size(0), -1)
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class KNET_Crispr_test15(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        # self.structure = structure
        self.n = number_of_kernel

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = kattn_module_(k_n=self.n, k_l=5)

        self.conv = nn.Sequential(
            nn.Conv1d(
            in_channels=self.n,
            out_channels=self.n,
            kernel_size=3,
            padding='same'
            ),
            nn.GELU(),
            # nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.fc_collect = nn.Sequential(
            nn.Linear(self.n*30, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        # key_padding_mask: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        pool = F.adaptive_max_pool2d(x1, output_size=(x1.size(2), 1)).squeeze(-1)
        agg = self.conv(pool)
        feats = agg.reshape(agg.size(0), -1)
        feats = self.fc_collect(feats) 

        x = self.regressor(feats)
        return self.classifier(x,cls_labels)

class self_kattn2(nn.Module):
    """
    K-attention + Add&Norm + FFN（Post-Norm）
    - 输入/残差/LayerNorm 都在 (B, L, C) 格式
    - V 由 1D Conv 生成并拆成 (H, D)
    - 注意力权重由 KattentionV4 给出 (B, H, L, L)
    - 注意力输出经 O-proj 投回 C 维 -> 残差相加 -> LayerNorm（Post-Norm）
    - FFN：Linear -> GELU -> Dropout -> Linear -> 残差相加 -> LayerNorm（Post-Norm）
    """
    def __init__(
        self,
        k_l: int,
        k_n: int = 64,             # 头数 H
        h_d: int = 12,             # 每头维度 D
        in_c: int = 4,             # 输入通道/特征维 C_in
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type: str = 'absolute_sinusoidal',
        ffn_multiplier: int = 4,   # FFN 扩张倍数
        dropout: float = 0.1,      # Dropout 概率
    ):
        super().__init__()
        self.k_n = k_n
        self.k_l = k_l
        self.in_c = in_c
        self.h_d = h_d
        self.hidden_dim = k_n * h_d  # H * D
        self.reverse = reverse

        # 位置编码（保持你的实现）
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=30, emb_dim=self.hidden_dim, pad_token_id=0,  # 如需与 x 维度匹配，可改为 emb_dim=in_c
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )
        else:
            self.position_embeddings = None

        # 生成 V 的投影：Conv1d (B, C, L) -> (B, H*D, L)
        self.v = nn.Sequential(
            nn.Conv1d(
                in_channels=in_c,
                out_channels=self.hidden_dim,
                kernel_size=k_l,
                padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

        # 你的 K-attention（输出 attn_logits: (B, H, L, L) 或 (B, L, L)）
        self.kattn = KattentionV4(
            channel_size=in_c,
            kernel_size=k_l,
            num_kernels=k_n,
            reverse=reverse
        )

        # LayerNorm（Post-Norm：放在残差之后）
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.k_n)

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.k_n),
        )
        self.drop_attn = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, C_in)
        return:
            out: (B, L, C_out)
        """
        B, L, C = x.shape
        assert C == self.in_c, f"Expected in_c={self.in_c}, got {C}"


        # ---------------- Self-Attention 子层（Post-Norm） ----------------
        # 生成 V: 输入要求 (B, C, L)
        v = self.v(x.transpose(1, 2))
        
        # 加位置编码（如果有）
        if self.position_embeddings is not None:
            position_ids = torch.arange(L, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, L)              # (B, L)
            pos = self.position_embeddings(position_ids)                        # (B, L, ?)
            v = v + pos.transpose(1, 2)                                                  
        v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)                # (B, H, L, D)

        # K-attn 权重
        pad = int((self.k_l - 1) / 2)
        x_ch_first = x.transpose(1, 2)                                         # (B, C, L)
        x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)  # (B, C, L+2*pad)
        if self.reverse:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"].flip([-1])
        else:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"]         # (B, H, L, L) 或 (B, L, L)

        attn = torch.softmax(A_logits/ math.sqrt(self.h_d), dim=-1)                                  # (B, H, L, L)

        # 注意力加权： (B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)
        attn_out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)                  # (B, H, L, D)
        attn_out = rearrange(attn_out, "b h q d -> b (h d) q")                  # (B, H*D, L)
        attn_out_d = self.drop_attn(attn_out)

        y1 = (v + attn_out_d).transpose(1, 2)       # (B, L, H*D)
        y1 = self.ln1(y1)                           # Norm（Post-Norm）

        y2 = self.ffn(y1)
        out = self.ln2(y2)
        return out.transpose(1, 2) 

class KNET_Crispr_test16(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn2(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn2(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.classifier = BaseClassifier_reg(
            in_features=number_of_kernel*30,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        pooled_x = torch.cat([x1, x2], dim=1)
        out = pooled_x.reshape(pooled_x.size(0), -1) #C*6*30

        return self.classifier(out,cls_labels)

# class self_kattn3(nn.Module):
#     """
#     K-attention + Add&Norm + FFN（Post-Norm）
#     - 输入/残差/LayerNorm 都在 (B, L, C) 格式
#     - V 由 1D Conv 生成并拆成 (H, D)
#     - 注意力权重由 KattentionV4 给出 (B, H, L, L)
#     - 注意力输出经 O-proj 投回 C 维 -> 残差相加 -> LayerNorm（Post-Norm）
#     - FFN：Linear -> GELU -> Dropout -> Linear -> 残差相加 -> LayerNorm（Post-Norm）
#     """
#     def __init__(
#         self,
#         k_l: int,
#         k_n: int = 64,             # 头数 H
#         h_d: int = 12,             # 每头维度 D
#         in_c: int = 4,             # 输入通道/特征维 C_in
#         reverse: bool = False,
#         activation: str = 'relu',
#         position_embedding_type: str = 'absolute_sinusoidal',
#         ffn_multiplier: int = 4,   # FFN 扩张倍数
#         dropout: float = 0.1,      # Dropout 概率
#     ):
#         super().__init__()
#         self.k_n = k_n
#         self.k_l = k_l
#         self.in_c = in_c
#         self.h_d = h_d
#         self.hidden_dim = k_n * h_d  # H * D
#         self.reverse = reverse

#         # 位置编码（保持你的实现）
#         if position_embedding_type is not None and "absolute" in position_embedding_type:
#             self.position_embeddings = AbsoluteLearnedEmbedding(
#                 max_seqlen=30, emb_dim=self.hidden_dim, pad_token_id=0,  # 如需与 x 维度匹配，可改为 emb_dim=in_c
#                 sinusoidal="sinusoidal" in position_embedding_type,
#                 learnable="learned" in position_embedding_type,
#                 base=10_000,
#             )
#         else:
#             self.position_embeddings = None

#         # 生成 V 的投影：Conv1d (B, C, L) -> (B, H*D, L)
#         self.v = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_c,
#                 out_channels=self.hidden_dim,
#                 kernel_size=k_l,
#                 padding='same'
#             ),
#             nn.ReLU() if activation == 'relu' else nn.Identity()
#         )

#         # 你的 K-attention（输出 attn_logits: (B, H, L, L) 或 (B, L, L)）
#         self.kattn = KattentionV4(
#             channel_size=in_c,
#             kernel_size=k_l,
#             num_kernels=k_n,
#             reverse=reverse
#         )

#         # LayerNorm（Post-Norm：放在残差之后）
#         self.ln1 = nn.LayerNorm(self.hidden_dim)
#         self.ln2 = nn.LayerNorm(self.k_n)

#         self.ffn = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.GELU(),
#             # nn.Dropout(dropout),
#             nn.Linear(self.hidden_dim, self.k_n),
#         )

#     def forward(self, x):
#         """
#         x: (B, L, C_in)
#         return:
#             out: (B, L, C_out)
#         """
#         B, L, C = x.shape
#         assert C == self.in_c, f"Expected in_c={self.in_c}, got {C}"


#         # ---------------- Self-Attention 子层（Post-Norm） ----------------
#         # 生成 V: 输入要求 (B, C, L)
#         v = self.v(x.transpose(1, 2))
        
#         # 加位置编码（如果有）
#         if self.position_embeddings is not None:
#             position_ids = torch.arange(L, dtype=torch.long, device=x.device)
#             position_ids = position_ids.unsqueeze(0).expand(B, L)              # (B, L)
#             pos = self.position_embeddings(position_ids)                        # (B, L, ?)
#             v = v + pos.transpose(1, 2)                                                  
#         v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)                # (B, H, L, D)

#         # K-attn 权重
#         pad = int((self.k_l - 1) / 2)
#         x_ch_first = x.transpose(1, 2)                                         # (B, C, L)
#         x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)  # (B, C, L+2*pad)
#         if self.reverse:
#             A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"].flip([-1])
#         else:
#             A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"]         # (B, H, L, L) 或 (B, L, L)

#         attn = torch.softmax(A_logits/ math.sqrt(self.h_d), dim=-1)                                  # (B, H, L, L)

#         # 注意力加权： (B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)
#         attn_out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)                  # (B, H, L, D)
#         attn_out = rearrange(attn_out, "b h q d -> b (h d) q")                  # (B, H*D, L)

#         y1 = (v + attn_out).transpose(1, 2)       # (B, L, H*D)
#         y1 = self.ln1(y1)                           # Norm（Post-Norm）

#         y2 = self.ffn(y1)
#         out = self.ln2(y2)
#         return out.transpose(1, 2) 

class KNET_Crispr_test17(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn3(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn3(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.classifier = BaseClassifier_reg(
            in_features=number_of_kernel*30,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        pooled_x = torch.cat([x1, x2], dim=1)
        out = pooled_x.reshape(pooled_x.size(0), -1) #C*6*30

        return self.classifier(out,cls_labels)

def max_of_main_diag_parallels(x: torch.Tensor) -> torch.Tensor:
    """
    对输入 (b, h, l, l) 的张量，每个 l×l 矩阵中：
    按副对角线方向（从右上到左下）依次遍历所有与主对角线平行的 2l-1 条线，
    分别在各自线上取最大值，返回形状 (b, h, 2l-1)。

    返回的最后一维顺序对应 offset = -(l-1), ..., 0, ..., +(l-1)，
    其中 offset 是主对角线平行线的对角偏移（i-j = offset）。
    """
    assert x.dim() == 4, "x 应为 (b, h, l, l)"
    b, h, l, l2 = x.shape
    assert l == l2, "最后两维需相等"

    outs = []
    # 主对角线平行线的偏移 k = i - j ∈ [-(l-1), ..., +(l-1)]
    for k in range(-(l-1), l):
        diag = x.diagonal(offset=k, dim1=-2, dim2=-1)  # (b, h, diag_len)
        m = diag.max(dim=-1).values                    # (b, h)
        outs.append(m)
    return torch.stack(outs, dim=-1)                   # (b, h, 2l-1)

class KattentionModel_diagmax(nn.Module):
    def __init__(
        self,
        embedding_method: Literal["onehot", "learned"] = "onehot",
        kattn_version: str = "v4",
        vocab_size: int = 10,
        kernel_size: int = 10,
        num_kernels: int = 32,
        cnn_config: Optional[CNNMixerConfig] = None,
        cls_mid_features: int | list[int] = 128,
        position_embedding_type='absolute_sinusoidal'
    ):
        super().__init__()
        if embedding_method == "learned":
            self.embedding = nn.Embedding(vocab_size, 128)
            input_embedding_dim = 128
        elif embedding_method == "onehot":
            self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()
            input_embedding_dim = vocab_size
        else:
            raise ValueError(f"embedding_method {embedding_method} not supported")

        if kattn_version == "v3":
            self.kattn = KattentionV3(
                hidden_dim=input_embedding_dim,
                first_kernel_size=kernel_size,
                num_kernel=num_kernels
            )
        elif kattn_version.startswith("v4"):
            self.kattn = KattentionV4(
                channel_size=4,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        elif kattn_version.startswith("v5"):
            self.kattn = KattentionV5(
                in_channels=input_embedding_dim,
                mixer_channels_per_kernel=16,
                kernel_size=kernel_size,
                num_kernels=num_kernels,
                reverse="rev" in kattn_version,
            )
        else:
            raise ValueError(f"kattn_version {kattn_version} not supported")

        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=100, emb_dim=4, pad_token_id=0,
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )

        if cnn_config is not None:
            cnn_config.in_channels = num_kernels
            self.cnnmixer = CNNMixer(cnn_config)
            cls_in_dim = self.cnnmixer.out_channels
        else:
            self.cnnmixer = None
            cls_in_dim = num_kernels

        self.linear = nn.Sequential(
            nn.Linear(177, 1),
            nn.GELU()
            # nn.Dropout(dropout),
        )

        self.classifier = BaseClassifier(
            in_features=num_kernels,
            mid_features=cls_mid_features,
        )
        # self.classifier = nn.Sequential(
        #     torch.nn.Linear(num_kernels, 1)
        # )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        r"""
        Parameters
        ----------------
        input_ids: torch.Tensor
            Shape: (batch_size, seq_len)
        labels: torch.Tensor
            Shape: (batch_size)
        key_padding_mask: torch.Tensor, optional
            Shape: (batch_size, seq_len), padding positions filled with 0
        """
        #TODO, position encoding

        X = self.embedding(input_ids)[:,:,-4:]
        b,seq_len,_ = X.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=X.device)
        position_ids = position_ids.unsqueeze(0).expand(X.shape[:2])
        position_embeddings = self.position_embeddings(position_ids)
        X = X + position_embeddings
        attn_logits = self.kattn(X, key_padding_mask)["attn_logits"]
        diag_max = max_of_main_diag_parallels(attn_logits)

        l = self.linear(diag_max)
        out = l.reshape(l.size(0), -1) #C*6*30
        return self.classifier(out, cls_labels)
        # return self.classifier(pooled_attn)

class KNET_Crispr(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                #  kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn(k_l=5,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.ds = nn.Sequential(
            nn.Conv1d(
            in_channels=number_of_kernel*hiddn_dim,
            out_channels=number_of_kernel,
            kernel_size=3,
            padding='same',
            groups=number_of_kernel
            ),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        self.fc_collect = nn.Sequential(
            nn.Linear(number_of_kernel*15, 80),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Linear(80, 60),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.classifier = BaseClassifier_reg(
            in_features=60,
            mid_features=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        crisproff: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        pooled_x = torch.cat([x1, x2], dim=1)
        feats1 = self.ds(pooled_x)
        feats2 = feats1.reshape(feats1.size(0), -1) #C*6*30

        feats3 = self.fc_collect(feats2) 

        feats4 = self.regressor(feats3)
        return self.classifier(feats4,cls_labels)

class self_kattn3(nn.Module):
    """
    K-attention + Add&Norm + FFN（Post-Norm）
    - 输入/残差/LayerNorm 都在 (B, L, C) 格式
    - V 由 1D Conv 生成并拆成 (H, D)
    - 注意力权重由 KattentionV4 给出 (B, H, L, L)
    - 注意力输出经 O-proj 投回 C 维 -> 残差相加 -> LayerNorm（Post-Norm）
    - FFN：Linear -> GELU -> Dropout -> Linear -> 残差相加 -> LayerNorm（Post-Norm）
    """
    def __init__(
        self,
        k_l: int,
        in_l: int = 100,
        k_n: int = 64,             # 头数 H
        h_d: int = 12,             # 每头维度 D
        in_c: int = 4,             # 输入通道/特征维 C_in
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type: str = 'absolute_sinusoidal',
        ffn_multiplier: int = 4,   # FFN 扩张倍数
        dropout: float = 0.1,      # Dropout 概率
        V: bool = True
    ):
        super().__init__()
        self.k_n = k_n
        self.k_l = k_l
        self.in_c = in_c
        self.h_d = h_d
        self.hidden_dim = k_n * h_d  # H * D
        self.reverse = reverse
        self.V = V
        # 位置编码（保持你的实现）
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=in_l, emb_dim=self.hidden_dim, pad_token_id=0,  # 如需与 x 维度匹配，可改为 emb_dim=in_c
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )
        else:
            self.position_embeddings = None

        # 生成 V 的投影：Conv1d (B, C, L) -> (B, H*D, L)
        self.v = nn.Sequential(
            nn.Conv1d(
                in_channels=in_c,
                out_channels=self.hidden_dim,
                kernel_size=k_l,
                padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

        # 你的 K-attention（输出 attn_logits: (B, H, L, L) 或 (B, L, L)）
        self.kattn = KattentionV4(
            channel_size=in_c,
            kernel_size=k_l,
            num_kernels=k_n,
            reverse=reverse
        )

        # LayerNorm（Post-Norm：放在残差之后）
        # self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.drop_attn = nn.Dropout(dropout)
        self.conv = Conv_layer(filters1=k_n, filters2=k_n,kernel_size=3)
        self.ds = nn.Sequential(
            nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.k_n,
            kernel_size=3,
            padding='same',
            groups=self.k_n
            ),
            nn.GELU(),
        )

        self.ln_ds = nn.LayerNorm(k_n)       # 作用在 channel 维


    def forward(self, x):
        """
        x: (B, L, C_in)
        return:
            out: (B, L, C_out)
        """
        B, L, C = x.shape
        assert C == self.in_c, f"Expected in_c={self.in_c}, got {C}"

        # ---------------- Self-Attention 子层（Post-Norm） ----------------
        # K-attn 权重
        pad = int((self.k_l - 1) / 2)
        x_ch_first = x.transpose(1, 2)                                         # (B, C, L)
        x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)  # (B, C, L+2*pad)
        if self.reverse:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"].flip([-1])
        else:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"]         # (B, H, L, L) 或 (B, L, L)
        if self.V:
            # 生成 V: 输入要求 (B, C, L)
            v = self.v(x.transpose(1, 2))
            
            # 加位置编码（如果有）
            if self.position_embeddings is not None:
                position_ids = torch.arange(L, dtype=torch.long, device=x.device)
                position_ids = position_ids.unsqueeze(0).expand(B, L)              # (B, L)
                pos = self.position_embeddings(position_ids)                        # (B, L, ?)
                v = v + pos.transpose(1, 2)                                                  
            v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)                # (B, H, L, D)
            # attn = torch.softmax(A_logits/ math.sqrt(self.h_d), dim=-1)  
            # attn = self.conv(A_logits)                             # (B, H, L, L)
            attn = A_logits                          # (B, H, L, L)

            # 注意力加权： (B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)
            attn_out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)                  # (B, H, L, D)
            attn_out = rearrange(attn_out, "b h q d -> b (h d) q")                  # (B, H*D, L)
            attn_out_c = self.drop_attn(attn_out)

            #（Post-Norm）
            y1 = (v + attn_out_c).transpose(1, 2)       # (B, L, H*D)
            out = self.ds(y1.transpose(1, 2))

            # out = self.ln1(y1)                                                        # Norm（Post-Norm）
            # feat_ds = self.ds(y1.transpose(1, 2)).transpose(1, 2)
            # out = self.ln_ds(feat_ds).transpose(1, 2)
        else:
            out = A_logits.max(dim=-1).values

        return out

class KNET_classifier(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                 kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 in_l:int =101,
                 V: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn3(k_l=kernel_size,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False,V=V,in_l=in_l)
        self.Kattention2 = self_kattn3(k_l=kernel_size,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True,V=V,in_l=in_l)

        # self.mlp = nn.Sequential(
        #     nn.Linear(number_of_kernel*10, number_of_kernel),
        #     nn.GELU()
        # )


        self.classifier = BaseClassifier(
            in_features=number_of_kernel,
            mid_features=outputdim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        icshape: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        X = torch.cat([x1, x2], dim=1)
        feats1 = F.adaptive_max_pool1d(X,1)

        # # x1 = self.Kattention1(X)
        # x2 = self.Kattention2(X)
        # # X = torch.cat([x1, x2], dim=1)
        # feats1 = F.adaptive_max_pool1d(x2,1)
        feats2 = feats1.reshape(feats1.size(0), -1) #C*6*30

        # feats3 = self.mlp(feats2) 

        return self.classifier(feats2,cls_labels)

class GatedFusionHead(nn.Module):
    def __init__(self, k_n, hidden_dim=64, per_token=False):
        """
        k_n: 通道数（即你的头数 H）
        hidden_dim: gate MLP 隐层维度
        per_token: True 时 gate 对每个位置不同；False 时对整条序列共享一个 gate
        """
        super().__init__()
        self.per_token = per_token

        self.ln_ds = nn.LayerNorm(k_n)
        self.ln_struct = nn.LayerNorm(k_n)

        # 产生 gate 的小 MLP（输入维度 2*k_n -> gate_dim）
        gate_in_dim = 2 * k_n
        gate_out_dim = k_n  # 输出每个通道一个 gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gate_out_dim),
        )

    def forward(self, feat_ds, feat_struct):
        """
        feat_ds, feat_struct: (B, H, L)
        return: fused_feat: (B, H, L)
        """
        B, H, L = feat_ds.shape

        # 转为 (B, L, H) 做 LayerNorm
        feat_ds_ln = self.ln_ds(feat_ds.transpose(1, 2))        # (B, L, H)
        feat_st_ln = self.ln_struct(feat_struct.transpose(1, 2))# (B, L, H)

        # 拼接后得到 (B, L, 2H)
        feat_cat = torch.cat([feat_ds_ln, feat_st_ln], dim=-1)

        if self.per_token:
            # 每个位置都有自己的 gate: (B, L, H)
            gate_logits = self.gate_mlp(feat_cat)
        else:
            # 对整个序列做一个 pooled gate: (B, 1, 2H) -> (B, 1, H)
            pooled = feat_cat.mean(dim=1, keepdim=True)  # (B, 1, 2H)
            gate_logits = self.gate_mlp(pooled)          # (B, 1, H)
            gate_logits = gate_logits.expand(B, L, H)    # broadcast 到所有位置

        gate = torch.sigmoid(gate_logits)                # (B, L, H) ∈ (0,1)

        # 再转回 (B, H, L)
        gate = gate.transpose(1, 2)                      # (B, H, L)
        feat_ds = feat_ds                                # (B, H, L)
        feat_struct = feat_struct                        # (B, H, L)

        fused = gate * feat_ds + (1.0 - gate) * feat_struct

        return fused, gate

class self_kattn4(nn.Module):
    """
    K-attention + Add&Norm + FFN（Post-Norm）
    - 输入/残差/LayerNorm 都在 (B, L, C) 格式
    - V 由 1D Conv 生成并拆成 (H, D)
    - 注意力权重由 KattentionV4 给出 (B, H, L, L)
    - 注意力输出经 O-proj 投回 C 维 -> 残差相加 -> LayerNorm（Post-Norm）
    - FFN：Linear -> GELU -> Dropout -> Linear -> 残差相加 -> LayerNorm（Post-Norm）
    """
    def __init__(
        self,
        k_l: int,
        in_l: int = 100,
        k_n: int = 64,             # 头数 H
        h_d: int = 12,             # 每头维度 D
        in_c: int = 4,             # 输入通道/特征维 C_in
        reverse: bool = False,
        activation: str = 'relu',
        position_embedding_type: str = 'absolute_sinusoidal',
        ffn_multiplier: int = 4,   # FFN 扩张倍数
        dropout: float = 0.1,      # Dropout 概率
    ):
        super().__init__()
        self.k_n = k_n
        self.k_l = k_l
        self.in_c = in_c
        self.h_d = h_d
        self.hidden_dim = k_n * h_d  # H * D
        self.reverse = reverse

        # 位置编码（保持你的实现）
        if position_embedding_type is not None and "absolute" in position_embedding_type:
            self.position_embeddings = AbsoluteLearnedEmbedding(
                max_seqlen=in_l, emb_dim=self.hidden_dim, pad_token_id=0,  # 如需与 x 维度匹配，可改为 emb_dim=in_c
                sinusoidal="sinusoidal" in position_embedding_type,
                learnable="learned" in position_embedding_type,
                base=10_000,
            )
        else:
            self.position_embeddings = None

        # 生成 V 的投影：Conv1d (B, C, L) -> (B, H*D, L)
        self.v = nn.Sequential(
            nn.Conv1d(
                in_channels=in_c,
                out_channels=self.hidden_dim,
                kernel_size=k_l,
                padding='same'
            ),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

        # 你的 K-attention（输出 attn_logits: (B, H, L, L) 或 (B, L, L)）
        self.kattn = KattentionV4(
            channel_size=in_c,
            kernel_size=k_l,
            num_kernels=k_n,
            reverse=reverse
        )

        # LayerNorm（Post-Norm：放在残差之后）
        # self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.drop_attn = nn.Dropout(dropout)

        self.ds = nn.Sequential(
            nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.k_n,
            kernel_size=3,
            padding='same',
            groups=self.k_n
            ),
            nn.GELU(),
        )

        self.ln_ds   = nn.LayerNorm(k_n)       # 作用在 channel 维
        self.ln_amax = nn.LayerNorm(k_n)

        self.fusion = GatedFusionHead(k_n=k_n, hidden_dim=64, per_token=False)
        self.ln_out = nn.LayerNorm(k_n)

    def forward(self, x):
        """
        x: (B, L, C_in)
        return:
            out: (B, L, C_out)
        """
        B, L, C = x.shape
        assert C == self.in_c, f"Expected in_c={self.in_c}, got {C}"


        # ---------------- Self-Attention 子层（Post-Norm） ----------------
        # 生成 V: 输入要求 (B, C, L)
        v = self.v(x.transpose(1, 2))
        
        # 加位置编码（如果有）
        if self.position_embeddings is not None:
            position_ids = torch.arange(L, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, L)              # (B, L)
            pos = self.position_embeddings(position_ids)                        # (B, L, ?)
            v = v + pos.transpose(1, 2)                                                  
        v_hd = rearrange(v, "b (h d) l -> b h l d", h=self.k_n)                # (B, H, L, D)

        # K-attn 权重
        pad = int((self.k_l - 1) / 2)
        x_ch_first = x.transpose(1, 2)                                         # (B, C, L)
        x_padded = F.pad(x_ch_first, (pad, pad), mode="constant", value=0.25)  # (B, C, L+2*pad)
        if self.reverse:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"].flip([-1])
        else:
            A_logits = self.kattn(x_padded.transpose(1, 2))["attn_logits"]         # (B, H, L, L) 或 (B, L, L)

        attn = torch.softmax(A_logits/ math.sqrt(self.h_d), dim=-1)                                  # (B, H, L, L)
        # 注意力加权： (B,H,L,L) @ (B,H,L,D) -> (B,H,L,D)
        attn_out = torch.einsum("bhqk,bhkd->bhqd", attn, v_hd)                  # (B, H, L, D)
        attn_out = rearrange(attn_out, "b h q d -> b (h d) q")                  # (B, H*D, L)
        attn_out_c = self.drop_attn(attn_out)

        # 残差相加，随后做 LayerNorm（Post-Norm）
        y1 = (v + attn_out_c).transpose(1, 2)       # (B, L, H*D)
        # out = self.ln1(y1)                                                        # Norm（Post-Norm）
        feat_ds = self.ds(y1.transpose(1, 2))
        A_max = A_logits.max(dim=-1).values
        # A_soft = torch.logsumexp(A_logits, dim=-1).detach()
        # out = out + A_max
        # out = torch.cat([out, A_max], dim=1)
        feat_ds_ln = self.ln_ds(feat_ds.transpose(1, 2)).transpose(1, 2)

        A_max_ln   = self.ln_amax(A_max.transpose(1, 2)).transpose(1, 2)

        fused, gate = self.fusion(feat_ds_ln, A_max_ln)      # (B, H, L)
        # fusion = self.fuse(fusion)                         # (B, H, L)
        fusion = self.ln_out(fused.transpose(1, 2)).transpose(1, 2)
        return fusion

class KNET_classifier1(nn.Module):
    def __init__(self,
                 vocab_size: int = 10,
                 kernel_size: int = 12,
                 number_of_kernel: int = 128,
                 hiddn_dim: int = 32,
                 outputdim: int = 1,
                 structure: Optional[bool] = None
                 ):
        super().__init__()
        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.Kattention1 = self_kattn4(k_l=kernel_size,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=False)
        self.Kattention2 = self_kattn4(k_l=kernel_size,k_n= number_of_kernel//2,h_d= hiddn_dim,reverse=True)

        self.mlp = nn.Sequential(
            nn.Linear(number_of_kernel, number_of_kernel//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(number_of_kernel//4, number_of_kernel//16),
            nn.GELU(),
            nn.Dropout(0.1),
        )


        self.classifier = BaseClassifier(
            in_features=number_of_kernel//16,
            mid_features=outputdim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cls_labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        X = self.embedding(input_ids)[:,:,-4:]

        x1 = self.Kattention1(X)
        x2 = self.Kattention2(X)
        X = torch.cat([x1, x2], dim=1)
        feats1 = F.adaptive_max_pool1d(X,1)
        feats2 = feats1.reshape(feats1.size(0), -1) #C*6*30

        feats3 = self.mlp(feats2) 

        return self.classifier(feats3,cls_labels)