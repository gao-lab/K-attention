import pdb
import sys
import math
import torch
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import random
import pickle
import glob

from einops.layers.torch import Rearrange
from einops import rearrange
from typing import Optional, Tuple, Literal

import matplotlib.pyplot as plt

plt.switch_backend('agg')


# Our module!
# sys.path.append("../../corecode/")
# from KattentionCore import *
# from MarkonvCore import *

def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)


class CNNTransformerRBP(nn.Module):
    """CNN-Transformer hybrid for RBP binding prediction (binary classification).

    Structure: input (B, L, 4) -> 3-layer CNN -> 2-layer Transformer -> global max pool
    -> linear -> sigmoid.
    """
    def __init__(self, in_channels=4, outputdim=1,
                 cnn_channels=(32, 64, 128), tf_hidden=128, tf_layers=2, tf_heads=4):
        super().__init__()
        c1, c2, c3 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),         nn.ReLU(), nn.BatchNorm1d(c2),
            nn.Conv1d(c2, c3, kernel_size=7, padding=3),         nn.ReLU(), nn.BatchNorm1d(c3),
        )
        self.proj = nn.Linear(c3, tf_hidden) if c3 != tf_hidden else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=tf_heads, dim_feedforward=tf_hidden * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.linear = nn.Linear(tf_hidden, outputdim)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous().float()    # (B, C, L)
        x = self.cnn(x)                                 # (B, 128, L)
        x = x.permute(0, 2, 1)                         # (B, L, 128)
        x = self.proj(x)
        x = self.transformer(x)                         # (B, L, tf_hidden)
        x = x.max(dim=1).values                        # (B, tf_hidden)
        output = self.linear(x)
        output = torch.sigmoid(output)
        return output.squeeze()


class CNNTransformerRBPMatched(nn.Module):
    """Parameter-matched CNN-Transformer hybrid for RBP (~80k params).

    Structure: input (B, L, 4) -> 2-layer CNN(4->32->64) -> 2-layer Transformer(64h, 4head)
    -> global max pool -> linear -> sigmoid.
    """
    def __init__(self, in_channels=4, outputdim=1,
                 cnn_channels=(32, 64), tf_hidden=64, tf_layers=2, tf_heads=4):
        super().__init__()
        c1, c2 = cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),         nn.ReLU(), nn.BatchNorm1d(c2),
        )
        self.proj = nn.Linear(c2, tf_hidden) if c2 != tf_hidden else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=tf_heads, dim_feedforward=tf_hidden * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.linear = nn.Linear(tf_hidden, outputdim)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous().float()    # (B, C, L)
        x = self.cnn(x)                                 # (B, 64, L)
        x = x.permute(0, 2, 1)                         # (B, L, 64)
        x = self.proj(x)
        x = self.transformer(x)                         # (B, L, tf_hidden)
        x = x.max(dim=1).values                        # (B, tf_hidden)
        output = self.linear(x)
        output = torch.sigmoid(output)
        return output.squeeze()


def buildModel(kernel_size, number_of_kernel, outputdim, modeltype, seqlen=101, device="CPU"):
    """

    Args:
        kernel_size:
        number_of_kernel:
        outputdim:
        modeltype:

    Returns:

    """
    if modeltype == "KNET_plus_seq2":
        net = KNET_plus_seq2(5, kernel_size, number_of_kernel, outputdim)
    elif modeltype == "KNET_plus_ic":
        net = KNET_plus_ic(5, kernel_size, number_of_kernel, outputdim)
    elif modeltype == "cnn_transformer":
        net = CNNTransformerRBP(in_channels=4, outputdim=outputdim)
    elif modeltype == "cnn_transformer_pm":
        net = CNNTransformerRBPMatched(in_channels=4, outputdim=outputdim)

    return net


class GlobalExpectationPooling1D(nn.Module):
    """Expect pooling operation for temporal data.
        # Arguments
            mode: int
            kernel_size: A integer,
                size of the max pooling window
            m_trainable: A boolean variable,
                if m_trainable == True, the base will be trainable,
                else the base will be a constant
            m_value: A float number,
                the value of the base to calculate the prob
        # Input shape
            `(batch_size, features, steps)`
        # Output shape
            2D tensor with shape:
            `(batch_size, features)`
        """

    def __init__(self,
                 m_trainable: bool = False,
                 m_value: float = 1) -> "GlobalExpectationPooling1D":

        super().__init__()

        if m_trainable:
            self.m = nn.Parameter(torch.tensor(m_value, dtype=torch.float))
        else:
            self.m = torch.tensor(m_value, dtype=torch.float)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = x - max(x)
        diff_1 = x - x.max(dim=-1, keepdim=True)[0]
        # x = mx
        diff = self.m * diff_1
        # prob =  exp(x_i)/sum(exp(x_j))
        prob = F.softmax(diff, dim=-1)
        # Expectation = sum(Prob*x)
        expectation = (prob * x).sum(dim=-1)

        return expectation

class KattentionV4(nn.Module):
    def __init__(self, channel_size: int, kernel_size: int = 10, num_heads: int = 32,
                 bias: bool = False, softmax_scale: Optional[float] = None, attn_dropout: float = 0.,
                 reverse: bool = False, hard_init_level: int = 0):
        super().__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads

        # Wattn inplace of Wq.T \codt Wk, by group conv
        self.Wattn = nn.Conv1d(
            in_channels=kernel_size * channel_size,
            out_channels=kernel_size * channel_size * num_heads,
            kernel_size=1,
            groups=kernel_size,
            bias=bias
        )

        self.softmax_scale = softmax_scale
        self.attn_drop = nn.Dropout(attn_dropout)

        self.reverse = reverse

        self.hard_init_level = hard_init_level
        self._init_weights()

    def _init_weights(self):
        # print(f"KattentionV4 initialized with hard_init_level {self.hard_init_level}")
        if self.hard_init_level == 0:
            nn.init.kaiming_normal_(self.Wattn.weight, mode='fan_out', nonlinearity='relu')
        else:
            if self.channel_size == 4:
                w_given = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float32)
            elif self.channel_size == 10:  # for compatability with RIFLE tokenizer
                w_given = torch.zeros(self.channel_size, self.channel_size, dtype=torch.float32)
                w_given[5:9, 5:9] = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                                                 dtype=torch.float32)
            w_given = w_given[None, None, :, :].repeat(self.kernel_size, self.num_heads, 1, 1)

            w_ = rearrange(w_given, "g h k1 k2 -> (g h k1) k2 1")
            if self.hard_init_level == 1:
                pass
            elif self.hard_init_level == 2:
                nn.init.kaiming_normal_(w_[::2], mode='fan_out', nonlinearity='relu')
            elif self.hard_init_level == 3:
                nn.init.kaiming_normal_(w_[::4], mode='fan_out', nonlinearity='relu')
            elif self.hard_init_level == 4:
                nn.init.kaiming_normal_(w_[1:], mode='fan_out', nonlinearity='relu')
            else:
                raise ValueError(f"hard_init_level {self.hard_init_level} not supported")

            with torch.no_grad():
                self.Wattn.weight.copy_(w_)

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
        Q = X.unfold(dimension=1, size=self.kernel_size * self.channel_size,
                     step=self.channel_size)  # (batch_size, seq_len - k + 1, kernel_size * channel_size)
        if self.reverse:
            K = X_rev.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        else:
            K = X.unfold(dimension=1, size=self.kernel_size * self.channel_size, step=self.channel_size)
        Q = Q.transpose(1, 2)  # (batch_size, kernel_size * channel_size, seq_len - k + 1)

        # attention operation as depthwise conv
        Q_W = self.Wattn(Q)  # (batch_size, kernel_size * channel_size * num_heads, seq_len - k + 1)
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

class Resi_Conv_layer(nn.Module):
    def __init__(self, poolsize=(1, 1), filters1=48, filters2=24, kernel_size=3, dilation_rate=1, dropout=0):
        super().__init__()
        self.filters1 = filters1
        self.filters2 = filters2
        self.poolsize = poolsize
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.dropout = dropout
        layers = []
        layers.append(
            nn.Sequential(
                nn.BatchNorm2d(self.filters1),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=self.filters1,
                    out_channels=self.filters2,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    # bias=False
                ),
                nn.BatchNorm2d(self.filters2),
                nn.GELU(),
                # nn.Conv2d(self.filters2, self.filters1, kernel_size=1, bias=False),
                # nn.BatchNorm2d(self.filters1),
                # nn.Dropout(self.dropout)
            )
        )
        self.layers = nn.Sequential(*layers)
        # self.pooling = nn.AdaptiveMaxPool2d(self.poolsize)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        out = x
        residual = self.layers(x)
        out = out + residual
        return self.pooling(out)

class ConcatDist2D(nn.Module):
    '''
    Concatenate the pairwise distance to 2d feature matrix.
    产生batch个距离矩阵 [1,1,512,512],
    tensor([[0, 1, 2, 3, 4],
           [1, 0, 1, 2, 3],
           [2, 1, 0, 1, 2],
           [3, 2, 1, 0, 1],
           [4, 3, 2, 1, 0]])
    '''

    def __init__(self):
        super(ConcatDist2D, self).__init__()

    def forward(self, inputs):
        batch_size, seq_len, features = inputs.shape[0], inputs.shape[3], inputs.shape[1]

        ## concat 2D distance ##
        pos = torch.arange(seq_len).unsqueeze(0).repeat(seq_len, 1)
        matrix_repr1 = pos
        matrix_repr2 = pos.t()
        dist = torch.abs(matrix_repr1 - matrix_repr2)
        dist = dist.float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)
        # 第二维与最后一维调换
        dist = torch.transpose(dist, 1, -1)
        return torch.cat([inputs, dist.cuda()], dim=1)

class Conv_layer(nn.Module):
    def __init__(self, filters1=48, filters2=24, kernel_size=3, dilation_rate=1, dropout=0):
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
                    out_channels=self.filters1 // 2,
                    kernel_size=1,
                    padding=0,
                ),
                nn.BatchNorm2d(self.filters1 // 2),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.filters1 // 2,
                    out_channels=self.filters2,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
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

class Conv_layer_(nn.Module):
    def __init__(self, filters1=48, filters2=24, kernel_size=3, dilation_rate=1, dropout=0):
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
                    out_channels=self.filters1,
                    kernel_size=1,
                    padding=0,
                ),
                nn.BatchNorm2d(self.filters1),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.filters1,
                    out_channels=self.filters2,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
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

class Conv_layer_dialated(nn.Module):
    def __init__(self, filters1=48, filters2=24, kernel_size=3, dilation_rate=1, dropout=0):
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
                nn.Conv2d(
                    in_channels=self.filters1,
                    out_channels=self.filters2,
                    kernel_size=3,
                    dilation=dilation_rate,
                    padding=dilation_rate
                ),
                nn.BatchNorm2d(self.filters2),
                nn.ReLU(),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class DepthwiseWeightedConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseWeightedConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                        bias=False)
        self.weights = nn.Parameter(torch.ones(in_channels))  # 每个通道的权重

    def forward(self, x):
        # 逐通道卷积
        x = self.depthwise_conv(x)
        # 对每个通道分配权重
        weighted_x = x * self.weights.view(1, -1, 1, 1)  # 权重广播
        return weighted_x.sum(dim=1, keepdim=True)  # 将所有通道加权求和

class Conv_group(nn.Module):
    def __init__(self, len, filters1=48, kernel_size=3, dilation_rate=1):
        super().__init__()
        self.n = filters1
        self.len = len
        self.act1 = nn.Sequential(
            nn.BatchNorm2d(filters1),
            nn.ReLU()
        )
        self.group_conv = nn.Conv2d(filters1, filters1, kernel_size, stride=1, padding=kernel_size // 2,
                                    groups=filters1)
        self.linear = torch.nn.Linear(self.n, int(self.n))
        self.act2 = nn.Sequential(
            nn.BatchNorm2d(filters1),
            nn.ReLU()
        )

    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # Apply Xavier initialization
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)  # Initialize bias to zero

    def forward(self, x):
        x = self.act1(x)
        x = self.group_conv(x)
        x = rearrange(x, "b c l1 l2 -> b (l1 l2) c")
        x = self.linear(x)
        x = rearrange(x, "b (l1 l2) c -> b c l1 l2", l1=self.len)
        x = self.act2(x)
        return x

class FFN(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=1024, output_size=1):
        super().__init__()

        # 创建全连接层
        self.fc1 = nn.Linear(input_size, hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, output_size)

    def forward(self, x):
        x = F.gelu(self.fc1(x))  # 第一层
        x = self.fc2(x)  # 第二层
        return x

class KNET_plus_seq2(nn.Module):
    def __init__(self, in_channels, kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.Kattention = KattentionV4(channel_size=4, kernel_size=kernel_size, num_heads=number_of_kernel,
                                       reverse=False)
        self.n = number_of_kernel

        self.conv1 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=1)
        self.conv2 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=3)
        self.conv3 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=5)
        self.conv4 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=7)

        self.linear = nn.Sequential(
            torch.nn.Linear(int(self.n * 4), outputdim)
        )

    def forward(self, x_seq, x_icshape):
        x_seq = x_seq.permute((0, 2, 1)).contiguous().float()
        x = self.Kattention(x_seq.transpose(1, 2))["attn_logits"]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        pooling1 = torch.max(x, dim=2).values
        pooling2 = torch.max(pooling1, dim=2).values
        pooling3 = torch.flatten(pooling2, start_dim=1)

        output = self.linear(pooling3)
        output1 = torch.sigmoid(output)
        output2 = output1.squeeze()
        return output2

class Conv_(nn.Module):
    def __init__(self, filters1=48, filters2=24, kernel_size=3, dilation_rate=1, dropout=0):
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
                    out_channels=self.filters2,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
                nn.BatchNorm2d(self.filters2),
                nn.ReLU(),
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

class DR2D_(nn.Module):
    def __init__(self, filters1=48, filters2=24, dilation_rate=1, dropout=0.1):
        super().__init__()
        self.filters1 = filters1
        self.filters2 = filters2
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filters1,
                    out_channels=self.filters2,
                    kernel_size=3,
                    dilation=dilation_rate,
                    padding=dilation_rate,
                    bias=False
                ),
                nn.BatchNorm2d(self.filters2),
                nn.ReLU(),
                # nn.Conv2d(self.filters2, self.filters1, kernel_size=1, bias=False),
                # nn.BatchNorm2d(self.filters1),
                nn.Dropout(self.dropout)
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        residual = self.layers(x)
        out = out + residual
        return out

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(16, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class DilatedResidual2D_no_bias(nn.Module):
    def __init__(self, filters1=48, filters2=24, dilation_rate=1, dropout=0.1):
        super().__init__()
        self.filters1 = filters1
        self.filters2 = filters2
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        layers = []
        layers.append(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.filters1,
                    out_channels=self.filters2,
                    kernel_size=3,
                    dilation=dilation_rate,
                    padding=dilation_rate,
                    bias=False
                ),
                nn.BatchNorm2d(self.filters2),
                nn.ReLU(),
                nn.Conv2d(self.filters2, self.filters1, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.filters1),
                nn.Dropout(self.dropout)
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        residual = self.layers(x)
        out = out + residual
        return out

class DilatedResidual2D_(nn.Module):
    def __init__(self, filters1=48, filters2=24, dilation_rate=1, dropout=0.1):
        super().__init__()
        self.filters1 = filters1
        self.filters2 = filters2
        self.dilation_rate = dilation_rate
        self.dropout = dropout

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.filters1,
                    out_channels=self.filters2,
                    kernel_size=3,
                    dilation=dilation_rate,
                    padding=dilation_rate,
                    bias=True
                ),
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # out = x
        residual = self.layers(x)
        # out = out + residual
        return residual

class KNET_plus_ic(nn.Module):
    def __init__(self, in_channels, kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.Kattention = KattentionV4(channel_size=5, kernel_size=kernel_size, num_heads=number_of_kernel,
                                       reverse=True)
        self.n = number_of_kernel

        self.conv1 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=1)
        self.conv2 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=3)
        self.conv3 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=5)
        self.conv4 = Conv_layer(filters1=self.n, filters2=self.n, kernel_size=7)

        self.linear = nn.Sequential(
            torch.nn.Linear(int(self.n * 4), outputdim)
        )

    def forward(self, x_seq, x_icshape):
        x_icshape = x_icshape.permute((0, 2, 1)).contiguous().float()
        x = self.Kattention(x_icshape.transpose(1, 2))["attn_logits"]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        pooling1 = torch.max(x, dim=2).values
        pooling2 = torch.max(pooling1, dim=2).values
        pooling3 = torch.flatten(pooling2, start_dim=1)

        output = self.linear(pooling3)
        output1 = torch.sigmoid(output)
        output2 = output1.squeeze()
        return output2

class TorchDataset_multi(Dataset):
    def __init__(self, dataset):
        self.X1 = dataset[0][0].astype("float32")
        self.X2 = dataset[0][1].astype("float32")
        self.Y = dataset[1].astype("float32")

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, i):
        return self.X1[i], self.X2[i], self.Y[i]

class TorchDataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset[0].astype("float32")
        self.Y = dataset[1].astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def optimizerSearch(opt, net):
    """
        ["RMSprop", "adam", "adamw", "adadelata", "sgd"]
    Args:
        opt:
    Returns:
    """

    if opt == "RMSprop":

        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.9)
    elif opt == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

    elif opt == "adadelata":
        optimizer = torch.optim.Adadelta(net.parameters(), lr=0.01)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), momentum=0.9)
    return optimizer

class DeepBind(nn.Module):
    def __init__(self, in_channels, num_filters, filter_size, output_dim=1):
        """
        DeepBind model implementation in PyTorch.

        Args:
            input_length (int): Length of the input sequence.
            num_filters (int): Number of convolutional filters.
            filter_size (int): Size of each convolutional filter.
            output_dim (int): Dimension of the output (default is 1 for regression).
        """
        super(DeepBind, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv1d(in_channels=in_channels,  # One-hot encoded DNA/RNA has 4 channels (A, C, G, T)
                              out_channels=num_filters,
                              kernel_size=filter_size)
        # Fully connected layer
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        """
        Forward pass of the DeepBind model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4, input_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Convolutional layer with ReLU activation
        x = x.permute((0, 2, 1)).contiguous().float()
        x = self.conv(x)
        x = F.relu(x)
        # Global max pooling
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        # Fully connected layer
        x = self.fc(x)
        output = torch.sigmoid(x)
        return output.squeeze()

class CNN(nn.Module):
    def __init__(self, in_channels, kernel_size, number_of_kernel, outputdim):
        super().__init__()
        self.Conv1d = torch.nn.Conv1d(in_channels, number_of_kernel, kernel_size, bias=False)
        self.Conv1d2 = torch.nn.Conv1d(number_of_kernel, number_of_kernel * 4, 1, bias=False)
        self.dropout = nn.Dropout(p=0.2)
        self.outputdim = outputdim
        self.linear = torch.nn.Linear(number_of_kernel, outputdim)
        torch.nn.init.xavier_uniform_(self.Conv1d.weight)

    def forward(self, x):
        x = x.permute((0, 2, 1)).contiguous().float()
        mar = self.Conv1d(x)
        pooling1 = torch.max(mar, dim=2).values
        output = self.dropout(pooling1)
        if self.outputdim:
            output = self.linear(output)
            output = torch.sigmoid(output)
        return output.squeeze()

class Down(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv1d(int(in_channels), int(output_channels), kernel_size, padding="same", bias=False)
        self.dropout = nn.Dropout(p=0.4)
        self.pooling1 = torch.nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        conv = self.conv(x)
        pool = self.pooling1(conv)
        # dropout = self.dropout(pool)
        return pool

class UP(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, output_padding=0):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(int(in_channels), int(output_channels), kernel_size, stride=2,
                                             output_padding=output_padding)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        conv = self.conv(x)
        # dropout = self.dropout(conv)
        return conv

def trainS(modelsave_output_prefix, data_set, number_of_kernel, kernel_size,
           random_seed, batch_size, epoch_scheme, DataName, opt, GPUID="0", outputName="Kattentionsingle"):
    '''
    Complete BConv training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    try:
        mkdir(modelsave_output_prefix + "/" + outputName)
    except:
        pass
    mkdir(modelsave_output_prefix.replace("result", "log") + "/" + outputName)

    modelsave_output_filename = modelsave_output_prefix + "/" + outputName + "/model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_opt-" + str(
        opt) + ".pt"
    modellogname = modelsave_output_prefix.replace("result", "log") + "/" + outputName + "/" + "model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                   str(kernel_size) + "_seed-" + str(random_seed) + "_opt_" + str(opt)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")

    # Load dataset
    training_set, test_set = data_set
    # print(training_set[0].shape)
    train_set, test_set = TorchDataset(training_set), TorchDataset(test_set)
    device = 'cuda:' + GPUID
    seqlen = training_set[0].shape[1]

    net = buildModel(kernel_size, number_of_kernel, 1, outputName, seqlen, device=device)
    # ini from tf

    net = net.to(device)
    # print(net)
    sys.stdout.flush()
    BCEloss = nn.BCELoss()

    if os.path.exists(test_prediction_output):
        trained = True
        print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0
    else:
        trained = False
        auc_records = []
        loss_records = []

        training_set_len = len(training_set[0])
        train_set_len = int(training_set_len * 0.8)
        train_set, valid_set = torch.utils.data.random_split(train_set,
                                                             [train_set_len, training_set_len - train_set_len])

        # optimizer = torch.optim.Adadelta(net.parameters(), lr=1, rho=0.9, eps=1.0e-8)
        optimizer = optimizerSearch(opt, net)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=10,
                                                                  min_lr=0.001)
        iterations = 0
        best_loss = 100000
        earlS_num = 0

        writer = SummaryWriter(modellogname)

        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch in range(int(epoch_scheme)):
            # 训练
            net.train()
            for X_iter, Y_true_iter in train_dataloader:
                X_iter = X_iter.to(device)
                Y_true_iter = Y_true_iter.to(device)
                optimizer.zero_grad()
                Y_pred = net(X_iter)
                # acc_train_batch = (Y_pred.argmax(dim=1) == Y_true_iter).float().mean().item() # 准确率
                loss = BCEloss(Y_pred, Y_true_iter.float())
                loss.backward()
                optimizer.step()
                # auc_train_batch = roc_auc_score(Y_true_iter.cpu().detach(), Y_pred.cpu().detach())

                iterations += 1
                if iterations % 64 == 0:
                    loss = loss.item()
                    # print(f'iterations={iterations}, loss={loss}, auc_train_batch={auc_train_batch}')
                    writer.add_scalar('train_batch_loss', loss, iterations)
            # 验证
            net.eval()
            with torch.no_grad():
                total_loss = 0.0
                tem = 0
                for X_iter, Y_true_iter in valid_dataloader:
                    X_iter = X_iter.to(device)
                    Y_true_iter = Y_true_iter.to(device)
                    Y_pred = net(X_iter)
                    loss_iter = BCEloss(Y_pred, Y_true_iter.float())
                    total_loss += loss_iter.item()
                    tem = tem + 1

            lr_scheduler.step(total_loss)
            # print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
            # print(f'valid: epoch={epoch}, val_loss={total_loss / tem}')
            sys.stdout.flush()
            writer.add_scalar('loss', total_loss, epoch)
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(net.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
                earlS_num = 0
                # print("Save the best model\n")
                sys.stdout.flush()
                # print(net.markonv.k_weights)
            else:
                earlS_num = earlS_num + 1

            if earlS_num >= 20:
                break

    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # 测试
    net.load_state_dict(torch.load(modelsave_output_filename.replace(".pt", ".checkpointer.pt"), map_location=device))
    net.eval()
    with torch.no_grad():
        Y_test = torch.tensor([])
        Y_pred = torch.tensor([])

        for X_iter, Y_test_iter in test_dataloader:
            Y_test = torch.concat([Y_test, Y_test_iter])
            X_iter = X_iter.to(device)
            Y_pred_iter = net(X_iter)
            try:
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
            except:
                Y_pred_iter = Y_pred_iter.cpu().detach()
                Y_pred_iter = torch.reshape(Y_pred_iter, (1,))
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
                pass

        loss = BCEloss(Y_pred, Y_test.float()).item()
        test_auc = roc_auc_score(Y_test, Y_pred)
        # print(f'test: test_auc={test_auc}, loss={loss}')
        print(f'record\t{DataName}\t{outputName}\t{kernel_size}\t{random_seed}\t{test_auc}\t{loss}')
        if not trained:
            report_dic = {}
            report_dic["auc"] = auc_records
            report_dic["loss"] = loss_records
            report_dic["test_auc"] = test_auc

            tmp_f = open(test_prediction_output, "wb")
            pickle.dump(np.array(report_dic), tmp_f)
            tmp_f.close()


#########################################################################################################################
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PrismNet import metrics

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class ResidualBlock2D(nn.Module):

    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResidualBlock1D(nn.Module):

    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes * 2, kernel_size=11, stride=1,
                            padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes * 2)
        self.c3 = nn.Conv1d(planes * 2, planes * 8, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes * 8),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Kattention_PrismNet(nn.Module):
    def __init__(self, in_channels, kernel_size, number_of_kernel, outputdim, mode="pu"):
        super(Kattention_PrismNet, self).__init__()
        self.mode = mode
        h_p, h_k = 2, 5
        if mode == "pu":
            self.n_features = 5
        elif mode == "seq":
            self.n_features = 4
            h_p, h_k = 1, 3
        elif mode == "str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"

        base_channel = 8
        self.conv = Conv2d(1, base_channel, kernel_size=(11, h_k), bn=True, same_padding=True)
        self.se = SEBlock(base_channel)
        self.res2d = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p))
        self.res1d = ResidualBlock1D(base_channel * 4)
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8, 1)
        self._initialize_weights()

        self.Kattention = Kattention(kernel_length=kernel_size, kernel_number=number_of_kernel,
                                     channel_size=in_channels, channel_last=0)
        self.dropout = nn.Dropout(p=0.2)
        self.outputdim = outputdim
        self.linear = torch.nn.Linear(int(number_of_kernel), outputdim)
        self.final_linear = nn.Linear(base_channel * 4 * 8 + int(number_of_kernel), outputdim)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_k, input):
        """[forward]

        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.mode == "seq":
            input = input[:, :, :, :4]
        elif self.mode == "str":
            input = input[:, :, :, 4:]
        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x * z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        # x = self.fc(x)

        k_x = input_k.permute((0, 2, 1)).contiguous().float()
        ken = self.Kattention(k_x)
        pooling1 = torch.max(ken, dim=2).values
        pooling2 = torch.max(pooling1, dim=2).values
        output = torch.flatten(pooling2, start_dim=1)
        output = torch.concat([output, x], dim=1)
        output = self.final_linear(output)

        output = torch.sigmoid(output)
        return output

class PrismNet(nn.Module):
    def __init__(self, mode="pu"):
        super(PrismNet, self).__init__()
        self.mode = mode
        h_p, h_k = 2, 5
        if mode == "pu":
            self.n_features = 5
        elif mode == "seq":
            self.n_features = 4
            h_p, h_k = 1, 3
        elif mode == "str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"

        base_channel = 8
        self.conv = Conv2d(1, base_channel, kernel_size=(11, h_k), bn=True, same_padding=True)
        self.se = SEBlock(base_channel)
        self.res2d = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p))
        self.res1d = ResidualBlock1D(base_channel * 4)
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        """[forward]

        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.mode == "seq":
            input = input[:, :, :, :4]
        elif self.mode == "str":
            input = input[:, :, :, 4:]
        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x * z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x

class PrismNet_plus(nn.Module):
    def __init__(self, mode="pu"):
        super(PrismNet_plus, self).__init__()
        self.mode = mode
        h_p, h_k = 2, 5
        if mode == "pu":
            self.n_features = 5
        elif mode == "seq":
            self.n_features = 4
            h_p, h_k = 1, 3
        elif mode == "str":
            self.n_features = 1
            h_p, h_k = 0, 1
        else:
            raise "mode error"

        base_channel = 32
        self.conv = Conv2d(1, base_channel, kernel_size=(11, h_k), bn=True, same_padding=True)
        self.se = SEBlock(base_channel)
        self.res2d = ResidualBlock2D(base_channel, kernel_size=(11, h_k), padding=(5, h_p))
        self.res1d = ResidualBlock1D(base_channel * 4)
        self.avgpool = nn.AvgPool2d((1, self.n_features))
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channel * 4 * 8, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        """[forward]

        Args:
            input ([tensor],N,C,W,H): input features
        """
        if self.mode == "seq":
            input = input[:, :, :, :4]
        elif self.mode == "str":
            input = input[:, :, :, 4:]
        x = self.conv(input)
        x = F.dropout(x, 0.1, training=self.training)
        z = self.se(x)
        x = self.res2d(x * z)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        x = self.res1d(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.gpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def train(batch_size, model, device, train_loader, criterion, optimizer):
    model.train()
    met = metrics.MLMetrics(objective='binary')
    for batch_idx, (x0, y0) in enumerate(train_loader):
        x, y = x0.float().to(device), y0.to(device).float()
        if y0.sum() == 0 or y0.sum() == batch_size:
            continue
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        prob = torch.sigmoid(output)

        y_np = y.to(device='cpu', dtype=torch.long).detach().numpy()
        p_np = prob.to(device='cpu').detach().numpy()
        met.update(y_np, p_np, [loss.item()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    return met

def validate(model, device, test_loader, criterion):
    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            # if y0.sum() ==0:
            #    import pdb; pdb.set_trace()
            output = model(x)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)

            y_np = y.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)

    met = metrics.MLMetrics(objective='binary')
    met.update(y_all, p_all, [l_all.mean()])

    return met, y_all, p_all

def trainPrismnet(modelsave_output_prefix, data_set, number_of_kernel, kernel_size,
                  random_seed, batch_size, epoch_scheme, DataName, opt, GPUID="0", outputName="Prismnet"):
    '''
    Complete BConv training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''
    iterations = 0
    best_loss = 100000
    earlS_num = 0
    nepochs = 200

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    mkdir(modelsave_output_prefix + "/" + outputName)
    mkdir(modelsave_output_prefix.replace("result", "log") + "/" + outputName)

    modelsave_output_filename = modelsave_output_prefix + "/" + outputName + "/model_seed-" + str(random_seed) + ".pt"
    modellogname = modelsave_output_prefix.replace("result", "log") + "/" + outputName + "/" + "model_seed-" + str(
        random_seed)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_", "/Report_")

    # Load dataset
    training_set, test_set = data_set
    train_set, test_set = TorchDataset(training_set), TorchDataset(test_set)
    device = 'cuda:' + GPUID
    if outputName == 'Prismnet_seq':
        net = PrismNet(mode="seq")
    elif outputName == 'Prismnet_plus':
        net = PrismNet_plus()
    else:
        net = PrismNet()
    # ini from tf

    net = net.to(device)
    # total_parameters = sum(p.numel() for p in net.parameters())
    # print(f'{total_parameters}')
    # print(net)
    sys.stdout.flush()
    BCEloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))

    if os.path.exists(test_prediction_output):
        trained = True
        print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0
    else:
        trained = False
        auc_records = []
        loss_records = []

        training_set_len = len(training_set[0])
        train_set_len = int(training_set_len * 0.8)
        train_set, valid_set = torch.utils.data.random_split(train_set,
                                                             [train_set_len, training_set_len - train_set_len])

        # optimizer = torch.optim.Adadelta(net.parameters(), lr=1, rho=0.9, eps=1.0e-8)
        optimizer = optimizerSearch(opt, net)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-6)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=10,
                                                                  min_lr=0.001)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=8, total_epoch=float(nepochs), after_scheduler=None)

        writer = SummaryWriter(modellogname)

        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=True)

        best_auc = 0
        best_acc = 0
        best_epoch = 0
        for epoch in range(1, nepochs + 1):
            t_met = train(batch_size, net, device, train_dataloader, BCEloss, optimizer)
            v_met, _, _ = validate(net, device, valid_dataloader, BCEloss)
            scheduler.step(epoch)
            lr = scheduler.get_lr()[0]
            color_best = 'green'
            if best_auc < v_met.auc:
                best_auc = v_met.auc
                best_acc = v_met.acc
                best_epoch = epoch
                color_best = 'red'
                torch.save(net.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
            if epoch - best_epoch > 15:
                # print("Early stop at %d" % (epoch))
                break
            # print('#'*10+"epoch_"+str(epoch))

        # print("auc: {:.4f} acc: {:.4f}".format(best_auc, best_acc))

        filename = modelsave_output_filename.replace(".pt", ".checkpointer.pt")
        # print("Loading model: {}".format(filename))
        net.load_state_dict(torch.load(filename))

    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # 测试
    net.load_state_dict(torch.load(modelsave_output_filename.replace(".pt", ".checkpointer.pt"), map_location=device))
    net.eval()
    with torch.no_grad():
        Y_test = torch.tensor([])
        Y_pred = torch.tensor([])

        for X_iter, Y_test_iter in test_dataloader:
            pdb.set_trace()
            Y_test = torch.concat([Y_test, Y_test_iter])
            X_iter = X_iter.to(device)
            Y_pred_iter = net(X_iter)
            try:
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
            except:
                Y_pred_iter = Y_pred_iter.cpu().detach()
                Y_pred_iter = torch.reshape(Y_pred_iter, (1,))
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
                pass

        loss = BCEloss(Y_pred, Y_test.float()).item()
        test_auc = roc_auc_score(Y_test, Y_pred)
        # print(f'test: test_auc={test_auc}, loss={loss}')
        # print(f'record\t{DataName}\t{outputName}\t{test_auc}\t{loss}')
        print(f'record\t{DataName}\t{outputName}\t{random_seed}\t{test_auc}\t{loss}')
        if not trained:
            report_dic = {}
            report_dic["auc"] = auc_records
            report_dic["loss"] = loss_records
            report_dic["test_auc"] = test_auc

            tmp_f = open(test_prediction_output, "wb")
            pickle.dump(np.array(report_dic), tmp_f)
            tmp_f.close()


def trainKattentionPrismnet(modelsave_output_prefix, data_set, number_of_kernel, kernel_size,
                            random_seed, batch_size, epoch_scheme, DataName, opt, GPUID="0",
                            outputName="Prismnet_kattention"):
    '''
    Complete BConv training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    mkdir(modelsave_output_prefix + "/" + outputName)
    mkdir(modelsave_output_prefix.replace("result", "log") + "/" + outputName)

    modelsave_output_filename = modelsave_output_prefix + "/" + outputName + "/model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_opt-" + str(
        opt) + ".pt"
    modellogname = modelsave_output_prefix.replace("result", "log") + "/" + outputName + "/" + "model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                   str(kernel_size) + "_seed-" + str(random_seed) + "_opt_" + str(opt)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")

    # Load dataset
    training_set, test_set = data_set
    train_set, test_set = TorchDataset_multi(training_set), TorchDataset_multi(test_set)
    device = 'cuda:' + GPUID

    net = buildModel(kernel_size, number_of_kernel, 1, outputName)
    # ini from tf
    net = net.to(device)
    total_parameters = sum(p.numel() for p in net.parameters())
    print(f'{total_parameters}')
    # print(net)
    sys.stdout.flush()

    BCEloss = nn.BCELoss()
    # BCEloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
    if os.path.exists(test_prediction_output):
        trained = True
        print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0
    else:
        trained = False
        auc_records = []
        loss_records = []

        training_set_len = len(training_set[1])
        train_set_len = int(training_set_len * 0.8)
        train_set, valid_set = torch.utils.data.random_split(train_set,
                                                             [train_set_len, training_set_len - train_set_len])

        # optimizer = torch.optim.Adadelta(net.parameters(), lr=1, rho=0.9, eps=1.0e-8)
        optimizer = optimizerSearch(opt, net)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=4,
                                                                  min_lr=0.0001)
        iterations = 0
        best_loss = 100000
        earlS_num = 0

        writer = SummaryWriter(modellogname)

        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch in range(int(epoch_scheme)):
            # 训练
            net.train()
            for X1_iter, X2_iter, Y_true_iter in train_dataloader:

                X1_iter = X1_iter.to(device)
                X2_iter = X2_iter.to(device)
                Y_true_iter = Y_true_iter.to(device)
                optimizer.zero_grad()
                Y_pred = net(X1_iter, X2_iter)
                # acc_train_batch = (Y_pred.argmax(dim=1) == Y_true_iter).float().mean().item() # 准确率

                loss = BCEloss(Y_pred, Y_true_iter.float())
                loss.backward()
                optimizer.step()
                # auc_train_batch = roc_auc_score(Y_true_iter.cpu().detach(), Y_pred.cpu().detach())

                iterations += 1
                if iterations % 64 == 0:
                    loss = loss.item()
                    # print(f'iterations={iterations}, loss={loss}, auc_train_batch={auc_train_batch}')
                    writer.add_scalar('train_batch_loss', loss, iterations)
            # 验证
            net.eval()
            with torch.no_grad():
                total_loss = 0.0
                tem = 0
                for X_iter1, X_iter2, Y_true_iter in valid_dataloader:
                    X_iter1 = X_iter1.to(device)
                    X_iter2 = X_iter2.to(device)
                    Y_true_iter = Y_true_iter.to(device)
                    Y_pred = net(X_iter1, X_iter2)
                    loss_iter = BCEloss(Y_pred, Y_true_iter.float())
                    total_loss += loss_iter.item()
                    tem = tem + 1

            lr_scheduler.step(total_loss)
            print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
            print(f'valid: epoch={epoch}, val_loss={total_loss / tem}')
            sys.stdout.flush()
            writer.add_scalar('loss', total_loss, epoch)
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(net.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
                earlS_num = 0
                print("Save the best model\n")
                sys.stdout.flush()
                # print(net.markonv.k_weights)
            else:
                earlS_num = earlS_num + 1

            if earlS_num >= 8:
                break

    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # 测试
    net.load_state_dict(torch.load(modelsave_output_filename.replace(".pt", ".checkpointer.pt"), map_location=device))
    net.eval()
    with torch.no_grad():
        Y_test = torch.tensor([])
        Y_pred = torch.tensor([])

        for X_iter1, X_iter2, Y_test_iter in test_dataloader:
            Y_test = torch.concat([Y_test, Y_test_iter])
            X_iter1 = X_iter1.to(device)
            X_iter2 = X_iter2.to(device)
            Y_pred_iter = net(X_iter1, X_iter2)
            try:
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
            except:
                Y_pred_iter = Y_pred_iter.cpu().detach()
                Y_pred_iter = torch.reshape(Y_pred_iter, (1,))
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
                pass

        loss = BCEloss(Y_pred, Y_test.float()).item()
        test_auc = roc_auc_score(Y_test, Y_pred)
        print(f'test: test_auc={test_auc}, loss={loss}')
        # print(f'record\t{DataName}\t{outputName}\t{test_auc}\t{loss}')
        print(f'record\t{DataName}\t{outputName}\t{kernel_size}\t{random_seed}\t{test_auc}\t{loss}')
        if not trained:
            report_dic = {}
            report_dic["auc"] = auc_records
            report_dic["loss"] = loss_records
            report_dic["test_auc"] = test_auc

            tmp_f = open(test_prediction_output, "wb")
            pickle.dump(np.array(report_dic), tmp_f)
            tmp_f.close()


def trainKNET(modelsave_output_prefix, data_set, number_of_kernel, kernel_size,
              random_seed, batch_size, epoch_scheme, DataName, opt, GPUID="0", outputName="Prismnet_kattention"):
    '''
    Complete BConv training for a specified data set, only save the best model
    :param input_shape:   Sequence shape
    :param modelsave_output_prefix:
                                    the path of the model to be saved, the results of all models are saved under the path.The saved information is:：
                                    lThe model parameter with the smallest loss: ：*.checkpointer.hdf5
                                     Historical auc and loss：Report*.pkl
    :param data_set:
                                    data[[training_x,training_y],[test_x,test_y]]
    :param number_of_kernel:
                                    kernel numbers
    :param kernel_size:
                                    kernel size
    :param random_seed:
                                    random seed
    :param batch_size:
                                    batch size
    :param epoch_scheme:           training epochs
    :return:                       model auc and model name which contains hpyer-parameters


    '''

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    try:
        mkdir(modelsave_output_prefix + "/" + outputName)
    except:
        pass
    mkdir(modelsave_output_prefix.replace("result", "log") + "/" + outputName)

    modelsave_output_filename = modelsave_output_prefix + "/" + outputName + "/model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_opt-" + str(
        opt) + ".pt"
    modellogname = modelsave_output_prefix.replace("result", "log") + "/" + outputName + "/" + "model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                   str(kernel_size) + "_seed-" + str(random_seed) + "_opt_" + str(opt)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")

    # Load dataset
    training_set, test_set = data_set
    train_set, test_set = TorchDataset_multi(training_set), TorchDataset_multi(test_set)
    device = 'cuda:' + GPUID

    net = buildModel(kernel_size, number_of_kernel, 1, outputName)
    # ini from tf
    total_parameters = sum(p.numel() for p in net.parameters())
    print(f'{total_parameters}')
    net = net.to(device)
    # print(net)
    sys.stdout.flush()

    # BCEloss = nn.BCELoss()
    BCEloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
    if os.path.exists(test_prediction_output):
        trained = True
        print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0
    else:
        trained = False
        auc_records = []
        loss_records = []

        training_set_len = len(training_set[1])
        train_set_len = int(training_set_len * 0.8)
        train_set, valid_set = torch.utils.data.random_split(train_set,
                                                             [train_set_len, training_set_len - train_set_len])

        # optimizer = torch.optim.Adadelta(net.parameters(), lr=1, rho=0.9, eps=1.0e-8)
        optimizer = optimizerSearch(opt, net)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=5,
                                                                  min_lr=0.0001)
        iterations = 0
        best_loss = 100000
        earlS_num = 0

        writer = SummaryWriter(modellogname)

        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=False)
        # pdb.set_trace()
        for epoch in range(int(epoch_scheme)):
            # 训练
            net.train()
            for X1_iter, X2_iter, Y_true_iter in train_dataloader:

                X1_iter = X1_iter.to(device)
                X2_iter = X2_iter.to(device)
                Y_true_iter = Y_true_iter.to(device)
                optimizer.zero_grad()
                Y_pred = net(X1_iter, X2_iter)
                # acc_train_batch = (Y_pred.argmax(dim=1) == Y_true_iter).float().mean().item() # 准确率

                loss = BCEloss(Y_pred, Y_true_iter.float())
                loss.backward()
                optimizer.step()
                # auc_train_batch = roc_auc_score(Y_true_iter.cpu().detach(), Y_pred.cpu().detach())

                iterations += 1
                if iterations % 64 == 0:
                    loss = loss.item()
                    # print(f'iterations={iterations}, loss={loss}, auc_train_batch={auc_train_batch}')
                    writer.add_scalar('train_batch_loss', loss, iterations)
            # 验证
            net.eval()
            with torch.no_grad():
                total_loss = 0.0
                tem = 0
                for X_iter1, X_iter2, Y_true_iter in valid_dataloader:
                    X_iter1 = X_iter1.to(device)
                    X_iter2 = X_iter2.to(device)
                    Y_true_iter = Y_true_iter.to(device)
                    Y_pred = net(X_iter1, X_iter2)
                    loss_iter = BCEloss(Y_pred, Y_true_iter.float())
                    total_loss += loss_iter.item()
                    tem = tem + 1

            lr_scheduler.step(total_loss)
            print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
            print(f'valid: epoch={epoch}, val_loss={total_loss / tem}')
            sys.stdout.flush()
            writer.add_scalar('loss', total_loss, epoch)
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(net.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
                earlS_num = 0
                print("Save the best model\n")
                sys.stdout.flush()
                # print(net.markonv.k_weights)
            else:
                earlS_num = earlS_num + 1

            if earlS_num >= 12:
                break

    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # 测试
    net.load_state_dict(torch.load(modelsave_output_filename.replace(".pt", ".checkpointer.pt"), map_location=device))
    net.eval()
    with torch.no_grad():
        Y_test = torch.tensor([])
        Y_pred = torch.tensor([])

        for X_iter1, X_iter2, Y_test_iter in test_dataloader:
            Y_test = torch.concat([Y_test, Y_test_iter])
            X_iter1 = X_iter1.to(device)
            X_iter2 = X_iter2.to(device)
            Y_pred_iter = net(X_iter1, X_iter2)
            try:
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
            except:
                Y_pred_iter = Y_pred_iter.cpu().detach()
                Y_pred_iter = torch.reshape(Y_pred_iter, (1,))
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
                pass

        loss = BCEloss(Y_pred, Y_test.float()).item()
        test_auc = roc_auc_score(Y_test, Y_pred)
        print(f'test: test_auc={test_auc}, loss={loss}')
        # print(f'record\t{DataName}\t{outputName}\t{test_auc}\t{loss}')
        print(f'record\t{DataName}\t{outputName}\t{kernel_size}\t{random_seed}\t{test_auc}\t{loss}')
        if not trained:
            report_dic = {}
            report_dic["auc"] = auc_records
            report_dic["loss"] = loss_records
            report_dic["test_auc"] = test_auc

            tmp_f = open(test_prediction_output, "wb")
            pickle.dump(np.array(report_dic), tmp_f)
            tmp_f.close()
