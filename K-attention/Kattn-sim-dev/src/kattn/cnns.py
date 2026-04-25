import torch
from torch import nn
import torch.nn.functional as F

from .modules import BaseClassifier, BaseClassifier_reg


# ---------------------------------------------------------------------------
# CNN-Transformer Hybrid baselines
# ---------------------------------------------------------------------------

class CNNTransformerModel(nn.Module):
    """标准 CNN-Transformer hybrid baseline (非参数匹配版).

    结构: 3层 CNN 局部特征提取 → 2层 Transformer Encoder 全局依赖 → 全局最大池化 → 分类头.
    参数量约 200k–400k（取决于序列长度），远大于 KNET，用于验证 K-attention 在标准设置下的优势.
    """
    def __init__(
        self,
        vocab_size: int = 10,
        cnn_channels: tuple = (32, 64, 128),
        tf_hidden: int = 128,
        tf_layers: int = 2,
        tf_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        c1, c2, c3 = cnn_channels

        self.cnn = nn.Sequential(
            nn.Conv1d(vocab_size, c1, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),         nn.ReLU(), nn.BatchNorm1d(c2),
            nn.Conv1d(c2, c3, kernel_size=7, padding=3),         nn.ReLU(), nn.BatchNorm1d(c3),
        )
        self.proj = nn.Linear(c3, tf_hidden) if c3 != tf_hidden else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=tf_heads, dim_feedforward=tf_hidden * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.classifier = BaseClassifier(in_features=tf_hidden, mid_features=1)

    def forward(self, input_ids: torch.Tensor, cls_labels: torch.Tensor, **kwargs):
        x = F.one_hot(input_ids, num_classes=self.vocab_size).permute(0, 2, 1).float()
        x = self.cnn(x)            # (B, C, L)
        x = x.permute(0, 2, 1)    # (B, L, C)
        x = self.proj(x)
        x = self.transformer(x)    # (B, L, tf_hidden)
        x = x.max(dim=1).values   # (B, tf_hidden)
        return self.classifier(x, cls_labels)


class CNNTransformerModelMatched(nn.Module):
    """参数匹配 CNN-Transformer hybrid baseline (~80k 参数，与 KNET 相当).

    结构: 2层 CNN (vocab→32→64) → 2层 Transformer (hidden=64, 4头, FFN=128)
    → 全局最大池化 → 分类头. 总参数量约 78k，与 KNET(kernel=12, num_kernels=64) 匹配.
    """
    def __init__(
        self,
        vocab_size: int = 10,
        cnn_channels: tuple = (32, 64),
        tf_hidden: int = 64,
        tf_layers: int = 2,
        tf_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        c1, c2 = cnn_channels

        self.cnn = nn.Sequential(
            nn.Conv1d(vocab_size, c1, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),         nn.ReLU(), nn.BatchNorm1d(c2),
        )
        self.proj = nn.Linear(c2, tf_hidden) if c2 != tf_hidden else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=tf_heads, dim_feedforward=tf_hidden * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.classifier = BaseClassifier(in_features=tf_hidden, mid_features=1)

    def forward(self, input_ids: torch.Tensor, cls_labels: torch.Tensor, **kwargs):
        x = F.one_hot(input_ids, num_classes=self.vocab_size).permute(0, 2, 1).float()
        x = self.cnn(x)            # (B, C, L)
        x = x.permute(0, 2, 1)    # (B, L, C)
        x = self.proj(x)
        x = self.transformer(x)    # (B, L, tf_hidden)
        x = x.max(dim=1).values   # (B, tf_hidden)
        return self.classifier(x, cls_labels)


class CNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10,
        cls_mid_channels: int | list[int] = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.features = nn.Sequential(
            nn.Conv1d(vocab_size, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(4)
        self.classifier = BaseClassifier(
            in_features=128 * 4,
            mid_features=cls_mid_channels
        )

    def forward(self, input_ids: torch.Tensor, cls_labels: torch.Tensor, **kwargs):
        """
        Parameters
        ----------
        input_ids: torch.Tensor
            The input tensor of shape (batch_size, seq_len = 256).
        """
        one_hot = F.one_hot(input_ids, num_classes=self.vocab_size).permute(0, 2, 1).float()
        x = self.features(one_hot)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x, cls_labels)


# ---------------------------------------------------------------------------
# CRISPR CNN-Transformer Hybrid baselines (regression)
# ---------------------------------------------------------------------------

class CNNTransformerCrispr(nn.Module):
    """CNN-Transformer hybrid for CRISPR gRNA efficiency prediction (regression).

    Structure: one-hot(30-mer, last 4 channels) -> 3-layer CNN -> 2-layer Transformer
    -> global max pool -> FC regression head (MSE loss).
    """
    def __init__(
        self,
        vocab_size: int = 10,
        cnn_channels: tuple = (32, 64, 128),
        tf_hidden: int = 128,
        tf_layers: int = 2,
        tf_heads: int = 4,
        fc_collect: int = 80,
        fc_hidden: int = 60,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        c1, c2, c3 = cnn_channels

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.cnn = nn.Sequential(
            nn.Conv1d(4, c1, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(c2),
            nn.Conv1d(c2, c3, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(c3),
        )
        self.proj = nn.Linear(c3, tf_hidden) if c3 != tf_hidden else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=tf_heads, dim_feedforward=tf_hidden * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        self.fc_collect = nn.Sequential(
            nn.Linear(tf_hidden, fc_collect),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(fc_collect, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = BaseClassifier_reg(in_features=fc_hidden, mid_features=1)

    def forward(self, input_ids: torch.Tensor, cls_labels: torch.Tensor, crisproff=None):
        X = self.embedding(input_ids)[:, :, -4:]       # (B, L, 4)
        x = X.permute(0, 2, 1)                         # (B, 4, L)
        x = self.cnn(x)                                 # (B, C, L)
        x = x.permute(0, 2, 1)                         # (B, L, C)
        x = self.proj(x)
        x = self.transformer(x)                         # (B, L, tf_hidden)
        x = x.max(dim=1).values                        # (B, tf_hidden)
        x = self.fc_collect(x)
        x = self.regressor(x)
        return self.classifier(x, cls_labels)


class CNNTransformerCrisprMatched(nn.Module):
    """Parameter-matched CNN-Transformer hybrid for CRISPR (regression).

    Smaller architecture to match KNET_Crispr parameter count.
    Structure: one-hot(30-mer) -> 2-layer CNN(4->32->64) -> 2-layer Transformer(64h, 4head)
    -> global max pool -> FC regression head.
    """
    def __init__(
        self,
        vocab_size: int = 10,
        cnn_channels: tuple = (32, 64),
        tf_hidden: int = 64,
        tf_layers: int = 2,
        tf_heads: int = 4,
        fc_collect: int = 80,
        fc_hidden: int = 60,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        c1, c2 = cnn_channels

        self.embedding = lambda x: F.one_hot(x, num_classes=vocab_size).float()

        self.cnn = nn.Sequential(
            nn.Conv1d(4, c1, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(c2),
        )
        self.proj = nn.Linear(c2, tf_hidden) if c2 != tf_hidden else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_hidden, nhead=tf_heads, dim_feedforward=tf_hidden * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        self.fc_collect = nn.Sequential(
            nn.Linear(tf_hidden, fc_collect),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(fc_collect, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = BaseClassifier_reg(in_features=fc_hidden, mid_features=1)

    def forward(self, input_ids: torch.Tensor, cls_labels: torch.Tensor, crisproff=None):
        X = self.embedding(input_ids)[:, :, -4:]       # (B, L, 4)
        x = X.permute(0, 2, 1)                         # (B, 4, L)
        x = self.cnn(x)                                 # (B, C, L)
        x = x.permute(0, 2, 1)                         # (B, L, C)
        x = self.proj(x)
        x = self.transformer(x)                         # (B, L, tf_hidden)
        x = x.max(dim=1).values                        # (B, tf_hidden)
        x = self.fc_collect(x)
        x = self.regressor(x)
        return self.classifier(x, cls_labels)
