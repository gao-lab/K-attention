import torch
from torch import nn
import torch.nn.functional as F

from .modules import BaseClassifier


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
