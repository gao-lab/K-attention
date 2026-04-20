from typing import Optional
import logging
from dataclasses import dataclass, field, fields

import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseClassifier(nn.Module):
    def __init__(
        self, in_features: int, mid_features: int | list[int] = 128
    ):
        super().__init__()
        mid_features = [mid_features] if isinstance(mid_features, int) else mid_features
        mid_features = [in_features] + mid_features
        # self.model = nn.Sequential(
        #     *[
        #         nn.Sequential(
        #             nn.Linear(i, o),
        #             nn.ReLU(),
        #         )
        #         for i, o in zip(mid_features[:-1], mid_features[1:])
        #     ],
        #     nn.Linear(mid_features[-1], 1),
        # )
        self.model = nn.Sequential(
            torch.nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor, cls_labels: torch.Tensor):
        """
        Parameters
        ------------------------------
        x: torch.Tensor
            shape of [batch_size, in_features]
        cls_labels: torch.Tensor
            shape of [batch_size,]
        """
        logits = self.model(x).squeeze(-1)

        probs = F.sigmoid(logits)

        cls_loss = F.binary_cross_entropy_with_logits(logits, cls_labels.float())
        return {
            "loss": cls_loss,
            "cls_logits": logits,
            "cls_probs": probs
        }


BaseClassifier_reg = BaseClassifier


@dataclass
class CNNMixerConfig:
    num_layers: int = 1
    in_channels: Optional[int] = None
    out_channels: int = field(init=False)

    # convolution
    conv_kernel_sizes: int | list[int] = 3
    conv_mid_channels: int | list[int] = 64

    def _reconcile_config(self):
        for field in fields(self):
            if not field.name.startswith("conv_"):
                continue
            if isinstance(getattr(self, field.name), (list, tuple)):
                assert len(getattr(self, field.name)) == self.num_layers
            else:
                setattr(
                    self,
                    field.name,
                    [getattr(self, field.name)] * self.num_layers
                )

    def __post_init__(self):
        self._reconcile_config()
        self.out_channels = self.conv_mid_channels[-1]


class CNNMixer(nn.Module):
    def __init__(
        self,
        config: CNNMixerConfig,
    ):
        super().__init__()
        self.config = config
        assert self.config.in_channels is not None, "in_channels should be set"
        cnn_channels = [self.config.in_channels] + self.config.conv_mid_channels
        self.out_channels = config.out_channels
        self.cnn = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(o),
                    nn.Dropout2d(0.1),
                )
                for i, o in zip(cnn_channels[:-1], cnn_channels[1:])
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.cnn(x)

