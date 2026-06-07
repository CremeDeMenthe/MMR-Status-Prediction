from typing import Tuple, Optional, Union

import torch
import torch.nn as nn


class ConvBNRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        **kwargs,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"Channel count must be positive: in={in_channels}, out={out_channels}"
            )
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"Channel count must be positive: in={in_channels}, out={out_channels}"
            )
        self.block = nn.Sequential(
            ConvBNRelu(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=padding),
            nn.BatchNorm3d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.block(x)
        out = out + res
        out = self.relu(out)
        return out


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
    ) -> None:
        super(ResNet18, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.num_class = num_class
        self.conv1 = nn.Sequential(
            ConvBNRelu(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 1, 1)
        self.layer2 = self._make_layer(64, 128, 1, 2)
        self.layer3 = self._make_layer(128, 256, 1, 2)
        self.layer4 = self._make_layer(256, 512, 1, 2)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        layers = [Block(in_channels, out_channels, stride)]
        for _ in range(1, n_blocks):
            layers.append(Block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
