from typing import Tuple, Optional

import torch
import torch.nn as nn

from models.resnet18 import ResNet18
from models.attention import MultiHeadAttention


class Single_ResNet18_mutli_attention(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
        d_model: int = 512,
        d_k: int = 64,
        n_heads: int = 8,
        d_v: int = 64,
    ) -> None:
        super(Single_ResNet18_mutli_attention, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if num_class <= 0:
            raise ValueError(f"num_class must be positive, got {num_class}")

        self.feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.multihead_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k, n_heads=n_heads, d_v=d_v,
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")

        x = self.feature(x)
        b, c, h, w, d = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(b, c, h, w, d)
        attn_output = attn_output.mean([2, 3, 4])
        return self.fc(attn_output)


class CP_ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
    ) -> None:
        super(CP_ResNet18, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.C_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.P_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 5 or y.dim() != 5:
            raise ValueError(f"Expected 5D inputs, got x={x.dim()}D, y={y.dim()}D")
        if x.size(0) != y.size(0):
            raise ValueError(f"Batch size mismatch: x={x.size(0)}, y={y.size(0)}")

        x = self.C_feature(x)
        x = x.view(x.size(0), -1)
        y = self.P_feature(y)
        y = y.view(y.size(0), -1)
        out = torch.cat((x, y), dim=1)

        if out.size(1) != 1024:
            raise ValueError(f"Expected concatenated dim 1024, got {out.size(1)}")

        out = self.fc1(out)
        return self.fc2(out)


class CP_ResNet18_mutli_attention(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
        d_model: int = 512,
        d_k: int = 64,
        n_heads: int = 8,
        d_v: int = 64,
    ) -> None:
        super(CP_ResNet18_mutli_attention, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.C_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.P_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.multihead_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k, n_heads=n_heads, d_v=d_v,
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 5 or y.dim() != 5:
            raise ValueError(f"Expected 5D inputs, got x={x.dim()}D, y={y.dim()}D")
        if x.size(0) != y.size(0):
            raise ValueError(f"Batch size mismatch: x={x.size(0)}, y={y.size(0)}")

        x = self.C_feature(x)
        y = self.P_feature(y)
        b, c, h, w, d = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        y = y.flatten(2)
        y = y.transpose(1, 2)
        attn_output, _ = self.multihead_attn(x, y, y)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(b, c, h, w, d)
        attn_output = attn_output.mean([2, 3, 4])
        return self.fc(attn_output)


class CPC_ResNet18(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
    ) -> None:
        super(CPC_ResNet18, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.C_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.P_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.fc0 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(65792, 4096),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 5 or y.dim() != 5:
            raise ValueError(f"Expected 5D image inputs, got x={x.dim()}D, y={y.dim()}D")
        if x.size(0) != y.size(0):
            raise ValueError(f"Batch size mismatch: x={x.size(0)}, y={y.size(0)}")

        x = self.C_feature(x)
        x = x.view(x.size(0), -1)
        y = self.P_feature(y)
        y = y.view(y.size(0), -1)
        z = z.view(z.size(0), -1)
        out = torch.cat((x, y, z), dim=1)
        out = self.fc0(out)
        out = self.fc1(out)
        return self.fc2(out)


class CPC_ResNet18_mutli_attention(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
        d_model: int = 512,
        d_k: int = 64,
        n_heads: int = 8,
        d_v: int = 64,
    ) -> None:
        super(CPC_ResNet18_mutli_attention, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.C_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.P_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.multihead_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k, n_heads=n_heads, d_v=d_v,
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")

        x = self.C_feature(x)
        b, c, h, w, d = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(b, c, h, w, d)
        attn_output = attn_output.mean([2, 3, 4])
        return self.fc(attn_output)


class CP_ResNet10_early(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_class: int = 2,
    ) -> None:
        super(CP_ResNet10_early, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        from models.resnet18 import ConvBNRelu, Block
        self.conv1 = nn.Sequential(
            ConvBNRelu(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 1, 1)
        self.layer2 = self._make_layer(64, 128, 1, 2)
        self.layer3 = self._make_layer(128, 256, 1, 2)
        self.layer4 = self._make_layer(256, 512, 1, 2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        from models.resnet18 import Block
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Multi_resNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_class: int = 2,
    ) -> None:
        super(Multi_resNet, self).__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.C_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.P_feature = ResNet18(in_channels=in_channels, num_class=num_class)
        self.T_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_class),
        )
        self.N_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_class),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 5 or y.dim() != 5:
            raise ValueError(f"Expected 5D inputs, got x={x.dim()}D, y={y.dim()}D")
        if x.size(0) != y.size(0):
            raise ValueError(f"Batch size mismatch: x={x.size(0)}, y={y.size(0)}")

        x = self.C_feature(x)
        x = x.view(x.size(0), -1)
        y = self.P_feature(y)
        y = y.view(y.size(0), -1)
        out = torch.cat((x, y), dim=1)
        out_t = self.T_fc(out)
        out_n = self.N_fc(out)
        return out_t, out_n
