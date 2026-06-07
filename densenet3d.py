import math
from typing import Tuple, List, Optional, OrderedDict
from collections import OrderedDict as OD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        efficient: bool = False,
    ) -> None:
        super(_DenseLayer, self).__init__()
        if num_input_features <= 0:
            raise ValueError(f"num_input_features must be positive, got {num_input_features}")
        if growth_rate <= 0:
            raise ValueError(f"growth_rate must be positive, got {growth_rate}")
        if bn_size <= 0:
            raise ValueError(f"bn_size must be positive, got {bn_size}")
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")

        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(
            num_input_features, bn_size * growth_rate,
            kernel_size=1, stride=1, bias=False,
        ))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False,
        ))
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features: torch.Tensor) -> torch.Tensor:
        if len(prev_features) == 0:
            raise ValueError("_DenseLayer requires at least one input feature")

        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(f.requires_grad for f in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        x = self.norm2(bottleneck_output)
        x = self.relu2(x)
        new_features = self.conv2(x)

        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training,
            )
        return new_features


def _bn_function_factory(
    norm: nn.Module,
    relu: nn.Module,
    conv: nn.Module,
):
    def bn_function(*inputs: torch.Tensor) -> torch.Tensor:
        concated_features = torch.cat(inputs, 1)
        out = norm(concated_features)
        out = relu(out)
        out = conv(out)
        return out
    return bn_function


class _Transition(nn.Sequential):
    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
    ) -> None:
        super(_Transition, self).__init__()
        if num_input_features <= 0 or num_output_features <= 0:
            raise ValueError(
                f"Feature counts must be positive: in={num_input_features}, out={num_output_features}"
            )
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(
            num_input_features, num_output_features,
            kernel_size=1, stride=1, bias=False,
        ))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        efficient: bool = False,
    ) -> None:
        super(_DenseBlock, self).__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 12,
        block_config: Tuple[int, ...] = (16, 16, 16),
        compression: float = 0.5,
        num_init_features: int = 24,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1,
        small_inputs: bool = False,
        efficient: bool = False,
    ) -> None:
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, f"compression must be in (0, 1], got {compression}"
        if len(block_config) == 0:
            raise ValueError("block_config must have at least one block")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        if small_inputs:
            self.features = nn.Sequential(OD([
                ('conv0', nn.Conv3d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OD([
                ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm3d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                num_output = int(num_features * compression)
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_output,
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_output

        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got {x.dim()}D")

        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
