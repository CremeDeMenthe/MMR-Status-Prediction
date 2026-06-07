from typing import Tuple, Optional

import torch
import torch.nn as nn


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be numeric, got {type(alpha)}")
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(
    x: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    return GradReverse.apply(x, alpha)


class OutputFusion(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super(OutputFusion, self).__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.fc = nn.Linear(854 + 13, num_classes)
        self.norm = nn.BatchNorm1d(854 + 13)

    def forward(
        self,
        outputs_P: torch.Tensor,
        outputs_C: torch.Tensor,
        clinic: torch.Tensor,
    ) -> torch.Tensor:
        if outputs_P.dim() != 2 or outputs_C.dim() != 2 or clinic.dim() != 2:
            raise ValueError("All inputs must be 2D tensors (B, features)")
        if outputs_P.size(0) != outputs_C.size(0) or outputs_P.size(0) != clinic.size(0):
            raise ValueError(
                f"Batch size mismatch: P={outputs_P.size(0)}, C={outputs_C.size(0)}, "
                f"clinic={clinic.size(0)}"
            )

        outputs = torch.cat((outputs_P, outputs_C, clinic), dim=1)
        expected_dim = 854 + 13
        if outputs.size(1) != expected_dim:
            raise ValueError(
                f"Concatenated feature dim mismatch: expected {expected_dim}, got {outputs.size(1)}"
            )

        outputs = self.norm(outputs)
        return self.fc(outputs)


class OutputDomain(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super(OutputDomain, self).__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.fc0 = nn.Sequential(
            nn.Linear(854, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        outputs_P: torch.Tensor,
        outputs_C: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        if outputs_P.dim() != 2 or outputs_C.dim() != 2:
            raise ValueError("outputs_P and outputs_C must be 2D tensors (B, features)")
        if outputs_P.size(0) != outputs_C.size(0):
            raise ValueError(
                f"Batch size mismatch: P={outputs_P.size(0)}, C={outputs_C.size(0)}"
            )

        outputs = torch.cat((outputs_P, outputs_C), dim=1)
        if outputs.size(1) != 854:
            raise ValueError(
                f"Concatenated feature dim mismatch: expected 854, got {outputs.size(1)}"
            )

        rev_feat = grad_reverse(outputs, alpha)
        return self.fc0(rev_feat)
