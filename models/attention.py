from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if Q.size(-1) == 0:
            raise ValueError("Key dimension cannot be zero")

        scores = torch.matmul(Q, K.transpose(-1, -2))
        scale = np.sqrt(Q.size(-1))
        scores = scores / scale
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        n_heads: int,
        d_v: int,
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        if d_model <= 0 or d_k <= 0 or d_v <= 0:
            raise ValueError(
                f"d_model, d_k, d_v must be positive: d_model={d_model}, d_k={d_k}, d_v={d_v}"
            )
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")

        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_model * 4, d_model, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_Q: torch.Tensor,
        input_K: torch.Tensor,
        input_V: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_Q.dim() != 3:
            raise ValueError(f"Expected 3D input (B,N,C), got {input_Q.dim()}D")

        batch_size, N, C = input_Q.shape
        residual = input_Q

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2)
        context = context.reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(context)
        output = self.layer_norm(output + residual)
        return output, attn
