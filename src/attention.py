"""Scaled dot-product and multi-head attention."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V.

    Args:
        query: Tensor of shape ``(batch, heads, query_len, d_k)``.
        key: Tensor of shape ``(batch, heads, key_len, d_k)``.
        value: Tensor of shape ``(batch, heads, key_len, d_v)``.
        mask: Optional broadcastable mask where True/1 means attend and False/0 means block.
        dropout: Optional dropout module applied to attention weights.

    Returns:
        A pair ``(output, attention_weights)`` with shapes
        ``(batch, heads, query_len, d_v)`` and ``(batch, heads, query_len, key_len)``.
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism from "Attention Is All You Need".

    Notes:
        This implementation accepts both ``h`` and ``num_heads`` for convenience.
        If both are provided, they must match.
    """

    def __init__(
        self,
        d_model: int = 512,
        h: int = 8,
        dropout: float = 0.1,
        *,
        num_heads: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_heads is not None:
            if h != 8 and h != num_heads:
                raise ValueError("If both h and num_heads are provided, they must match")
            h = num_heads

        if d_model % h != 0:
            raise ValueError("d_model must be divisible by number of heads")

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

    def _prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        prepared_mask = self._prepare_mask(mask) if mask is not None else None
        attended, weights = scaled_dot_product_attention(q, k, v, mask=prepared_mask, dropout=self.dropout)

        attended = attended.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.d_model)
        output = self.out_proj(attended)
        return output, weights
