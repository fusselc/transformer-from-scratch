import math
from typing import Optional, Tuple

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by number of heads")

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
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
        attended, weights = self.attention(q, k, v, prepared_mask)

        attended = attended.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.d_model)
        output = self.out_proj(self.dropout(attended))
        return output, weights
