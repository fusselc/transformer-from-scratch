from typing import List, Optional, Tuple

import torch
from torch import nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        all_attention_weights: List[torch.Tensor] = []
        for layer in self.layers:
            x, attention_weights = layer(x, src_mask)
            all_attention_weights.append(attention_weights)
        return x, all_attention_weights
