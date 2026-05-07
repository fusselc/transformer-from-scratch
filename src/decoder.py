from typing import List, Optional, Tuple

import torch
from torch import nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_attended, self_attention_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attended))

        cross_attended, cross_attention_weights = self.cross_attention(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(cross_attended))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x, self_attention_weights, cross_attention_weights


class Decoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        self_attention_weights_list: List[torch.Tensor] = []
        cross_attention_weights_list: List[torch.Tensor] = []

        for layer in self.layers:
            x, self_weights, cross_weights = layer(x, memory, tgt_mask, memory_mask)
            self_attention_weights_list.append(self_weights)
            cross_attention_weights_list.append(cross_weights)

        return x, self_attention_weights_list, cross_attention_weights_list
