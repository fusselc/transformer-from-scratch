import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.position_encoding(self.token_embedding(tokens))
