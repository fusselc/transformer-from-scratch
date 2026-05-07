import math

import torch

from src.embedding import PositionalEncoding, TransformerEmbedding


def test_positional_encoding_is_registered_buffer() -> None:
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_len=20)
    assert "pe" in dict(pe.named_buffers())
    assert "pe" not in dict(pe.named_parameters())


def test_positional_encoding_matches_formula_at_position_zero() -> None:
    pe = PositionalEncoding(d_model=8, dropout=0.0, max_len=10)
    values = pe.pe[0, 0]
    assert torch.allclose(values[0::2], torch.zeros(4), atol=1e-7)
    assert torch.allclose(values[1::2], torch.ones(4), atol=1e-7)


def test_transformer_embedding_shape() -> None:
    embedding = TransformerEmbedding(vocab_size=100, d_model=32, dropout=0.0, max_len=50)
    tokens = torch.randint(0, 100, (4, 12))
    out = embedding(tokens)
    assert out.shape == (4, 12, 32)
