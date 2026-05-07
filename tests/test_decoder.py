import torch

from src.decoder import DecoderLayer
from src.utils import generate_square_subsequent_mask


def test_decoder_layer_shapes() -> None:
    layer = DecoderLayer(d_model=32, h=8, d_ff=64, dropout=0.0)
    x = torch.randn(2, 4, 32)
    memory = torch.randn(2, 5, 32)
    tgt_mask = generate_square_subsequent_mask(4)

    out, self_attn, cross_attn = layer(x, memory, tgt_mask=tgt_mask)

    assert out.shape == (2, 4, 32)
    assert self_attn.shape == (2, 8, 4, 4)
    assert cross_attn.shape == (2, 8, 4, 5)


def test_causal_mask_blocks_future_attention() -> None:
    layer = DecoderLayer(d_model=32, h=8, d_ff=64, dropout=0.0)
    x = torch.randn(1, 4, 32)
    memory = torch.randn(1, 4, 32)
    tgt_mask = generate_square_subsequent_mask(4)

    _, self_attn, _ = layer(x, memory, tgt_mask=tgt_mask)

    upper = torch.triu(self_attn[0, 0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)
