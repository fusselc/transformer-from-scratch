import torch

from src.encoder import EncoderLayer


def test_encoder_layer_output_shape_and_attention_shape() -> None:
    layer = EncoderLayer(d_model=32, h=8, d_ff=64, dropout=0.0)
    x = torch.randn(2, 5, 32)
    out, attn = layer(x)

    assert out.shape == (2, 5, 32)
    assert attn.shape == (2, 8, 5, 5)
