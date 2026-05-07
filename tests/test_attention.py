import torch

from src.attention import MultiHeadAttention, ScaledDotProductAttention


def test_scaled_dot_product_attention_output_and_weights_shapes() -> None:
    attention = ScaledDotProductAttention()
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)

    output, weights = attention(q, k, v)

    assert output.shape == (2, 4, 3, 8)
    assert weights.shape == (2, 4, 3, 5)


def test_scaled_dot_product_attention_masking() -> None:
    attention = ScaledDotProductAttention()
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])
    mask = torch.tensor([[[[True, False]]]])

    output, weights = attention(q, k, v, mask=mask)

    assert torch.allclose(weights[..., 1], torch.zeros_like(weights[..., 1]))
    assert torch.allclose(output, torch.tensor([[[[10.0, 0.0]]]]), atol=1e-6)


def test_multi_head_attention_shape_and_gradients() -> None:
    module = MultiHeadAttention(d_model=32, h=8, dropout=0.0)
    x = torch.randn(2, 6, 32, requires_grad=True)

    output, weights = module(x, x, x)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 6, 32)
    assert weights.shape == (2, 8, 6, 6)
    assert x.grad is not None
