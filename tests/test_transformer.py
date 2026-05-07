import torch

from src.transformer import Transformer


def test_transformer_default_layer_counts() -> None:
    model = Transformer(src_vocab_size=50, tgt_vocab_size=50)
    assert len(model.encoder.layers) == 6
    assert len(model.decoder.layers) == 6


def test_transformer_shared_embeddings_when_vocab_matches() -> None:
    model = Transformer(src_vocab_size=50, tgt_vocab_size=50, d_model=32, h=8, d_ff=64)
    assert model.src_embedding.token_embedding.embedding.weight.data_ptr() == model.tgt_embedding.token_embedding.embedding.weight.data_ptr()


def test_transformer_forward_probabilities_and_gradients() -> None:
    model = Transformer(
        src_vocab_size=40,
        tgt_vocab_size=40,
        d_model=32,
        h=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
    )
    src = torch.randint(0, 40, (2, 5))
    tgt = torch.randint(0, 40, (2, 5))

    probs = model(src, tgt)
    assert probs.shape == (2, 5, 40)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)

    loss = probs.mean()
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_transformer_default_memory_mask_blocks_source_padding_in_cross_attention() -> None:
    model = Transformer(
        src_vocab_size=40,
        tgt_vocab_size=40,
        d_model=32,
        h=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        pad_token_id=0,
    )
    src = torch.tensor([[7, 8, 0, 0]])
    tgt = torch.tensor([[1, 9, 10, 11]])

    _, attention_info = model(src, tgt, return_attention=True)

    for layer_cross_attention in attention_info["decoder_cross"]:
        assert torch.allclose(
            layer_cross_attention[..., 2:],
            torch.zeros_like(layer_cross_attention[..., 2:]),
            atol=1e-6,
        )
