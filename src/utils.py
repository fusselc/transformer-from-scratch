from typing import Optional

import torch


def generate_square_subsequent_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.tril(torch.ones((size, size), device=device, dtype=torch.bool))


def create_padding_mask(tokens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    return (tokens != pad_token_id).unsqueeze(1).unsqueeze(2)


def combine_padding_and_causal_masks(padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    if causal_mask.dim() == 2:
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    return padding_mask & causal_mask


def transformer_learning_rate(step: int, d_model: int, warmup: int = 4000) -> float:
    if step <= 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))
