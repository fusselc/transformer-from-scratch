"""Masking helpers and the Transformer/Noam learning-rate schedule."""

from __future__ import annotations

from typing import Optional

import torch


def generate_square_subsequent_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a lower-triangular causal mask with shape (size, size)."""

    return torch.tril(torch.ones((size, size), device=device, dtype=torch.bool))


def create_padding_mask(tokens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create padding mask with shape (batch, 1, 1, seq_len); True means keep."""

    return (tokens != pad_token_id).unsqueeze(1).unsqueeze(2)


def combine_padding_and_causal_masks(padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    """Combine a padding mask and causal mask into a single broadcastable attention mask."""

    if causal_mask.dim() == 2:
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    return padding_mask & causal_mask


def transformer_learning_rate(step: int, d_model: int, warmup: int = 4000) -> float:
    """Noam learning-rate schedule from the Transformer paper."""

    if step <= 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))


def combine_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Combine broadcastable boolean masks by logical AND."""

    result: Optional[torch.Tensor] = None
    for mask in masks:
        if mask is None:
            continue
        result = mask if result is None else (result & mask)
    return result


class TransformerLRScheduler:
    """Noam learning-rate schedule from the Transformer paper."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int = 512, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def rate(self, step: Optional[int] = None) -> float:
        step = self.step_num if step is None else step
        if step <= 0:
            return 0.0
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

    def step(self) -> float:
        self.step_num += 1
        lr = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        return self.rate()
