import argparse
from typing import Tuple

import torch
from torch import nn

from src.transformer import Transformer
from src.utils import transformer_learning_rate


def build_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    bos_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    tgt_input = torch.cat([torch.full((batch_size, 1), bos_token_id, device=device), src[:, :-1]], dim=1)
    tgt_output = src
    return src, tgt_input, tgt_output


def train_copy_task(steps: int = 200, batch_size: int = 32, seq_len: int = 12, vocab_size: int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bos_token_id = 1

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        h=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=512,
        dropout=0.1,
        share_embeddings=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.NLLLoss()

    model.train()
    for step in range(1, steps + 1):
        src, tgt_in, tgt_out = build_copy_batch(batch_size, seq_len, vocab_size, bos_token_id, device)

        probabilities = model(src, tgt_in)
        log_probabilities = torch.log(probabilities + 1e-9)
        loss = loss_fn(log_probabilities.view(-1, vocab_size), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = transformer_learning_rate(step, d_model=128, warmup=4000)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % 20 == 0 or step == 1:
            print(f"step={step:04d} loss={loss.item():.4f} lr={lr:.8f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer on a toy copy task")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_copy_task(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
