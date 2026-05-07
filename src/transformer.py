from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .decoder import Decoder
from .embedding import TransformerEmbedding
from .encoder import Encoder
from .utils import combine_padding_and_causal_masks, create_padding_mask, generate_square_subsequent_mask


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        h: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = True,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id

        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, dropout=dropout, max_len=max_len)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, dropout=dropout, max_len=max_len)

        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.tgt_embedding.token_embedding.embedding.weight = self.src_embedding.token_embedding.embedding.weight

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            h=h,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            h=h,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def make_src_mask(self, src_tokens: torch.Tensor) -> torch.Tensor:
        return create_padding_mask(src_tokens, pad_token_id=self.pad_token_id)

    def make_tgt_mask(self, tgt_tokens: torch.Tensor) -> torch.Tensor:
        padding_mask = create_padding_mask(tgt_tokens, pad_token_id=self.pad_token_id)
        causal_mask = generate_square_subsequent_mask(tgt_tokens.size(1), device=tgt_tokens.device)
        return combine_padding_and_causal_masks(padding_mask, causal_mask)

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_attention: bool = False,
    ) -> Any:
        src_mask = src_mask if src_mask is not None else self.make_src_mask(src_tokens)
        tgt_mask = tgt_mask if tgt_mask is not None else self.make_tgt_mask(tgt_tokens)
        memory_mask = memory_mask if memory_mask is not None else src_mask

        src_embedded = self.src_embedding(src_tokens)
        tgt_embedded = self.tgt_embedding(tgt_tokens)

        memory, encoder_attention = self.encoder(src_embedded, src_mask)
        decoder_output, decoder_self_attention, decoder_cross_attention = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        logits = self.output_projection(decoder_output)
        output = logits if return_logits else self.softmax(logits)

        if not return_attention:
            return output

        attention_info: Dict[str, Any] = {
            "encoder": encoder_attention,
            "decoder_self": decoder_self_attention,
            "decoder_cross": decoder_cross_attention,
        }
        return output, attention_info
