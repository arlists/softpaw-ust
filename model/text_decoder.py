"""
Text Decoder — Autoregressive character-level text generation.

Takes stroke embeddings from a detected text group and generates
the text content character by character.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TextDecoderConfig


class TextDecoder(nn.Module):
    """Autoregressive decoder for text recognition.

    Architecture:
        - Token embedding (vocab_size → hidden_dim) + sinusoidal position
        - 4-layer Transformer decoder:
            - Causal self-attention (on generated characters so far)
            - Cross-attention to stroke embeddings of the text group
            - FFN
        - Output projection → vocab_size logits per position

    During training: teacher forcing (feed ground truth characters).
    During inference: autoregressive generation with greedy or beam search.
    """

    def __init__(self, cfg: TextDecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)

        # Positional encoding for output sequence
        self.pos_encoding = nn.Embedding(cfg.max_length, cfg.hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.num_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        # Tie weights with token embedding
        self.output_proj.weight = self.token_embed.weight

        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        stroke_embeddings: torch.Tensor,
        stroke_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            stroke_embeddings: (batch, num_strokes_in_group, hidden_dim)
                — stroke embeddings belonging to this text group
            stroke_mask: (batch, num_strokes_in_group) — 1 for real, 0 for padding
            target_ids: (batch, seq_len) — target character IDs (with BOS prepended)

        Returns:
            logits: (batch, seq_len, vocab_size) — predicted logits at each position
        """
        B, T = target_ids.shape

        # Token + position embeddings
        positions = torch.arange(T, device=target_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(target_ids) + self.pos_encoding(positions)

        # Causal mask for self-attention (prevent attending to future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Memory key padding mask
        memory_key_padding_mask = (stroke_mask == 0) if stroke_mask is not None else None

        # Transformer decoder
        x = self.transformer(
            tgt=x,
            memory=stroke_embeddings,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)  # (B, T, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        stroke_embeddings: torch.Tensor,
        stroke_mask: torch.Tensor,
        bos_id: int = 1,
        eos_id: int = 2,
        max_length: int = 256,
    ) -> torch.Tensor:
        """Autoregressive greedy generation.

        Args:
            stroke_embeddings: (batch, num_strokes, hidden_dim)
            stroke_mask: (batch, num_strokes)
            bos_id: ID of the BOS token
            eos_id: ID of the EOS token
            max_length: maximum generation length

        Returns:
            generated_ids: (batch, generated_length) — generated token IDs
        """
        B = stroke_embeddings.size(0)
        device = stroke_embeddings.device

        # Start with BOS
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        memory_key_padding_mask = (stroke_mask == 0) if stroke_mask is not None else None

        for step in range(max_length - 1):
            T = generated.size(1)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            x = self.token_embed(generated) + self.pos_encoding(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

            x = self.transformer(
                tgt=x,
                memory=stroke_embeddings,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            x = self.norm(x)
            logits = self.output_proj(x[:, -1:])  # (B, 1, vocab_size)
            next_token = logits.argmax(dim=-1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return generated
