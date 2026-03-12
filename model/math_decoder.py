"""
Math Decoder — Autoregressive LaTeX token generation.

Takes stroke embeddings from a detected math group and generates
LaTeX markup token by token.

Separate from the text decoder because:
- Different vocabulary (LaTeX tokens vs characters)
- Different output structure (LaTeX has nested syntax)
- Allows independent capacity tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MathDecoderConfig


class MathDecoder(nn.Module):
    """Autoregressive decoder for math → LaTeX recognition.

    Architecture mirrors TextDecoder but with LaTeX token vocabulary (~500 tokens).

    During training: teacher forcing.
    During inference: autoregressive with greedy or beam search.
    """

    def __init__(self, cfg: MathDecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)

        # Positional encoding
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
        # Tie weights with embedding
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
            stroke_mask: (batch, num_strokes_in_group) — 1=real, 0=pad
            target_ids: (batch, seq_len) — target LaTeX token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = target_ids.shape

        positions = torch.arange(T, device=target_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(target_ids) + self.pos_encoding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        memory_key_padding_mask = (stroke_mask == 0) if stroke_mask is not None else None

        x = self.transformer(
            tgt=x,
            memory=stroke_embeddings,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        stroke_embeddings: torch.Tensor,
        stroke_mask: torch.Tensor,
        bos_id: int = 1,
        eos_id: int = 2,
        max_length: int = 256,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive greedy generation.

        Args:
            stroke_embeddings: (batch, num_strokes, hidden_dim)
            stroke_mask: (batch, num_strokes)
            bos_id: BOS token ID
            eos_id: EOS token ID
            max_length: max output length
            temperature: sampling temperature (1.0 = greedy argmax)

        Returns:
            generated_ids: (batch, generated_length)
        """
        B = stroke_embeddings.size(0)
        device = stroke_embeddings.device

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
            logits = self.output_proj(x[:, -1:])  # (B, 1, V)

            if temperature <= 0 or temperature == 1.0:
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits.squeeze(1) / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == eos_id)
            if finished.all():
                break

        return generated

    @torch.no_grad()
    def beam_search(
        self,
        stroke_embeddings: torch.Tensor,
        stroke_mask: torch.Tensor,
        beam_width: int = 5,
        bos_id: int = 1,
        eos_id: int = 2,
        max_length: int = 256,
    ) -> torch.Tensor:
        """Beam search decoding for higher quality math recognition.

        Args:
            stroke_embeddings: (1, num_strokes, hidden_dim) — single example
            stroke_mask: (1, num_strokes)
            beam_width: number of beams
            bos_id, eos_id, max_length: standard args

        Returns:
            best_sequence: (1, seq_length) — highest scoring beam
        """
        device = stroke_embeddings.device

        # Expand memory for beam search
        memory = stroke_embeddings.expand(beam_width, -1, -1)
        mem_mask = stroke_mask.expand(beam_width, -1) if stroke_mask is not None else None
        memory_key_padding_mask = (mem_mask == 0) if mem_mask is not None else None

        # Initialize beams: (beam_width, 1)
        beams = torch.full((beam_width, 1), bos_id, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_width, device=device)
        beam_scores[1:] = -1e9  # only first beam is active initially

        finished_beams = []

        for step in range(max_length - 1):
            T = beams.size(1)
            positions = torch.arange(T, device=device).unsqueeze(0).expand(beam_width, -1)
            x = self.token_embed(beams) + self.pos_encoding(positions)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

            x = self.transformer(
                tgt=x,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            x = self.norm(x)
            logits = self.output_proj(x[:, -1])  # (beam_width, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # Score all possible next tokens
            next_scores = beam_scores.unsqueeze(-1) + log_probs  # (beam_width, vocab_size)
            next_scores = next_scores.reshape(-1)  # (beam_width * vocab_size)

            # Select top beam_width candidates
            top_scores, top_indices = next_scores.topk(beam_width, dim=0)
            beam_indices = top_indices // self.cfg.vocab_size
            token_indices = top_indices % self.cfg.vocab_size

            # Update beams
            beams = torch.cat([
                beams[beam_indices],
                token_indices.unsqueeze(-1),
            ], dim=1)
            beam_scores = top_scores

            # Check for completed beams
            for i in range(beam_width):
                if token_indices[i].item() == eos_id:
                    finished_beams.append((beam_scores[i].item(), beams[i]))
                    beam_scores[i] = -1e9

            if len(finished_beams) >= beam_width:
                break

        # Return best beam
        if finished_beams:
            best = max(finished_beams, key=lambda x: x[0])
            return best[1].unsqueeze(0)
        else:
            return beams[0:1]
