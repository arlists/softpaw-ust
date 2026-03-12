"""
SoftPaw UST — CoreML export pipeline.

Converts the trained PyTorch model to CoreML .mlpackage for on-device inference.

The model is exported as separate components for flexibility:
1. Backbone (stroke encoder + page transformer + group decoder) — runs once per page analysis
2. Text decoder — runs per detected text group (autoregressive)
3. Math decoder — runs per detected math group (autoregressive)
4. Relationship head — runs once after group detection

Usage:
    python export_coreml.py --checkpoint checkpoints/best.pt --output ./export
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from config import SoftPawConfig
from model import SoftPawUST


def export_backbone(model: SoftPawUST, output_dir: Path, cfg: SoftPawConfig):
    """Export the backbone (stroke encoder + page transformer + group decoder).

    This is the main inference component — runs on the full page.
    """
    import coremltools as ct

    class BackboneWrapper(torch.nn.Module):
        """Wraps backbone for clean tracing."""
        def __init__(self, model):
            super().__init__()
            self.stroke_encoder = model.stroke_encoder
            self.page_transformer = model.page_transformer
            self.group_decoder = model.group_decoder

        def forward(self, stroke_points, stroke_masks, page_mask, centroids, temporal_order):
            # Stroke encoder
            embeds = self.stroke_encoder.encode_page_strokes(
                stroke_points, stroke_masks, page_mask
            )
            # Page transformer
            ctx = self.page_transformer(embeds, centroids, temporal_order, page_mask)
            # Group decoder
            group_out = self.group_decoder(ctx, page_mask)
            return (
                group_out["class_logits"],
                group_out["mask_logits"],
                group_out["query_embeddings"],
                ctx,  # pass through for decoder heads
            )

    backbone = BackboneWrapper(model).eval()

    max_strokes = cfg.data.stroke.max_strokes_per_page
    max_points = cfg.data.stroke.max_points_per_stroke

    # Trace with dummy input
    dummy_points = torch.randn(1, max_strokes, max_points, 6)
    dummy_stroke_masks = torch.ones(1, max_strokes, max_points)
    dummy_page_mask = torch.ones(1, max_strokes)
    dummy_centroids = torch.rand(1, max_strokes, 2)
    dummy_order = torch.arange(max_strokes).unsqueeze(0)

    traced = torch.jit.trace(
        backbone,
        (dummy_points, dummy_stroke_masks, dummy_page_mask, dummy_centroids, dummy_order),
    )

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="stroke_points", shape=(1, max_strokes, max_points, 6)),
            ct.TensorType(name="stroke_masks", shape=(1, max_strokes, max_points)),
            ct.TensorType(name="page_mask", shape=(1, max_strokes)),
            ct.TensorType(name="centroids", shape=(1, max_strokes, 2)),
            ct.TensorType(name="temporal_order", shape=(1, max_strokes)),
        ],
        outputs=[
            ct.TensorType(name="class_logits"),
            ct.TensorType(name="mask_logits"),
            ct.TensorType(name="query_embeddings"),
            ct.TensorType(name="stroke_embeddings"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )

    # Quantize
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"),
    ))

    out_path = output_dir / "SoftPawBackbone.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Exported backbone: {out_path}")
    print(f"  Size: {sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")


def export_text_decoder(model: SoftPawUST, output_dir: Path, cfg: SoftPawConfig):
    """Export text decoder as a single-step model for autoregressive use in Swift."""
    import coremltools as ct

    class TextDecoderStep(torch.nn.Module):
        """Single-step text decoder for autoregressive inference."""
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, token_ids, stroke_embeddings, stroke_mask):
            logits = self.decoder(stroke_embeddings, stroke_mask, token_ids)
            return logits[:, -1:]  # only last position logits

    step_model = TextDecoderStep(model.text_decoder).eval()

    max_strokes = 64  # max strokes in a single text group
    max_seq = cfg.model.text_decoder.max_length
    d = cfg.model.page_transformer.hidden_dim

    dummy_tokens = torch.zeros(1, max_seq, dtype=torch.long)
    dummy_strokes = torch.randn(1, max_strokes, d)
    dummy_mask = torch.ones(1, max_strokes)

    traced = torch.jit.trace(step_model, (dummy_tokens, dummy_strokes, dummy_mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token_ids", shape=(1, max_seq), dtype=np.int32),
            ct.TensorType(name="stroke_embeddings", shape=(1, max_strokes, d)),
            ct.TensorType(name="stroke_mask", shape=(1, max_strokes)),
        ],
        outputs=[
            ct.TensorType(name="next_logits"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )

    # FP16 quantization (needs precision for text)
    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="float16"),
    ))

    out_path = output_dir / "SoftPawTextDecoder.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Exported text decoder: {out_path}")


def export_math_decoder(model: SoftPawUST, output_dir: Path, cfg: SoftPawConfig):
    """Export math decoder as a single-step model."""
    import coremltools as ct

    class MathDecoderStep(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, token_ids, stroke_embeddings, stroke_mask):
            logits = self.decoder(stroke_embeddings, stroke_mask, token_ids)
            return logits[:, -1:]

    step_model = MathDecoderStep(model.math_decoder).eval()

    max_strokes = 64
    max_seq = cfg.model.math_decoder.max_length
    d = cfg.model.page_transformer.hidden_dim

    dummy_tokens = torch.zeros(1, max_seq, dtype=torch.long)
    dummy_strokes = torch.randn(1, max_strokes, d)
    dummy_mask = torch.ones(1, max_strokes)

    traced = torch.jit.trace(step_model, (dummy_tokens, dummy_strokes, dummy_mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token_ids", shape=(1, max_seq), dtype=np.int32),
            ct.TensorType(name="stroke_embeddings", shape=(1, max_strokes, d)),
            ct.TensorType(name="stroke_mask", shape=(1, max_strokes)),
        ],
        outputs=[
            ct.TensorType(name="next_logits"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="float16"),
    ))

    out_path = output_dir / "SoftPawMathDecoder.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Exported math decoder: {out_path}")


def export_relationship_head(model: SoftPawUST, output_dir: Path, cfg: SoftPawConfig):
    """Export relationship head."""
    import coremltools as ct

    rel_head = model.relationship_head.eval()

    d = cfg.model.page_transformer.hidden_dim
    Q = cfg.model.group_decoder.num_queries

    dummy_embeds = torch.randn(1, Q, d)
    dummy_mask = torch.ones(1, Q)

    traced = torch.jit.trace(rel_head, (dummy_embeds, dummy_mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="group_embeddings", shape=(1, Q, d)),
            ct.TensorType(name="group_mask", shape=(1, Q)),
        ],
        outputs=[
            ct.TensorType(name="rel_logits"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )

    out_path = output_dir / "SoftPawRelationships.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Exported relationship head: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export SoftPaw UST to CoreML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="./export", help="Output directory")
    args = parser.parse_args()

    cfg = SoftPawConfig()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = SoftPawUST(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Print param counts
    params = model.count_parameters()
    print("Parameters:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Export each component
    print("\nExporting components...")
    export_backbone(model, output_dir, cfg)
    export_text_decoder(model, output_dir, cfg)
    export_math_decoder(model, output_dir, cfg)
    export_relationship_head(model, output_dir, cfg)

    print(f"\nAll models exported to {output_dir}/")
    print("Components:")
    for f in sorted(output_dir.glob("*.mlpackage")):
        size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file()) / 1024 / 1024
        print(f"  {f.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
