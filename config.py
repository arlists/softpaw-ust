"""
SoftPaw Unified Stroke Transformer — Configuration

All hyperparameters and settings in one place.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class StrokeConfig:
    """Raw stroke preprocessing."""
    num_features: int = 6           # x, y, pressure, time, altitude, azimuth
    max_points_per_stroke: int = 128
    max_strokes_per_page: int = 512
    page_width: float = 768.0       # PencilKit default page width (points)
    page_height: float = 1024.0     # PencilKit default page height (points)


@dataclass
class DataConfig:
    """Dataset paths and composition parameters."""
    # Dataset root directories (set before training)
    mathwriting_dir: str = "./datasets/mathwriting"
    iam_online_dir: str = "./datasets/iam_online"
    quickdraw_dir: str = "./datasets/quickdraw"
    composed_cache_dir: str = "./datasets/composed_pages"

    # Composition
    num_train_pages: int = 4_000_000
    num_val_pages: int = 100_000
    num_test_pages: int = 100_000
    num_composer_workers: int = 16

    # Content sampling ranges per page
    min_text_groups: int = 2
    max_text_groups: int = 8
    min_math_groups: int = 0
    max_math_groups: int = 4
    min_drawing_groups: int = 0
    max_drawing_groups: int = 3
    min_gestures: int = 0
    max_gestures: int = 5

    # QuickDraw subset
    quickdraw_categories: int = 100
    quickdraw_samples_per_category: int = 20_000

    # Synthetic handwriting (from fonts — unlimited text diversity)
    synthetic_text_samples: int = 500_000

    # Synthetic gestures
    synthetic_gestures_per_type: int = 100_000

    stroke: StrokeConfig = field(default_factory=StrokeConfig)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """On-the-fly data augmentation parameters."""
    spatial_jitter: float = 0.05        # max translation as fraction of page
    scale_range: tuple = (0.8, 1.2)     # random scale
    rotation_range: float = 3.0         # max rotation in degrees
    stroke_point_dropout: float = 0.05  # probability of dropping a point
    pressure_noise_std: float = 0.05    # gaussian noise on pressure
    speed_scale_range: tuple = (0.8, 1.2)  # random time scaling
    altitude_noise_std: float = 0.05    # noise on altitude
    azimuth_noise_std: float = 0.05     # noise on azimuth
    group_dropout_prob: float = 0.05    # probability of dropping entire group
    feature_channel_dropout_prob: float = 0.2  # prob of zeroing pressure/altitude/azimuth per page
    enabled: bool = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class StrokeEncoderConfig:
    """Stroke Encoder (per-stroke transformer)."""
    input_dim: int = 6
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1


@dataclass
class PageTransformerConfig:
    """Page-Level Transformer (cross-stroke attention)."""
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    spatial_embed_dim: int = 128    # MLP output for spatial position


@dataclass
class GroupDecoderConfig:
    """DETR-style Group Decoder."""
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    ffn_dim: int = 1024
    num_queries: int = 32
    num_classes: int = 9            # text, math, drawing, 5 gesture types, no_object
    dropout: float = 0.1


@dataclass
class TextDecoderConfig:
    """Autoregressive text decoder."""
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 1024
    vocab_size: int = 256           # must be >= TextTokenizer().vocab_size (~202)
    max_length: int = 256
    dropout: float = 0.1


@dataclass
class MathDecoderConfig:
    """Autoregressive math/LaTeX decoder."""
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 1024
    vocab_size: int = 400           # must be >= MathTokenizer().vocab_size (~339)
    max_length: int = 256
    dropout: float = 0.1


@dataclass
class RelationshipHeadConfig:
    """MLP for inter-group relationship prediction."""
    input_dim: int = 512            # two concatenated group embeddings
    hidden_dim: int = 256
    num_relations: int = 7          # none, precedes, emphasizes, selects, connects, deletes, proximity


@dataclass
class ModelConfig:
    """Full model configuration."""
    stroke_encoder: StrokeEncoderConfig = field(default_factory=StrokeEncoderConfig)
    page_transformer: PageTransformerConfig = field(default_factory=PageTransformerConfig)
    group_decoder: GroupDecoderConfig = field(default_factory=GroupDecoderConfig)
    text_decoder: TextDecoderConfig = field(default_factory=TextDecoderConfig)
    math_decoder: MathDecoderConfig = field(default_factory=MathDecoderConfig)
    relationship_head: RelationshipHeadConfig = field(default_factory=RelationshipHeadConfig)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    """Multi-task loss weights."""
    # Hungarian matching cost weights
    match_cost_class: float = 2.0
    match_cost_mask_bce: float = 5.0
    match_cost_mask_dice: float = 5.0

    # Loss weights
    weight_class: float = 2.0
    weight_mask_bce: float = 5.0
    weight_mask_dice: float = 5.0
    weight_text: float = 2.0
    weight_math: float = 2.0
    weight_relationship: float = 1.0

    # Class weight for no_object (handles imbalance)
    no_object_weight: float = 0.1

    # Label smoothing
    label_smoothing: float = 0.1

    # Auxiliary losses from intermediate decoder layers
    aux_loss_weight: float = 0.5


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Schedule
    scheduler: str = "cosine"
    warmup_steps: int = 2000
    min_lr: float = 1e-6

    # Phased training: only grouping losses during warmup
    recognition_loss_start_step: int = 2000

    # Training loop
    epochs: int = 40
    batch_size: int = 32            # per GPU
    gradient_clip_max_norm: float = 1.0
    early_stopping_patience: int = 5

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    # Data loading
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Logging
    log_interval: int = 100         # log every N steps
    val_interval: int = 5000        # validate every N steps
    save_interval: int = 5000       # save checkpoint every N steps
    wandb_project: str = "softpaw-ust"
    wandb_entity: Optional[str] = None

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # Distributed
    distributed: bool = True


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@dataclass
class ExportConfig:
    """CoreML export settings."""
    output_dir: str = "./export"
    quantization: str = "mixed"     # "none", "fp16", "int8", "mixed"
    min_deployment_target: str = "iOS17"
    compute_units: str = "ALL"      # "ALL", "CPU_AND_GPU", "CPU_AND_NE"


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class SoftPawConfig:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    seed: int = 42


# Group class names and indices
GROUP_CLASSES = [
    "text",                 # 0
    "math",                 # 1
    "drawing",              # 2
    "gesture_circle",       # 3
    "gesture_underline",    # 4
    "gesture_arrow",        # 5
    "gesture_strikethrough",# 6
    "gesture_bracket",      # 7
    "no_object",            # 8
]

RELATIONSHIP_TYPES = [
    "none",                 # 0
    "precedes",             # 1
    "emphasizes",           # 2
    "selects",              # 3
    "connects",             # 4
    "deletes",              # 5
    "spatial_proximity",    # 6
]

NUM_GROUP_CLASSES = len(GROUP_CLASSES)
NUM_RELATIONSHIP_TYPES = len(RELATIONSHIP_TYPES)
NO_OBJECT_INDEX = GROUP_CLASSES.index("no_object")
