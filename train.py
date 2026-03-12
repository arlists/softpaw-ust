"""
SoftPaw UST — Main training script.

Usage:
    # Single GPU
    python train.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train.py

    # Resume from checkpoint
    python train.py --resume checkpoints/step_10000.pt

    # Override config
    python train.py --lr 5e-5 --batch_size 16 --epochs 60
"""

import os
import sys
import math
import time
import argparse
import dataclasses
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler

import wandb
from tqdm import tqdm

from config import SoftPawConfig, ModelConfig, GROUP_CLASSES
from model import SoftPawUST
from losses import SoftPawLoss
from data.page_composer import ComposedPageDataset, PageComposer
from data.mathwriting import MathWritingDataset
from data.iam_online import IAMOnlineDataset
from data.quickdraw import QuickDrawDataset
from data.augmentation import augment_page
from data.synthetic_handwriting import SyntheticHandwritingGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Train SoftPaw UST")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--data_dir", type=str, default="./datasets", help="Dataset root")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation run (~30 min on 4090): 50K pages, 5 epochs")
    parser.add_argument("--medium", action="store_true",
                        help="Medium run (~8-12 hrs on 4090): 500K pages, 10 epochs")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Cosine annealing with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_datasets(cfg: SoftPawConfig):
    """Load all source datasets and build the page composer.

    Data sources (in order of importance for text):
    1. Synthetic handwriting from fonts — unlimited, diverse vocabulary
    2. IAM Online — real human handwriting (13K, used as seed + validation)
    3. MathWriting — 630K real math expressions with strokes
    4. QuickDraw — drawings (mouse-drawn, supplemented by sketch datasets)
    """
    data_cfg = cfg.data
    print("Loading datasets...")

    # ---- TEXT ----
    # Primary: synthetic handwriting from writing stroke definitions + text corpus
    text_samples = []

    fonts_base = os.path.join(os.path.dirname(data_cfg.iam_online_dir), "fonts")
    synth_gen = SyntheticHandwritingGenerator(
        font_dir=os.path.join(fonts_base, "handwriting"),
        corpus_file=os.path.join(fonts_base, "corpus.txt"),
    )
    synth_count = data_cfg.synthetic_text_samples
    print(f"  Generating {synth_count:,} synthetic handwriting samples...")
    for strokes, text in synth_gen.iter_samples(count=synth_count):
        if strokes and text:
            text_samples.append((strokes, text))
    print(f"  Synthetic handwriting: {len(text_samples)} samples")

    # Secondary: IAM Online real handwriting
    iam = IAMOnlineDataset(data_cfg.iam_online_dir, split="train")
    iam_count = 0
    for strokes, text in iam.iter_samples():
        if strokes and text:
            text_samples.append((strokes, text))
            iam_count += 1
    print(f"  IAM Online: {iam_count} real handwriting samples")

    # ---- MATH ----
    math_samples = []
    mw = MathWritingDataset(data_cfg.mathwriting_dir, split="train")
    for strokes, latex in mw.iter_samples():
        if strokes and latex:
            math_samples.append((strokes, latex))
    print(f"  MathWriting: {len(math_samples)} math samples")

    # ---- DRAWINGS ----
    drawing_samples = []
    qd = QuickDrawDataset(
        data_cfg.quickdraw_dir,
        max_per_category=data_cfg.quickdraw_samples_per_category,
    )
    for strokes, category in qd.iter_samples():
        if strokes:
            drawing_samples.append((strokes, category))
    print(f"  QuickDraw: {len(drawing_samples)} drawing samples")

    # ---- FALLBACKS ----
    # These only kick in if real datasets aren't downloaded yet.
    # They're intentionally minimal so you notice and fix the data.
    if not text_samples:
        print("  CRITICAL: No text data at all. Using minimal fallback.")
        print("  Fix: download IAM data OR add handwriting fonts to datasets/fonts/handwriting/")
        import numpy as np
        from data.stroke import Stroke
        for _ in range(500):
            n = np.random.randint(10, 30)
            s = Stroke.from_xy(
                x=np.linspace(0, np.random.uniform(50, 200), n),
                y=np.full(n, np.random.uniform(10, 50)) + np.random.normal(0, 2, n),
            )
            text_samples.append(([s], "placeholder"))

    if not math_samples:
        print("  CRITICAL: No math data. Download MathWriting dataset.")
        import numpy as np
        from data.stroke import Stroke
        for _ in range(500):
            n = np.random.randint(5, 20)
            s = Stroke.from_xy(
                x=np.linspace(0, np.random.uniform(30, 100), n),
                y=np.full(n, np.random.uniform(10, 30)) + np.random.normal(0, 1.5, n),
            )
            math_samples.append(([s], "x^2"))

    if not drawing_samples:
        print("  CRITICAL: No drawing data. Download QuickDraw dataset.")
        import numpy as np
        from data.stroke import Stroke
        for _ in range(500):
            t = np.linspace(0, 2 * np.pi, 30)
            s = Stroke.from_xy(
                x=np.cos(t) * np.random.uniform(20, 60) + 50,
                y=np.sin(t) * np.random.uniform(20, 60) + 50,
            )
            drawing_samples.append(([s], "circle"))

    print(f"\n  TOTAL: {len(text_samples):,} text + {len(math_samples):,} math + {len(drawing_samples):,} drawings")

    # Build composer
    composer = PageComposer(
        text_samples=text_samples,
        math_samples=math_samples,
        drawing_samples=drawing_samples,
        cfg=data_cfg,
    )

    return composer


def train():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    # Config
    cfg = SoftPawConfig()
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.resume:
        cfg.training.resume_from = args.resume
    if args.data_dir:
        cfg.data.mathwriting_dir = os.path.join(args.data_dir, "mathwriting")
        cfg.data.iam_online_dir = os.path.join(args.data_dir, "iam_online")
        cfg.data.quickdraw_dir = os.path.join(args.data_dir, "quickdraw")

    # Preset overrides (--quick / --medium)
    if args.quick:
        cfg.data.num_train_pages = 50_000
        cfg.data.num_val_pages = 5_000
        cfg.data.synthetic_text_samples = 5_000
        cfg.data.quickdraw_samples_per_category = 500
        cfg.data.stroke.max_strokes_per_page = 256   # pages have ~30-150 strokes, not 512
        cfg.data.stroke.max_points_per_stroke = 64    # synth strokes are 10-30 pts, not 128
        cfg.training.epochs = 5
        cfg.training.batch_size = 8
        cfg.training.warmup_steps = 200
        cfg.training.recognition_loss_start_step = 200
        cfg.training.val_interval = 500
        cfg.training.save_interval = 500
        cfg.training.log_interval = 20
        cfg.training.num_workers = 4
        args.compile = True  # torch.compile for free speedup
        print("  Mode: QUICK (~30 min on 4090)")
    elif args.medium:
        cfg.data.num_train_pages = 500_000
        cfg.data.num_val_pages = 20_000
        cfg.data.synthetic_text_samples = 50_000
        cfg.data.quickdraw_samples_per_category = 5_000
        cfg.training.epochs = 10
        cfg.training.batch_size = 16
        cfg.training.warmup_steps = 500
        cfg.training.recognition_loss_start_step = 500
        cfg.training.val_interval = 2000
        cfg.training.save_interval = 2000
        cfg.training.log_interval = 50
        print("  Mode: MEDIUM (~8-12 hrs on 4090)")

    # Set seed
    torch.manual_seed(cfg.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"SoftPaw UST Training")
        print(f"  Device: {device}")
        print(f"  World size: {world_size}")
        print(f"  Batch size: {cfg.training.batch_size} x {world_size} = {cfg.training.batch_size * world_size}")

    # Model
    model = SoftPawUST(cfg.model).to(device)

    if is_main:
        params = model.count_parameters()
        print(f"  Parameters:")
        for name, count in params.items():
            print(f"    {name}: {count:,}")

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Loss
    criterion = SoftPawLoss(cfg.loss).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=cfg.training.betas,
        eps=cfg.training.eps,
    )

    # Dataset
    composer = load_datasets(cfg)
    train_dataset = ComposedPageDataset(
        composer=composer,
        num_pages=cfg.data.num_train_pages,
        cfg=cfg.data,
        split="train",
        augmentation_cfg=cfg.augmentation,
    )
    val_dataset = ComposedPageDataset(
        composer=composer,
        num_pages=min(cfg.data.num_val_pages, 10_000),  # smaller for quick val
        cfg=cfg.data,
        split="val",
    )

    # Data loaders
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        prefetch_factor=cfg.training.prefetch_factor,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers // 2,
        pin_memory=True,
        drop_last=False,
    )

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = cfg.training.epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=cfg.training.min_lr / cfg.training.learning_rate,
    )

    # AMP scaler (only needed for float16, not bfloat16)
    use_scaler = cfg.training.use_amp and cfg.training.amp_dtype == "float16" and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_scaler)
    amp_dtype = torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if cfg.training.resume_from:
        checkpoint = torch.load(cfg.training.resume_from, map_location=device, weights_only=False)
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if is_main:
            print(f"  Resumed from step {global_step} (epoch {start_epoch})")

    # W&B
    if is_main and not args.no_wandb:
        wandb.init(
            project=cfg.training.wandb_project,
            entity=cfg.training.wandb_entity,
            config={
                "model": dataclasses.asdict(cfg.model),
                "training": dataclasses.asdict(cfg.training),
                "loss": dataclasses.asdict(cfg.loss),
            },
            resume="allow" if cfg.training.resume_from else None,
        )

    # Checkpoint dir
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    patience_counter = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses = {}

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.training.epochs}",
            disable=not is_main,
        )

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            with autocast(device.type, dtype=amp_dtype, enabled=cfg.training.use_amp):
                outputs = model(batch)
                losses = criterion(
                    outputs, batch,
                    step=global_step,
                    recognition_start_step=cfg.training.recognition_loss_start_step,
                )

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # Accumulate losses for logging
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()

            # Log
            if is_main and global_step % cfg.training.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.4f}",
                    "cls": f"{losses['class'].item():.4f}",
                    "mask": f"{(losses['mask_bce'] + losses['mask_dice']).item():.4f}",
                    "text": f"{losses['text'].item():.4f}",
                    "math": f"{losses['math'].item():.4f}",
                    "lr": f"{lr:.2e}",
                })

                if not args.no_wandb:
                    log_dict = {
                        f"train/{k}": v.item() for k, v in losses.items()
                    }
                    log_dict["train/lr"] = lr
                    log_dict["train/step"] = global_step
                    wandb.log(log_dict, step=global_step)

            # Validation
            if global_step % cfg.training.val_interval == 0:
                should_stop = False

                if is_main:
                    val_loss = validate(model, val_loader, criterion, device, cfg, global_step)

                    if not args.no_wandb:
                        wandb.log({"val/total_loss": val_loss}, step=global_step)

                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(
                            model, optimizer, scheduler, scaler,
                            epoch, global_step, best_val_loss,
                            ckpt_dir / "best.pt"
                        )
                        print(f"  New best val loss: {val_loss:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.training.early_stopping_patience:
                            print(f"  Early stopping at step {global_step}")
                            should_stop = True

                    model.train()

                # Sync early stopping across all DDP ranks to avoid deadlock
                if world_size > 1:
                    stop_tensor = torch.tensor(
                        [int(should_stop)], dtype=torch.int, device=device
                    )
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())

                if should_stop:
                    break

            # Save checkpoint
            if is_main and global_step % cfg.training.save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, best_val_loss,
                    ckpt_dir / f"step_{global_step}.pt"
                )

        if patience_counter >= cfg.training.early_stopping_patience:
            break

        # End of epoch logging
        if is_main:
            n_batches = len(train_loader)
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch+1} avg loss: {avg_losses.get('total', 0):.4f}")

    # Final save
    if is_main:
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            cfg.training.epochs, global_step, best_val_loss,
            ckpt_dir / "final.pt"
        )
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    cleanup_distributed()


@torch.no_grad()
def validate(model, val_loader, criterion, device, cfg, global_step):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    count = 0
    amp_dtype = torch.bfloat16 if cfg.training.amp_dtype == "bfloat16" else torch.float16

    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with autocast(device.type, dtype=amp_dtype, enabled=cfg.training.use_amp):
            outputs = model(batch)
            losses = criterion(
                outputs, batch,
                step=global_step,
                recognition_start_step=cfg.training.recognition_loss_start_step,
            )

        total_loss += losses["total"].item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    print(f"  Validation loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_val_loss, path):
    """Save training checkpoint."""
    model_to_save = model.module if isinstance(model, DDP) else model
    torch.save({
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
    }, path)
    print(f"  Saved checkpoint: {path}")


if __name__ == "__main__":
    train()
