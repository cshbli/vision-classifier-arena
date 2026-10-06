#!/usr/bin/env python3
"""Frozen OpenCLIP image encoder + linear probe on DermaMNIST@224.

Backbone (and text tower) stay frozen. Only Linear(image_dim → num_classes)
is trained. Uses OpenCLIP preprocess (not ImageNet norms). Same CE / class
weights / early-stop-on-bacc protocol as train_dinov3_linear_derma.py.

Default: ViT-B/16 + DataComp-XL (`datacomp_xl_s13b_b90k`).

Requires
--------
    conda activate torch
    pip install open_clip_torch

Example
-------
    cd /path/to/vision-classifier-arena
    python train_clip_linear_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_convnext_base_derma import (
    CLASS_NAMES,
    NpyImageDataset,
    REPO_ROOT,
    class_weights_from_labels,
    evaluate,
    load_split,
    set_seed,
)
from zeroshot_clip_derma import DEFAULT_ARCH, DEFAULT_PRETRAINED, import_open_clip


class FrozenCLIPLinear(nn.Module):
    """Frozen OpenCLIP encode_image + trainable linear classifier."""

    def __init__(self, clip: nn.Module, num_classes: int, embed_dim: int):
        super().__init__()
        self.clip = clip
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()
        self.head = nn.Linear(embed_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.clip.encode_image(images)
        return self.head(feats)


def build_model(
    arch: str, pretrained: str, num_classes: int, device: torch.device
) -> tuple[FrozenCLIPLinear, object, object]:
    open_clip = import_open_clip()
    clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained
    )
    clip = clip.to(device).eval()
    # Do not use getattr(..., clip.embed_dim): the default is evaluated even
    # when visual.output_dim exists, and CLIP has no embed_dim attribute.
    if hasattr(clip.visual, "output_dim"):
        embed_dim = int(clip.visual.output_dim)
    else:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            embed_dim = int(clip.encode_image(dummy).shape[-1])
    model = FrozenCLIPLinear(clip, num_classes=num_classes, embed_dim=embed_dim)
    return model.to(device), preprocess_train, preprocess_val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCLIP linear probe on DermaMNIST@224.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "dermamnist_224")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "clip_linear_derma")
    p.add_argument("--arch", default=DEFAULT_ARCH, help="open_clip model name.")
    p.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help="open_clip pretrained tag (openai, laion2b_s34b_b88k, …).",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="LR for the linear head only.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10, help="Early stop on val balanced acc.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights in CrossEntropyLoss.",
    )
    p.add_argument(
        "--eval-test",
        action="store_true",
        help="After training, evaluate best checkpoint on the test split.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    data_dir = args.data if args.data.is_absolute() else (REPO_ROOT / args.data)
    out_dir = args.out if args.out.is_absolute() else (REPO_ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_images, train_labels = load_split(data_dir, "train")
    val_images, val_labels = load_split(data_dir, "val")
    num_classes = int(max(train_labels.max(), val_labels.max()) + 1)
    if num_classes != len(CLASS_NAMES):
        print(f"Note: num_classes={num_classes} (expected {len(CLASS_NAMES)} for DermaMNIST)")

    model, train_tf, eval_tf = build_model(
        args.arch, args.pretrained, num_classes, device
    )
    train_ds = NpyImageDataset(train_images, train_labels, train_tf)
    val_ds = NpyImageDataset(val_images, val_labels, eval_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        weights = class_weights_from_labels(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(
            "Class weights:",
            {
                CLASS_NAMES[i] if i < len(CLASS_NAMES) else i: float(weights[i])
                for i in range(num_classes)
            },
        )

    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict] = []
    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_path = out_dir / "best.pt"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for p in model.clip.visual.parameters())
    print(
        f"Device={device} model=OpenCLIP {args.arch} ({args.pretrained}) frozen "
        f"vision={vision/1e6:.1f}M  head=Linear "
        f"trainable={trainable:,} / {total:,} ({100 * trainable / total:.4f}%) "
        f"train={len(train_ds)} val={len(val_ds)} classes={num_classes} batch={args.batch_size}"
    )
    print(
        f"Linear probe: lr={args.lr} wd={args.weight_decay} "
        f"epochs={args.epochs} patience={args.patience}"
    )

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        scheduler.step()

        train_loss = running / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, criterion, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.4f}  "
            f"bacc={val_metrics['balanced_acc']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        metric = val_metrics["balanced_acc"]
        if metric > best_metric + 1e-4:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "head": model.head.state_dict(),
                    "arch": args.arch,
                    "pretrained": args.pretrained,
                    "num_classes": num_classes,
                    "class_names": CLASS_NAMES[:num_classes],
                    "args": vars(args),
                    "val": val_metrics,
                    "trainable_params": trainable,
                    "total_params": total,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    elapsed = time.time() - t0
    summary = {
        "arch": "openclip_linear",
        "clip_arch": args.arch,
        "pretrained": args.pretrained,
        "trainable_params": trainable,
        "total_params": total,
        "vision_params": vision,
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_metric,
        "elapsed_sec": elapsed,
        "best_checkpoint": str(best_path),
        "history": history,
    }

    if args.eval_test and best_path.is_file():
        test_images, test_labels = load_split(data_dir, "test")
        test_ds = NpyImageDataset(test_images, test_labels, eval_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.head.load_state_dict(ckpt["head"])
        test_metrics = evaluate(model, test_loader, device, criterion, num_classes)
        summary["test"] = test_metrics
        print(
            f"test  loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
            f"bacc={test_metrics['balanced_acc']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}"
        )

    (out_dir / "history.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {best_path} and {out_dir / 'history.json'}")
    print(f"Best val balanced_acc={best_metric:.4f} @ epoch {best_epoch} ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
