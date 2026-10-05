#!/usr/bin/env python3
"""LoRA fine-tune ImageNet ConvNeXt-Base on local DermaMNIST@224.

Freezes the backbone. Injects classic LoRA (rank-r adapters) on the **Linear**
layers inside ConvNeXt blocks (the 1×1 / inverted-bottleneck projections).
Does **not** LoRA the 7×7 depthwise convs. The classification head stays fully
trainable. No extra PEFT package required.

Same data / metrics protocol as train_convnext_base_derma.py.

Example
-------
    conda activate torch
    cd /path/to/vision-classifier-arena
    python train_convnext_base_lora_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_convnext_base_derma import (
    CLASS_NAMES,
    NpyImageDataset,
    REPO_ROOT,
    build_model,
    build_transforms,
    class_weights_from_labels,
    evaluate,
    load_split,
    set_seed,
)


class LoRALinear(nn.Module):
    """y = W x + (α / r) B A x, with W frozen."""

    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if r < 1:
            raise ValueError("LoRA rank r must be >= 1")
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.empty(linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        z = F.linear(self.dropout(x), self.lora_A)
        z = F.linear(z, self.lora_B)
        return base + self.scale * z


def inject_lora_on_linears(module: nn.Module, r: int, alpha: float, dropout: float) -> int:
    """Replace nn.Linear children with LoRALinear. Returns how many were wrapped."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            n += 1
        else:
            n += inject_lora_on_linears(child, r, alpha, dropout)
    return n


def apply_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> int:
    n = inject_lora_on_linears(model.features, r, alpha, dropout)
    for p in model.features.parameters():
        p.requires_grad = False
    for m in model.features.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    return n


def count_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def param_groups(
    model: nn.Module, lr_lora: float, lr_head: float, weight_decay: float
) -> list[dict]:
    lora, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier."):
            head.append(p)
        else:
            lora.append(p)
    if not lora:
        raise RuntimeError("No LoRA parameters found — inject_lora may have failed.")
    return [
        {"params": lora, "lr": lr_lora, "weight_decay": weight_decay},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune ConvNeXt-Base on DermaMNIST@224.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "dermamnist_224")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "convnext_base_lora_derma")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr-lora", type=float, default=1e-3)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
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
    train_tf, eval_tf = build_transforms(args.image_size)

    train_images, train_labels = load_split(data_dir, "train")
    val_images, val_labels = load_split(data_dir, "val")
    num_classes = int(max(train_labels.max(), val_labels.max()) + 1)
    if num_classes != len(CLASS_NAMES):
        print(f"Note: num_classes={num_classes} (expected {len(CLASS_NAMES)} for DermaMNIST)")

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

    model = build_model(num_classes)
    n_wrapped = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model = model.to(device)
    trainable, total = count_params(model)

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
        param_groups(model, args.lr_lora, args.lr_head, args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict] = []
    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_path = out_dir / "best.pt"

    print(
        f"Device={device} model=ConvNeXt-Base + LoRA "
        f"(wrapped {n_wrapped} Linear layers in features) "
        f"trainable={trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%) "
        f"train={len(train_ds)} val={len(val_ds)} classes={num_classes} batch={args.batch_size}"
    )
    print(
        f"LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} "
        f"lr_lora={args.lr_lora} lr_head={args.lr_head} "
        f"wd={args.weight_decay} epochs={args.epochs} patience={args.patience}"
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
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
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
                    "model": model.state_dict(),
                    "arch": "convnext_base_lora",
                    "lora": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                    },
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
        "arch": "convnext_base_lora",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "trainable_params": trainable,
        "total_params": total,
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
        model.load_state_dict(ckpt["model"])
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
