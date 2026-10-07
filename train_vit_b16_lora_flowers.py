#!/usr/bin/env python3
"""LoRA fine-tune ImageNet ViT-B/16 on local Oxford Flowers-102.

Freezes the backbone. Injects classic LoRA on **MLP Linear** layers inside the
encoder. Attention is left alone: torchvision `MultiheadAttention` reads
`out_proj.weight` via the fused functional path (so wrapping `out_proj` breaks),
and QKV lives in fused `in_proj_weight` (not an `nn.Linear`). The classification
head (`heads.*`) stays fully trainable.

Same Flowers data / metrics protocol as train_vit_b16_flowers.py.
No PEFT package required.

Example
-------
    conda activate torch
    cd /path/to/vision-classifier-arena
    python train_vit_b16_lora_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
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
    REPO_ROOT,
    build_transforms,
    class_weights_from_labels,
    evaluate,
    set_seed,
)
from train_convnext_base_lora_derma import count_params
from train_resnet50_flowers import (
    CLASS_NAMES,
    NUM_CLASSES,
    FlowersImageDataset,
    load_flowers_split,
)
from train_vit_b16_derma import build_model
from train_vit_b16_lora_derma import apply_lora, param_groups


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune ViT-B/16 on Oxford Flowers-102.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "102flowers")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "vit_b16_lora_flowers")
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

    train_paths, train_labels = load_flowers_split(data_dir, "train")
    val_paths, val_labels = load_flowers_split(data_dir, "val")
    num_classes = int(max(train_labels.max(), val_labels.max()) + 1)
    if num_classes != NUM_CLASSES:
        print(f"Note: num_classes={num_classes} (expected {NUM_CLASSES} for Flowers-102)")

    train_ds = FlowersImageDataset(train_paths, train_labels, train_tf)
    val_ds = FlowersImageDataset(val_paths, val_labels, eval_tf)
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
    n_wrapped = apply_lora(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model = model.to(device)
    trainable, total = count_params(model)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        weights = class_weights_from_labels(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(
            "Class weights: min={:.3f} max={:.3f} mean={:.3f}".format(
                float(weights.min()), float(weights.max()), float(weights.mean())
            )
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
        f"Device={device} model=ViT-B/16 + LoRA "
        f"(wrapped {n_wrapped} MLP Linear layers; attention frozen) "
        f"trainable={trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%) "
        f"train={len(train_ds)} val={len(val_ds)} classes={num_classes} batch={args.batch_size}"
    )
    print(
        f"LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} "
        f"lr_lora={args.lr_lora} lr_head={args.lr_head} "
        f"wd={args.weight_decay} epochs={args.epochs} patience={args.patience}"
    )
    print(
        "Note: LoRA on MLP only — torchvision MHA uses fused in_proj + functional "
        "out_proj.weight path (wrapping out_proj breaks)."
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
                    "arch": "vit_b_16_lora",
                    "dataset": "flowers102",
                    "lora": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "targets": "encoder MLP Linear only",
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
        "arch": "vit_b_16_lora",
        "dataset": "flowers102",
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": "encoder MLP Linear only",
        },
        "trainable_params": trainable,
        "total_params": total,
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_metric,
        "elapsed_sec": elapsed,
        "best_checkpoint": str(best_path),
        "history": history,
    }

    if args.eval_test and best_path.is_file():
        test_paths, test_labels = load_flowers_split(data_dir, "test")
        test_ds = FlowersImageDataset(test_paths, test_labels, eval_tf)
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
