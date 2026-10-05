#!/usr/bin/env python3
"""Full fine-tune ImageNet ConvNeXt-Base on local DermaMNIST@224 (.npy export).

Same protocol as train_resnet50_derma.py: differential LRs (backbone << head),
inverse-frequency class weights, early stop on validation balanced accuracy.

Example
-------
    conda activate torch
    cd /path/to/vision-classifier-arena
    python train_convnext_base_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Base_Weights

REPO_ROOT = Path(__file__).resolve().parent

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class NpyImageDataset(Dataset):
    """DermaMNIST export: `{split}_images.npy` (N,H,W,3 uint8) + `{split}_labels.npy`."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected images (N,H,W,3), got {images.shape}")
        labels = np.asarray(labels).reshape(-1)
        if len(images) != len(labels):
            raise ValueError("images/labels length mismatch")
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        x = transforms.functional.to_pil_image(img)
        if self.transform is not None:
            x = self.transform(x)
        y = int(self.labels[idx])
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    images = np.load(data_dir / f"{split}_images.npy")
    labels = np.load(data_dir / f"{split}_labels.npy")
    return images, labels


def class_weights_from_labels(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    labels = np.asarray(labels).reshape(-1)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    correct = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        correct += (pred == y).sum().item()
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            conf[t, p] += 1

    acc = correct / max(n, 1)
    recalls = []
    f1s = []
    for c in range(num_classes):
        tp = conf[c, c]
        support = conf[c].sum()
        pred_pos = conf[:, c].sum()
        recall = tp / support if support > 0 else 0.0
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        recalls.append(recall)
        f1s.append(f1)

    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "balanced_acc": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def build_model(num_classes: int) -> nn.Module:
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = models.convnext_base(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def param_groups(
    model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float
) -> list[dict]:
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier."):
            head.append(p)
        else:
            backbone.append(p)
    return [
        {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ConvNeXt-Base on DermaMNIST@224.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "dermamnist_224")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "convnext_base_derma")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
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

    model = build_model(num_classes).to(device)
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
        param_groups(model, args.lr_backbone, args.lr_head, args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict] = []
    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_path = out_dir / "best.pt"

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"Device={device} model=ConvNeXt-Base ({n_params:.1f}M) "
        f"train={len(train_ds)} val={len(val_ds)} classes={num_classes} batch={args.batch_size}"
    )
    print(
        f"Full FT: lr_backbone={args.lr_backbone} lr_head={args.lr_head} "
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
            "lr_backbone": optimizer.param_groups[0]["lr"],
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
                    "arch": "convnext_base",
                    "num_classes": num_classes,
                    "class_names": CLASS_NAMES[:num_classes],
                    "args": vars(args),
                    "val": val_metrics,
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
        "arch": "convnext_base",
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
