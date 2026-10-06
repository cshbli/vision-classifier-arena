#!/usr/bin/env python3
"""Full fine-tune ImageNet ResNet50 on local Oxford Flowers-102.

Expects repo-root `102flowers/` with `jpg/`, `imagelabels.mat`, `setid.mat`
(see README). Same protocol as train_resnet50_derma.py: differential LRs,
inverse-frequency class weights, early stop on validation balanced accuracy.

No scipy required — labels/splits are loaded with a small MATLAB Level-5 reader.

Example
-------
    conda activate torch
    cd /path/to/vision-classifier-arena
    python train_resnet50_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import ResNet50_Weights

from train_convnext_base_derma import (
    REPO_ROOT,
    build_transforms,
    class_weights_from_labels,
    evaluate,
    set_seed,
)

NUM_CLASSES = 102
SPLIT_KEYS = {"train": "trnid", "val": "valid", "test": "tstid"}

# Torchvision Flowers102 class names (index 0 = label 1 in the .mat files).
CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


def _read_tag(buf: bytes, off: int) -> tuple[int, int, int, int]:
    """Return (dtype, nbytes, data_offset, next_offset)."""
    w0 = struct.unpack_from("<I", buf, off)[0]
    if w0 >> 16:  # small-data element
        return w0 & 0xFFFF, w0 >> 16, off + 4, off + 8
    nbytes = struct.unpack_from("<I", buf, off + 4)[0]
    data_off = off + 8
    next_off = data_off + nbytes
    if next_off % 8:
        next_off += 8 - (next_off % 8)
    return w0, nbytes, data_off, next_off


def _parse_matrix(payload: bytes) -> tuple[str, np.ndarray]:
    off = 0
    _, _, _, off = _read_tag(payload, off)  # array flags
    _, nb, d0, off = _read_tag(payload, off)  # dims
    dims = struct.unpack_from("<" + "i" * (nb // 4), payload, d0)
    _, nb, d0, off = _read_tag(payload, off)  # name
    name = payload[d0 : d0 + nb].split(b"\0")[0].decode("ascii")
    dt, nb, d0, _ = _read_tag(payload, off)  # real data
    type_map = {
        1: np.int8,
        2: np.uint8,
        3: np.int16,
        4: np.uint16,
        5: np.int32,
        6: np.uint32,
        7: np.float32,
        9: np.float64,
    }
    if dt not in type_map:
        raise ValueError(f"Unsupported MATLAB type {dt} for array {name!r}")
    arr = (
        np.frombuffer(payload[d0 : d0 + nb], dtype=type_map[dt])
        .copy()
        .reshape(dims, order="F")
    )
    return name, arr


def _read_elements(buf: bytes, off: int = 0, end: int | None = None) -> dict[str, np.ndarray]:
    if end is None:
        end = len(buf)
    arrays: dict[str, np.ndarray] = {}
    while off + 8 <= end:
        mdtype, nbytes, d0, _padded_next = _read_tag(buf, off)
        # Flowers setid.mat packs miCOMPRESSED elements back-to-back without
        # 8-byte padding; advance by exact payload end.
        next_off = d0 + nbytes
        if next_off > end:
            break
        payload = buf[d0:next_off]
        if mdtype == 15:  # miCOMPRESSED
            arrays.update(_read_elements(zlib.decompress(payload)))
        elif mdtype == 14:  # miMATRIX
            name, arr = _parse_matrix(payload)
            arrays[name] = arr
        off = next_off
    return arrays


def load_mat(path: Path) -> dict[str, np.ndarray]:
    """Load arrays from a MATLAB Level-5 .mat (compressed miMATRIX ok). No scipy."""
    blob = path.read_bytes()
    if blob[:4] != b"MATL":
        raise ValueError(f"Not a MATLAB Level-5 file: {path}")
    return _read_elements(blob, 128)


def load_flowers_split(
    data_dir: Path, split: str
) -> tuple[list[Path], np.ndarray]:
    """Return image paths and 0-based labels for an official Flowers-102 split."""
    if split not in SPLIT_KEYS:
        raise ValueError(f"split must be one of {tuple(SPLIT_KEYS)}, got {split!r}")
    jpg_dir = data_dir / "jpg"
    labels_path = data_dir / "imagelabels.mat"
    setid_path = data_dir / "setid.mat"
    for p in (jpg_dir, labels_path, setid_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. See README: Oxford Flowers-102 — download and extract."
            )

    labels_1based = np.asarray(load_mat(labels_path)["labels"]).reshape(-1)
    set_ids = load_mat(setid_path)
    image_ids = np.asarray(set_ids[SPLIT_KEYS[split]]).reshape(-1).astype(np.int64)
    # .mat labels are 1–102; image ids are 1-indexed into that array.
    y = (labels_1based[image_ids - 1] - 1).astype(np.int64)
    paths = [jpg_dir / f"image_{int(i):05d}.jpg" for i in image_ids]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images, e.g. {missing[0]}")
    return paths, y


class FlowersImageDataset(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform=None):
        self.paths = paths
        self.labels = np.asarray(labels).astype(np.int64)
        self.transform = transform
        if len(self.paths) != len(self.labels):
            raise ValueError("paths/labels length mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def build_model(num_classes: int) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def param_groups(
    model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float
) -> list[dict]:
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head.append(p)
        else:
            backbone.append(p)
    return [
        {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ResNet50 on Oxford Flowers-102.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "102flowers")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "resnet50_flowers")
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

    model = build_model(num_classes).to(device)
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        weights = class_weights_from_labels(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        # Official split is balanced (10/class); weights ≈ 1 — still logged for parity.
        print(
            "Class weights: min={:.3f} max={:.3f} mean={:.3f}".format(
                float(weights.min()), float(weights.max()), float(weights.mean())
            )
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
        f"Device={device} model=ResNet50 ({n_params:.1f}M) "
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
                    "arch": "resnet50",
                    "dataset": "flowers102",
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
        "arch": "resnet50",
        "dataset": "flowers102",
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
