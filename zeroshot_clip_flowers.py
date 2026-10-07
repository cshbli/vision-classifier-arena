#!/usr/bin/env python3
"""Zero-shot OpenCLIP on local Oxford Flowers-102.

No training. Frozen image + text towers; class scores are cosine similarity
between image embeddings and prompt embeddings (CLIP logit scale).

Default: OpenCLIP ViT-B/16 + DataComp-XL (`datacomp_xl_s13b_b90k`) so the
vision encoder matches the arena Base slot (~86M).

Requires
--------
    conda activate torch
    pip install open_clip_torch

Example
-------
    cd /path/to/vision-classifier-arena
    python zeroshot_clip_flowers.py --data 102flowers --eval-test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_convnext_base_derma import REPO_ROOT, evaluate, set_seed
from train_resnet50_flowers import (
    CLASS_NAMES,
    NUM_CLASSES,
    FlowersImageDataset,
    load_flowers_split,
)
from zeroshot_clip_derma import (
    DEFAULT_ARCH,
    DEFAULT_PRETRAINED,
    ZeroShotCLIP,
    build_text_features,
    import_open_clip,
)

# Natural-image flower prompts (class names are already readable English).
DEFAULT_TEMPLATES = [
    "a photo of a {class}, a type of flower.",
    "a close-up photo of the flower {class}.",
    "a photo of {class}.",
    "a flower photo of {class}.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot OpenCLIP on Oxford Flowers-102.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "102flowers")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "clip_zeroshot_flowers")
    p.add_argument("--arch", default=DEFAULT_ARCH, help="open_clip model name.")
    p.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help="open_clip pretrained tag (openai, laion2b_s34b_b88k, …).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--eval-test",
        action="store_true",
        help="Also evaluate the test split (recommended; no training).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    data_dir = args.data if args.data.is_absolute() else (REPO_ROOT / args.data)
    out_dir = args.out if args.out.is_absolute() else (REPO_ROOT / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    open_clip = import_open_clip()
    device = torch.device(args.device)

    t0 = time.time()
    clip, _, preprocess = open_clip.create_model_and_transforms(
        args.arch, pretrained=args.pretrained
    )
    clip = clip.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.arch)

    val_paths, val_labels = load_flowers_split(data_dir, "val")
    num_classes = int(max(val_labels.max() + 1, NUM_CLASSES))
    class_names = CLASS_NAMES[:num_classes]
    if num_classes != NUM_CLASSES:
        print(f"Note: num_classes={num_classes} (expected {NUM_CLASSES} for Flowers-102)")

    text_features = build_text_features(
        clip, tokenizer, class_names, DEFAULT_TEMPLATES, device
    )
    model = ZeroShotCLIP(clip, text_features, clip.logit_scale.detach()).to(device).eval()

    n_params = sum(p.numel() for p in clip.parameters()) / 1e6
    print(
        f"Device={device} model=OpenCLIP {args.arch} ({args.pretrained}) "
        f"{n_params:.1f}M  val={len(val_labels)} classes={num_classes} "
        f"batch={args.batch_size}  templates={len(DEFAULT_TEMPLATES)}"
    )
    print(f"Templates: {DEFAULT_TEMPLATES}")

    val_ds = FlowersImageDataset(val_paths, val_labels, preprocess)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()
    val_metrics = evaluate(model, val_loader, device, criterion, num_classes)
    print(
        f"val   loss={val_metrics['loss']:.4f}  acc={val_metrics['acc']:.4f}  "
        f"bacc={val_metrics['balanced_acc']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}"
    )

    ckpt_path = out_dir / "zeroshot.pt"
    torch.save(
        {
            "arch": args.arch,
            "pretrained": args.pretrained,
            "dataset": "flowers102",
            "num_classes": num_classes,
            "class_names": class_names,
            "templates": DEFAULT_TEMPLATES,
            "text_features": text_features.cpu(),
            "logit_scale": clip.logit_scale.detach().cpu(),
            "args": vars(args),
            "val": val_metrics,
        },
        ckpt_path,
    )

    elapsed = time.time() - t0
    summary = {
        "arch": f"openclip_{args.arch}",
        "pretrained": args.pretrained,
        "dataset": "flowers102",
        "templates": DEFAULT_TEMPLATES,
        "class_names": class_names,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_path),
        "val": val_metrics,
    }

    if args.eval_test:
        test_paths, test_labels = load_flowers_split(data_dir, "test")
        test_ds = FlowersImageDataset(test_paths, test_labels, preprocess)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        test_metrics = evaluate(model, test_loader, device, criterion, num_classes)
        summary["test"] = test_metrics
        print(
            f"test  loss={test_metrics['loss']:.4f}  acc={test_metrics['acc']:.4f}  "
            f"bacc={test_metrics['balanced_acc']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}"
        )

    (out_dir / "history.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {ckpt_path} and {out_dir / 'history.json'} ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
