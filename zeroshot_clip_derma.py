#!/usr/bin/env python3
"""Zero-shot OpenCLIP on local DermaMNIST@224 (.npy export).

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
    python zeroshot_clip_derma.py --data dermamnist_224 --eval-test
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
    evaluate,
    load_split,
    set_seed,
)

DEFAULT_ARCH = "ViT-B-16"
DEFAULT_PRETRAINED = "datacomp_xl_s13b_b90k"

# MedMNIST DermaMNIST names (abbreviations are too opaque for CLIP text).
CLASS_DISPLAY = {
    "akiec": "actinic keratoses and intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesion",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}

DEFAULT_TEMPLATES = [
    "a dermoscopic photo of {class}.",
    "a dermoscopy image of {class}.",
    "a close-up photo of {class} on the skin.",
    "a photo of {class}, a type of skin lesion.",
]


class ZeroShotCLIP(nn.Module):
    """encode_image → cosine sim vs frozen text classifier."""

    def __init__(
        self, clip: nn.Module, text_features: torch.Tensor, logit_scale: torch.Tensor
    ):
        super().__init__()
        self.clip = clip
        for p in self.clip.parameters():
            p.requires_grad = False
        self.register_buffer("text_features", text_features)
        self.register_buffer("logit_scale", logit_scale)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ self.text_features.T


def import_open_clip():
    try:
        import open_clip
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: install with\n"
            "  pip install open_clip_torch\n"
            f"Original error: {e}"
        ) from e
    return open_clip


@torch.inference_mode()
def build_text_features(
    clip: nn.Module,
    tokenizer,
    class_names: list[str],
    templates: list[str],
    device: torch.device,
) -> torch.Tensor:
    display = [CLASS_DISPLAY.get(c, c.replace("_", " ")) for c in class_names]
    zeroshots = []
    for name in display:
        texts = [t.format(class_=name, **{"class": name}) for t in templates]
        tokens = tokenizer(texts).to(device)
        feats = clip.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean = feats.mean(dim=0)
        mean = mean / mean.norm()
        zeroshots.append(mean)
    return torch.stack(zeroshots, dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot OpenCLIP on DermaMNIST@224.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "dermamnist_224")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "clip_zeroshot_derma")
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

    val_images, val_labels = load_split(data_dir, "val")
    num_classes = int(val_labels.max()) + 1
    class_names = CLASS_NAMES[:num_classes]
    if num_classes != len(CLASS_NAMES):
        print(f"Note: num_classes={num_classes} (expected {len(CLASS_NAMES)} for DermaMNIST)")

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
    for i, c in enumerate(class_names):
        print(f"  class {i} {c}: {CLASS_DISPLAY.get(c, c)}")

    val_ds = NpyImageDataset(val_images, val_labels, preprocess)
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
            "num_classes": num_classes,
            "class_names": class_names,
            "class_display": [CLASS_DISPLAY.get(c, c) for c in class_names],
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
        "templates": DEFAULT_TEMPLATES,
        "class_display": [CLASS_DISPLAY.get(c, c) for c in class_names],
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_path),
        "val": val_metrics,
    }

    if args.eval_test:
        test_images, test_labels = load_split(data_dir, "test")
        test_ds = NpyImageDataset(test_images, test_labels, preprocess)
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
