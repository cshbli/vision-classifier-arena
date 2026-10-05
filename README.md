# Vision Classifier Arena

A benchmark suite comparing image classification performance across small, custom, and domain-specific datasets. Evaluates supervised CNNs (ResNet, ConvNeXt), self-supervised backbones (DINOv2/DINOv3 + MLP), and pre-trained Vision Transformers (ViT, Swin) under frozen feature probing, PEFT (LoRA), and full fine-tuning.

| Architecture Family | Specific Model | Training Strategy | Strengths on Small Data |
|---------------------|----------------|-------------------|-------------------------|
| Classic CNN | ResNet50 | Full Fine-Tuning | Strong baseline, fast convergence. |
| Modern CNN | ConvNeXt-Base | Full / LoRA Fine-Tuning | High accuracy; inductive bias resists overfitting vs standard ViT. |
| Self-Supervised ViT | DINOv2 / DINOv3 | Linear Probe / MLP Head | No backbone tuning; preserves global features. |
| Supervised ViT | ViT-B/16 | Full vs. **LoRA** Fine-Tuning | High ceiling; full FT overfits easily — LoRA helps a lot. |
| Hierarchical ViT | Swin-T / Swin-B | Full / LoRA Fine-Tuning | CNN–Transformer hybrid; more data-efficient than plain ViT. |
| Multimodal | CLIP / OpenCLIP / EVA-CLIP | **Zero-shot** vs. Linear Probe | Zero training images (zero-shot); linear probe is a strong small-data SOTA with low compute. |
| PEFT (strategy) | LoRA on ViT / Swin / ConvNeXt | Train ~1% of params | Higher accuracy + faster training than full FT on small samples. |

## Datasets

Open, compact classification sets for small / fine-grained / domain-specific evaluation (no large-scale corpora).

| Dataset | Domain | Approx. scale | Why it fits | Download |
|---------|--------|---------------|-------------|----------|
| **Oxford-IIIT Pets** | Fine-grained animals | ~7k images, 37 classes | Small, clean, standard transfer baseline | [Project page](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [images](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz) · [annotations](https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz) |
| **Oxford Flowers-102** | Fine-grained | ~8k, 102 classes | Harder fine-grained; few shots per class | [Project page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) · [images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) · [labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) · [splits](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat) |
| **CUB-200-2011** | Fine-grained birds | ~12k, 200 classes | Classic small-data / fine-grained stress test | [Project page](https://www.vision.caltech.edu/datasets/cub_200_2011/) · [Caltech DATA record](https://data.caltech.edu/records/65de6-vp158) |
| **Stanford Cars** | Fine-grained products | ~16k, 196 classes | “Custom catalog” style recognition | [Project page](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) · via [torchvision `StanfordCars`](https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html) / Hugging Face mirrors if Stanford HTTP links fail |
| **FGVC Aircraft** | Fine-grained | ~10k, 100 classes | Additional fine-grained check | [Project page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) · [data archive](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz) |
| **MedMNIST** (2D first; see below) | Medical | Small–medium, many tasks | Domain shift; easy to automate | [medmnist.com](https://medmnist.com/) · [GitHub](https://github.com/MedMNIST/MedMNIST) · [Zenodo MedMNIST+](https://zenodo.org/records/10519652) · start with [`dermamnist_224.npz`](https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1) |
| **EuroSAT** | Satellite / remote sensing | ~27k, 10 classes | Non-natural-image domain | [GitHub](https://github.com/phelber/EuroSAT) · [RGB zip (DFKI)](http://madm.dfki.de/files/sentinel/EuroSAT.zip) |

Use **official splits** when available. Match each backbone’s expected preprocessing (ImageNet norms vs CLIP’s own).

### MedMNIST — which 2D task first

MedMNIST has many subsets (2D + 3D). Skip 3D for now. Prefer size **224** (or 128) for ResNet/ViT/CLIP, not only 28×28.

| Priority | Dataset | Why |
|----------|---------|-----|
| **1. Start** | **DermaMNIST** | Small (~10k), multi-class (7), RGB dermatoscopy, strong domain shift, fast loops |
| **2. Next** | **PathMNIST** | Larger (~107k), 9 tissue classes, histology — harder transfer test |
| **3. Optional** | **OrganAMNIST** | CT organs, 11 classes; pick **one** of A/C/S (A is the usual default) |
| Later | BloodMNIST, TissueMNIST | More bioimage variety once the pipeline is solid |
| Skip at first | ChestMNIST | Multi-label (14) — different metrics/training |
| Skip at first | PneumoniaMNIST | Binary and often too easy — weak ranking signal |
| Skip at first | RetinaMNIST | Ordinal / small; more awkward than plain multi-class |
| Skip at first | BreastMNIST | Tiny binary; easy to overfit and over-interpret |

Wire the arena on **DermaMNIST@224**, then add **PathMNIST** when you want a heavier medical benchmark.

### DermaMNIST@224 — download and extract

Official file: [`dermamnist_224.npz`](https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1) (~1.1 GB) from [Zenodo MedMNIST+ v3.0](https://zenodo.org/records/10519652). An `.npz` is a zip of NumPy arrays. Extract it to `dermamnist_224/` at the **repo root** (gitignored).

The folder should contain these six files:
| File | Shape | Role |
|------|-------|------|
| `train_images.npy` | `(7007, 224, 224, 3)` uint8 | Training images |
| `train_labels.npy` | `(7007, 1)` uint8 | Training labels (0–6) |
| `val_images.npy` | `(1003, 224, 224, 3)` uint8 | Validation images |
| `val_labels.npy` | `(1003, 1)` uint8 | Validation labels |
| `test_images.npy` | `(2005, 224, 224, 3)` uint8 | Test images |
| `test_labels.npy` | `(2005, 1)` uint8 | Test labels |

7 classes (HAM10000): 0 akiec, 1 bcc, 2 bkl, 3 df, 4 mel, 5 nv, 6 vasc. License: CC BY-NC 4.0.

### Dataset size — is DermaMNIST enough? What about &lt;1k / class?

**DermaMNIST (~10k images, 7 classes)** is a decent size for **ImageNet-pretrained** fine-tuning of ResNet50, ConvNeXt, and LoRA / linear-probe setups. Rough average is ~1.4k images/class, but HAM10000 is **imbalanced** — some lesion types have far fewer — so treat it as “medium small,” not huge. Full ViT fine-tuning can still overfit; prefer LoRA or a linear probe there.

For **&lt;1k per class** (or only hundreds of images total), DermaMNIST alone is **not** the best stress test. Gaps between full FT, LoRA, probes, and CLIP zero-shot show up more clearly on fine-grained sets that are small **by design**:

| Dataset | Scale | ~Images / class | Why it fits scarce-data comparisons |
|---------|-------|-----------------|-------------------------------------|
| **Oxford Flowers-102** | ~8k, 102 cls | **~40–80** overall; official train is often **10 / class** | Classic few-shot transfer; harder than Derma |
| **CUB-200-2011** | ~12k, 200 cls | **~60** | Standard small-data / fine-grained stress test |
| **Stanford Cars** | ~16k, 196 cls | **~80** | Catalog-style; still &lt;1k/class |
| **FGVC Aircraft** | ~10k, 100 cls | **~100** | Similar few-per-class regime |
| **BreastMNIST** / **RetinaMNIST** | Tiny medical | Very small | Domain shift, but binary/ordinal — weaker for architecture ranking |

**Practical arena split**

1. **DermaMNIST@224** — medium medical transfer (first medical run).
2. **Flowers-102 (official split)** or **CUB** — when you want tens of images per class and clearer gaps between full FT vs LoRA vs probe vs CLIP zero-shot.
3. Optional: **subsample** Derma/Path (e.g. 50–100 / class) for a controlled few-shot curve on the same medical domain.

## ResNet50 baseline — full fine-tune on DermaMNIST

First arena CNN baseline: ImageNet-pretrained **ResNet50**, full fine-tune on local `dermamnist_224/` (train 7007 / val 1003 / test 2005, 7 classes). Script: [`train_resnet50_derma.py`](train_resnet50_derma.py).

### Full FT vs freeze the backbone?

| Strategy | When | Verdict on DermaMNIST@224 |
|----------|------|---------------------------|
| **Full FT + differential LR** | Medium data, strong baseline | **Do this first** |
| Freeze backbone, train head only | Linear-probe baseline / tiny data | Secondary comparison only |
| Freeze then unfreeze | Extra care / unstable FT | Optional if full FT overfits |

With ~7k train images, a frozen ResNet usually **underperforms** full FT. Keep a frozen run later as a probe baseline for the arena, not as the primary ResNet result.

### Differential learning rates

Yes — use **different LRs** for backbone and head:

| Group | Default LR | Role |
|-------|------------|------|
| **Backbone** (all but `fc`) | `1e-4` | Gentle updates to pretrained features |
| **Head** (new `fc`) | `1e-3` (~10×) | Fast adapt of the classifier |

Optimizer: AdamW, weight decay `0.01`, cosine schedule. Train **all** layers. Inverse-frequency **class weights** are on by default (HAM10000 imbalance). Early stop on **val balanced accuracy**; also log accuracy and macro-F1.

### Usage

```bash
conda activate torch
python train_resnet50_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `dermamnist_224` | Folder with `{train,val,test}_{images,labels}.npy` |
| `--out` | `runs/resnet50_derma` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Differential FT |
| `--batch-size` | `32` | Try `16`–`64` on 16 GB GPUs |
| `--patience` | `10` | Early stop on val balanced acc |
| `--no-class-weights` | off | Plain CE if you want unweighted training |
| `--eval-test` | off | Run test metrics after training |

`runs/` and `*.pt` are gitignored. Prefer `best.pt` (best val balanced accuracy), not the last epoch.

### Baseline run analysis — `resnet50_derma`

Command: `--epochs 40 --batch-size 32 --eval-test`. Early stop at epoch **37**; best val balanced accuracy at epoch **27** (~42 min). Checkpoint: `runs/resnet50_derma/best.pt`.

| Split | Acc | Balanced acc | Macro-F1 |
|-------|----:|-------------:|---------:|
| Val (best) | — | **0.827** | — |
| Test | 0.881 | **0.792** | 0.794 |

(Val acc at the best-bacc epoch was ~0.88; train loss kept falling while val loss plateaued — mild overfit, early stop handled it.)

#### What do the class weights mean?

Inverse-frequency weights in CrossEntropyLoss so rare classes are not ignored:

```text
w_c = N / (C * n_c)
```

`N` = train size, `C` = 7, `n_c` = count of class `c`. Mean weight ≈ 1.

| Class | Weight | Meaning |
|-------|-------:|---------|
| **nv** | 0.21 | Very common (many nevi) → down-weighted |
| bkl, mel | ~1.3 | Near average |
| bcc, akiec | 2.8–4.4 | Less common → up-weighted |
| **vasc**, **df** | 10–12.5 | Rare → mistakes cost much more |

The model is pushed to care about rare lesions, not only dominant **nv**. That helps balanced accuracy / minority recall, sometimes at a small cost to overall accuracy.

#### What does macro-F1 mean?

For each class: precision, recall, then `F1 = 2 * P * R / (P + R)`. **Macro-F1** = **unweighted average** of the 7 per-class F1s — every class counts equally, so **nv** cannot hide weak df/vasc/mel.

| Metric | Meaning |
|--------|---------|
| **acc** | Overall % correct (skewed by majority class) |
| **bacc** | Mean per-class recall (class-equal) |
| **macro_f1** | Mean per-class F1 (class-equal; also penalizes false positives) |

#### Takeaways

- Solid first ResNet50 full-FT baseline on DermaMNIST@224.
- Prefer `best.pt` (epoch 27), not the last epoch.
- Test ~79% balanced / ~88% overall is a reasonable starting point; ~3–4 pt bacc drop val→test is normal.
- Checkpointing on **bacc** (not acc) is the right default under class imbalance.

## A. Modern CNN (ConvNeXt)

**Why include it:** ResNet (2015) is a classic baseline, but ConvNeXt modernized CNN architectures using design choices borrowed from Vision Transformers (7×7 depthwise convolutions, inverted bottlenecks, LayerNorm).

ConvNeXt comes in multiple capacity variants (same block design; different width/depth). This arena defaults to **ConvNeXt-Base**.

| Variant | Params | Disk size (`.pth`) |
|---------|--------|--------------------|
| **Tiny (T)** | ~29M | ~109 MB |
| **Small (S)** | ~50M | ~192 MB |
| **Base (B)** | ~89M | ~338 MB |
| **Large (L)** | ~198M | ~755 MB |
| **XLarge (XL)** | ~350M | ~1.3 GB |

T–L disk sizes are torchvision ImageNet-1K checkpoints; XL is estimated from ~4 bytes × params (not in torchvision).

**Small-data impact:** ConvNeXt often outperforms ViTs on small custom datasets because its strong inductive bias prevents overfitting better than a standard ViT.

**Full fine-tune on DermaMNIST:** Same protocol as ResNet50 — ImageNet-pretrained **ConvNeXt-Base**, differential LRs, class weights, early stop on val balanced accuracy. The classifier head is `model.classifier[2]` (not ResNet’s `fc`). Script: [`train_convnext_base_derma.py`](train_convnext_base_derma.py).

```bash
conda activate torch
python train_convnext_base_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/convnext_base_derma` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Head = `classifier.*` |
| `--batch-size` | `32` | Drop to `16` if GPU OOM |

### Run analysis — `convnext_base_derma`

Command: `--epochs 40 --batch-size 32 --eval-test`. Early stop at epoch **34**; best val balanced accuracy at epoch **24** (~26 min). Checkpoint: `runs/convnext_base_derma/best.pt`.

| Split | Acc | Balanced acc | Macro-F1 |
|-------|----:|-------------:|---------:|
| Val (best, epoch 24) | 0.894 | **0.861** | 0.827 |
| Test | 0.895 | **0.868** | 0.840 |

Train loss **1.28 → ~0.005**. Val CE loss bottomed ~epoch **6** (~0.52), then rose (~0.85 at the best-bacc epoch) — mild overfitting on the loss, while **bacc still climbed** until epoch 24. Val acc and bacc can disagree (e.g. epoch 19: acc 0.901, bacc 0.799); checkpointing on **bacc** avoided locking onto majority **nv**. Test bacc is slightly *above* val (0.868 vs 0.861); with a 1k val set that is noise, not a leak.

| Model | Params | Time | Best epoch | Val bacc | Test acc | Test bacc | Test macro-F1 |
|-------|-------:|-----:|-----------:|---------:|---------:|----------:|--------------:|
| ResNet50 | ~26M | ~42 min | 27 | 0.827 | 0.881 | 0.792 | 0.794 |
| **ConvNeXt-Base** | ~88M | ~26 min | 24 | **0.861** | **0.895** | **0.868** | **0.840** |

ConvNeXt-Base is the stronger full-FT CNN here: **+7.6 pt test bacc**, **+4.6 pt macro-F1**, **+1.4 pt acc**. Acc moves less because **nv** already dominates; the gain is mostly rarer lesion types. Prefer `best.pt` (epoch 24), not epoch 34. Later arena methods (LoRA, probes, CLIP) should beat **~87% test bacc / ~84% macro-F1**, not only overall accuracy.

**LoRA on ConvNeXt:** LoRA is often thought of as “Transformers only” because the original method targets dense / Linear weights (especially attention). It is **not** limited to Transformers: low-rank updates apply wherever there are suitable weight matrices. **1×1 convolutions** behave like linear layers over channels, so classic LoRA fits naturally; larger kernels may use Conv-LoRA-style factorizations. **ConvNeXt** is a modern CNN with Transformer-inspired blocks, so PEFT/LoRA is much more natural here than on older CNN designs — which is why this arena lists **Full / LoRA** for ConvNeXt alongside ViT/Swin.

Script: [`train_convnext_base_lora_derma.py`](train_convnext_base_lora_derma.py). Freezes the backbone; wraps the **Linear** layers in `features` (the inverted-bottleneck projections) with rank-`r` adapters; trains those adapters plus the full `classifier` head. Depthwise 7×7 convs stay frozen (no Conv-LoRA). No `peft` package — self-contained `LoRALinear`. Same data, class weights, and early-stop-on-bacc protocol as full FT.

```bash
conda activate torch
python train_convnext_base_lora_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/convnext_base_lora_derma` | Writes `best.pt` + `history.json` |
| `--lora-r` / `--lora-alpha` | `8` / `16` | Rank and scale (`ΔW` mixed at `α/r`) |
| `--lora-dropout` | `0.05` | Dropout on LoRA input |
| `--lr-lora` / `--lr-head` | `1e-3` / `1e-3` | Adapters + classifier; backbone frozen |

With `r=8` this is roughly **~1–2%** of ConvNeXt-Base parameters trainable. Compare against full-FT test **bacc 0.868 / macro-F1 0.840**.

### Run analysis — `convnext_base_lora_derma`

Command: `--epochs 40 --batch-size 32 --eval-test` (`r=8`, `α=16`, dropout `0.05`). Early stop at epoch **19**; best val balanced accuracy at epoch **9** (~9.8 min). Checkpoint: `runs/convnext_base_lora_derma/best.pt`. Trainable **1.45M / 89.0M (1.63%)**.

| Split | Acc | Balanced acc | Macro-F1 |
|-------|----:|-------------:|---------:|
| Val (best, epoch 9) | 0.854 | **0.855** | 0.791 |
| Test | 0.870 | **0.865** | 0.791 |

Train loss **1.21 → ~0.09** by early stop. Best epoch also had the **lowest val CE** (0.446) — unlike full FT, where val loss rose while bacc kept climbing. After epoch 9, val bacc bounced (0.79–0.85) without a new best; patience-10 stopped at 19. Test bacc again slightly above val (noise on a 1k val set).

| Method | Trainable | Time | Best epoch | Val bacc | Test acc | Test bacc | Test macro-F1 |
|--------|----------:|-----:|-----------:|---------:|---------:|----------:|--------------:|
| ResNet50 full FT | ~26M | ~42 min | 27 | 0.827 | 0.881 | 0.792 | 0.794 |
| ConvNeXt-Base full FT | ~88M | ~26 min | 24 | 0.861 | **0.895** | **0.868** | **0.840** |
| **ConvNeXt-Base LoRA** | **1.45M** | **~10 min** | **9** | 0.855 | 0.870 | 0.865 | 0.791 |

**Vs full FT:** LoRA matches full FT on the arena’s primary metric (**test bacc 0.865 vs 0.868**, −0.3 pt) with **~60× fewer trainable params** and **~2.6× less wall-clock**. Overall acc (−2.5 pt) and macro-F1 (−4.9 pt) lag — full FT still better when you care about precision on rare classes, not only per-class recall. Prefer `best.pt` (epoch 9). On this medium-sized medical set, LoRA is the better **accuracy-per-compute** default; full FT remains the ceiling for macro-F1.

## Self-supervised ViT — DINOv2 / DINOv3 (linear probe / MLP)

**Why include it:** DINOv2/DINOv3 are self-supervised ViTs trained to produce strong general visual features **without** ImageNet class labels. On small / domain-shifted data, a **frozen** DINO backbone + tiny head often beats full fine-tuning a supervised CNN, because you keep the pretrained representation instead of overfitting it.

**Protocol:** freeze the backbone → extract the **CLS** token → train only a head with CE (+ class weights), early stop on val balanced accuracy. Same DermaMNIST@224 splits and ImageNet mean/std as the CNN runs.

| Head | What it is | When to use |
|------|------------|-------------|
| **Linear probe** | One `Linear(D → C)` on CLS | **Start here** — standard DINO eval; tests linear separability of features |
| **MLP head** | Small MLP (e.g. Linear → GELU → Dropout → Linear) | Second run if linear plateaus; slightly more capacity, slightly more overfit risk |

Both keep DINO frozen. They differ only in head capacity — not in backbone training.

**DINOv2 linear probe (no license gate)** — script: [`train_dinov2_linear_derma.py`](train_dinov2_linear_derma.py). Official weights via PyTorch Hub [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2). Default: **`dinov2_vitb14`** (ViT-B/14, ~86M) so it matches ConvNeXt-Base / the planned DINOv3-B slot. DINOv2 uses **patch 14** (vs DINOv3’s 16); 224×224 is still the usual input. No Hugging Face login.

```bash
conda activate torch
python train_dinov2_linear_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--hub-model` | `dinov2_vitb14` | Also `dinov2_vits14`, `dinov2_vitl14`, `dinov2_vitg14` |
| `--out` | `runs/dinov2_linear_derma` | Writes `best.pt` + `history.json` |
| `--lr` | `1e-3` | Head only (backbone frozen) |

First run clones the Hub repo and downloads weights into `~/.cache/torch/hub/`. Same CE / class weights / early-stop-on-bacc protocol as ConvNeXt. Keep DINOv3 below for when Meta/HF access is approved.

**DINOv3 linear probe** — script: [`train_dinov3_linear_derma.py`](train_dinov3_linear_derma.py). Default backbone: [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) (ViT-B/16). Needs Hugging Face Transformers (not in the base `torch` env until you install it):

DINOv3 weights on Hugging Face are **gated**. A bare `from_pretrained` without login fails with `401` / `GatedRepoError`. One-time setup:

1. Create / log in at [huggingface.co](https://huggingface.co/)
2. Open [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) → **Agree** to the Meta license / access conditions
3. Create an access token: [Settings → Access Tokens](https://huggingface.co/settings/tokens) (read is enough)
4. Authenticate in the `torch` env:

```bash
conda activate torch
pip install "transformers>=4.56" huggingface_hub
hf auth login   # paste the token (huggingface-cli login is deprecated)
# or: export HF_TOKEN=hf_...
python train_dinov3_linear_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

If you already agreed on the website but still see `401`, you are not logged in locally — re-run `hf auth login`.

`403` / “not in the authorized list” means the token **is** valid, but **this HF account has not been granted the gated repo**. Login cannot skip that. Open the model page **while logged into the same account as the token**, click **Agree and access repository** (or submit the access form), wait until the page says you have access, then rerun the script. Approval is often instant; if Meta reviews it, wait until the repo lists you as authorized.

| Flag | Default | Notes |
|------|---------|-------|
| `--model-id` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Try `…/dinov3-vits16-pretrain-lvd1689m` for a smaller/faster backbone |
| `--out` | `runs/dinov3_linear_derma` | Writes `best.pt` + `history.json` |
| `--lr` | `1e-3` | Head only (backbone frozen) |
| `--image-size` | `224` | Matches HF processor default for these checkpoints |

**Why this checkpoint:** Meta’s official open DINOv3 release has **12** backbones ([MODEL_CARD](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)). We default to **ViT-B/16 @ LVD-1689M** (~86M) so it lines up with the arena’s **ConvNeXt-Base (~89M)** mid-size slot — strong enough for a fair SSL vs supervised CNN comparison, still easy to run frozen + linear probe. **LVD** is web/natural-image pretraining (right domain for DermaMNIST); **SAT-493M** is satellite and would be the wrong domain here.

| Family | Pretrain data | Open models |
|--------|---------------|-------------|
| **ViT** | LVD-1689M (web) | S/16 (21M), S+/16 (29M), **B/16 (86M)** ← default, L/16 (300M), H+/16 (840M), 7B/16 (~6.7B) |
| **ConvNeXt** | LVD-1689M | T (29M), S (50M), B (89M), L (198M) |
| **ViT** | SAT-493M (satellite) | L/16 (300M), 7B/16 (~6.7B) |

HF ids look like `facebook/dinov3-vitb16-pretrain-lvd1689m`, `…-vits16-…`, `…-convnext-base-…`, `…-vitl16-pretrain-sat493m`. For DermaMNIST stick to **LVD ViT** (or LVD ConvNeXt if you want a DINO-distilled CNN). Backbone weights cache under `~/.cache/huggingface/hub/` after the first download.

### Run analysis — `dinov3_linear_derma`

Command: `--epochs 40 --batch-size 32 --eval-test` (`facebook/dinov3-vitb16-pretrain-lvd1689m`). Early stop at epoch **35**; best val balanced accuracy at epoch **25** (~8.0 min). Checkpoint: `runs/dinov3_linear_derma/best.pt`. Trainable **5,383 / 85.7M (0.0063%)** — head only.

| Split | Acc | Balanced acc | Macro-F1 |
|-------|----:|-------------:|---------:|
| Val (best, epoch 25) | 0.771 | **0.782** | 0.677 |
| Test | 0.780 | **0.781** | 0.694 |

Train loss **1.31 → ~0.45**. Val CE fell steadily through ~epoch 19, then plateaued (~0.62). Val bacc climbed to **0.782** at epoch 25 with the usual noise under class imbalance (acc and bacc often disagree). Val and test bacc are essentially identical — the linear head did not overfit the 1k val set.

| Method | Trainable | Time | Best epoch | Val bacc | Test acc | Test bacc | Test macro-F1 |
|--------|----------:|-----:|-----------:|---------:|---------:|----------:|--------------:|
| ResNet50 full FT | ~26M | ~42 min | 27 | 0.827 | 0.881 | 0.792 | 0.794 |
| ConvNeXt-Base full FT | ~88M | ~26 min | 24 | 0.861 | **0.895** | **0.868** | **0.840** |
| ConvNeXt-Base LoRA | 1.45M | ~10 min | 9 | 0.855 | 0.870 | 0.865 | 0.791 |
| DINOv3-B linear probe | 5.4k | ~8 min | 25 | 0.782 | 0.780 | 0.781 | 0.694 |
| **DINOv3-B MLP head** | **397k** | **~9 min** | **32** | **0.820** | 0.858 | **0.805** | **0.779** |

**Takeaways:** Frozen DINOv3 + linear is a clean, cheap SSL baseline, but on DermaMNIST it **lags ConvNeXt full FT / LoRA by ~8–9 pt test bacc** and ~10–15 pt macro-F1. That is expected: a single linear layer cannot adapt web/LVD features to dermatoscopy the way full/LoRA FT can. Prefer `best.pt` (epoch 25).

**DINOv3 MLP head** — script: [`train_dinov3_mlp_derma.py`](train_dinov3_mlp_derma.py). Same frozen backbone and data protocol; head is `Linear(D→H) → GELU → Dropout → Linear(H→C)` (default `H=512`, dropout `0.2`). Uses the cached HF weights (no re-download if linear already ran). Compare against linear test **bacc 0.781**.

```bash
conda activate torch
python train_dinov3_mlp_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/dinov3_mlp_derma` | Writes `best.pt` + `history.json` |
| `--mlp-hidden` | `512` | Hidden width of the MLP |
| `--dropout` | `0.2` | After GELU |
| `--lr` | `1e-3` | Head only |

### Run analysis — `dinov3_mlp_derma`

Command: `--epochs 40 --batch-size 32 --eval-test` (`mlp_hidden=512`, `dropout=0.2`). Ran all **40** epochs (no early stop); best val balanced accuracy at epoch **32** (~9.1 min). Checkpoint: `runs/dinov3_mlp_derma/best.pt`. Trainable **397k / 86.1M (0.46%)**.

| Split | Acc | Balanced acc | Macro-F1 |
|-------|----:|-------------:|---------:|
| Val (best, epoch 32) | 0.855 | **0.820** | 0.784 |
| Test | 0.858 | **0.805** | 0.779 |

Train loss **1.18 → ~0.12**. Val CE bottomed mid-run (~0.62) then rose while train loss kept falling — mild head overfit; bacc still peaked at epoch 32. Val→test bacc drop is small (**−1.5 pt**).

**Vs linear probe (same backbone):** MLP gains **+2.4 pt test bacc** (0.805 vs 0.781), **+7.8 pt test acc**, and **+8.5 pt macro-F1** (0.779 vs 0.694). Nonlinearity helps rare-class precision/recall, not only overall accuracy. Prefer `best.pt` (epoch 32), not epoch 40.

**Vs ConvNeXt:** Still **~6 pt behind** LoRA/full FT on test bacc (0.805 vs ~0.87). Frozen DINOv3 + MLP is a stronger SSL head than linear, but does not replace domain adaptation of the backbone. Next: DINOv2 linear/MLP, or light DINO LoRA if you want to close the CNN gap.

## Supervised ViT — ViT-B/16 (full fine-tune)

**Why include it:** ImageNet-supervised **ViT-B/16** (~87M) is the classic Transformer classification baseline. On small / imbalanced data it often **overfits more** than ConvNeXt (weaker inductive bias), which is why the arena table lists **Full vs LoRA** — full FT first, then LoRA as the PEFT comparison.

**Full fine-tune on DermaMNIST:** Same protocol as ConvNeXt — torchvision `vit_b_16` + `IMAGENET1K_V1`, differential LRs, class weights, early stop on val balanced accuracy. Classifier head is `model.heads.head` (not ResNet `fc` / ConvNeXt `classifier[2]`). Script: [`train_vit_b16_derma.py`](train_vit_b16_derma.py).

```bash
conda activate torch
python train_vit_b16_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/vit_b16_derma` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Head = `heads.*` |
| `--batch-size` | `32` | Drop to `16` if GPU OOM |

### Run analysis — `vit_b16_derma`

Command: `--epochs 40 --batch-size 32 --eval-test`. Ran all **40** epochs (no early stop — val bacc kept edging up late); best val balanced accuracy at epoch **40** (~21.9 min). Checkpoint: `runs/vit_b16_derma/best.pt`.

| Split | Acc | Balanced acc | Macro-F1 |
|-------|----:|-------------:|---------:|
| Val (best, epoch 40) | 0.867 | **0.823** | 0.804 |
| Test | 0.846 | **0.798** | 0.797 |

Train loss **1.62 → ~0.016**. Val CE bottomed ~epoch **10** (0.617), then climbed to **~1.27** by epoch 40 while train loss collapsed — **clear overfit**. Val bacc still improved slowly after epoch 23 (0.805 → 0.823), so checkpointing on **bacc** (not val loss) selected a late epoch; test bacc is **−2.5 pt** vs that val peak.

| Method | Trainable | Time | Best epoch | Val bacc | Test acc | Test bacc | Test macro-F1 |
|--------|----------:|-----:|-----------:|---------:|---------:|----------:|--------------:|
| ConvNeXt-Base full FT | ~88M | ~26 min | 24 | 0.861 | **0.895** | **0.868** | **0.840** |
| ConvNeXt-Base LoRA | 1.45M | ~10 min | 9 | 0.855 | 0.870 | 0.865 | 0.791 |
| **ViT-B/16 LoRA (MLP)** | **0.74M** | **~16 min** | **21** | 0.838 | 0.881 | **0.859** | **0.849** |
| DINOv3-B MLP (frozen) | 397k | ~9 min | 32 | 0.820 | 0.858 | 0.805 | 0.779 |
| ViT-B/16 full FT | ~86M | ~22 min | 40 | 0.823 | 0.846 | 0.798 | 0.797 |

**Takeaways:** Supervised ViT-B/16 full FT **underperforms ConvNeXt** on this set (**−7.0 pt test bacc** vs ConvNeXt full FT). Test bacc was roughly on par with frozen DINOv3 + MLP (0.798 vs 0.805) — plain ViT full FT overfits easily. Prefer `best.pt`. **LoRA** (below) largely closes the gap.

**LoRA fine-tune** — script: [`train_vit_b16_lora_derma.py`](train_vit_b16_lora_derma.py). Freezes ViT-B/16; wraps encoder **MLP** `nn.Linear` layers with rank-`r` adapters; trains those + the full `heads` classifier. Attention stays frozen: torchvision fuses QKV into `in_proj_weight`, and `MultiheadAttention` reads `out_proj.weight` via the fused functional path (wrapping `out_proj` raises `AttributeError`). Same `r=8` / `α=16` defaults as ConvNeXt LoRA. No `peft` package.

```bash
conda activate torch
python train_vit_b16_lora_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/vit_b16_lora_derma` | Writes `best.pt` + `history.json` |
| `--lora-r` / `--lora-alpha` | `8` / `16` | Rank and scale |
| `--lora-dropout` | `0.05` | Dropout on LoRA input |
| `--lr-lora` / `--lr-head` | `1e-3` / `1e-3` | Adapters + classifier |

### Run analysis — `vit_b16_lora_derma`

Command: `--epochs 40 --batch-size 32 --eval-test` (`r=8`, `α=16`, MLP-only LoRA). Early stop at epoch **31**; best val balanced accuracy at epoch **21** (~16.4 min). Checkpoint: `runs/vit_b16_lora_derma/best.pt`. Trainable **0.74M / 86.5M (0.86%)**.

| Split | Acc | Balanced acc | Macro-F1 |
|-------|----:|-------------:|---------:|
| Val (best, epoch 21) | 0.861 | **0.838** | 0.821 |
| Test | 0.881 | **0.859** | 0.849 |

Train loss **1.18 → ~0.01**. Val CE bottomed ~epoch **8** (0.529), then rose while train loss kept falling — milder overfit than full FT (val CE stayed ~0.7–1.0, not the ~1.27 collapse of full FT). Test bacc is **above** val (+2.1 pt); with a 1k val set that is noise, not a leak.

**Vs ViT full FT:** LoRA wins clearly — **+6.1 pt test bacc** (0.859 vs 0.798), **+3.5 pt acc**, **+5.2 pt macro-F1**, with **~116× fewer** trainable params and less CE overfit. Prefer `best.pt` (epoch 21).

**Vs ConvNeXt:** Within **~1 pt test bacc** of ConvNeXt LoRA (0.859 vs 0.865) and close to ConvNeXt full FT (0.868). On DermaMNIST, **ViT + LoRA** is the right supervised-Transformer recipe; full FT is the weak baseline the arena predicted.

## B. Zero-shot & linear probe CLIP (multimodal models)

**Why include it:** Models like CLIP (OpenAI / OpenCLIP) or EVA-CLIP are pre-trained on billions of image–text pairs.

**Small-data impact:** Zero-shot CLIP requires **zero** training images, while a CLIP image encoder paired with a **linear probe** often sets a strong benchmark for small datasets with minimal compute.

## C. Parameter-efficient fine-tuning (PEFT / LoRA for vision)

**Why include it:** Full fine-tuning of ViTs on small datasets often leads to severe overfitting.

**Small-data impact:** Applying LoRA (Low-Rank Adaptation) to frozen ViT/Swin (and similarly **ConvNeXt**) backbones fine-tunes only ~1% of parameters, usually yielding higher accuracy and faster training on small sample sizes. See also [LoRA on ConvNeXt](#a-modern-cnn-convnext).

## D. Hierarchical Vision Transformers (e.g. Swin Transformer)

**Why include it:** Standard ViT uses non-overlapping square patches at a single scale. Swin Transformer uses shifted windows and hierarchical representations.

**Small-data impact:** Swin behaves more like a hybrid between a CNN and a Transformer, making it significantly more data-efficient on small custom datasets.
