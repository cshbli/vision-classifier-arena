# Vision Classifier Arena

A benchmark suite comparing image classification performance across small, custom, and domain-specific datasets. Evaluates supervised CNNs (ResNet, ConvNeXt), self-supervised backbones (DINOv3 + MLP), and pre-trained Vision Transformers (ViT, Swin) under frozen feature probing, PEFT (LoRA), and full fine-tuning.

| Architecture Family | Specific Model | Params | Pretrain data | Label size | Training Strategy | Strengths on Small Data |
|---------------------|----------------|-------:|---------------|------------|-------------------|-------------------------|
| Classic CNN | ResNet50 | 25.6M | ImageNet-1K (~1.28M images) | 1,000 classes | Full Fine-Tuning | Strong baseline, fast convergence. |
| Modern CNN | ConvNeXt-Base | 88.6M | ImageNet-1K (~1.28M images) | 1,000 classes | Full / LoRA Fine-Tuning | High accuracy; inductive bias resists overfitting vs standard ViT. |
| Self-Supervised ViT | DINOv3 ViT-B/16 | 85.7M | LVD-1689M (~1.69B images) | — (no class labels; SSL) | Linear Probe / MLP Head | No backbone tuning; preserves global features. |
| Supervised ViT | ViT-B/16 | 86.6M | ImageNet-1K (~1.28M images) | 1,000 classes | Full / LoRA Fine-Tuning | High ceiling; full FT overfits easily — LoRA helps a lot. |
| Hierarchical ViT | Swin-B | 87.8M | ImageNet-1K (~1.28M images) | 1,000 classes | Full / LoRA Fine-Tuning | CNN–Transformer hybrid; more data-efficient than plain ViT. |
| Multimodal | OpenCLIP ViT-B/16 (DataComp-XL) | 86.2M vision / 150M with text | DataComp-1B (~1.4B image–text pairs; ~13B samples seen) | — (captions, not a closed label set) | **Zero-shot** / Linear Probe | Zero training images (zero-shot); linear probe is a strong small-data SOTA with low compute. |

## Datasets

Open, compact classification sets for small / fine-grained / domain-specific evaluation (no large-scale corpora).

| Dataset | Domain | Approx. scale | Why it fits | Download |
|---------|--------|---------------|-------------|----------|
| **Oxford-IIIT Pets** | Fine-grained animals | ~7k images, 37 classes | Small, clean, standard transfer baseline | [Project page](https://www.robots.ox.ac.uk/~vgg/data/pets/) · [images](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz) · [annotations](https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz) |
| **Oxford Flowers-102** | Fine-grained | ~8k, 102 classes | Harder fine-grained; few shots per class | [Project page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) · [images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) · [labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) · [splits](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat) |
| **DermaMNIST@224** | Medical (dermatoscopy) | ~10k images, 7 classes | Domain shift; imbalanced; current arena run | [medmnist.com](https://medmnist.com/) · [Zenodo MedMNIST+](https://zenodo.org/records/10519652) · [`dermamnist_224.npz`](https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1) |

Use **official splits** when available. Match each backbone’s expected preprocessing (ImageNet norms vs CLIP’s own).

### Oxford Flowers-102 — download and extract

Official files from the [Oxford Flowers-102 page](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/):

| File | Role |
|------|------|
| [`102flowers.tgz`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) | Images archive (~330 MB) |
| [`imagelabels.mat`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) | Per-image class labels (1–102) |
| [`setid.mat`](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat) | Official train / val / test image ids |

Extract the archive and place the labels/splits beside it under `102flowers/` at the **repo root** (gitignored):

```bash
cd /path/to/vision-classifier-arena
mkdir -p 102flowers
# download the three files into 102flowers/, then:
tar -xzf 102flowers/102flowers.tgz -C 102flowers
# tgz expands to 102flowers/jpg/…; keep imagelabels.mat and setid.mat in 102flowers/
```

The folder should look like:

| Path | Role |
|------|------|
| `jpg/image_00001.jpg` … `jpg/image_08189.jpg` | **8189** RGB images (variable resolution) |
| `imagelabels.mat` | `labels` array length 8189; class ids **1–102** |
| `setid.mat` | `trnid`, `valid`, `tstid` — **1-indexed** image ids for each split |

Official split sizes (10 train + 10 val per class; remainder test):

| Split | Count | Notes |
|-------|------:|-------|
| Train (`trnid`) | 1020 | 10 images / class |
| Val (`valid`) | 1020 | 10 images / class |
| Test (`tstid`) | 6149 | Rest of each class |

Image `image_XXXXX.jpg` corresponds to label index `XXXXX` (1-based) in `imagelabels.mat` and to ids in `setid.mat`. Convert labels to **0–101** for PyTorch. Resize/crop to **224** at train time (same as DermaMNIST). Fine-grained few-shot stress test: ~10 labeled examples per class in the official train split.

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

#### Class Imbalance and Class Weights

DermaMNIST is **imbalanced** (especially **nv**). Arena scripts use inverse-frequency weights in CrossEntropyLoss so rare classes are not ignored:

$$
w_c = \frac{N}{C \cdot n_c}
$$

where $N$ = train size, $C$ = 7 classes, and $n_c$ = count of class $c$. Mean weight $\approx 1$.

| Class | Weight | Meaning |
|-------|-------:|---------|
| **nv** | 0.21 | Very common (many nevi) → down-weighted |
| bkl, mel | ~1.3 | Near average |
| bcc, akiec | 2.8–4.4 | Less common → up-weighted |
| **vasc**, **df** | 10–12.5 | Rare → mistakes cost much more |

The model is pushed to care about rare lesions, not only dominant **nv**. That helps balanced accuracy / minority recall, sometimes at a small cost to overall accuracy.

## Training Strategy

### Full FT vs freeze the backbone?

| Strategy | When | Verdict on DermaMNIST@224 |
|----------|------|---------------------------|
| **Full FT + differential LR** | Medium data, strong baseline | **Do this first** |
| Freeze backbone, train head only | Linear-probe baseline / tiny data | Secondary comparison only |
| Freeze then unfreeze | Extra care / unstable FT | Optional if full FT overfits |

With ~7k train images, a frozen ResNet usually **underperforms** full FT. Keep a frozen run later as a probe baseline for the arena, not as the primary ResNet result.

### Differential learning rates

Use **different LRs** for backbone and head:

| Group | Default LR | Role |
|-------|------------|------|
| **Backbone** (all but `fc`) | `1e-4` | Gentle updates to pretrained features |
| **Head** (new `fc`) | `1e-3` (~10×) | Fast adapt of the classifier |

Optimizer: AdamW, weight decay `0.01`, cosine schedule. Train **all** layers. Inverse-frequency **class weights** are on by default (HAM10000 imbalance). Early stop on **val balanced accuracy**; also log accuracy and macro-F1.

## Benchmarking Criteria

For each class $c$, with precision $P_c$ and recall $R_c$:

$$
\mathrm{F1}_c = \frac{2 P_c R_c}{P_c + R_c}, \qquad
\mathrm{macro\text{-}F1} = \frac{1}{C}\sum_{c=1}^{C} \mathrm{F1}_c
$$

Every class counts equally, so **nv** cannot hide weak df/vasc/mel.

| Metric | Meaning |
|--------|---------|
| **acc** | Overall % correct (skewed by majority class) |
| **bacc** | Mean per-class recall (class-equal) |
| **macro_f1** | Mean per-class F1 (class-equal; also penalizes false positives) |

## Benchmark results on DermaMNIST@224

Primary metric: **test balanced accuracy**. Full per-method write-ups are in the sections below.

| Method | Trainable | Val bacc | Test acc | Test bacc | Test macro-F1 |
|--------|----------:|---------:|---------:|----------:|--------------:|
| ResNet50 | ~26M | 0.827 | **0.881** | 0.792 | 0.794 |
| ConvNeXt Base full FT | ~88M | 0.861 | **0.895** | **0.868** | **0.840** |
| ConvNeXt Base LoRA | 1.45M |  0.855 | **0.870** | **0.865** | 0.791 |
| DINOv3 linear | 5.4k | 0.782 | 0.780 | 0.781 | 0.694 |
| DINOv3 MLP | 397k | 0.820 | **0.858** | 0.805 | 0.779 |
| ViT-B/16 full FT | ~86M | 0.823 | **0.846** | 0.798 | 0.797 |
| ViT-B/16 LoRA | 0.74M |0.838 | **0.881** | **0.859** | **0.849** |
| Swin-B full FT | ~87M | 0.845 | **0.830** | **0.859** | 0.776 |
| Swin-B LoRA | 1.01M | 0.863 | **0.872** | **0.872** | **0.835** |
| OpenCLIP B/16 zero-shot | 0 | 0.311 | 0.255 | 0.350 | 0.198 |
| OpenCLIP B/16 linear | 3.6k | 0.741 | 0.741 | 0.723 | 0.595 |

## Benchmark results on Oxford Flowers-102

Primary metric: **test balanced accuracy**. Full per-method write-ups are in the sections below.

| Method | Trainable | Val bacc | Test acc | Test bacc | Test macro-F1 |
|--------|----------:|---------:|---------:|----------:|--------------:|
| ResNet50 | ~24M | 0.925 | 0.892 | 0.910 | 0.888 |
| ConvNeXt-Base full FT | ~88M | 0.979 | 0.968 | 0.972 | 0.966 |
| ConvNeXt-Base LoRA | 1.55M | 0.977 | 0.965 | 0.971 | 0.964 |
| DINOv3 linear | 78k | **0.995** | **0.997** | **0.997** | **0.997** |
| DINOv3 MLP | 446k | 0.995 | 0.996 | 0.997 | 0.996 |
| ViT-B/16 full FT | ~86M | 0.950 | 0.932 | 0.946 | 0.931 |
| ViT-B/16 LoRA | 0.82M | 0.932 | 0.917 | 0.933 | 0.917 |
| Swin-B full FT | ~87M | 0.955 | 0.942 | 0.952 | 0.939 |
| Swin-B LoRA | 1.11M | 0.957 | 0.943 | 0.952 | 0.941 |

## ResNet50 — full fine-tune on DermaMNIST

ImageNet-pretrained **ResNet50**, full fine-tune on local `dermamnist_224/` (train 7007 / val 1003 / test 2005, 7 classes). Script: [`train_resnet50_derma.py`](train_resnet50_derma.py).

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

### Run analysis — `resnet50_derma`

Command: `--epochs 40 --batch-size 32 --eval-test`. Early stop at epoch **37**; best val balanced accuracy at epoch **27** (~42 min). Checkpoint: `runs/resnet50_derma/best.pt`.

- Test ~79% balanced / ~88% overall is a reasonable starting point; ~3–4 pt bacc drop val→test is normal.
- Checkpointing on **bacc** (not acc) is the right default under class imbalance.

## ResNet50 - Full fine-tune on Oxford Flowers-102

Fine-grained few-shot check: ImageNet-pretrained **ResNet50**, full fine-tune on local `102flowers/` (official split **1020 / 1020 / 6149**, **102** classes, 10 train images per class). Script: [`train_resnet50_flowers.py`](train_resnet50_flowers.py). Loads `imagelabels.mat` / `setid.mat` without scipy; labels converted to **0–101**.

```bash
conda activate torch
python train_resnet50_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/resnet50_flowers` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Differential FT |
| `--batch-size` | `32` | Drop to `16` if GPU OOM |
| `--patience` | `10` | Early stop on val balanced acc |
| `--eval-test` | off | Score the large official test split after training |

Official train is **balanced** (10/class), so class weights ≈ 1; kept on for protocol parity with Derma. Expect higher variance than Derma — only 10 shots per class.

### Run analysis — `resnet50_flowers`

Early stop at epoch **30**; best val balanced accuracy at epoch **20** (~1.7 min). Checkpoint: `runs/resnet50_flowers/best.pt`. Model **23.7M** (ResNet50 with 102-way head).

- With only **10 images/class**, ImageNet ResNet50 still reaches **~91% test bacc** on Flowers — natural-image fine-grained transfer is much easier than Derma’s dermatoscopy domain shift (Derma test bacc **0.792**). 
- Training is fast (~2 min) because the train set is tiny (1020). Acc/bacc/macro-F1 stay close (balanced train/val). 

## ConvNeXt - Full fine-tune on Oxford Flowers-102

Same protocol as ResNet50 Flowers / ConvNeXt Derma — ImageNet-pretrained **ConvNeXt-Base**, differential LRs, class weights, early stop on val balanced accuracy. Head is `classifier[2]`. Script: [`train_convnext_base_flowers.py`](train_convnext_base_flowers.py). Reuses the Flowers loader from [`train_resnet50_flowers.py`](train_resnet50_flowers.py).

```bash
conda activate torch
python train_convnext_base_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/convnext_base_flowers` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Head = `classifier.*` |
| `--batch-size` | `32` | Drop to `16` if GPU OOM |
| `--eval-test` | off | Score the large official test split after training |

Compare against ResNet50 Flowers test **bacc 0.910**.

### Run analysis — `convnext_base_flowers`

Early stop at epoch **35**; best val balanced accuracy at epoch **25** (~5.0 min). Checkpoint: `runs/convnext_base_flowers/best.pt`. Model **87.7M**.

**Vs ResNet50 Flowers:** ConvNeXt wins clearly — **+6.2 pt test bacc** (0.972 vs 0.910), **+7.6 pt acc**, **+7.8 pt macro-F1**. Same 10-shot/class protocol; the modern CNN inductive bias helps a lot on this fine-grained natural-image set.

## ConvNeXt - LoRA on Oxford Flowers-102

Same LoRA recipe as Derma — freeze ConvNeXt-Base; wrap **Linear** layers in `features` with rank-`r` adapters; train adapters + full `classifier`. Depthwise 7×7 convs stay frozen. Script: [`train_convnext_base_lora_flowers.py`](train_convnext_base_lora_flowers.py). Defaults `r=8` / `α=16`. No `peft` package.

```bash
conda activate torch
python train_convnext_base_lora_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/convnext_base_lora_flowers` | Writes `best.pt` + `history.json` |
| `--lora-r` / `--lora-alpha` | `8` / `16` | Rank and scale |
| `--lora-dropout` | `0.05` | Dropout on LoRA input |
| `--lr-lora` / `--lr-head` | `1e-3` / `1e-3` | Adapters + classifier |
| `--eval-test` | off | Score the large official test split after training |

Compare against ConvNeXt Flowers full FT test **bacc 0.972**.

### Run analysis — `convnext_base_lora_flowers`

Early stop at epoch **30**; best val balanced accuracy at epoch **20** (~3.4 min). Checkpoint: `runs/convnext_base_lora_flowers/best.pt`. Trainable **1.55M / 89.1M (1.74%)**.

**Vs ConvNeXt full FT:** Essentially tied on the primary metric — test bacc **0.971 vs 0.972** (−0.1 pt) with **~57× fewer** trainable params and less wall-clock (~3.4 vs ~5 min). Acc/macro-F1 also within ~0.5 pt. On Flowers, LoRA is the better **accuracy-per-compute** default; full FT remains the slight ceiling.

**Vs ResNet50 full FT:** Still **+6.1 pt test bacc** (0.971 vs 0.910) despite training far fewer params than ResNet full FT.

## DINOv3 — linear probe on Oxford Flowers-102

Same frozen-backbone protocol as Derma — **DINOv3 ViT-B/16** ([`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)), train only `Linear(CLS → 102)`. ImageNet mean/std, inverse-freq CE, early stop on val balanced accuracy. Script: [`train_dinov3_linear_flowers.py`](train_dinov3_linear_flowers.py). Reuses the Flowers loader from [`train_resnet50_flowers.py`](train_resnet50_flowers.py) and the model wrapper from [`train_dinov3_linear_derma.py`](train_dinov3_linear_derma.py).

Weights are **gated** on Hugging Face — one-time login + license accept (see [Self-supervised ViT — DINOv3](#self-supervised-vit--dinov3-linear-probe--mlp) below). Needs `transformers>=4.56`.

```bash
conda activate torch
python train_dinov3_linear_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/dinov3_linear_flowers` | Writes `best.pt` + `history.json` |
| `--model-id` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Try `…/dinov3-vits16-…` for a smaller/faster backbone |
| `--lr` | `1e-3` | Head only (backbone frozen) |
| `--eval-test` | off | Score the large official test split after training |

Compare against ConvNeXt Flowers full FT / LoRA test **bacc ~0.97** and ResNet50 **0.910**.

### Run analysis — `dinov3_linear_flowers`

Early stop at epoch **20**; best val balanced accuracy at epoch **10** (~1.3 min). Checkpoint: `runs/dinov3_linear_flowers/best.pt`. Trainable **78,438 / 85.7M (0.0915%)** — head only.

- Val bacc **0.967** after epoch 1 and **0.993** by epoch 2 — Flowers features are nearly linearly separable under LVD DINOv3.
- Test **bacc / acc 0.997**, macro-F1 **0.997** on the large 6149-image split — essentially saturated.

**Vs ConvNeXt full FT / LoRA:** Clear win — **+2.5 pt test bacc** (0.997 vs 0.972 / 0.971) with **~1,100× fewer** trainable params than full FT and ~4× less wall-clock than LoRA. On in-domain natural images, frozen SSL beats supervised CNN fine-tuning.

**Vs Derma DINOv3 linear:** Opposite story — Derma test bacc **0.781** (lags ConvNeXt); Flowers **0.997** (leads). Domain match (web/LVD ↔ photos) vs dermatoscopy shift is the difference, not the linear-probe recipe.

## DINOv3 — MLP head on Oxford Flowers-102

Same frozen backbone as Flowers linear probe; head is `Linear(D→H) → GELU → Dropout → Linear(H→102)` (default `H=512`, dropout `0.2`). Script: [`train_dinov3_mlp_flowers.py`](train_dinov3_mlp_flowers.py). Reuses the Flowers loader from [`train_resnet50_flowers.py`](train_resnet50_flowers.py) and the model wrapper from [`train_dinov3_mlp_derma.py`](train_dinov3_mlp_derma.py).

HF gated access + `transformers>=4.56` — same setup as [DINOv3 linear Flowers](#dinov3--linear-probe-on-oxford-flowers-102).

```bash
conda activate torch
python train_dinov3_mlp_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/dinov3_mlp_flowers` | Writes `best.pt` + `history.json` |
| `--model-id` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Same backbone as linear probe |
| `--mlp-hidden` / `--dropout` | `512` / `0.2` | MLP width and dropout after GELU |
| `--lr` | `1e-3` | Head only (backbone frozen) |
| `--eval-test` | off | Score the large official test split after training |

Compare against Flowers linear probe test **bacc 0.997**.

### Run analysis — `dinov3_mlp_flowers`

Early stop at epoch **13**; best val balanced accuracy at epoch **3** (~0.9 min). Checkpoint: `runs/dinov3_mlp_flowers/best.pt`. Trainable **446k / 86.1M (0.52%)** (`mlp_hidden=512`, `dropout=0.2`).

- Val bacc **0.991** after epoch 1; best val **0.995** by epoch 3 — same ceiling as linear.
- Test **bacc 0.997** (0.9965), acc **0.996**, macro-F1 **0.996** — within noise of linear.

**Vs linear probe (same backbone):** No gain — test bacc **0.9965 vs 0.9971** (−0.06 pt), macro-F1 slightly lower, with **~5.7× more** trainable params. Prefer linear on Flowers; the extra nonlinearity only helped on Derma where features were not already linearly separable.

**Vs ConvNeXt:** Still **+2.5 pt test bacc** vs full FT / LoRA, same story as linear — frozen LVD DINO dominates supervised CNN FT on this set.

## ViT-B/16 — full fine-tune on Oxford Flowers-102

Same protocol as ViT Derma / ConvNeXt Flowers — ImageNet-pretrained **ViT-B/16** (torchvision `vit_b_16` + `IMAGENET1K_V1`), differential LRs, class weights, early stop on val balanced accuracy. Head is `heads.head`. Script: [`train_vit_b16_flowers.py`](train_vit_b16_flowers.py). Reuses the Flowers loader from [`train_resnet50_flowers.py`](train_resnet50_flowers.py) and `build_model` / `param_groups` from [`train_vit_b16_derma.py`](train_vit_b16_derma.py).

```bash
conda activate torch
python train_vit_b16_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/vit_b16_flowers` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Differential FT; head = `heads.*` |
| `--eval-test` | off | Score the large official test split after training |

Compare against ConvNeXt Flowers full FT test **bacc 0.972** and DINOv3 linear **0.997**.

### Run analysis — `vit_b16_flowers`

Early stop at epoch **31**; best val balanced accuracy at epoch **21** (~3.4 min). Checkpoint: `runs/vit_b16_flowers/best.pt`. Model **85.9M**.

- Train loss collapses by epoch 4 (~0.04) while val bacc plateaus ~0.94–0.95 — classic ViT overfit on 10-shot/class.
- Test **bacc 0.946**, acc **0.932**, macro-F1 **0.931**; val→test drop ~0.4 pt bacc (mild).

**Vs ConvNeXt full FT:** Trails by **−2.6 pt test bacc** (0.946 vs 0.972) and ~3–4 pt acc/F1 — same lesson as Derma: supervised ViT full FT loses to ConvNeXt at matched capacity.

**Vs ResNet50:** **+3.6 pt test bacc** (0.946 vs 0.910) — beats the smaller CNN, still not competitive with ConvNeXt or frozen DINO.

**Vs DINOv3 linear:** Far behind — **−5.1 pt test bacc** (0.946 vs 0.997) despite training **~1,100× more** params. On Flowers, keep the SSL backbone frozen; full FT of ImageNet ViT is the wrong recipe.

## ViT-B/16 — LoRA on Oxford Flowers-102

Same LoRA recipe as Derma — freeze ViT-B/16; wrap encoder **MLP** `nn.Linear` layers with rank-`r` adapters; train adapters + full `heads` classifier. Attention stays frozen (torchvision fused QKV / functional `out_proj`). Defaults `r=8` / `α=16`. No `peft` package. Script: [`train_vit_b16_lora_flowers.py`](train_vit_b16_lora_flowers.py). Reuses the Flowers loader from [`train_resnet50_flowers.py`](train_resnet50_flowers.py) and LoRA helpers from [`train_vit_b16_lora_derma.py`](train_vit_b16_lora_derma.py).

```bash
conda activate torch
python train_vit_b16_lora_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/vit_b16_lora_flowers` | Writes `best.pt` + `history.json` |
| `--lora-r` / `--lora-alpha` | `8` / `16` | Rank and scale |
| `--lora-dropout` | `0.05` | Dropout on LoRA input |
| `--lr-lora` / `--lr-head` | `1e-3` / `1e-3` | Adapters + classifier |
| `--eval-test` | off | Score the large official test split after training |

Compare against ViT Flowers full FT test **bacc 0.946** and ConvNeXt LoRA **0.971**.

### Run analysis — `vit_b16_lora_flowers`

Early stop at epoch **21**; best val balanced accuracy at epoch **11** (~2.3 min). Checkpoint: `runs/vit_b16_lora_flowers/best.pt`. Trainable **0.82M / 86.6M (0.94%)** (24 MLP Linears wrapped; attention frozen).

- Val peaks early at **0.932** and does not climb toward full-FT’s **0.950**.
- Test **bacc 0.933**, acc **0.917**, macro-F1 **0.917**.

**Vs ViT full FT:** LoRA **loses** — **−1.3 pt test bacc** (0.933 vs 0.946), **−1.5 pt acc**, **−1.4 pt macro-F1**, despite **~105× fewer** trainable params and less wall-clock. Opposite of Derma (where LoRA was **+6.1 pt**). On in-domain Flowers, full backbone FT still helps ImageNet ViT; MLP-only adapters under-adapt.

**Vs ConvNeXt LoRA:** Trails by **−3.8 pt test bacc** (0.933 vs 0.971) at similar PEFT budget — ConvNeXt remains the better supervised LoRA on this set.

**Vs DINOv3 linear:** **−6.4 pt test bacc** with ~10× more trainable params — frozen SSL still dominates.

## Swin-B — full fine-tune on Oxford Flowers-102

Same protocol as Swin Derma / ViT Flowers — ImageNet-pretrained **Swin-B** (torchvision `swin_b` + `IMAGENET1K_V1`), differential LRs, class weights, early stop on val balanced accuracy. Head is `model.head` (`Linear(1024 → 102)`). Script: [`train_swin_b_flowers.py`](train_swin_b_flowers.py). Reuses the Flowers loader from [`train_resnet50_flowers.py`](train_resnet50_flowers.py) and `build_model` / `param_groups` from [`train_swin_b_derma.py`](train_swin_b_derma.py).

```bash
conda activate torch
python train_swin_b_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/swin_b_flowers` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Differential FT; head = `head.*` |
| `--eval-test` | off | Score the large official test split after training |

### Run analysis — `swin_b_flowers`

Early stop at epoch **27**; best val balanced accuracy at epoch **17** (~3.1 min). Checkpoint: `runs/swin_b_flowers/best.pt`. Model **86.8M**.

- Val climbs to **0.955** by epoch 17; train loss already tiny by epoch 5 — less plateaued than plain ViT.
- Test **bacc 0.952**, acc **0.942**, macro-F1 **0.939**; val→test drop ~0.3 pt bacc.

**Vs ViT full FT:** Modest win — **+0.6 pt test bacc** (0.952 vs 0.946), **+1.0 pt acc**, **+0.8 pt macro-F1**. Hierarchical / windowed bias helps a bit on 10-shot Flowers, but the gap is much smaller than on Derma (+6.1 pt).

**Vs ConvNeXt full FT:** Trails by **−2.0 pt test bacc** (0.952 vs 0.972) and ~2–3 pt acc/F1 — ConvNeXt remains the stronger supervised full-FT CNN.

**Vs DINOv3 linear:** Still far behind — **−4.5 pt test bacc** (0.952 vs 0.997). Best supervised Transformer full FT so far on Flowers, but frozen SSL stays the leader.

## Swin-B — LoRA on Oxford Flowers-102

Same LoRA recipe as Derma — freeze Swin-B; wrap **MLP** `nn.Linear` layers and **PatchMerging.reduction** with rank-`r` adapters; train adapters + full `head` classifier. Attention stays frozen (torchvision functional `qkv` / `proj` path). Defaults `r=8` / `α=16`. No `peft` package. Script: [`train_swin_b_lora_flowers.py`](train_swin_b_lora_flowers.py). Reuses the Flowers loader from [`train_resnet50_flowers.py`](train_resnet50_flowers.py) and LoRA helpers from [`train_swin_b_lora_derma.py`](train_swin_b_lora_derma.py).

```bash
conda activate torch
python train_swin_b_lora_flowers.py --data 102flowers --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data` | `102flowers` | Folder with `jpg/`, `imagelabels.mat`, `setid.mat` |
| `--out` | `runs/swin_b_lora_flowers` | Writes `best.pt` + `history.json` |
| `--lora-r` / `--lora-alpha` | `8` / `16` | Rank and scale |
| `--lora-dropout` | `0.05` | Dropout on LoRA input |
| `--lr-lora` / `--lr-head` | `1e-3` / `1e-3` | Adapters + classifier |
| `--eval-test` | off | Score the large official test split after training |

Compare against Swin Flowers full FT test **bacc 0.952**, ViT LoRA **0.933**, and ConvNeXt LoRA **0.971**. On Derma, Swin LoRA beat full FT; on Flowers full FT already ~0.95 — expect a tighter full-vs-LoRA gap.

### Run analysis — `swin_b_lora_flowers`

Early stop at epoch **40**; best val balanced accuracy at epoch **30** (~4.5 min). Checkpoint: `runs/swin_b_lora_flowers/best.pt`. Trainable **1.11M / 87.9M (1.26%)** (51 Linears wrapped; MLP + PatchMerging; attention frozen).

- Val climbs steadily to **0.957** at epoch 30 (slightly above full FT’s **0.955**).
- Test **bacc 0.952**, acc **0.943**, macro-F1 **0.941** — essentially tied with full FT.

**Vs Swin full FT:** Match on primary metric — test bacc **0.9523 vs 0.9520** (+0.03 pt), with **~79× fewer** trainable params. Prefer LoRA for accuracy-per-compute; full FT is not needed for the ceiling here.

**Vs ViT LoRA:** Clear win — **+1.9 pt test bacc** (0.952 vs 0.933). Hierarchical Swin + PatchMerging adapters transfer better than plain ViT MLP-only LoRA on Flowers.

**Vs ConvNeXt LoRA:** Trails by **−1.9 pt test bacc** (0.952 vs 0.971) at similar PEFT budget.

**Vs DINOv3 linear:** Still **−4.5 pt test bacc** — frozen SSL remains the Flowers leader.

## ConvNeXt - Full fine-tune on DermaMNIST
Same protocol as ResNet50 — ImageNet-pretrained **ConvNeXt-Base**, differential LRs, class weights, early stop on val balanced accuracy. Script: [`train_convnext_base_derma.py`](train_convnext_base_derma.py).

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

Early stop at epoch **34**; best val balanced accuracy at epoch **24** (~26 min). Checkpoint: `runs/convnext_base_derma/best.pt`.

## ConvNeXt - LoRA on DermaMNIST

LoRA is often thought of as “Transformers only” because the original method targets dense / Linear weights (especially attention). It is **not** limited to Transformers: low-rank updates apply wherever there are suitable weight matrices. **1×1 convolutions** behave like linear layers over channels, so classic LoRA fits naturally; larger kernels may use Conv-LoRA-style factorizations. **ConvNeXt** is a modern CNN with Transformer-inspired blocks, so PEFT/LoRA is much more natural here than on older CNN designs — which is why this arena lists **Full / LoRA** for ConvNeXt alongside ViT/Swin.

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

Early stop at epoch **19**; best val balanced accuracy at epoch **9** (~9.8 min). Checkpoint: `runs/convnext_base_lora_derma/best.pt`. Trainable **1.45M / 89.0M (1.63%)**.

**Vs full FT:** LoRA matches full FT on the arena’s primary metric (**test bacc 0.865 vs 0.868**, −0.3 pt) with **~60× fewer trainable params** and **~2.6× less wall-clock**. Overall acc (−2.5 pt) and macro-F1 (−4.9 pt) lag — full FT still better when you care about precision on rare classes, not only per-class recall.

## Self-supervised ViT — DINOv3 (linear probe / MLP)

DINOv3 is a self-supervised ViT trained to produce strong general visual features **without** ImageNet class labels. On small / domain-shifted data, a **frozen** DINO backbone + tiny head often beats full fine-tuning a supervised CNN, because you keep the pretrained representation instead of overfitting it.

**Protocol:** freeze the backbone → extract the **CLS** token → train only a head with CE (+ class weights), early stop on val balanced accuracy. Same DermaMNIST@224 splits and ImageNet mean/std as the CNN runs.

| Head | What it is | When to use |
|------|------------|-------------|
| **Linear probe** | One `Linear(D → C)` on CLS | **Start here** — standard DINO eval; tests linear separability of features |
| **MLP head** | Small MLP (e.g. Linear → GELU → Dropout → Linear) | Second run if linear plateaus; slightly more capacity, slightly more overfit risk |

Both keep DINO frozen. They differ only in head capacity — not in backbone training.

**DINOv3 linear probe** — Derma: [`train_dinov3_linear_derma.py`](train_dinov3_linear_derma.py); Flowers: [`train_dinov3_linear_flowers.py`](train_dinov3_linear_flowers.py). Default backbone: [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) (ViT-B/16). Needs Hugging Face Transformers (not in the base `torch` env until you install it):

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

**Takeaways:** Frozen DINOv3 + linear is a clean, cheap SSL baseline, but on DermaMNIST it **lags ConvNeXt full FT / LoRA by ~8–9 pt test bacc** and ~10–15 pt macro-F1. That is expected: a single linear layer cannot adapt web/LVD features to dermatoscopy the way full/LoRA FT can. 

**DINOv3 MLP head** — Derma: [`train_dinov3_mlp_derma.py`](train_dinov3_mlp_derma.py); Flowers: [`train_dinov3_mlp_flowers.py`](train_dinov3_mlp_flowers.py). Same frozen backbone and data protocol; head is `Linear(D→H) → GELU → Dropout → Linear(H→C)` (default `H=512`, dropout `0.2`). 

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

**Vs linear probe (same backbone):** MLP gains **+2.4 pt test bacc** (0.805 vs 0.781), **+7.8 pt test acc**, and **+8.5 pt macro-F1** (0.779 vs 0.694). Nonlinearity helps rare-class precision/recall, not only overall accuracy. Prefer `best.pt` (epoch 32), not epoch 40.

**Vs ConvNeXt:** Still **~6 pt behind** LoRA/full FT on test bacc (0.805 vs ~0.87). Frozen DINOv3 + MLP is a stronger SSL head than linear, but does not replace domain adaptation of the backbone. Next: light DINO LoRA if you want to close the CNN gap.

## Supervised ViT — ViT-B/16 (full fine-tune)

ImageNet-supervised **ViT-B/16** (~87M) is the classic Transformer classification baseline. On small / imbalanced data it often **overfits more** than ConvNeXt (weaker inductive bias), which is why the arena table lists **Full vs LoRA** — full FT first, then LoRA as the PEFT comparison.

**Full fine-tune** — Derma: [`train_vit_b16_derma.py`](train_vit_b16_derma.py); Flowers: [`train_vit_b16_flowers.py`](train_vit_b16_flowers.py). Same protocol as ConvNeXt — torchvision `vit_b_16` + `IMAGENET1K_V1`, differential LRs, class weights, early stop on val balanced accuracy.

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

**Takeaways:** Supervised ViT-B/16 full FT **underperforms ConvNeXt** on this set (**−7.0 pt test bacc** vs ConvNeXt full FT). Test bacc was roughly on par with frozen DINOv3 + MLP (0.798 vs 0.805) — plain ViT full FT overfits easily. 

**LoRA fine-tune** — Derma: [`train_vit_b16_lora_derma.py`](train_vit_b16_lora_derma.py); Flowers: [`train_vit_b16_lora_flowers.py`](train_vit_b16_lora_flowers.py). Freezes ViT-B/16; wraps encoder **MLP** `nn.Linear` layers with rank-`r` adapters; trains those + the full `heads` classifier. Attention stays frozen: torchvision fuses QKV into `in_proj_weight`, and `MultiheadAttention` reads `out_proj.weight` via the fused functional path (wrapping `out_proj` raises `AttributeError`). Same `r=8` / `α=16` defaults as ConvNeXt LoRA. No `peft` package.

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

**Vs ViT full FT:** LoRA wins clearly — **+6.1 pt test bacc** (0.859 vs 0.798), **+3.5 pt acc**, **+5.2 pt macro-F1**, with **~116× fewer** trainable params and less CE overfit. Prefer `best.pt` (epoch 21).

**Vs ConvNeXt:** Within **~1 pt test bacc** of ConvNeXt LoRA (0.859 vs 0.865) and close to ConvNeXt full FT (0.868). On DermaMNIST, **ViT + LoRA** is the right supervised-Transformer recipe; full FT is the weak baseline the arena predicted.

## Hierarchical ViT — Swin-B (full fine-tune)

Standard ViT uses non-overlapping square patches at a **single scale**. **Swin** uses shifted-window attention and a hierarchical feature pyramid (patch merging across stages), so it behaves more like a CNN–Transformer hybrid — often more data-efficient than plain ViT on small / domain data.

**Why Swin-B (not Swin-T):** Arena mid-size slot is ~86–89M (ConvNeXt-Base, ViT-B/16, DINOv3-B). Torchvision **Swin-B** is ~88M; **Swin-T** is ~28M (Tiny class). Use **Swin-B** for fair comparison; Swin-T only as a cheaper ablation later.

**Full fine-tune on DermaMNIST:** Same protocol as ConvNeXt / ViT — torchvision `swin_b` + `IMAGENET1K_V1`, differential LRs, class weights, early stop on val balanced accuracy. Classifier head is `model.head` (`Linear(1024 → num_classes)`). Script: [`train_swin_b_derma.py`](train_swin_b_derma.py). Flowers: [`train_swin_b_flowers.py`](train_swin_b_flowers.py).

```bash
conda activate torch
python train_swin_b_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/swin_b_derma` | Writes `best.pt` + `history.json` |
| `--lr-backbone` / `--lr-head` | `1e-4` / `1e-3` | Head = `head.*` |
| `--batch-size` | `32` | Drop to `16` if GPU OOM |

### Run analysis — `swin_b_derma`

Command: `--epochs 40 --batch-size 32 --eval-test`. Early stop at epoch **23**; best val balanced accuracy at epoch **13** (~12.8 min). Checkpoint: `runs/swin_b_derma/best.pt`. Model **86.8M** params (full FT).

**Vs ViT full FT:** Swin wins clearly — **+6.1 pt test bacc** (0.859 vs 0.798), earlier peak (epoch 13 vs 40), and **~1.7× faster**. Hierarchical / windowed bias helps on this small dermatology set where plain ViT overfits.

**Vs ConvNeXt:** Within **~1 pt test bacc** of ConvNeXt full FT (0.859 vs 0.868) and on par with ConvNeXt LoRA (0.865). Acc and macro-F1 lag ConvNeXt more (**−6.5 pt** acc, **−6.4 pt** F1 vs ConvNeXt full) — Swin’s bacc is strong (rare classes), but overall precision/recall balance is weaker. Still the best **supervised Transformer full FT** so far.

**Vs ViT LoRA:** Same test bacc (**0.859**); ViT LoRA has higher acc/F1 with far fewer trainable params. **Swin-B LoRA** (below) tests whether PEFT matches or beats this full-FT result with less overfit.

**LoRA fine-tune** — script: [`train_swin_b_lora_derma.py`](train_swin_b_lora_derma.py). Freezes Swin-B; wraps **MLP** `nn.Linear` layers and **PatchMerging.reduction** with rank-`r` adapters; trains those + the full `head` classifier. Attention stays frozen: torchvision `ShiftedWindowAttention` reads `qkv.weight` / `proj.weight` via the functional `shifted_window_attention` path (wrapping them raises `AttributeError` — same issue as ViT `out_proj`). Same `r=8` / `α=16` defaults as ConvNeXt / ViT LoRA. No `peft` package. Default run wraps **51** Linears; trainable **~1.01M / 87.8M (1.15%)**.

```bash
conda activate torch
python train_swin_b_lora_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/swin_b_lora_derma` | Writes `best.pt` + `history.json` |
| `--lora-r` / `--lora-alpha` | `8` / `16` | Rank and scale |
| `--lora-dropout` | `0.05` | Dropout on LoRA input |
| `--lr-lora` / `--lr-head` | `1e-3` / `1e-3` | Adapters + classifier |

### Run analysis — `swin_b_lora_derma`

Command: `--epochs 40 --batch-size 32 --eval-test` (`r=8`, `α=16`, MLP + PatchMerging LoRA). Early stop at epoch **29**; best val balanced accuracy at epoch **19** (~14.9 min). Checkpoint: `runs/swin_b_lora_derma/best.pt`. Trainable **1.01M / 87.8M (1.15%)**.

**Vs Swin full FT:** LoRA wins on every test metric — **+1.4 pt bacc** (0.872 vs 0.859), **+4.2 pt acc**, **+5.9 pt macro-F1**, with **~87× fewer** trainable params and less CE overfit. On DermaMNIST, **Swin + LoRA** is the right hierarchical-ViT recipe, matching the ViT full-vs-LoRA lesson.

**Vs ConvNeXt:** **Best test bacc in the arena so far** (0.872 vs ConvNeXt full 0.868 / LoRA 0.865). Acc still trails ConvNeXt full (**−2.3 pt**); macro-F1 is essentially tied (0.835 vs 0.840). Primary metric (balanced acc) favors Swin LoRA.

**Vs ViT LoRA:** **+1.4 pt test bacc** (0.872 vs 0.859). ViT LoRA keeps a slight acc/F1 edge (0.881 / 0.849 vs 0.872 / 0.835) with fewer adapters. Hierarchical bias helps rare-class bacc; plain ViT LoRA is still competitive on overall accuracy.

## B. Zero-shot & linear probe CLIP (multimodal models)

**Why include it:** CLIP-style models are trained on image–text pairs (not ImageNet class labels). **Zero-shot** uses the text tower and needs **zero** training images; a **linear probe** on the frozen image encoder is the fair multimodal counterpart to DINOv3 linear (same protocol, tiny head only).

**Why OpenCLIP ViT-B/16 DataComp-XL (not OpenAI CLIP or EVA-CLIP first):** Arena mid-size slot is ~86–89M (ConvNeXt-Base, ViT-B/16, DINOv3-B, Swin-B). A CLIP **B/16** image encoder matches that. Use **OpenCLIP** so one library can load OpenAI / LAION / DataComp / EVA later.

| Family | Concrete pick | Vision size | Role |
|--------|----------------|------------:|------|
| OpenAI CLIP | `ViT-B-16` + `openai` | ~86M | Classic citation; weaker zero-shot |
| **OpenCLIP (default)** | **`ViT-B-16` + `datacomp_xl_s13b_b90k`** | **~86M** | Capacity-matched; strongest common B/16 CLIP |
| OpenCLIP (alt) | `ViT-B-16` + `laion2b_s34b_b88k` | ~86M | LAION-2B paper checkpoint |
| EVA-CLIP | `EVA02-B-16` | ~86M | Stronger training recipe, different ViT; later ablation |
| Skip for now | CLIP-L/14, EVA-L/E | 300M–1B+ | Breaks the Base comparison |

Same **ViT-B/16** architecture; DataComp-XL public weights are stronger than original CLIP (ImageNet zero-shot ~**73.5%** vs OpenAI ~**68%** vs LAION-2B ~**70%**). Do **not** start with EVA-CLIP: EVA02-B is still Base-sized and often a bit stronger, but it is a different backbone, so a win vs ViT-B would mix “CLIP training” with “EVA architecture.” Use it as a follow-up, not the headline CLIP.

**What to run on DermaMNIST:**
1. **Zero-shot** — frozen image + text towers; prompts like `a dermoscopic photo of {class}` (no training images). Script: [`zeroshot_clip_derma.py`](zeroshot_clip_derma.py). Default: OpenCLIP **`ViT-B-16` + `datacomp_xl_s13b_b90k`**. Uses CLIP’s own preprocess (not ImageNet norms). Text side expands DermaMNIST abbreviations to MedMNIST names (`mel` → melanoma, etc.) and averages **4** dermoscopic templates. Needs `open_clip_torch` (not in the base `torch` env until you install it):

```bash
conda activate torch
pip install open_clip_torch
python zeroshot_clip_derma.py --data dermamnist_224 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/clip_zeroshot_derma` | Writes `zeroshot.pt` + `history.json` |
| `--arch` / `--pretrained` | `ViT-B-16` / `datacomp_xl_s13b_b90k` | Try `openai` or `laion2b_s34b_b88k` |
| `--eval-test` | off | Also score the test split (recommended; no training) |

First run downloads weights via Hugging Face Hub into `~/.cache/huggingface/` / OpenCLIP’s cache.

### Run analysis — `clip_zeroshot_derma`

Command: `--eval-test` (OpenCLIP `ViT-B-16` / `datacomp_xl_s13b_b90k`, 4 dermoscopic templates). No training; ~1.6 min including weight download. Classifier: `runs/clip_zeroshot_derma/zeroshot.pt` (text embeddings only). Reported **149.6M** is **image + text** towers (vision encoder is still ~86M).

**Takeaways:** Web CLIP **fails as a dermatoscope classifier** without labeled data — **−43 pt test bacc** vs frozen DINOv3 linear, **−52 pt** vs Swin LoRA. Expected: DataComp/LAION text is natural-image captions, not HAM10000 lesion types under dermoscopy. Prompt tweaks will not close that gap. The **image encoder** may still be useful; **linear probe** (below) is the fair CLIP-vs-DINO comparison.

2. **Linear probe** — frozen CLIP **image** encoder + linear head; text tower unused. Same protocol as DINOv3 linear (inverse-freq CE, early stop on val bacc). Script: [`train_clip_linear_derma.py`](train_clip_linear_derma.py). Uses OpenCLIP train/eval preprocess (CLIP norms, not ImageNet). Checkpoint stores the **head only**.

```bash
conda activate torch
python train_clip_linear_derma.py --data dermamnist_224 --epochs 40 --batch-size 32 --eval-test
```

| Flag | Default | Notes |
|------|---------|-------|
| `--out` | `runs/clip_linear_derma` | Writes `best.pt` + `history.json` |
| `--arch` / `--pretrained` | `ViT-B-16` / `datacomp_xl_s13b_b90k` | Same as zero-shot |
| `--lr` | `1e-3` | Head only (backbone frozen) |

### Run analysis — `clip_linear_derma`

Command: `--epochs 40 --batch-size 32 --eval-test` (OpenCLIP `ViT-B-16` / `datacomp_xl_s13b_b90k`). Early stop at epoch **34**; best val balanced accuracy at epoch **24** (~6.9 min). Checkpoint: `runs/clip_linear_derma/best.pt` (head only). Trainable **3,591 / 149.6M (0.0024%)**; frozen vision encoder **86.2M**.

**Vs zero-shot:** Linear probe is the CLIP number that matters — **+37 pt test bacc** (0.723 vs 0.350), **+49 pt acc**. The image encoder **does** carry dermoscopic signal; the text tower / prompts were the bottleneck.

**Vs DINOv3 linear (fair frozen-probe match):** CLIP trails by **−5.8 pt test bacc** (0.723 vs 0.781) and **−10 pt macro-F1**. Same ViT-B/16 class, but DINO’s SSL features transfer better to this medical set than CLIP’s image–text embeddings. Frozen CLIP is **not** a substitute for DINO or for domain FT (Swin LoRA 0.872).

Skip CLIP LoRA unless you want to test whether adapters close the remaining ~6 pt vs DINO linear. Original OpenAI B/16 is optional later if you want a “classic CLIP” point.

## Modern CNN (ConvNeXt)

ResNet (2015) is a classic baseline, but ConvNeXt modernized CNN architectures using design choices borrowed from Vision Transformers (7×7 depthwise convolutions, inverted bottlenecks, LayerNorm).

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
