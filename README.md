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
| **MedMNIST** (e.g. PathMNIST, OrganAMNIST, DermaMNIST) | Medical | Small–medium, many tasks | Domain shift; easy to automate | [medmnist.com](https://medmnist.com/) · [GitHub](https://github.com/MedMNIST/MedMNIST) · `pip install medmnist` |
| **EuroSAT** | Satellite / remote sensing | ~27k, 10 classes | Non-natural-image domain | [GitHub](https://github.com/phelber/EuroSAT) · [RGB zip (DFKI)](http://madm.dfki.de/files/sentinel/EuroSAT.zip) |

Use **official splits** when available. Match each backbone’s expected preprocessing (ImageNet norms vs CLIP’s own).

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

**LoRA on ConvNeXt:** LoRA is often thought of as “Transformers only” because the original method targets dense / Linear weights (especially attention). It is **not** limited to Transformers: low-rank updates apply wherever there are suitable weight matrices. **1×1 convolutions** behave like linear layers over channels, so classic LoRA fits naturally; larger kernels may use Conv-LoRA-style factorizations. **ConvNeXt** is a modern CNN with Transformer-inspired blocks, so PEFT/LoRA is much more natural here than on older CNN designs — which is why this arena lists **Full / LoRA** for ConvNeXt alongside ViT/Swin.

## B. Zero-shot & linear probe CLIP (multimodal models)

**Why include it:** Models like CLIP (OpenAI / OpenCLIP) or EVA-CLIP are pre-trained on billions of image–text pairs.

**Small-data impact:** Zero-shot CLIP requires **zero** training images, while a CLIP image encoder paired with a **linear probe** often sets a strong benchmark for small datasets with minimal compute.

## C. Parameter-efficient fine-tuning (PEFT / LoRA for vision)

**Why include it:** Full fine-tuning of ViTs on small datasets often leads to severe overfitting.

**Small-data impact:** Applying LoRA (Low-Rank Adaptation) to frozen ViT/Swin (and similarly **ConvNeXt**) backbones fine-tunes only ~1% of parameters, usually yielding higher accuracy and faster training on small sample sizes. See also [LoRA on ConvNeXt](#a-modern-cnn-convnext).

## D. Hierarchical Vision Transformers (e.g. Swin Transformer)

**Why include it:** Standard ViT uses non-overlapping square patches at a single scale. Swin Transformer uses shifted windows and hierarchical representations.

**Small-data impact:** Swin behaves more like a hybrid between a CNN and a Transformer, making it significantly more data-efficient on small custom datasets.
