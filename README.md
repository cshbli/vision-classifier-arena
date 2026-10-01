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

## A. Modern CNN (ConvNeXt)

**Why include it:** ResNet (2015) is a classic baseline, but ConvNeXt modernized CNN architectures using design choices borrowed from Vision Transformers (7×7 depthwise convolutions, inverted bottlenecks, LayerNorm).

**Small-data impact:** ConvNeXt often outperforms ViTs on small custom datasets because its strong inductive bias prevents overfitting better than a standard ViT.

## B. Zero-shot & linear probe CLIP (multimodal models)

**Why include it:** Models like CLIP (OpenAI / OpenCLIP) or EVA-CLIP are pre-trained on billions of image–text pairs.

**Small-data impact:** Zero-shot CLIP requires **zero** training images, while a CLIP image encoder paired with a **linear probe** often sets a strong benchmark for small datasets with minimal compute.

## C. Parameter-efficient fine-tuning (PEFT / LoRA for vision)

**Why include it:** Full fine-tuning of ViTs on small datasets often leads to severe overfitting.

**Small-data impact:** Applying LoRA (Low-Rank Adaptation) to frozen ViT/Swin backbones fine-tunes only ~1% of parameters, usually yielding higher accuracy and faster training on small sample sizes.

## D. Hierarchical Vision Transformers (e.g. Swin Transformer)

**Why include it:** Standard ViT uses non-overlapping square patches at a single scale. Swin Transformer uses shifted windows and hierarchical representations.

**Small-data impact:** Swin behaves more like a hybrid between a CNN and a Transformer, making it significantly more data-efficient on small custom datasets.
