# HOI-DETR: Hand-Object Interaction Detection Transformer

> **Improving and Evaluating Hand-Object Interaction Detection**

HOI-DETR is a transformer-based framework for detecting hands, hand-held objects, and their interactions in images and video. Built on the [Co-DETR](https://github.com/Sense-X/Co-DETR) architecture, it adds a lightweight interaction module that jointly predicts all visible hands, 1st objects (objects in direct contact with a hand), and 2nd objects (objects acted upon through a tool), along with their pairwise interaction links — all in a single forward pass.

![HOI-DETR method overview](assets/method.png)

---

## Highlights

- **State-of-the-art HOI detection** across four diverse benchmarks: Hands23, HOIST, FineBio, and HD-EPIC-HOI
- **+20 mAP points** improvement over the previous state of the art on Hands23 and FineBio
- **Interaction module** that predicts hand → 1st object and 1st object → 2nd object relations from decoder token pairs
- **New video benchmark** HD-EPIC-HOI with spatiotemporal consistency metrics (ST-STC, LTC)
- **Refined Hands23 annotations** correcting duplicate bounding boxes across 26k images

---

## Demo

Edit the paths and settings at the top of `demo/demo.py`, then run:

```bash
export PYTHONPATH=".:$PYTHONPATH"
python demo/demo.py
```

Key settings in `demo/demo.py`:

```python
MODEL_CONFIG = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py'
CHECKPOINT   = 'checkpoints/epoch_5.pth'
DEVICE       = 'cuda:0'
INPUT_DIR    = 'demo/example_images'   # directory of input images
OUTPUT_DIR   = None                    # None -> demo/results/<input_dir_name>/
SCORE_THR    = 0.3
VERBOSE_LABELS = False
```

Results are saved to `demo/results/<input_dir_name>/` by default, preserving original filenames.

---

## Evaluation

```bash
bash -c 'export PYTHONPATH=".:$PYTHONPATH" && \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
  bash ./tools/dist_test.sh \
    projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py \
    checkpoints/epoch_5.pth \
    4 \
    --eval bbox'
```

---

## Checkpoints

| Model | Backbone | Hands23 AP50 | Download |
|-------|----------|-------------|----------|
| HOI-DETR | ViT-L/16 | 86.1 | [epoch_5.pth](#) |

---

## Installation

We provide separate installation guides for two platforms:

| Platform | Guide |
|----------|-------|
| Standard x86 GPU (RTX 3090/4090, A100) | [INSTALL.md](INSTALL.md) |
| ARM64 HPC with Hopper GPUs (GH200, Isambard-AI) | [INSTALL_HOPPER.md](INSTALL_HOPPER.md) |

---

## Model Overview

HOI-DETR extends Co-DETR with three components:

**Backbone and Encoder.** A ViT-L/16 backbone processes the input image into patch embeddings, which are refined by a 6-layer transformer encoder using multi-scale deformable attention.

**Decoder.** A transformer decoder maps Q query tokens to detections across four role-based classes: `{hand, 1st object, 2nd object, background}`. Unlike semantic detection, labels depend on the object's role in the scene — a pan held in a hand is a 1st object; the same pan resting on a stove is background.

**Interaction Module.** An MLP head operates on pairs of decoder token embeddings to predict binary interaction relationships. Only valid pairs are evaluated: `hand → 1st object` and `1st object → 2nd object`. The module is supervised at every decoder layer with a focal loss and trained end-to-end with the detector.

---

## Datasets

| Dataset | Images | Hand | 1st obj | 2nd obj |
|---------|--------|------|---------|---------|
| Hands23 (refined) | 24.6k | 39.8k | 2.5k | 2.5k |
| HD-EPIC-HOI | 41.9k | — | 26k | — |
| HOIST | 3.5k | — | 3.9k | — |
| FineBio | 238 | 465 | 372 | — |

**Refined Hands23.** We corrected duplicate object annotations arising from the hand-centric annotation pipeline in the original Hands23 dataset. Across 26.2k reviewed images, 56.2% required correction, yielding cleaner training targets and more reliable evaluation.

**HD-EPIC-HOI.** A new video benchmark derived from HD-EPIC, consisting of 911 sequences (41.9k frames) centred around object contact events. Ground-truth masks are propagated with SAM2. Two spatiotemporal metrics are introduced: Short-Term Spatio-Temporal Consistency (ST-STC) and Long-Term Consistency (LTC).

---

## Results

**Hands23 (val)**

| Method | Hand AP50 | 1st obj AP50 | 2nd obj AP50 | F1 inter |
|--------|-----------|-------------|-------------|----------|
| Hands23 | 85.2 | 59.4 | 46.2 | 90.7 |
| HOI-DETR (ours) | **93.1** | **86.5** | **78.7** | **95.5** |

**Cross-dataset (zero-shot)**

| Method | HOIST | HD-EPIC-HOI | FineBio (1st obj) |
|--------|-------|-------------|-------------------|
| Hands23 | 43.1 | 42.4 | 26.0 |
| HOIST | 70.7 | 28.4 | — |
| HOI-DETR (ours) | **76.6** | **67.6** | **55.8** |

**Spatiotemporal consistency (HD-EPIC-HOI)**

| Method | ST-STC | LTC |
|--------|--------|-----|
| Hands23 | 91.9 | 37.2 |
| HOIST | 92.4 | 32.4 |
| HOI-DETR (ours) | **95.4** | **61.9** |

---

## Acknowledgements

This work builds on [Co-DETR](https://github.com/Sense-X/Co-DETR), [MMDetection](https://github.com/open-mmlab/mmdetection), and [Hands23](https://github.com/shan-it/Hands23). We thank the authors of HD-EPIC, HOIST, and FineBio for making their datasets available.
