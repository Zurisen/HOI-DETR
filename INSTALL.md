# Installation — Standard x86 GPU

Tested on: Ubuntu 24.04, RTX 4090 (×2), CUDA Driver 12.2, Python 3.7.

For ARM64 HPC clusters with Hopper GPUs, see [INSTALL_HOPPER.md](INSTALL_HOPPER.md).

---

## Requirements

- Python 3.7
- CUDA-capable NVIDIA GPU (tested on RTX 3090/4090, A100)
- CUDA driver ≥ 11.3 (CUDA 12.x drivers are backward compatible)
- Anaconda or Miniconda

---

## Steps

### 1. Create conda environment

```bash
conda create -n codetr python=3.7 -y
conda activate codetr
```

### 2. Install PyTorch

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

> CUDA 11.3 wheels work with CUDA 12.x drivers due to NVIDIA's backward compatibility guarantee.

### 3. Install mmcv

```bash
pip install mmcv-full==1.5.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
```

### 4. Clone and install Co-DETR

```bash
git clone https://github.com/your-org/Co-DETR.git
cd Co-DETR
pip install -e .
```

### 5. Install remaining dependencies

```bash
pip install timm==0.6.13 fairscale==0.4.6 scipy==1.7.3 yapf==0.40.1 \
  opencv-python numpy==1.21.6 pycocotools
```

> You may see pip resolver warnings about other pre-existing packages in your environment.
> These are unrelated to Co-DETR and can be safely ignored as long as the install succeeds.

### 6. Verify

```bash
python -c "import torch, mmcv; print(torch.__version__, mmcv.__version__)"
# Expected: 1.11.0+cu113  1.5.0
```

---

## Tested Configuration

| Component | Version |
|-----------|---------|
| Python | 3.7 |
| PyTorch | 1.11.0+cu113 |
| torchvision | 0.12.0+cu113 |
| mmcv-full | 1.5.0 |
| numpy | 1.21.6 |
| scipy | 1.7.3 |
| CUDA driver | 12.2 |
| GPU | RTX 4090 |
