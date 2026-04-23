# Installation — Hopper HPC (aarch64)

Tested on: Isambard-AI, GH200 Hopper GPUs, aarch64, CUDA 12.0, Python 3.10.

For standard x86 machines (RTX 3090/4090, A100), see [INSTALL.md](INSTALL.md).

> **Why a separate guide?** aarch64 has no prebuilt PyPI wheels for PyTorch or torchvision.
> conda-forge is used instead, and mmcv must be compiled from source. Several compatibility
> patches are also required due to API changes in PyTorch 2.4.

---

## Requirements

- Python 3.10
- CUDA 12.x (loaded via environment modules)
- GCC 12.3+
- Miniforge or Miniconda

If Miniforge is not installed:

```bash
cd $HOME
curl --location --remote-name \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
source ~/miniforge3/bin/activate
```

---

## Steps

> **Important:** Run all installation steps on a compute node, not the login node.
> Request one before starting:
> ```bash
> srun -p workq -N 1 -t 4:00:00 --gres=gpu:1 --pty bash
> source ~/miniforge3/bin/activate
> ```

### 1. Create conda environment

```bash
conda create -n codetr python=3.10 -y
conda activate codetr
```

### 2. Install PyTorch

> Do **not** use `pip install torch` on aarch64 — prebuilt pip wheels are unavailable for
> this platform and pip will silently install a mismatched version.

```bash
conda install -c conda-forge pytorch=2.4.1=*cu* -y
```

Verify CUDA is enabled:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: 2.4.1  True
```

If `cuda.is_available()` returns `False`, the install resolved to a CPU build. Rerun:

```bash
conda install -c conda-forge pytorch=2.4.1=*cu* --force-reinstall -y
```

### 3. Install torchvision

```bash
conda install -c conda-forge torchvision=0.19.1 -y
```

### 4. Install dependencies

> **numpy must be pinned before building mmcv.** The `<2.0` constraint is critical — mmcv
> ops and scipy are compiled against the numpy 1.x ABI and will crash silently with numpy 2.x.

```bash
pip install --no-cache-dir "numpy<2.0"
pip install --no-cache-dir "scipy<1.13"
pip install --no-cache-dir opencv-python-headless==4.8.0.76
pip install --no-cache-dir fvcore tensorboard einops
pip install --no-cache-dir timm fairscale==0.4.13 yapf==0.40.1
pip install --no-cache-dir pycocotools
```

Verify:

```bash
python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
# Expected: 1.26.4  1.12.x
```

### 5. Load CUDA and compiler modules

```bash
module load cuda/12.6
module load gcc-native/12.3
gcc --version    # confirm GCC 12.3
nvcc --version   # confirm CUDA toolkit on PATH
```

> Module names may differ across clusters. Ensure `nvcc` is available before building mmcv.

### 6. Build mmcv from source

```bash
git clone --branch v1.7.2 https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install --force-reinstall "setuptools==60.2.0"
```

Set build flags for Hopper (SM 9.0):

```bash
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=4
```

Build:

```bash
MMCV_WITH_OPS=1 python setup.py develop
```

Verify:

```bash
python -c "import mmcv; from mmcv.ops import RoIPool; print(mmcv.__version__)"
# Expected: 1.7.2
```

### 7. Clone Co-DETR and install mmdet

```bash
git clone https://github.com/your-org/Co-DETR.git
cd Co-DETR
pip install --no-cache-dir openmim
mim install mmdet==2.25.3
```

### 8. Fix mmdet version check

mmdet 2.25.3 enforces a strict upper-bound mmcv version check that rejects the mmcv 1.7.2
source build. Relax it:

```bash
MMDET_INIT=$(python -c "import mmdet, os; print(os.path.dirname(mmdet.__file__))")/__init__.py
nano $MMDET_INIT
```

Find this assertion:

```python
assert mmcv_version >= digit_version(mmcv_minimum_version) and \
       mmcv_version < digit_version(mmcv_maximum_version)
```

Change it to:

```python
assert mmcv_version >= digit_version(mmcv_minimum_version)
```

### 9. Verify

```bash
python -c "import torch, mmcv, torchvision; print(torch.__version__, mmcv.__version__, torchvision.__version__)"
# Expected: 2.4.1  1.7.2  0.19.1
```

---

## Troubleshooting

### Multi-GPU evaluation fails with `_use_replicated_tensor_module`

```
AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'
```

This is a torch 2.4 / mmcv DDP incompatibility. Apply the following patch:

```bash
sed -i 's/self._use_replicated_tensor_module else self.module/getattr(self, "_use_replicated_tensor_module", False) else self.module/' \
    /path/to/mmcv/mmcv/parallel/distributed.py
```

No reinstall needed — mmcv is installed in editable mode.

---

### `torch.cuda.is_available()` returns `False`

The conda install resolved to a CPU-only build. Reinstall:

```bash
conda install -c conda-forge pytorch=2.4.1=*cu* --force-reinstall -y
```

---

### `numpy.dtype size changed` or `_ARRAY_API not found`

numpy 2.x is present and conflicts with packages compiled against numpy 1.x:

```bash
pip install "numpy<2.0" --force-reinstall
pip install "scipy<1.13" --force-reinstall --no-cache-dir
```

---

### `ModuleNotFoundError: No module named 'pkg_resources'`

setuptools was upgraded and dropped `pkg_resources`. Fix:

```bash
pip install --force-reinstall "setuptools==60.2.0"
```

---

### pip silently installs torch 2.6.0 and breaks the environment

Never run `pip install torch` on aarch64. If this has happened:

```bash
pip uninstall torch torchvision -y
conda install -c conda-forge pytorch=2.4.1=*cu* --force-reinstall -y
conda install -c conda-forge torchvision=0.19.1 --force-reinstall -y
```

---

## Tested Configuration

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.4.1 (cu120) |
| torchvision | 0.19.1 |
| mmcv-full | 1.7.2 (source) |
| mmdet | 2.25.3 |
| numpy | 1.26.4 |
| scipy | 1.12.x |
| opencv-python-headless | 4.8.0.76 |
| fairscale | 0.4.13 |
| CUDA driver | 12.0 |
| GPU | GH200 (SM 9.0) |
