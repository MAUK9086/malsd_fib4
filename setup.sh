#!/bin/bash
set -e

echo "=== MASLD FIB-4 Study Environment Setup ==="
echo "Target: NVIDIA Blackwell (CUDA 12.8)"

# 1. Verify CUDA 12.8
if ! nvcc --version 2>/dev/null | grep -q "12.8"; then
    echo "WARNING: CUDA 12.8 not found. Install from:"
    echo "  https://developer.nvidia.com/cuda-12-8-0-download-archive"
    echo "Continuing anyway — CPU-only inference will be used as fallback."
fi

# 2. Create Python 3.12 virtual environment
if [ ! -d ".venv" ]; then
    python3.12 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists, skipping creation"
fi
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip wheel setuptools --quiet

# 4. Install PyTorch 2.7+ with CUDA 12.8 (Blackwell native support)
echo "Installing PyTorch 2.7 with CUDA 12.8 support..."
pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --quiet

# 5. Verify GPU is detected
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cc = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cc[0]}.{cc[1]}')
    if cc[0] >= 12:
        print('Blackwell GPU confirmed.')
    else:
        print('WARNING: Not a Blackwell GPU. llama-cpp build flags may differ.')
"

# 6. Install llama-cpp-python with CUDA 12.8 / Blackwell support
# Builds from source to get sm_120 compute architecture support
echo "Building llama-cpp-python with CUDA 12.8 (sm_120)..."
CMAKE_ARGS="-DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES=120 \
            -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8" \
FORCE_CMAKE=1 pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128

# Verify llama-cpp-python
python -c "
from llama_cpp import Llama
print('llama-cpp-python imported successfully')
"

# 7. Install all other dependencies
echo "Installing requirements.txt..."
pip install -r requirements.txt --quiet

# 8. Create required output directories
mkdir -p data/raw results/exp01 results/exp02 results/exp03 \
         results/exp04 results/exp05 results/exp06 results/exp07 \
         results/exp08 results/exp09 results/exp10 logs models

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  python src/download_nhanes.py   # ~30 min, ~200 MB"
echo "  python src/merge_nhanes.py"
echo "  python src/define_cohort.py"
