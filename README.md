Perfect! Here's your updated CMake command with only modern GPU architectures to eliminate those deprecation warnings:

```bash
# Clean and reconfigure with modern GPU architectures only
cd ~/faiss
rm -rf build

# Set up Python paths (if in venv)
PYTHON_EXECUTABLE=$(which python)
PYTHON_INCLUDE_DIR="/usr/include/python3.12"
PYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.12.so"
NUMPY_INCLUDE_DIR=$(python -c "import numpy; print(numpy.get_include())")

# Configure with only modern CUDA architectures
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_CUVS=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DFAISS_ENABLE_C_API=OFF \
    -DFAISS_OPT_LEVEL=generic \
    -DBUILD_TESTING=OFF \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" \
    -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DPython_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
    -DPython_LIBRARY="$PYTHON_LIBRARY" \
    -DPython_NumPy_INCLUDE_DIR="$NUMPY_INCLUDE_DIR" \
    -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
    .
```

## What These Architectures Cover

- **75**: Turing (RTX 20 series, GTX 16 series, Tesla T4, Quadro RTX)
- **80**: Ampere (RTX 30 series, A100, A10)  
- **86**: Ampere (RTX 30 series mobile, A40, A30, A16, A2)
- **89**: Ada Lovelace (RTX 40 series, L40S, L40, L20, L4, L2)
- **90**: Hopper (H100, H200, GH200)

This covers all GPUs from 2018 onwards, which should be more than sufficient for most use cases.

## Verify Your GPU is Supported

Double-check your GPU is covered:

```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

If your GPU shows compute capability 7.0 or below, you might need to add it back, but those warnings will return. Most modern GPUs should be 7.5 or higher.

Now you can build without those annoying deprecation warnings:

```bash
# Build (should be warning-free now)
make -C build -j$(nproc) faiss
make -C build -j$(nproc) swigfaiss

# Install
cd build/faiss/python
python setup.py install
```

This approach is cleaner than suppressing warnings and ensures you're building only for actively supported GPU architectures!
