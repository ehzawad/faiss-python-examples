#!/usr/bin/env bash

# CPU-only FAISS builder for Apple Silicon (arm64)
# - Builds FAISS from source with OpenBLAS + OpenMP
# - Builds and installs Python bindings from the CMake build tree
# - No CUDA/ROCm, no PyTorch requirement
#
# Usage:
#   bash build_metal_faiss_m4.sh [venv_path] [python_cmd]
#
# Examples:
#   bash build_metal_faiss_m4.sh               # uses ~/venv-faiss and python3
#   bash build_metal_faiss_m4.sh .venv-faiss python3.12

set -euo pipefail

echo "🚀 Building FAISS (CPU) for Apple Silicon (arm64)"
echo "=================================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
VENV_PATH="${1:-$HOME/venv-faiss}"
PYTHON_CMD="${2:-python3}"
BUILD_DIR="faiss_build"
SRC_DIR="$(pwd)"
NUM_CORES="$(sysctl -n hw.ncpu)"

# Sanity: ensure we're at the FAISS repo root (contains CMakeLists.txt)
if [[ ! -f "CMakeLists.txt" ]] || ! grep -qiE "^\s*project\(\s*faiss\b" CMakeLists.txt 2>/dev/null; then
  echo "❌ Please run this script from the FAISS repository root (where CMakeLists.txt lives)."
  exit 1
fi

cat <<INFO
Using configuration:
- CMake source dir: $SRC_DIR
- Build dir:        $BUILD_DIR
- Python launcher:  $PYTHON_CMD
- Virtual env:      $VENV_PATH
- Parallel jobs:    $NUM_CORES
INFO

# -----------------------------------------------------------------------------
# Step 1: Homebrew dependencies
# -----------------------------------------------------------------------------
 echo "📦 Ensuring Homebrew dependencies (openblas, libomp, swig, gflags, cmake)..."
 deps=(openblas libomp swig gflags cmake)
 for dep in "${deps[@]}"; do
   if ! brew list --versions "$dep" >/dev/null; then
     echo "Installing $dep..."
     brew install "$dep"
   else
     echo "✅ $dep already installed"
   fi
 done
 OPENBLAS_ROOT="$(brew --prefix openblas)"
 LIBOMP_ROOT="$(brew --prefix libomp)"
 echo "OpenBLAS: $OPENBLAS_ROOT"
 echo "libomp:   $LIBOMP_ROOT"
 echo

# -----------------------------------------------------------------------------
# Step 2: Python virtual environment
# -----------------------------------------------------------------------------
 echo "🐍 Creating virtual environment at: $VENV_PATH ..."
 rm -rf "$VENV_PATH"
 "$PYTHON_CMD" -m venv "$VENV_PATH"
 # shellcheck disable=SC1090
 source "$VENV_PATH/bin/activate"
 python -m pip install --upgrade pip setuptools wheel build
 python -m pip install numpy packaging
 PYTHON_EXECUTABLE="$VENV_PATH/bin/python"
 PY_VER="$($PYTHON_EXECUTABLE -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
 echo "Python in venv: $PY_VER"
 echo

# -----------------------------------------------------------------------------
# Step 3: Configure toolchain (Apple Clang + OpenMP)
# -----------------------------------------------------------------------------
 echo "🔧 Configuring toolchain..."
 export CC="/usr/bin/clang"
 export CXX="/usr/bin/clang++"
 export CMAKE_OSX_ARCHITECTURES="arm64"
 export CMAKE_APPLE_SILICON_PROCESSOR="arm64"

 # Prefer rpath over DYLD at runtime
 export CMAKE_INSTALL_RPATH="$OPENBLAS_ROOT/lib;$LIBOMP_ROOT/lib"
 export CMAKE_BUILD_RPATH="$CMAKE_INSTALL_RPATH"

 # Headers and libs for brew packages
 export CPPFLAGS="-I$OPENBLAS_ROOT/include -I$LIBOMP_ROOT/include"
 export LDFLAGS="-L$OPENBLAS_ROOT/lib -L$LIBOMP_ROOT/lib"
 export PKG_CONFIG_PATH="$OPENBLAS_ROOT/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
 echo "✅ Toolchain configured"
 echo

# -----------------------------------------------------------------------------
# Step 4: Configure CMake (out-of-source)
# -----------------------------------------------------------------------------
 echo "🏗️  Configuring CMake..."
 rm -rf "$BUILD_DIR"
 echo "Running CMake configure with:" \
 && echo "  CXX=$CXX" \
 && echo "  CC=$CC" \
 && echo "  BUILD_DIR=$BUILD_DIR" \
 && echo "  INSTALL_PREFIX=$VENV_PATH" \
 && echo "  OpenBLAS=$OPENBLAS_ROOT libomp=$LIBOMP_ROOT" \
 && cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_CXX_COMPILER="$CXX" \
   -DCMAKE_OSX_ARCHITECTURES=arm64 \
   -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 \
   -DCMAKE_INSTALL_PREFIX="$VENV_PATH" \
   -DCMAKE_INSTALL_RPATH="$CMAKE_INSTALL_RPATH" \
   -DCMAKE_BUILD_RPATH="$CMAKE_BUILD_RPATH" \
   -DBUILD_TESTING=OFF \
   -DBUILD_SHARED_LIBS=ON \
   -DFAISS_ENABLE_GPU=OFF \
   -DFAISS_ENABLE_MKL=OFF \
   -DFAISS_ENABLE_C_API=OFF \
   -DFAISS_ENABLE_PYTHON=ON \
   -DFAISS_OPT_LEVEL=generic \
   -DBLA_VENDOR=OpenBLAS \
   -DBLAS_LIBRARIES="$OPENBLAS_ROOT/lib/libopenblas.dylib" \
   -DLAPACK_LIBRARIES="$OPENBLAS_ROOT/lib/libopenblas.dylib" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_ROOT/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$LIBOMP_ROOT/lib/libomp.dylib" \
   -DPython_EXECUTABLE="$PYTHON_EXECUTABLE"
 echo "✅ CMake configured"
 echo

# -----------------------------------------------------------------------------
# Step 5: Build swigfaiss target (builds FAISS core and Python extension)
# -----------------------------------------------------------------------------
 echo "⚙️  Building FAISS + Python bindings..."
 cmake --build "$BUILD_DIR" --target swigfaiss -j "$NUM_CORES"
 echo "✅ Build finished"
 echo

# -----------------------------------------------------------------------------
# Step 6: Install FAISS libraries to venv prefix
# -----------------------------------------------------------------------------
 echo "📦 Installing FAISS to: $VENV_PATH ..."
 cmake --install "$BUILD_DIR" --prefix "$VENV_PATH"
 echo "✅ Libraries installed"
 echo

# -----------------------------------------------------------------------------
# Step 7: Install Python package from build tree
# -----------------------------------------------------------------------------
 PY_BUILD_PY_DIR="$BUILD_DIR/faiss/python"
 if [[ ! -d "$PY_BUILD_PY_DIR" ]]; then
   echo "❌ Expected python build dir not found: $PY_BUILD_PY_DIR"
   exit 1
 fi
echo "🐍 Installing Python package from: $PY_BUILD_PY_DIR"
pushd "$PY_BUILD_PY_DIR" >/dev/null

# Work around potential nested contrib trees copied from repo state
if [[ -d contrib/contrib ]]; then
  echo "Fixing nested contrib: removing contrib/contrib"
  rm -rf contrib/contrib
fi

# Patch setup.py to be tolerant if contrib copy fails
"$PYTHON_EXECUTABLE" - <<'PY'
from pathlib import Path
p = Path('setup.py')
if p.exists():
    txt = p.read_text()
    old = 'shutil.copytree("contrib", "faiss/contrib")'
    new = (
        'try:\n'
        '    shutil.copytree("contrib", "faiss/contrib")\n'
        'except Exception as e:\n'
        '    print(f"Warning: contrib copy failed: {e}")\n'
        '    os.makedirs("faiss/contrib", exist_ok=True)\n'
        '    open("faiss/contrib/__init__.py", "w").close()\n'
    )
    if old in txt:
        p.write_text(txt.replace(old, new))
PY

# Ensure a modern PEP 517 build (standard frontend via `python -m build`)
if [[ ! -f pyproject.toml ]]; then
cat > pyproject.toml <<'PYPROJECT'
[build-system]
requires = [
  "setuptools>=61",
  "wheel",
  "numpy>=1.22",
  "packaging"
]
build-backend = "setuptools.build_meta"
PYPROJECT
fi

# Clean previous artifacts and build a wheel using PEP 517 (isolated)
rm -rf dist build *.egg-info
python -m build --wheel

# Install the built wheel into the venv
built_wheel=$(ls -1 dist/*.whl 2>/dev/null | head -n 1)
if [[ -z "${built_wheel:-}" ]]; then
  echo "❌ Wheel build failed: no wheel found in dist/"
  exit 1
fi
python -m pip install --no-deps "$built_wheel"

popd >/dev/null
 echo "✅ Python package installed"
 echo

# -----------------------------------------------------------------------------
# Step 8: Quick validation
# -----------------------------------------------------------------------------
 echo "🧪 Validating import..."
 "$PYTHON_EXECUTABLE" - <<'PY'
import sys
print("Python:", sys.version)
import faiss
print("faiss module:", faiss.__file__)
print("faiss version:", getattr(faiss, "__version__", "n/a"))
PY
 echo "✅ Import OK"
 echo

 echo "🎉 BUILD COMPLETE (CPU-only)"
 echo "Activate with: source \"$VENV_PATH/bin/activate\""
 echo "Test import:  python -c 'import faiss, numpy as np; print(faiss.__file__)'"

 # Notes:
 # - If you see runtime errors about libomp.dylib, ensure your shell has:
 #     export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$(brew --prefix openblas)/lib:$DYLD_LIBRARY_PATH"
 #   or ensure your environment does not strip rpaths.
