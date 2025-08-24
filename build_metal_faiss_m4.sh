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
#   bash build_metal_faiss_m4.sh               # auto-detects active venv or uses ~/venv-faiss
#   bash build_metal_faiss_m4.sh .venv-faiss python3.12

set -euo pipefail

echo "🚀 Building FAISS (CPU) for Apple Silicon (arm64)"
echo "=================================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Auto-detect active virtual environment
echo "📋 Checking for active virtual environment..."
echo "   VIRTUAL_ENV variable: ${VIRTUAL_ENV:-'(not set)'}"
echo "   Command line arg 1: ${1:-'(not provided)'}"
echo "   Command line arg 2: ${2:-'(not provided)'}"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  VENV_PATH="${1:-$VIRTUAL_ENV}"
  PYTHON_CMD="${2:-python}"
  echo "🔍 Detected active virtual environment: $VIRTUAL_ENV"
  echo "   Using VENV_PATH: $VENV_PATH"
  echo "   Using PYTHON_CMD: $PYTHON_CMD"
else
  VENV_PATH="${1:-$HOME/venv-faiss}"
  PYTHON_CMD="${2:-python3}"
  echo "❌ No active virtual environment detected"
  echo "   Fallback VENV_PATH: $VENV_PATH"
  echo "   Fallback PYTHON_CMD: $PYTHON_CMD"
fi
BUILD_DIR="faiss_build"
SRC_DIR="$(pwd)"
NUM_CORES="$(sysctl -n hw.ncpu)"

# Sanity: ensure we're at the FAISS repo root (contains CMakeLists.txt)
echo "🔍 Validating FAISS repository root..."
echo "   Current directory: $(pwd)"
echo "   Looking for CMakeLists.txt..."
if [[ ! -f "CMakeLists.txt" ]]; then
  echo "❌ CMakeLists.txt not found in current directory"
  echo "❌ Please run this script from the FAISS repository root (where CMakeLists.txt lives)."
  exit 1
fi
echo "   ✅ CMakeLists.txt found"
echo "   Checking if it's a FAISS project..."
if ! grep -qiE "^\s*project\(\s*faiss\b" CMakeLists.txt 2>/dev/null; then
  echo "❌ CMakeLists.txt doesn't appear to be for FAISS project"
  echo "❌ Please run this script from the FAISS repository root."
  exit 1
fi
echo "   ✅ Confirmed FAISS project"

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
 echo "   Checking Homebrew installation..."
 if ! command -v brew >/dev/null; then
   echo "❌ Homebrew not found. Please install Homebrew first."
   exit 1
 fi
 echo "   ✅ Homebrew found at: $(which brew)"
 
 deps=(openblas libomp swig gflags cmake)
 for dep in "${deps[@]}"; do
   echo "   Checking dependency: $dep"
   if ! brew list --versions "$dep" >/dev/null 2>&1; then
     echo "   📥 Installing $dep..."
     brew install "$dep"
     echo "   ✅ $dep installed successfully"
   else
     version=$(brew list --versions "$dep")
     echo "   ✅ $dep already installed: $version"
   fi
 done
 
 echo "   🔍 Getting brew prefixes..."
 OPENBLAS_ROOT="$(brew --prefix openblas)"
 LIBOMP_ROOT="$(brew --prefix libomp)"
 echo "   OpenBLAS: $OPENBLAS_ROOT"
 echo "   libomp:   $LIBOMP_ROOT"
 
 echo "   🔍 Verifying library files exist..."
 if [[ -f "$OPENBLAS_ROOT/lib/libopenblas.dylib" ]]; then
   echo "   ✅ OpenBLAS library found"
 else
   echo "   ❌ OpenBLAS library not found at expected location"
 fi
 if [[ -f "$LIBOMP_ROOT/lib/libomp.dylib" ]]; then
   echo "   ✅ libomp library found"
 else
   echo "   ❌ libomp library not found at expected location"
 fi
 echo

# -----------------------------------------------------------------------------
# Step 2: Python virtual environment
# -----------------------------------------------------------------------------
echo "🔍 Virtual environment setup decision logic..."
echo "   VIRTUAL_ENV is set: $([[ -n "${VIRTUAL_ENV:-}" ]] && echo 'YES' || echo 'NO')"
echo "   VENV_PATH equals VIRTUAL_ENV: $([[ "$VENV_PATH" == "$VIRTUAL_ENV" ]] && echo 'YES' || echo 'NO')"

if [[ -n "${VIRTUAL_ENV:-}" && "$VENV_PATH" == "$VIRTUAL_ENV" ]]; then
  echo "🐍 Using current active virtual environment: $VENV_PATH"
  PYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python"
  echo "   Python executable: $PYTHON_EXECUTABLE"
  echo "   Checking if python executable exists..."
  if [[ -f "$PYTHON_EXECUTABLE" ]]; then
    echo "   ✅ Python executable found"
  else
    echo "   ❌ Python executable not found at: $PYTHON_EXECUTABLE"
    exit 1
  fi
  echo "   📦 Installing/upgrading required packages..."
  python -m pip install --upgrade pip setuptools wheel build numpy packaging
  echo "   ✅ Packages installed/upgraded"
else
  echo "🐍 Creating virtual environment at: $VENV_PATH ..."
  echo "   Checking if target directory exists..."
  if [[ -d "$VENV_PATH" ]]; then
    echo "   🗑️  Removing existing directory: $VENV_PATH"
  fi
  rm -rf "$VENV_PATH"
  echo "   🏗️  Creating new venv with: $PYTHON_CMD -m venv $VENV_PATH"
  "$PYTHON_CMD" -m venv "$VENV_PATH"
  echo "   ✅ Virtual environment created"
  echo "   🔧 Activating virtual environment..."
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
  echo "   ✅ Virtual environment activated"
  PYTHON_EXECUTABLE="$VENV_PATH/bin/python"
  echo "   Python executable: $PYTHON_EXECUTABLE"
  echo "   📦 Installing base packages..."
  python -m pip install --upgrade pip setuptools wheel build
  echo "   📦 Installing numpy and packaging..."
  python -m pip install numpy packaging
  echo "   ✅ All packages installed"
fi
echo "🔍 Getting Python version..."
PY_VER="$($PYTHON_EXECUTABLE -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "✅ Python in venv: $PY_VER"
echo "   Full Python version: $($PYTHON_EXECUTABLE --version)"
echo

# -----------------------------------------------------------------------------
# Step 3: Configure toolchain (Apple Clang + OpenMP)
# -----------------------------------------------------------------------------
 echo "🔧 Configuring toolchain..."
 echo "   Setting up compilers..."
 export CC="/usr/bin/clang"
 export CXX="/usr/bin/clang++"
 echo "   CC=$CC"
 echo "   CXX=$CXX"
 
 echo "   Verifying compilers exist..."
 if [[ -f "$CC" ]]; then
   echo "   ✅ C compiler found: $($CC --version | head -n1)"
 else
   echo "   ❌ C compiler not found at: $CC"
 fi
 if [[ -f "$CXX" ]]; then
   echo "   ✅ C++ compiler found: $($CXX --version | head -n1)"
 else
   echo "   ❌ C++ compiler not found at: $CXX"
 fi
 
 echo "   Setting architecture flags..."
 export CMAKE_OSX_ARCHITECTURES="arm64"
 export CMAKE_APPLE_SILICON_PROCESSOR="arm64"
 echo "   CMAKE_OSX_ARCHITECTURES=$CMAKE_OSX_ARCHITECTURES"
 echo "   CMAKE_APPLE_SILICON_PROCESSOR=$CMAKE_APPLE_SILICON_PROCESSOR"

 echo "   Getting Python library information..."
 PYTHON_LIB_DIR="$($PYTHON_EXECUTABLE -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
 PYTHON_INCLUDE_DIR="$($PYTHON_EXECUTABLE -c 'import sysconfig; print(sysconfig.get_path("include"))')"
 PYTHON_LIBRARY_NAME="$($PYTHON_EXECUTABLE -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY") or sysconfig.get_config_var("LIBRARY") or "")')"
 PYTHON_LIBRARY_PATH="$($PYTHON_EXECUTABLE -c 'import sysconfig; lib=sysconfig.get_config_var("LIBDIR"); name=sysconfig.get_config_var("LDLIBRARY") or sysconfig.get_config_var("LIBRARY"); print(f"{lib}/{name}" if lib and name else "")')"
 echo "   Python lib dir: $PYTHON_LIB_DIR"
 echo "   Python include dir: $PYTHON_INCLUDE_DIR"
 echo "   Python library name: $PYTHON_LIBRARY_NAME"
 echo "   Python library path: $PYTHON_LIBRARY_PATH"
 
 # Check if the Python library exists
 if [[ -n "$PYTHON_LIBRARY_PATH" && -f "$PYTHON_LIBRARY_PATH" ]]; then
   echo "   ✅ Python library found at: $PYTHON_LIBRARY_PATH"
 else
   echo "   ⚠️ Python library not found, will use framework linking"
   PYTHON_LIBRARY_PATH=""
 fi

 echo "   Setting up rpath configuration..."
 export CMAKE_INSTALL_RPATH="$OPENBLAS_ROOT/lib;$LIBOMP_ROOT/lib;$PYTHON_LIB_DIR"
 export CMAKE_BUILD_RPATH="$CMAKE_INSTALL_RPATH"
 echo "   CMAKE_INSTALL_RPATH=$CMAKE_INSTALL_RPATH"
 echo "   CMAKE_BUILD_RPATH=$CMAKE_BUILD_RPATH"

 echo "   Setting up compiler and linker flags..."
 export CPPFLAGS="-I$OPENBLAS_ROOT/include -I$LIBOMP_ROOT/include -I$PYTHON_INCLUDE_DIR"
 export LDFLAGS="-L$OPENBLAS_ROOT/lib -L$LIBOMP_ROOT/lib -L$PYTHON_LIB_DIR"
 export PKG_CONFIG_PATH="$OPENBLAS_ROOT/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
 echo "   CPPFLAGS=$CPPFLAGS"
 echo "   LDFLAGS=$LDFLAGS"
 echo "   PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
 echo "✅ Toolchain configured"
 echo

# -----------------------------------------------------------------------------
# Step 4: Configure CMake (out-of-source)
# -----------------------------------------------------------------------------
 echo "🏗️  Configuring CMake..."
 echo "   Checking if cmake is available..."
 if ! command -v cmake >/dev/null; then
   echo "   ❌ CMake not found in PATH"
   exit 1
 fi
 echo "   ✅ CMake found: $(cmake --version | head -n1)"
 
 echo "   Cleaning build directory: $BUILD_DIR"
 if [[ -d "$BUILD_DIR" ]]; then
   echo "   🗑️  Removing existing build directory"
 fi
 rm -rf "$BUILD_DIR"
 echo "   ✅ Build directory cleaned"
 
 echo "   📋 CMake configuration summary:"
 echo "     Source dir: $SRC_DIR"
 echo "     Build dir: $BUILD_DIR"
 echo "     Install prefix: $VENV_PATH"
 echo "     C compiler: $CC"
 echo "     C++ compiler: $CXX"
 echo "     Python executable: $PYTHON_EXECUTABLE"
 echo "     Python include dir: $PYTHON_INCLUDE_DIR"
 echo "     Python lib dir: $PYTHON_LIB_DIR"
 echo "     OpenBLAS root: $OPENBLAS_ROOT"
 echo "     libomp root: $LIBOMP_ROOT"
 
 echo "   🚀 Running CMake configure..."
 cmake_args=(
   -S "$SRC_DIR" -B "$BUILD_DIR"
   -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_CXX_COMPILER="$CXX"
   -DCMAKE_OSX_ARCHITECTURES=arm64
   -DCMAKE_APPLE_SILICON_PROCESSOR=arm64
   -DCMAKE_INSTALL_PREFIX="$VENV_PATH"
   -DCMAKE_INSTALL_RPATH="$CMAKE_INSTALL_RPATH"
   -DCMAKE_BUILD_RPATH="$CMAKE_BUILD_RPATH"
   -DCMAKE_SHARED_LINKER_FLAGS="-undefined dynamic_lookup"
   -DBUILD_TESTING=OFF
   -DBUILD_SHARED_LIBS=ON
   -DFAISS_ENABLE_GPU=OFF
   -DFAISS_ENABLE_MKL=OFF
   -DFAISS_ENABLE_C_API=OFF
   -DFAISS_ENABLE_PYTHON=ON
   -DFAISS_OPT_LEVEL=generic
   -DBLA_VENDOR=OpenBLAS
   -DBLAS_LIBRARIES="$OPENBLAS_ROOT/lib/libopenblas.dylib"
   -DLAPACK_LIBRARIES="$OPENBLAS_ROOT/lib/libopenblas.dylib"
   -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_ROOT/include"
   -DOpenMP_CXX_LIB_NAMES=omp
   -DOpenMP_omp_LIBRARY="$LIBOMP_ROOT/lib/libomp.dylib"
   -DPython_EXECUTABLE="$PYTHON_EXECUTABLE"
   -DPython_INCLUDE_DIRS="$PYTHON_INCLUDE_DIR"
   -DPython_LIBRARY_DIRS="$PYTHON_LIB_DIR"
   -DPYTHON_INCLUDE_DIRS="$PYTHON_INCLUDE_DIR"
   -DPYTHON_LIBRARY_DIRS="$PYTHON_LIB_DIR"
 )
 
 # Add Python library if found
 if [[ -n "$PYTHON_LIBRARY_PATH" ]]; then
   cmake_args+=(-DPython_LIBRARIES="$PYTHON_LIBRARY_PATH")
   cmake_args+=(-DPYTHON_LIBRARIES="$PYTHON_LIBRARY_PATH")
   echo "   Adding Python library: $PYTHON_LIBRARY_PATH"
 fi
 
 cmake "${cmake_args[@]}"
   
 if [[ $? -eq 0 ]]; then
   echo "✅ CMake configured successfully"
 else
   echo "❌ CMake configuration failed"
   exit 1
 fi
 echo

# -----------------------------------------------------------------------------
# Step 5: Build swigfaiss target (builds FAISS core and Python extension)
# -----------------------------------------------------------------------------
 echo "⚙️  Building FAISS + Python bindings..."
 echo "   Build target: swigfaiss"
 echo "   Parallel jobs: $NUM_CORES"
 echo "   Build directory: $BUILD_DIR"
 
 echo "   🚀 Starting build process..."
 start_time=$(date +%s)
 cmake --build "$BUILD_DIR" --target swigfaiss -j "$NUM_CORES"
 build_result=$?
 end_time=$(date +%s)
 build_duration=$((end_time - start_time))
 
 if [[ $build_result -eq 0 ]]; then
   echo "✅ Build finished successfully in ${build_duration}s"
 else
   echo "❌ Build failed after ${build_duration}s"
   exit 1
 fi
 echo

# -----------------------------------------------------------------------------
# Step 6: Install FAISS libraries to venv prefix
# -----------------------------------------------------------------------------
 echo "📦 Installing FAISS to: $VENV_PATH ..."
 echo "   Install prefix: $VENV_PATH"
 echo "   Build directory: $BUILD_DIR"
 
 echo "   🚀 Running cmake install..."
 cmake --install "$BUILD_DIR" --prefix "$VENV_PATH"
 install_result=$?
 
 if [[ $install_result -eq 0 ]]; then
   echo "   ✅ Libraries installed successfully"
   echo "   🔍 Checking installed files..."
   if [[ -d "$VENV_PATH/lib" ]]; then
     echo "     Found lib directory with: $(ls -1 "$VENV_PATH/lib" | wc -l) files"
   fi
   if [[ -d "$VENV_PATH/include" ]]; then
     echo "     Found include directory with: $(find "$VENV_PATH/include" -name '*.h' | wc -l) headers"
   fi
 else
   echo "   ❌ Installation failed"
   exit 1
 fi
 echo

# -----------------------------------------------------------------------------
# Step 7: Install Python package from build tree
# -----------------------------------------------------------------------------
 echo "🔍 Locating Python build directory..."
 PY_BUILD_PY_DIR="$BUILD_DIR/faiss/python"
 echo "   Expected path: $PY_BUILD_PY_DIR"
 if [[ ! -d "$PY_BUILD_PY_DIR" ]]; then
   echo "   ❌ Python build directory not found"
   echo "   🔍 Searching for alternative python directories..."
   find "$BUILD_DIR" -name "python" -type d 2>/dev/null || true
   echo "   ❌ Expected python build dir not found: $PY_BUILD_PY_DIR"
   exit 1
 fi
 echo "   ✅ Python build directory found"
 echo "   📂 Contents: $(ls -la "$PY_BUILD_PY_DIR" | wc -l) items"

echo "🐍 Installing Python package from: $PY_BUILD_PY_DIR"
echo "   Changing to build directory..."
pushd "$PY_BUILD_PY_DIR" >/dev/null
echo "   ✅ Now in: $(pwd)"

# Work around potential nested contrib trees copied from repo state
echo "   🔍 Checking for nested contrib directories..."
if [[ -d contrib/contrib ]]; then
  echo "   🔧 Fixing nested contrib: removing contrib/contrib"
  rm -rf contrib/contrib
  echo "   ✅ Nested contrib directory removed"
else
  echo "   ✅ No nested contrib directory found"
fi

# Patch setup.py to be tolerant if contrib copy fails
echo "   🔧 Patching setup.py for contrib robustness..."
if [[ -f setup.py ]]; then
  echo "   ✅ setup.py found, applying patch"
else
  echo "   ❌ setup.py not found in build directory"
  echo "   📂 Files in current directory:"
  ls -la
fi

"$PYTHON_EXECUTABLE" - <<'PY'
from pathlib import Path
p = Path('setup.py')
if p.exists():
    print(f"   Reading setup.py ({p.stat().st_size} bytes)")
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
        print("   ✅ setup.py patched successfully")
    else:
        print("   ⚠️  setup.py patch not needed (pattern not found)")
else:
    print("   ❌ setup.py not found for patching")
PY
echo "   ✅ setup.py patch completed"

# Ensure a modern PEP 517 build (standard frontend via `python -m build`)
echo "   🔍 Checking for pyproject.toml..."
if [[ ! -f pyproject.toml ]]; then
  echo "   📝 Creating pyproject.toml for PEP 517 build..."
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
  echo "   ✅ pyproject.toml created"
else
  echo "   ✅ pyproject.toml already exists"
  echo "   📋 Contents:"
  cat pyproject.toml | sed 's/^/     /'
fi

# Clean previous artifacts and build a wheel using PEP 517 (isolated)
echo "   🧹 Cleaning previous build artifacts..."
echo "     Removing: dist/ build/ *.egg-info"
rm -rf dist build *.egg-info
echo "   ✅ Build artifacts cleaned"

echo "   🏗️  Building wheel with PEP 517..."
start_time=$(date +%s)
python -m build --wheel
build_result=$?
end_time=$(date +%s)
build_duration=$((end_time - start_time))

if [[ $build_result -eq 0 ]]; then
  echo "   ✅ Wheel built successfully in ${build_duration}s"
else
  echo "   ❌ Wheel build failed after ${build_duration}s"
  exit 1
fi

# Install the built wheel into the venv
echo "   🔍 Looking for built wheel..."
if [[ -d dist ]]; then
  echo "     dist/ directory contents:"
  ls -la dist/ | sed 's/^/       /'
else
  echo "     ❌ dist/ directory not found"
fi

built_wheel=$(ls -1 dist/*.whl 2>/dev/null | head -n 1)
if [[ -z "${built_wheel:-}" ]]; then
  echo "   ❌ Wheel build failed: no wheel found in dist/"
  exit 1
fi
echo "   ✅ Found built wheel: $built_wheel"
wheel_size=$(ls -lh "$built_wheel" | awk '{print $5}')
echo "     Wheel size: $wheel_size"

echo "   📦 Installing wheel into venv..."
python -m pip install --no-deps "$built_wheel"
install_result=$?
if [[ $install_result -eq 0 ]]; then
  echo "   ✅ Wheel installed successfully"
else
  echo "   ❌ Wheel installation failed"
  exit 1
fi

echo "   🔙 Returning to original directory..."
popd >/dev/null
echo "   ✅ Returned to: $(pwd)"
echo "✅ Python package installed successfully"
echo

# -----------------------------------------------------------------------------
# Step 8: Quick validation
# -----------------------------------------------------------------------------
 echo "🧪 Validating installation..."
 echo "   Python executable: $PYTHON_EXECUTABLE"
 echo "   🐍 Running import test..."
 "$PYTHON_EXECUTABLE" - <<'PY'
import sys
print(f"   Python: {sys.version}")
print(f"   Platform: {sys.platform}")
print(f"   Executable: {sys.executable}")

try:
    import faiss
    print(f"   ✅ faiss module: {faiss.__file__}")
    print(f"   ✅ faiss version: {getattr(faiss, '__version__', 'n/a')}")
    
    # Quick functionality test
    import numpy as np
    d = 64
    nb = 1000
    x = np.random.random((nb, d)).astype('float32')
    index = faiss.IndexFlatL2(d)
    index.add(x)
    print(f"   ✅ Basic functionality test passed (indexed {index.ntotal} vectors)")
    
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Functionality test failed: {e}")
    sys.exit(1)
PY
 import_result=$?
 if [[ $import_result -eq 0 ]]; then
   echo "✅ Import and functionality test passed"
 else
   echo "❌ Import or functionality test failed"
   exit 1
 fi
 echo

 echo "🎉 BUILD COMPLETE (CPU-only)"
 echo
 echo "📊 Build Summary:"
 echo "   Source directory: $SRC_DIR"
 echo "   Build directory: $BUILD_DIR"
 echo "   Install prefix: $VENV_PATH"
 echo "   Python version: $PY_VER"
 echo "   Python executable: $PYTHON_EXECUTABLE"
 echo
 echo "🚀 Usage Instructions:"
 if [[ "$VENV_PATH" != "$VIRTUAL_ENV" ]]; then
   echo "   Activate with: source \"$VENV_PATH/bin/activate\""
 else
   echo "   ✅ Already using the correct virtual environment"
 fi
 echo "   Test import:  python -c 'import faiss, numpy as np; print(faiss.__file__)'"
 echo "   Quick test:   python -c 'import faiss; print(f\"FAISS version: {getattr(faiss, \"__version__\", \"n/a\")}\")'"

 # Notes:
 # - If you see runtime errors about libomp.dylib, ensure your shell has:
 #     export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$(brew --prefix openblas)/lib:$DYLD_LIBRARY_PATH"
 #   or ensure your environment does not strip rpaths.
