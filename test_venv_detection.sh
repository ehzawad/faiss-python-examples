#!/usr/bin/env bash

# Simple test script to demonstrate virtual environment detection logic
# This mimics the key parts of build_metal_faiss_m4.sh for testing

echo "🧪 Testing Virtual Environment Detection"
echo "========================================"

# Simulate the detection logic from the updated script
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  VENV_PATH="${1:-$VIRTUAL_ENV}"
  PYTHON_CMD="${2:-python}"
  echo "🔍 Detected active virtual environment: $VIRTUAL_ENV"
  echo "   Will use: $VENV_PATH"
  echo "   Python command: $PYTHON_CMD"
  
  # Test if we can find the python executable
  PYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python"
  if [[ -f "$PYTHON_EXECUTABLE" ]]; then
    echo "✅ Python executable found: $PYTHON_EXECUTABLE"
    PY_VER="$($PYTHON_EXECUTABLE -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "   Python version: $PY_VER"
  else
    echo "❌ Python executable not found at: $PYTHON_EXECUTABLE"
  fi
  
  echo "🐍 Would use current active virtual environment (no new venv creation)"
else
  VENV_PATH="${1:-$HOME/venv-faiss}"
  PYTHON_CMD="${2:-python3}"
  echo "❌ No active virtual environment detected"
  echo "   Would create new venv at: $VENV_PATH"
  echo "   Python command: $PYTHON_CMD"
  echo "🐍 Would create new virtual environment"
fi

echo
echo "Summary:"
echo "- VIRTUAL_ENV: ${VIRTUAL_ENV:-'(not set)'}"
echo "- VENV_PATH: $VENV_PATH"
echo "- PYTHON_CMD: $PYTHON_CMD"
echo
echo "Run this script with and without an active venv to see the difference!"