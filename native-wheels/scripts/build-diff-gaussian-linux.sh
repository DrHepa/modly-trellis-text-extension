#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
OUT_DIR="native-wheels/wheelhouse"
WORK_DIR="native-wheels/work/diff-gaussian-linux"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
CUDA_ROOT="${CUDA_PATH:-${CUDA_HOME:-}}"
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9;9.0+PTX"
REPO_URL="https://github.com/autonomousvision/mip-splatting.git"
REPO_REF="dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"
PACKAGE_SUBDIR="submodules/diff-gaussian-rasterization"
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --torch-version) TORCH_VERSION="$2"; shift 2 ;;
    --torchvision-version) TORCHVISION_VERSION="$2"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="$2"; shift 2 ;;
    --cuda-root) CUDA_ROOT="$2"; shift 2 ;;
    --torch-cuda-arch-list) TORCH_CUDA_ARCH_LIST="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CUDA_ROOT" ]]; then
  echo "CUDA Toolkit root was not resolved. Pass --cuda-root or set CUDA_PATH/CUDA_HOME." >&2
  exit 1
fi

OUT_DIR="$(realpath -m "$OUT_DIR")"
WORK_DIR="$(realpath -m "$WORK_DIR")"
VENV_DIR="$WORK_DIR/venv"
SRC_DIR="$WORK_DIR/src"
REPO_DIR="$SRC_DIR/mip-splatting"
PACKAGE_DIR="$REPO_DIR/$PACKAGE_SUBDIR"

if [[ "$CLEAN" == "1" && -d "$WORK_DIR" ]]; then
  rm -rf "$WORK_DIR"
fi

mkdir -p "$OUT_DIR" "$SRC_DIR"

export CUDA_HOME="$CUDA_ROOT"
export CUDA_PATH="$CUDA_ROOT"
export CUDACXX="$CUDA_ROOT/bin/nvcc"
export TORCH_CUDA_ARCH_LIST
export PATH="$CUDA_ROOT/bin:${PATH}"
export CPATH="$CUDA_ROOT/include${CPATH:+:${CPATH}}"

LIB_ENTRIES=()
if [[ -d "$CUDA_ROOT/lib" ]]; then
  LIB_ENTRIES+=("$CUDA_ROOT/lib")
fi
if [[ -d "$CUDA_ROOT/lib64" ]]; then
  LIB_ENTRIES+=("$CUDA_ROOT/lib64")
fi
if [[ -d "$CUDA_ROOT/targets/x86_64-linux/lib" ]]; then
  LIB_ENTRIES+=("$CUDA_ROOT/targets/x86_64-linux/lib")
fi
if [[ ${#LIB_ENTRIES[@]} -gt 0 ]]; then
  LIB_JOINED="$(IFS=:; printf '%s' "${LIB_ENTRIES[*]}")"
  export LIBRARY_PATH="$LIB_JOINED${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="$LIB_JOINED${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[native-wheels] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

"$PYTHON" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel ninja
"$VENV_PYTHON" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"

if [[ -d "$REPO_DIR" ]]; then
  rm -rf "$REPO_DIR"
fi

git clone "$REPO_URL" "$REPO_DIR"
git -C "$REPO_DIR" checkout "$REPO_REF"
git -C "$REPO_DIR" submodule update --init --recursive

if [[ ! -d "$PACKAGE_DIR/third_party/glm" ]]; then
  echo "GLM submodule was not populated under diff-gaussian-rasterization/third_party/glm." >&2
  exit 1
fi

"$VENV_PYTHON" -m pip wheel "$PACKAGE_DIR" --no-build-isolation -w "$OUT_DIR"

echo "[native-wheels] diff_gaussian_rasterization wheel build complete: $OUT_DIR"
