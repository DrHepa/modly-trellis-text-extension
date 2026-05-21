#!/usr/bin/env bash
set -euo pipefail

PYTHON="python3"
WHEEL_DIR="native-wheels/wheelhouse"
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
KEEP_VENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --wheel-dir) WHEEL_DIR="$2"; shift 2 ;;
    --torch-version) TORCH_VERSION="$2"; shift 2 ;;
    --torchvision-version) TORCHVISION_VERSION="$2"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="$2"; shift 2 ;;
    --keep-venv) KEEP_VENV=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

WHEEL_DIR="$(realpath -m "$WHEEL_DIR")"
if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "Wheel directory not found: $WHEEL_DIR" >&2
  exit 1
fi

shopt -s nullglob
NVDIFFRAST_WHEELS=("$WHEEL_DIR"/nvdiffrast-*.whl)
DIFF_GAUSSIAN_WHEELS=("$WHEEL_DIR"/diff_gaussian_rasterization-*.whl)
shopt -u nullglob

if [[ ${#NVDIFFRAST_WHEELS[@]} -eq 0 ]]; then
  echo "No nvdiffrast wheel found in wheelhouse." >&2
  exit 1
fi
if [[ ${#DIFF_GAUSSIAN_WHEELS[@]} -eq 0 ]]; then
  echo "No diff_gaussian_rasterization wheel found in wheelhouse." >&2
  exit 1
fi

TEMP_ROOT="$(mktemp -d -t modly-trellis-text-native-smoke-XXXXXX)"
VENV_DIR="$TEMP_ROOT/venv"

cleanup() {
  if [[ "$KEEP_VENV" != "1" && -d "$TEMP_ROOT" ]]; then
    rm -rf "$TEMP_ROOT"
  fi
}
trap cleanup EXIT

"$PYTHON" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$TORCH_INDEX_URL"
"$VENV_PYTHON" -m pip install --no-deps "${NVDIFFRAST_WHEELS[0]}" "${DIFF_GAUSSIAN_WHEELS[0]}"
"$VENV_PYTHON" -c "import torch; import torchvision; import nvdiffrast.torch; import diff_gaussian_rasterization; print('native smoke OK', torch.__version__, torchvision.__version__)"

echo "[native-wheels] Smoke test passed."
