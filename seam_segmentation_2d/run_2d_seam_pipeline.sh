#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/python.exe" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/python.exe"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: python/python3 not found in PATH."
  exit 1
fi

echo "Using Python: $PYTHON_BIN"

run_step() {
  local module_name="$1"
  echo "==> Running $module_name"
  (cd "$ROOT_DIR" && "$PYTHON_BIN" -m "$module_name")
}

run_step "seam_segmentation_2d.generate_data"
run_step "seam_segmentation_2d.train_model"
run_step "seam_segmentation_2d.predict_masks"
run_step "seam_segmentation_2d.analyze_seam_mask"
run_step "seam_segmentation_2d.evaluate_width"

echo "All scripts finished successfully."
