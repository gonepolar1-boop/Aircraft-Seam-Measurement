#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$ROOT_DIR"

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
  local script_name="$1"
  echo "==> Running $script_name"
  "$PYTHON_BIN" "$CODE_DIR/$script_name"
}

run_step "generate_data.py"
run_step "train_model.py"
run_step "predict_masks.py"
run_step "analyze_seam_mask.py"
run_step "evaluate_width.py"

echo "All scripts finished successfully."
