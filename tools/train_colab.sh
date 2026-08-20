#!/usr/bin/env bash
set -euo pipefail
export SNUNET_DATA_ROOT="/content/LEVIRCD_256"
export SNUNET_OUTPUT_ROOT="/content/drive/MyDrive/SNUNet/LEVIRCD"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
python -u train.py 2>&1 | tee -a "$SNUNET_OUTPUT_ROOT/train.log"
