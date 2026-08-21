#!/usr/bin/env bash
set -euo pipefail
cd /content/SNUNet
PYTHONPATH=/content/SNUNet python -u train.py 2>&1 | tee -a /content/drive/MyDrive/SNUNet/SYSU/logs/train.log
