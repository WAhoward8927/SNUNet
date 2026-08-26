set -euo pipefail
cd /content/SNUNet
mkdir -p /content/drive/MyDrive/SNUNet/SYSU/checkpoints /content/drive/MyDrive/SNUNet/SYSU/logs
PYTHONPATH=/content/SNUNet python -u train.py 2>&1 | tee -a /content/drive/MyDrive/SNUNet/SYSU/logs/train.log
