#!/usr/bin/env bash
# CometSpark-Exp 训练脚本
# 用法: ./scripts/train.sh [--steps 30] [其他 run.py train 参数]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 run.py train "$@"
