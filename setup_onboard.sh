#!/usr/bin/env bash
set -euo pipefail
# G1 機体側(Jetson / Ubuntu / ARM64)でコックピットを動かすためのセットアップ。
# 使い方: bash setup_onboard.sh
cd "$(dirname "$0")"

PY=python3
"$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# ベース依存(numpy/mujoco/unitree_sdk2py/cyclonedds)
pip install -r requirements_jetpack.txt

# Jetson 用 torch は NVIDIA の index から。JetPack 6 を既定とし、
# 入らなければ JetPack 5 を試す。
TORCH_IDX6="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/"
TORCH_IDX5="https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/"
pip install torch --extra-index-url "$TORCH_IDX6" \
  || pip install torch --extra-index-url "$TORCH_IDX5"

# import 確認
"$PY" - <<'PY'
import numpy, mujoco, torch
import unitree_sdk2py  # noqa: F401
print("numpy", numpy.__version__, "/ torch", torch.__version__,
      "/ mujoco", mujoco.__version__)
PY
echo "完了: source .venv/bin/activate して python3 real/cockpit.py"
