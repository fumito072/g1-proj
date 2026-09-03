#!/bin/bash
# G1 オンボード演算コックピット 起動スクリプト
#
# 機体(Jetson)の上でコックピットを走らせ、制御・推論・DDSをすべて機体内で
# 完結させる。ネットワークを渡るのはUIの画面だけなので、通信が切れても
# 制御は止まらない。
#
# 配置: 機体の ~/cl-workspace/start_onboard.sh
#   ~/cl-workspace/g1_cockpit/ 以下に real/ deploy/ model/ motions/ を置く
# 使い方(PCから):
#   ssh unitree@192.168.179.100 '~/cl-workspace/start_onboard.sh'
#
# ★lowcmd の二重送信はトルク異常の原因になる。競合源を必ず潰してから起動する。
#   詳細は docs/オンボード運用.md
set -u
WS=/home/unitree/cl-workspace/g1_cockpit
LOG=/home/unitree/cl-workspace/cockpit_onboard.log
PW=123

echo "=== 1. 競合するlowcmd送信源を停止 ==="
# 大林デモ(rt/lowcmd を出せるウェブサーバ)。enabled のままなので機体を
# 再起動するたびに復活する → 毎回止める。disable はしない(先方のサービス)。
if systemctl is-active --quiet 3f_demo.service; then
  echo "$PW" | sudo -S systemctl stop 3f_demo.service 2>/dev/null
  echo "  3f_demo.service を停止しました"
else
  echo "  3f_demo.service は停止済み"
fi
if pgrep -f "cockpit\.py" >/dev/null 2>&1; then
  pkill -f "cockpit\.py" 2>/dev/null; sleep 2
  echo "  既存のコックピットを停止しました"
fi
echo "  port5000: $(ss -ltn 2>/dev/null | grep -c :5000) 件 (0なら清浄)"

echo "=== 2. オンボード起動 ==="
cd "$WS/real" || exit 1
ulimit -n 8192
# ★--iface は付けない。省略すると ChannelFactoryInitialize(0) になり、
#   機体内ループバックDDS(実測1043Hz)を掴む。空文字 "" を明示的に渡すと
#   channel factory init error になるので注意。
# ★MUJOCO_GL=disable が必須。機体にGL環境が無く、EGLの初期化で落ちる。
#   MuJoCoはFK計算にしか使っていないので描画は要らない。
MUJOCO_GL=disable nohup python3 -u cockpit.py --port 8090 --host 0.0.0.0 > "$LOG" 2>&1 &
echo "  起動しました (import torch に約45秒かかります)"

echo "=== 3. 待受を待つ ==="
for i in $(seq 1 20); do
  sleep 5
  if ss -ltn 2>/dev/null | grep -q :8090; then
    echo "  待受OK ($((i*5))秒)"
    echo ""
    echo "  無線: http://192.168.179.100:8090"
    echo "  有線: http://192.168.123.164:8090"
    exit 0
  fi
done
echo "  ★待受しません。ログを確認してください: $LOG"
tail -15 "$LOG"
exit 1
