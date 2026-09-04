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
# ★自動起動(systemd)を入れた後は、電源投入で勝手に立ち上がる。このスクリプトは
#   コード更新後の「立て直し」に使う(サービスの restart に化ける)。
#
# ★lowcmd の二重送信はトルク異常の原因になる。競合源を必ず潰してから起動する。
#   詳細は docs/オンボード運用.md
set -u
WS=/home/unitree/cl-workspace/g1_cockpit
LOG=/home/unitree/cl-workspace/cockpit_onboard.log
PW=123

# ★2026-09-04: 自動起動(systemd)が入っていれば、サービスの再起動に置き換える。
#   (real/install_autostart.sh で g1-lidar-bridge.service / g1-cockpit.service を登録。
#    機体の電源投入で勝手に立ち上がる。手動の nohup 起動と二重にならないようにここで分岐)
# ★走行中(方策の実行・保持・制御権あり・自動歩行)に再起動すると lowcmd が途絶えて機体が倒れる。
#   コックピットが答えるなら状態を見て、IDLE/DAMP/内蔵制御/歩行待機 以外では止める(2026-09-04 午後)
ST=$(curl -s -m 3 http://127.0.0.1:8090/state 2>/dev/null | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d.get("fsm","?"))
except Exception:
    print("?")' 2>/dev/null)
case "$ST" in
  ""|"?"|IDLE|DAMP|STD:*|WALK:ready) ;;
  *) echo "★コックピットが走行中です(状態 $ST)。いま再起動すると機体が倒れます。ダンプかスタンドロックにしてから再実行してください"; exit 2 ;;
esac

if [ -f /etc/systemd/system/g1-cockpit.service ]; then
  echo "=== 自動起動(systemd)が入っているのでサービスを再起動します ==="
  pkill -f "lidar_bridge\.py" 2>/dev/null; pkill -f "python3 -u cockpit\.py" 2>/dev/null
  echo "$PW" | sudo -S -p "" systemctl restart g1-lidar-bridge.service g1-cockpit.service
  for i in $(seq 1 24); do
    sleep 5
    if ss -ltn 2>/dev/null | grep -q :8090; then
      echo "  待受OK ($((i*5))秒)"
      systemctl --no-pager --lines=0 status g1-lidar-bridge.service g1-cockpit.service | grep -E "Active:"
      echo ""
      echo "  無線: http://192.168.179.100:8090"
      echo "  有線: http://192.168.123.164:8090"
      exit 0
    fi
  done
  echo "  ★待受しません。journalctl -u g1-cockpit -n 50 / $LOG を確認してください"
  tail -15 "$LOG"
  exit 1
fi

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

echo "=== 1b. LiDAR ブリッジ(Livox SDK2 直結 → /dev/shm) ==="
# この FW では Unitree の rt/utlidar/* が配信されないので、機体内の Livox SDK2 を
# 直接叩いて点群を /dev/shm/g1_lidar.npy へ書く(real/lidar_bridge.py)。
# コックピットは DDS と /dev/shm の両方を見る。
if pgrep -f "lidar_bridge\.py" >/dev/null 2>&1; then
  pkill -f "lidar_bridge\.py" 2>/dev/null; sleep 1
fi
cd "$WS/real" || exit 1
nohup python3 -u lidar_bridge.py > /home/unitree/cl-workspace/lidar_bridge.log 2>&1 &
echo "  lidar_bridge.py を起動しました(ログ: ~/cl-workspace/lidar_bridge.log)"

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
