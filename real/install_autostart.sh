#!/bin/bash
# 機体(Jetson)で実行: LiDAR ブリッジとコックピットを systemd に登録して自動起動にする。
#   ssh unitree@192.168.123.164 'bash ~/cl-workspace/g1_cockpit/real/install_autostart.sh'
# 取り消し: bash install_autostart.sh --remove
set -u
PW=123
HERE=$(cd "$(dirname "$0")" && pwd)
S() { echo "$PW" | sudo -S -p "" "$@"; }
if [ "${1:-}" = "--remove" ]; then
  S systemctl disable --now g1-cockpit.service g1-lidar-bridge.service 2>/dev/null
  S rm -f /etc/systemd/system/g1-cockpit.service /etc/systemd/system/g1-lidar-bridge.service
  S systemctl daemon-reload
  echo "自動起動を外しました(手動は ~/cl-workspace/start_onboard.sh)"
  exit 0
fi
ST=$(curl -s -m 3 http://127.0.0.1:8090/state 2>/dev/null | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); print(d.get("fsm","?"))
except Exception:
    print("?")' 2>/dev/null)
case "$ST" in
  ""|"?"|IDLE|DAMP|STD:*|WALK:ready) ;;
  *) echo "★コックピットが走行中です(状態 $ST)。いま再起動すると機体が倒れます"; exit 2 ;;
esac
for u in g1-lidar-bridge g1-cockpit; do
  [ -f "$HERE/systemd/$u.service" ] || { echo "★ $HERE/systemd/$u.service が無い"; exit 1; }
  S install -m 644 "$HERE/systemd/$u.service" "/etc/systemd/system/$u.service"
done
S systemctl daemon-reload
S systemctl enable g1-lidar-bridge.service g1-cockpit.service
# 手動起動(nohup)の取り残しを止めてからサービスで立て直す
pkill -f "lidar_bridge\.py" 2>/dev/null; pkill -f "python3 -u cockpit\.py" 2>/dev/null; sleep 1
S systemctl restart g1-lidar-bridge.service
S systemctl restart g1-cockpit.service
echo "--- 登録状態"
systemctl is-enabled g1-lidar-bridge.service g1-cockpit.service
for i in $(seq 1 24); do
  sleep 5
  if ss -ltn 2>/dev/null | grep -q :8090; then echo "待受OK ($((i*5))秒)"; break; fi
done
systemctl --no-pager --lines=0 status g1-lidar-bridge.service g1-cockpit.service | grep -E "●|Active:"
echo "  無線: http://192.168.179.100:8090   有線: http://192.168.123.164:8090"
