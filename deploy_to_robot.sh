#!/usr/bin/env bash
# PC → 機体(Jetson)へコックピット一式を同期して、機体側で起動する。
#
#   ./deploy_to_robot.sh            # 有線(192.168.123.164)へ同期して起動
#   ./deploy_to_robot.sh wifi       # 無線(192.168.179.100)へ同期して起動
#   ./deploy_to_robot.sh wifi sync  # 同期だけ(起動しない)
#
# ★同期は有線を推奨(docs/オンボード運用.md §9: 無線のrsyncは実際に失敗する)。
# ★機体側のパスワード(123)は sshpass が無ければ手で入力する。
# ★logs/ は同期しない(機体の走行ログを上書きしない)。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOST="192.168.123.164"
[[ "${1:-}" == "wifi" ]] && HOST="192.168.179.100"
ONLY_SYNC="${2:-}"
DST="unitree@${HOST}:/home/unitree/cl-workspace/g1_cockpit/"

echo "=== 同期先 ${DST} ==="
ping -c 1 -W 2 "${HOST}" >/dev/null || { echo "★${HOST} に届きません(接続とIPを確認)"; exit 1; }
RS=(rsync -av --delete
    --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pyc'
    --exclude '.venv/' --exclude 'video/' --exclude 'MUJOCO_LOG.TXT')
if command -v sshpass >/dev/null 2>&1; then
  export SSHPASS=123
  RS+=(-e "sshpass -e ssh -o StrictHostKeyChecking=no")
fi
"${RS[@]}" "${HERE}/real" "${HERE}/deploy" "${HERE}/model" "${HERE}/motions" "${DST}"
# 起動スクリプトは cl-workspace 直下(docs/オンボード運用.md §0 の配置)
"${RS[@]}" "${HERE}/real/start_onboard.sh" "unitree@${HOST}:/home/unitree/cl-workspace/start_onboard.sh"
echo "=== 同期完了 ==="
[[ "${ONLY_SYNC}" == "sync" ]] && exit 0

echo "=== 機体側で起動(競合サービス停止 → cockpit.py → 待受確認) ==="
if command -v sshpass >/dev/null 2>&1; then
  sshpass -e ssh -o StrictHostKeyChecking=no "unitree@${HOST}" 'bash ~/cl-workspace/start_onboard.sh'
else
  ssh -o StrictHostKeyChecking=no "unitree@${HOST}" 'bash ~/cl-workspace/start_onboard.sh'
fi
echo ""
echo "Android: 機体と同じWi-Fi(SSID SPW_X11_fce0f9)に接続して http://192.168.179.100:8090 を開く"
