#!/usr/bin/env python3
"""実機のLowStateを常時監視して記録する。**送信は一切しない。**

コックピット本体の記録は方策/補間の実行中しか動かないので、標準モード
(ゼロトルク/ダンプ/立つ/座る)の最中は時系列が残らない。このツールは別
プロセスで購読だけを行い、モードに関係なく全部残す。同時に走らせてよい
(DDSは複数購読できる。lowcmd は publish しない)。

  python3 real/watch_live.py                 # Ctrl-C まで監視
  python3 real/watch_live.py --sec 600       # 10分で終了

1秒ごとに1行出し、異常があれば ★ を付ける。npzは5秒ごとに追記保存する
ので、強制終了しても残る。
"""
import argparse
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "real"))

from log_view import rated_torque              # noqa: E402

JOINTS = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
          "left_ankle_pitch", "left_ankle_roll",
          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
          "right_ankle_pitch", "right_ankle_roll",
          "waist_yaw", "waist_roll", "waist_pitch",
          "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
          "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
          "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
          "right_elbow", "right_wrist_roll", "right_wrist_pitch",
          "right_wrist_yaw"]
RATED = rated_torque(JOINTS)

# 注意を促す閾値(止める閾値ではない。これは監視専用で何も送らない)
WARN_TILT = 25.0
WARN_DQ = 8.0
WARN_RATIO = 1.0
WARN_TEMP = 65.0


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Watcher:
    def __init__(self, iface, hz=100):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, iface) if iface else ChannelFactoryInitialize(0)
        self.lock = threading.Lock()
        self.buf = []            # 保存用(間引き)
        self.win = []            # 直近1秒の統計用(全コマ)
        self.t0 = None
        self.n = 0
        self.min_dt = 1.0 / hz
        self._last_keep = 0.0
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on, 10)

    def _on(self, msg):
        now = time.time()
        q = np.array([msg.motor_state[i].q for i in range(29)])
        dq = np.array([msg.motor_state[i].dq for i in range(29)])
        tau = np.array([msg.motor_state[i].tau_est for i in range(29)])
        tp = np.array([msg.motor_state[i].temperature[0] for i in range(29)])
        quat = np.array(list(msg.imu_state.quaternion))
        gyro = np.array(list(msg.imu_state.gyroscope))
        up = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up)))))
        ratio = float((np.abs(tau) / RATED).max())
        with self.lock:
            self.n += 1
            if self.t0 is None:
                self.t0 = now
            self.win.append((tilt, float(np.abs(dq).max()), ratio,
                             float(tp.max()), int(np.argmax(np.abs(tau) / RATED))))
            if now - self._last_keep >= self.min_dt:
                self._last_keep = now
                self.buf.append(np.concatenate([[now - self.t0, tilt],
                                                q, dq, tau, tp, quat, gyro]))

    def tick(self):
        with self.lock:
            w, n = self.win, self.n
            self.win = []
        if not w:
            return None
        a = np.array([x[:4] for x in w], dtype=float)
        hot = int(np.bincount([x[4] for x in w], minlength=29).argmax())
        return dict(n=n, hz=len(w), tilt=a[:, 0].max(), dq=a[:, 1].max(),
                    ratio=a[:, 2].max(), temp=a[:, 3].max(), hot=hot)

    def save(self, path):
        with self.lock:
            if not self.buf:
                return 0
            rec = np.array(self.buf, dtype=np.float32)
        cols = (["t", "tilt_deg"] + [f"q{i}" for i in range(29)]
                + [f"dq{i}" for i in range(29)] + [f"tau{i}" for i in range(29)]
                + [f"temp{i}" for i in range(29)]
                + ["quat_w", "quat_x", "quat_y", "quat_z"]
                + ["gyro_x", "gyro_y", "gyro_z"])
        np.savez_compressed(path, rec=rec, cols=np.array(cols),
                            joints=np.array(JOINTS))
        return len(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp46s0")
    ap.add_argument("--sec", type=float, default=0.0, help="0=Ctrl-Cまで")
    a = ap.parse_args()

    out_dir = ROOT / "logs" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / time.strftime("watch_%Y%m%d_%H%M%S.npz")
    print(f"監視のみ(送信しません)  iface={a.iface}")
    print(f"保存先: {out.name}  (5秒ごとに書き出し)")
    w = Watcher(a.iface)
    t0 = time.time()
    while w.t0 is None and time.time() - t0 < 5:
        time.sleep(0.1)
    if w.t0 is None:
        print("★LowStateが来ない")
        return 1
    print(f"{'経過':>6s} {'受信':>6s} {'傾き':>7s} {'|dq|':>7s} "
          f"{'トルク比':>8s} {'温度':>6s}  最大トルクの関節")
    t0 = time.time()
    last_save = t0
    try:
        while a.sec <= 0 or time.time() - t0 < a.sec:
            time.sleep(1.0)
            s = w.tick()
            if s is None:
                print("  ★受信が止まっている")
                continue
            flag = ""
            if s["tilt"] > WARN_TILT:
                flag += " ★傾き"
            if s["dq"] > WARN_DQ:
                flag += " ★速度"
            if s["ratio"] > WARN_RATIO:
                flag += " ★トルク"
            if s["temp"] > WARN_TEMP:
                flag += " ★温度"
            print(f"{time.time()-t0:6.0f} {s['hz']:5d}Hz {s['tilt']:6.1f}度 "
                  f"{s['dq']:7.2f} {s['ratio']:8.2f} {s['temp']:5.0f}度  "
                  f"{JOINTS[s['hot']]}{flag}", flush=True)
            if time.time() - last_save > 5.0:
                last_save = time.time()
                w.save(out)
    except KeyboardInterrupt:
        pass
    finally:
        n = w.save(out)
        print(f"\n保存: {out}  ({n}コマ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
