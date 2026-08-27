#!/usr/bin/env python3
"""実機のLowStateを購読して表示するだけのツール。**何も送らない。**

コックピットを立ち上げる前の配線・IMU・関節の健全性確認に使う。
rt/lowcmd への publish も MotionSwitcher の ReleaseMode も行わないので、
ロボットは今のモード(リモコンのdamp等)のまま一切動かない。

  python3 real/listen_only.py --iface enp46s0
  python3 real/listen_only.py --iface enp46s0 --sec 10 --save   # npzに保存

見るところ:
  受信Hz      500前後。低い/途切れるならLAN・NIC設定を疑う
  傾き        立位なら数度。ここが大きいとコックピットが即DAMPする
  関節速度    静止しているのに大きいならノイズ過大
  温度        高いモータがあれば無理をさせない
"""
import argparse
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Listener:
    def __init__(self, iface):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        if iface:
            ChannelFactoryInitialize(0, iface)
        else:
            ChannelFactoryInitialize(0)
        self.lock = threading.Lock()
        self.n = 0
        self.t0 = None
        self.last = None
        self.buf = []
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on, 10)

    def _on(self, msg):
        q = np.array([msg.motor_state[i].q for i in range(29)])
        dq = np.array([msg.motor_state[i].dq for i in range(29)])
        tau = np.array([msg.motor_state[i].tau_est for i in range(29)])
        tp = np.array([msg.motor_state[i].temperature[0] for i in range(29)])
        quat = np.array(list(msg.imu_state.quaternion))
        gyro = np.array(list(msg.imu_state.gyroscope))
        with self.lock:
            self.n += 1
            if self.t0 is None:
                self.t0 = time.time()
            self.last = dict(q=q, dq=dq, tau=tau, temp=tp, quat=quat, gyro=gyro,
                             mode_machine=int(msg.mode_machine))
            self.buf.append(np.concatenate([[time.time() - self.t0], q, dq, tau,
                                            tp, quat, gyro]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp46s0")
    ap.add_argument("--sec", type=float, default=8.0)
    ap.add_argument("--save", action="store_true", help="npzに保存")
    a = ap.parse_args()

    print(f"購読のみ(送信しません): iface={a.iface}")
    L = Listener(a.iface)
    t0 = time.time()
    while time.time() - t0 < 5.0 and L.last is None:
        time.sleep(0.1)
    if L.last is None:
        print("★LowStateが来ない。LANケーブル・NICのIP・電源を確認")
        return 1

    print(f"{'経過':>5s} {'受信Hz':>7s} {'傾き':>7s} {'|dq|max':>8s} "
          f"{'|tau|max':>9s} {'温度max':>8s}")
    t0 = time.time()
    while time.time() - t0 < a.sec:
        time.sleep(1.0)
        with L.lock:
            n, d = L.n, dict(L.last)
        hz = n / max(1e-9, time.time() - L.t0)
        up = float(quat_to_mat(d["quat"])[2, 2])
        tilt = np.degrees(np.arccos(min(1.0, max(-1.0, up))))
        print(f"{time.time()-t0:5.1f} {hz:7.0f} {tilt:6.1f}度 "
              f"{np.abs(d['dq']).max():8.2f} {np.abs(d['tau']).max():9.1f} "
              f"{d['temp'].max():7.0f}度")

    with L.lock:
        d, buf = dict(L.last), np.array(L.buf)
    print(f"\nmode_machine = {d['mode_machine']}  (LowCmdにそのまま返す値)")
    hot = np.argsort(-d["temp"])[:3]
    print("温度上位: " + " / ".join(f"j{i} {d['temp'][i]:.0f}度" for i in hot))
    print("関節角(rad):")
    for k, lab in ((slice(0, 6), "左脚"), (slice(6, 12), "右脚"),
                   (slice(12, 15), "腰 "), (slice(15, 22), "左腕"),
                   (slice(22, 29), "右腕")):
        print(f"  {lab} " + " ".join(f"{v:+6.3f}" for v in d["q"][k]))
    if a.save:
        out = ROOT / "logs" / "real" / time.strftime("listen_%Y%m%d_%H%M%S.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, rec=buf.astype(np.float32),
                            cols=np.array(["t"] + [f"q{i}" for i in range(29)]
                                          + [f"dq{i}" for i in range(29)]
                                          + [f"tau{i}" for i in range(29)]
                                          + [f"temp{i}" for i in range(29)]
                                          + ["quat_w", "quat_x", "quat_y", "quat_z"]
                                          + ["gyro_x", "gyro_y", "gyro_z"]))
        print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
