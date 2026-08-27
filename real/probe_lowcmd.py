#!/usr/bin/env python3
"""`rt/lowcmd` を購読して、誰がどれだけ指令を出しているかを見る。**送信しない。**

目的: `ReleaseMode()` が本当に効いたかを、CheckModeではなく**実測**で判定する。

  - 内蔵制御が動いている間: 400Hz前後で kp/kd が流れる(REAL_MODES.md)
  - 解放されている間      : **0 Hz**(誰も出さない)
  - こちらのコックピットが送信中: 500Hz で kp が方策の値(100/150/200等)

これを見れば「解放したつもりで内蔵制御がまだ出している(=モータの取り合い)」
を検出できる。2026-08-20 の崩落(偏差1.4radに対しトルク3Nm)は、この取り合いが
原因と考えられる。

  python3 real/probe_lowcmd.py --iface enp46s0          # Ctrl-Cまで
  python3 real/probe_lowcmd.py --iface enp46s0 --sec 30
"""
import argparse
import sys
import threading
import time

import numpy as np


class Probe:
    def __init__(self, iface):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        ChannelFactoryInitialize(0, iface) if iface else ChannelFactoryInitialize(0)
        self.lock = threading.Lock()
        self.win = []
        self.sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub.Init(self._on, 10)

    def _on(self, msg):
        kp = np.array([msg.motor_cmd[i].kp for i in range(29)])
        kd = np.array([msg.motor_cmd[i].kd for i in range(29)])
        with self.lock:
            self.win.append((float(kp.max()), float(kd.max()),
                             float(kp[3]), float(kd[3]),      # 左膝
                             int(msg.mode_machine)))

    def tick(self):
        with self.lock:
            w, self.win = self.win, []
        if not w:
            return dict(hz=0, kinds=[], mm=None)
        a = np.array([x[:4] for x in w])
        # kp最大の値で「誰の指令か」をざっくり分ける
        kinds = {}
        for kpmax, kdmax, kknee, kdknee, mm in w:
            key = (round(kpmax), round(kdmax, 1))
            kinds[key] = kinds.get(key, 0) + 1
        return dict(hz=len(w), kinds=sorted(kinds.items(), key=lambda x: -x[1]),
                    mm=w[-1][4], knee_kp=a[:, 2].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp46s0")
    ap.add_argument("--sec", type=float, default=0.0)
    a = ap.parse_args()
    print("rt/lowcmd を購読(送信しません)")
    p = Probe(a.iface)
    time.sleep(1.0)
    print(f"{'経過':>5s} {'送信Hz':>7s} {'mode_machine':>13s}  "
          f"kp最大/kd最大 の内訳(回数)")
    t0 = time.time()
    try:
        while a.sec <= 0 or time.time() - t0 < a.sec:
            time.sleep(1.0)
            s = p.tick()
            if s["hz"] == 0:
                print(f"{time.time()-t0:5.0f} {0:6d}Hz {'-':>13s}  "
                      f"★誰も指令を出していない(=解放中)")
                continue
            det = "  ".join(f"kp{k[0]}/kd{k[1]}×{n}" for k, n in s["kinds"][:3])
            note = ""
            if len(s["kinds"]) >= 2:
                note = "  ★2種類の指令が混在=モータの取り合い"
            print(f"{time.time()-t0:5.0f} {s['hz']:6d}Hz {s['mm']:>13d}  "
                  f"{det}{note}")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
