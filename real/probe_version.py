#!/usr/bin/env python3
"""実機の素性と、コックピットが前提にしている値を**読むだけ**で確認する。

  python3 real/probe_version.py --iface enp46s0

**1バイトも送信しない。**LowStateの購読と、GET系RPCだけ。
ロボットは damp でも stand でも、どの状態でも安全に実行できる。

ファームウェアを更新したら、方策を動かす前に必ずこれを通すこと。
コックピットは以下を「実測で決めた値」として前提にしており、
ファーム更新はこの前提を変えうる(実機セッション手順 付録C)。

  - LowState.mode_machine(実測5)を LowCmd にそのまま返す
  - mode_pr = 0(PR)
  - FSM id: 0=ゼロトルク 1=ダンプ 4=立ち上がり 200=運用制御
"""
import argparse
import json
import sys
import threading
import time

import numpy as np

EXPECT = {"mode_machine": 5, "mode_pr": 0}
JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
LOCO_GETS = [(7001, "FSM ID"), (7002, "FSM MODE"), (7003, "バランスモード"),
             (7004, "スイング高さ"), (7005, "立ち高さ")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp46s0")
    ap.add_argument("--sec", type=float, default=3.0, help="観測する秒数")
    a = ap.parse_args()

    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    print("=" * 66)
    print("実機の素性チェック(★送信しません。読み取りのみ)")
    print("=" * 66)

    try:
        import unitree_sdk2py
        v = getattr(unitree_sdk2py, "__version__", "(不明)")
    except Exception:                              # noqa: BLE001
        v = "(不明)"
    print(f"PC側 unitree_sdk2py = {v}")

    ChannelFactoryInitialize(0, a.iface) if a.iface else ChannelFactoryInitialize(0)
    box = {"msg": None, "n": 0, "t0": 0.0, "t1": 0.0}
    lock = threading.Lock()

    def on_state(msg):
        with lock:
            box["msg"] = msg
            box["n"] += 1
            if box["t0"] == 0.0:
                box["t0"] = time.time()
            box["t1"] = time.time()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    t0 = time.time()
    while box["msg"] is None:
        if time.time() - t0 > 5.0:
            print("★LowStateが5秒来ない。配線とIP設定を確認")
            return 1
        time.sleep(0.05)
    time.sleep(a.sec)

    with lock:
        m = box["msg"]
        hz = (box["n"] - 1) / max(box["t1"] - box["t0"], 1e-9)

    print(f"\n--- LowState(受信 {hz:.0f} Hz / {box['n']}件) " + "-" * 22)
    ver = list(m.version)
    print(f"  version        = {ver}  "
          f"(生値。Unitreeは詳細な公開チェンジログを出していない。"
          f"更新前後でこの値を控えること)")
    print(f"  tick           = {m.tick}")
    for k, got in (("mode_machine", int(m.mode_machine)),
                   ("mode_pr", int(m.mode_pr))):
        exp = EXPECT[k]
        mark = "OK  " if got == exp else "★差異"
        print(f"  {k:14s} = {got}   期待 {exp}   [{mark}]")
    if int(m.mode_pr) != EXPECT["mode_pr"]:
        print("     ★mode_pr が想定と違う。足首(pitch/roll)と並列モータ(A/B)の")
        print("       意味が変わる。cockpitは 0 決め打ちで送るので、方策を")
        print("       動かす前に real_robot.py の c.mode_pr を合わせること")

    q = np.array([m.motor_state[i].q for i in range(29)])
    dq = np.array([m.motor_state[i].dq for i in range(29)])
    tau = np.array([m.motor_state[i].tau_est for i in range(29)])
    tp = np.array([m.motor_state[i].temperature[0] for i in range(29)])
    st = np.array([m.motor_state[i].mode for i in range(29)])
    print(f"\n--- モータ29軸 " + "-" * 47)
    print(f"  mode(各軸)     = {sorted(set(int(x) for x in st))}"
          f"  (0=無効 1=有効)")
    print(f"  温度           = 最小{tp.min():.0f} / 中央{np.median(tp):.0f} / "
          f"最大{tp.max():.0f}度 ({JOINTS[int(tp.argmax())]})")
    if tp.max() >= 60:
        print("     ★60度を超えている軸がある。冷めるまで方策を回さない")
    print(f"  関節速度       = 最大 {np.abs(dq).max():.2f} rad/s "
          f"({JOINTS[int(np.abs(dq).argmax())]})")
    print(f"  推定トルク     = 最大 {np.abs(tau).max():.1f} Nm "
          f"({JOINTS[int(np.abs(tau).argmax())]})")
    if np.abs(tau).max() > 1.0:
        print("     → 内蔵制御がトルクを出している(立位保持中など)")
    else:
        print("     → ほぼ無トルク(damp/ゼロトルクの状態)")

    quat = np.array(m.imu_state.quaternion)
    w, x, y, z = quat
    up_z = 1 - 2 * (x * x + y * y)
    tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
    gyro = np.array(m.imu_state.gyroscope)
    print(f"\n--- IMU " + "-" * 54)
    print(f"  傾き           = {tilt:.1f} 度")
    print(f"  角速度|xy|     = {np.linalg.norm(gyro[:2]):.3f} rad/s")
    if tilt > 15 and np.linalg.norm(gyro[:2]) < 0.25:
        print("     ★静的な前傾。この姿勢からは方策を開始できない(自動で拒否)")

    print(f"\n--- 内蔵制御サービス " + "-" * 42)
    name = "?"
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client \
            import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(3.0)
        msc.Init()
        code, res = msc.CheckMode()
        name = (res or {}).get("name") or "(解放中)"
        print(f"  CheckMode      = code={code}  {res}")
        print(f"  → 現在の制御   = {name}")
        if name == "(解放中)":
            print("     ★解放中。リモコンかコックピットの[ダンプ]で"
                  "内蔵制御へ戻してから始めること")
    except Exception as e:                         # noqa: BLE001
        print(f"  ★MotionSwitcher に繋がらない: {e}")

    print(f"\n--- LocoClient(GETのみ。設定は送らない) " + "-" * 23)
    try:
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        c = LocoClient()
        c.SetTimeout(3.0)
        c.Init()
        for api, label in LOCO_GETS:
            try:
                c._RegistApi(api, 0)
            except Exception:                      # noqa: BLE001
                pass
            try:
                code, data = c._Call(api, "{}")
                try:
                    data = json.loads(data) if isinstance(data, str) else data
                except Exception:                  # noqa: BLE001
                    pass
                print(f"  {label:14s} (api {api}): code={code}  {data}")
            except Exception as e:                 # noqa: BLE001
                print(f"  {label:14s} (api {api}): 例外 {e}")
    except Exception as e:                         # noqa: BLE001
        print(f"  ★LocoClient に繋がらない: {e}")

    print("\n" + "=" * 66)
    ok = (int(m.mode_machine) == EXPECT["mode_machine"]
          and int(m.mode_pr) == EXPECT["mode_pr"] and hz > 100)
    if ok:
        print("前提値は8/24の実測と一致。コックピットを実機モードで開いてよい")
    else:
        print("★前提値に差異がある。上の★印を潰してから方策を動かすこと")
    print("  ※FSM id(0/1/4/200)の効きは、実際にボタンを押して確かめるしかない。")
    print("    最初は必ず吊り下げ・物理E-STOPを手元に。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
