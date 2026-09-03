#!/usr/bin/env python3
"""`ReleaseMode()` が固まるのかを、**ロボットを動かさずに**確かめる。

固まる条件は「内蔵制御が動作中に、こちらが500Hz送信しながら ReleaseMode を
呼ぶ」。これを **damp の状態で**再現する。dampは元々脱力なので、解放が
成功してもしなくてもロボットの状態は変わらない(こちらは kp=kd=0 = 無トルク
で送るため、掴むこともしない)。

見たいこと:
  1. ReleaseMode は何秒で返るか(固まるか)
  2. 固まるとき、GILを握ったままか(= 他のPythonスレッドも止まるか)

  python3 real/probe_release.py --iface enp46s0

★実行前にロボットを damp にしておくこと。支えは不要(元々脱力)。

--- 2026-08-26 改訂 ------------------------------------------------------
旧版は存在しないメソッド(wait_release / _select_ai)を解放後に呼んでいて、
**解放したまま例外終了し、復帰処理をせずに終わる**状態だった。実行禁止
だったものを、次の2点を直して実行可能にした。

  1. 復帰処理を finally に置いた。途中で何が起きても必ず内蔵制御へ戻す
  2. 解放の確認に rt/lowcmd の購読を使わない。
     **同一プロセスで同じトピックを publish しながら subscribe すると
     cyclonedds 内部でデッドロックする**(2026-08-20 A/B実測: 購読ありで
     即停止、なしで心拍49Hz維持)。この試験自体が publish しているので、
     ここで購読してはいけない。lowcmd を見たいときは、**別ターミナルで**
     `python3 real/probe_lowcmd.py --iface enp46s0` を同時に走らせること
     (あちらは送信しないので安全)。
"""
import argparse
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp46s0")
    a = ap.parse_args()
    from real_robot import RealRobot

    print("=" * 60)
    print("★別ターミナルで probe_lowcmd.py を走らせておくと、"
          "解放が効いたかを実測で見られます(このプロセスでは購読しません)")
    r = RealRobot(iface=a.iface)               # watch_cmd=False のまま
    mode = r.current_mode()
    print(f"接続。内蔵制御 = {mode}")
    if mode in ("(解放中)", "?"):
        print("★既に解放中です。この試験は内蔵制御が動作中でないと意味が"
              "ありません。コックピットで[ダンプ]を押してから実行してください")
        r.close()
        return 1

    stop = threading.Event()
    beats = {"n": 0}

    def heartbeat():
        while not stop.is_set():
            beats["n"] += 1
            time.sleep(0.02)                       # 50Hz
    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        # --- こちらの500Hz送信を開始。kp=kd=0 なのでトルクは出ない(無害)
        q = r.state()[0]
        r.set_target(q, np.zeros(29), np.zeros(29), latch=True)
        r._stream_on = True
        time.sleep(1.0)
        alive, age = r.send_alive()
        print(f"500Hz送信を開始(kp=kd=0 = 無トルク。ロボットは動きません) "
              f"送信={'OK' if alive else '★出ていない'} "
              f"{r._send_n}パケット")
        if not alive:
            print(f"★送信が出ていません({r._send_err_msg})。試験を中止します")
            return 1
        n0 = beats["n"]
        n_pkt0 = r._send_n

        # --- 本命: ReleaseMode を呼んで、返るまでの時間を測る
        print("\nReleaseMode() を呼びます…")
        t0 = time.time()
        done = {}

        def call():
            try:
                done["r"] = r._msc.ReleaseMode()
                done["ok"] = True
            except Exception as e:                 # noqa: BLE001
                done["err"] = e
        th = threading.Thread(target=call, daemon=True)
        th.start()
        while th.is_alive() and time.time() - t0 < 20.0:
            time.sleep(0.1)
        dt = time.time() - t0
        hb = (beats["n"] - n0) / max(dt, 1e-9)
        pkt = (r._send_n - n_pkt0) / max(dt, 1e-9)

        if th.is_alive():
            print(f"★ReleaseMode が {dt:.1f}秒たっても返りません")
            print(f"  心拍(別スレッド) = {hb:.0f} Hz (正常なら約50Hz)")
            print(f"  lowcmd送信       = {pkt:.0f} Hz (正常なら約500Hz)")
            if hb < 5:
                print("  → **GILを握ったままブロックしています。**")
                print("     プロセス内の対策(スレッド+timeout)では"
                      "指令の送信まで止まります。別プロセス化が必要です。")
            else:
                print("  → GILは解放されています。スレッド+timeout で回避可能"
                      "(real_robot.py の _rpc / _rpc_async がこれ)")
        else:
            print(f"ReleaseMode は {dt:.2f}秒で返りました "
                  f"({'成功' if done.get('ok') else 'エラー: ' + str(done.get('err'))})")
            print(f"  心拍 = {hb:.0f} Hz / lowcmd送信 = {pkt:.0f} Hz")
        print(f"\nCheckMode の言い分 = {r.current_mode()}")
        print("  ※CheckMode は当てにならない(解放済みと返るのに制御権が"
              "移らない実測あり)。実測は probe_lowcmd.py の Hz で見ること")
        return 0
    finally:
        # --- 後始末は**必ず**通す。解放したまま終わらせない
        print("\n後始末: 送信停止 → 内蔵制御へ復帰")
        stop.set()
        r._stream_on = False
        time.sleep(0.1)
        back = r._select_ai()
        print(f"  復帰 = {'OK' if back else '★失敗(リモコンでdampにすること)'}"
              f"  内蔵制御 = {r.current_mode()}")
        r.close()


if __name__ == "__main__":
    sys.exit(main())
