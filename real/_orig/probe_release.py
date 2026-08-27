#!/usr/bin/env python3
"""`ReleaseMode()` が固まるのかを、**ロボットを動かさずに**確かめる。

固まる条件は「内蔵制御が動作中に、こちらが500Hz送信しながら ReleaseMode を
呼ぶ」。これを **damp の状態で**再現する。dampは元々脱力なので、解放が
成功してもしなくてもロボットの状態は変わらない(こちらは kp=kd=0 = 無トルク
で送るため、掴むこともしない)。

見たいこと:
  1. ReleaseMode は何秒で返るか(固まるか)
  2. 固まるとき、GILを握ったままか(= 他のPythonスレッドも止まるか)
  3. 解放後、内蔵制御は lowcmd を出さなくなるか

  python3 real/probe_release.py --iface enp46s0

★実行前にロボットを damp にしておくこと。支えは不要(元々脱力)。
"""
import argparse
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp46s0")
    a = ap.parse_args()
    from real_robot import RealRobot

    print("=" * 60)
    r = RealRobot(iface=a.iface)
    print(f"接続。内蔵制御 = {r.current_mode()}")
    if r.current_mode() in ("(解放中)", "?"):
        print("★既に解放中です。この試験は内蔵制御が動作中でないと意味が"
              "ありません。コックピットで[ダンプ]を押してから実行してください")
        return 1

    # --- 他のPythonスレッドが動き続けるかを見る心拍
    beats = {"n": 0, "last": 0.0}
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            beats["n"] += 1
            beats["last"] = time.time()
            time.sleep(0.02)                       # 50Hz
    threading.Thread(target=heartbeat, daemon=True).start()

    # --- こちらの500Hz送信を開始。kp=kd=0 なのでトルクは出ない(無害)
    q = r.state()[0]
    r.set_target(q, np.zeros(29), np.zeros(29))
    r._stream_on = True
    print("500Hz送信を開始(kp=kd=0 = 無トルク。ロボットは動きません)")
    time.sleep(1.0)
    n0 = beats["n"]

    # --- 本命: ReleaseMode を呼んで、返るまでの時間を測る
    print("\nReleaseMode() を呼びます…")
    t0 = time.time()
    done = {"ok": False}

    def call():
        try:
            r._msc.ReleaseMode()
            done["ok"] = True
        except Exception as e:                     # noqa: BLE001
            done["err"] = e
    th = threading.Thread(target=call, daemon=True)
    th.start()
    while th.is_alive() and time.time() - t0 < 20.0:
        time.sleep(0.1)
    dt = time.time() - t0
    n1 = beats["n"]
    hb = (n1 - n0) / max(dt, 1e-9)

    if th.is_alive():
        print(f"★ReleaseMode が {dt:.1f}秒たっても返りません")
        print(f"  この間の心拍(別スレッド) = {hb:.0f} Hz (正常なら約50Hz)")
        if hb < 5:
            print("  → **GILを握ったままブロックしています。**")
            print("     プロセス内の対策(スレッド+timeout)では回避できません。")
            print("     別プロセスでRPCを呼ぶ設計が必要です。")
        else:
            print("  → GILは解放されています。スレッド+timeout で回避できます")
    else:
        print(f"ReleaseMode は {dt:.2f}秒で返りました "
              f"({'成功' if done.get('ok') else 'エラー: '+str(done.get('err'))})")
        print(f"  この間の心拍 = {hb:.0f} Hz")

    # --- 解放されたか(lowcmdの実測)
    ok, wdt = r.wait_release(quiet=0.4, timeout=6.0)
    print(f"\n内蔵制御のlowcmd停止: {'確認' if ok else '★まだ出し続けている'}"
          f" ({wdt:.1f}秒)")
    print(f"CheckMode の言い分 = {r.current_mode()}")

    # --- 後始末: 送信を止めて内蔵制御へ戻す
    print("\n後始末: 送信停止 → 内蔵制御へ復帰")
    r._stream_on = False
    stop.set()
    back = r._select_ai()
    print(f"  復帰 = {'OK' if back else '★失敗'}  内蔵制御 = {r.current_mode()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
