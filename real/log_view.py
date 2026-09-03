#!/usr/bin/env python3
"""実機/simの走行ログを読んで数値で要約する。グラフPNGも出せる。

コックピットは `logs/real/cockpit_<日時>/` に
  run<NN>_<フェーズ>.npz   毎コマの記録(100コマごとに逐次保存)
  run<NN>_設定.json        その走行の条件(方策・ゲイン・閾値)
  イベント.log             時刻つきの経過(切替・中止理由)
を残す。このツールはそれを読む。

  python3 real/log_view.py                     # 最新セッションを要約
  python3 real/log_view.py --list              # セッション一覧
  python3 real/log_view.py <セッションのパス>    # 指定して要約
  python3 real/log_view.py --plot              # 併せてPNGを書き出す
  python3 real/log_view.py --csv               # 併せてCSVを書き出す
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "real"
FSM_NAME = {0: "IDLE", 1: "MOVING", 2: "WAIT", 3: "RUNNING", 4: "HOLD", 5: "DAMP"}
# 連続定格。トルク余裕の判定に使う(SIM_SUMMARY.md 1章)
TAU_RATED = None


def rated_torque(names):
    out = []
    for n in names:
        if "knee" in n:
            out.append(139.0)
        elif "ankle" in n or "waist_roll" in n or "waist_pitch" in n:
            out.append(35.0)
        elif "hip" in n or "waist" in n:
            out.append(88.0)
        elif "wrist_pitch" in n or "wrist_yaw" in n:
            out.append(5.0)
        else:
            out.append(25.0)
    return np.array(out)


def sessions():
    return sorted([d for d in LOGS.iterdir() if d.is_dir()]) if LOGS.is_dir() else []


def col_slices(cols):
    cols = [str(c) for c in cols]
    def rng(prefix, n):
        i = cols.index(f"{prefix}0")
        return slice(i, i + n)
    return dict(q=rng("q", 29), dq=rng("dq", 29), tau=rng("tau", 29),
                target=rng("target", 29), act=rng("act", 29),
                quat=slice(cols.index("quat_w"), cols.index("quat_w") + 4),
                gyro=slice(cols.index("gyro_x"), cols.index("gyro_x") + 3))


def summarize(path, meta, plot=False, csv=False):
    z = np.load(path)
    rec, cols = z["rec"], z["cols"]
    s = col_slices(cols)
    n = len(rec)
    if n == 0:
        print(f"  {path.name}: 空")
        return
    names = meta.get("joint_names") if meta else None
    names = names or [f"j{i}" for i in range(29)]
    rated = rated_torque(names)
    q, dq, tau = rec[:, s["q"]], rec[:, s["dq"]], rec[:, s["tau"]]
    tgt, act = rec[:, s["target"]], rec[:, s["act"]]
    tilt = rec[:, 4]
    fsm = rec[:, 1].astype(int)
    dur = float(rec[-1, 3] - rec[0, 3]) if n > 1 else 0.0
    hz = (n - 1) / dur if dur > 0 else 0.0
    # 追従誤差は「送っている目標との差」で見る(参照との差は方策の残差を含む)
    err = np.abs(q - tgt)
    leg = err[:, :15]
    ratio = np.abs(tau) / np.maximum(rated, 1e-6)
    sat = np.mean(np.abs(act) > 0.99, axis=1)
    fin = "完了" if bool(z["final"]) else "途中(保存中に終了)"
    states = " ".join(f"{FSM_NAME.get(k,k)}×{int(np.sum(fsm==k))}"
                      for k in sorted(set(fsm.tolist())))
    print(f"\n  ── {path.name}  {n}コマ / {dur:.1f}秒 / 実効{hz:.1f}Hz  [{fin}]")
    print(f"     状態: {states}")
    print(f"     傾き        中央{np.median(tilt):5.1f}度  最大{tilt.max():5.1f}度")
    # **値と関節名は必ず同じ配列から出す。** 以前は最大値を脚腰15関節から、
    # 関節名を全29関節から取っていたため、「脚腰 最大0.776rad(右肘)」のような
    # 実在しない組み合わせが出ていた(肘は脚腰に含まれない)。腕は別行で出す
    # ようにして、上半身の追従破綻を脚の値で隠さない(2026-08-25)。
    arm = err[:, 15:]
    print(f"     追従誤差    脚腰 中央{np.median(leg):5.3f}  最大{leg.max():5.3f} rad"
          f"  (最大の関節: {names[int(np.argmax(leg.max(axis=0)))]})")
    print(f"                 腕  中央{np.median(arm):5.3f}  最大{arm.max():5.3f} rad"
          f"  (最大の関節: {names[15 + int(np.argmax(arm.max(axis=0)))]})")
    print(f"     関節速度    最大{np.abs(dq).max():5.2f} rad/s")
    print(f"     トルク比    中央{np.median(ratio):5.2f}  最大{ratio.max():5.2f}"
          f"  (最大の関節: {names[int(np.argmax(ratio.max(axis=0)))]})")
    print(f"     行動の飽和  中央{np.median(sat)*100:4.0f}%  最大{sat.max()*100:4.0f}%")
    hot = np.argsort(-ratio.max(axis=0))[:3]
    print("     トルク上位: " + " / ".join(
        f"{names[i]} {ratio[:, i].max():.2f}" for i in hot))
    if "obs" in z:
        print(f"     観測も保存済み: {z['obs'].shape}(simとの突き合わせに使える)")
    if csv:
        out = path.with_suffix(".csv")
        np.savetxt(out, rec, delimiter=",", fmt="%.6g",
                   header=",".join(str(c) for c in cols), comments="")
        print(f"     CSV: {out.name}")
    if plot:
        _plot(path, rec, tilt, leg, ratio, sat, names)


def _plot(path, rec, tilt, leg, ratio, sat, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("     (matplotlib が無いのでグラフは省略)")
        return
    t = np.arange(len(rec)) / 50.0
    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(t, tilt); ax[0].axhline(40, color="r", ls="--")
    ax[0].set_ylabel("tilt [deg]")
    ax[1].plot(t, leg.max(axis=1)); ax[1].set_ylabel("track err [rad]")
    ax[2].plot(t, ratio.max(axis=1)); ax[2].axhline(1.0, color="r", ls="--")
    ax[2].set_ylabel("torque / rated")
    ax[3].plot(t, sat * 100); ax[3].set_ylabel("action sat [%]")
    ax[3].set_xlabel("time [s]")
    fig.suptitle(path.stem)
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"     グラフ: {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", help="セッションのパス(省略で最新)")
    ap.add_argument("--list", action="store_true", help="セッション一覧")
    ap.add_argument("--plot", action="store_true", help="PNGを書き出す")
    ap.add_argument("--csv", action="store_true", help="CSVを書き出す")
    a = ap.parse_args()

    ss = sessions()
    if a.list:
        for d in ss:
            runs = len(list(d.glob("run*.npz")))
            print(f"{d.name}   走行{runs}件")
        return 0
    if not ss and not a.session:
        print(f"ログがない: {LOGS}")
        return 1
    d = pathlib.Path(a.session) if a.session else ss[-1]
    if not d.is_dir():
        print(f"見つからない: {d}")
        return 1
    print(f"セッション: {d}")
    ev = d / "イベント.log"
    if ev.exists():
        lines = ev.read_text(encoding="utf-8").splitlines()
        print(f"\nイベント({len(lines)}行) 末尾:")
        for l in lines[-8:]:
            print("   " + l)
    for npz in sorted(d.glob("run*.npz")):
        mp = d / (npz.name.split("_")[0] + "_設定.json")
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
        if meta:
            tag = "SIM" if meta.get("is_sim") else "実機"
            print(f"\n[{tag}] run{meta['run']:02d} "
                  f"{' → '.join(meta['phases'])}  開始 {meta['started']}")
        summarize(npz, meta, plot=a.plot, csv=a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
