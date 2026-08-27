#!/usr/bin/env python3
"""実機ログを**方策ごと**に集計する。rc と rb のような同条件比較のための道具。

log_view.py が「1回の走行を詳しく」見るのに対して、こちらは
「どの方策が何回走って何回完走したか」を横に並べる。
実機セッション_20260824.md §6-4「rc vs rb の実機成功率を集計 →
較正シム予測(rc 83% vs rb 74%)と比較」がそのまま出る。

  python3 real/ab_report.py                 # 実機ログ全部
  python3 real/ab_report.py --today         # 今日のセッションだけ
  python3 real/ab_report.py --since 20260824
  python3 real/ab_report.py --sim           # simの回も含める(既定は実機のみ)
  python3 real/ab_report.py --runs          # 1回ずつの明細も出す

完走の判定: その方策の参照コマ数まで RUNNING で到達したか。
途中でDAMPした回も**必ず行として出す**(中止ログこそ較正の材料)。
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
LOGS = ROOT / "logs" / "real"
DEPLOY = ROOT / "deploy"
FSM_RUNNING, FSM_DAMP = 3, 5


def ref_frames(name):
    """方策の参照コマ数。設定.jsonに無い古いログのために deploy から読む。"""
    try:
        return int(len(np.load(DEPLOY / name / "reference.npz")["ref_q"]))
    except Exception:                              # noqa: BLE001
        return 0


def col_index(cols, prefix, n=29):
    cols = [str(c) for c in cols]
    if f"{prefix}0" not in cols:
        return None
    i = cols.index(f"{prefix}0")
    return slice(i, i + n)


def rated_torque(names):
    """連続定格[N·m]。log_view.py と同じ表(SIM_SUMMARY.md 1章)。"""
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


def scan_run(npz_path, meta):
    """1本の記録を1行分の数値にする。読めなければ None。"""
    name = "_".join(npz_path.stem.split("_")[2:])   # runNN_<i>_<方策>
    if not name or name.startswith("interp"):
        return None
    z = np.load(npz_path)
    rec, cols = z["rec"], z["cols"]
    if len(rec) == 0:
        return None
    fsm = rec[:, 1].astype(int)
    run = rec[fsm == FSM_RUNNING]
    n_ref = int((meta or {}).get("n_frames") or 0) or ref_frames(name)
    reached = int(run[:, 0].max()) + 1 if len(run) else 0
    tau = rec[:, col_index(cols, "tau")]
    names = (meta or {}).get("joint_names") or [f"j{i}" for i in range(29)]
    ratio = np.abs(tau) / np.maximum(rated_torque(names), 1e-6)
    ts = col_index(cols, "temp")
    temp = rec[:, ts] if ts is not None else None
    knee = ankle = (np.nan, np.nan)
    if temp is not None and np.any(temp):
        knee = (float(temp[0, [3, 9]].max()), float(temp[-1, [3, 9]].max()))
        ankle = (float(temp[0, [4, 5, 10, 11]].max()),
                 float(temp[-1, [4, 5, 10, 11]].max()))
    return dict(
        policy=name, path=npz_path,
        started=(meta or {}).get("started", ""),
        is_sim=bool((meta or {}).get("is_sim", False)),
        n_ref=n_ref, reached=reached,
        done=bool(n_ref and reached >= n_ref),
        damped=bool(np.any(fsm == FSM_DAMP)) or not bool(z["final"]),
        # 秒は RUNNING の区間だけ(記録には保持中のコマも入っている)
        secs=(float(run[-1, 3] - run[0, 3]) if len(run) > 1 else 0.0),
        tilt_max=float(rec[:, 4].max()),
        tau_ratio=float(ratio.max()),
        knee=knee, ankle=ankle)


def collect(sessions, want_sim):
    rows = []
    for d in sessions:
        for npz in sorted(d.glob("run*.npz")):
            mp = d / (npz.name.split("_")[0] + "_設定.json")
            meta = (json.loads(mp.read_text(encoding="utf-8"))
                    if mp.exists() else None)
            try:
                r = scan_run(npz, meta)
            except Exception as e:                 # noqa: BLE001
                print(f"  (読めない: {npz.name} — {e})")
                continue
            if r is None or (r["is_sim"] and not want_sim):
                continue
            r["session"] = d.name
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="YYYYMMDD 以降のセッション")
    ap.add_argument("--today", action="store_true", help="今日のセッションだけ")
    ap.add_argument("--sim", action="store_true", help="simの回も含める")
    ap.add_argument("--runs", action="store_true", help="1回ずつの明細も出す")
    a = ap.parse_args()

    if not LOGS.is_dir():
        print(f"ログがない: {LOGS}")
        return 1
    since = time.strftime("%Y%m%d") if a.today else a.since
    ss = [d for d in sorted(LOGS.iterdir()) if d.is_dir()
          and (not since or d.name[len("cockpit_"):] >= since)]
    if not ss:
        print("対象のセッションがない")
        return 1
    rows = collect(ss, a.sim)
    if not rows:
        print(f"対象の走行がない(セッション{len(ss)}件を見た)。"
              + ("" if a.sim else " simの回も見るなら --sim"))
        return 1

    print(f"セッション{len(ss)}件 / 走行{len(rows)}件"
          + ("(sim含む)" if a.sim else "(実機のみ)"))
    if a.runs:
        print(f"\n{'方策':<16s}{'開始':<20s}{'到達':>10s} {'秒':>5s} "
              f"{'傾き最大':>8s} {'τ比':>5s}  結果")
        for r in rows:
            res = "完走" if r["done"] else "中止"
            if not r["done"] and r["damped"]:
                res += "(DAMP)"
            print(f"{r['policy']:<16s}{r['started']:<20s}"
                  f"{r['reached']:>5d}/{r['n_ref']:<4d} {r['secs']:5.1f} "
                  f"{r['tilt_max']:7.1f}度 {r['tau_ratio']:5.2f}  {res}")

    print(f"\n{'方策':<16s}{'回数':>4s}{'完走':>5s}{'成功率':>7s}"
          f"{'到達率中央':>11s}{'傾き最大の中央':>15s}{'τ比最大':>8s}  温度 膝(開始→終了)")
    for name in sorted({r["policy"] for r in rows}):
        g = [r for r in rows if r["policy"] == name]
        done = sum(r["done"] for r in g)
        prog = [r["reached"] / r["n_ref"] for r in g if r["n_ref"]]
        kn = [r["knee"] for r in g if not np.isnan(r["knee"][0])]
        tnote = (f"{min(k[0] for k in kn):.0f}→{max(k[1] for k in kn):.0f}度"
                 if kn else "—")
        print(f"{name:<16s}{len(g):>4d}{done:>5d}{100*done/len(g):>6.0f}%"
              f"{100*np.median(prog) if prog else 0:>10.0f}%"
              f"{np.median([r['tilt_max'] for r in g]):>13.1f}度"
              f"{max(r['tau_ratio'] for r in g):>8.2f}  {tnote}")
    print("\n完走=参照コマ数まで RUNNING で到達。中止した回もこの表に入っている。"
          "\n較正シムの予測(実機由来の開始7状態): sit_up_rc_r2 83% / sit_up_rb_r2 74%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
