#!/usr/bin/env python3
"""実機の走行ログと、同じ条件のシミュレーションを突き合わせる。**微調整の材料。**

  python3 real/sim_compare.py                       # 最新セッションの全runを比較
  python3 real/sim_compare.py --policy sit_up_deep_r2 --all   # その方策の全実機ログ
  python3 real/sim_compare.py --run logs/real/<session>/run01_0_sit_up_deep_r2.npz
  python3 real/sim_compare.py --run <npz> --mode policy       # 方策ロールアウト
  python3 real/sim_compare.py --run <npz> --plot              # PNGも書く

★実機の制御中には走らせないこと(MuJoCoの物理を回すのでCPUを食う)。
  走行と走行の間か、帰ってから使う。実機には一切繋がない。

--- 3つのモード ------------------------------------------------------------
--mode step(既定) **短時間予測誤差 = シム較正に使うのはこれ**
    各コマで、シムを**実機の実測状態にリセット**してから、実機に出ていた
    指令を STEP_HORIZON 秒だけ与えて、その先の実機と比べる。
    誤差が積み上がらないので、差はその瞬間のモデル誤差(摩擦・慣性・接触)
    だけを表す。関節ごと・時刻ごとに出るので、どこが合っていないか分かる。

--mode policy     **方策ロールアウト**
    実機と同じ開始姿勢から、方策をシムの中で閉ループで回す。
    → 「シムならこの試行はどうなったか」。実機の失敗がシムで再現するかを見る。

--mode replay     **指令リプレイ(open-loop)。★balance課題では使えない**
    記録した target をそのまま最後まで流す。方策は**フィードバック制御**
    なので、出力だけを開ループで流すと必ず発散する。
    実測(2026-08-26 deep 完走回)でシムは0.8秒で傾き100度に達した。
    差はモデル誤差ではなく発散なので、較正には使えない。残してあるのは
    「なぜ使えないか」を再現できるようにするため。

--- 置き方について(ここは正直に) -------------------------------------------
実機の水平位置は分からない(脚オドメトリはドリフトする)ので、シムでは
**参照軌道が想定する開始位置** `ref_xy_abs[0]` に置く。姿勢(関節角とIMU)は
実機の実測を使い、高さは足が床に着くように決める。
つまり「椅子との位置関係は設計どおりだった場合」の比較になる。
実機の椅子の位置がずれていたなら、その分は差として出る。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

LOGS = ROOT / "logs" / "real"
# --mode step の予測ホライズン[秒]。短いほどモデル誤差だけを見る。
# 0.1秒(5コマ)は、接触と摩擦の効果が出るには十分で、姿勢の発散が
# 効いてくるには短い。
STEP_HORIZON = 0.10

LEG = slice(0, 12)
WAIST = slice(12, 15)
ARM = slice(15, 29)


def load_run(path):
    """実機ログ(npz)を読む。cols名で引くので列を足しても壊れない"""
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    rec = z["rec"]
    g = lambda p: [cols.index(f"{p}{i}") for i in range(29)]      # noqa: E731
    out = dict(
        q=rec[:, g("q")].astype(float),
        dq=rec[:, g("dq")].astype(float),
        tau=rec[:, g("tau")].astype(float),
        target=rec[:, g("target")].astype(float),
        act=rec[:, g("act")].astype(float),
        temp=rec[:, g("temp")].astype(float),
        quat=rec[:, [cols.index(c) for c in
                     ("quat_w", "quat_x", "quat_y", "quat_z")]].astype(float),
        gyro=rec[:, [cols.index(c) for c in
                     ("gyro_x", "gyro_y", "gyro_z")]].astype(float),
        tilt=rec[:, cols.index("tilt_deg")].astype(float),
        final=bool(z["final"]) if "final" in z else None,
    )
    for c in ("dt_ms", "ms_infer"):
        if c in cols:
            out[c] = rec[:, cols.index(c)].astype(float)
    out["n"] = len(rec)
    return out


def run_meta(path):
    p = pathlib.Path(path)
    m = p.parent / f"{p.stem.split('_')[0]}_設定.json"
    return json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}


class Sim:
    """比較用のMuJoCo。sim_robot.py と同じPD・同じサブステップ数を使う"""

    def __init__(self, policy_name):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(
            str(ROOT / "model" / "scene_task.xml"))
        self.d = mujoco.MjData(self.m)
        self.qadr, self.dofadr, self.acts = [], [], []
        for i in range(self.m.nu):
            jid = self.m.actuator_trnid[i, 0]
            lo, hi = self.m.jnt_range[jid]
            if lo == 0.0 and hi == 0.0:
                continue
            self.qadr.append(int(self.m.jnt_qposadr[jid]))
            self.dofadr.append(int(self.m.jnt_dofadr[jid]))
            self.acts.append(i)
        self.qadr = np.array(self.qadr)
        self.dofadr = np.array(self.dofadr)
        self.acts = np.array(self.acts)
        self.tau_lo = self.m.actuator_ctrlrange[self.acts, 0].copy()
        self.tau_hi = self.m.actuator_ctrlrange[self.acts, 1].copy()
        self.fid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                        "pelvis")
        self.chair = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                       "chair")
        z = np.load(ROOT / "deploy" / policy_name / "reference.npz")
        self.ref = z
        self.ref_q = z["ref_q"]

    def place(self, q, quat, dq=None, gyro=None, z_hint=None):
        """実機の姿勢で置く。xyは参照の開始位置、高さは足が床に着くところ。

        dq / gyro を渡すと関節速度と体幹角速度も実機に合わせる
        (--mode step では速度まで合わせないと短時間予測にならない)。
        """
        d = self.d
        self.mj.mj_resetData(self.m, d)
        d.qpos[0:2] = self.ref["ref_xy_abs"][0][:2]
        d.qpos[2] = 1.0
        d.qpos[3:7] = quat / np.linalg.norm(quat)
        d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, d)
        if z_hint is not None:
            d.qpos[2] = float(z_hint)
        else:
            # ★足を「床(z=0)」ではなく**参照が想定する足の高さ**に置く。
            #   着座フェーズは段の上から始まる(sit_up_deep_r2 の
            #   min(ref_foot_z[0]) = 0.244 m)。床に置くと椅子のベース箱に
            #   めり込んで、置いた瞬間に16接触・即転倒になる(実測)。
            zmin = min(d.xpos[self.fid[0]][2], d.xpos[self.fid[1]][2])
            foot0 = float(np.min(self.ref["ref_foot_z"][0]))
            d.qpos[2] = 1.0 - zmin + foot0
        d.qvel[:] = 0.0
        if dq is not None:
            d.qvel[self.dofadr] = dq
        if gyro is not None:
            d.qvel[3:6] = gyro
        self.mj.mj_forward(self.m, d)

    def step(self, target, kp, kd, n_sub=10):
        d = self.d
        for _ in range(n_sub):
            q = d.qpos[self.qadr]
            dq = d.qvel[self.dofadr]
            tau = np.clip(kp * (target - q) - kd * dq, self.tau_lo, self.tau_hi)
            d.ctrl[self.acts] = tau
            self.mj.mj_step(self.m, d)

    def _body_kind(self, bid):
        """そのボディがロボットのどこか。seat(尻・腿) / foot(足) / other"""
        while bid > 0:
            nm = self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_BODY, bid)
            if nm:
                if "ankle" in nm or "foot" in nm:
                    return "foot"
                if "pelvis" in nm or "hip" in nm or "knee" in nm \
                        or "waist" in nm or "torso" in nm:
                    return "seat"
            bid = int(self.m.body_parentid[bid])
        return "other"

    def seat_load(self):
        """ロボットが**椅子から受けている鉛直力**[N]を、尻/足に分けて返す。

        ずり落ちの直接の指標(実機セッション手順 §6: 一度座ってから
        尻荷重が219N→76Nへ抜けていく現象)。
        ★椅子のベース箱がそのまま段差なので、足で段を踏んでいる力と
          尻で座面を押している力を**必ず分ける**。混ぜると体重の倍以上の
          数字になって意味がなくなる(実測8593N。ロボットは約35kg=350N)。
        ★ロボットのgeomと椅子のgeomの接触だけを数える。椅子の凸分割どうしの
          自己接触や、床と椅子の接触を拾わない。
        """
        d, m = self.d, self.m
        rob = int(m.body_rootid[self.pelvis])
        cha = int(m.body_rootid[self.chair])
        seat = foot = 0.0
        buf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            r1, r2 = int(m.body_rootid[b1]), int(m.body_rootid[b2])
            if r1 == rob and r2 == cha:
                rb = b1
            elif r2 == rob and r1 == cha:
                rb = b2
            else:
                continue                            # ロボット×椅子 以外は無視
            self.mj.mj_contactForce(m, d, i, buf)
            # 接触フレームは行が軸。ワールドへ戻して鉛直成分を取る
            fw = c.frame.reshape(3, 3).T @ buf[0:3]
            fz = abs(float(fw[2]))
            if self._body_kind(rb) == "foot":
                foot += fz
            else:
                seat += fz
        return seat, foot

    def read(self):
        d = self.d
        w, x, y, z = d.qpos[3:7]
        up_z = 1 - 2 * (x * x + y * y)
        seat, foot = self.seat_load()
        return dict(
            q=d.qpos[self.qadr].copy(),
            dq=d.qvel[self.dofadr].copy(),
            tau=d.ctrl[self.acts].copy(),
            quat=d.qpos[3:7].copy(),
            tilt=float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z))))),
            pelvis_z=float(d.xpos[self.pelvis][2]),
            seat_n=seat,
            foot_n=foot,
        )


def rollout_step(sim, real, kp, kd):
    """各コマで実機の状態へリセットし、STEP_HORIZON 秒だけ先を予測する。

    誤差が積み上がらないので、差 = その瞬間のモデル誤差。
    返すのは「H秒後のシム」と「H秒後の実機」の組。
    """
    hz = 50
    H = max(1, int(round(STEP_HORIZON * hz)))
    out = []
    for t in range(real["n"] - H):
        sim.place(real["q"][t], real["quat"][t],
                  dq=real["dq"][t], gyro=real["gyro"][t])
        for k in range(H):
            sim.step(real["target"][t + k], kp, kd)
        r = sim.read()
        r["ref_t"] = t + H
        out.append(r)
    return out


def rollout_replay(sim, real, kp, kd):
    """実機に出た指令をそのままシムへ流す(open-loop)"""
    sim.place(real["q"][0], real["quat"][0])
    out = []
    for t in range(real["n"]):
        sim.step(real["target"][t], kp, kd)
        out.append(sim.read())
    return out


def rollout_policy(sim, real, kp, kd, policy_name):
    """同じ開始姿勢から、方策をシムの中で閉ループで回す"""
    from run_fsm import ACTION_SCALE, ObsBuilder, Policy
    pol = Policy(policy_name)
    ob = ObsBuilder(pol)
    ob.reset(est_xy=pol.ref["ref_xy_abs"][0][:2])
    sim.place(real["q"][0], real["quat"][0])
    out = []
    for t in range(min(real["n"], pol.n)):
        s = sim.read()
        gyro = sim.d.qvel[3:6].copy()
        obs = ob.build(pol, t, s["q"], s["dq"], s["quat"], gyro)
        a = pol.act(obs)
        ob.last_cmd = a.copy()
        sim.step(pol.ref_q[min(t, pol.n - 1)] + a * ACTION_SCALE, kp, kd)
        r = sim.read()
        r["act"] = a
        out.append(r)
    return out


def summarize(name, real, sim_out, meta, mode):
    n = len(sim_out)
    sq = np.array([s["q"] for s in sim_out])
    st = np.array([s["tilt"] for s in sim_out])
    sp = np.array([s["pelvis_z"] for s in sim_out])
    sn = np.array([s["seat_n"] for s in sim_out])
    fn = np.array([s["foot_n"] for s in sim_out])
    if mode == "step":
        idx = np.array([s["ref_t"] for s in sim_out])
        rq = real["q"][idx]
        rt = real["tilt"][idx]
    else:
        rq = real["q"][:n]
        rt = real["tilt"][:n]
    d = np.abs(sq - rq)
    hz = meta.get("control_hz", 50)

    print(f"\n{'=' * 70}")
    print(f"{name}   {mode}   {n}コマ / {n / hz:.1f}秒"
          f"   実機の結果: {'完走' if real['final'] else '中断'}")
    print("=" * 70)

    print("\n[1] 傾き — 実機 vs シム")
    print(f"{'時刻':>6s} {'実機':>7s} {'シム':>7s} {'差':>7s}")
    for k in range(0, n, max(1, n // 10)):
        print(f"{k / hz:5.1f}s {rt[k]:6.1f}° {st[k]:6.1f}° "
              f"{st[k] - rt[k]:+6.1f}°")
    print(f"  最大: 実機 {rt.max():.1f}° / シム {st.max():.1f}°"
          f"   差のRMS {np.sqrt(np.mean((st - rt) ** 2)):.1f}°")

    print("\n[2] 関節の差(シム − 実機)")
    print(f"{'部位':>6s} {'RMS':>8s} {'最大':>8s}  最大の関節")
    names = meta.get("joint_names", [f"j{i}" for i in range(29)])
    for lbl, sl in (("脚", LEG), ("腰", WAIST), ("腕", ARM)):
        dd = d[:, sl]
        j = int(np.unravel_index(dd.argmax(), dd.shape)[1])
        print(f"{lbl:>6s} {np.sqrt(np.mean(dd ** 2)):8.4f} {dd.max():8.4f}"
              f"  {names[sl.start + j]}")
    worst = np.argsort(-d.max(axis=0))[:5]
    print("  差の大きい関節: " + ", ".join(
        f"{names[i]} {d[:, i].max():.3f}" for i in worst))

    print("\n[3] 骨盤の高さ(シム) — ずり落ちの指標")
    print("  " + " ".join(f"{sp[k]:.3f}" for k in range(0, n, max(1, n // 12))))
    print(f"  着座後の沈み: {(sp[int(n * 0.6):].max() - sp[-1]) * 1000:.0f} mm"
          f"   (シムでの正常は0〜6mm。67mmが「ずり落ち」1本)")

    print("\n[4] 荷重の行き先(シム) — ずり落ちの直接の指標")
    print(f"{'時刻':>6s} {'尻→座面':>9s} {'足→段':>8s}")
    for k in range(0, n, max(1, n // 8)):
        print(f"{k / hz:5.1f}s {sn[k]:8.0f} N {fn[k]:7.0f} N")
    tail = slice(int(n * 0.85), n)
    print(f"  終端: 尻 {sn[tail].mean():.0f} N / 足 {fn[tail].mean():.0f} N"
          f"   (ロボットは約35kg = 350N)")
    print(f"  → 尻が受けている割合 {100 * sn[tail].mean() / max(1e-6, sn[tail].mean() + fn[tail].mean()):.0f}%"
          f"   ★座った後に尻の荷重が抜けていくなら「ずり落ち」")

    print("\n[5] 腕の動き(実機) — 「手の動きが激しい」の定量")
    adq = np.abs(real["dq"][:n, ARM])
    aact = np.abs(real["act"][:n, ARM])
    ldq = np.abs(real["dq"][:n, LEG])
    print(f"  腕の関節速度: 平均 {adq.mean():.2f} / 最大 {adq.max():.2f} rad/s"
          f"   (脚は 平均 {ldq.mean():.2f} / 最大 {ldq.max():.2f})")
    print(f"  腕のaction飽和率 {100 * np.mean(aact > 0.99):.1f}%"
          f"   (脚 {100 * np.mean(np.abs(real['act'][:n, LEG]) > 0.99):.1f}%)")
    j = int(adq.max(axis=0).argmax())
    print(f"  いちばん動く腕関節: {names[ARM.start + j]} "
          f"最大 {adq[:, j].max():.2f} rad/s / "
          f"可動幅 {np.ptp(real['q'][:n, ARM.start + j]):.2f} rad")
    if "dt_ms" in real:
        dt = real["dt_ms"][1:n]
        print(f"\n[6] ループ周期(実機) 中央 {np.median(dt):.1f} ms / "
              f"最大 {dt.max():.1f} ms (予算20ms)"
              f"  推論 中央 {np.median(real['ms_infer'][1:n]):.1f} ms")
    return dict(sim_q=sq, sim_tilt=st, sim_pelvis_z=sp, sim_seat_n=sn,
                sim_foot_n=fn, real_q=rq, real_tilt=rt, diff=d)


def save_and_plot(path, res, mode, do_plot):
    p = pathlib.Path(path)
    tag = {"policy": "P", "step": "S", "replay": "R"}[mode]
    out = p.parent / f"sim{tag}_{p.stem}.npz"
    np.savez_compressed(out, **res)
    print(f"\n保存: {out}")
    if not do_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(res["sim_tilt"])
        t = np.arange(n) / 50.0
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        ax[0].plot(t, res["real_tilt"], label="実機", lw=1.4)
        ax[0].plot(t, res["sim_tilt"], label="シム", lw=1.4)
        ax[0].set_ylabel("tilt [deg]"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot(t, res["sim_pelvis_z"], lw=1.4)
        ax[1].set_ylabel("pelvis z [m] (sim)"); ax[1].grid(alpha=.3)
        ax[2].plot(t, res["diff"][:, LEG].max(axis=1), label="脚")
        ax[2].plot(t, res["diff"][:, ARM].max(axis=1), label="腕")
        ax[2].set_ylabel("|sim-real| [rad]"); ax[2].set_xlabel("t [s]")
        ax[2].legend(); ax[2].grid(alpha=.3)
        png = out.with_suffix(".png")
        fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
        print(f"保存: {png}")
    except Exception as e:                         # noqa: BLE001
        print(f"(PNGは書けなかった: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="実機ログのnpz")
    ap.add_argument("--policy", default="", help="この方策の実機ログを対象にする")
    ap.add_argument("--all", action="store_true", help="全セッションから探す")
    ap.add_argument("--mode", default="step",
                    choices=("step", "policy", "replay"))
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    # ★実機の操縦中に走らせない。2026-08-26 11:40 に、このツールを回した
    #   直後の走行で制御ループが19Hzまで落ち、実機が転倒した。
    #   torchは既定で全コアを使うので、コックピットの50Hzループを食う。
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                              # noqa: BLE001
        pass
    import subprocess
    try:
        ps = subprocess.run(["ps", "-eo", "comm,args"], capture_output=True,
                            text=True, timeout=5).stdout
        live = [l for l in ps.splitlines()
                if l.startswith("python") and "cockpit.py" in l
                and "--sim" not in l]
        if live:
            print("★実機のコックピットが動いています。")
            print("  このツールはCPUを食うので、**走行中は実行しないこと**。")
            print("  2026-08-26 11:40: 直後の走行で制御ループが19Hzに落ち転倒。")
            print("  続けるなら走行の合間に。中断するなら Ctrl+C。")
            for l in live:
                print(f"    {l.strip()[:100]}")
    except Exception:                              # noqa: BLE001
        pass

    runs = []
    if a.run:
        runs = [pathlib.Path(a.run)]
    else:
        sess = sorted(LOGS.glob("cockpit_*"))
        if not a.all:
            sess = sess[-1:]
        for s in sess:
            for f in sorted(s.glob("run*_[0-9]_*.npz")):
                if a.policy and not f.stem.endswith(a.policy):
                    continue
                runs.append(f)
    if not runs:
        print("対象の実機ログが見つからない")
        return 1

    for f in runs:
        meta = run_meta(f)
        if meta.get("is_sim"):
            print(f"(skip: simの回) {f.name}")
            continue
        name = "_".join(f.stem.split("_")[2:])
        if not (ROOT / "deploy" / name / "reference.npz").exists():
            print(f"(skip: deployに {name} が無い) {f.name}")
            continue
        real = load_run(f)
        if real["n"] < 5:
            print(f"(skip: 短すぎる {real['n']}コマ) {f.name}")
            continue
        kp = np.array(meta.get("kp") or [0] * 29, float)
        kd = np.array(meta.get("kd") or [0] * 29, float)
        if kp.max() <= 0:
            print(f"(skip: 設定.jsonにkpが無い) {f.name}")
            continue
        sim = Sim(name)
        if a.mode == "policy":
            out = rollout_policy(sim, real, kp, kd, name)
        elif a.mode == "step":
            out = rollout_step(sim, real, kp, kd)
        else:
            print("★--mode replay は balance 課題では発散します"
                  "(モデル誤差ではなく発散を見ることになる)")
            out = rollout_replay(sim, real, kp, kd)
        res = summarize(f"{f.parent.name}/{f.name}", real, out, meta, a.mode)
        save_and_plot(f, res, a.mode, a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
