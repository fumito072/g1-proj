#!/usr/bin/env python3
"""着座の「形」を数値にする: 座面接触の時刻 / 骨盤の後退 / 膝の前方位置 / ヨー(回転)。

2026-09-04 の解析で、「座る直前に膝が前へ出る」「座りながら回転する」の2つは
次の量で言い切れることが分かった(docs/着座_膝と回転の解析_20260904.md):

  座面接触   両膝の保持トルクが降下中ピークの40%を切った時刻。実機は 0.9〜1.2秒、
             参照(骨盤高さが座面到達)は 1.84秒 — **0.6〜0.9秒早く座面に着く**
  骨盤の後退 足首中点から骨盤までの前後距離(ヨーを除いた水平面。足首固定の仮定)。
             参照は 30.5cm 後ろまで下がるが、実機は接触時に 25〜35cm、終端は 13〜28cm
  膝の前方   足首に対する膝の前後位置。参照の終端は 0(スネが鉛直)、実機は +6〜22cm
             = 「膝が出た」姿勢。骨盤が後ろへ行き切らないまま座面に乗るので、膝が前へ出る
  ヨー       IMUの向きの変化。足が固定なら 体のヨー = -(左右の股ヨーの平均変化) になる
             はずで、その差が**足の滑り(または尻の旋回)**。実機は接触前後で半分ずつ、
             滑り分が 11本中9本で同じ向き(時計回り)

  python3 real/sit_shape.py                     # 最新セッション
  python3 real/sit_shape.py --policy sit_up_ln23_r2 --all
  python3 real/sit_shape.py --run <npz>

コックピットは走行直後に同じ関数(shape_metrics)で計算し、イベントログと[走行の統計]に出す。
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
CONTACT_TAU_RATIO = 0.40     # 膝トルク合計が降下中ピークのこの割合を切ったら座面に乗った
CONTACT_T_MIN = 45           # これより前(0.9秒)の落ち込みは接触とみなさない
CONTACT_PEAK_WINDOW = (25, 100)


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SagittalFK:
    """骨盤・膝の、足首中点に対する前後位置(ヨーを除いた水平面)。MuJoCoはFKだけに使う。

    既に読んであるモデル(ObsBuilder の m/d/qadr)を渡せば二重に読まない。
    """

    def __init__(self, m=None, d=None, qadr=None, joint_names=None):
        import mujoco
        self.mj = mujoco
        if m is None:
            m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "scene_task.xml"))
            d = mujoco.MjData(m)
            qadr = []
            for nm in joint_names:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_joint")
                if jid < 0:
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
                qadr.append(int(m.jnt_qposadr[jid]))
        self.m, self.d, self.qadr = m, d, np.asarray(qadr)
        b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)   # noqa: E731
        self.bid = dict(pelvis=b("pelvis"), kl=b("left_knee_link"), kr=b("right_knee_link"),
                        al=b("left_ankle_roll_link"), ar=b("right_ankle_roll_link"))

    def __call__(self, q, quat):
        d, m, mj = self.d, self.m, self.mj
        mj.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, 1.0]
        d.qpos[3:7] = quat
        d.qpos[self.qadr] = q
        mj.mj_kinematics(m, d)
        P = {k: d.xpos[v].copy() for k, v in self.bid.items()}
        ank = (P["al"] + P["ar"]) / 2
        yaw = yaw_of(quat)
        c, s = math.cos(-yaw), math.sin(-yaw)

        def rel(p, origin):
            v = p - origin
            return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
        pel = rel(P["pelvis"], ank)
        kl = rel(P["kl"], P["al"])
        kr = rel(P["kr"], P["ar"])
        return dict(pelvis_back=-float(pel[0]), pelvis_h=float(pel[2]),
                    knee_fwd=float((kl[0] + kr[0]) / 2),
                    knee_fwd_l=float(kl[0]), knee_fwd_r=float(kr[0]))


def contact_frame(tau, n):
    """両膝トルク合計の落ち込みで座面接触のコマを推定。取れなければ None"""
    kt = np.abs(tau[:, 3]) + np.abs(tau[:, 9])
    a, b = CONTACT_PEAK_WINDOW
    if n <= a + 5:
        return None
    pk = float(kt[a:min(b, n)].max())
    if pk <= 1e-6:
        return None
    idx = np.where((kt < CONTACT_TAU_RATIO * pk) & (np.arange(n) >= CONTACT_T_MIN))[0]
    return int(idx[0]) if len(idx) else None


def shape_metrics(q, tau, quat, t_ref, ref_q, ref_quat, fk):
    """1走行ぶんの指標。q/tau/quat は (n,29)/(n,29)/(n,4)、t_ref は各コマの参照コマ番号。

    戻り値(単位: 秒 / m / 度):
      tc_s       座面接触(実機)   ref_tc_s  参照の座面到達(骨盤高さが最低+1cm)
      back_c / back_e   骨盤の後退 接触時/終端      ref_back_e  参照の終端
      kneex_c / kneex_e 膝の前方 接触時/終端        ref_kneex_e 参照の終端
      kdev_e     終端の膝角 実測−参照(度)
      yaw_c / yaw_e     ヨー変化 接触時/終端       slip_e  足固定で説明できない分(終端)
    """
    D = 180.0 / math.pi
    n = len(q)
    out = dict(n=n)
    tc = contact_frame(tau, n)
    out["tc_s"] = None if tc is None else round(tc / 50.0, 2)
    ref_n = len(ref_q)
    # 参照側: 座面到達 = 骨盤高さが終端+1cm以内へ入った最初のコマ
    ref_h = np.array([fk(ref_q[k], ref_quat[k])["pelvis_h"] for k in range(0, ref_n, 5)])
    kk = np.arange(0, ref_n, 5)
    idx = np.where(ref_h <= ref_h[-1] + 0.01)[0]
    out["ref_tc_s"] = round(float(kk[idx[0]]) / 50.0, 2) if len(idx) else None
    fe = fk(ref_q[ref_n - 1], ref_quat[ref_n - 1])
    out["ref_back_e"] = round(fe["pelvis_back"], 3)
    out["ref_kneex_e"] = round(fe["knee_fwd"], 3)
    # 実機側
    k_e = n - 1
    e = fk(q[k_e], quat[k_e])
    out["back_e"] = round(e["pelvis_back"], 3)
    out["kneex_e"] = round(e["knee_fwd"], 3)
    tr = int(min(t_ref[k_e], ref_n - 1))
    out["kdev_e"] = round(float((q[k_e, 3] + q[k_e, 9] - ref_q[tr, 3] - ref_q[tr, 9]) / 2 * D), 1)
    yaw0 = yaw_of(quat[0])
    out["yaw_e"] = round(wrap(yaw_of(quat[k_e]) - yaw0) * D, 1)
    dhy = float(((q[k_e, 2] + q[k_e, 8]) - (q[0, 2] + q[0, 8])) / 2 * D)
    out["slip_e"] = round(out["yaw_e"] + dhy, 1)          # 足固定なら yaw = -dhy
    if tc is not None:
        c = fk(q[tc], quat[tc])
        out["back_c"] = round(c["pelvis_back"], 3)
        out["kneex_c"] = round(c["knee_fwd"], 3)
        out["yaw_c"] = round(wrap(yaw_of(quat[tc]) - yaw0) * D, 1)
    else:
        out["back_c"] = out["kneex_c"] = out["yaw_c"] = None
    return out


def describe(mt):
    """イベントログ向けの1行"""
    f = lambda v, s=100.0, u="": ("--" if v is None else f"{v * s:+.0f}{u}")   # noqa: E731
    return (f"着座の形: 座面接触 {mt['tc_s'] if mt['tc_s'] is not None else '--'}秒"
            f"(参照{mt['ref_tc_s']}) / 骨盤の後退 接触時{f(mt['back_c'])}→終端{f(mt['back_e'])}cm"
            f"(参照{f(mt['ref_back_e'])}) / 膝の前方 終端{f(mt['kneex_e'])}cm"
            f"(参照{f(mt['ref_kneex_e'])}) / 膝角 参照より{mt['kdev_e']:+.0f}度"
            f" / ヨー 接触時{mt['yaw_c'] if mt['yaw_c'] is not None else '--'}度→終端"
            f"{mt['yaw_e']:+.0f}度(うち足の滑り{mt['slip_e']:+.0f}度)")


# ---------------------------------------------------------------- CLI
def _load(npz_path):
    z = np.load(npz_path)
    cols = [str(c) for c in z["cols"]]
    rec = z["rec"]
    rec = rec[rec[:, cols.index("fsm")].astype(int) == 3]
    g = lambda p: [cols.index(f"{p}{i}") for i in range(29)]      # noqa: E731
    qw = cols.index("quat_w")
    return dict(q=rec[:, g("q")], tau=rec[:, g("tau")], quat=rec[:, qw:qw + 4],
                t=rec[:, cols.index("t")].astype(int), n=len(rec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--policy", default="")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    logs = ROOT / "logs" / "real"
    if a.run:
        files = [pathlib.Path(a.run)]
    else:
        sess = sorted(logs.glob("cockpit_*"))
        if not a.all:
            sess = sess[-1:]
        files = [f for s in sess for f in sorted(s.glob("run*_[0-9]_*.npz"))]
    fks, refs = {}, {}
    print(f"{'run':>30s} {'コマ':>4s} {'接触s':>5s} {'参照':>4s} {'後退c':>5s} {'後退e':>5s} {'参照':>5s}"
          f" {'膝前c':>5s} {'膝前e':>5s} {'参照':>5s} {'膝角Δ':>5s} {'yaw_c':>6s} {'yaw_e':>6s} {'滑り':>5s}")
    for f in files:
        name = "_".join(f.stem.split("_")[2:])
        meta = f.parent / f"{f.stem.split('_')[0]}_設定.json"
        if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("is_sim"):
            continue
        rp = ROOT / "deploy" / name / "reference.npz"
        if not rp.exists():
            continue
        if name not in refs:
            z = np.load(rp)
            refs[name] = {k: z[k] for k in ("ref_q", "ref_quat", "joint_names")}
            fks[name] = SagittalFK(joint_names=[str(s) for s in refs[name]["joint_names"]])
        r = _load(f)
        if r["n"] < 60:
            continue
        mt = shape_metrics(r["q"], r["tau"], r["quat"], r["t"], refs[name]["ref_q"],
                           refs[name]["ref_quat"], fks[name])
        cm = lambda v: "  --" if v is None else f"{v * 100:+4.0f}"      # noqa: E731
        print(f"{(f.parent.name[8:] + '/' + f.stem[:6]):>30s} {r['n']:4d}"
              f" {mt['tc_s'] if mt['tc_s'] is not None else '  --':>5} {mt['ref_tc_s']:4}"
              f" {cm(mt['back_c']):>5s} {cm(mt['back_e']):>5s} {cm(mt['ref_back_e']):>5s}"
              f" {cm(mt['kneex_c']):>5s} {cm(mt['kneex_e']):>5s} {cm(mt['ref_kneex_e']):>5s}"
              f" {mt['kdev_e']:+5.0f} {mt['yaw_c'] if mt['yaw_c'] is not None else '--':>6}"
              f" {mt['yaw_e']:+6.1f} {mt['slip_e']:+5.0f}   {name}")
    print("単位: 秒 / cm / 度。後退=足首中点→骨盤の後ろ向き距離、膝前=足首→膝の前向き距離、"
          "滑り=足固定では説明できないヨー")
    return 0


if __name__ == "__main__":
    sys.exit(main())
