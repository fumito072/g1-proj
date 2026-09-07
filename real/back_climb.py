#!/usr/bin/env python3
"""後ろ向き段差登り(`G1_後ろ向き段差登り_一式`、2026-08-21)の方策を本流コックピットで動かすための部品。

元の実行系 `06_コード/5_実機/g1_live/` の observe.py / estimator.py / policy.py を、コックピットの
`Policy` / `ObsBuilder` と同じ顔(reset / build / act)に合わせて移植したもの(2026-09-07)。

  NumpyPolicy    観測正規化 + 183 → 512 → ELU → 256 → ELU → 29 の MLP(PyTorch 不要)
  BaseEstimator  脚オドメトリ。IMU で予測し、接地足の「錨」で位置と速度を補正する
  BackClimbObs   学習環境 rl/motion_env.py の _observe と同じ 183 次元(履歴なし)

観測のうち実機で直接取れない量(体幹の高さ・速度・水平位置、接地、荷重配分)は推定で作る。
接地は「参照のスケジュールを前提に、運動学で拒否権」(トルクからの推定は当たらなかった)。
足元の地形は、推定した足の位置で本流の MuJoCo シーン(model/scene_task.xml、椅子と段)へ
光線を落として測る。学習シーンと同じファイルなので、段の位置・高さは学習時と一致する。

★方策が学習で見た区間は 0〜260 コマ(exec_hi)。それ以降へ進めると未知の観測になって転倒する
  (元の実験: 素通し 0/10、260 で凍結 9/10)ので、Policy.n = 261 にして 260 で凍結・保持する。
★ゲインは学習値(脚 kp400/kd12、腕腰 kp70/kd2.5)。公式例の 4〜6 倍固い。剛性を下げたら
  膝が体重を支えられず後ろへ倒れた(元の実験報告 2026-08-20)。
"""
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ACT_BETA = 0.5             # 行動のローパス(学習 train_back_stages.py と同じ)
CHAIR_GEOMGROUP = np.zeros(6, np.uint8)
CHAIR_GEOMGROUP[3] = 1      # 椅子・段の当たり判定 geom(class chair_col、group 3)


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


class NumpyPolicy:
    """学習側の ActorCritic.pi と RunningNorm を numpy で再現(deterministic)"""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.mean = z["norm_mean"].astype(np.float64)
        self.var = z["norm_var"].astype(np.float64)
        self.obs_dim = int(z["obs_dim"])
        self.act_dim = int(z["act_dim"])
        self.steps = int(z["steps"])
        self.layers = []
        i = 0
        while f"pi{i}.weight" in z:
            self.layers.append((z[f"pi{i}.weight"].astype(np.float32), z[f"pi{i}.bias"].astype(np.float32)))
            i += 2
        if not self.layers:
            raise RuntimeError(f"方策の重みが見つかりません: {npz_path}")

    def act(self, obs):
        x = np.clip((np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)
        x = x.astype(np.float32)
        for j, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if j < len(self.layers) - 1:
                x = elu(x)
        return x


def _quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _yaw_of(q):
    R = _quat_to_mat(q)
    return math.atan2(R[1, 0], R[0, 0])


class BaseEstimator:
    """体幹の位置・速度を、IMU と脚の運動学だけから推定する(脚オドメトリ)。元の estimator.py と同じ"""

    G = 9.81

    def __init__(self, model, data, fid, sole_off=0.035, dt=0.02):
        import mujoco
        self.mj = mujoco
        self.model, self.data = model, data
        self.fid = fid
        self.sole_off = sole_off
        self.dt = dt
        self.qadr = np.array([model.jnt_qposadr[model.dof_jntid[d]] for d in range(6, model.nv)])
        self.dofadr = np.arange(6, model.nv)
        self.leg_dofs = [self.dofadr[np.arange(0, 6)], self.dofadr[np.arange(6, 12)]]
        self.leg_joints = [np.arange(0, 6), np.arange(6, 12)]
        self._jacp = np.zeros((3, model.nv))
        import os
        self.kv = float(os.environ.get("BC_KV", 1.0))        # 接地足の速度で引く強さ(1.0=接地中は脚オドメトリの速度を全面的に信じる。0.6 だと 13/20、1.0 で 19/20。2026-09-07 シム)
        self.kp_ = float(os.environ.get("BC_KP", 0.5))       # 接地足の錨で位置を引く強さ
        self.use_imu = os.environ.get("BC_IMU", "1") == "1"   # IMU 加速度で予測するか
        self.v_lp = float(os.environ.get("BC_VLP", 0.0))     # 速度観測のローパス(0=なし)
        self.reset()

    def reset(self, p=(0.0, 0.0, 0.79), v=(0.0, 0.0, 0.0)):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.anchor = [None, None]
        self.conf = np.zeros(2)

    def kin(self, q, quat):
        """体幹を原点に置いたときの 重心・足の位置(R·p)・脚ヤコビアン"""
        d, m = self.data, self.model
        self.mj.mj_resetData(m, d)
        d.qpos[0:3] = 0.0
        d.qpos[3:7] = quat
        d.qpos[self.qadr] = q
        self.mj.mj_forward(m, d)
        feet, jac = [], []
        for k in (0, 1):
            feet.append(d.xpos[self.fid[k]].copy())
            self.mj.mj_jacBody(m, d, self._jacp, None, self.fid[k])
            jac.append(self._jacp[:, self.leg_dofs[k]].copy())
        com = d.subtree_com[1].copy()
        return com, feet, jac

    def contact_conf(self, feet, ground_z, ref_contact=None, h_on=0.06, h_off=0.18):
        conf = np.zeros(2)
        for k in (0, 1):
            prior = 1.0 if ref_contact is None else float(ref_contact[k])
            h = (self.p[2] + feet[k][2] - self.sole_off) - ground_z[k]
            gate = float(np.clip(1.0 - (h - h_on) / (h_off - h_on), 0.0, 1.0))
            conf[k] = prior * gate
        if conf.sum() <= 1e-6 and ref_contact is not None:
            k = int(np.argmin([self.p[2] + feet[i][2] - ground_z[i] for i in (0, 1)]))
            conf[k] = 0.5
        return conf

    def update(self, q, qd, quat, gyro, acc_imu, ground_z, ref_contact=None):
        R = _quat_to_mat(quat)
        com, feet, jac = self.kin(q, quat)
        conf = self.contact_conf(feet, ground_z, ref_contact)
        w = R @ np.asarray(gyro, dtype=float)
        qdl = np.asarray(qd, dtype=float)
        a_world = (R @ np.asarray(acc_imu, dtype=float) - np.array([0.0, 0.0, self.G])) if self.use_imu else np.zeros(3)
        self.v = self.v + a_world * self.dt
        self.p = self.p + self.v * self.dt
        tot = conf.sum()
        if tot > 1e-6:
            v_meas = np.zeros(3)
            p_meas = np.zeros(3)
            for k in (0, 1):
                if conf[k] <= 0:
                    continue
                rp = feet[k]
                vk = -(np.cross(w, rp) + jac[k] @ qdl[self.leg_joints[k]])
                v_meas += conf[k] * vk
                if self.anchor[k] is None:
                    a = self.p + rp
                    a[2] = ground_z[k] + self.sole_off
                    self.anchor[k] = a
                p_meas += conf[k] * (self.anchor[k] - rp)
            v_meas /= tot
            p_meas /= tot
            kv = self.kv * min(tot, 1.0)
            kp_ = self.kp_ * min(tot, 1.0)
            if self.v_lp > 0:
                v_meas = self.v_lp * self.v + (1 - self.v_lp) * v_meas
            self.v += kv * (v_meas - self.v)
            self.p += kp_ * (p_meas - self.p)
        for k in (0, 1):
            if conf[k] <= 0:
                self.anchor[k] = None
        self.conf = conf
        return self.p.copy(), self.v.copy(), conf, feet, com


class BackClimbObs:
    """コックピットの ObsBuilder と同じ顔(reset / build / last_cmd / pitch_bias / est_xy)で 183 次元を作る"""

    def __init__(self, policy, robot=None, control_hz=50.0):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "scene_task.xml"))
        self.data = mujoco.MjData(self.model)
        self.robot = robot
        self.ref = policy.ref
        self.ref_len = int(self.ref["ref_len"]) if "ref_len" in self.ref.files else len(self.ref["ref_q"])
        self.exec_hi = int(getattr(policy, "exec_hi", self.ref_len - 1))
        self.hz = control_hz
        self.dt = 1.0 / control_hz
        self.fid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.sole_off = 0.035
        self.est = BaseEstimator(self.model, self.data, self.fid, self.sole_off, self.dt)
        self.last_cmd = np.zeros(29)          # コックピットが代入する(この観測では使わない)
        self.pitch_bias = 0.0                 # 着座用の観測バイアス。登りでは使わない
        self.hist = None
        self.yaw_off = 0.0
        self._yaw_fix = np.array([1.0, 0.0, 0.0, 0.0])
        self.start_xy = np.zeros(2)
        self.vel = np.zeros(3)
        self.ground_z = np.zeros(2)
        self.conf = np.zeros(2)
        self.reset()

    # ---- コックピットの ObsBuilder と同じ入口
    def reset(self, est_xy=(0.0, 0.0), quat=None, ref_quat=None):
        """参照の開始位置に機体を置いたものとして推定を初期化する。quat/ref_quat があればヨーを参照へ揃える。
        戻り値: ヨーのずれ[rad](コックピットがログに出す)"""
        xy0 = np.asarray(self.ref["ref_xy_abs"][0][:2], dtype=float)
        self.start_xy = xy0 - np.asarray(self.ref["ref_xy"][0][:2], dtype=float)   # 学習シーンの世界オフセット
        self.est.reset(p=(xy0[0], xy0[1], float(self.ref["ref_z"][0])))
        self.yaw_off = 0.0
        self._yaw_fix = np.array([1.0, 0.0, 0.0, 0.0])
        if quat is not None and ref_quat is not None:
            dy = _yaw_of(ref_quat) - _yaw_of(quat)
            dy = (dy + math.pi) % (2 * math.pi) - math.pi
            self.yaw_off = float(dy)
            self._yaw_fix = np.array([math.cos(dy / 2), 0.0, 0.0, math.sin(dy / 2)])
        self.vel = np.zeros(3)
        self.ground_z = np.zeros(2)
        self.conf = np.zeros(2)
        return self.yaw_off

    @property
    def est_xy(self):
        return self.est.p[:2].copy()

    # ---- 地形(推定した足の位置で、椅子と段へ光線を落とす)
    def terrain_z(self, x, y, z_from):
        gid = np.zeros(1, np.int32)
        z0 = float(z_from) + 0.12
        dist = self.mj.mj_ray(self.model, self.data, np.array([float(x), float(y), z0]),
                              np.array([0.0, 0.0, -1.0]), CHAIR_GEOMGROUP, 1, -1, gid)
        return max(0.0, z0 - dist) if dist >= 0 else 0.0

    def build(self, pol, t, q, dq, quat, gyro, tau=None, acc=None):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(dq, dtype=float)
        quat_a = _quat_mul(self._yaw_fix, np.asarray(quat, dtype=float))   # 参照の世界へヨーを回す
        quat_a = quat_a / max(np.linalg.norm(quat_a), 1e-9)
        R = _quat_to_mat(quat_a)
        t = int(np.clip(t, 0, self.exec_hi))
        rc = [float(self.ref["ref_contact"][t, k]) for k in (0, 1)]
        # 足元の地形は「前のコマの推定位置」から。開始直後は参照の床(0)
        p_prev = self.est.p
        _com, feet, _jac = self.est.kin(q, quat_a)
        gz = [self.terrain_z(p_prev[0] + feet[k][0], p_prev[1] + feet[k][1], p_prev[2] + feet[k][2])
              for k in (0, 1)]
        self.ground_z = np.array(gz)
        if acc is None:
            acc = R.T @ np.array([0.0, 0.0, BaseEstimator.G])    # 加速度計が無い(モック)ときは等速とみなす
        p, v, conf, feet, com = self.est.update(q, qd, quat_a, gyro, acc, gz, ref_contact=rc)
        self.vel = v
        self.conf = conf
        base = p
        lf = feet[0] + base
        rf = feet[1] + base
        com_w = com + base
        # ★学習環境の「接地」フラグは、床(task_floor)との接触を数えない癖がある(robot_bodies に world が入っていて
        #   足×床が「ロボット同士」扱いになる)。段・椅子との接触だけが 1 になる。方策はその意味で学習したので、
        #   ここでも「足が段(地形の高さ 0.1m 超)に乗っている」ときだけ 1 にする(2026-09-07 パリティ検証)
        contact = [bool(conf[k] > 0.5 and gz[k] > 0.10) for k in (0, 1)]
        # 荷重配分は接触力の比(実機では取れない)。準静的に、両足の間での重心の位置の比で代用する。
        # 片足支持(確信度で判定)なら 1/0
        if conf[0] > 0.5 and conf[1] <= 0.5:
            load_ratio = 1.0
        elif conf[1] > 0.5 and conf[0] <= 0.5:
            load_ratio = 0.0
        elif "ref_load" in self.ref.files:
            # ★学習環境の荷重配分は、段の上で左足の内部接触力(1700N 級)が二重に数えられる物理の癖で 0.9 前後になる。
            #   実機では測れない量なので、学習環境を真の観測で駆動した 6 本の位相ごとの中央値(ref_load)を使う
            load_ratio = float(np.clip(self.ref["ref_load"][t], 0.0, 1.0))
        else:
            dlr = lf[:2] - rf[:2]
            L2 = float(dlr @ dlr)
            load_ratio = 0.5 if L2 < 1e-6 else float(np.clip(((com_w[:2] - rf[:2]) @ dlr) / L2, 0.0, 1.0))
        ahead = [min(t + k, self.ref_len - 1) for k in (0, int(0.15 * self.hz), int(0.4 * self.hz))]
        ref_q = self.ref["ref_q"]
        # 支持の中心も学習環境と同じ癖(段に乗っている足だけ。無ければ両足の中点)
        pts = [f[:2] for f, c in ((lf, contact[0]), (rf, contact[1])) if c]
        sup = np.mean(pts, axis=0) if pts else (lf[:2] + rf[:2]) / 2
        rq = _quat_to_mat(self.ref["ref_quat"][t])
        h = max(com_w[2] - max(gz), 0.15)
        dcm = com_w[:2] + v[:2] / math.sqrt(BaseEstimator.G / h)
        obs = np.concatenate([
            [base[2] - float(self.ref["ref_z"][t])],
            R[:2].reshape(-1),
            v * 0.3,
            np.asarray(gyro, dtype=float) * 0.2,
            q,
            qd * 0.1,
            *[ref_q[a] - q for a in ahead],
            rq[:2].reshape(-1),
            (np.asarray(self.ref["ref_xy"][t][:2], dtype=float) + self.start_xy) - base[:2],
            [t / self.ref_len],
            R.T @ (com_w - base),
            R.T @ (lf - base),
            R.T @ (rf - base),
            R.T[:2, :2] @ (com_w[:2] - sup),
            R.T[:2, :2] @ (dcm - sup),
            [float(contact[0]), float(contact[1])],
            [load_ratio],
        ]).astype(np.float32)
        return obs
