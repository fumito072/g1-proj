"""GPU(mjlab / BeyondMimic 追従)で学習した後ろ向き登りの方策を実機で回すための観測と方策(2026-09-07 夕)。

観測 160 次元(mjlab の tracking 課題の actor 群、この順):
  ref_q[t] 29 / ref_qd[t] 29                      … 参照の関節角・関節速度(現在コマ)
  anchor_pos_b 3 / anchor_ori_b 6                  … 参照の胴(torso_link)の位置・姿勢を、機体の胴の座標系で見た値
                                                     (姿勢は回転行列の先頭 2 列)
  imu_lin_vel 3 / gyro 3                           … 骨盤 IMU サイト(imu_in_pelvis)の速度計・ジャイロ(サイト系 = 骨盤系)
  q − default_q 29 / dq 29                          … mjlab の既定姿勢(膝曲げ立位)からの差、関節速度
  last_action 29                                   … 前回の方策出力(スケール前)
行動: 目標角 = ref_q[t] + a × action_scale_v(脚腰 0.70 / 腕 0.20)。低域通過なし。
機体の骨盤の位置・速度は back_climb.BaseEstimator(脚オドメトリ + IMU)で推定し、胴の姿勢は MuJoCo の順運動学で出す。
世界座標は BackClimbObs と同じ(参照の開始位置に置いた前提、開始時にヨーを参照へ揃える)。
"""
from __future__ import annotations

import math

import numpy as np

from back_climb import BackClimbObs, BaseEstimator, _quat_mul, _quat_to_mat


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


class NumpyMlp:
    """rsl_rl の MLPModel(EmpiricalNormalization → Linear/ELU × 3 → Linear)を numpy で。決定論的(平均)出力"""

    def __init__(self, path):
        with np.load(path, allow_pickle=True) as z:
            n = int(z["n_layers"]) if "n_layers" in z.files else 4
            self.W = [np.asarray(z[f"W{i}"], dtype=np.float64) for i in range(n)]
            self.b = [np.asarray(z[f"b{i}"], dtype=np.float64) for i in range(n)]
            self.mean = np.asarray(z["obs_mean"], dtype=np.float64)
            self.std = np.asarray(z["obs_std"], dtype=np.float64)
            self.eps = float(z["obs_eps"]) if "obs_eps" in z.files else 1e-2
            self.family = str(z["family"]) if "family" in z.files else "back_climb_mjlab"
        self.obs_dim = self.W[0].shape[1]
        self.act_dim = self.W[-1].shape[0]

    def __call__(self, obs):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / (self.std + self.eps)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = W @ x + b
            if i < len(self.W) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))     # ELU(alpha=1)
        return x


class MjlabClimbObs(BackClimbObs):
    """BackClimbObs の推定(脚オドメトリ・地形)をそのまま使い、観測だけ mjlab の 160 次元にする"""

    OBS_DIM = 160

    def __init__(self, policy, robot=None, control_hz=50.0):
        super().__init__(policy, robot, control_hz)
        z = self.ref
        self.default_q = np.asarray(z["default_q"], dtype=float)
        self.imu_site = np.asarray(z["imu_site_pos"], dtype=float) if "imu_site_pos" in z.files else np.array([0.04525, 0.0, -0.08339])
        self.ref_torso_pos = np.asarray(z["ref_torso_pos"], dtype=float)
        self.ref_torso_quat = np.asarray(z["ref_torso_quat"], dtype=float)
        self.tid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "torso_link")
        self.pid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "pelvis")
        self.last_cmd = np.zeros(29)
        self.debug = {}

    def torso_pose(self, p, quat_a, q):
        """骨盤の位置・姿勢と関節角から胴(torso_link)の世界位置・姿勢(順運動学)"""
        d = self.data
        d.qpos[0:3] = p
        d.qpos[3:7] = quat_a
        d.qpos[7:7 + len(q)] = q
        self.mj.mj_kinematics(self.model, d)
        return d.xpos[self.tid].copy(), d.xquat[self.tid].copy()

    def build(self, pol, t, q, dq, quat, gyro, tau=None, acc=None, p_true=None, v_true=None):
        """p_true / v_true を渡すと推定の代わりにその値を使う(学習環境とのパリティ確認用)"""
        q = np.asarray(q, dtype=float)
        qd = np.asarray(dq, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        quat_a = _quat_mul(self._yaw_fix, np.asarray(quat, dtype=float))
        quat_a = quat_a / max(np.linalg.norm(quat_a), 1e-9)
        R = _quat_to_mat(quat_a)
        t = int(np.clip(t, 0, self.exec_hi))
        # 推定(BackClimbObs と同じ手順。地形は前のコマの推定位置から)
        p_prev = self.est.p
        _com, feet, _jac = self.est.kin(q, quat_a)
        gz = [self.terrain_z(p_prev[0] + feet[k][0], p_prev[1] + feet[k][1], p_prev[2] + feet[k][2]) for k in (0, 1)]
        self.ground_z = np.array(gz)
        if acc is None:
            acc = R.T @ np.array([0.0, 0.0, BaseEstimator.G])
        rc = [float(self.ref["ref_contact"][t, k]) for k in (0, 1)] if "ref_contact" in self.ref.files else None
        p, v, conf, feet, _com2 = self.est.update(q, qd, quat_a, gyro, acc, gz, ref_contact=rc)
        self.vel = v
        self.conf = conf
        if p_true is not None:
            p = np.asarray(p_true, dtype=float)
        if v_true is not None:
            v = np.asarray(v_true, dtype=float)
        # 胴(anchor)の相対姿勢
        tp, tq = self.torso_pose(p, quat_a, q)
        q_inv = _quat_conj(tq)
        rel_p = _quat_to_mat(q_inv) @ (self.ref_torso_pos[t] - tp)
        rel_q = _quat_mul(q_inv, self.ref_torso_quat[t])
        M = _quat_to_mat(rel_q / max(np.linalg.norm(rel_q), 1e-9))
        ori6 = M[:, :2].reshape(-1)                      # 先頭 2 列 [m00 m01 m10 m11 m20 m21]
        # 骨盤 IMU サイトの速度計(サイト系 = 骨盤系): v_site = v + ω × (R r_site)
        w_world = R @ gyro
        v_site = R.T @ (v + np.cross(w_world, R @ self.imu_site))
        obs = np.concatenate([
            self.ref["ref_q"][t], self.ref["ref_qd"][t],
            rel_p, ori6,
            v_site, gyro,
            q - self.default_q, qd,
            np.asarray(self.last_cmd, dtype=float),
        ]).astype(np.float32)
        self.debug = dict(p=p.copy(), v=v.copy(), torso_p=tp, rel_p=rel_p.copy())
        return obs
