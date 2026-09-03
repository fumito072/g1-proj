#!/usr/bin/env python3
"""コックピットの結合試験用: MuJoCoが実機のふりをするモックロボット。

実機と同じインターフェース(state / set_target / set_damp)を持ち、
500Hz相当でPD制御しながら物理を進める。実機なしでコックピットの
FSM・観測構築・UI・安全系を通しで検証できる(WP6の「DDSモック」)。
"""
import pathlib
import threading
import time

import mujoco
import numpy as np

from real_robot import GUARD_RANGE_MARGIN, GUARD_STEP_MAX  # ガードは1箇所で決める

ROOT = pathlib.Path(__file__).resolve().parent.parent


class SimRobot:
    """実機と同じAPIを持つMuJoCoモック。realtime倍率は --speed で変更可"""

    def __init__(self, speed=1.0):
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "scene_task.xml"))
        self.d = mujoco.MjData(self.m)
        # 可動関節(29軸、DDS順=MJCF順)
        self.qadr, self.dofadr, self.acts, names = [], [], [], []
        for i in range(self.m.nu):
            jid = self.m.actuator_trnid[i, 0]
            lo, hi = self.m.jnt_range[jid]
            if lo == 0.0 and hi == 0.0:
                continue
            self.qadr.append(int(self.m.jnt_qposadr[jid]))
            self.dofadr.append(int(self.m.jnt_dofadr[jid]))
            self.acts.append(i)
            names.append(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        self.qadr = np.array(self.qadr)
        self.dofadr = np.array(self.dofadr)
        self.acts = np.array(self.acts)
        self.joint_names = names
        self.tau_lo = self.m.actuator_ctrlrange[self.acts, 0].copy()
        self.tau_hi = self.m.actuator_ctrlrange[self.acts, 1].copy()
        self.lock = threading.Lock()
        self.target_q = np.zeros(29)
        self.kp = np.zeros(29)
        self.kd = np.zeros(29)
        self.speed = float(speed)
        self.mode_machine = 5                       # 実機と同じ値を模す
        # 実機と同じく、起動しただけでは制御権を持たない(ensure_customで取る)。
        # Trueにしておくと実機では有り得ない「IDLEなのに制御権あり」になり、
        # 安全監視の掛かり方がsimと実機でずれる
        self.custom_active = False
        # 実機と同じ目標ガード(NaN拒否・可動域・変化量)を掛けるための下ごしらえ
        jids = self.m.actuator_trnid[self.acts, 0]
        self.q_lo = self.m.jnt_range[jids, 0].copy()
        self.q_hi = self.m.jnt_range[jids, 1].copy()
        self.guard_n_clip = 0
        self.guard_n_over = 0
        self.guard_over_max = 0.0
        self.guard_n_rate = 0
        self.guard_n_nan = 0
        self._nan_streak = 0
        self._estop_latched = False
        self._estop_why = ""
        self._renderer = None
        self._stop = False
        # 物理は壁時計と切り離し、エンジンが tick() で明示的に進める。
        # Windowsのsleep精度(〜15ms)では500Hzスレッドが実時間で回りきれず、
        # 方策の参照だけが先へ進む「スローモーション物理」になって転倒した(実測)。

    # ---- 実機と同じAPI
    def state(self):
        with self.lock:
            q = self.d.qpos[self.qadr].copy()
            dq = self.d.qvel[self.dofadr].copy()
            quat = self.d.qpos[3:7].copy()
            gyro = self.d.qvel[3:6].copy()
            tau = self.d.ctrl[self.acts].copy()
        return q, dq, quat, gyro, tau

    def set_target(self, q, kp, kd, latch=False):
        """実機と同じガードを掛ける(NaN拒否・可動域・変化量)。
        実機だけで発動するガードは、モックで手順を練習する意味がなくなる。"""
        if self._estop_latched:
            return False, "E-STOPラッチ中"
        q = np.asarray(q, dtype=float)
        kp = np.asarray(kp, dtype=float)
        kd = np.asarray(kd, dtype=float)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(kp))
                and np.all(np.isfinite(kd))):
            self.guard_n_nan += 1
            self._nan_streak += 1
            return False, f"NaN/Infを含む指令を拒否({self._nan_streak}回連続)"
        self._nan_streak = 0
        q = q.copy()
        over = float(np.maximum(np.max(q - self.q_hi), np.max(self.q_lo - q)))
        if over > 0:
            self.guard_n_over += 1
            self.guard_over_max = max(self.guard_over_max, over)
        lo = self.q_lo - GUARD_RANGE_MARGIN
        hi = self.q_hi + GUARD_RANGE_MARGIN
        n_out = int(np.count_nonzero((q < lo) | (q > hi)))
        if n_out:
            self.guard_n_clip += n_out
            q = np.clip(q, lo, hi)
        with self.lock:
            if not latch:
                d = q - self.target_q
                n_fast = int(np.count_nonzero(np.abs(d) > GUARD_STEP_MAX))
                if n_fast:
                    self.guard_n_rate += n_fast
                    q = self.target_q + np.clip(d, -GUARD_STEP_MAX,
                                                GUARD_STEP_MAX)
            self.target_q = q
            self.kp = kp.copy()
            self.kd = kd.copy()
        return True, ""

    def _set_damp_raw(self):
        kd = np.full(29, 5.0)
        kd[[4, 5, 10, 11]] = 0.2
        with self.lock:
            self.target_q = np.zeros(29)
            self.kp = np.zeros(29)
            self.kd = kd

    def set_damp(self):
        self._set_damp_raw()

    def estop(self, why="緊急停止"):
        """実機と同じ即時停止。ラッチが立ち、clear_estop まで指令を受けない"""
        self._estop_latched = True
        self._estop_why = why
        self._set_damp_raw()
        return True

    def clear_estop(self):
        was = self._estop_latched
        self._estop_latched = False
        self._estop_why = ""
        return was

    def healthy(self):
        return True

    def send_alive(self):
        return True, 0.0

    def motor_temps(self):
        return np.zeros(29)

    def state_full(self):
        """実機と同じ形。simで取れない量は0で埋める(列は揃える)"""
        with self.lock:
            ddq = self.d.qacc[self.dofadr].copy()
        return dict(ddq=ddq, vol=np.zeros(29), temps2=np.zeros(29),
                    mstate=np.zeros(29), mmode=np.ones(29),
                    accel=np.zeros(3), rpy=np.zeros(3), imu_temp=0.0,
                    tick=0, mode_pr=0, mode_machine=5,
                    remote=np.zeros(40), version=np.zeros(2),
                    ls_reserve=np.zeros(4), crc=0.0,
                    msensor=np.zeros((29, 2)), mreserve=np.zeros((29, 4)),
                    mot_ext=np.zeros((6, 5)))

    def current_mode(self):
        return "(sim)"

    def health_detail(self):
        return {"state_age_ms": 0.0, "send_hz_ok": True, "send_age_ms": 0.0,
                "send_n": 0, "send_err": 0, "send_err_msg": "",
                "estop_latched": bool(self._estop_latched),
                "guard_clip": int(self.guard_n_clip),
                "guard_rate": int(self.guard_n_rate),
                "guard_nan": int(self.guard_n_nan),
                "guard_over": int(self.guard_n_over),
                "guard_over_max": round(float(self.guard_over_max), 3)}

    def stop_move(self):
        """実機の stop_move と同じ入口(simでは何もしない)"""
        return

    def check_authority(self, jid=15, delta=0.06, dur=0.15):
        """実機の check_authority と同じ戻り値。simでは常に到達している"""
        return True, delta, 0.0, ""

    # ---- 標準モードのエミュレーション(実機はSDK。simは近似)
    def ensure_custom(self, kp=None, kd=None):
        if self._estop_latched:
            return False
        if kp is not None:
            q, _, _, _, _ = self.state()
            self.set_target(q, kp, kd, latch=True)  # 現姿勢保持で引き継ぐ
        self.custom_active = True
        return True

    def standard_mode(self, name):
        self.custom_active = False
        if name == "zero":
            self.set_target(np.zeros(29), np.zeros(29), np.zeros(29))
        elif name == "damp":
            self.set_damp()
        elif name in ("stand", "walk", "sit", "seated"):
            # 標準コントローラのバランス制御はsimに無い。
            # stand/walk/sit(スクワット)/seated(着座)は現姿勢PD保持で近似
            # (数秒で釣り合いを失う点に注意。simでは姿勢遷移そのものは起きない)
            q, _, _, _, _ = self.state()
            kp = np.full(29, 100.0)
            kd = np.full(29, 3.0)
            self.set_target(q, kp, kd, latch=True)
        return True

    def loco_move(self, vx, vy, omega):
        pass                                        # simでは非対応

    # ---- モック専用
    def place(self, qpos_joints, quat, xy, z):
        """指定姿勢に置く(コックピットの「開始位置に配置」用)"""
        with self.lock:
            mujoco.mj_resetData(self.m, self.d)
            self.d.qpos[0:2] = xy
            self.d.qpos[2] = z
            self.d.qpos[3:7] = quat
            self.d.qpos[self.qadr] = qpos_joints
            mujoco.mj_forward(self.m, self.d)
            self.target_q = np.asarray(qpos_joints, dtype=float).copy()

    def render_jpeg(self, size=480):
        import cv2
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, height=size, width=size)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [0, -0.15, 0.55]
            self._cam.distance, self._cam.elevation, self._cam.azimuth = 2.6, -10, 40
        with self.lock:
            self._renderer.update_scene(self.d, self._cam)
            img = self._renderer.render()
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""

    def close(self, flush=0.0):
        self._set_damp_raw()
        self._stop = True

    def tick(self, n_sub=10):
        """1制御ステップぶん(既定10サブステップ=20ms相当)物理を進める。
        エンジンの50Hzループから呼ぶ(論理時間。壁時計に依存しない)"""
        with self.lock:
            for _ in range(n_sub):
                q = self.d.qpos[self.qadr]
                dq = self.d.qvel[self.dofadr]
                tau = np.clip(self.kp * (self.target_q - q) - self.kd * dq,
                              self.tau_lo, self.tau_hi)
                self.d.ctrl[self.acts] = tau
                mujoco.mj_step(self.m, self.d)
