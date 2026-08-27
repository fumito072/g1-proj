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

ROOT = pathlib.Path(__file__).parent.parent


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
        self.custom_active = True
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

    def set_target(self, q, kp, kd):
        with self.lock:
            self.target_q = np.asarray(q, dtype=float).copy()
            self.kp = np.asarray(kp, dtype=float).copy()
            self.kd = np.asarray(kd, dtype=float).copy()

    def set_damp(self):
        kd = np.full(29, 5.0)
        kd[[4, 5, 10, 11]] = 0.2
        self.set_target(np.zeros(29), np.zeros(29), kd)

    def healthy(self):
        return True

    def stop_move(self):
        """実機の stop_move と同じ入口(simでは何もしない)"""
        return

    def check_authority(self, jid=15, delta=0.06, dur=0.15):
        """実機の check_authority と同じ戻り値。simでは常に到達している"""
        return True, delta, 0.0, ""

    # ---- 標準モードのエミュレーション(実機はSDK。simは近似)
    def ensure_custom(self, kp=None, kd=None):
        if kp is not None:
            q, _, _, _, _ = self.state()
            self.set_target(q, kp, kd)             # 現姿勢保持で引き継ぐ
        self.custom_active = True

    def standard_mode(self, name):
        self.custom_active = False
        if name == "zero":
            self.set_target(np.zeros(29), np.zeros(29), np.zeros(29))
        elif name == "damp":
            self.set_damp()
        elif name in ("stand", "walk"):
            # 標準コントローラのバランス制御はsimに無い。
            # standは現姿勢PD保持で近似(数秒で釣り合いを失う点に注意)
            q, _, _, _, _ = self.state()
            kp = np.full(29, 100.0)
            kd = np.full(29, 3.0)
            self.set_target(q, kp, kd)
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

    def close(self):
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
