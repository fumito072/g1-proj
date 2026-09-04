#!/usr/bin/env python3
"""コックピットの結合試験用: MuJoCoが実機のふりをするモックロボット。

実機と同じインターフェース(state / set_target / set_damp)を持ち、
500Hz相当でPD制御しながら物理を進める。実機なしでコックピットの
FSM・観測構築・UI・安全系を通しで検証できる(WP6の「DDSモック」)。
"""
import math
import pathlib
import threading
import time

import mujoco
import numpy as np

from real_robot import GUARD_RANGE_MARGIN, GUARD_STEP_MAX  # ガードは1箇所で決める

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class _SimLidar:
    """内蔵歩行モックの合成LiDAR。世界座標(frame_id="odom")の点群を返す。

    床(z=0)と箱の障害物(前面・側面・天面)を毎回サンプルする。実機と同じ
    「世界座標 → オドメトリ+ヨーで体基準へ」の経路を通すための最低限。
    """

    def __init__(self, robot):
        self.r = robot
        self.enabled = False
        self.n_recv = 0
        self.frame_id = "odom"

    def enable(self, on):
        self.enabled = bool(on)

    def stop(self):
        pass

    def latest(self):
        if not self.enabled:
            return None
        r = self.r
        rng = np.random.default_rng()
        x0, y0 = r._wx, r._wy
        n = 1500
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = np.sqrt(rng.uniform(0, 1, n)) * 4.0
        floor = np.stack([x0 + rad * np.cos(ang), y0 + rad * np.sin(ang),
                          rng.normal(0, 0.01, n)], 1)
        parts = [floor]
        for b in r._boxes:
            bx, by, w, d, h = b["x"], b["y"], b["w"], b["d"], b["h"]
            if math.hypot(bx - x0, by - y0) > 6.0:
                continue
            m = 150
            # 前面(x = bx - d/2)・後面・左右側面・天面
            ys = rng.uniform(by - w / 2, by + w / 2, m)
            zs = rng.uniform(0.0, h, m)
            parts.append(np.stack([np.full(m, bx - d / 2), ys, zs], 1))
            parts.append(np.stack([np.full(m, bx + d / 2), ys, zs], 1))
            xs = rng.uniform(bx - d / 2, bx + d / 2, m)
            parts.append(np.stack([xs, np.full(m, by - w / 2), zs], 1))
            parts.append(np.stack([xs, np.full(m, by + w / 2), zs], 1))
            parts.append(np.stack([xs, ys, np.full(m, h)], 1))
        pts = np.concatenate(parts).astype(np.float32)
        pts += rng.normal(0, 0.005, pts.shape).astype(np.float32)
        self.n_recv += 1
        return time.time(), pts, self.frame_id


class _SimOdom:
    def __init__(self, robot):
        self.r = robot
        self.enabled = False
        self.n_recv = 0

    def enable(self, on):
        self.enabled = bool(on)

    def stop(self):
        pass

    def latest(self):
        if not self.enabled:
            return None
        r = self.r
        self.n_recv += 1
        return (time.time(), r._wx + np.random.normal(0, 0.003),
                r._wy + np.random.normal(0, 0.003), r._wyaw,
                r._wv[0], r._wv[1])


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
        # --- 内蔵歩行のモック(自動歩行の結合試験用)。
        #     内蔵バランス制御そのものは模さない。歩行モード中は物理を止め、
        #     速度指令を運動学的に積分してオドメトリ(_wx,_wy,_wyaw)を動かす。
        self._fsm_id = 1                          # 起動直後=ダンピング相当
        self.walk_mode = False
        self.balance_hold = False                 # 標準モード(立位等)中は物理を止める
        self._wcmd = (0.0, 0.0, 0.0, 0.0)         # vx, vy, om, 失効時刻
        self._wv = [0.0, 0.0, 0.0]                # 実速度(1次遅れ)
        self._wx, self._wy, self._wyaw = 0.0, 0.0, 0.0
        # 既定の障害物: 2.0m先に 幅0.6×奥行0.3×高さ1.0 の箱
        self._boxes = [dict(x=2.0, y=0.0, w=0.6, d=0.3, h=1.0)]
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
        self.balance_hold = False                   # 物理を再開(以後は方策が支える)
        self.walk_mode = False
        return True

    def standard_mode(self, name):
        self.custom_active = False
        self.walk_mode = (name == "walk")
        self._fsm_id = {"zero": 0, "damp": 1, "stand": 4, "walk": 802,
                        "sit": 2, "seated": 3}.get(name, self._fsm_id)
        # ★2026-09-04 内蔵バランス制御の近似: 立位/歩行/スクワット/着座の標準モード中は
        #   物理を進めず、直立で固定する(以前は素のPD保持で数秒で倒れ、手順の練習=
        #   [スタンドロック]→[着座の確認]→着座 が通らなかった)。
        #   方策へ引き継ぐ(ensure_custom)と物理は再開する。
        self.balance_hold = name in ("stand", "walk", "sit", "seated")
        if name in ("stand", "walk"):
            self._upright()
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
        self.set_velocity(vx, vy, omega, 1.0)

    def return_to_balance(self, log=print):
        """実機の return_to_balance と同じ入口。simでは内蔵立位(PD保持)で近似"""
        if self._estop_latched:
            return False, None
        self.standard_mode("stand")
        self._upright()
        log("(sim) 内蔵バランス制御へ返却(ロック立位4の近似)")
        return True, 4

    # ---- 内蔵歩行のモック(実機の set_velocity/yaw/open_lidar/open_odom/
    #      ensure_walk_mode と同じ入口)
    def set_velocity(self, vx, vy, om, duration=0.5):
        with self.lock:
            now = time.time()
            nz = (abs(vx) > 1e-6 or abs(vy) > 1e-6 or abs(om) > 1e-6)
            prev = self._wcmd
            prev_nz = (abs(prev[0]) > 1e-6 or abs(prev[1]) > 1e-6 or abs(prev[2]) > 1e-6) and now < prev[3]
            if nz and not prev_nz:
                self._wstart = now                 # 歩き出し(内蔵歩行の立ち上がり遅れの模擬)
            self._wcmd = (float(vx), float(vy), float(om), now + float(duration))
        return True

    def yaw(self):
        return float(self._wyaw)

    def get_fsm_id(self):
        return self._fsm_id

    def _upright(self):
        """モックの機体を直立に戻す(内蔵バランス制御の代わり)。

        standで待つ間は素のPD保持なので、モックは数秒で倒れる(実測)。実機なら
        内蔵バランスが立て直しているので、歩行モードに入る時点で直立に置き直す。
        歩行モード中は物理を進めない(tick 参照)ので、以後は倒れない。
        """
        with self.lock:
            self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.d.qvel[:] = 0.0
            mujoco.mj_forward(self.m, self.d)

    def ensure_walk_mode(self, log=print):
        if self.custom_active:
            return False, "方策側が制御権を持っています"
        if self._fsm_id not in (4, 500, 501, 801, 802):
            return False, f"FSM={self._fsm_id}: 先に[立つ](ロック立位4)で立たせてから"
        self._fsm_id = 802
        self.walk_mode = True
        self._upright()
        log("(sim) 歩行モード 802 に入りました(運動学モック。バランス制御は模さない)")
        return True, 802

    def open_lidar(self):
        return _SimLidar(self)

    def open_odom(self):
        return _SimOdom(self)

    def sim_obstacles(self, boxes):
        """試験用: 障害物の箱を差し替える [{x,y,w,d,h}, ...](世界座標)"""
        self._boxes = list(boxes)

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
        if self.walk_mode:
            # 歩行モック: 物理は進めず、速度指令を1次遅れで積分する
            dt = n_sub * float(self.m.opt.timestep)
            with self.lock:
                vx, vy, om, texp = self._wcmd
                tgt = (vx, vy, om) if time.time() < texp else (0.0, 0.0, 0.0)
                # ★実機の内蔵歩行の癖の模擬(2026-09-04 実測: 0.08m/s×0.5sのパルスでは一歩も出ない):
                #   不感帯 |v|<0.10m/s・|ω|<0.15rad/s は動かない。歩き出しに 0.25秒かかる
                if (abs(tgt[0]) < 0.10 and abs(tgt[1]) < 0.10 and abs(tgt[2]) < 0.15) \
                        or time.time() - getattr(self, "_wstart", 0.0) < 0.25:
                    tgt = (0.0, 0.0, 0.0)
                a = min(1.0, dt / 0.4)
                self._wv = [self._wv[i] + (tgt[i] - self._wv[i]) * a
                            for i in range(3)]
                c, s = math.cos(self._wyaw), math.sin(self._wyaw)
                self._wx += (c * self._wv[0] - s * self._wv[1]) * dt
                self._wy += (s * self._wv[0] + c * self._wv[1]) * dt
                self._wyaw = _wrap(self._wyaw + self._wv[2] * dt)
            return
        if self.balance_hold:
            return                                  # 内蔵バランスが支えている近似
        with self.lock:
            for _ in range(n_sub):
                q = self.d.qpos[self.qadr]
                dq = self.d.qvel[self.dofadr]
                tau = np.clip(self.kp * (self.target_q - q) - self.kd * dq,
                              self.tau_lo, self.tau_hi)
                self.d.ctrl[self.acts] = tau
                mujoco.mj_step(self.m, self.d)
