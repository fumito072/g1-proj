#!/usr/bin/env python3
"""実機G1へのDDS接続(unitree_sdk2py)。コックピット/FSMランナー用。

前提(docs/REAL_ROBOT.md / REAL_CALIB.md の実測):
  - 有線LAN、PCは 192.168.123.0/24(例 192.168.123.222)、実機は .161
  - 送信前に MotionSwitcher の ReleaseMode() で標準制御を解放する
    (解放しないと rt/lowcmd をkp=0の指令が占有し、モータを取り合う)
  - LowCmd.mode_machine は LowState から読んだ値(実測5)をそのまま返す
  - mode_pr はPR(0)のまま
  - CRCを毎パケット計算する

※ probe.py / calib.py(実機PCに動作実績あり)と同じAPIを使っている。
  もし本ファイルで初期化に失敗する場合は、それらの初期化部と差分を比較すること。
"""
import threading
import time

import numpy as np

DAMP_KD_LEG = 5.0
DAMP_KD_ANKLE = 0.2


class RealRobot:
    """実機。SimRobotと同じAPI(state / set_target / set_damp / healthy)"""

    def __init__(self, iface="", send_hz=500):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        if iface:
            ChannelFactoryInitialize(0, iface)      # 例: "enp46s0"
        else:
            ChannelFactoryInitialize(0)

        self._crc = CRC()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self.lock = threading.Lock()
        self.q = np.zeros(29)
        self.dq = np.zeros(29)
        self.tau = np.zeros(29)
        self.quat = np.array([1.0, 0, 0, 0])
        self.gyro = np.zeros(3)
        self.temps = np.zeros(29)
        self.mode_machine = 5
        self._last_state_t = 0.0
        self.target_q = np.zeros(29)
        self.kp = np.zeros(29)
        self.kd = np.zeros(29)

        # --- MotionSwitcher / LocoClient(標準モードとの排他制御に使う)
        self._msc = None
        self._loco = None
        self.custom_active = False      # True=カスタム方策がモータを持つ
        self._stream_on = False         # 500Hz送信の有効/無効
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client \
                import MotionSwitcherClient
            self._msc = MotionSwitcherClient()
            self._msc.SetTimeout(5.0)
            self._msc.Init()
        except Exception as e:                     # noqa: BLE001
            print(f"★MotionSwitcher初期化に失敗: {e}")
        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            self._loco = LocoClient()
            self._loco.SetTimeout(10.0)
            self._loco.Init()
        except Exception as e:                     # noqa: BLE001
            print(f"★LocoClient初期化に失敗: {e}(標準モードボタンは使用不可)")
        # 起動時は制御権を取らない(標準制御のスタンドロック等を維持)。
        # 方策を使う直前に ensure_custom() でシームレスに引き継ぐ。

        # --- 購読
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        # --- 送信
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        # 最初のLowStateを待つ(mode_machineを読むため)
        t0 = time.time()
        while self._last_state_t == 0.0:
            if time.time() - t0 > 5.0:
                raise TimeoutError("LowStateが5秒来ない。配線とIP設定を確認")
            time.sleep(0.05)
        self._stop = False
        self._send_dt = 1.0 / send_hz
        self._th = threading.Thread(target=self._send_loop, daemon=True)
        self._th.start()

    def _on_state(self, msg):
        with self.lock:
            for i in range(29):
                m = msg.motor_state[i]
                self.q[i] = m.q
                self.dq[i] = m.dq
                self.tau[i] = m.tau_est
                self.temps[i] = m.temperature[0]
            self.quat[:] = msg.imu_state.quaternion
            self.gyro[:] = msg.imu_state.gyroscope
            self.mode_machine = msg.mode_machine
            self._last_state_t = time.time()

    # ---- 標準モード(SDK) / カスタム制御の排他
    def ensure_custom(self, kp=None, kd=None):
        """制御権をシームレスに引き継ぐ(脱力なし)。

        順序が重要:
          1) 現在の関節角 q_now を読む
          2) 先に 目標=q_now・指定ゲイン で500Hz送信を開始
             (標準制御と同じ姿勢の指令なので、解放までの重複期間も動かない)
          3) それから ReleaseMode で標準制御を解放
             → 解放の瞬間から当方のPDが同じ姿勢を保持し続ける
        逆順(先に解放)だと、解放〜初回指令の間モータが無支配になり脱力する。
        """
        if self.custom_active and self._stream_on:
            return
        q_now, _, _, _, _ = self.state()
        if kp is None:                             # 既定は公式ゲイン
            kp = np.zeros(29)
            kd = np.zeros(29)
            for i in range(12):                    # 脚: 股100/膝150/足首40
                kp[i] = 100.0 if i % 6 < 3 else (150.0 if i % 6 == 3 else 40.0)
                kd[i] = 3.0 if i % 6 < 3 else (4.5 if i % 6 == 3 else 1.5)
            kp[12:15] = 200.0; kd[12:15] = 6.0     # 腰
            kp[15:] = 70.0; kd[15:] = 2.5          # 腕
        self.set_target(q_now, kp, kd)             # 2) 現姿勢を保持する指令を先に
        self._stream_on = True
        time.sleep(0.05)
        if self._msc is not None:                  # 3) それから解放
            try:
                code, result = self._msc.CheckMode()
                n = 0
                while result and result.get("name") and n < 10:
                    self._msc.ReleaseMode()
                    time.sleep(0.3)
                    code, result = self._msc.CheckMode()
                    n += 1
            except Exception as e:                 # noqa: BLE001
                print(f"★ReleaseMode失敗: {e}")
        self.custom_active = True
        print("カスタム制御: シームレス引き継ぎ完了(現姿勢を保持中)")

    def standard_mode(self, name):
        """Unitree標準モードへ(SDK)。こちらの送信は停止する。
        name: zero / damp / stand / walk
        FSM id(g1_loco_client): 0=ゼロトルク 1=ダンプ 4=立ち上がり 200=運用制御
        """
        self._stream_on = False        # 送信停止(モータの取り合いを防ぐ)
        self.custom_active = False
        time.sleep(0.05)
        if self._msc is not None:
            try:                        # 制御サービスを復帰させる
                code, result = self._msc.CheckMode()
                if not (result and result.get("name")):
                    self._msc.SelectMode("ai")     # 実測でaiが動作(REAL_ROBOT.md)
                    time.sleep(1.0)
            except Exception as e:                 # noqa: BLE001
                print(f"★SelectMode失敗: {e}")
        if self._loco is None:
            print("★LocoClient未初期化のため標準モード不可")
            return False
        try:
            if name == "zero":
                self._loco.SetFsmId(0)
            elif name == "damp":
                self._loco.Damp()
            elif name == "stand":
                self._loco.SetFsmId(4)             # 立ち上がり→バランス立位
            elif name == "walk":
                self._loco.SetFsmId(200)           # 運用制御(歩行可能)
            print(f"標準モード: {name}")
            return True
        except Exception as e:                     # noqa: BLE001
            print(f"★標準モード({name})失敗: {e}")
            return False

    def stop_move(self):
        """内蔵制御の速度指令をゼロにする。**引き継ぎの前に必ず呼ぶ。**

        ウォーキングFSM(200)は内蔵の目標が毎tick変わる。その途中で
        「いまの関節角」をラッチして保持すると、重心移動の途中の姿勢を
        固定することになる。2026-08-24の実測でも、walkから引き継いだ1回だけ
        方策開始時の傾きが7度(standからの6回は0〜4度)と最大だった。
        stand(FSM 4)では無害なので、モードを問わず呼んでよい。
        """
        if self._loco is None:
            return
        try:
            if hasattr(self._loco, "StopMove"):
                self._loco.StopMove()
            else:
                self._loco.Move(0.0, 0.0, 0.0)
        except Exception as e:                     # noqa: BLE001
            print(f"★StopMove失敗(続行): {e}")

    def loco_move(self, vx, vy, omega):
        """ウォーキングモード中の速度指令(SDK)"""
        if self._loco is not None:
            try:
                self._loco.Move(vx, vy, omega)
            except Exception as e:                 # noqa: BLE001
                print(f"★Move失敗: {e}")

    def _send_loop(self):
        while not self._stop:
            t0 = time.time()
            if not self._stream_on:
                time.sleep(0.02)
                continue
            with self.lock:
                tq = self.target_q.copy()
                kp = self.kp.copy()
                kd = self.kd.copy()
                mm = self.mode_machine
            c = self._cmd
            c.mode_pr = 0
            c.mode_machine = mm
            for i in range(29):
                c.motor_cmd[i].mode = 1
                c.motor_cmd[i].q = float(tq[i])
                c.motor_cmd[i].dq = 0.0
                c.motor_cmd[i].kp = float(kp[i])
                c.motor_cmd[i].kd = float(kd[i])
                c.motor_cmd[i].tau = 0.0
            c.crc = self._crc.Crc(c)
            self._pub.Write(c)
            rest = self._send_dt - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)

    # ---- 共通API
    def state(self):
        with self.lock:
            return (self.q.copy(), self.dq.copy(), self.quat.copy(),
                    self.gyro.copy(), self.tau.copy())

    def set_target(self, q, kp, kd):
        with self.lock:
            self.target_q = np.asarray(q, dtype=float).copy()
            self.kp = np.asarray(kp, dtype=float).copy()
            self.kd = np.asarray(kd, dtype=float).copy()

    def set_damp(self):
        kd = np.full(29, DAMP_KD_LEG)
        kd[[4, 5, 10, 11]] = DAMP_KD_ANKLE
        self.set_target(np.zeros(29), np.zeros(29), kd)

    def healthy(self):
        """LowStateが0.2秒以上途絶えたら不健康(→コックピットが自動DAMP)"""
        return (time.time() - self._last_state_t) < 0.2

    def check_authority(self, jid=15, delta=0.06, dur=0.15):
        """指令が実際にモータへ届いているかを**実測**する。方策を始める前に呼ぶ。

        CheckMode は当てにならない(稼働中のFSMから解放すると「解放済み」と
        返すのに制御権が移らない実測あり。膝の偏差1.4radに対しトルク3〜17N·m
        = kp150なら225N·m出るはずの状況だった)。

        ★他の関節の kp/kd を 0 にしてはいけない。初期実装は試験する腕以外を
          全部0にしており、脚が1.2秒間まったくの脱力になって膝が
          0.55→2.91rad(31度→167度)まで沈んだ。終了時に set_damp するのも同じ誤り。
        ここでは**全関節を現姿勢で保持したまま**、左肩ピッチの目標だけを
        delta ずらして応答を見る。実測で姿勢変化0.000rad・所要0.15秒。
        """
        q0, _, _, _, _ = self.state()
        with self.lock:
            kp = self.kp.copy()
            kd = self.kd.copy()
        if float(kp.max()) <= 0.0:
            return False, 0.0, 0.0, "ゲインが0(先に ensure_custom を呼ぶこと)"
        tgt = q0.copy()
        tgt[jid] += delta
        self.set_target(tgt, kp, kd)
        time.sleep(dur)
        q1, _, _, _, tau1 = self.state()
        moved = float(q1[jid] - q0[jid])
        tau = float(tau1[jid])
        self.set_target(q0, kp, kd)                # 現姿勢へ戻す(dampしない)
        ok = (abs(moved) > 0.3 * delta) or (abs(tau) > 0.5)
        why = ("" if ok else
               "指令が届いていない(mode_pr / 二重publish / 制御権を疑う)")
        return ok, moved, tau, why

    def close(self):
        self._stop = True
        self.set_damp()
