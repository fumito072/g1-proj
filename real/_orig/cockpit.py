#!/usr/bin/env python3
"""G1コックピット — deployの方策を実機で走らせる操縦システム。

  python3 real/cockpit.py --sim            # MuJoCoモックで結合試験(実機不要)
  python3 real/cockpit.py --iface enp46s0  # 実機(有線LAN)
ブラウザで http://localhost:8090 を開く。使い方は real/COCKPIT.md。

構成:
  ロボットIF: RealRobot(DDS) / SimRobot(MuJoCoモック、--sim)
  FSMエンジン: 50Hzループ。フェーズ=deployの方策。遷移は操作者確認(既定)
  Webサーバ: 状態JSON・映像(simのみ)・コマンド受付
安全:
  E-STOP(UIの赤ボタン)→即damp / 傾き40度・関節速度超過・受信途絶→自動damp
"""
import argparse
import gc
import json
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from run_fsm import (Policy, ObsBuilder, quat_to_mat, ACTION_SCALE,   # noqa: E402
                     CONTROL_HZ, TILT_LIMIT_DEG)

DEPLOY = ROOT / "deploy"
VEL_HARD = 32.0            # 全関節の速度ハード上限[rad/s]
# 引き継ぎ直後の「静的な前傾」を開始前に弾くための境目。
# 2026-08-21の実バンク学習で、傾き16.1度と36.9度・角速度|gyro|≒0.04rad/s の
# 2状態だけが 0/10 だった。成功した7状態は傾き12〜20度に戻り方向の角速度〜1rad/s。
# 傾きだけでは決まらない(11.9度で失敗・18.5度で成功の例がある)ので、
# 「大きく傾いているのに動いていない」の組で判定する。
HANDOVER_STATIC_TILT_DEG = 15.0
HANDOVER_STATIC_RATE = 0.25       # rad/s

# ログ形式は 2026-08-24 の実機セッションのものに合わせる。
# real/log_view.py と real/ab_report.py がこの形式を前提にしており
# (`rec` 行列 + `cols` 列名 + `run<NN>_設定.json`)、別形式で書くと
# 現場で取ったデータを既存の解析ツールで読めなくなる。
FSM_CODE = {"IDLE": 0, "MOVING": 1, "WAIT_CONFIRM": 2, "RUNNING": 3,
            "HOLD": 4, "DAMP": 5}
REC_COLS = (["t", "fsm", "phase_i", "elapsed_s", "tilt_deg"]
            + [f"q{i}" for i in range(29)]
            + [f"dq{i}" for i in range(29)]
            + [f"tau{i}" for i in range(29)]
            + ["quat_w", "quat_x", "quat_y", "quat_z"]
            + ["gyro_x", "gyro_y", "gyro_z"]
            + [f"target{i}" for i in range(29)]
            + [f"act{i}" for i in range(29)]
            + [f"temp{i}" for i in range(29)]
            # ここから先は2026-08-26に追加した列。既存ツールは名前で引くので
            # 末尾に足す分には影響しない。dt_ms/ms_inferはPythonの尾側の
            # 遅れが実機で出ているかを後から数値で確かめるため
            + ["dt_ms", "ms_infer"])
REC_SAVE_EVERY = 100              # 途中で落ちても失わないよう逐次保存


def list_patterns():
    out = {"climb": [], "turn": [], "sit": []}
    for d in sorted(DEPLOY.iterdir()):
        if not (d / "policy.pt").exists():
            continue
        n = d.name
        if n.startswith("climb"):
            out["climb"].append(n)
        elif n.startswith(("turn", "bridge")):
            out["turn"].append(n)
        elif n.startswith("sit"):
            out["sit"].append(n)
    return out


class Engine:
    """FSMエンジン。stateはUIへそのまま出す"""

    def __init__(self, robot, is_sim):
        self.robot = robot
        self.is_sim = is_sim
        self.lock = threading.Lock()
        self.cmd_q = []
        self.fsm = "IDLE"                # IDLE/MOVING/WAIT_CONFIRM/RUNNING/HOLD/DAMP
        self.phase_i = -1
        self.phases = []                 # [(名前, Policy), ...]
        self.t = 0
        self.n = 1
        self.msg = "起動しました。パターンを選んでARMしてください"
        self.step_mode = True            # フェーズ間で操作者確認
        self.armed = False
        self.sel = {"climb": "climb_slow_r2", "turn": "turn_wide_r2",
                    "sit": "sit_up_rc_r2"}   # 実機較正シム再学習版を既定に(2026-08-23)
        self.obs_b = None
        self.log_dir = None
        self.logs = []
        self.run_i = 0                   # 1セッション中の実行回数(run01, run02…)
        self.single_task = None
        self._rec_rows = []
        self._rec_obs = []
        self._rec_path = None
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def log(self, s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-200:]
            self.msg = s
        # セッションの全操作を「イベント.log」へ追記する。何時何分に何を
        # 押したか・引き継ぎに何秒かかったかは、後から数値で追える唯一の記録
        if self.log_dir is not None:
            try:
                with open(self.log_dir / "イベント.log", "a",
                          encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
            except Exception:                      # noqa: BLE001
                pass

    def command(self, cmd, arg=None):
        with self.lock:
            self.cmd_q.append((cmd, arg))

    def _pop(self):
        with self.lock:
            return self.cmd_q.pop(0) if self.cmd_q else (None, None)

    def _safety(self):
        q, dq, quat, gyro, tau = self.robot.state()
        if not self.robot.healthy():
            return "受信途絶"
        up_z = quat_to_mat(quat)[2, 2]
        if up_z < np.cos(np.radians(TILT_LIMIT_DEG)):
            return f"傾き{np.degrees(np.arccos(min(1, max(-1, up_z)))):.0f}度"
        if float(np.abs(dq).max()) > VEL_HARD:
            return "関節速度超過"
        return None

    def _estop(self, why):
        # 中断した回の記録も必ず残す(final=False で「途中で終わった」印になる)。
        # 失敗した回のログの方が価値が高い
        self._save_rec(final=False)
        self.hold_pol = None
        if getattr(self.robot, "custom_active", True):
            self.robot.set_damp()
        else:
            self.robot.standard_mode("damp")       # 標準モード中はSDKのDamp
        self.fsm = "DAMP"
        self.armed = False
        self.log(f"★DAMP: {why}")

    def _handover(self, pol):
        """内蔵制御からの引き継ぎ。**準備が全部終わってから**呼ぶこと。

        順序: 目標=現姿勢で500Hz送信を開始 → 解放 → 指令の到達を実測。
        重い準備(Policy/ObsBuilder/ログ)を制御権取得後にやると、その間は
        PD保持だけでバランスが取れず体幹が傾く(実測: 合計3秒で37度・自動DAMP。
        準備を前へ出して3秒→0.73秒)。所要時間もログに残す。
        """
        t0 = time.time()
        # 内蔵の速度指令をゼロにしてから静定を待つ。ウォーキングFSM(200)は
        # 目標が毎tick変わるので、その途中の姿勢をラッチすると重心移動の
        # 途中で固めることになる(2026-08-24実測: walkから引き継いだ1回だけ
        # 方策開始時の傾き7度。standからの6回は0〜4度)
        if hasattr(self.robot, "stop_move"):
            self.robot.stop_move()
            time.sleep(0.3)
        q0, _, quat0, gyro0, _ = self.robot.state()
        rate0 = float(np.linalg.norm(gyro0[:2]))
        self.log(f"内蔵制御から引き継ぎます(姿勢を保持したまま解放)"
                 f" 角速度{rate0:.2f}rad/s")
        self.robot.ensure_custom(kp=pol.kp, kd=pol.kd)
        ok, moved, tau, why = self.robot.check_authority()
        if ok:
            self.log(f"指令の到達を確認: OK — 左肩ピッチを+0.060rad指令 → "
                     f"実測{moved:.3f}rad / トルク{tau:.1f}Nm"
                     f"(他の関節は保持したまま)")
        else:
            # ここで damp してはいけない。指令が届いていないなら damp も
            # 届かず、届くなら脱力させることになる。現姿勢保持のまま止める
            self.log(f"★指令が届いていません — {why}。開始を中止します"
                     f"(実測{moved:.3f}rad / トルク{tau:.1f}Nm)")
            self.fsm = "IDLE"
            return False
        q, _, quat, _, _ = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        self.log(f"解放から方策開始まで {time.time() - t0:.2f}秒(傾き{tilt:.0f}度)")
        return True

    # ---------------- メインループ(50Hz)
    def _loop(self):
        dt = 1.0 / CONTROL_HZ
        interp = None                    # (q0, q_goal, kp, kd, i, steps)
        while True:
            t0 = time.time()
            cmd, arg = self._pop()
            # --- コマンド処理
            if cmd == "estop":
                self._estop("操作者による緊急停止")
            elif cmd == "select":
                k, v = arg
                self.sel[k] = v
                self.log(f"パターン選択: {k}={v}")
            elif cmd == "mode":
                self.step_mode = (arg == "step")
                self.log(f"進行モード: {'ステップ(各フェーズ前に確認)' if self.step_mode else '自動'}")
            elif cmd == "arm" and self.fsm in ("IDLE", "DAMP", "HOLD"):
                try:
                    self.phases = []
                    for k in ("climb", "turn", "sit"):
                        if self.sel[k] != "(skip)":
                            self.phases.append((self.sel[k], Policy(self.sel[k])))
                    self.obs_b = ObsBuilder(self.phases[0][1])
                    # 開始は自然な両足立位から(climb系の参照fr0は片脚立ちだが、
                    # 蒸留済み方策は両足立位スタートでも完走する。10/10で実測)
                    sp = ROOT / "motions" / "climb_stand.npz"
                    self.stand = dict(np.load(sp)) if (
                        sp.exists() and self.phases[0][0].startswith("climb")
                    ) else None
                    self.log_dir = (ROOT / "logs" / "real"
                                    / time.strftime("cockpit_%Y%m%d_%H%M%S"))
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    self.single_task = None
                    self.armed = True
                    self.phase_i = -1
                    self.hold_pol = None
                    self.fsm = "IDLE"
                    self.log("ARM完了: " + " → ".join(n for n, _ in self.phases))
                except Exception as e:              # noqa: BLE001
                    self.log(f"★ARM失敗: {e}")
            elif cmd == "place_sim" and self.is_sim and self.armed:
                pol = self.phases[0][1]
                z = pol.ref
                if self.stand is not None:
                    s = self.stand
                    self.robot.place(s["q"], s["quat"], s["xy"][:2], float(s["z"]))
                    self.robot.set_target(s["q"], pol.kp, pol.kd)
                    self.log("(sim)開始位置に両足立位で配置しました")
                else:
                    self.robot.place(pol.ref_q[0], z["ref_quat"][0],
                                     z["ref_xy_abs"][0][:2], float(z["ref_z"][0]))
                    self.robot.set_target(pol.ref_q[0], pol.kp, pol.kd)
                    self.log("(sim)参照開始姿勢で配置しました")
                # 方策なしのPD保持は両足立位でも数秒で倒れる(実測)。
                # simでは待機中は物理を凍結(実機では操作者が支える)
                self.sim_frozen = True
            elif cmd == "start" and self.armed and self.fsm in ("IDLE", "HOLD"):
                pol = self.phases[0][1]
                if not self.robot.custom_active:
                    if not self._handover(pol):
                        continue
                self.sim_frozen = False
                q, _, _, _, _ = self.robot.state()
                # 目標は両足立位(保持可能。蒸留方策はここから完走できる)。
                # 立位ファイルが無い課題は参照fr0へ(片脚立ちなら即開始)。
                q_goal = (self.stand["q"] if self.stand is not None
                          else pol.ref_q[0])
                err = float(np.abs(q - q_goal).max())
                if err < 0.30:  # 蒸留方策は開始ずれに頑健。補間(方策なしPD)は最小限に
                    self.phase_i = 0
                    self._begin_phase()
                else:
                    steps = int(1.5 * CONTROL_HZ)
                    interp = [q.copy(), np.asarray(q_goal).copy(),
                              pol.kp, pol.kd, 0, steps]
                    self.fsm = "MOVING"
                    self.log(f"開始立位へ1.5秒補間(関節差 最大{err:.2f}rad)")
            elif cmd == "next" and self.fsm == "WAIT_CONFIRM":
                self._begin_phase()
            elif cmd == "damp":
                self._estop("操作者によるdamp")
            elif cmd and cmd.startswith("mode_"):
                # 標準モード(SDK): zero/damp/stand/walk。方策実行は中断
                name = cmd[5:]
                self.armed = False
                self.sim_frozen = False
                ok = self.robot.standard_mode(name)
                self.fsm = f"STD:{name}" if ok else "DAMP"
                self.log(f"標準モード {name}" + ("" if ok else "(失敗→要確認)"))
            elif cmd == "custom":
                # シームレス引き継ぎ: 現姿勢を保持したまま制御権を取る
                # (脱力しない。スタンドロックからそのまま移行できる)
                self.robot.ensure_custom()
                self.fsm = "IDLE"
                self.log("カスタム制御へ引き継ぎ(現姿勢を保持中)。ARM→実行へ")
            elif cmd == "run_task" and arg in ("climb", "turn", "sit"):
                # 単体タスク実行: 選択中のパターンを1フェーズだけ走らせる
                if self.sel[arg] == "(skip)":
                    self.log(f"★{arg} のパターンが(skip)です")
                else:
                    try:
                        # 重い準備は**引き継ぎより前**に全部済ませる。
                        # ObsBuilder は MuJoCo モデルを読むので1〜2秒かかり、
                        # 制御権を取ってから作ると、その間PD保持だけで
                        # バランスが取れず体幹が傾く(実測: 合計3秒で37度傾き、
                        # 開始直後に自動DAMP。references/handover.md §3)
                        pol = Policy(self.sel[arg])
                        self.phases = [(self.sel[arg], pol)]
                        self.obs_b = ObsBuilder(pol)
                        sp = ROOT / "motions" / "climb_stand.npz"
                        self.stand = (dict(np.load(sp))
                                      if arg == "climb" and sp.exists() else None)
                        self.log_dir = (ROOT / "logs" / "real" /
                                        time.strftime("cockpit_%Y%m%d_%H%M%S"))
                        self.log_dir.mkdir(parents=True, exist_ok=True)
                        self.single_task = arg
                        if not self.robot.custom_active:
                            # 準備が済んでから、脱力なしで引き継ぐ
                            if not self._handover(pol):
                                continue
                        self.armed = True
                        self.sim_frozen = False
                        q, _, quat, _, _ = self.robot.state()
                        up_z = float(quat_to_mat(quat)[2, 2])
                        if up_z > np.cos(np.radians(20.0)):
                            # 直立していれば方策を直接開始する。
                            # ハンドオフ後の姿勢は参照と最大0.4rad程度ずれるが、
                            # 蒸留方策はその分布で学習済み(バンク)。方策なしの
                            # 補間はかえって転倒する(実測)
                            self.phase_i = 0
                            self._begin_phase()
                        else:
                            q_goal = (self.stand["q"] if self.stand is not None
                                      else pol.ref_q[0])
                            steps = int(1.5 * CONTROL_HZ)
                            interp = [q.copy(), np.asarray(q_goal).copy(),
                                      pol.kp, pol.kd, 0, steps]
                            self.fsm = "MOVING"
                            self.log(f"単体実行 {self.sel[arg]}: 姿勢が崩れている"
                                     f"ため開始姿勢へ補間(実機では支えること)")
                    except Exception as e:          # noqa: BLE001
                        self.log(f"★単体実行失敗: {e}")

            # --- 状態処理
            why = self._safety() if self.fsm in ("MOVING", "RUNNING", "HOLD") else None
            if why:
                self._estop(why)
                interp = None
            if self.fsm == "MOVING" and interp:
                q0, qg, kp, kd, i, steps = interp
                w = (i + 1) / steps
                w = w * w * (3 - 2 * w)
                # 目標は現在姿勢から始まる(=初期トルク0)ので、kpは最初から
                # フル値でよい。kpをランプすると序盤の支持力が消えて崩れる(実測)
                self.robot.set_target((1 - w) * q0 + w * qg, kp, kd)
                interp[4] = i + 1
                if interp[4] >= steps:
                    interp = None
                    self.phase_i = 0
                    if self.step_mode and self.stand is not None:
                        self.fsm = "WAIT_CONFIRM"
                        if self.is_sim:
                            self.sim_frozen = True
                        self.log(f"立位で待機中(実機では支えて)。"
                                 f"[NEXT]で {self.phases[0][0]} 開始")
                    else:
                        self._begin_phase()
            elif self.fsm == "RUNNING":
                name, pol = self.phases[self.phase_i]
                q, dq, quat, gyro, tau = self.robot.state()
                _ti = time.perf_counter()
                obs = self.obs_b.build(pol, self.t, q, dq, quat, gyro)
                a = pol.act(obs)
                self._ms_infer = (time.perf_counter() - _ti) * 1000.0
                self.obs_b.last_cmd = a.copy()
                target = pol.ref_q[min(self.t, pol.n - 1)] + a * ACTION_SCALE
                self.robot.set_target(target, pol.kp, pol.kd)
                self._rec(name, q, dq, quat, gyro, tau, obs, a, target)
                self.t += 1
                if self.t >= pol.n:
                    self._end_phase()
            elif (self.fsm in ("HOLD", "WAIT_CONFIRM")
                  and getattr(self, "hold_pol", None) is not None):
                # 待機中も直前フェーズの方策で最終コマを維持(バランスあり)
                pol = self.hold_pol
                q, dq, quat, gyro, tau = self.robot.state()
                obs = self.obs_b.build(pol, pol.n - 1, q, dq, quat, gyro)
                a = pol.act(obs)
                self.obs_b.last_cmd = a.copy()
                self.robot.set_target(pol.ref_q[pol.n - 1] + a * ACTION_SCALE,
                                      pol.kp, pol.kd)
            # simモックは論理時間で進める(壁時計非依存。実機は実時間)
            if self.is_sim and not getattr(self, "sim_frozen", False):
                self.robot.tick(10)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def _begin_phase(self):
        name, pol = self.phases[self.phase_i]
        q, _dq, quat, gyro, _tau = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        rate = float(np.linalg.norm(gyro[:2]))
        # 立位→方策の引き継ぎでは、目標=現在姿勢のPDが重力に釣り合うまで沈む。
        # 2026-08-21実測: 姿勢チェック時2.5〜2.8度が方策開始時点で12〜19度に育ち、
        # 開始0.4秒後に31〜40度のピーク(中止しきい値40度に対し余裕2〜8度)。
        # ここで開始時点の値を必ず残す。これが無いと後から切り分けられない。
        self.log(f"引き継ぎ計測: 傾き{tilt:.1f}度 角速度{rate:.2f}rad/s")
        # 傾きが大きいのに角速度が小さい = 静的な前傾は、重心が支持多角形の外で
        # 静止した「倒れ確定」姿勢で、踏み出し無しでは物理的に回復不能
        # (2026-08-21 §5: 実バンクの該当2状態だけが0/10。除外して初めて合格した)。
        # 動的なトランジェント(戻り方向の角速度つき)とは区別して止める。
        if tilt > HANDOVER_STATIC_TILT_DEG and rate < HANDOVER_STATIC_RATE:
            # **ここで damp してはいけない。** 機体はいま自前PDで立っている。
            # kp=0 にすればその瞬間に崩れる(スキル原則1: 安全のために足した
            # 機構が事故を起こす)。現姿勢の保持を続けたまま開始だけ拒否する。
            self.robot.set_target(q, pol.kp, pol.kd)
            self.fsm = "IDLE"
            self.log(f"★開始を拒否: 静的な前傾{tilt:.0f}度"
                     f"(角速度{rate:.2f}rad/s)。現姿勢を保持中 — "
                     f"支えて立て直してから再実行してください")
            return
        self.sim_frozen = False
        self.hold_pol = None
        self.obs_b.reset(est_xy=pol.ref["ref_xy_abs"][0][:2])
        self.t = 0
        self.n = pol.n
        self.start_tilt = tilt
        self.start_rate = rate
        self._rec_rows = []
        self._rec_obs = []
        self._rec_prev_t = None
        self._phase_t0 = time.time()
        if self.phase_i == 0:            # 通しの1回 = run。設定を先に残す
            self.run_i += 1
            self._write_run_meta()
        self._rec_path = (self.log_dir /
                          f"run{self.run_i:02d}_{self.phase_i}_{name}.npz")
        self.fsm = "RUNNING"
        tp = np.asarray(getattr(self.robot, "temps", np.zeros(29)))
        self.log(f"フェーズ開始: {name}({pol.n}コマ / {pol.n / CONTROL_HZ:.1f}秒)"
                 f"  温度 膝 L{tp[3]:.0f}/R{tp[9]:.0f}度"
                 f"  足首 L{tp[4]:.0f}/R{tp[10]:.0f}度"
                 f"  最大{tp.max():.0f}度(j{int(np.argmax(tp))})")

    def _write_run_meta(self):
        """run<NN>_設定.json — 何をどの設定で回したかの唯一の記録。
        ab_report.py がこれと npz を突き合わせて集計する"""
        name0, pol0 = self.phases[0]
        meta = {
            "run": self.run_i,
            "phases": [n for n, _ in self.phases],
            "single_task": self.single_task,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_sim": bool(self.is_sim),
            "step_mode": bool(self.step_mode),
            "control_hz": CONTROL_HZ,
            "action_scale": ACTION_SCALE,
            "tilt_limit_deg": TILT_LIMIT_DEG,
            "vel_hard": VEL_HARD,
            "kp": [float(x) for x in pol0.kp],
            "kd": [float(x) for x in pol0.kd],
            "joint_names": list(pol0.joint_names),
            "n_frames": int(pol0.n),
            "duration_s": round(pol0.n / CONTROL_HZ, 2),
            "temps_start": [float(x) for x in
                            np.asarray(getattr(self.robot, "temps",
                                               np.zeros(29)))],
            "policy_meta": {n: p.meta for n, p in self.phases},
        }
        try:
            (self.log_dir / f"run{self.run_i:02d}_設定.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:                     # noqa: BLE001
            self.log(f"★設定の保存に失敗: {e}")

    def _save_rec(self, final):
        """逐次保存。途中でDAMPしても、そこまでの記録は必ず残る"""
        if self._rec_path is None or not self._rec_rows:
            return
        try:
            np.savez_compressed(
                self._rec_path,
                rec=np.asarray(self._rec_rows, dtype=np.float32),
                cols=np.array(REC_COLS),
                obs=np.asarray(self._rec_obs, dtype=np.float32),
                final=np.array(bool(final)))
        except Exception as e:                     # noqa: BLE001
            self.log(f"★記録の保存に失敗: {e}")

    def _end_phase(self):
        name, pol = self.phases[self.phase_i]
        self._save_rec(final=True)
        tp = np.asarray(getattr(self.robot, "temps", np.zeros(29)))
        self.log(f"フェーズ完走: {name}  温度 膝 L{tp[3]:.0f}/R{tp[9]:.0f}度"
                 f"  足首 L{tp[4]:.0f}/R{tp[10]:.0f}度"
                 f"  最大{tp.max():.0f}度(j{int(np.argmax(tp))})")
        # 待機中も方策を最終コマで動かし続ける(方策は終端の静止保持を
        # 学習済み。素のPD保持は数秒で釣り合いを失う。実測)
        self.hold_pol = pol
        if self.phase_i + 1 < len(self.phases):
            self.phase_i += 1
            if self.step_mode:
                self.fsm = "WAIT_CONFIRM"
                self.log(f"{name} 完了(方策で姿勢維持中)。"
                         f"[NEXT]で {self.phases[self.phase_i][0]} を開始")
            else:
                self._begin_phase()
        else:
            self.fsm = "HOLD"
            self.log(f"{name} 完了。全フェーズ終了 — 方策で姿勢維持中")

    def _rec(self, name, q, dq, quat, gyro, tau, obs, a, target):
        """1コマを REC_COLS の並びで平坦な行にして積む(2026-08-24形式)。
        温度は摩擦の温度依存をシムの同定と突き合わせるため、
        dt_ms/ms_infer はPythonの遅れを実機で数値確認するために入れてある。"""
        now = time.perf_counter()
        prev = self._rec_prev_t
        self._rec_prev_t = now
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        tp = np.asarray(getattr(self.robot, "temps", np.zeros(29)), dtype=float)
        self._rec_rows.append(np.concatenate([
            [float(self.t), float(FSM_CODE.get(self.fsm, -1)),
             float(self.phase_i), time.time() - self._phase_t0, tilt],
            q, dq, tau, quat, gyro, target, a, tp,
            [0.0 if prev is None else (now - prev) * 1000.0,
             float(getattr(self, "_ms_infer", 0.0))]]))
        self._rec_obs.append(np.asarray(obs, dtype=np.float32))
        if len(self._rec_rows) % REC_SAVE_EVERY == 0:
            self._save_rec(final=False)

    def snapshot(self):
        q, dq, quat, gyro, tau = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        with self.lock:
            return {
                "fsm": self.fsm, "msg": self.msg, "armed": self.armed,
                "step_mode": self.step_mode, "sel": dict(self.sel),
                "phases": [n for n, _ in self.phases],
                "phase_i": self.phase_i, "t": self.t, "n": self.n,
                "tilt_deg": float(np.degrees(np.arccos(min(1, max(-1, up_z))))),
                "qd_rms": float(np.sqrt(np.mean(dq ** 2))),
                "tau_max": float(np.abs(tau).max()),
                "healthy": bool(self.robot.healthy()),
                "is_sim": self.is_sim,
                "logs": list(self.logs[-25:]),
            }


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>G1 Cockpit</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#111;--card:#1c1c1c;--line:#333;--t1:#eee;--t2:#9a9a9a;
 --ok:#1baf7a;--warn:#eda100;--bad:#e34948;--acc:#3987e5}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--t1);font:14px/1.5 "Segoe UI",sans-serif;
 padding:16px;max-width:1100px;margin:auto}
h1{font-size:17px;margin-bottom:10px}
.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:12px;color:var(--t2);margin-bottom:8px;letter-spacing:.05em}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.tile .k{font-size:10px;color:var(--t2)}.tile .v{font-size:20px;font-weight:700}
select,button{font:inherit;border-radius:8px;border:1px solid var(--line);
 background:#242424;color:var(--t1);padding:8px 12px}
button{cursor:pointer}button:disabled{opacity:.35;cursor:default}
button.go{background:var(--acc);border:none;font-weight:700}
button.next{background:var(--ok);border:none;font-weight:700}
#estop{background:var(--bad);border:none;color:#fff;font-size:22px;
 font-weight:900;width:100%;padding:18px;border-radius:12px;margin-bottom:12px}
.bar{height:10px;background:#2c2c2c;border-radius:5px;overflow:hidden;margin:6px 0}
.bar>div{height:100%;background:var(--acc)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0}
.log{font:12px/1.5 Consolas,monospace;white-space:pre-wrap;color:var(--t2);
 max-height:220px;overflow-y:auto}
.state{font-size:26px;font-weight:900}
img{width:100%;border-radius:8px}
.lbl{font-size:11px;color:var(--t2)}
</style></head><body>
<h1>🤖 G1 Cockpit <span id="mode" class="lbl"></span></h1>
<button id="estop" onclick="cmd('estop')">■ E-STOP(即damp)</button>
<div class="tiles">
 <div class="tile"><div class="k">FSM</div><div class="v state" id="fsm">-</div></div>
 <div class="tile"><div class="k">傾き</div><div class="v" id="tilt">-</div></div>
 <div class="tile"><div class="k">関節速度RMS</div><div class="v" id="qd">-</div></div>
 <div class="tile"><div class="k">受信</div><div class="v" id="hl">-</div></div>
</div>
<div class="grid">
<div>
 <div class="card">
  <h2>ロボットモード(Unitree標準制御/SDK)</h2>
  <div class="row">
   <button onclick="cmd('mode_zero')">ゼロトルク</button>
   <button onclick="cmd('mode_damp')">ダンプ</button>
   <button onclick="if(confirm('標準制御で立ち上がります。周囲OK?'))cmd('mode_stand')">立つ</button>
   <button onclick="if(confirm('運用制御(歩行可能)に入ります。OK?'))cmd('mode_walk')">ウォーキング</button>
   <button onclick="cmd('custom')">カスタム制御へ</button>
  </div>
  <div class="lbl">標準モード中はこちらの送信を停止しSDK経由で指令。方策を使う前に[カスタム制御へ]</div>
 </div>
 <div class="card" style="margin-top:12px">
  <h2>シーケンス構成 / 単体実行</h2>
  <div class="row">登り <select id="sel_climb" onchange="sel('climb')"></select>
   <button class="go" onclick="if(confirm('登りを単体実行します。位置と吊り具OK?'))cmd('run_task','climb')">▶ 登壇</button></div>
  <div class="row">旋回 <select id="sel_turn" onchange="sel('turn')"></select>
   <button class="go" onclick="if(confirm('旋回を単体実行します。段上の立位からOK?'))cmd('run_task','turn')">▶ 旋回</button></div>
  <div class="row">着座 <select id="sel_sit" onchange="sel('sit')"></select>
   <button class="go" onclick="if(confirm('着座を単体実行します。椅子前の立位からOK?'))cmd('run_task','sit')">▶ 座る</button></div>
  <div class="row">進行
   <select id="sel_mode" onchange="cmd('mode',this.value)">
    <option value="step">ステップ(各フェーズ前に確認)</option>
    <option value="auto">自動</option></select></div>
  <div class="row">
   <button onclick="cmd('arm')">1. ARM(方策読込)</button>
   <button id="place" onclick="cmd('place_sim')" style="display:none">1.5 (sim)配置</button>
   <button class="go" onclick="if(confirm('開始姿勢へ動きます。吊り具/周囲OK?'))cmd('start')">2. START</button>
   <button class="next" id="next" onclick="cmd('next')">▶ NEXT(次フェーズ)</button>
   <button onclick="cmd('damp')">damp</button>
  </div>
 </div>
 <div class="card" style="margin-top:12px">
  <h2>進行状況</h2>
  <div id="phases" class="lbl">-</div>
  <div class="bar"><div id="prog" style="width:0%"></div></div>
  <div class="lbl" id="tn">-</div>
  <div style="margin-top:8px;font-size:15px" id="msg">-</div>
 </div>
 <div class="card" style="margin-top:12px"><h2>ログ</h2><div class="log" id="log"></div></div>
</div>
<div>
 <div class="card"><h2>ビュー(simモードのみ)</h2>
  <img id="cam" style="display:none"><div class="lbl" id="noimg">実機モードでは映像なし(テレメトリ参照)</div></div>
</div>
</div>
<script>
let S={sel:{}};
function cmd(c,a){fetch('/cmd?c='+c+(a?('&a='+encodeURIComponent(a)):''),{method:'POST'})}
function sel(k){cmd('select',k+':'+document.getElementById('sel_'+k).value)}
function fill(id,arr,cur,skip){const e=document.getElementById(id);
 if(e.dataset.done)return; e.dataset.done=1;
 const items=skip?['(skip)',...arr]:arr;
 e.innerHTML=items.map(x=>`<option ${x===cur?'selected':''}>${x}</option>`).join('')}
async function tick(){
 let d;try{d=await(await fetch('/state')).json()}catch(e){return}
 S=d;
 document.getElementById('mode').textContent=d.is_sim?'[SIMモック]':'[実機]';
 const f=document.getElementById('fsm');f.textContent=d.fsm;
 f.style.color=d.fsm==='DAMP'?'var(--bad)':(d.fsm==='RUNNING'?'var(--ok)':'var(--t1)');
 document.getElementById('tilt').textContent=d.tilt_deg.toFixed(0)+'°';
 document.getElementById('tilt').style.color=d.tilt_deg>25?'var(--warn)':'var(--t1)';
 document.getElementById('qd').textContent=d.qd_rms.toFixed(2);
 document.getElementById('hl').textContent=d.healthy?'OK':'途絶';
 document.getElementById('hl').style.color=d.healthy?'var(--ok)':'var(--bad)';
 fill('sel_climb',d.patterns.climb,d.sel.climb,true);
 fill('sel_turn',d.patterns.turn,d.sel.turn,true);
 fill('sel_sit',d.patterns.sit,d.sel.sit,true);
 document.getElementById('phases').textContent=
  d.phases.map((p,i)=>(i===d.phase_i?'▶':'')+p).join('  →  ')||'-';
 document.getElementById('prog').style.width=(100*d.t/Math.max(d.n,1))+'%';
 document.getElementById('tn').textContent=`フェーズ ${d.phase_i+1}/${d.phases.length}  コマ ${d.t}/${d.n}`;
 document.getElementById('msg').textContent=d.msg;
 document.getElementById('log').textContent=d.logs.join('\\n');
 document.getElementById('next').disabled=(d.fsm!=='WAIT_CONFIRM');
 document.getElementById('place').style.display=d.is_sim?'inline-block':'none';
 if(d.is_sim){const c=document.getElementById('cam');c.style.display='block';
  document.getElementById('noimg').style.display='none';
  c.src='/frame.jpg?t='+Date.now();}
}
setInterval(tick,200);tick();
document.addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();cmd('estop')}});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None
    patterns = None

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/state":
            d = self.engine.snapshot()
            d["patterns"] = self.patterns
            self._send(json.dumps(d).encode(), "application/json")
        elif p == "/frame.jpg" and self.engine.is_sim:
            self._send(self.engine.robot.render_jpeg(), "image/jpeg")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        q = parse_qs(urlparse(self.path).query)
        c = q.get("c", [""])[0]
        a = q.get("a", [None])[0]
        if c == "select" and a and ":" in a:
            k, v = a.split(":", 1)
            self.engine.command("select", (k, v))
        elif c in ("estop", "arm", "start", "next", "damp", "place_sim", "mode",
                   "mode_zero", "mode_damp", "mode_stand", "mode_walk",
                   "custom", "run_task"):
            self.engine.command(c, a)
        self._send(b"ok", "text/plain")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="MuJoCoモックで結合試験")
    ap.add_argument("--iface", default="", help="実機NIC名(例 enp46s0)")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    if a.sim:
        from sim_robot import SimRobot
        robot = SimRobot()
        print("SIMモックで起動")
    else:
        from real_robot import RealRobot
        robot = RealRobot(iface=a.iface)
        print("実機に接続しました")
    # 制御ループの尾側の遅れ対策。推論そのものは50Hz予算20msに対して
    # 中央1.1msしか使っていないが、GCの一時停止で数十msの飛びが出る。
    # 50Hzが1コマ遅れても500Hz送信スレッドが直前の目標を送り続けるので
    # 実害は小さいが、GILを握ったまま止まると送信自体が途切れうる。
    #  - freeze(): 起動時までに作った全オブジェクトを恒久世代へ移し、
    #              以降のGCの走査対象から外す(解放はしない=安全)
    #  - set_threshold: 世代0の閾値を上げて走行中の回収頻度を下げる
    # 効いているかは、ログの dt_ms / ms_infer を実機で見て判断する
    import torch
    torch.set_num_threads(1)          # 推論スレッドが送信スレッドを食わないように
    gc.collect()
    gc.freeze()
    gc.set_threshold(100_000, 50, 50)
    eng = Engine(robot, a.sim)
    Handler.engine = eng
    Handler.patterns = list_patterns()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"コックピット: http://localhost:{a.port}")
    try:
        srv.serve_forever()
    finally:
        robot.set_damp()


if __name__ == "__main__":
    main()

