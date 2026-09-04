#!/usr/bin/env python3
"""実機G1のFSMランナー(雛形)。床→段→180度旋回→着座を状態機械で実行する。

★これは雛形であり、実機投入前に必ず吊りテストで検証すること(WP8)。
★DDSの初期化・送受信は、実機PCで動作実績のある real/probe.py /
  real/calib.py と同じ流儀に合わせて TODO 箇所を埋めること。

構成:
  受信スレッド: rt/lowstate を購読 → 最新状態を共有
  送信スレッド: rt/lowcmd へ500Hzで「現在の目標」を送り続ける
  メイン:      50HzのFSMループ(状態ごとの処理と遷移判定・安全監視)

使い方:
  python3 run_fsm.py --dry-run                 # 送信なしで観測とFSMだけ回す
  python3 run_fsm.py --climb climb_slow_r2 --turn standard --sit sit_up_r2

前提(docs/REAL_ROBOT.md の実測):
  - 送信前に MotionSwitcher の ReleaseMode() で標準制御を解放する
  - LowCmd の mode_machine は LowState から読んだ値(=5)をそのまま返す
  - mode_pr はPR(足首・腰の意味が変わる)
"""
import argparse
import json
import pathlib
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEPLOY = ROOT / "deploy"

CONTROL_HZ = 50
SEND_HZ = 500
OBS_HIST = 3
ACTION_SCALE = 0.70

# 安全しきい値(超えたら即DAMP)
TILT_LIMIT_DEG = 40.0
VEL_LIMIT = {"knee": 20.0, "hip": 32.0, "ankle": 30.0, "waist": 32.0}
DAMP_KD_LEG = 5.0          # dampモードの実測値(REAL_MODES.md)
DAMP_KD_ANKLE = 0.2

# 静止判定(フェーズ遷移用): 関節速度RMSがこの値未満をこの時間続ける
STILL_QD_RMS = 0.15
STILL_SECONDS = 0.5


class _Ref(dict):
    """reference.npz を**丸ごとメモリへ読み出した**辞書。

    ★2026-09-03。以前は np.load が返す NpzFile をそのまま持ち回っていたが、
      NpzFile は遅延読み込みで、`ref["ref_xy_abs"]` のたびに開いたままの
      zip を seek して解凍する。これを50Hzの制御ループから触ると
        - zipの共有ファイルハンドルへの読みが交錯し、zipfile の zip爆弾
          ガードが誤発火して `BadZipFile: Overlapped entries` で制御ループが
          落ちる(機体で実際に発生し自動DAMPになった)
        - 毎コマ解凍するぶん制御周期も食う
      ので、読み込み時に全部materializeして zip は閉じる。
      NpzFile と同じ使い勝手にするため `.files` も生やしてある。
    """

    @property
    def files(self):
        return list(self.keys())


# ---------------------------------------------------------------- 方策
class Policy:
    """deploy/<name>/ の TorchScript 方策と参照データ"""

    def __init__(self, name):
        import torch
        d = DEPLOY / name
        self.name = name
        self.net = torch.jit.load(str(d / "policy.pt")).eval()
        # ★暖機(2026-09-04): TorchScript の最初の推論はプロファイル実行で 120〜270ms かかる(実機ログ)。
        #   走行の1コマ目で払わず、読込のここで済ませておく
        try:
            _nin = int(json.loads((d / "meta.json").read_text(encoding="utf-8")).get("obs_dim", 615))
        except Exception:                          # noqa: BLE001
            _nin = 615
        with torch.no_grad():
            for _ in range(4):
                self.net(torch.zeros(1, _nin))
        # ★zipは読み切って閉じる(遅延読み込みのまま制御ループへ渡さない)。
        #   理由は _Ref のコメントを参照。
        with np.load(d / "reference.npz") as _z:
            z = _Ref({k: _z[k] for k in _z.files})
        self.ref_q = z["ref_q"]                    # (n, 29)
        self.kp = z["kp"]
        self.kd = z["kd"]
        self.joint_names = [str(s) for s in z["joint_names"]]
        self.n = len(self.ref_q)
        self.meta = json.loads((d / "meta.json").read_text())
        self.ref = z                               # ref_quat/ref_xy_abs/ref_z等
        # ★関節ごとの残差スケール(2026-09-02の配布から)。
        #   target = ref_q + action * scale_v で、学習時と同じ幅にする。
        #   ln20系は腕14関節が0.2、脚腰が0.7。**これを読まないと腕が
        #   学習時の3.5倍の幅で動く**(配布元 README_必読.md)。
        #   古い方策には無いので、その場合は従来どおり全関節 ACTION_SCALE。
        if "action_scale_v" in z.files:
            self.action_scale = np.asarray(z["action_scale_v"], dtype=float)
            self.has_scale_v = True
        else:
            sc = float(z["action_scale"]) if "action_scale" in z.files else ACTION_SCALE
            self.action_scale = np.full(29, sc, dtype=float)
            self.has_scale_v = False

    def act(self, obs):
        import torch
        with torch.no_grad():
            a = self.net(torch.as_tensor(obs, dtype=torch.float32)[None])
        return np.clip(a.numpy()[0], -1.0, 1.0)


# ---------------------------------------------------------------- 観測
class ObsBuilder:
    """rl/real_env.py の _observe_now と同一レイアウトの観測を実機信号から作る。

    使う量はエンコーダ・IMU・モデルFK・脚オドメトリのみ。
    MuJoCoのモデルはFK計算機として使う(シミュレーションはしない)。
    """

    def __init__(self, policy):
        import mujoco
        self.mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "scene_task.xml"))
        self.d = mujoco.MjData(self.m)
        # 関節 -> qposアドレス(DDS順=MJCF順は確認済み)。
        # 名前が見つからないと mj_name2id は -1 を返し、そのまま添字にすると
        # **黙って最後の関節/ボディを指す**。関節マッピングのずれは実機で
        # 四肢が伸びる・抽搐する形で出るので、ここで必ず落とす
        def _id(kind, name):
            i = mujoco.mj_name2id(self.m, kind, name)
            return int(i)

        self.qadr = []
        for nm in policy.joint_names:
            jid = _id(mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_joint")
            if jid < 0:
                jid = _id(mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid < 0:
                raise KeyError(f"MJCFに関節が無い: {nm}(scene_task.xml と "
                               f"deploy の reference.npz の関節名が不一致)")
            self.qadr.append(int(self.m.jnt_qposadr[jid]))
        self.qadr = np.array(self.qadr)
        self.fid = []
        for n in ("left_ankle_roll_link", "right_ankle_roll_link"):
            b = _id(mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                raise KeyError(f"MJCFにボディが無い: {n}")
            self.fid.append(b)
        self.com_bid = _id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.com_bid < 0:
            raise KeyError("MJCFにボディが無い: pelvis")
        # 接触観測を使う方策か(書き出し時の meta.json が唯一の真実)。
        # ここが sim 側の G1_CONTACT_OBS とずれると、方策は別物の入力を受け取る
        extra = policy.meta.get("obs_extra", {})
        self.contact_obs = bool(extra.get("contact", False))
        self.contact_dim = int(extra.get("contact_dim", 0))
        self.reset()

    def reset(self, est_xy=(0.0, 0.0), quat=None, ref_quat=None):
        """フェーズ開始時に呼ぶ。est_xy には**そのフェーズの参照開始位置**
        (pol.ref['ref_xy_abs'][0])を渡すこと。ロボットを参照開始位置に
        置いた前提で、以降は脚オドメトリで積分する。

        quat/ref_quat を渡すと**ヨー合わせ**を行う(2026-08-27追加)。

        観測には体の向きがワールド座標の回転行列 R[:2] として入り、参照側の
        向き ref_quat も同じワールド座標で入る。学習時のロボットは常に参照と
        同じ向き(この方策なら-90度)を向いていたので、方策はその近傍しか
        知らない。ところが実機のIMUのヨーには絶対基準が無く、電源を入れ
        直すたびに原点が変わる。2026-08-27の実測では、参照との差が
        -23度だった回は最大傾き23度で完走、+78度だった回は34度、+90度の
        回は膝の左右差1.16radという壊れ方をした
        (符号つき相関 r=0.83 / 実機13本)。

        差はワールドのヨー回転ひとつぶんなので、実測側を参照の座標系へ
        回してやれば消える。椅子はロボットとの相対位置で置かれていて、
        ワールド原点に対する向きには意味が無いから、この回転は課題を
        変えない。
        """
        self.hist = None
        self.prev_feet_rel = None
        self.last_cmd = np.zeros(29)
        self.est_xy = np.array(est_xy, dtype=float)   # 脚オドメトリの積分値
        self.yaw_off = 0.0
        self.Rz = np.eye(3)
        if quat is not None and ref_quat is not None:
            d = _yaw_of(ref_quat) - _yaw_of(quat)
            d = (d + np.pi) % (2 * np.pi) - np.pi
            self.yaw_off = float(d)
            c, s = np.cos(d), np.sin(d)
            self.Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return self.yaw_off

    def _contact_tail(self, z, t, n):
        """参照側の接触スケジュール(10次元)。rl/motion_env._contact_obs_tail と同一。

        参照の接触ラベル3 × 先読み3点 = 9、座面に触れるべき時刻までの残り[秒] = 1。
        自己位置(xy_off)は脚オドメトリの積分なのでドリフトするが、
        この予定表はドリフトしない。書き出し時に reference.npz へ同梱してある。
        """
        if not self.contact_obs:
            return np.zeros(0)
        ahead = [min(t + k, n - 1) for k in (0, int(0.15 * CONTROL_HZ),
                                             int(0.4 * CONTROL_HZ))]
        ref = np.asarray(z["ref_contact3"])[ahead].astype(np.float64).reshape(-1)
        dt_seat = float(np.clip((int(z["seat_onset"]) - t) / CONTROL_HZ,
                                -2.0, 2.0))
        return np.concatenate([ref, [dt_seat]])

    def _fk(self, q, quat):
        d = self.d
        self.mujoco.mj_resetData(self.m, d)
        d.qpos[0:3] = [0, 0, 1.0]                  # 位置は相対量にしか使わない
        d.qpos[3:7] = quat
        d.qpos[self.qadr] = q
        self.mujoco.mj_kinematics(self.m, d)
        self.mujoco.mj_comPos(self.m, d)
        base = d.qpos[0:3].copy()
        return (d.xpos[self.fid[0]].copy() - base,
                d.xpos[self.fid[1]].copy() - base,
                d.subtree_com[self.com_bid].copy() - base)

    def build(self, pol, t, q, dq, quat, gyro):
        """観測を作る。全て実機で取れる量のみ。

        素の方策は 205×履歴3 = 615次元。接触観測つきの方策(ContactMimic系)は
        各フレームの末尾に参照側の接触スケジュール10次元が付き、215×3 = 645次元。
        並び順は rl/real_env.py の _observe_now と一致していること
        (tests/test_obs_parity.py が番人)。
        """
        R = quat_to_mat(quat)                      # 実測(生のワールド)
        # ヨー合わせ後の向き。obsに入る「体の向き」と脚オドメトリの積分だけ
        # こちらを使う。体幹基準の量(R.T @ …)はヨー回転で不変なので生のRのまま
        Ra = self.Rz @ R
        z = pol.ref
        n = pol.n
        t = min(t, n - 1)
        ahead = [min(t + k, n - 1) for k in (0, int(0.15 * 50), int(0.4 * 50))]
        lf_w, rf_w, com_w = self._fk(q, quat)      # 体幹原点基準(ワールド向き)
        # 足基準の相対高さ
        h_rel = -min(lf_w[2], rf_w[2])
        ref_foot_z = z["ref_foot_z"][t]
        h_rel_ref = float(z["ref_z"][t]) - float(min(ref_foot_z))
        # 脚オドメトリ: 体幹から見た足位置の変化から体幹速度を推定
        feet_rel = np.stack([R.T @ lf_w, R.T @ rf_w])
        if self.prev_feet_rel is None:
            self.prev_feet_rel = feet_rel.copy()
        v_cands = []
        for k in (0, 1):
            v_rel = (feet_rel[k] - self.prev_feet_rel[k]) * 50.0
            v_cands.append(-(v_rel + np.cross(gyro, feet_rel[k])))
        v_est = min(v_cands, key=lambda v: float(v @ v))
        # 水平位置の推定も脚オドメトリで積分(開始時に床マーキングで原点合わせ)
        self.est_xy += (Ra @ v_est)[:2] / 50.0
        self.prev_feet_rel = feet_rel.copy()
        xy_off = z["ref_xy_abs"][t][:2] - self.est_xy   # TODO: AprilTag導入時は置換
        obs = np.concatenate([
            [h_rel - h_rel_ref],
            Ra[:2].reshape(-1),
            v_est * 0.3,
            gyro * 0.2,
            q,
            dq * 0.1,
            *[pol.ref_q[a] - q for a in ahead],
            quat_to_mat(z["ref_quat"][t])[:2].reshape(-1),
            xy_off,
            [t / n],
            R.T @ com_w,
            R.T @ lf_w,
            R.T @ rf_w,
            self.last_cmd,
            self._contact_tail(z, t, n),
        ]).astype(np.float32)
        if self.hist is None:
            self.hist = [obs.copy() for _ in range(OBS_HIST)]
        else:
            self.hist.append(obs)
            self.hist = self.hist[-OBS_HIST:]
        return np.concatenate(self.hist)


def _yaw_of(q):
    """クォータニオンからヨー角[rad]。DDSと同じ (w,x,y,z) の並び。"""
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- DDS入出力
class Robot:
    """LowState購読とLowCmd送信。★probe.py/calib.py の流儀で実装すること"""

    def __init__(self, dry_run):
        self.dry = dry_run
        self.lock = threading.Lock()
        self.q = np.zeros(29)
        self.dq = np.zeros(29)
        self.tau = np.zeros(29)
        self.quat = np.array([1.0, 0, 0, 0])
        self.gyro = np.zeros(3)
        self.mode_machine = 5
        self.fresh = False
        # 送信目標(送信スレッドが500Hzで読み続ける)
        self.target_q = np.zeros(29)
        self.kp = np.zeros(29)
        self.kd = np.zeros(29)
        if not dry_run:
            self._init_dds()

    def _init_dds(self):
        # TODO: real/probe.py と同じ ChannelFactoryInitialize / 購読 / 送信を
        #       ここに移植する。要点:
        #   - MotionSwitcherClient.ReleaseMode() を必ず先に呼ぶ
        #   - LowCmd.mode_machine = LowState.mode_machine(実測5)
        #   - mode_pr のままにする
        #   - CRC計算(unitree_sdk2py の実装どおり)
        raise NotImplementedError("probe.py の DDS 初期化をここへ移植")

    def state(self):
        with self.lock:
            return (self.q.copy(), self.dq.copy(), self.quat.copy(),
                    self.gyro.copy(), self.tau.copy())

    def set_target(self, q, kp, kd):
        with self.lock:
            self.target_q = q.copy()
            self.kp = kp.copy()
            self.kd = kd.copy()

    def set_damp(self):
        kd = np.full(29, DAMP_KD_LEG)
        kd[[4, 5, 10, 11]] = DAMP_KD_ANKLE         # 足首はdamp実測値
        self.set_target(np.zeros(29), np.zeros(29), kd)


# ---------------------------------------------------------------- FSM
def still_check(robot, seconds=STILL_SECONDS):
    """関節速度が十分小さい状態が続いたか"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        _, dq, _, gyro, _ = robot.state()
        if float(np.sqrt(np.mean(dq ** 2))) > STILL_QD_RMS:
            return False
        time.sleep(0.02)
    return True


def safety_ok(robot):
    q, dq, quat, gyro, tau = robot.state()
    up_z = quat_to_mat(quat)[2, 2]
    if up_z < np.cos(np.radians(TILT_LIMIT_DEG)):
        return False, "tilt"
    # 関節速度(脚のみ厳格)
    # TODO: 関節indexとVEL_LIMITの対応を joint_names で引く
    if float(np.abs(dq[:12]).max()) > 32.0:
        return False, "joint_vel"
    return True, ""


def run_policy_phase(robot, obs_b, pol, logf, dry):
    """1フェーズを50Hzで実行。参照終端で正常終了、安全違反でFalse"""
    obs_b.hist = None
    obs_b.prev_feet_rel = None
    dt = 1.0 / CONTROL_HZ
    for t in range(pol.n):
        t0 = time.time()
        q, dq, quat, gyro, tau = robot.state()
        obs = obs_b.build(pol, t, q, dq, quat, gyro)
        a = pol.act(obs)
        obs_b.last_cmd = a.copy()
        target = pol.ref_q[min(t, pol.n - 1)] + a * pol.action_scale
        robot.set_target(target, pol.kp, pol.kd)
        ok, why = safety_ok(robot)
        if not ok:
            print(f"★安全違反({why})→DAMP")
            robot.set_damp()
            return False
        if logf:
            np.save(logf, dict(t=t, q=q, dq=dq, quat=quat, gyro=gyro,
                               tau=tau, obs=obs, act=a, target=target),
                    allow_pickle=True)
        rest = dt - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)
    return True


def interpolate_to(robot, q_goal, kp_goal, kd_goal, seconds=3.0):
    """現在姿勢から目標姿勢へゆっくり補間(kpも0からランプ)"""
    q0, _, _, _, _ = robot.state()
    steps = int(seconds * CONTROL_HZ)
    for i in range(steps):
        w = (i + 1) / steps
        w = w * w * (3 - 2 * w)
        # 目標は現在姿勢から始まる(=初期トルク0)ので、kpは最初からフル値。
        # kpをランプすると序盤の支持力が消えて立位が崩れる(モックで実測)
        robot.set_target((1 - w) * q0 + w * q_goal, kp_goal, kd_goal)
        ok, why = safety_ok(robot)
        if not ok:
            robot.set_damp()
            return False
        time.sleep(1.0 / CONTROL_HZ)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--climb", default="climb_slow_r2")
    ap.add_argument("--turn", default="standard",
                    help="standard=Unitree標準コントローラ / turn_wide_r2等=方策")
    ap.add_argument("--sit", default="sit_up_r2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    climb = Policy(a.climb)
    sit = Policy(a.sit)
    turn = None if a.turn == "standard" else Policy(a.turn)
    robot = Robot(a.dry_run)
    obs_b = ObsBuilder(climb)
    log = (ROOT / "logs" / "real" / time.strftime("fsm_%Y%m%d_%H%M%S"))
    log.mkdir(parents=True, exist_ok=True)
    print(f"FSM開始 climb={a.climb} turn={a.turn} sit={a.sit} log={log}")

    # --- MOVE_TO_START
    input("段の手前のマーキングに立たせ、吊り具を確認してEnter(中止はCtrl+C)")
    if not interpolate_to(robot, climb.ref_q[0], climb.kp, climb.kd):
        return
    # --- CLIMB
    obs_b.reset(est_xy=climb.ref["ref_xy_abs"][0][:2],
                 quat=robot.quat.copy(), ref_quat=climb.ref["ref_quat"][0])
    if not run_policy_phase(robot, obs_b, climb, log / "climb.npy", a.dry_run):
        return
    if not still_check(robot):
        print("★登頂後に静止せず→DAMP")
        robot.set_damp()
        return
    # --- TURN
    if turn is None:
        # TODO: MotionSwitcherClient.SelectMode で標準制御へ渡し、
        #       LocoClient で低速のその場旋回+位置合わせを指令、
        #       完了後に ReleaseMode で取り戻す(WP5。要実機検証)
        print("標準コントローラの旋回は未実装(TODO)。方策指定で代替可")
        robot.set_damp()
        return
    obs_b.reset(est_xy=turn.ref["ref_xy_abs"][0][:2],
                 quat=robot.quat.copy(), ref_quat=turn.ref["ref_quat"][0])
    if not run_policy_phase(robot, obs_b, turn, log / "turn.npy", a.dry_run):
        return
    # --- ALIGN CHECK(±3cm/±15度は脚オドメトリ推定。AprilTag導入で置換)
    if not still_check(robot):
        robot.set_damp()
        return
    # --- SIT
    obs_b.reset(est_xy=sit.ref["ref_xy_abs"][0][:2],
                 quat=robot.quat.copy(), ref_quat=sit.ref["ref_quat"][0])
    if not run_policy_phase(robot, obs_b, sit, log / "sit.npy", a.dry_run):
        return
    print("完了。着座姿勢で保持中(Ctrl+CでDAMPして終了)")
    try:
        while True:
            time.sleep(0.5)
            ok, _ = safety_ok(robot)
            if not ok:
                break
    finally:
        robot.set_damp()


if __name__ == "__main__":
    main()
