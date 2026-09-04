#!/usr/bin/env python3
"""実機G1へのDDS接続(unitree_sdk2py)。コックピット/FSMランナー用。

前提(docs/REAL_ROBOT.md / REAL_CALIB.md の実測):
  - 有線LAN、PCは 192.168.123.0/24(例 192.168.123.222)、実機は .161
  - 送信前に MotionSwitcher の ReleaseMode() で標準制御を解放する
    (解放しないと rt/lowcmd をkp=0の指令が占有し、モータを取り合う)
  - LowCmd.mode_machine は LowState から読んだ値(実測5)をそのまま返す
  - mode_pr はPR(0)のまま
  - CRCを毎パケット計算する

★2026-08-26 改訂(実機投入前レビューの指摘に対応)。変更点:
  1. SDKのRPCは全て _rpc / _rpc_async 経由。**呼び出し側は絶対にブロックしない**
     (2026-08-20実測: ReleaseMode が5分以上返らずUIごと停止した)
  2. estop(): どのスレッドからでも即座に呼べる緊急停止。ブロックしない。
     一度掛かると **_estop_latched** が立ち、解除するまで set_target を受け付けない
     (50Hzループが別処理でブロックしていても、後からdampを上書きされない)
  3. 500Hz送信スレッドの生存監視。Write()の例外を握り潰さず、心拍が途絶えたら
     healthy() が False を返す(=コックピットが自動DAMP)。スレッドが死んだら
     自動で再起動する
  4. set_target() のガード: NaN/Inf の混入を弾き、目標を関節可動域+余裕に収め、
     1ステップの変化量を制限する。**ゲインは絶対に下げない**(保持力を奪わない)
  5. ensure_custom() は解放の成否を返す。失敗したら custom_active を立てない
  6. close(): damp を送ってから、実際に送信されるのを待って停止する(順序が逆だった)

※ probe.py / calib.py(実機PCに動作実績あり)と同じAPIを使っている。
  もし本ファイルで初期化に失敗する場合は、それらの初期化部と差分を比較すること。
"""
import pathlib
import threading
import time

import numpy as np

DAMP_KD_LEG = 5.0
DAMP_KD_ANKLE = 0.2

# --- set_target のガード定数 -------------------------------------------------
# ★ガードは「壊れた指令」だけを捕まえる外枠にする。正常な方策の指令に
#   触れてはいけない。安全のために足した機構が挙動を変えて事故になる
#   (2026-08-20の check_authority が脚を脱力させた件と同じ失敗)。
#
# 可動域の外へどれだけ目標を出してよいか[rad]。
#   target = ref_q + action*ACTION_SCALE で、action は方策側で ±1 に
#   クリップ済み。つまり**正常な指令のはみ出しは構造的に0.70radが上限**。
#   0.75 はその外側なので、正常運転では絶対に掛からない。
#   実測(2026-08-20/24の全34ログ、のべ約1.4万コマ)でも、可動域を最大
#   +0.54rad 超えるコマが climb_slow_r2 で0.66%あった。ここを0.15radなどに
#   締めると、**実機で完走している登り動作の足首指令を書き換えてしまう**。
#   締めるのではなく、はみ出し量をログに残して後で数える(guard_over)。
GUARD_RANGE_MARGIN = 0.75
# 1制御ステップ(20ms)で目標が飛んでよい量[rad]。
#   参照軌道からの計算上の最大は0.49rad/step、実機ログの実測最大は0.80rad/step
#   (2026-08-24 rd の転倒時)。1.5 はその外側で、正常運転では掛からない。
GUARD_STEP_MAX = 1.5
# NaN/Infをこの回数連続で受けたら、呼び出し側へ「異常」を返す
GUARD_NAN_TRIP = 3

# SwitchToUserCtrl(7110) のあと FSM=1000 を確認するまで待つ秒数。
# ★2026-09-03 オンボード実行(機体上でコックピットを走らせる)で 2.0秒では
#   足りず、切替に失敗したと誤判定していた。機体上では切替の最中に
#   コックピット(DDS受信 recvMC が1コアの7割 + user_lowcmdの500Hz送信)と
#   制御サービスが同じJetsonのCPUを奪い合い、GetFsmId が None(RPC不達)を
#   返し続ける。PC制御のときは500Hz送信がPC側にあり、機体は受信だけで
#   済んでいたのでこの競合が出なかった。
#   この窓の間に出しているのは user_lowcmd のストリームだけで、UserCtrlに
#   入るまでロボットは無視するので、延ばしても機体は動かない。
USER_CTRL_CONFIRM_S = 6.0
# 速度指令(SetVelocity)を受け付ける内蔵歩行のFSM(新FW)。自動歩行・手動操作用
WALK_FSMS = {500, 501, 801, 802}


def _rpc(label, fn, *args, timeout=2.5):
    """SDKのRPCを**必ず有限時間で**打ち切って呼ぶ。

    unitree_sdk2py の MotionSwitcher/LocoClient は SetTimeout を設定しても
    返ってこないことがある(2026-08-20 実測: ReleaseMode 呼び出しから5分以上
    戻らずUIが停止した)。戻らないスレッドは殺せないので、こちらが待つのを
    やめる。放置したスレッドはdaemonなのでプロセス終了時に消える。

    戻り値: (完了したか, 戻り値)
    """
    box = {}

    def run():
        try:
            box["r"] = fn(*args)
        except Exception as e:                     # noqa: BLE001
            box["e"] = e
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        print(f"★{label}: {timeout:.1f}秒で応答なし — 待つのをやめます")
        return False, None
    if "e" in box:
        print(f"★{label}: 例外 {box['e']}")
        return False, None
    return True, box.get("r")


def _rpc_async(label, fn, *args):
    """応答を待たずにRPCを投げる。

    引き継ぎ経路と緊急停止では、待ち時間がそのまま「バランス制御が無い時間」
    「停止が効かない時間」になる。StopMove も ReleaseMode も Damp も
    **効けばよい**もので、応答を確認する必要はない(効いたかどうかは
    check_authority と LowState で実測する)。
    2026-08-20 実測: StopMove の応答待ちで1.5秒溶けて引き継ぎが1.69秒になった。
    """
    threading.Thread(target=lambda: _rpc(label, fn, *args, timeout=5.0),
                     daemon=True).start()


def _load_joint_limits(joint_names=None):
    """MJCFから関節可動域を読む。読めなければ (None, None) を返してガードを外す。

    ガードのために model/ を読むだけ。物理シミュレーションはしない。
    """
    try:
        import mujoco
        root = pathlib.Path(__file__).resolve().parent.parent
        m = mujoco.MjModel.from_xml_path(str(root / "model" / "scene_task.xml"))
        if joint_names is None:
            z = None
            for d in sorted((root / "deploy").iterdir()):
                if (d / "reference.npz").exists():
                    z = np.load(d / "reference.npz")
                    break
            if z is None:
                return None, None
            joint_names = [str(s) for s in z["joint_names"]]
        lo, hi = [], []
        for nm in joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_joint")
            if jid < 0:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid < 0:
                return None, None
            lo.append(float(m.jnt_range[jid][0]))
            hi.append(float(m.jnt_range[jid][1]))
        return np.array(lo), np.array(hi)
    except Exception as e:                         # noqa: BLE001
        print(f"(可動域ガードは無効: {e})")
        return None, None


class RealRobot:
    """実機。SimRobotと同じAPI(state / set_target / set_damp / healthy / estop)"""

    def __init__(self, iface="", send_hz=500, watch_cmd=False):
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
        # --- 2026-08-27 追加: 再学習のために取れるものは全部取る。
        #     LowStateには載っているのに記録していなかった量:
        #       accel  … IMUの加速度計。動力学/摩擦の同定に効く
        #       rpy    … IMUが出す姿勢角(quaternionとの突き合わせ用)
        #       ddq    … 関節加速度。摩擦モデルの同定に直接使う
        #       vol    … モータ電圧。負荷での垂れ下がりが見える
        #       temp2  … 2つ目の温度センサ(巻線側)
        #       mstate … モータのエラーフラグ
        self.accel = np.zeros(3)
        self.rpy = np.zeros(3)
        self.imu_temp = 0.0
        self.ddq = np.zeros(29)
        self.vol = np.zeros(29)
        self.temps2 = np.zeros(29)
        self.mstate = np.zeros(29)
        self.mmode = np.zeros(29)
        self.tick = 0
        self.mode_pr = 0
        self.mode_machine = 5
        # --- 低頻度で拾うもの(意味が未文書 or ほぼ一定)。
        #     1042Hzのコールバックで毎回読むとGILを食うので **20回に1回**
        #     (=約50Hz。記録は50Hzなので十分)。
        self.msensor = np.zeros((29, 2))     # motor_state[i].sensor(uint32×2)
        self.mreserve = np.zeros((29, 4))    # motor_state[i].reserve(uint32×4)
        self.mot_ext = np.zeros((6, 5))      # 29〜34番(29dofでは未使用)
        self.remote = np.zeros(40)           # リモコンのボタン状態
        self.version = np.zeros(2)
        self.ls_reserve = np.zeros(4)
        self.crc = 0
        self._nstate = 0
        self._recv_err = 0
        self._last_state_t = 0.0
        self.target_q = np.zeros(29)
        self.kp = np.zeros(29)
        self.kd = np.zeros(29)

        # --- 送信スレッドの健全性(★指令だけ止まる故障を検出するため)
        self._send_beat = 0.0        # 最後に1パケット送れた時刻
        self._send_n = 0             # 送信できたパケット数
        self._send_err = 0           # 連続で送信に失敗した回数
        self._send_err_msg = ""

        # --- 緊急停止のラッチ。立っている間は set_target を一切受け付けない
        self._estop_latched = False
        self._estop_why = ""

        # --- 目標ガード
        self.q_lo, self.q_hi = _load_joint_limits()
        self.guard_n_clip = 0        # 可動域+余裕で実際に丸めた回数(=異常)
        self.guard_n_over = 0        # 素の可動域を超えた回数(=正常でも起きる)
        self.guard_over_max = 0.0    # そのはみ出しの最大[rad]
        self.guard_n_rate = 0        # 変化量で丸めた回数
        self.guard_n_nan = 0         # NaN/Infを弾いた回数
        self._nan_streak = 0

        # --- MotionSwitcher / LocoClient(標準モードとの排他制御に使う)
        self._msc = None
        self._loco = None
        self.custom_active = False      # True=カスタム方策がモータを持つ
        self._stream_on = False         # 500Hz送信の有効/無効
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client \
                import MotionSwitcherClient
            self._msc = MotionSwitcherClient()
            self._msc.SetTimeout(2.0)
            self._msc.Init()
        except Exception as e:                     # noqa: BLE001
            print(f"★MotionSwitcher初期化に失敗: {e}")
        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            self._loco = LocoClient()
            self._loco.SetTimeout(3.0)
            self._loco.Init()
        except Exception as e:                     # noqa: BLE001
            print(f"★LocoClient初期化に失敗: {e}(標準モードボタンは使用不可)")
        # 起動時は制御権を取らない(標準制御のスタンドロック等を維持)。
        # 方策を使う直前に ensure_custom() でシームレスに引き継ぐ。

        # --- 購読
        #   ★2026-09-03 ハンドラを付けずポーリングで読む。
        #     SDKは handler を渡すと DataReader に Listener を付ける。すると
        #     cyclonedds の受信スレッド(recvMC)が **1043回/秒 Python に入って
        #     GILを握り**、同じプロセスの50Hz制御ループを餓死させる。機体
        #     (Jetson ARM)ではこれで制御周期が71ms(14Hz)まで落ち、方策が
        #     自動DAMPされた。PC(x86)は単スレッド性能が高く表面化しなかった。
        #     handler=None なら Listener が付かず受信スレッドはPythonに入らない。
        #     読み取りは _state_poll_loop が約180Hzで行う(下記)。
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(None, 0)
        self._poll_stop = False
        self._th_poll = threading.Thread(target=self._state_poll_loop,
                                         name="lowstate_poll", daemon=True)
        self._th_poll.start()
        # --- 送信
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        # --- 新FWの User Development Mode 用トピック(rt/user_lowcmd)。
        #   ★これが ReleaseMode 方式との決定的な差:
        #     切替(7110)の前から流しておける。ロボットは切替まで
        #     user_lowcmd を**無視する**ので、競合窓が原理的にゼロになる。
        #     ReleaseMode は解放〜初回指令の間が必ず空く(実測0.20〜0.50秒)。
        #   使うときだけ作る(旧FWでは存在しないトピックへのpublishになるだけ)。
        self._LowCmd_cls = LowCmd_
        self._pub_user = None
        self._use_user_topic = False
        # --- lowcmd の購読(★既定で無効)。
        #   「内蔵制御がまだ指令を出しているか」を実測できるが、
        #   **同一プロセスで同じトピックを publish しながら subscribe すると
        #   cyclonedds 内部でデッドロックする**(2026-08-20 A/B実測: 購読ありで
        #   即停止、なしで心拍49Hz維持)。コックピットでは絶対に有効にしない。
        #   送信しない診断ツール(probe_lowcmd.py)だけが使ってよい。
        self._other_t = 0.0
        self._sub_cmd = None
        if watch_cmd:
            print("★watch_cmd=True: 送信しながらの lowcmd 購読は"
                  "cyclonedds のデッドロック実績があります(診断専用)")
            self._sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
            self._sub_cmd.Init(self._on_cmd, 10)
        # 最初のLowStateを待つ(mode_machineを読むため)
        t0 = time.time()
        while self._last_state_t == 0.0:
            if time.time() - t0 > 5.0:
                raise TimeoutError("LowStateが5秒来ない。配線とIP設定を確認")
            time.sleep(0.05)
        self._stop = False
        self._send_dt = 1.0 / send_hz
        self._th = None
        self._start_send_thread()

    # ---- 送信スレッド ------------------------------------------------------
    def _start_send_thread(self):
        self._th = threading.Thread(target=self._send_loop, daemon=True,
                                    name="lowcmd_send")
        self._th.start()

    def _send_loop(self):
        while not self._stop:
            t0 = time.time()
            if not self._stream_on:
                time.sleep(0.02)
                continue
            try:
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
                if self._use_user_topic and self._pub_user is not None:
                    self._pub_user.Write(c)
                else:
                    self._pub.Write(c)
            except Exception as e:                 # noqa: BLE001
                # ★握り潰さない。ここが黙って死ぬと、画面は正常なのに指令だけ
                #   止まる(=一番危ない壊れ方)。心拍を更新しないので healthy()
                #   が False になり、コックピットが自動DAMPする
                self._send_err += 1
                self._send_err_msg = str(e)
                if self._send_err in (1, 10, 100):
                    print(f"★lowcmd送信に失敗({self._send_err}回連続): {e}")
                time.sleep(0.002)
                continue
            self._send_err = 0
            self._send_beat = time.time()
            self._send_n += 1
            rest = self._send_dt - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)

    def send_alive(self):
        """500Hz送信が生きているか。(健全か, 最後に送れてからの経過秒)

        送信していないとき(_stream_on=False)は「健全」扱い。
        送信中なのに50ms心拍が来なければ異常(500Hzなら2msに1回来る)。
        """
        if not self._stream_on:
            return True, 0.0
        age = time.time() - self._send_beat
        alive = (self._th is not None and self._th.is_alive() and age < 0.05)
        return alive, age

    def _ensure_send_thread(self):
        """送信スレッドが死んでいたら作り直す(1回で戻らなければ諦めて報告)"""
        if self._stop:
            return
        if self._th is None or not self._th.is_alive():
            print("★lowcmd送信スレッドが停止していました — 再起動します")
            self._start_send_thread()

    # ---- 内蔵制御サービスの状態 --------------------------------------------
    # ---- 新FW: User Development Mode(7110/7111) -------------------------
    def set_topic(self, user):
        """送信先を rt/lowcmd ←→ rt/user_lowcmd で切り替える"""
        if user and self._pub_user is None:
            from unitree_sdk2py.core.channel import ChannelPublisher
            self._pub_user = ChannelPublisher("rt/user_lowcmd",
                                              self._LowCmd_cls)
            self._pub_user.Init()
        self._use_user_topic = bool(user)

    def _regist_user_apis(self):
        """★vendored SDK の LocoClient は 7110/7111 を登録していない。

        登録しないと `_Call` がロボットへ送信すらせず、クライアント内部で
        3103(API_NOT_REG)を返す(2026-08-26 実機12:43)。
        登録するだけなら旧FWでも無害。
        """
        if self._loco is None:
            return False
        for api in (7001, 7002, 7110, 7111):
            try:
                self._loco._RegistApi(api, 0)
            except Exception:                      # noqa: BLE001
                pass
        return True

    def get_fsm_id(self):
        """いまのFSM番号を読む(GETのみ)。取れなければ None"""
        if self._loco is None:
            return None
        try:
            self._loco._RegistApi(7001, 0)
        except Exception:                          # noqa: BLE001
            pass
        ok, res = _rpc("GetFsmId", self._loco._Call, 7001, "{}", timeout=2.5)
        if not ok or res is None:
            return None
        try:
            code, data = res
            import json as _j
            d = _j.loads(data) if isinstance(data, str) else data
            return int(d.get("data")) if isinstance(d, dict) else None
        except Exception:                          # noqa: BLE001
            return None

    def switch_to_user_ctrl(self):
        """API 7110。**戻り値は成否を表さない**(下記)。

        ★2026-08-26 実機: `{"data": false}` のような無効な引数でも code=0 を
          返しながら、実際には何も切り替わらない。歩行(500/501)は静止して
          いても「動的(fsm_mode=1)」扱いで、公式docの「状態が不適切なときは
          モード変更を禁止する」が黙って発動している。
          → **必ず GetFsmId()==1000 とウィグルで所有を実測すること。**
        """
        import json as _j
        if self._loco is None:
            return -1
        self._regist_user_apis()
        ok, res = _rpc("SwitchToUserCtrl(7110)", self._loco._Call, 7110,
                       _j.dumps({"data": False}), timeout=4.0)
        if not ok or res is None:
            return -1
        return int(res[0])

    def switch_to_internal_ctrl(self, mode=1):
        """API 7111。mode: 0=LAST(進入前へ) / 1=PASSIVE(ダンピング) / 2=WALKRUN

        ★既定は 1(PASSIVE)。2(WALKRUN)は当FWで **802(走行)** に入り、
          吊った機体が空中で走行動作を始めて暴れた実績がある(実機13:58)。
        """
        import json as _j
        if self._loco is None:
            return -1
        self._regist_user_apis()
        ok, res = _rpc(f"SwitchToInternalCtrl({mode})", self._loco._Call, 7111,
                       _j.dumps({"data": int(mode)}), timeout=4.0)
        if not ok or res is None:
            return -1
        self._use_user_topic = False
        return int(res[0])

    def return_to_balance(self, log=print):
        """こちらの制御(方策なしの現姿勢PD保持)を**内蔵バランス制御へ返す**。脱力しない。

        UI(スマホ/PC)との通信が途絶えたとき、方策で支えていない状態
        (=現姿勢PD保持。バランスが無く数秒で倒れる)を内蔵制御に引き取らせる
        ための経路(2026-09-04)。dampへ落とすと立っている機体が崩れる
        (実機で転倒事例)。「立っている途中なら、バランスを保ってそこで静止」がこれ。

          UserCtrl経路   : SwitchToInternalCtrl(LAST=0) → 進入前の状態(802 走行=
                           バランス立位、速度ゼロ)へ戻る。実機実績: 2026-08-26 15:22
                           往復実証 / 15:44 LAST返却で崩落なし(docs/ポータブル版 09)
          ReleaseMode経路: SelectMode(ai) → ロック立位(FSM 4)

        ★送信は切替の**後**に止める。先に止めると UserCtrl のまま指令が無い窓が
          できる。切替後は user_lowcmd が無視されるだけなので、RPC が返ってから
          止めれば競合しない(rt/lowcmd へ戻る前に必ず止める)。
        戻り値 (成功か, FSM)
        """
        if self._estop_latched:
            return False, None
        if self._loco is None:
            return False, None
        if self._use_user_topic:
            import json as _j
            self._regist_user_apis()
            ok, res = _rpc("SwitchToInternalCtrl(LAST)", self._loco._Call, 7111,
                           _j.dumps({"data": 0}), timeout=4.0)
            code = int(res[0]) if (ok and res is not None) else -1
            self._stream_on = False                # ← 先に送信を止めてから
            self.custom_active = False
            self._use_user_topic = False           #    送信先を rt/lowcmd へ戻す
            f = None
            t0 = time.time()
            while time.time() - t0 < 3.0:
                f = self.get_fsm_id()
                if f is not None and f != 1000:
                    break
                time.sleep(0.2)
            good = (f is not None and f != 1000)
            log(f"内蔵バランス制御へ返却(LAST): code={code} → FSM {f}"
                f"({time.time() - t0:.1f}秒)" + ("" if good else " ★切り替わっていない"))
            return good, f
        good = self.standard_mode("stand")
        log("内蔵制御へ返却: SelectMode(ai) → ロック立位(4)" + ("" if good else " ★失敗"))
        return good, (4 if good else None)

    def current_mode(self, timeout=2.5):
        """内蔵制御サービスの状態を返す(表示用)。'(解放中)' なら未使用。

        ★'?' は **CheckModeが読めなかっただけ** で、解放されている証拠では
          ない。両者を混同しないこと(_select_ai のコメント参照)。
        """
        if self._msc is None:
            return "?"
        ok, res = _rpc("CheckMode", self._msc.CheckMode, timeout=timeout)
        if not ok or res is None:
            return "?"
        code, result = res
        name = result.get("name") if result else None
        return name or "(解放中)"

    def _on_cmd(self, msg):
        """rt/lowcmd を監視。自分以外(=内蔵制御)の指令を見たら時刻を記録する。
        ★publish と同一プロセスで使わないこと(冒頭の注意)。"""
        kd = max(msg.motor_cmd[i].kd for i in range(29))
        kp = max(msg.motor_cmd[i].kp for i in range(29))
        if kd > 0.5 or kp > 0.5:            # 待機中の自分は kp=kd=0 で出す
            self._other_t = time.time()

    def wait_release(self, quiet=0.4, timeout=6.0):
        """内蔵制御が lowcmd を出さなくなるまで待つ。解放の**実測**確認。

        watch_cmd=True で作った RealRobot でのみ意味がある(それ以外は
        _other_t が更新されないので常に成功と返る)。
        呼ぶ側は kp=kd=0 で送信し続けていること。
        """
        if self._sub_cmd is None:
            return False, 0.0           # 購読していない=判定できない
        t0 = time.time()
        while time.time() - t0 < timeout:
            if time.time() - self._other_t > quiet:
                return True, time.time() - t0
            time.sleep(0.02)
        return False, time.time() - t0

    def _state_poll_loop(self):
        """LowStateを約180Hzで読む。**制御ループとは別スレッド**。

        Read() は take_one()。履歴QoSが KEEP_LAST(1) なので、溜まった古い
        サンプルではなく**常に最新**が返る(実測: tickの差分が中央5=1043/180
        で、キューに追従遅れが出ていないことを確認済み)。
        LowStateは1043Hzで来るので Read() は待たずに返り、実質この sleep が
        取得レートを決める。50Hz制御に対して遅れは最大5.5ms。

        ★例外を外へ出さないこと。このスレッドが死ぬとLowStateが更新されず、
          コックピットは受信途絶と判定して自動DAMPする。
        """
        while not self._poll_stop:
            try:
                msg = self._sub.Read()
                if msg is not None:
                    self._on_state(msg)
            except Exception as e:                 # noqa: BLE001
                self._recv_err += 1
                if self._recv_err in (1, 10, 100):
                    print(f"★LowStateの読み取りで例外({self._recv_err}回): {e}")
            time.sleep(0.0055)                     # ≒180Hz

    def _on_state(self, msg):
        """1042Hzで呼ばれる。**1回のループで全部読む**(GILを長く握らないため)

        ★例外を絶対に外へ出さないこと。ここで例外が出ると cyclonedds の
          受信スレッドが死に、**LowStateが二度と届かなくなる**(コックピットは
          「受信途絶」と判定して自動DAMPする)。2026-08-27に wireless_remote を
          bytes のまま numpy へ代入して実際にこれを起こした。
        """
        try:
            self._on_state_body(msg)
        except Exception as e:                     # noqa: BLE001
            self._recv_err += 1
            if self._recv_err in (1, 10, 100):
                print(f"★LowStateの取り込みで例外({self._recv_err}回): {e}")
            self._last_state_t = time.time()       # 受信自体は来ている

    def _on_state_body(self, msg):
        """★2026-09-03 オンボード対策。この関数は1043Hzで呼ばれる。

        従来は毎回29関節×9項目を numpy へ**1要素ずつ**代入していた
        (約260回/コール = 27万回/秒)。PC(x86)は単スレッド性能が高いので
        問題にならなかったが、機体(Jetson ARM 1.98GHz)ではこれだけで1コアを
        使い切り、**GIL越しに50Hz制御ループを餓死**させて制御周期71ms(14Hz)
        まで落ちた(実測。方策が自動DAMPされた)。

        いまは _state_poll_loop が約180Hzで呼ぶ(1043Hzの全コマではない)ので、
        制御に要る値は毎回取り込む。記録用の値だけ4回に1回=約45Hzに間引く
        (記録は50Hzなので実用上そろう)。
        あわせて1要素ずつの代入をやめ、内包表記＋スライス一括代入にする
        (numpyのスカラ代入は boxing のぶん遅い)。
        """
        n = self._nstate = self._nstate + 1
        now = time.time()
        ms = msg.motor_state
        im = msg.imu_state
        with self.lock:
            self.q[:] = [ms[i].q for i in range(29)]
            self.dq[:] = [ms[i].dq for i in range(29)]
            self.tau[:] = [ms[i].tau_est for i in range(29)]
            self.quat[:] = im.quaternion
            self.gyro[:] = im.gyroscope
            self.accel[:] = im.accelerometer
            self.rpy[:] = im.rpy
            self.tick = msg.tick
            self.mode_pr = msg.mode_pr
            self.mode_machine = msg.mode_machine
            if n % 4 == 0:                 # ≒45Hz。記録の頻度に合わせる
                self.ddq[:] = [ms[i].ddq for i in range(29)]
                self.vol[:] = [ms[i].vol for i in range(29)]
                self.mstate[:] = [ms[i].motorstate for i in range(29)]
                self.mmode[:] = [ms[i].mode for i in range(29)]
                self.temps[:] = [ms[i].temperature[0] for i in range(29)]
                self.temps2[:] = [ms[i].temperature[1] for i in range(29)]
                self.imu_temp = im.temperature
                for i in range(29):
                    m = ms[i]
                    self.msensor[i, :] = m.sensor
                    self.mreserve[i, :] = m.reserve
                for k in range(6):
                    j = 29 + k
                    if j < len(ms):
                        x = ms[j]
                        self.mot_ext[k, :] = (x.mode, x.q, x.dq, x.ddq,
                                              x.tau_est)
                # wireless_remote は bytes。numpyへ入れるには変換が要る
                self.remote[:] = np.frombuffer(bytes(msg.wireless_remote),
                                               dtype=np.uint8)
                self.version[:] = msg.version
                self.ls_reserve[:] = msg.reserve
                self.crc = msg.crc
            self._last_state_t = now

    # ---- 標準モード(SDK) / カスタム制御の排他
    def ensure_custom(self, kp=None, kd=None):
        """制御権をシームレスに引き継ぐ(脱力なし)。成否を返す。

        順序が重要:
          1) 現在の関節角 q_now を読む
          2) 先に 目標=q_now・指定ゲイン で500Hz送信を開始
             (標準制御と同じ姿勢の指令なので、解放までの重複期間も動かない)
          3) それから ReleaseMode で標準制御を解放
             → 解放の瞬間から当方のPDが同じ姿勢を保持し続ける
        逆順(先に解放)だと、解放〜初回指令の間モータが無支配になり脱力する。

        ★ReleaseMode の応答は待たない(_rpc_async)。待ち時間はそのまま
          「バランス制御が無い時間」になる。効いたかどうかは、この後に
          呼び出し側が check_authority() で**実測**すること。CheckMode は
          稼働中のFSMから解放すると「解放済み」と返すのに制御権が移らない
          実測があり、判定に使えない。
        """
        if self._estop_latched:
            print("★E-STOPラッチ中です。clear_estop() してから引き継いでください")
            return False
        if self.custom_active and self._stream_on:
            return True
        self._ensure_send_thread()
        q_now, _, _, _, _ = self.state()
        if kp is None:                             # 既定は公式ゲイン
            kp = np.zeros(29)
            kd = np.zeros(29)
            for i in range(12):                    # 脚: 股100/膝150/足首40
                kp[i] = 100.0 if i % 6 < 3 else (150.0 if i % 6 == 3 else 40.0)
                kd[i] = 3.0 if i % 6 < 3 else (4.5 if i % 6 == 3 else 1.5)
            kp[12:15] = 200.0; kd[12:15] = 6.0     # 腰
            kp[15:] = 70.0; kd[15:] = 2.5          # 腕
        # 2) 現姿勢を保持する指令を先に。ここは引き継ぎのラッチなので
        #    変化量ガードを掛けない(直前の目標がゼロでも一気に現姿勢へ乗せる)
        self.set_target(q_now, kp, kd, latch=True)
        self._stream_on = True
        time.sleep(0.05)                           # 数十パケット流してから
        ok_send, age = self.send_alive()
        if not ok_send:
            print(f"★送信が出ていません(心拍{age * 1000:.0f}ms前)。"
                  f"解放しません — {self._send_err_msg}")
            self._stream_on = False
            return False
        if self._msc is not None:                  # 3) それから解放
            _rpc_async("ReleaseMode", self._msc.ReleaseMode)
        time.sleep(0.03)
        self.custom_active = True
        print("カスタム制御: シームレス引き継ぎ完了(現姿勢を保持中)。"
              "制御権は check_authority で確認します")
        return True

    def enter_user_ctrl(self, kp=None, kd=None, log=print, settle=1.5):
        """★純正立位 → 走行(802) → UserCtrl(1000) を **脱力窓ゼロ**で通す。

        2026-08-26 の実機知見(docs/ポータブル版_設計メモ/09)にそのまま従う。

        手順:
          1. 制御サービスを有効化(SelectMode ai)
          2. FSM 802(走行)へ遷移   ★ここが危険。下記の警告
          3. 802 で静止するのを待つ
          4. 現姿勢をラッチして **rt/user_lowcmd** へ事前ストリーム開始
             → 切替までロボットは無視するので競合窓ゼロ
          5. SwitchToUserCtrl(7110)
          6. GetFsmId()==1000 を確認
          7. ウィグルで所有を実測(7110のACKは信用できない)

        ★警告(実機13:58): 802 は「静止立位に見えても走行制御が走っている状態」。
          吊った機体をここに入れて空中で走行動作を始め暴れた実績がある。
          さらに **ゼロトルク(FSM 0)への遷移が 802 から拒否**された。
          止めるときは ダンピング(FSM 1) か 7111 PASSIVE を使うこと。

        戻り値: (成功か, 説明)
        """
        if self._estop_latched:
            return False, "E-STOPラッチ中"
        self._regist_user_apis()
        if self._loco is None:
            return False, "LocoClient が無い"
        # 1) 制御サービスを確実に有効に
        if not self._select_ai():
            return False, "制御サービスを復帰できない"
        f0 = self.get_fsm_id()
        log(f"UserCtrl進入: いまのFSM = {f0}")
        # 2) 走行(802)へ
        if f0 != 802:
            log("★FSM 802(走行)へ遷移します — 走行制御が動きます。"
                "リモコンE-STOPを握ってください")
            ok, _ = _rpc("SetFsmId(802)", self._loco.SetFsmId, 802, timeout=4.0)
            t0 = time.time()
            while time.time() - t0 < 6.0:
                time.sleep(0.4)
                f = self.get_fsm_id()
                if f == 802:
                    break
            else:
                f = self.get_fsm_id()
            if f != 802:
                log(f"★802へ入れなかった(いまFSM={f})。中止します")
                return False, f"802へ遷移できない(FSM={f})"
            log(f"FSM 802 に到達({time.time() - t0:.1f}秒)")
        # 3) 静止待ち
        t0 = time.time()
        while time.time() - t0 < settle:
            _q, dq, _qt, _g, _t = self.state()
            if float(np.abs(dq).max()) < 0.15:
                break
            time.sleep(0.1)
        _q, dq, _qt, _g, _t = self.state()
        log(f"802で静止待ち {time.time() - t0:.1f}秒 "
            f"(関節速度 最大{float(np.abs(dq).max()):.3f}rad/s)")
        # 4) ラッチして user_lowcmd へ事前ストリーム
        q_now, _, _, _, _ = self.state()
        if kp is None:
            kp = np.zeros(29); kd = np.zeros(29)
            for i in range(12):
                kp[i] = 100.0 if i % 6 < 3 else (150.0 if i % 6 == 3 else 40.0)
                kd[i] = 3.0 if i % 6 < 3 else (4.5 if i % 6 == 3 else 1.5)
            kp[12:15] = 200.0; kd[12:15] = 6.0
            kp[15:] = 70.0; kd[15:] = 2.5
        self.set_topic(True)
        self.set_target(q_now, kp, kd, latch=True)
        self._stream_on = True
        self._ensure_send_thread()
        time.sleep(0.25)                       # ストリーム確立(まだ無視される)
        alive, age = self.send_alive()
        if not alive:
            self.set_topic(False); self._stream_on = False
            return False, f"user_lowcmdの送信が出ていない({self._send_err_msg})"
        # 5) 切替
        t_sw = time.time()
        code = self.switch_to_user_ctrl()
        log(f"SwitchToUserCtrl(7110) → code={code} ({time.time() - t_sw:.2f}秒)"
            f"  ※このACKは成否を表さない")
        # 6) FSM=1000 の確認
        f = None
        seen = []                       # 途中経過。?はGetFsmIdがRPC不達
        t0 = time.time()
        while time.time() - t0 < USER_CTRL_CONFIRM_S:
            f = self.get_fsm_id()
            seen.append("?" if f is None else str(f))
            if f == 1000:
                break
            time.sleep(0.2)
        if f == 1000 and len(seen) > 1:
            log(f"FSM 1000 の確認に {time.time() - t0:.1f}秒 かかりました"
                f"(経過 {'→'.join(seen[-8:])})")
        if f != 1000:
            log(f"★FSMが1000にならない(いま{f})。切り替わっていません。"
                f"送信を止めて中止します  経過 {'→'.join(seen[:24])}"
                f"  ※?はGetFsmIdがRPC不達(CPU競合の疑い)")
            self._stream_on = False
            self.set_topic(False)
            return False, f"UserCtrlへ入れない(FSM={f})"
        log(f"FSM 1000(UserCtrl)を確認({time.time() - t0:.1f}秒)")
        # 7) ウィグルで所有を実測
        self.custom_active = True
        ok, moved, tau, why = self.check_authority()
        sag = float(np.abs(self.state()[0] - q_now).max())
        if not ok:
            log(f"★ウィグルで所有を確認できない — {why}。中止します")
            self.custom_active = False
            self._stream_on = False
            self.set_topic(False)
            return False, "所有を確認できない"
        log(f"所有を確認: 左肩ピッチ +0.060rad指令 → 実測{moved:.3f}rad / "
            f"トルク{tau:.1f}Nm  掴んだ姿勢からのずれ {sag:.3f}rad")
        return True, "UserCtrl(FSM 1000)"

    def standard_mode(self, name):
        """Unitree標準モードへ(SDK)。こちらの送信は停止する。
        name: zero / damp / stand / walk / sit / seated
        FSM id(g1_loco_client): 0=ゼロトルク 1=ダンプ 2=スクワット 3=着座
                                4=立ち上がり 200=運用制御

        ★この関数はRPCを含むので**50Hz制御ループから直接呼ばない**こと
          (コックピットはワーカースレッドで呼ぶ)。
        """
        self._stream_on = False        # 送信停止(モータの取り合いを防ぐ)
        self.custom_active = False
        time.sleep(0.05)
        if not self._select_ai():
            print("★制御サービスを復帰できない。標準モードは効かない")
            return False
        if self._loco is None:
            print("★LocoClient未初期化のため標準モード不可")
            return False
        # ★これらは「効けばよい」FSM指令。応答が遅いだけで失敗扱いにしない。
        #   2026-08-26 実測: SelectMode("ai") は3.1秒で復帰しているのに
        #   Damp の応答が2.5秒で返らず「失敗→要確認」と出ていた(誤報)。
        #   実際に効いたかは LowState と操作者の目で分かる。
        if name == "zero":
            ok, _ = _rpc("SetFsmId(0)", self._loco.SetFsmId, 0, timeout=4.0)
        elif name == "damp":
            ok, _ = _rpc("Damp", self._loco.Damp, timeout=4.0)
        elif name == "stand":
            ok, _ = _rpc("SetFsmId(4)", self._loco.SetFsmId, 4, timeout=4.0)
        elif name == "walk":
            ok, _ = _rpc("SetFsmId(200)", self._loco.SetFsmId, 200, timeout=4.0)
        elif name == "sit":
            ok, _ = _rpc("SetFsmId(2)", self._loco.SetFsmId, 2, timeout=4.0)
        elif name == "seated":
            # FSM 3 = 着座。方策で座り終えた姿勢から内蔵制御へ渡す先。
            # スクワット(2)は立位でしゃがむモードなので、座った状態からは
            # 立ち上がろうとしうる。座った姿勢の引き継ぎにはこちらを使う。
            ok, _ = _rpc("SetFsmId(3)", self._loco.SetFsmId, 3, timeout=4.0)
        else:
            print(f"★未知の標準モード: {name}")
            return False
        if ok:
            print(f"標準モード: {name}")
        else:
            print(f"標準モード: {name}(送信済み。応答は返らなかった "
                  f"— 制御サービスは復帰しているので効いているはず)")
        return True                                # 制御サービスは復帰済み

    def _select_ai(self, timeout=6.0):
        """内蔵制御サービスを復帰させる。復帰を確認するまで粘る。

        ReleaseMode() の直後は SelectMode("ai") が通らないことがある(実測:
        1回だけ呼ぶ実装では4秒経っても解放中のままだった。別の回は0.5秒で
        戻った=間欠的)。SDKは戻り値でしか失敗を伝えないので、CheckMode で
        復帰を確認するまで繰り返す。
        ★連打はしない。0.5秒ごとに SelectMode を送ると切替完了前に上書きし
          続けて、いつまでも解放中のままになる(実測: 連打で6秒粘っても復帰
          せず、呼ぶのをやめた0.5秒後に復帰していた)。

        ★2026-09-03。CheckModeが返らないとき current_mode() は '?' を返すが、
          これは**読めなかっただけ**で解放されている証拠ではない。以前は
          '(解放中)' と同一視してここで6秒粘り、続けて呼び出し元のRPC(4秒)が
          走るので、UIのボタンが最大10秒返らなかった。その間 busy なので
          他のボタンも全部「処理中です」で弾かれ、操作不能に見えていた。
          コントローラーで歩かせた直後は loco 側が忙しく CheckMode の応答が
          遅れるため、必ずここに嵌まっていた(オンボードだとコックピットが
          制御サービスとCPUを分け合うぶん、さらに出やすい)。
          → 読めないときは粘らずに先へ進む。指令を送るほうが、送らずに
            諦めるより操作者の意図に沿う(諦めても機体の状態は改善しない)。
        """
        if self._msc is None:
            return False
        t0 = time.time()
        unread = 0
        for attempt in range(3):
            if time.time() - t0 > timeout:         # 実時間で必ず打ち切る
                break
            m = self.current_mode(timeout=0.8)     # 探りは短く(応答性優先)
            if m not in ("(解放中)", "?"):
                if attempt:
                    print(f"  制御サービス復帰({time.time() - t0:.1f}秒)")
                return True
            if m == "?":                           # 読めないだけ。粘らない
                unread += 1
                if unread >= 2:
                    print(f"  CheckModeが読めない({time.time() - t0:.1f}秒)。"
                          f"解放中とは限らないのでこのまま指令を送ります")
                    return True
                continue
            _rpc("SelectMode(ai)", self._msc.SelectMode, "ai")
            t1 = time.time()
            while time.time() - t1 < timeout / 3 and time.time() - t0 < timeout:
                if self.current_mode(timeout=0.8) not in ("(解放中)", "?"):
                    print(f"  制御サービス復帰({time.time() - t0:.1f}秒)")
                    return True
                time.sleep(0.3)
        return False

    def stop_move(self):
        """内蔵制御の速度指令をゼロにする。**引き継ぎの前に必ず呼ぶ。**

        ウォーキングFSM(200)は内蔵の目標が毎tick変わる。その途中で
        「いまの関節角」をラッチして保持すると、重心移動の途中の姿勢を
        固定することになる。2026-08-24の実測でも、walkから引き継いだ1回だけ
        方策開始時の傾きが7度(standからの6回は0〜4度)と最大だった。
        stand(FSM 4)では無害なので、モードを問わず呼んでよい。
        ★応答は待たない(待ち時間がそのまま引き継ぎの遅れになる)。
        """
        if self._loco is None:
            return
        fn = getattr(self._loco, "StopMove", None)
        if fn is not None:
            _rpc_async("StopMove", fn)
        else:
            _rpc_async("Move(0,0,0)", self._loco.Move, 0.0, 0.0, 0.0)

    def loco_move(self, vx, vy, omega):
        """ウォーキングモード中の速度指令(SDK)"""
        if self._loco is not None:
            _rpc_async("Move", self._loco.Move, vx, vy, omega)

    # ---- 内蔵歩行(自動歩行・スマホ手動操作)の入口(2026-09-04) ------------------
    def set_velocity(self, vx, vy, om, duration=0.5):
        """内蔵歩行への速度指令(同期・有限時間)。autowalk.VelSender だけが呼ぶ。

        duration 秒で内蔵側が勝手に止まるので、こちらのプロセスが死んでも
        機体は暴走しない。応答は0.8秒で打ち切る(送信スレッドの周期を守る)。
        """
        if self._loco is None:
            return False
        ok, _res = _rpc("SetVelocity", self._loco.SetVelocity, float(vx),
                        float(vy), float(om), float(duration), timeout=0.8)
        return bool(ok)

    def yaw(self):
        """IMUのヨー[rad](LowState)。直進保持と点群の体基準化に使う"""
        with self.lock:
            return float(self.rpy[2])

    def open_lidar(self):
        """rt/utlidar/cloud_livox_mid360 のポーリング読み手(autowalk.LidarReader)"""
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        from autowalk import LidarReader
        sub = ChannelSubscriber("rt/utlidar/cloud_livox_mid360", PointCloud2_)
        sub.Init(None, 0)                          # ハンドラを付けない(§3)
        return LidarReader(sub)

    def open_odom(self):
        """rt/odommodestate のポーリング読み手(autowalk.OdomReader)"""
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        from autowalk import OdomReader
        sub = ChannelSubscriber("rt/odommodestate", SportModeState_)
        sub.Init(None, 0)
        return OdomReader(sub)

    def ensure_walk_mode(self, log=print):
        """内蔵の歩行FSM(802。だめなら501)へ入れる。**ロック立位(4)からだけ**遷移する。

        ダンピング/ゼロトルクからは自動で立ち上げない(立ち上がりは操作者が
        [立つ]を押して目で見ながら行う)。
        ★802 は静止していても走行制御が動いている状態。吊った機体を入れると
          空中で走行動作を始めて暴れる(実機13:58)。**必ず接地させ、リモコンの
          E-STOPを握って**押すこと。2026-09-03 の実機では 4→802 が0.4秒で通っている。
        戻り値: (成功か, いまのFSM)
        """
        if self.custom_active or self._stream_on:
            return False, "方策側が制御権を持っています。先に[ダンプ]→[立つ]で内蔵制御へ戻すこと"
        if self._loco is None:
            return False, "LocoClient が無い"
        if not self._select_ai():
            return False, "制御サービスを復帰できない"
        f = self.get_fsm_id()
        if f in WALK_FSMS:
            return True, f
        if f != 4:
            return False, f"FSM={f}: 先に[立つ](ロック立位4)で立たせ、接地を確認してから"
        log("★FSM 802(走行)へ遷移します — 走行制御が動きます。"
            "機体を接地させ、リモコンE-STOPを握ってください")
        _rpc("SetFsmId(802)", self._loco.SetFsmId, 802, timeout=4.0)
        t0 = time.time()
        while time.time() - t0 < 6.0:
            time.sleep(0.4)
            f = self.get_fsm_id()
            if f in WALK_FSMS:
                log(f"FSM {f} に到達({time.time() - t0:.1f}秒)")
                return True, f
        log(f"802へ入れなかった(FSM={f})。501(腰3DoF歩行)を試します")
        _rpc("SetFsmId(501)", self._loco.SetFsmId, 501, timeout=4.0)
        t0 = time.time()
        while time.time() - t0 < 5.0:
            time.sleep(0.4)
            f = self.get_fsm_id()
            if f in WALK_FSMS:
                log(f"FSM {f} に到達({time.time() - t0:.1f}秒)")
                return True, f
        return False, f

    # ---- 共通API
    def state(self):
        with self.lock:
            return (self.q.copy(), self.dq.copy(), self.quat.copy(),
                    self.gyro.copy(), self.tau.copy())

    def motor_temps(self):
        with self.lock:
            return self.temps.copy()

    def state_full(self):
        """記録用に、LowStateから取れるものを全部返す(コピー)。

        state() は制御ループが毎コマ使うので軽いまま据え置き、
        こちらは記録のときだけ呼ぶ。
        """
        with self.lock:
            return dict(ddq=self.ddq.copy(), vol=self.vol.copy(),
                        temps2=self.temps2.copy(), mstate=self.mstate.copy(),
                        mmode=self.mmode.copy(), accel=self.accel.copy(),
                        rpy=self.rpy.copy(), imu_temp=float(self.imu_temp),
                        tick=int(self.tick), mode_pr=int(self.mode_pr),
                        mode_machine=int(self.mode_machine),
                        remote=self.remote.copy(),
                        version=self.version.copy(),
                        ls_reserve=self.ls_reserve.copy(), crc=float(self.crc),
                        msensor=self.msensor.copy(),
                        mreserve=self.mreserve.copy(),
                        mot_ext=self.mot_ext.copy())

    def set_target(self, q, kp, kd, latch=False):
        """目標を差し替える。**ここが実機へ出る指令の唯一の入口。**

        戻り値: (受理したか, 理由)。受理しなかったときは直前の目標を保持し続ける
        (=脱力しない)。

        ガードの原則(スキル原則1: 安全のために足した機構が事故を起こす):
          - **ゲインは絶対に触らない。** kp/kd を下げるガードは、崩れかけの
            機体から支持力を奪うので、事故そのものになる
          - NaN/Inf は受け付けない。直前の目標を保持する方が必ず安全
          - 可動域の外は「余裕 GUARD_RANGE_MARGIN まで」に丸める。
            機械的ストッパに当たった先を突っ張らせても意味がない
          - 1ステップの変化量は GUARD_STEP_MAX まで。通常運転では絶対に
            掛からない値(参照軌道の実測最大は0.49rad/step)
        latch=True は引き継ぎ・補間開始など「意図的に目標を大きく動かす」入口で、
        変化量ガードを外す。
        """
        if self._estop_latched:
            return False, "E-STOPラッチ中"
        q = np.asarray(q, dtype=float)
        kp = np.asarray(kp, dtype=float)
        kd = np.asarray(kd, dtype=float)
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(kp))
                and np.all(np.isfinite(kd))):
            self.guard_n_nan += 1
            self._nan_streak += 1
            return False, (f"NaN/Infを含む指令を拒否({self._nan_streak}回連続)")
        self._nan_streak = 0
        q = q.copy()
        if self.q_lo is not None:
            # まず素の可動域からのはみ出しを**数えるだけ**。方策が機械的
            # ストッパへ押し付けている量で、シムでも同じことが起きている。
            # 止めずに記録する(run設定.jsonとイベントログに残る)
            over = float(np.maximum(np.max(q - self.q_hi),
                                    np.max(self.q_lo - q)))
            if over > 0:
                self.guard_n_over += 1
                self.guard_over_max = max(self.guard_over_max, over)
            # 外枠での丸めは、構造的に有り得ない値(=壊れた指令)だけに掛かる
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
        """ガードとラッチを迂回してdampを書き込む(緊急停止と終了処理専用)"""
        kd = np.full(29, DAMP_KD_LEG)
        kd[[4, 5, 10, 11]] = DAMP_KD_ANKLE
        with self.lock:
            self.target_q = np.zeros(29)
            self.kp = np.zeros(29)
            self.kd = kd

    def set_damp(self):
        self._set_damp_raw()

    def estop(self, why="緊急停止"):
        """★どのスレッドからでも呼べる即時停止。**絶対にブロックしない。**

        - カスタム制御中: dampを送信バッファへ直接書く。500Hz送信スレッドが
          次のパケット(<2ms)で出す
        - 標準モード中 : SDKのDampを**応答を待たずに**投げる
        いずれの場合もラッチが立ち、clear_estop() するまで set_target() は
        一切受け付けない。50Hzループが別処理(RPC等)でブロックしていて、
        あとから古い目標を書きに来ても上書きされない。
        """
        self._estop_latched = True
        self._estop_why = why
        self._set_damp_raw()
        if self.custom_active:
            self._stream_on = True     # dampを確実に流す
            self._ensure_send_thread()
            if self._use_user_topic:
                # ★UserCtrl中は、dampを流すのに加えて内蔵制御へ返す。
                #   PASSIVE(1)=ダンピング。新FWのFSM表で「1は常に遷移可」。
                #   ★ゼロトルク(0)は 802 から拒否された実績があるので使わない。
                _rpc_async("SwitchToInternalCtrl(PASSIVE)",
                           self.switch_to_internal_ctrl, 1)
        elif self._loco is not None:
            _rpc_async("Damp(E-STOP)", self._loco.Damp)
        return True

    def clear_estop(self):
        """ラッチ解除。**再実行の直前に、操作者の明示操作でのみ呼ぶこと。**"""
        was = self._estop_latched
        self._estop_latched = False
        self._estop_why = ""
        return was

    def healthy(self):
        """LowStateが0.2秒以上途絶えた、または送信が止まったら不健康。
        (→コックピットが自動DAMP)

        ★受信だけを見ていると「画面は正常なのに指令だけ止まっている」故障を
          見逃す。送信スレッドの心拍も条件に入れてある。
        """
        if (time.time() - self._last_state_t) >= 0.2:
            return False
        alive, _ = self.send_alive()
        if not alive:
            self._ensure_send_thread()
            return False
        return True

    def health_detail(self):
        """UI/ログ用。何が不健全なのかを文字列で返す"""
        state_age = time.time() - self._last_state_t
        alive, send_age = self.send_alive()
        return {
            "state_age_ms": round(state_age * 1000, 1),
            "send_hz_ok": bool(alive),
            "send_age_ms": round(send_age * 1000, 1),
            "send_n": int(self._send_n),
            "send_err": int(self._send_err),
            "send_err_msg": self._send_err_msg,
            "recv_err": int(self._recv_err),
            "estop_latched": bool(self._estop_latched),
            "guard_clip": int(self.guard_n_clip),
            "guard_rate": int(self.guard_n_rate),
            "guard_nan": int(self.guard_n_nan),
            "guard_over": int(self.guard_n_over),
            "guard_over_max": round(float(self.guard_over_max), 3),
        }

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
        alive, age = self.send_alive()
        if not alive:
            return False, 0.0, 0.0, (f"500Hz送信が出ていない(心拍{age*1000:.0f}ms前"
                                     f" / {self._send_err_msg})")
        tgt = q0.copy()
        tgt[jid] += delta
        self.set_target(tgt, kp, kd, latch=True)
        time.sleep(dur)
        q1, _, _, _, tau1 = self.state()
        moved = float(q1[jid] - q0[jid])
        tau = float(tau1[jid])
        self.set_target(q0, kp, kd, latch=True)    # 現姿勢へ戻す(dampしない)
        # ★2026-08-27 14:32:15 の事故。旧判定は
        #     ok = 動いた or |トルク|>0.5
        #   で、**実測0.000rad・トルク0.7Nm を OK と返した**。
        #   トルクは「誰が指令していても」モータが報告する量で、重力と摩擦だけで
        #   0.5Nm は軽く出る。つまりトルク側は制御権の証拠にならない。
        #   ReleaseMode は既に送ってあるので、ここを通してしまうと
        #   「内蔵制御は手を離した / こちらの指令も届いていない」状態のまま
        #   走行に進み、脚を誰も持たずに膝から崩れる。
        #   **動いたことだけが証拠。** 過去の成功例は全て 0.049〜0.075rad で、
        #   しきい値0.018radに対し十分な余裕がある(3日間25回の実測)。
        ok = abs(moved) > 0.3 * delta
        if not ok:
            why = (f"指令が届いていない(動き{moved:+.4f}rad "
                   f"< 必要{0.3 * delta:.3f}rad / mode_pr・二重publish・"
                   f"制御権を疑う)")
            if abs(tau) > 0.5:
                why += f" ※トルク{tau:.1f}Nmは出ているが重力と摩擦でも出る量"
        else:
            why = ""
        return ok, moved, tau, why

    def close(self, flush=0.3):
        """終了処理。**damp を実際に送ってから**送信を止める。

        旧実装は _stop=True(送信スレッド停止)→ set_damp の順で、
        dampが1パケットも出ないままプロセスが終わり得た。
        """
        try:
            self._set_damp_raw()
            self._stream_on = True
            self._ensure_send_thread()
            t0 = time.time()
            n0 = self._send_n
            while time.time() - t0 < flush and self._send_n - n0 < 20:
                time.sleep(0.005)
            sent = self._send_n - n0
            print(f"終了: damp を {sent} パケット送信しました")
        except Exception as e:                     # noqa: BLE001
            print(f"★終了時のdamp送信に失敗: {e}")
        finally:
            self._stop = True
            self._poll_stop = True             # LowStateのポーリングも止める
            time.sleep(0.02)
