#!/usr/bin/env python3
"""自動歩行(前進 → 壁の手前で自然に停止 / 障害物は回り込んで元の経路へ / 横歩きは
細かい足踏み)と、スマホからの手動操作(押している間だけ動く)のための速度送信。

★方策(rt/lowcmd)は一切使わない。Unitree内蔵の歩行制御(loco FSM 500/501。802 では速度指令が効かない)に
  LocoClient.SetVelocity(vx, vy, omega, duration) を10Hzで送るだけ。
  バランスは内蔵制御が取る。こちらが送るのは速度の希望値だけ。

構成:
  LidarReader      rt/utlidar/cloud_livox_mid360 をポーリング購読(実機)
  OdomReader       rt/odommodestate(位置・速度)をポーリング購読(実機)
  ObstacleDetector 点群 → 体基準(前・左・床からの高さ)へ変換 → 前方の最近点距離、
                   その物体の横幅(壁か障害物か)、左右どちらへ回り込めるか、横の空き
  VelSender        10Hzの速度送信スレッド。指令が0.5秒更新されなければ自動で停止
  AutoWalk         前進 / 回り込み / 横歩き の状態機械(スレッド)
  WalkController   上の全部をまとめてコックピット(Engine)へ1つの入口で見せる

前進の速さ(2026-09-04): 壁までの残り距離に応じて段階的に落とし、最後は忍び足で止まる
  残り ≥1.5m 100% / 1.0〜1.5m 80% / 0.6〜1.0m 55% / 0.25〜0.6m 35% / 0〜0.25m 15%(最低0.08m/s)
  さらに加減速に上限(加速0.3・減速0.6 m/s²)を掛けて滑らかにする
回り込み: 前方の物体が「壁」(横幅1.4m超、または両側とも1m以上)でなく、左右どちらかに
  体が通る幅(物体の端+肩幅+余裕)が空いていれば、そちらへ横移動 → 物体の奥まで前進 →
  元の経路線へ横移動して戻り → 前進を続ける(オドメトリで経路線を保持)
横歩き: 0.5秒進んで0.4秒止まる「足踏みパルス」を繰り返し、残り距離に応じて歩幅を縮める
  (到達許容2cm)。5cm刻みの微調整にも同じ経路を使う

安全(このファイルの中で守っていること):
  - 速度指令には必ず duration=0.5秒 を付ける。**こちらのプロセスが死んでも0.5秒で止まる**
  - 送信は VelSender の1スレッドだけ。指令が0.5秒更新されなければゼロを送る(デッドマン)
  - 自動歩行は LiDAR途絶 / オドメトリ途絶 / 傾き / UIハートビート途絶 / 最大距離 / 時間 の
    どれでも止まる。止め方は「速度ゼロ」であって damp ではない(歩行中に damp すると倒れる)
  - 障害物の判定は連続2回で成立。「見えない」ことは安全側に倒し、LiDARが届かなければ前進しない

座標の約束(この中では全部これ):
  fwd = 機体前方[m]、lat = 機体左[m](右は負)、h = 床からの高さ[m]
  vx = 前進[m/s]、vy = 左[m/s]、omega = 反時計回り[rad/s](SDKと同じ)
  経路線 = 開始時の位置と向き。s = 経路に沿った進み、e = 経路からの左へのずれ
"""
import json
import os
import math
import pathlib
import threading
import time

import numpy as np

# ---------------------------------------------------------------- 既定値
WALK_DEFAULTS = dict(
    v_fwd=0.50,        # 前進の巡航速度[m/s](かんたん画面の「速さ」: ゆっくり0.3/ふつう0.5/はやい0.7)
    v_side=0.25,       # 横歩き・後退の巡航速度[m/s](普通の歩行。横は前進より不安定なので低め)
    v_creep=0.15,      # 歩き続けられる最低速度[m/s]。★内蔵歩行は 0.1m/s 未満では歩かない(2026-09-04 実測の疑い)
    a_dec=0.15,        # 壁・目標へ向けた減速度[m/s²]。v²/(2a) 手前から滑らかに落とす(0.5m/s なら 0.8m 手前)
    cmd_lag=0.80,      # 内蔵歩行の応答遅れの見込み[s](実測: 指令を変えてから実速度が追いつくまで約 1 秒)。この分だけ早めに減速する
    align_wall=True,   # 前進の前に、正面の壁が斜めならその場で回転して正対する(2026-09-04 操作者の指示)
    align_tol_deg=3.0, # 正対したとみなす角度[deg]
    om_turn=0.30,      # 正対の回転速度の上限[rad/s]
    align_inplace_deg=8.0,  # 壁の角度がこれを超えていたら前進の前にその場で回る。以下なら歩きながら合わせる
    wall_track=True,   # 歩行中も壁の角度を測り続け、向きを壁に垂直へ寄せ続ける(2026-09-04 操作者の指示)
    veer_v=0.35,       # 回り込み(斜め歩き)中の前進速度の上限[m/s]
    # 最新 SDK(unitree_sdk2 例: Start→StandUp→SetSpeedMode→ContinuousGait→Move)の歩き方(2026-09-04 夕)
    speed_mode=0,      # SetSpeedMode(7107): -1/0/1/2(SDK 例の取りうる値。0=標準)。歩行モードに入るときに送る
    gait_cont=False,   # ContinuousGait(SetBalanceMode 1): 速度ゼロでも足踏みを続ける。★実機 14:09 で壁の手前で
                       #   足踏みのまま居座った(操作者の指摘)ので既定 OFF。0 なら指令ゼロで自然に立ち止まる
                       #   段差を無くす。歩行が終わったら 0(静止立位)へ戻す
    # ★速度の較正(2026-09-04 14:08 の実機ログ): 指令 0.50 で実速度 0.15〜0.20、0.42→0.13、0.35→0.08、0.20→0.04、
    #   0.15→0.02。実速度 ≈ k·(指令 − 不感帯) で k≈0.4、不感帯≈0.10。この層が無いと「遅い」「壁の前で足踏みのまま居座る」
    k_vx=0.45,         # 前後: 実速度/(指令−不感帯)。歩きながら実測で更新し walk_calib.json に残す
    k_vy=0.35,         # 横: 同上
    v_dead=0.10,       # 不感帯[m/s](これ未満の指令では歩かない)
    cmd_min_walk=0.30, # 動かすときの指令の下限[m/s]。★これ未満の指令だと内蔵歩行は歩き出さず、揺り戻り(後ろへ)だけが残る
    calib_learn=False, # 速度係数の実測更新。★歩くたびに係数が大きくなり指令が小さくなって後退量が増えた(操作者の報告)→ 既定 OFF
    cmd_max=0.90,      # 指令の上限[m/s]
    final_zone=0.60,   # 目標までこの距離[m]に入ったら一定のゆっくり歩き(実速度 0.08 目安)で寄せ、残り 8cm で一度だけ止める
    anchor_s=6.0,      # 止めた後、この秒数は位置を見張り、後ろへ 5cm 以上ずれたら前へ寄せ直す(アンカー保持)
    stop_lock=True,    # 目的地で止まったら足踏みをやめてロック立位(FSM 4)で静止する(操作者の指示 2026-09-04)。
                       #   次の[前進]/[横歩き]は自動で歩行(200)へ戻ってから動く
    yaw_autocal=False, # LiDAR ヨーの自動較正。★壁が主な景色だと壁沿いの平行移動が決まらず ±20° の雑音になった(14:0x)。既定 OFF
    cmd_dur=2.0,       # SetVelocity の duration[s]。SDK の Move(continuous) は 864000 だが、自プロセスが死んでも
                       #   2 秒で止まるように有限にする。送信は 10Hz で上書きし続ける
    stop_dist=0.60,    # 壁(障害物)のこの距離手前で止まる[m](骨盤基準)
    side_dist=0.50,    # 横歩きの距離[m]
    side_dir="left",   # 横歩きの向き "left"/"right"
    max_fwd=4.0,       # 壁が無いときに前進をやめる距離[m](安全上限)
    half_w=0.35,       # 前方コリドーの半幅[m]。G1の肩幅約0.45mに余裕
    h_min=0.12,        # 床からこの高さ未満の点は無視(床の凹凸・敷物)
    h_max=1.80,        # この高さ超は無視(天井・照明)
    side_clear=0.45,   # 横移動中、進行方向にこの距離未満で点があれば止まる[m]
    self_fwd=0.40,     # 自分の体(頭のLiDARの真下〜腕)を除く範囲: 前方この距離まで[m]
    self_lat=0.50,     # 同・左右この幅まで[m]
    yaw_fix_deg=0.0,   # センサ座標の点群に足すヨー[deg](lidar_mount.json の値 − ブリッジが適用済みの値)。[前後を反転]で±180
    front_offset=0.15, # センサ軸→つま先の前方距離[m]。距離は「つま先から」で出す
    avoid=True,        # 壁以外の障害物は回り込んで避ける
    wall_width=1.4,    # 前方の物体の横幅がこれ以上なら「壁」(回り込まない)[m]
    detour_margin=0.08, # 回り込みで障害物の端と体の側面との余白[m]。★ギリギリ(操作者の指示)
    body_half=0.25,    # 体の半幅[m](肩 0.22 + 腕の振り)。回り込みの横位置 = 端 + body_half + detour_margin  # 回り込みで物体の端から取る余裕[m]
    detour_max=0.9,    # 回り込みの最大横オフセット[m](これ以上要るなら壁扱い)
    side_tol=0.02,     # 横移動の到達許容[m]
    pulse_on=0.5,      # 足踏みパルス: 動く時間[s]
    pulse_off=0.4,     # 足踏みパルス: 止まって測る時間[s]
    dry_run=False,     # True = 速度を一切送らず、判断とログだけ
    tele_vx=0.35, tele_vy=0.20, tele_om=0.45,   # 手動操作の上限
    mode="both",       # "both"=前進→横移動 / "forward"=前進して壁の手前で止まるだけ / "side"=横移動だけ
                       # "back"=後退だけ(椅子との距離を詰める。足踏みパルス、LiDAR不要)
    back_dist=0.05,    # 後退の距離[m]
    # 小刻みステップ(2026-09-04 午後): 内蔵歩行が「確実に一歩出る」短い速度指令を出し、
    # 止めてオドメトリで測る、を繰り返す。弱いパルス(0.08m/s×0.5s)では一歩も出なかった実測から
    step_v=0.15,       # 1歩の望む実速度[m/s](較正で指令に直す。指令は 0.4 前後になる)
    step_on=0.6,       # 1歩の指令時間[s]
    step_off=1.0,      # 止めて着地と計測を待つ時間[s]
    step_min_on=0.3,   # 残りが小さいときの最短指令時間[s]
    step_est=0.06,     # 1歩の推定移動量[m]。実測で更新
    step_max=20,       # 1回の指示で出す最大歩数
    step_dir="left",   # mode="step"(1歩だけ)の向き: left/right/back/fwd
)
WALK_FSMS = {200, 500, 501}        # 速度指令を受ける内蔵FSM(loco)。802/801 では効かない(2026-09-04 実測)
YAW_KP = 1.6                       # 直進保持のゲイン[(rad/s)/rad](旧コックピット実績値)
YAW_OM_MAX = 0.30                  # 直進保持の補正上限[rad/s]
LAT_KP = 0.6                       # 経路線への横ずれ補正[(m/s)/m]
LAT_VY_MAX = 0.05                  # 同・上限[m/s](望む実速度。較正後の指令は 0.25 程度)
ACC_UP = 0.50                      # 加速の上限[m/s²](0.5m/s まで 1 秒)
ACC_DOWN = 0.80                    # 急な減速の上限[m/s²](プロファイル自体は a_dec で滑らか)
SEND_HZ = 10.0                     # 速度送信の周期
CMD_HOLD_S = 0.5                   # 指令の有効期間。これを過ぎたらゼロを送る
TILT_ABORT_DEG = 25.0              # 歩行中にこれを超えたら自動歩行を止める
SENSOR_FWD_OFFSET = 0.10           # センサ座標の点群を使うときのセンサ→骨盤の前方オフセット[m]
MAX_DETOURS = 2                    # 1回の前進で回り込む回数の上限

# Livox Mid-360 の点の型(unitree utlidar 配信の実測。フィールド情報が来れば
# そちらを優先して組み立て直す)
PT_DTYPE_DEFAULT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("intensity", "<f4"), ("ring", "<u2"),
                             ("time", "<f4")])
_PF_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def dtype_from_fields(fields, point_step):
    """PointCloud2 の fields/point_step から numpy dtype を組む。
    x/y/z が無い・組めないときは None(呼ぶ側は既定dtypeへ倒す)。"""
    try:
        names, formats, offsets = [], [], []
        for f in fields:
            t = _PF_TYPES.get(int(f.datatype))
            if t is None:
                continue
            names.append(str(f.name))
            formats.append("<" + t)
            offsets.append(int(f.offset))
        if not {"x", "y", "z"} <= set(names):
            return None
        return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                         "itemsize": int(point_step)})
    except Exception:                              # noqa: BLE001
        return None


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def speed_profile(v_max, d_rem, v_min=0.15, a_dec=0.25, v_now=0.0, lag=0.3, tol=0.03):
    """残り距離 d_rem[m](目標・停止点まで)に応じた目標速度[m/s]。**連続で滑らか**。
    減速度 a_dec で止まれる速度 sqrt(2·a·d) を上限に、巡航 v_max と最低 v_min の間に収める。
    応答遅れ lag の分(v_now·lag)だけ手前から減速する。残りが tol 以内なら 0。
    (2026-09-04 午後: 段階式 stage_speed を置き換え。操作者の「距離に応じて可変・なめらか」)"""
    if d_rem is None:
        return float(v_max)
    d = float(d_rem) - float(v_now) * float(lag)
    if float(d_rem) <= tol:
        return 0.0
    if d <= 0.0:
        return float(v_min)
    v = math.sqrt(2.0 * float(a_dec) * d)
    return float(min(v_max, max(v_min, v)))


def stage_speed(v_max, d_rem, v_creep=0.15):
    """互換: 段階式の名残。いまは speed_profile と同じ連続プロファイル"""
    return speed_profile(v_max, d_rem, v_min=v_creep)


# ---------------------------------------------------------------- 実機の購読
class LidarReader:
    """rt/utlidar/cloud_livox_mid360 を**ポーリング**で読む(実機)。

    ★ハンドラ方式は使わない。SDKの handler は cyclonedds の受信スレッドが
      Python に入ってGILを握る(docs/オンボード運用.md §3)。点群は10Hzなので
      LowStateほどではないが、500KBの取り込みを制御ループの裏でやるので
      **必要なときだけ**読む(enable)。方策の走行中は読まない。
    """

    # 機体内の lidar_bridge.py(Livox SDK2 直結)が書く共有メモリ。DDS の rt/utlidar/* が
    # 無い FW(1.5.3.8 で実測)でも、こちらから点群が入る。両方を見て新しい方を使う。
    SHM_NPY = "/dev/shm/g1_lidar.npy"
    SHM_JSON = "/dev/shm/g1_lidar.json"

    def __init__(self, sub, hz=20.0, keep=3):
        self._sub = sub                # DDS購読(None なら共有メモリだけ)
        self._dt = 1.0 / hz
        self._keep = keep
        self._lock = threading.Lock()
        self._frames = []              # [(t, pts(N,3) float32)]
        self.frame_id = ""
        self.n_recv = 0
        self.err = 0
        self.enabled = False
        self._stop = False
        self._dtype = None
        self._shm_seq = -1
        self.source = "-"
        self.mount = None              # lidar_bridge が推定した取り付け(表示用)
        # ★DDS の Read() は配信が無いと**永久にブロック**する(take_one にタイムアウトを
        #   付けると SDK が毎回 "[Reader] take sample timeout" を印字する)。
        #   そこで DDS は専用スレッドでブロックさせておき、/dev/shm の読み取りは別の
        #   スレッドで回す(2026-09-04 実機で「共有メモリを読む番が来ない」不具合の修正)
        self._th = threading.Thread(target=self._loop, daemon=True,
                                    name="lidar_poll")
        self._th.start()
        if self._sub is not None:
            threading.Thread(target=self._dds_loop, daemon=True,
                             name="lidar_dds").start()

    def _dds_loop(self):
        while not self._stop:
            if not self.enabled:
                time.sleep(0.1)
                continue
            try:
                msg = self._sub.Read()             # 配信が無ければここで待ち続ける(無害)
                if msg is not None:
                    self._ingest(msg)
                    self.source = "dds"
            except Exception:                      # noqa: BLE001
                self.err += 1
                time.sleep(0.05)

    def _poll_shm(self):
        """lidar_bridge.py の出力(/dev/shm)を読む。新しい連番のときだけ取り込む"""
        try:
            with open(self.SHM_JSON) as f:
                meta = json.load(f)
        except Exception:                          # noqa: BLE001
            return
        seq = int(meta.get("seq", -1))
        if seq == self._shm_seq:
            return
        if time.time() - float(meta.get("t", 0)) > 3.0:
            return                                 # 古い(ブリッジが止まっている)
        try:
            P = np.load(self.SHM_NPY)
        except Exception:                          # noqa: BLE001
            return
        self._shm_seq = seq
        pts = np.ascontiguousarray(P[:, :3], dtype=np.float32)
        self.frame_id = str(meta.get("frame_id", "livox_frame"))
        self.mount = meta.get("mount")
        self.source = "shm"
        with self._lock:
            self._frames.append((float(meta.get("t", time.time())), pts))
            self._frames = self._frames[-self._keep:]
            self.n_recv += 1

    def enable(self, on):
        self.enabled = bool(on)

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            if not self.enabled:
                time.sleep(0.1)
                continue
            try:
                self._poll_shm()
            except Exception:                      # noqa: BLE001
                self.err += 1
            time.sleep(self._dt)

    def _ingest(self, msg):
        n = int(msg.width) * int(msg.height)
        if n <= 0:
            return
        if self._dtype is None:
            self._dtype = (dtype_from_fields(msg.fields, msg.point_step)
                           or PT_DTYPE_DEFAULT)
        buf = bytes(msg.data)
        cnt = min(n, len(buf) // self._dtype.itemsize)
        if cnt <= 0:
            return
        p = np.frombuffer(buf, dtype=self._dtype, count=cnt)
        pts = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float32)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        try:
            self.frame_id = str(msg.header.frame_id)
        except Exception:                          # noqa: BLE001
            pass
        with self._lock:
            self._frames.append((time.time(), pts))
            self._frames = self._frames[-self._keep:]
            self.n_recv += 1

    def latest(self):
        """(最新の受信時刻, 直近K枚を連結した点群(N,3), frame_id) / 未受信は None"""
        with self._lock:
            if not self._frames:
                return None
            t = self._frames[-1][0]
            pts = (np.concatenate([f[1] for f in self._frames])
                   if len(self._frames) > 1 else self._frames[-1][1])
        return t, pts, self.frame_id


class OdomReader:
    """rt/odommodestate(unitree_go SportModeState_)をポーリングで読む。
    position[0:2] が世界(odom)座標の位置、imu_state.rpy[2] がヨー。
    約500Hzで来るので、ハンドラは付けず20Hzで最新だけ拾う。"""

    def __init__(self, sub, hz=20.0):
        self._sub = sub
        self._dt = 1.0 / hz
        self._lock = threading.Lock()
        self._v = None
        self.n_recv = 0
        self.err = 0
        self.enabled = False
        self._stop = False
        threading.Thread(target=self._loop, daemon=True,
                         name="odom_poll").start()

    def enable(self, on):
        self.enabled = bool(on)

    def stop(self):
        self._stop = True

    def _loop(self):
        while not self._stop:
            if not self.enabled:
                time.sleep(0.1)
                continue
            try:
                m = self._sub.Read()
                if m is not None:
                    with self._lock:
                        self._v = (time.time(), float(m.position[0]),
                                   float(m.position[1]),
                                   float(m.imu_state.rpy[2]),
                                   float(m.velocity[0]), float(m.velocity[1]))
                        self.n_recv += 1
            except Exception:                      # noqa: BLE001
                self.err += 1
            time.sleep(self._dt)

    def latest(self):
        """(t, x, y, yaw, vx, vy) / 未受信は None"""
        with self._lock:
            return self._v


# ---------------------------------------------------------------- 障害物
class ObstacleDetector:
    """点群 → 前方の最近点距離 / その物体の横幅と壁判定 / 回り込める側 / 横の空き / 後方(椅子)。

    点群の座標系は header.frame_id で切り替える:
      "odom"/"map"/"world" を含む → 世界座標。オドメトリ位置とIMUヨーで体基準へ
      それ以外(livox_frame/body など)  → センサ座標。x前・y左とみなし
                                         SENSOR_FWD_OFFSET を足す
    どちらでも**床平面を点群からフィット**して「床からの高さ」を出すので、
    z=0 がどこか(センサか床か)に依存しない。センサが傾いていても水平化される。
    """

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.floor = None
        self._rng = np.random.default_rng(0)
        self.floor_prior = None     # ブリッジ推定の床の z(センサ座標、livox_level のとき −高さ)            # (a, b, c, n_inlier): z = a*fwd + b*lat + c
        self.frame = "?"

    @staticmethod
    def is_world_frame(frame_id):
        f = (frame_id or "").lower()
        return any(k in f for k in ("odom", "map", "world"))

    def to_body(self, pts, frame_id, odom_xy, yaw):
        """(N,3) → (fwd, lat, z)。世界座標ならオドメトリ+ヨーで回す。"""
        if self.is_world_frame(frame_id):
            if odom_xy is None or yaw is None:
                return None
            self.frame = "world"
            c, s = math.cos(-yaw), math.sin(-yaw)
            rx = pts[:, 0] - odom_xy[0]
            ry = pts[:, 1] - odom_xy[1]
            fwd = c * rx - s * ry
            lat = s * rx + c * ry
            z = pts[:, 2]
        else:
            self.frame = "sensor"
            yf = math.radians(float(self.cfg.get("yaw_fix_deg", 0.0)))
            if abs(yf) > 1e-6:
                c, s = math.cos(yf), math.sin(yf)
                fwd = c * pts[:, 0] - s * pts[:, 1]
                lat = s * pts[:, 0] + c * pts[:, 1]
            else:
                fwd = pts[:, 0].copy()
                lat = pts[:, 1]
            # ★距離は「つま先から」(2026-09-04 午後)。以前は骨盤基準(センサ+0.10m)で、壁までの距離が
            #   実物(機体の前面から測る)より 25cm ほど大きく出ていた
            fwd = fwd - float(self.cfg.get("front_offset", 0.15))
            z = pts[:, 2]
        return fwd, lat, z

    def fit_floor(self, fwd, lat, z):
        """床平面。低い側半分の点から RANSAC(3点→3cm以内の点数が最大の面)で種を作り、
        反復LSQ(3cm以内を採用)で仕上げる。戻り (a,b,c,n) / 失敗は None。
        ★以前の「低い側35%へLSQ」は、頭の LiDAR が真下の自分の体や周りの物を多く見て
          床点が 35% に満たない実機の点群(2026-09-04)で失敗した。"""
        m = (fwd > -1.0) & (fwd < 4.0) & (np.abs(lat) < 2.0)
        if int(m.sum()) < 300:
            return None
        f, l, zz = fwd[m], lat[m], z[m]
        low = zz < np.percentile(zz, 50)
        if self.floor_prior is not None:
            # ブリッジの取り付け推定(水平化済み)があれば、その ±35cm の点だけを候補にする
            low = low & (np.abs(zz - self.floor_prior) < 0.35)
        idx = np.nonzero(low)[0]
        if len(idx) < 100:
            return None
        Pc = np.stack([f[idx], l[idx], zz[idx]], 1)
        rng = self._rng
        best, best_n = None, 0
        cos_max = math.cos(math.radians(20.0))
        for _ in range(48):
            i3 = rng.choice(len(idx), 3, replace=False)
            p0, p1, p2 = Pc[i3]
            nrm = np.cross(p1 - p0, p2 - p0)
            nn = float(np.linalg.norm(nrm))
            if nn < 1e-6:
                continue
            nrm /= nn
            if nrm[2] < 0:
                nrm = -nrm
            if nrm[2] < cos_max:            # 20度以上傾いた面は床ではない
                continue
            d = (Pc - p0) @ nrm
            n_in = int((np.abs(d) < 0.03).sum())
            if n_in > best_n:
                best_n, best = n_in, (nrm, p0)
        if best is None or best_n < 60:
            return None
        nrm, p0 = best
        # 種の面の3cm以内 → LSQ を2回
        sel_c = np.abs((Pc - p0) @ nrm) < 0.03
        sel = np.zeros(len(zz), dtype=bool)
        sel[idx[sel_c]] = True
        coef = None
        for _ in range(2):
            if int(sel.sum()) < 60:
                return None
            A = np.c_[f[sel], l[sel], np.ones(int(sel.sum()))]
            coef, *_ = np.linalg.lstsq(A, zz[sel], rcond=None)
            resid = zz - (f * coef[0] + l * coef[1] + coef[2])
            sel = np.abs(resid) < 0.03
        n = int(sel.sum())
        if n < 60:
            return None
        if math.hypot(coef[0], coef[1]) > math.tan(math.radians(20.0)):
            return None
        if self.floor_prior is not None and abs(coef[2] - self.floor_prior) > 0.35:
            return None
        return float(coef[0]), float(coef[1]), float(coef[2]), n

    def update(self, pts, frame_id, odom_xy, yaw, side_dir=1):
        """判定を1回。戻り値は UI/ログ用の辞書(dist が None = 前方3m以内に何も無い)。"""
        cfg = self.cfg
        out = dict(ok=False, n=int(len(pts)), dist=None, n_obs=0, ahead=None,
                   side_free=None, side_free_l=None, side_free_r=None,
                   rear_n=0, rear_dist=None, rear_h=None,
                   floor_ok=False, floor_h=None, frame="?", why="")
        if len(pts) < 50:
            out["why"] = "点群が少ない"
            return out
        b = self.to_body(pts, frame_id, odom_xy, yaw)
        if b is None:
            out["why"] = "世界座標の点群にはオドメトリとヨーが要る(未受信)"
            return out
        fwd, lat, z = b
        out["frame"] = self.frame
        fl = self.fit_floor(fwd, lat, z)
        if fl is not None:
            self.floor = fl
        elif self.floor is None and self.floor_prior is not None and self.frame == "sensor":
            # 1度も床が取れていなければブリッジの取り付け推定(水平化済み)を床にする
            self.floor = (0.0, 0.0, float(self.floor_prior), 0)
        if self.floor is not None:
            a, bcoef, c, _ = self.floor
            nrm = np.array([-a, -bcoef, 1.0])
            nrm /= np.linalg.norm(nrm)
            h = (z - (fwd * a + lat * bcoef + c)) * nrm[2]
            # センサが傾いているとき(センサ座標)は前方軸を床面へ射影して水平化
            if nrm[2] < math.cos(math.radians(3.0)):
                fx = np.array([1.0, 0.0, 0.0]) - nrm * nrm[0]
                fx /= np.linalg.norm(fx)
                lax = np.cross(nrm, fx)
                P = np.stack([fwd, lat, z], 1)
                fwd = P @ fx
                lat = P @ lax
            out["floor_ok"] = True
            out["floor_h"] = round(float(-c * nrm[2]), 3)
        else:
            # 床が取れないときは「高さ」を使わず、極端な上下だけ除く(安全側)
            h = z - np.median(z)
            out["why"] = "床平面が取れない(高さ判定なし)"
        hm = (h > cfg["h_min"]) & (h < cfg["h_max"])
        # ★自分の体を除く。頭の LiDAR は真下の胸・肩・腕を見る(実機 2026-09-04: 前方0.11mに
        #   「幅1.57mの壁」= 自分の腕)。前方 self_fwd・左右 self_lat の箱の中は無視する
        own = (fwd > -0.8) & (fwd < cfg.get("self_fwd", 0.4)) & (np.abs(lat) < cfg.get("self_lat", 0.5))
        hm = hm & ~own
        hw = cfg["half_w"]
        m = hm & (fwd > 0.10) & (fwd < 3.0) & (np.abs(lat) < hw)
        n_obs = int(m.sum())
        out["n_obs"] = n_obs
        if n_obs >= 12:
            d0 = float(np.percentile(fwd[m], 5))
            out["dist"] = round(d0, 3)
            # --- 前方の物体の横幅と、回り込める側(壁か障害物か)
            mc = hm & (fwd > d0 - 0.15) & (fwd < d0 + 0.7) & (np.abs(lat) < 1.6)
            if int(mc.sum()) >= 12:
                lo = float(np.percentile(lat[mc], 3))
                hi = float(np.percentile(lat[mc], 97))
                depth = min(0.5, max(0.0, float(np.percentile(fwd[mc], 95)) - d0))
                wall = ((hi - lo) > cfg["wall_width"]) or (lo < -1.0 and hi > 1.0)
                mgn = cfg["detour_margin"]

                def free(e_lo, e_hi):
                    mb = (hm & (fwd > d0 - 0.4) & (fwd < d0 + depth + 0.5)
                          & (lat > e_lo) & (lat < e_hi))
                    return int(mb.sum()) < 6
                bh = float(cfg.get("body_half", 0.25))       # 体の半幅(ギリギリを通る)
                eL = hi + bh + mgn
                eR = lo - bh - mgn
                free_l = eL if (not wall and eL <= cfg["detour_max"]
                                and free(hi + 0.03, eL + bh)) else None
                free_r = eR if (not wall and -eR <= cfg["detour_max"]
                                and free(eR - bh, lo - 0.03)) else None
                out["ahead"] = dict(lat_lo=round(lo, 3), lat_hi=round(hi, 3),
                                    depth=round(depth, 3), wall=bool(wall),
                                    free_l=(None if free_l is None else round(free_l, 3)),
                                    free_r=(None if free_r is None else round(free_r, 3)))
        # --- 横の空き(左右とも)
        for s, key in ((1.0, "side_free_l"), (-1.0, "side_free_r")):
            ms = hm & (fwd > -0.35) & (fwd < 0.45) & (s * lat > 0.12) & (s * lat < 1.5)
            if int(ms.sum()) >= 12:
                out[key] = round(float(np.percentile(s * lat[ms], 5)), 3)
        out["side_free"] = out["side_free_l"] if side_dir >= 0 else out["side_free_r"]
        # --- 後方(椅子側): 座面の高さ帯(床から0.35〜0.75m)にある点。着座前の位置確認の
        #     参考値。★頭のLiDARは後ろ下が見えない可能性が高い(未検出=無いとは言えない)
        mr = (fwd < -0.05) & (fwd > -0.8) & (np.abs(lat) < 0.30) & (h > 0.35) & (h < 0.75)
        out["rear_n"] = int(mr.sum())
        if out["rear_n"] >= 12:
            out["rear_dist"] = round(float(-np.percentile(fwd[mr], 95)), 3)
            out["rear_h"] = round(float(np.median(h[mr])), 3)
        # --- 正面の壁: 前方セクタの点へ水平面内の直線を RANSAC で当てる(2026-09-04 午後)。
        #     コリドーの最近点は「手前の物」も拾うので、壁そのものの距離は別に出す
        out["wall_dist"], out["wall_ang"], out["wall_len"] = self.fit_wall(fwd, lat, h, hm)
        # --- 4 方向の最近距離(前/後/左/右 ±20 度、h 0.2〜1.6、自分の体の外)。前後の向きの確認用
        rr = np.hypot(fwd + float(cfg.get("front_offset", 0.15)), lat)
        azd = np.degrees(np.arctan2(lat, fwd + float(cfg.get("front_offset", 0.15))))
        hm2 = (h > 0.2) & (h < 1.6) & (rr > 0.45)
        dirs = {}
        for key, a0 in (("front", 0.0), ("left", 90.0), ("back", 180.0), ("right", -90.0)):
            da = (azd - a0 + 180.0) % 360.0 - 180.0
            mm = hm2 & (np.abs(da) < 20.0)
            dirs[key] = (round(float(np.percentile(rr[mm], 5)), 2) if int(mm.sum()) >= 10 else None)
        out["dirs"] = dirs
        out["ok"] = True
        return out

    def fit_wall(self, fwd, lat, h, hm):
        """前方セクタ(|lat|<1.2m, 0.2<fwd<6m, h 0.25〜1.6)の点に、進行方向にほぼ垂直な直線を当てる。
        戻り値 (壁までの距離[m] つま先基準・進行線との交点, 壁の法線と進行方向の角度[deg], 壁の見えている幅[m])
        取れなければ (None, None, None)"""
        m = hm & (h > 0.25) & (h < 1.6) & (np.abs(lat) < 1.2) & (fwd > 0.2) & (fwd < 6.0)
        n = int(m.sum())
        if n < 40:
            return None, None, None
        X = np.stack([fwd[m], lat[m]], 1)
        rng = self._rng
        cands = []                                     # (内点数, 法線, 通る点, 進行線との交点x)
        cos35 = math.cos(math.radians(35.0))
        for _ in range(80):
            i, j = rng.choice(n, 2, replace=False)
            d = X[j] - X[i]
            L = float(np.hypot(d[0], d[1]))
            if L < 0.15:
                continue
            nrm = np.array([-d[1], d[0]]) / L         # 直線の法線
            if abs(nrm[0]) < cos35:                    # 進行方向にほぼ垂直な面だけ(法線が前方向き)
                continue
            r = (X - X[i]) @ nrm
            cnt = int((np.abs(r) < 0.04).sum())
            if cnt >= 30:
                # 進行線(lat=0)との交点の fwd: nrm·(p - X[i]) = 0 で p=(x,0) → x = X[i]·nrm / nrm[0]
                xh = float((X[i] @ nrm) / nrm[0]) if abs(nrm[0]) > 1e-6 else 1e9
                cands.append((cnt, nrm, X[i], xh))
        if not cands:
            return None, None, None
        # ★内点が最大の 70% 以上ある候補のうち、いちばん手前の面を採る(壁の奥の面や箱の天面ではなく、
        #   機体が実際にぶつかる面)。同じ面の候補は交点で寄せる
        top = max(c[0] for c in cands)
        good = [c for c in cands if c[0] >= 0.7 * top]
        best = min(good, key=lambda c: c[3])
        _cnt, nrm, p0, _xh = best
        inl = X[np.abs((X - p0) @ nrm) < 0.04]
        cen = inl.mean(axis=0)
        u, s, vt = np.linalg.svd(inl - cen, full_matrices=False)
        d = vt[0]                                       # 直線の向き
        nrm = np.array([-d[1], d[0]])
        if nrm[0] < 0:
            nrm = -nrm
        ext = float((inl @ d).max() - (inl @ d).min())  # 見えている幅
        if ext < 0.6:
            return None, None, None
        # 進行線(lat=0)との交点: cen + t*d で lat=0 → t = -cen[1]/d[1]
        if abs(d[1]) < 1e-6:
            return None, None, None
        t = -cen[1] / d[1]
        x_hit = float(cen[0] + t * d[0])
        ang = math.degrees(math.atan2(nrm[1], nrm[0]))
        if x_hit < 0.05:
            return None, None, None
        return round(x_hit, 3), round(ang, 1), round(ext, 2)


# ---------------------------------------------------------------- 速度送信
class VelSender:
    """10Hzで速度を送る唯一のスレッド。指令は CMD_HOLD_S 秒で失効しゼロを送る。"""

    def __init__(self, send_fn, log=print):
        self._send = send_fn         # send_fn(vx, vy, om, duration) -> bool
        self.duration = 2.0          # 指令の有効時間[s](cmd_dur)。10Hz で上書きし続ける
        self._log = log
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._t = 0.0
        self._src = ""
        self._dry = False
        self._need_stop = 0
        self.last_sent = (0.0, 0.0, 0.0)
        self.last_sent_t = 0.0
        self.n_sent = 0
        self.n_fail = 0
        self._stop = False
        threading.Thread(target=self._loop, daemon=True,
                         name="vel_sender").start()

    def set(self, vx, vy, om, src, dry=False):
        with self._lock:
            self._cmd = (float(vx), float(vy), float(om))
            self._t = time.time()
            self._src = src
            self._dry = bool(dry)

    def stop(self, src="stop"):
        """即座にゼロを送る(3回)。"""
        with self._lock:
            self._cmd = (0.0, 0.0, 0.0)
            self._t = 0.0
            self._src = src
            self._need_stop = 3

    def close(self):
        self._stop = True

    def _loop(self):
        dt = 1.0 / SEND_HZ
        while not self._stop:
            t0 = time.time()
            with self._lock:
                cmd, tset, dry, need = self._cmd, self._t, self._dry, self._need_stop
            fresh = (time.time() - tset) < CMD_HOLD_S
            moving = fresh and any(abs(v) > 1e-6 for v in cmd)
            try:
                if need > 0:
                    if not dry:
                        ok = self._send(0.0, 0.0, 0.0, self.duration)
                        self.n_sent += 1
                        if not ok:
                            self.n_fail += 1
                    self.last_sent = (0.0, 0.0, 0.0)
                    self.last_sent_t = time.time()
                    with self._lock:
                        self._need_stop = max(0, self._need_stop - 1)
                elif moving:
                    if not dry:
                        ok = self._send(cmd[0], cmd[1], cmd[2], self.duration)
                        self.n_sent += 1
                        if not ok:
                            self.n_fail += 1
                    self.last_sent = cmd
                    self.last_sent_t = time.time()
                elif any(abs(v) > 1e-6 for v in self.last_sent):
                    # 動いていた指令が失効した = デッドマン。ゼロを送って止める
                    with self._lock:
                        self._need_stop = 3
                        self._cmd = (0.0, 0.0, 0.0)
            except Exception as e:                 # noqa: BLE001
                self.n_fail += 1
                self._log(f"★速度送信で例外: {e}")
            time.sleep(max(0.0, dt - (time.time() - t0)))


# ---------------------------------------------------------------- 自動歩行
# ---------------------------------------------------------------- LiDAR ヨーの自動較正
CAL_CELL = 0.05                    # 占有格子のセル[m]
CAL_HALF = 4.0                     # 格子の半径[m]
CAL_SHIFT = 0.6                    # 探す平行移動の範囲[m]


def occupancy_grid(fwd, lat, mask):
    """体座標の点を 5cm 格子の占有(bool)にする。static な景色の比較用"""
    n = int(round(2 * CAL_HALF / CAL_CELL))
    G = np.zeros((n, n), dtype=bool)
    f = fwd[mask]
    l = lat[mask]
    i = np.floor((f + CAL_HALF) / CAL_CELL).astype(int)
    j = np.floor((l + CAL_HALF) / CAL_CELL).astype(int)
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    G[i[ok], j[ok]] = True
    return G


def scene_shift(G0, G1):
    """G0(前)から G1(今)への景色の平行移動 (dx, dy)[m] を総当たりの相関で求める。
    戻り値 (dx, dy, 最良スコア, スコアの中央値)。景色は機体が動いた分だけ逆向きに流れる"""
    n = G0.shape[0]
    k = int(round(CAL_SHIFT / CAL_CELL))
    best, bi, bj = -1, 0, 0
    scores = []
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            a0 = G0[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)]
            a1 = G1[max(0, di):n - max(0, -di), max(0, dj):n - max(0, -dj)]
            sc = int(np.count_nonzero(a0 & a1))
            scores.append(sc)
            if sc > best:
                best, bi, bj = sc, di, dj
    med = float(np.median(scores)) if scores else 0.0
    return bi * CAL_CELL, bj * CAL_CELL, best, med


def yaw_error_from_motion(shift, db):
    """景色の平行移動 shift(検出器の座標)と、オドメトリの機体座標での移動 db から、
    検出器の前方軸が機体の前方に対して何度ずれているか[deg]を返す(検出器座標で測った機体前方の角)。
    yaw_fix はこの値を**引く**と合う。整合しなければ None"""
    sx, sy = shift
    dbx, dby = db
    ns = math.hypot(sx, sy)
    nd = math.hypot(dbx, dby)
    if nd < 0.15 or ns < 0.6 * nd or ns > 1.4 * nd:
        return None
    e = math.degrees(math.atan2(-sy, -sx) - math.atan2(dby, dbx))
    return (e + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- 速度の較正(指令 → 実速度)
CALIB_PATH = os.environ.get("G1_WALK_CALIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_calib.json")


class VelCalib:
    """内蔵歩行の「指令 → 実速度」の非線形(実速度 ≈ k·(指令 − 不感帯))を持ち、望む実速度から指令を作る。
    歩行中に (指令, オドメトリの実速度) を 1 秒窓で集めて k を更新し、walk_calib.json に残す(次回も使う)。"""

    def __init__(self, p):
        self.k = {"x": float(p.get("k_vx", 0.45)), "y": float(p.get("k_vy", 0.35))}
        self.dead = float(p.get("v_dead", 0.10))
        self.cmd_max = float(p.get("cmd_max", 0.9))
        self.cmd_min = float(p.get("cmd_min_walk", 0.30))
        self.learn = bool(p.get("calib_learn", False))
        self.n_upd = {"x": 0, "y": 0}
        self._win = {"x": [], "y": []}
        if self.learn:                              # 学習 OFF のときは固定値だけを使う(ファイルは読まない)
            try:
                with open(CALIB_PATH) as f:
                    d = json.load(f)
                for ax in ("x", "y"):
                    if 0.15 <= float(d.get("k_" + ax, 0)) <= 1.5:
                        self.k[ax] = float(d["k_" + ax])
            except Exception:                      # noqa: BLE001
                pass

    def to_cmd(self, v_des, axis="x"):
        """望む実速度[m/s] → 指令[m/s]。|v_des| が 2cm/s 未満なら 0(止める)。
        動かすときは cmd_min 以上を必ず出す(内蔵歩行が歩き出さない小さな指令を出さない)"""
        if abs(v_des) < 0.02:
            return 0.0
        c = self.dead + abs(v_des) / max(0.15, self.k[axis])
        c = max(self.cmd_min, min(self.cmd_max, c))
        return math.copysign(c, v_des)

    def per_burst(self, cmd, t_on, axis="x"):
        """指令 cmd を t_on 秒出したときの見込み移動量[m](歩き出しの遅れ 0.25 秒を引く)"""
        return max(0.0, self.k[axis] * max(0.0, abs(cmd) - self.dead) * max(0.0, t_on - 0.25))

    def observe(self, axis, cmd, x, t):
        """(指令, 位置, 時刻) を積み、|指令| が一定で 1 秒続いた窓から k を更新する(学習 ON のときだけ)"""
        if not self.learn:
            return
        w = self._win[axis]
        w.append((t, cmd, x))
        w[:] = [e for e in w if t - e[0] <= 1.2]
        if len(w) < 6 or t - w[0][0] < 1.0:
            return
        cmds = [abs(e[1]) for e in w]
        if min(cmds) < self.dead + 0.15 or max(cmds) - min(cmds) > 0.06:
            return
        dx = abs(w[-1][2] - w[0][2]) / (w[-1][0] - w[0][0])
        c = sum(cmds) / len(cmds)
        k_obs = dx / max(0.05, c - self.dead)
        if 0.1 <= k_obs <= 1.6:
            a = 0.3 if self.n_upd[axis] < 5 else 0.15
            self.k[axis] = min(1.5, max(0.15, (1 - a) * self.k[axis] + a * k_obs))
            self.n_upd[axis] += 1
        w[:] = w[-3:]

    def save(self):
        if not self.learn:
            return
        try:
            with open(CALIB_PATH, "w") as f:
                json.dump({"k_x": round(self.k["x"], 3), "k_y": round(self.k["y"], 3),
                           "n_x": self.n_upd["x"], "n_y": self.n_upd["y"], "t": time.time()}, f)
        except Exception:                          # noqa: BLE001
            pass


class _Abort(Exception):
    pass


class AutoWalk:
    """前進(壁の手前で自然に停止・障害物は回り込み) / 横歩き(足踏みパルス)。10Hzのスレッド。

    io は WalkController が渡す関数の束:
      obstacle()  → ObstacleDetector.update の結果(最新)と、その時刻
      odom()      → (t, x, y, yaw_odom, vx, vy) / None
      yaw()       → IMUヨー[rad](LowState)
      tilt_deg()  → 体幹の傾き[度]
      hb_ok()     → UIハートビートが生きているか
      vel(vx,vy,om) / stop(why)
      log(str)
    """

    def __init__(self, params, io, log_path=None):
        self.p = dict(params)
        self.io = io
        self.phase = "INIT"
        self.msg = ""
        self.done = False
        self.result = ""
        self.traveled = 0.0          # 経路に沿った前進[m]
        self.traveled_base = 0.0     # ステップ機関が後退前の進みを引き継ぐための基準
        self.side_traveled = 0.0     # 横歩きした距離[m](SIDE中)
        self.offset = 0.0            # 経路線からの左へのずれ[m]
        self.detours = 0
        self.v = 0.0
        self.t_start = time.time()
        # 横移動だけ(mode=side)は LiDAR が無くてもオドメトリで動かせる(横の空きは見ない)。
        # 前進は LiDAR が必須(壁・障害物を見るのが目的)
        mode0 = self.p.get("mode", "both")
        self.need_lidar = (mode0 not in ("side", "back")
                           and not (mode0 == "step" and self.p.get("step_dir", "left") != "fwd"))
        self.cal = VelCalib(self.p)  # 指令 → 実速度 の較正(walk_calib.json)
        self.step_last = None        # 直近の1歩の移動量[m](指令方向が正)
        self.step_est = float(self.p.get("step_est", 0.06))
        self.steps = 0
        self._stop_req = None
        self._log_path = pathlib.Path(log_path) if log_path else None
        self._rows = []
        self._th = threading.Thread(target=self._run, daemon=True, name="autowalk")
        # 経路線(開始時に決める)
        self._x0 = self._y0 = 0.0
        self._fx = self._fy = 0.0
        self._lx = self._ly = 0.0
        self._yaw_ref = 0.0

    def start(self):
        self._th.start()

    def request_stop(self, why):
        self._stop_req = why

    # ---- 本体
    def _run(self):
        try:
            self._main()
        except _Abort as e:
            self.result = str(e)
            self.io["log"](f"★自動歩行 {self.result}")
        except Exception as e:                     # noqa: BLE001
            self.result = f"例外: {type(e).__name__}: {e}"
            self.io["log"](f"★自動歩行 {self.result}")
        finally:
            self.io["stop"]("自動歩行 終了")
            self.phase = "DONE" if self.result.startswith("完了") else "ABORT"
            try:
                self.cal.save()
            except Exception:                      # noqa: BLE001
                pass
            self.done = True
            self._flush()

    # ---- 経路線とセンサ
    def _set_path(self, od):
        self._x0, self._y0 = od[1], od[2]
        yaw_od = od[3]
        self._fx, self._fy = math.cos(yaw_od), math.sin(yaw_od)
        self._lx, self._ly = -math.sin(yaw_od), math.cos(yaw_od)
        self._yaw_ref = self.io["yaw"]()
        self._path_yaw_ref = self._yaw_ref

    def _pose(self, od):
        dx, dy = od[1] - self._x0, od[2] - self._y0
        return dx * self._fx + dy * self._fy, dx * self._lx + dy * self._ly

    def _sense(self):
        """(od, obs) を読み、中止条件を検査してから返す"""
        od = self.io["odom"]()
        obs, obs_t = self.io["obstacle"]()
        now = time.time()
        if self._stop_req:
            raise _Abort(f"中止({self.phase}): {self._stop_req}")
        if not self.io["hb_ok"]():
            raise _Abort(f"中止({self.phase}): UIハートビート途絶(操作者が見ていない)")
        tilt = self.io["tilt_deg"]()
        if tilt > TILT_ABORT_DEG:
            raise _Abort(f"中止({self.phase}): 体幹の傾き{tilt:.0f}度")
        if od is None or now - od[0] > 0.8:
            raise _Abort(f"中止({self.phase}): オドメトリ途絶")
        if self.need_lidar and (obs_t is None or now - obs_t > 1.5):
            raise _Abort(f"中止({self.phase}): LiDAR途絶")
        if not self.need_lidar and (obs_t is None or now - obs_t > 1.5):
            obs = dict(ok=False, why="LiDAR無し")
        return od, obs

    def _om(self):
        """向きの保持。内蔵歩行は小さな回転指令に応じないので、3 度を超えるずれには最低 0.15 rad/s を出し、
        3 度以内は 0(小刻みに回し続けない)"""
        yaw = self.io["yaw"]()
        err = _wrap(yaw - self._yaw_ref)
        if abs(err) <= math.radians(3.0):
            return 0.0
        om = float(np.clip(-YAW_KP * err, -YAW_OM_MAX, YAW_OM_MAX))
        return math.copysign(max(0.15, abs(om)), om)

    def _rec(self, **kw):
        kw["t"] = round(time.time() - self.t_start, 3)
        kw["phase"] = self.phase
        self._rows.append(kw)
        if len(self._rows) >= 50:
            self._flush()

    def _flush(self):
        if self._log_path is None or not self._rows:
            self._rows = []
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                for r in self._rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception:                          # noqa: BLE001
            pass
        self._rows = []

    def _hold(self, secs, msg):
        """速度ゼロで secs 秒待つ(監視は続ける)"""
        t0 = time.time()
        while time.time() - t0 < secs:
            time.sleep(0.1)
            self._sense()
            self.io["vel"](0.0, 0.0, 0.0)
            self.msg = msg

    # ---- 小刻みステップ(2026-09-04 午後。足踏みパルスを置き換え)
    def _step_axis(self, axis, target, label, single=False):
        """axis='e'(横。+左) / 's'(経路に沿って。+前)。target はその軸の目標値[m]。
        1歩 = step_v [m/s] を step_on 秒 → step_off 秒止めて着地を待ち、オドメトリで測る。
        残りが1歩の推定より小さいときは指令時間を縮める(最短 step_min_on)。
        1歩の推定は実測で更新する。3歩で1cmも進まなければ中止(内蔵歩行が速度指令に
        応じていない = 2026-09-04 11:47 の実測)。single=True は1歩だけ出して戻る。
        戻り値: 到達したか(横に障害物 / 前が停止距離以内なら False)"""
        p = self.p
        tol = p["side_tol"]
        est = max(0.01, float(self.step_est))

        def cur():
            od, obs = self._sense()
            s, e = self._pose(od)
            self.offset = e
            return (s if axis == "s" else e), obs, s

        x, obs, s_now = cur()
        s_ref = s_now
        t0 = time.time()
        t_limit = 15.0 + abs(target - x) / 0.03 * (p["step_on"] + p["step_off"])
        n = 0
        last_sgn = 0
        reversals = 0
        self._step_recent = []
        net = 0.0                              # 指令方向へ進んだ量の合計(オドメトリ雑音に強い判定用)
        best = 0.0                             # 1歩で最も進んだ量
        while True:
            x, obs, s_now = cur()
            if axis == "s":
                self.traveled = s_now - s_ref + self.traveled_base
            rem = target - x
            if abs(rem) <= tol and not single:
                self._hold(0.5, f"{label}: 到達(残り{rem * 100:+.0f}cm)")
                return True
            if n >= int(p.get("step_max", 20)):
                raise _Abort(f"中止({self.phase}): {n}歩で到達せず(残り{rem * 100:+.0f}cm)")
            if time.time() - t0 > t_limit:
                raise _Abort(f"中止({self.phase}): 時間切れ({t_limit:.0f}秒、残り{rem * 100:+.0f}cm)")
            sgn = 1.0 if rem > 0 else -1.0
            if n > 0 and sgn != last_sgn:
                reversals += 1
                if reversals > 1 or abs(rem) < 0.6 * est:
                    # 半歩未満の行き過ぎは追わない(往復して椅子や壁に近づくのを避ける)
                    self._hold(0.5, f"{label}: 半歩未満の残り{rem * 100:+.0f}cm は追いません")
                    return True
            if axis == "e":
                sf = obs.get("side_free_l" if sgn > 0 else "side_free_r") if obs.get("ok") else None
                if sf is not None and sf < p["side_clear"]:
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 横{sf:.2f}mに障害物 — 止めます"
                    return False
            elif sgn > 0:
                d_front = obs.get("dist") if obs.get("ok") else None
                if d_front is not None and d_front < p["stop_dist"] + 0.05:
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 前方{d_front:.2f}m(停止距離{p['stop_dist']:.2f}m) — 前へは出しません"
                    return False
            frac = 1.0 if single else min(1.0, abs(rem) / est)
            t_on = max(float(p["step_min_on"]), float(p["step_on"]) * frac)
            # step_v は「望む実速度」。較正で指令に直す(実機は指令の 4 割ほどしか出ない)
            v = sgn * abs(self.cal.to_cmd(float(p["step_v"]), "y" if axis == "e" else "x"))
            n += 1
            last_sgn = sgn
            xa = x
            t1 = time.time()
            while time.time() - t1 < t_on:
                time.sleep(0.1)
                od, obs = self._sense()
                s, e = self._pose(od)
                if axis == "e":
                    self.io["vel"](0.0, v, self._om())
                else:
                    vy = float(np.clip(-LAT_KP * e, -LAT_VY_MAX, LAT_VY_MAX))
                    self.io["vel"](v, vy, self._om())
                self.offset = e
                self.msg = (f"{label}: {n}歩目 指令{v:+.2f}m/s×{t_on:.1f}s"
                            f" 残り{(target - (s if axis == 's' else e)) * 100:+.0f}cm")
                self._rec(step=n, v=round(v, 3), t_on=round(t_on, 2),
                          x=round(float(s if axis == "s" else e), 3))
            self._hold(float(p["step_off"]), f"{label}: {n}歩目の着地を待つ")
            x, obs, s_now = cur()
            d = (x - xa) * sgn                    # この歩で進んだ量(指令方向が正)
            self.step_last = d
            self.steps = n
            net += d
            best = max(best, d)
            recent = getattr(self, "_step_recent", [])
            recent = (recent + [d])[-3:]
            self._step_recent = recent
            if d > 0.005:
                est = d if n == 1 else 0.6 * est + 0.4 * d
                self.step_est = est
            self.io["log"](f"{label}: {n}歩目 {d * 100:+.1f}cm(指令{v:+.2f}m/s×{t_on:.1f}s)"
                           f" 残り{(target - x) * 100:+.1f}cm 1歩の推定{est * 100:.1f}cm")
            if n >= 3 and sum(recent) < 0.015:      # 直近 3 歩の合計が 1.5cm 未満(1 歩の雑音 ±0.5cm に強い)
                self.io["vel"](0.0, 0.0, 0.0)
                raise _Abort(f"中止({self.phase}): 3歩で進み{net * 100:.1f}cm — 歩行モードが"
                             "速度指令に応じていない(十字キーで歩けるか確認。docs 自動歩行 §6b-3)")
            if single:
                return True

    NUDGE_MAX = 0.10                # これ未満の移動は小刻みステップ(1歩)、以上は普通の歩行

    def _move_axis(self, axis, target, label):
        """普通の歩行で目標まで動く(2026-09-04 午後。小刻みステップは 10cm 未満の微調整だけ)。
        axis='e'(横。+左) / 's'(経路に沿って。+前)。速度は残り距離の連続プロファイル
        (sqrt(2·a·d)、最低 v_creep、上限 v_side)。行き過ぎたら逆向きにゆっくり戻す(2 回まで)。
        戻り値: 到達したか(横に障害物 / 前が停止距離以内なら False)"""
        p = self.p
        tol = max(0.03, float(p["side_tol"]))
        v_max = float(p["v_side"])
        v_min = float(p["v_creep"])
        a_dec = min(0.2, float(p.get("a_dec", 0.25)))
        lag = float(p.get("cmd_lag", 0.3))
        od, obs = self._sense()
        s0, e0 = self._pose(od)
        x0 = s0 if axis == "s" else e0
        t_limit = 15.0 + abs(target - x0) / max(v_min, 0.05) * 2.0
        t0 = time.time()
        t_prev = t0
        v = 0.0
        last_sgn = 0
        reversals = 0
        hist = []                                   # (時刻, 位置) 動かない検出用
        while True:
            time.sleep(0.1)
            now = time.time()
            dt = max(0.02, min(0.3, now - t_prev))
            t_prev = now
            od, obs = self._sense()
            s, e = self._pose(od)
            self.offset = e
            if axis == "s":
                self.traveled = s
            x = s if axis == "s" else e
            rem = target - x
            hist.append((now, x))
            # 応答遅れの分だけ手前で止める(止めた後もその間は進む)
            if abs(rem) <= tol + abs(v) * lag:
                self._hold(0.6, f"{label}: 停止(残り{rem * 100:+.0f}cm)")
                od, obs = self._sense()
                s, e = self._pose(od)
                self.offset = e
                x = s if axis == "s" else e
                rem = target - x
                if abs(rem) <= tol * 1.5:
                    self.msg = f"{label}: 到達(残り{rem * 100:+.0f}cm)"
                    return True
                v = 0.0
                hist = []
                continue
            # 指令を出しているのに 4 秒で 2cm も動かない → 内蔵歩行が応じていない
            if abs(v) >= v_min - 1e-6 and len(hist) >= 2 and now - hist[0][0] >= 4.0:
                x_old = [xx for tt, xx in hist if tt <= now - 4.0]
                if x_old and abs(x - x_old[-1]) < 0.02:
                    self.io["vel"](0.0, 0.0, 0.0)
                    raise _Abort(f"中止({self.phase}): 4秒送って進み{abs(x - x_old[-1]) * 100:.1f}cm — 歩行モードが"
                                 "速度指令に応じていない(十字キーで歩けるか確認。docs 自動歩行 §6b-5)")
            if now - t0 > t_limit:
                raise _Abort(f"中止({self.phase}): 時間切れ({t_limit:.0f}秒、残り{rem * 100:+.0f}cm)")
            sgn = 1.0 if rem > 0 else -1.0
            if last_sgn and sgn != last_sgn:
                reversals += 1
                if reversals > 2:
                    self._hold(0.6, f"{label}: 行き過ぎ {rem * 100:+.0f}cm はこれ以上追いません")
                    return True
                v = 0.0                             # 向きが変わるときは一度止める
            last_sgn = sgn
            if axis == "e":
                sf = obs.get("side_free_l" if sgn > 0 else "side_free_r") if obs.get("ok") else None
                if sf is not None and sf < p["side_clear"]:
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 横{sf:.2f}mに障害物 — 止めます"
                    return False
            elif sgn > 0:
                d_front = obs.get("dist") if obs.get("ok") else None
                if d_front is not None and d_front < p["stop_dist"] + 0.05:
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 前方{d_front:.2f}m(停止距離{p['stop_dist']:.2f}m) — 前へは出しません"
                    return False
            # 残りが final_zone 以内なら「最後の一歩」で寄せて自然に止まる
            if abs(rem) <= float(p.get("final_zone", 0.25)):
                self._final_approach(axis, target, label)
                return True
            v_t = speed_profile(v_max, abs(rem), v_min, a_dec, abs(v), lag, tol)
            if v_t > abs(v):
                v = min(v_t, abs(v) + ACC_UP * dt)
            else:
                v = max(v_t, abs(v) - ACC_DOWN * dt)
            vcmd = sgn * v
            cax = "y" if axis == "e" else "x"
            c = self.cal.to_cmd(vcmd, cax)
            if axis == "e":
                self.io["vel"](0.0, c, self._om())
            else:
                vy = float(np.clip(-LAT_KP * e, -LAT_VY_MAX, LAT_VY_MAX))
                self.io["vel"](c, self.cal.to_cmd(vy, "y"), self._om())
            self.cal.observe(cax, c, x, now)
            self.msg = f"{label}: 望む{vcmd:+.2f}m/s(指令{c:+.2f}) 残り{rem * 100:+.0f}cm"
            self._rec(v=round(vcmd, 3), c=round(c, 3), x=round(float(x), 3), rem=round(float(rem), 3))

    def _final_approach(self, axis, target, label):
        """最後の寄せ(目標まで final_zone 以内): 一定のゆっくりした実速度で歩き続け、残りが 5cm 以下になった
        瞬間に**一度だけ**止める。行き過ぎても戻さない(ピタッと止まる。操作者の指示)。
        ★以前の「短い指令→止めて測る」の繰り返しは、指令の出し止めのたびに機体が後ろへ揺り戻り
          (実機 14:27〜14:28: 1 回で −4〜−16cm)、目的の距離の後で少しずつ後ろへ戻って見えた。
        axis='s'(経路に沿って) / 'e'(横)。target はその軸の値(経路座標)。"""
        cax = "y" if axis == "e" else "x"
        v_slow = 0.08                                    # 望む実速度[m/s](指令は下限 cmd_min_walk=0.30 になる)
        tol = 0.08                                       # 多少のずれは許容し、少し手前で一度だけ止める(操作者の指示)
        od, obs = self._sense()
        s, e = self._pose(od)
        x0 = s if axis == "s" else e
        sgn = 1.0 if target >= x0 else -1.0
        t0 = time.time()
        t_limit = 6.0 + abs(target - x0) / 0.05
        while True:
            time.sleep(0.1)
            od, obs = self._sense()
            s, e = self._pose(od)
            x = s if axis == "s" else e
            rem = (target - x) * sgn
            self.offset = e
            if axis == "s":
                self.traveled = getattr(self, "traveled_base", 0.0) + s
            if rem <= tol:
                # 急にゼロにせず、0.3 秒だけ弱い指令(0.20)を挟んでからゼロ(急停止の踏み替えを避ける)
                t2 = time.time()
                while time.time() - t2 < 0.3:
                    time.sleep(0.1)
                    self._sense()
                    self.io["vel"](sgn * 0.20 if axis == "s" else 0.0, sgn * 0.20 if axis == "e" else 0.0, 0.0)
                self.io["vel"](0.0, 0.0, 0.0)
                self._hold(0.8, f"{label}: 停止(残り{rem * 100:+.0f}cm)")
                self.msg = f"{label}: 到達(残り{rem * 100:+.0f}cm)"
                return True
            if time.time() - t0 > t_limit:
                self.io["vel"](0.0, 0.0, 0.0)
                raise _Abort(f"中止({self.phase}): 最後の寄せの時間切れ({t_limit:.0f}秒、残り{rem * 100:+.0f}cm)")
            c = sgn * abs(self.cal.to_cmd(v_slow, cax))
            # 最後の寄せでは向き・横ずれの補正はしない(止まる直前に別の動きを足すと揺れる)
            if axis == "e":
                self.io["vel"](0.0, c, 0.0)
            else:
                self.io["vel"](c, 0.0, 0.0)
            self.msg = f"{label}: 最後の寄せ 指令{c:+.2f} 残り{rem * 100:+.0f}cm"
            self._rec(final=1, c=round(c, 3), x=round(float(x), 3), rem=round(float(rem), 3))

    def _lateral_to(self, e_target, label):
        """経路線からのずれ e を e_target へ。10cm 以上は普通の歩行、未満は小刻みステップ。
        戻り値: 到達したか(横に障害物なら False)"""
        od, _obs = self._sense()
        _s, e = self._pose(od)
        if abs(e_target - e) <= self.NUDGE_MAX + 1e-6:
            return self._step_axis("e", e_target, label)
        return self._move_axis("e", e_target, label)

    def _back_to(self, dist, label):
        """後退(椅子との距離を詰める)。経路に沿って dist だけ下がる。10cm 以上は普通の歩行、
        未満は小刻みステップ。後ろは LiDAR が見えないので操作者が目で見る前提。"""
        od, _obs = self._sense()
        s0, _e = self._pose(od)
        self.traveled_base = self.traveled
        if dist <= self.NUDGE_MAX + 1e-6:
            return self._step_axis("s", s0 - dist, label)
        return self._move_axis("s", s0 - dist, label)

    def _step_once(self, direction):
        """1歩だけ(かんたん画面の [1歩] ボタン)。戻り値: 結果の文"""
        od, _obs = self._sense()
        s0, e0 = self._pose(od)
        self.traveled_base = self.traveled
        far = 1.0
        if direction == "left":
            self._step_axis("e", e0 + far, "左へ1歩", single=True)
        elif direction == "right":
            self._step_axis("e", e0 - far, "右へ1歩", single=True)
        elif direction == "fwd":
            self._step_axis("s", s0 + far, "前へ1歩", single=True)
        else:
            self._step_axis("s", s0 - far, "後ろへ1歩", single=True)
        d = self.step_last
        return (f"1歩({ {'left': '左', 'right': '右', 'fwd': '前', 'back': '後ろ'}.get(direction, direction)}): "
                f"{(d or 0.0) * 100:+.1f}cm 動きました")

    def _rebase_path(self, od, yaw_ref_imu):
        """経路線を「いまの位置・向き yaw_ref(IMU)」で張り直す。進み(traveled)は引き継ぐ。
        壁に垂直な向きへ寄せ続けると最初の経路線からずれるので、向きが 5 度以上変わったら呼ぶ"""
        s, _e = self._pose(od)
        self.traveled_base = getattr(self, "traveled_base", 0.0) + s
        yaw_od = od[3] + _wrap(yaw_ref_imu - self.io["yaw"]())    # 望む向きをオドメトリ座標で
        self._x0, self._y0 = od[1], od[2]
        self._fx, self._fy = math.cos(yaw_od), math.sin(yaw_od)
        self._lx, self._ly = -math.sin(yaw_od), math.cos(yaw_od)
        self._path_yaw_ref = yaw_ref_imu

    def _track_wall_heading(self, obs, od):
        """壁の面の角度 wall_ang(壁の法線が進行方向から左へ何度か)を毎コマ見て、目標の向き
        _yaw_ref を壁に垂直へ寄せる(ローパス 0.3)。向きが 5 度以上変わったら経路線も張り直す。
        壁が見えないときは前回の _yaw_ref をそのまま保持(IMU)"""
        if not self.p.get("wall_track", True) or not obs.get("ok"):
            return
        ang = obs.get("wall_ang")
        wl = obs.get("wall_len") or 0.0
        wd = obs.get("wall_dist")
        if ang is None or wl < 0.8 * float(self.p["wall_width"]) or wd is None or wd > 5.0:
            return                                  # 壁幅の 8 割未満の面(箱・机)は追わない
        yaw_now = self.io["yaw"]()
        target = _wrap(yaw_now + math.radians(float(ang)))
        self._yaw_ref = _wrap(self._yaw_ref + 0.3 * _wrap(target - self._yaw_ref))
        if abs(_wrap(self._yaw_ref - getattr(self, "_path_yaw_ref", self._yaw_ref))) > math.radians(5.0):
            self._rebase_path(od, self._yaw_ref)

    def _align_to_wall(self):
        """正面の壁が斜めなら、その場で回転して壁に正対する(壁の面フィットの角度 wall_ang を使う)。
        wall_ang = 壁の法線が進行方向から左へ何度ずれているか。正なら左へ回る。
        戻り値: 回った角度[deg](壁が見えない/正対済みなら 0)"""
        p = self.p
        tol = float(p.get("align_tol_deg", 3.0))
        om_max = float(p.get("om_turn", 0.3))
        self._hold(0.4, "正対: 壁の向きを読んでいます")     # 歩行モードに入った直後の新しい点群を待つ
        od, obs = self._sense()
        ang = obs.get("wall_ang") if obs.get("ok") else None
        if ang is None or abs(ang) <= max(tol, float(p.get("align_inplace_deg", 8.0))):
            return 0.0                                  # 小さなずれは歩きながら合わせる(_track_wall_heading)
        self.phase = "ALIGN"
        self.io["log"](f"正面の壁が {ang:+.1f}° 斜めです — その場で回転して正対します(壁 {obs.get('wall_dist')}m)")
        yaw0 = self.io["yaw"]()
        t0 = time.time()
        ok_n = 0
        miss = 0
        while True:
            time.sleep(0.1)
            od, obs = self._sense()
            ang = obs.get("wall_ang") if obs.get("ok") else None
            if time.time() - t0 > 20.0:
                self.io["vel"](0.0, 0.0, 0.0)
                raise _Abort("中止(ALIGN): 正対の時間切れ(20秒)")
            if ang is None:
                miss += 1
                if miss > 10:                       # 1 秒以上壁を見失った
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.io["log"]("正対: 回転中に壁の面を見失いました — そのまま前進します")
                    break
                continue
            miss = 0
            if abs(ang) <= tol:
                ok_n += 1
                if ok_n >= 3:
                    self.io["vel"](0.0, 0.0, 0.0)
                    break
            else:
                ok_n = 0
            # 角度に比例(1.2 rad/s per rad)。小さすぎると足踏みで回らないので最低 0.12 rad/s
            om = math.radians(ang) * 1.2
            om = max(-om_max, min(om_max, om))
            if abs(om) < 0.12:
                om = math.copysign(0.12, om)
            self.io["vel"](0.0, 0.0, om)
            self.msg = f"正対中: 壁の角度 {ang:+.1f}° 回転 {om:+.2f}rad/s"
            self._rec(om=round(om, 3), wall_ang=ang)
        self._hold(0.8, "正対: 静定")
        turned = math.degrees(_wrap(self.io["yaw"]() - yaw0))
        self.io["log"](f"正対しました(回転 {turned:+.1f}°、壁の角度 {ang if ang is None else round(ang, 1)}°)")
        return turned

    def _log_post_stop(self):
        """止めた後の処理。stop_lock=True(既定): 足踏みをやめてロック立位(FSM 4)で静止し、3 秒の位置変化をログ。
        stop_lock=False: アンカー保持(下記)。"""
        if self.p.get("stop_lock", True):
            try:
                od, _obs = self._sense()
                s0, e0 = self._pose(od)
                self.io["vel"](0.0, 0.0, 0.0)
                ok = self.io["lock_stand"]()
                self.io["log"]("足踏みをやめてロック立位(FSM 4)で静止します" + ("" if ok else "(★ロック立位へ入れませんでした)"))
                t0 = time.time()
                while time.time() - t0 < 3.0:
                    time.sleep(0.1)
                    self._sense()
                od, _obs = self._sense()
                s1, e1 = self._pose(od)
                self.io["log"](f"停止後 3 秒の位置変化: 前後 {(s1 - s0) * 100:+.1f}cm 横 {(e1 - e0) * 100:+.1f}cm(ロック立位)")
            except _Abort:
                pass
            except Exception as ex:                # noqa: BLE001
                self.io["log"](f"(停止後の処理で例外: {ex})")
            return
        self._anchor_hold()

    def _anchor_hold(self):
        """アンカー保持: 止めた後 anchor_s 秒、位置を見張る。後ろ(経路の −s)へ 5cm 以上ずれたら前へ寄せ直し、
        横へ 8cm 以上ずれたら横へ寄せ直す(いずれも歩き出す下限の指令、2cm 以内で止める)。前へのずれ(壁側)は
        追わない。終わりに位置の変化をログに出す(「停止後 N 秒の位置変化」)。
        ★実機 14:49〜14:53: 後ろ向きの指令を一度も出していないのに停止直後に 20〜50cm 後退した(内蔵の急停止か
          ハーネスの張力)。ここで見張って戻す"""
        secs = float(self.p.get("anchor_s", 6.0))
        try:
            od, _obs = self._sense()
            s0, e0 = self._pose(od)
            t0 = time.time()
            n_fix = 0
            moving = None                            # ("s"|"e", sgn) 寄せ直し中
            while time.time() - t0 < secs:
                time.sleep(0.1)
                od, obs = self._sense()
                s, e = self._pose(od)
                ds, de = s - s0, e - e0
                if moving is None:
                    if ds < -0.05:
                        moving = ("s", 1.0)
                        n_fix += 1
                        self.io["log"](f"アンカー保持: 後ろへ {-ds * 100:.0f}cm ずれた — 前へ寄せ直します")
                    elif abs(de) > 0.08:
                        moving = ("e", -1.0 if de > 0 else 1.0)
                        n_fix += 1
                        self.io["log"](f"アンカー保持: 横へ {de * 100:+.0f}cm ずれた — 寄せ直します")
                if moving is None:
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"アンカー保持 {time.time() - t0:.0f}/{secs:.0f}s ずれ 前後{ds * 100:+.0f}cm 横{de * 100:+.0f}cm"
                    continue
                ax, sg = moving
                done = (ds > -0.02) if ax == "s" else (abs(de) < 0.02)
                # 壁側へ寄せるときは LiDAR の停止距離を守る
                d_front = obs.get("dist") if obs.get("ok") else None
                if ax == "s" and d_front is not None and d_front < float(self.p["stop_dist"]) - 0.10:
                    done = True
                if done:
                    self.io["vel"](0.0, 0.0, 0.0)
                    moving = None
                    continue
                c = sg * float(self.cal.cmd_min)
                if ax == "s":
                    self.io["vel"](c, 0.0, 0.0)
                else:
                    self.io["vel"](0.0, c, 0.0)
                self.msg = f"アンカー保持: 寄せ直し中 前後{ds * 100:+.0f}cm 横{de * 100:+.0f}cm"
                self._rec(anchor=1, c=round(c, 3), ds=round(ds, 3), de=round(de, 3))
            self.io["vel"](0.0, 0.0, 0.0)
            od, _obs = self._sense()
            s1, e1 = self._pose(od)
            self.io["log"](f"停止後 {secs:.0f} 秒の位置変化: 前後 {(s1 - s0) * 100:+.1f}cm 横 {(e1 - e0) * 100:+.1f}cm"
                           f"(寄せ直し {n_fix} 回)")
        except _Abort:
            self.io["vel"](0.0, 0.0, 0.0)
        except Exception:                          # noqa: BLE001
            self.io["vel"](0.0, 0.0, 0.0)

    def _forward(self):
        """壁の手前(stop_dist)で止まるか、最大前進距離で止まるまで進む。
        - 速度は残り距離の連続プロファイル(speed_profile)
        - 壁の角度を毎コマ測り、向きを壁に垂直へ寄せ続ける(_track_wall_heading)
        - 壁でない障害物は**歩きながら斜めに**空いている側へ寄せて脇を通り(余白 detour_margin)、
          通り過ぎたら歩きながら元の経路線へ戻る(2026-09-04 操作者の指示: ギリギリ・スマートに)
        戻り値: "wall"(壁の手前で停止) / "max"(壁なしで最大距離)"""
        p = self.p
        self.phase = "FORWARD"
        self.v = 0.0
        hit = 0
        veer = None                                 # 回り込みの状態 {e_t, s_end, stage}
        vy_now = 0.0
        t_phase = time.time()
        t_prev = t_phase
        t_limit = p["max_fwd"] / max(p["v_fwd"], 0.05) * 2.5 + 20.0
        v_min = float(p["v_creep"])
        while True:
            time.sleep(0.1)
            now = time.time()
            dt = max(0.02, min(0.3, now - t_prev))
            t_prev = now
            od, obs = self._sense()
            if now - t_phase > t_limit:
                raise _Abort(f"中止(FORWARD): 時間切れ({t_limit:.0f}秒)")
            s, e = self._pose(od)
            self.traveled, self.offset = getattr(self, "traveled_base", 0.0) + s, e
            if not obs.get("ok"):
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                self.msg = f"待機: 障害物の判定不能({obs.get('why')})"
                self._rec(vx=0, dist=None, s=round(s, 3))
                continue
            self._track_wall_heading(obs, od)      # ★壁に垂直な向きを保ち続ける
            dist = obs.get("dist")
            ah = obs.get("ahead")
            # 「壁」として使うのは面の見えている幅が壁幅の 8 割以上の面だけ(箱の前面を壁と誤認しない)
            wall_d = (obs.get("wall_dist") if (obs.get("wall_len") or 0.0) >= 0.8 * float(p["wall_width"])
                      else None)
            # --- 回り込みの開始: 壁でない物体が前に来たら、空いている側へ歩きながら寄せる
            if (veer is None and p.get("avoid", True) and dist is not None and ah is not None
                    and not ah["wall"] and self.detours < MAX_DETOURS
                    and dist <= max(p["stop_dist"] + 0.9, 1.5)):
                cands = [c for c in (ah.get("free_l"), ah.get("free_r")) if c is not None]
                if cands:
                    e_t = min(cands, key=abs)
                    veer = dict(e_t=e + e_t, s_end=s + dist + ah["depth"] + 0.25, stage="out",
                                dir=(1.0 if e_t > 0 else -1.0))
                elif now - getattr(self, "_t_noveer", 0.0) > 2.0:
                    self._t_noveer = now
                    self.io["log"](f"障害物 {dist:.2f}m(横幅{ah['lat_hi'] - ah['lat_lo']:.2f}m)に回り込み先が無い"
                                   f"(左 {ah.get('free_l')} / 右 {ah.get('free_r')}、横 {p['detour_max']:.1f}m 以内に空きなし) — 手前で止まります")
                    self.io["log"](f"障害物 {dist:.2f}m(横幅{ah['lat_hi'] - ah['lat_lo']:.2f}m・奥行{ah['depth']:.2f}m) — "
                                   f"歩きながら{'左' if e_t > 0 else '右'}へ{abs(e_t):.2f}m寄せて脇を通ります")
                    self.phase = "DETOUR_OUT"
            # --- 停止判定: 回り込み中は壁までの距離だけを見る(脇を通る物体では止めない)。
            #     ただし極端に近い物(0.35m 未満)は止める
            if veer is not None:
                d_stop = None if wall_d is None else wall_d - p["stop_dist"]
                if dist is not None and dist < 0.35:
                    d_stop = dist - 0.30
            else:
                d_stop = None if dist is None else dist - p["stop_dist"]
            near = d_stop is not None and d_stop <= 0.03
            hit = hit + 1 if near else 0
            if hit >= 2:
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                what = "壁" if (ah and ah["wall"]) or (veer is not None and wall_d is not None) else "障害物"
                self.io["log"](f"{what} {(wall_d if (veer is not None and wall_d is not None) else dist):.2f}m"
                               f"(停止距離{p['stop_dist']:.2f}m) — 停止します  前進{self.traveled:.2f}m")
                return "wall"
            if self.traveled >= p["max_fwd"]:
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                return "max"
            # --- 前進速度: 連続プロファイル(回り込み中は veer_v 以下)
            v_t = speed_profile(p["v_fwd"], d_stop, v_min, p.get("a_dec", 0.25), self.v, p.get("cmd_lag", 0.3))
            if veer is not None:
                # 脇へ寄せ切るまでは障害物へ近づくほど落とす(0.35m 手前を基準)。ただし最低 v_min は保つ
                v_cap = speed_profile(float(p.get("veer_v", 0.35)), None if dist is None else dist - 0.35,
                                      v_min, p.get("a_dec", 0.25), self.v, p.get("cmd_lag", 0.3))
                v_t = min(v_t, max(v_cap, v_min))
                v_t = max(v_t, v_min) if d_stop is None or d_stop > 0.03 else v_t
            if v_t > self.v:
                self.v = min(v_t, self.v + ACC_UP * dt)
            else:
                self.v = max(v_t, self.v - ACC_DOWN * dt)
            # --- 最終接近: 残りが final_zone 以内で回り込み中でなければ「最後の一歩」で寄せて自然に止まる
            #     (小さな指令を出し続けると足踏みのまま居座る: 14:09 の実機)
            if (veer is None and d_stop is not None and 0.03 < d_stop <= float(p.get("final_zone", 0.6))
                    and self.v <= 0.35):
                self._final_approach("s", s + d_stop, "壁の手前")     # ★経路座標 s(進み traveled ではない)。後ろへは歩かない
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                what = "壁" if (ah and ah["wall"]) or wall_d is not None else "障害物"
                od, obs = self._sense()
                self.io["log"](f"{what} {obs.get('dist') if obs.get('dist') is not None else dist}m 手前で止まりました"
                               f"(停止距離{p['stop_dist']:.2f}m)  前進{self.traveled:.2f}m")
                return "wall"
            # --- 横速度: 回り込み中は目標の横位置へ(連続プロファイル)、それ以外は経路線を保つ
            if veer is not None:
                rem_e = veer["e_t"] - e
                if veer["stage"] == "out":
                    # 寄せている向き(固定)にだけ側方の物を見る。目標に届いた後の微調整では見ない
                    sgn = veer["dir"]
                    sf = obs.get("side_free_l" if sgn > 0 else "side_free_r")
                    if abs(rem_e) > 0.04 and sf is not None and sf < p["side_clear"]:
                        self.io["vel"](0.0, 0.0, 0.0)
                        self.v = 0.0
                        raise _Abort(f"中止(DETOUR_OUT): 回り込む側 {sf:.2f}m に障害物")
                    if abs(rem_e) <= 0.04 and s >= veer["s_end"]:
                        veer["stage"] = "back"
                        veer["e_t"] = 0.0
                        self.phase = "DETOUR_BACK"
                        self.io["log"](f"障害物を通り過ぎました(進み{self.traveled:.2f}m) — 歩きながら元の経路へ戻ります")
                else:
                    if abs(rem_e) <= 0.04:
                        veer = None
                        self.detours += 1
                        self.phase = "FORWARD"
                        self.io["log"](f"元の経路へ戻りました(回り込み{self.detours}回目)")
                if veer is not None:
                    vy_t = math.copysign(speed_profile(float(p["v_side"]), abs(rem_e), 0.10, 0.2, abs(vy_now), 0.3, 0.04)
                                         if abs(rem_e) > 0.04 else 0.0, rem_e)
                else:
                    vy_t = 0.0
            else:
                vy_t = float(np.clip(-LAT_KP * e, -LAT_VY_MAX, LAT_VY_MAX)) if (self.v > 0.05 and abs(e) > 0.06) else 0.0
            if vy_t > vy_now:
                vy_now = min(vy_t, vy_now + ACC_UP * dt)
            else:
                vy_now = max(vy_t, vy_now - ACC_UP * dt)
            om = self._om() if (self.v > 0.03 or abs(vy_now) > 0.03) else 0.0
            cx = self.cal.to_cmd(self.v, "x")
            cy = self.cal.to_cmd(vy_now, "y")
            self.io["vel"](cx, cy, om)
            self.cal.observe("x", cx, s, now)
            if abs(cy) > 0.02:
                self.cal.observe("y", cy, e, now)
            wa = obs.get("wall_ang")
            wa_s = "" if wa is None else f" 壁の角度{wa:+.0f}°"
            d_s = "---" if dist is None else f"{dist:.2f}m"
            self.msg = (f"{'回り込み' if veer is not None else '前進'} v={self.v:.2f} 横{vy_now:+.2f} 回転{om:+.2f} | 前方 "
                        f"{d_s}{'(壁)' if (ah and ah['wall']) else ''}{wa_s}"
                        f" | {self.traveled:.2f}m ずれ{e * 100:+.0f}cm")
            self._rec(vx=round(self.v, 3), vy=round(vy_now, 3), om=round(om, 3), cx=round(cx, 3), cy=round(cy, 3),
                      dist=dist, s=round(self.traveled, 3), e=round(e, 3), n=obs.get("n_obs"),
                      wall_ang=obs.get("wall_ang"), wall_len=obs.get("wall_len"),
                      ah_wall=(None if ah is None else bool(ah["wall"])),
                      ah_w=(None if ah is None else round(ah["lat_hi"] - ah["lat_lo"], 2)),
                      free_l=(None if ah is None else ah.get("free_l")), free_r=(None if ah is None else ah.get("free_r")),
                      veer=(None if veer is None else veer["stage"]))

    def _main(self):
        p = self.p
        io = self.io
        log = io["log"]
        dry = bool(p.get("dry_run", False))
        sdir = 1.0 if p.get("side_dir", "left") == "left" else -1.0
        mode = p.get("mode", "both")
        od, obs = self._sense()
        self._set_path(od)
        self.traveled_base = 0.0
        log(f"自動歩行 開始{'(ドライラン: 速度は送らない)' if dry else ''}"
            f"[{ {'both': '前進→横移動', 'forward': '前進のみ', 'side': '横移動のみ', 'back': '後退', 'step': '1歩'}.get(mode, mode)}]: "
            f"前進{p['v_fwd']:.2f}m/s 停止距離{p['stop_dist']:.2f}m "
            f"横{'左' if sdir > 0 else '右'}{p['side_dist']:.2f}m@{p['v_side']:.2f}m/s "
            f"最大前進{p['max_fwd']:.1f}m 回り込み{'あり' if p.get('avoid', True) else 'なし'}"
            f"  点群座標系={obs.get('frame')}")
        if mode == "step":
            self.phase = "STEP"
            self.result = "完了: " + self._step_once(p.get("step_dir", "left"))
            log(f"自動歩行 {self.result}")
            return
        if mode == "back":
            self.phase = "BACK"
            ok = self._back_to(float(p.get("back_dist", 0.05)), "後退")
            self.result = (f"完了: 後ろへ{-self.traveled:.2f}m下がりました" if ok
                           else f"中止(BACK): 後退{-self.traveled:.2f}m")
            log(f"自動歩行 {self.result}")
            return
        if mode == "side":
            self.phase = "SIDE"
            ok = self._lateral_to(sdir * p["side_dist"], f"横歩き({'左' if sdir > 0 else '右'})")
            self.side_traveled = sdir * self.offset
            self._log_post_stop()
            if not ok:
                self.result = f"中止(SIDE): 横方向に障害物(横移動{self.side_traveled:.2f}m/{p['side_dist']:.2f}m)"
                log(f"★自動歩行 {self.result}")
                return
            self.result = f"完了: {'左' if sdir > 0 else '右'}へ{self.side_traveled:.2f}m横移動"
            log(f"自動歩行 {self.result}")
            return
        if p.get("align_wall", True) and mode in ("both", "forward"):
            turned = self._align_to_wall()
            if abs(turned) > 0.5:
                od, obs = self._sense()
                self._set_path(od)                  # 正対した向きで経路(直進の基準)を張り直す
        how = self._forward()
        self._log_post_stop()
        if how == "max":
            self.result = (f"完了(壁なし): 最大前進距離{p['max_fwd']:.1f}mに到達"
                           f"{'。横移動はしません' if mode == 'both' else ''}")
            log(f"自動歩行 {self.result}")
            return
        self.phase = "STOPPING"
        self._hold(1.2, "停止中(1.2秒静定)")
        if mode == "forward":
            self.result = (f"完了: 前進{self.traveled:.2f}mで壁の手前に停止"
                           + (f"(回り込み{self.detours}回)" if self.detours else ""))
            log(f"自動歩行 {self.result}")
            return
        self.phase = "SIDE"
        e0 = self.offset
        log(f"横移動 開始: {'左' if sdir > 0 else '右'}へ {p['side_dist']:.2f}m")
        ok = self._lateral_to(e0 + sdir * p["side_dist"], f"横歩き({'左' if sdir > 0 else '右'})")
        self.side_traveled = sdir * (self.offset - e0)
        if not ok:
            self.result = f"中止(SIDE): 横方向に障害物(横移動{self.side_traveled:.2f}m/{p['side_dist']:.2f}m)"
            log(f"★自動歩行 {self.result}")
            return
        self.result = (f"完了: 前進{self.traveled:.2f}mで壁の手前に停止 → "
                       f"{'左' if sdir > 0 else '右'}へ{self.side_traveled:.2f}m横移動"
                       + (f"(回り込み{self.detours}回)" if self.detours else ""))
        log(f"自動歩行 {self.result}")


# ---------------------------------------------------------------- まとめ役
class WalkController:
    """コックピット(Engine)から見た唯一の入口。

    robot に要るもの:
      set_velocity(vx, vy, om, duration) -> bool   内蔵歩行への速度指令(同期・有限時間)
      open_lidar() / open_odom()                    .latest()/.enable()/.stop() を持つ読み手
      yaw() -> rad                                  IMUヨー(LowState)
      state() -> (q, dq, quat, gyro, tau)           傾きの計算用
      ensure_walk_mode(log) -> (ok, fsm)            歩行FSMへ
    """

    RANGES = {"v_fwd": (0.1, 1.0), "v_side": (0.1, 0.5), "v_creep": (0.1, 0.25), "a_dec": (0.1, 0.6), "cmd_lag": (0.0, 0.8),
              "stop_dist": (0.3, 2.5), "side_dist": (0.02, 3.0),
              "max_fwd": (0.3, 10.0), "half_w": (0.2, 0.6), "h_min": (0.05, 0.5),
              "h_max": (0.5, 2.5), "side_clear": (0.2, 1.5), "wall_width": (0.8, 3.0),
              "detour_margin": (0.05, 0.4), "detour_max": (0.3, 1.5),
              "side_tol": (0.01, 0.1), "pulse_on": (0.2, 1.0), "pulse_off": (0.2, 1.0),
              "step_v": (0.1, 0.4), "step_on": (0.2, 1.5), "step_off": (0.3, 2.0),
              "step_min_on": (0.15, 0.8), "step_est": (0.01, 0.3), "step_max": (1, 60),
              "back_dist": (0.02, 0.5), "self_fwd": (0.1, 0.8), "self_lat": (0.2, 0.8),
              "yaw_fix_deg": (-360.0, 360.0), "front_offset": (-0.3, 0.5),
              "align_tol_deg": (1.0, 15.0), "om_turn": (0.1, 0.6),
              "align_inplace_deg": (2.0, 45.0), "veer_v": (0.15, 0.6),
              "body_half": (0.15, 0.45),
              "speed_mode": (-1, 2), "cmd_dur": (0.5, 5.0),
              "k_vx": (0.15, 1.5), "k_vy": (0.15, 1.5), "v_dead": (0.0, 0.3), "cmd_max": (0.3, 1.2), "final_zone": (0.1, 0.5),
              "cmd_min_walk": (0.15, 0.6), "anchor_s": (0.0, 20.0),
              "tele_vx": (0.05, 0.6), "tele_vy": (0.05, 0.3), "tele_om": (0.1, 0.8)}

    def __init__(self, robot, log=print, hb_ok=lambda: True, is_sim=False):
        self.robot = robot
        self.log = log
        self.hb_ok = hb_ok
        self.is_sim = is_sim
        self.params = dict(WALK_DEFAULTS)
        self.lidar = None
        self.odom = None
        self.det = ObstacleDetector(self.params)
        self._cal = None             # LiDAR ヨー較正: 静止中の景色の占有格子とそのときのオドメトリ
        self._cal_last = 0.0
        self.sender = VelSender(self._send_vel, log=log)
        self.auto = None
        self.fsm_id = None
        self.ready = False
        self.enabled = False            # センサ読み取り+判定スレッドが動いているか
        self._obs = (dict(ok=False, why="未判定"), None)
        self._obs_lock = threading.Lock()
        self._tele_t = 0.0
        self._closing = False
        self.log_dir = None
        threading.Thread(target=self._perception, daemon=True,
                         name="walk_perception").start()

    # ---- 設定
    def set_params(self, d):
        changed = []
        for k, v in (d or {}).items():
            if k not in WALK_DEFAULTS:
                continue
            if k == "side_dir":
                v = "left" if str(v) == "left" else "right"
            elif k == "mode":
                v = str(v) if str(v) in ("both", "forward", "side", "back", "step") else "both"
            elif k == "step_dir":
                v = str(v) if str(v) in ("left", "right", "back", "fwd") else "left"
            elif k in ("dry_run", "avoid", "align_wall", "wall_track", "gait_cont", "yaw_autocal", "calib_learn", "stop_lock"):
                v = bool(v) if not isinstance(v, str) else (v.lower() in ("1", "true", "on", "yes"))
            else:
                try:
                    v = float(v)
                except Exception:                  # noqa: BLE001
                    continue
                lo, hi = self.RANGES.get(k, (-1e9, 1e9))
                v = min(hi, max(lo, v))
            if self.params.get(k) != v:
                self.params[k] = v
                changed.append(f"{k}={v}")
        self.det.cfg = dict(self.params)
        return changed

    # ---- センサ
    def enable_sensors(self, on):
        if on and not self.enabled:
            try:
                if self.lidar is None:
                    self.lidar = self.robot.open_lidar()
                if self.odom is None:
                    self.odom = self.robot.open_odom()
            except Exception as e:                 # noqa: BLE001
                self.log(f"★LiDAR/オドメトリの購読を開けない: {e}")
                return False
            self.lidar.enable(True)
            self.odom.enable(True)
            self.enabled = True
            self.log("LiDAR(rt/utlidar/cloud_livox_mid360)と"
                     "オドメトリ(rt/odommodestate)の読み取りを開始")
        elif not on and self.enabled:
            if self.lidar is not None:
                self.lidar.enable(False)
            if self.odom is not None:
                self.odom.enable(False)
            self.enabled = False
        return True

    MOUNT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_mount.json")

    def mount_yaw_file(self):
        """lidar_mount.json の yaw_offset_deg(無ければ 0)。5 秒キャッシュ"""
        now = time.time()
        if now - getattr(self, "_myf_t", 0.0) < 5.0:
            return getattr(self, "_myf", 0.0)
        v = 0.0
        try:
            with open(self.MOUNT_PATH) as f:
                v = float(json.load(f).get("yaw_offset_deg", 0.0))
        except Exception:                          # noqa: BLE001
            pass
        self._myf, self._myf_t = v, now
        return v

    def set_mount_yaw(self, deg):
        """lidar_mount.json の yaw_offset_deg を書き換える([前後を反転]・自動較正)。即時に反映"""
        deg = (float(deg) + 180.0) % 360.0 - 180.0
        d = {}
        try:
            with open(self.MOUNT_PATH) as f:
                d = json.load(f)
        except Exception:                          # noqa: BLE001
            d = {}
        d["yaw_offset_deg"] = round(deg, 1)
        with open(self.MOUNT_PATH, "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        self._myf_t = 0.0
        return deg

    def _yaw_calib_step(self, pts, frame, od, t):
        """前進したときに、景色の流れ(LiDAR)とオドメトリの移動方向を比べて LiDAR のヨーを直す。
        センサ座標の点群だけ。静止中の点群を覚えておき、0.25m 以上動いたら比べる。
        向きが 4 度以上変わったら比べない(平行移動だけを見る)。20 秒に 1 回まで"""
        if (not self.params.get("yaw_autocal", False) or od is None
                or self.det.is_world_frame(frame) or self.det.floor is None):
            self._cal = None
            return
        fwd, lat, z = self.det.to_body(pts, frame, None, None)
        a, b, c, _n = self.det.floor
        h = z - (fwd * a + lat * b + c)
        cfg = self.det.cfg
        own = (fwd > -0.8) & (fwd < cfg.get("self_fwd", 0.4)) & (np.abs(lat) < cfg.get("self_lat", 0.5))
        mask = (h > 0.3) & (h < 1.6) & ~own
        cal = getattr(self, "_cal", None)
        if cal is None:
            self._cal = dict(G=occupancy_grid(fwd, lat, mask), x=od[1], y=od[2], yaw=od[3], t=t)
            return
        dx, dy = od[1] - cal["x"], od[2] - cal["y"]
        d = math.hypot(dx, dy)
        dyaw = abs(_wrap(od[3] - cal["yaw"]))
        if dyaw > math.radians(4.0):
            self._cal = None
            return
        if d < 0.03:
            if t - cal["t"] > 2.0:                 # 静止中は新しい景色で覚え直す
                self._cal = dict(G=occupancy_grid(fwd, lat, mask), x=od[1], y=od[2], yaw=od[3], t=t)
            return
        if d < 0.25:
            return
        if time.time() - getattr(self, "_cal_last", 0.0) < 20.0:
            self._cal = None
            return
        G1 = occupancy_grid(fwd, lat, mask)
        sx, sy, best, med = scene_shift(cal["G"], G1)
        cy, sy0 = math.cos(cal["yaw"]), math.sin(cal["yaw"])
        db = (cy * dx + sy0 * dy, -sy0 * dx + cy * dy)      # 機体座標での移動(前, 左)
        e = yaw_error_from_motion((sx, sy), db)
        self._cal = None
        self._cal_last = time.time()
        if e is None or best < max(30, 2.0 * med):
            self.log(f"LiDAR ヨー較正: 判定できず(移動{d:.2f}m 景色の流れ({sx:+.2f},{sy:+.2f}) 相関{best}/中央{med:.0f})")
            return
        if abs(e) < 3.0:
            self.log(f"LiDAR ヨー較正: 向きは合っています(ずれ {e:+.1f}°、移動{d:.2f}m)")
            return
        e = max(-60.0, min(60.0, e)) if abs(e) < 120.0 else e   # 前後逆(±180 近く)はそのまま
        cur = self.mount_yaw_file()
        new = self.set_mount_yaw(cur - e)
        self.log(f"★LiDAR のヨーを自動較正: 検出器の前方が機体の前方から {e:+.1f}° ずれていました → "
                 f"yaw_offset_deg {cur:.0f}→{new:.0f}(lidar_mount.json、即時反映。移動{d:.2f}m 相関{best})")

    def _perception(self):
        """5Hzで点群を判定。方策の走行中は enable_sensors(False) されて止まる。"""
        while not self._closing:
            time.sleep(0.2)
            if not self.enabled or self.lidar is None:
                continue
            try:
                lf = self.lidar.latest()
                od = self.odom.latest() if self.odom is not None else None
                if lf is None:
                    continue
                t, pts, frame = lf
                mt = getattr(self.lidar, "mount", None)
                if isinstance(mt, dict) and mt.get("height") and str(frame).startswith("livox_level"):
                    self.det.floor_prior = -float(mt["height"])
                    # lidar_mount.json のヨー − ブリッジが適用済みのヨー = 今この場で足す分
                    #  ([前後を反転] を押した直後から効く。ブリッジは次の起動で json を読む)
                    yf = float(self.mount_yaw_file()) - float(mt.get("yaw_offset_deg", 0.0))
                    yf = (yf + 180.0) % 360.0 - 180.0
                    if abs(yf - float(self.params.get("yaw_fix_deg", 0.0))) > 0.01:
                        self.params["yaw_fix_deg"] = yf
                        self.det.cfg = dict(self.params)
                odxy = (od[1], od[2]) if od is not None else None
                sdir = 1 if self.params.get("side_dir", "left") == "left" else -1
                r = self.det.update(pts, frame, odxy, self.robot.yaw(), sdir)
                r["age_ms"] = round((time.time() - t) * 1000.0)
                r["frame_id"] = frame
                try:
                    self._yaw_calib_step(pts, frame, od, t)
                except Exception as e:             # noqa: BLE001
                    self._cal = None
                    self.log(f"(LiDAR ヨー較正の例外: {e})")
                with self._obs_lock:
                    self._obs = (r, t)
            except Exception as e:                 # noqa: BLE001
                with self._obs_lock:
                    self._obs = (dict(ok=False, why=f"判定の例外: {e}"), None)

    def obstacle(self):
        with self._obs_lock:
            return self._obs


    # ---- 送信
    def _send_vel(self, vx, vy, om, dur):
        return bool(self.robot.set_velocity(vx, vy, om, dur))

    def _tilt_deg(self):
        try:
            _q, _dq, quat, _g, _t = self.robot.state()
            w, x, y, z = [float(v) for v in quat]
            up_z = 1.0 - 2.0 * (x * x + y * y)
            return float(math.degrees(math.acos(min(1.0, max(-1.0, up_z)))))
        except Exception:                          # noqa: BLE001
            return 0.0

    # ---- 操作
    def prepare(self):
        """歩行FSMへ入れ、センサ読み取りを始める。ワーカースレッドで呼ぶこと(RPC)。"""
        ok, fsm = self.robot.ensure_walk_mode(log=self.log)
        self.fsm_id = fsm
        self.ready = bool(ok)
        if ok:
            self.locked = False
            self.enable_sensors(True)
            fn = getattr(self.robot, "set_speed_mode", None)
            if fn is not None:
                fn(int(self.params.get("speed_mode", 0)), log=self.log)
            fn2 = getattr(self.robot, "set_gait_continuous", None)
            if fn2 is not None and not self.params.get("gait_cont", False):
                fn2(False, log=self.log)           # 速度ゼロで足踏みをやめる設定(BalanceMode 0)を明示
            self.log(f"歩行モード(FSM {fsm})。[前進][横歩き]、十字キーの手動操作が押せます")
        else:
            self.log(f"★歩行モードへ入れませんでした(FSM {fsm})")
        return ok

    def tele(self, vx, vy, om):
        """手動操作(押している間だけ届く)。上限で切って送信スレッドへ渡す。"""
        if not self.ready:
            return False
        if self.auto is not None and not self.auto.done:
            return False                           # 自動歩行中は手動を受けない
        if getattr(self, "locked", False):
            if time.time() - getattr(self, "_t_lockmsg", 0.0) > 3.0:
                self._t_lockmsg = time.time()
                self.log("ロック立位で静止中です。十字キーで動かすには [歩行モード] を押してください([前進][横歩き]は自動で戻ります)")
            return False
        p = self.params
        vx = float(np.clip(vx, -p["tele_vx"] * 0.6, p["tele_vx"]))   # 後退は6割
        vy = float(np.clip(vy, -p["tele_vy"], p["tele_vy"]))
        om = float(np.clip(om, -p["tele_om"], p["tele_om"]))
        self._tele_t = time.time()
        self.sender.set(vx, vy, om, "tele", dry=bool(p.get("dry_run")))
        return True

    def _lock_stand(self):
        """目的地でロック立位(FSM 4)へ。足踏みをやめてバランスだけで静止する。戻り値: 成功か"""
        try:
            self.sender.stop("ロック立位へ")
            time.sleep(0.3)
            ok = bool(self.robot.standard_mode("stand"))
        except Exception as ex:                    # noqa: BLE001
            self.log(f"★ロック立位へ切り替えられません: {ex}")
            ok = False
        self.locked = ok
        if ok:
            self.fsm_id = 4
        return ok

    def _unlock_if_needed(self):
        """ロック立位(4)に居たら歩行(200)へ戻す。戻り値: 歩行 FSM に居るか"""
        f = None
        try:
            f = self.robot.get_fsm_id()
        except Exception:                          # noqa: BLE001
            pass
        if f in WALK_FSMS:
            self.locked = False
            return True
        self.log("ロック立位から歩行へ戻します")
        ok, f2 = self.robot.ensure_walk_mode(log=self.log)
        self.fsm_id = f2
        self.locked = not ok
        if ok:
            time.sleep(0.5)                        # 歩行制御の立ち上がりを待つ
        return bool(ok)

    def start_auto(self, overrides=None):
        """自動歩行を始める。overrides でこの1回だけのパラメータ(mode, side_dist 等)を上書き"""
        if not self.ready:
            self.log("★先に[歩行モードへ]を押してください")
            return False
        if self.auto is not None and not self.auto.done:
            self.log("★自動歩行はすでに実行中です")
            return False
        if not self._unlock_if_needed():
            self.log("★歩行モードへ戻れませんでした")
            return False
        if not self.enabled:
            self.enable_sensors(True)
        obs, t = self.obstacle()
        od = self.odom.latest() if self.odom is not None else None
        now = time.time()
        params = dict(self.params)
        if overrides:
            tmp = WalkController.__new__(WalkController)
            tmp.params = params
            tmp.det = self.det
            WalkController.set_params(tmp, overrides)
            self.det.cfg = dict(self.params)       # 検出器の設定は本体のまま
        side_only = (params.get("mode", "both") in ("side", "back"))
        lidar_ok = (t is not None and now - t <= 1.5)
        if od is None or now - od[0] > 0.8:
            self.log("★オドメトリが届いていません(rt/odommodestate)。"
                     "自動歩行は始めません")
            return False
        if not lidar_ok:
            if side_only:
                # 横歩きだけならオドメトリで動かせる。横の空きは見ない(操作者が見る)
                self.log("△LiDAR点群が無いので、横歩きは横方向の障害物を見ずに動きます"
                         "(オドメトリのみ)。周囲を目で確認してください")
            else:
                self.log("★LiDAR点群が届いていません(rt/utlidar/cloud_livox_mid360)。"
                         "前進は壁が見えないので始めません(横歩き・十字キーは使えます)")
                return False
        elif not obs.get("ok") and not side_only:
            self.log(f"★障害物の判定ができません: {obs.get('why')}。自動歩行は始めません")
            return False
        io = dict(obstacle=self.obstacle, lock_stand=self._lock_stand,
                  odom=lambda: self.odom.latest(),
                  yaw=self.robot.yaw,
                  tilt_deg=self._tilt_deg,
                  hb_ok=self.hb_ok,
                  vel=lambda vx, vy, om: self.sender.set(vx, vy, om, "auto",
                                                         dry=bool(params.get("dry_run"))),
                  stop=lambda why: self.sender.stop(why),
                  log=self.log)
        lp = None
        if self.log_dir is not None:
            lp = pathlib.Path(self.log_dir) / time.strftime("walk_%Y%m%d_%H%M%S.jsonl")
        self.sender.duration = float(self.params.get('cmd_dur', 2.0))
        fn = getattr(self.robot, 'set_gait_continuous', None)
        if fn is not None and self.params.get('gait_cont', True) and not self.params.get('dry_run'):
            fn(True, log=self.log)
            self._gait_on = True

        self.auto = AutoWalk(params, io, log_path=lp)
        self.auto.start()
        return True

    def stop(self, why="停止"):
        fn = getattr(self.robot, "set_gait_continuous", None)
        if fn is not None and getattr(self, "_gait_on", False):
            self._gait_on = False
            fn(False, log=self.log)
        """速度ゼロ。damp ではない(内蔵バランスに立たせたまま止める)。"""
        if self.auto is not None and not self.auto.done:
            self.auto.request_stop(why)
        self.sender.stop(why)

    def close(self):
        self._closing = True
        self.stop("終了")
        self.sender.close()
        for r in (self.lidar, self.odom):
            if r is not None:
                try:
                    r.stop()
                except Exception:                  # noqa: BLE001
                    pass

    # ---- UI
    def status(self):
        obs, t = self.obstacle()
        od = self.odom.latest() if self.odom is not None else None
        now = time.time()
        a = self.auto
        ah = obs.get("ahead") or {}
        return {
            "ready": bool(self.ready), "fsm_id": self.fsm_id,
            "locked": bool(getattr(self, "locked", False)),
            "sensors": bool(self.enabled),
            "auto": bool(a is not None and not a.done),
            "phase": (a.phase if a is not None else "-"),
            "mode": (a.p.get("mode") if a is not None else self.params.get("mode")),
            "msg": (a.msg if a is not None and not a.done else
                    (a.result if a is not None else "")),
            "dist": obs.get("dist"), "n_obs": obs.get("n_obs"),
            "wall": ah.get("wall"), "width": (None if not ah else round(ah["lat_hi"] - ah["lat_lo"], 2)),
            "free_l": ah.get("free_l"), "free_r": ah.get("free_r"),
            "side_free": obs.get("side_free"),
            "side_free_l": obs.get("side_free_l"), "side_free_r": obs.get("side_free_r"),
            "rear_n": obs.get("rear_n"), "rear_dist": obs.get("rear_dist"),
            "rear_h": obs.get("rear_h"),
            "frame": obs.get("frame_id", obs.get("frame")),
            "floor_ok": obs.get("floor_ok"), "floor_h": obs.get("floor_h"),
            "why": obs.get("why", ""),
            "wall_dist": obs.get("wall_dist"), "wall_ang": obs.get("wall_ang"), "wall_len": obs.get("wall_len"),
            "dirs": obs.get("dirs"),
            "yaw_fix_deg": float(self.params.get("yaw_fix_deg", 0.0)),
            "k_vx": (round(a.cal.k["x"], 2) if a is not None else None),
            "k_vy": (round(a.cal.k["y"], 2) if a is not None else None),
            "lidar_age_ms": (None if t is None else round((now - t) * 1000)),
            "lidar_n": (self.lidar.n_recv if self.lidar is not None else 0),
            "lidar_src": getattr(self.lidar, "source", "-"),
            "mount": getattr(self.lidar, "mount", None),
            "odom_age_ms": (None if od is None else round((now - od[0]) * 1000)),
            "odom": (None if od is None else [round(od[1], 3), round(od[2], 3),
                                              round(math.degrees(od[3]), 1)]),
            "traveled": (round(a.traveled, 2) if a is not None else 0.0),
            "side_traveled": (round(a.side_traveled, 2) if a is not None else 0.0),
            "offset": (round(a.offset, 3) if a is not None else 0.0),
            "detours": (a.detours if a is not None else 0),
            "steps": (a.steps if a is not None else 0),
            "step_last_cm": (None if a is None or a.step_last is None else round(a.step_last * 100, 1)),
            "step_est_cm": round((a.step_est if a is not None else float(self.params.get("step_est", 0.06))) * 100, 1),
            "v": (round(a.v, 2) if a is not None else 0.0),
            "sent": [round(v, 2) for v in self.sender.last_sent],
            "sent_age_ms": round((now - self.sender.last_sent_t) * 1000)
            if self.sender.last_sent_t else None,
            "n_sent": self.sender.n_sent, "n_fail": self.sender.n_fail,
            "params": dict(self.params),
        }
