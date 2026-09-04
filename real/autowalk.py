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
    # ---- 速度指令(内蔵歩行 SetVelocity 7105 に送る値。2026-09-04 の調査と実測に基づく)
    v_fwd=0.70,        # 前進の指令の上限[m/s](かんたん画面の「速さ」: ゆっくり 0.5 / ふつう 0.7 / はやい 0.9)
    v_side=0.50,       # 横歩き・後退の指令の上限[m/s]
    cmd_min=0.30,      # 動かすときの指令の下限[m/s]。★これ未満は足踏みだけで進まない(実測: 0.20→0.04m/s)。止めるときは 0
    k_dist=0.90,       # 前進: 指令 = k × (LiDAR の距離 − 停止距離)[m/s per m]。1m 手前で上限、0.33m 手前で下限
    k_side=1.20,       # 横・後退: 指令 = k × 残り距離(オドメトリ)
    stop_lead=0.15,    # 前進: 残り(LiDAR)がこれ以下[m]になったら指令ゼロ(下限指令 0.30 の実速度 ≈0.08 × 応答遅れ ≈1.5 s)
    slew_up=0.60,      # 指令の上げ幅の上限[m/s²]
    slew_down=0.50,    # 指令の下げ幅の上限[m/s²]
    cmd_dur=0.30,      # SetVelocity の duration[s]。10Hz で上書きし続ける。送信が止まれば 0.3 秒で内蔵が自動停止(デッドマン)
    settle_s=1.0,      # 停止: fsm_mode==0(静止)がこの秒数続いたら「止まった」
    settle_max=6.0,    # 停止: 最長これだけ待つ[s]
    stop_dist=0.60,    # 前の壁の手前で止まる距離[m](つま先から)
    side_dist=0.50,    # 横歩きの距離[m]
    side_dir="left",   # 横歩きの向き
    max_fwd=4.0,       # 前進の上限距離[m](壁が無いとき)
    back_dist=0.05,    # 後退の距離[m]
    mode="both",       # both=前進→横移動 / forward / side / back / step
    step_dir="left",   # mode="step"(1歩だけ)の向き: left/right/back/fwd
    step_on=0.6,       # 微調整(10cm 以下)の 1 歩: 指令時間[s]
    step_off=1.0,      # 同: 止めて測る時間[s]
    step_max=20,
    tele_vx=0.5,       # 十字キーの前進上限[m/s]
    tele_vy=0.4,       # 同・横
    tele_om=0.45,      # 同・旋回[rad/s]
    dry_run=False,     # 速度を送らない(判定と表示だけ)
    # ---- 壁・障害物(ObstacleDetector)
    half_w=0.35,       # 進行方向の帯の半幅[m](この中の点を障害物とみなす)
    h_min=0.12,        # 障害物とみなす高さ帯[m](床から)
    h_max=1.80,
    side_clear=0.45,   # 横移動中、進行方向にこの距離未満で点があれば止まる[m]
    self_fwd=0.40,     # 自分の体(頭のLiDARの真下〜腕)を除く範囲: 前方この距離まで[m]
    self_lat=0.50,     # 同・左右この幅まで[m]
    yaw_fix_deg=0.0,   # センサ座標の点群に足すヨー[deg](lidar_mount.json の値 − ブリッジが適用済みの値)。[前後を反転]で±180
    front_offset=0.15, # センサ軸→つま先の前方距離[m]。距離は「つま先から」で出す
    yaw_autocal=False, # LiDAR ヨーの自動較正(壁だけの景色では決まらないので既定 OFF)
    # ---- 壁への正対と追従
    align_wall=True,   # 前進の前に、正面の壁が斜めならその場で回転して正対する
    align_tol_deg=3.0,
    align_inplace_deg=8.0,  # これを超えていたらその場で回る。以下なら歩きながら合わせる
    om_turn=0.30,      # 正対の回転速度の上限[rad/s]
    wall_track=True,   # 歩行中も壁の角度を測り続け、向きを壁に垂直へ寄せ続ける
    # ---- 障害物の回り込み(歩きながら斜めに)
    avoid=True,
    wall_width=1.4,    # 横幅がこれより広い物は壁(回り込まない)[m]
    detour_margin=0.08,# 障害物の端と体の側面との余白[m]
    body_half=0.25,    # 体の半幅[m]
    detour_max=0.9,    # 回り込みで横へ出る上限[m]
    veer_v=0.50,       # 回り込み(斜め歩き)中の前進指令の上限[m/s]
    side_tol=0.03,     # 横移動の到達許容[m]
)
WALK_FSMS = {200, 500, 501}        # 速度指令を受ける内蔵FSM(loco)。802/801 では効かない(2026-09-04 実測)
YAW_KP = 1.6                       # 直進保持のゲイン[(rad/s)/rad](旧コックピット実績値)
YAW_OM_MAX = 0.30                  # 直進保持の補正上限[rad/s]
SEND_HZ = 10.0                     # 速度送信の周期
LIDAR_STALE_S = 0.8                # 点群がこれ以上古ければ前進を中止(0.8秒 = 巡航で約20cm)
CMD_HOLD_S = 0.5                   # 指令の有効期間。これを過ぎたらゼロを送る
TILT_ABORT_DEG = 25.0              # 歩行中にこれを超えたら自動歩行を止める
SENSOR_FWD_OFFSET = 0.10           # センサ座標の点群を使うときのセンサ→骨盤の前方オフセット[m]
MAX_DETOURS = 2                    # 1回の前進で回り込む回数の上限
EMERGENCY_STOP_M = 0.45            # 回り込み中でも、これより近い点があれば止める[m](自己除外 0.40m の外側)

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
                   side_fwd_l=None, side_fwd_r=None,
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
            # 真横(体の脇)。★|lat| < self_lat は自分の体として除外されるので、これより近い物は見えない
            #   (真横の物は操作者が目で見る前提。2026-09-04 レビュー)
            ms = hm & (fwd > -0.35) & (fwd < 0.45) & (s * lat > 0.12) & (s * lat < 1.5)
            if int(ms.sum()) >= 12:
                out[key] = round(float(np.percentile(s * lat[ms], 5)), 3)
            # 斜め前(0.45〜1.2m)。横歩きのときだけ使う(回り込み中は避けている物が入るので使わない)
            mf = hm & (fwd > 0.45) & (fwd < 1.2) & (s * lat > 0.12) & (s * lat < 1.5)
            if int(mf.sum()) >= 12:
                out["side_fwd_l" if s > 0 else "side_fwd_r"] = round(float(np.percentile(s * lat[mf], 5)), 3)
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
        self.duration = 0.3          # 指令の有効時間[s](cmd_dur)。10Hz で上書きし続ける。止まれば 0.3 秒で内蔵が自動停止
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
            # ★ゼロ指令も新しいうちは送り続ける(「最新の指令はゼロ」を内蔵へ確実に伝える。
            #   2026-09-04 の調査: 止め方は最後の指令をゼロにして持続させるのが公式の作法)
            moving = fresh
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
class _Abort(Exception):
    pass


class AutoWalk:
    """前進(壁の手前で止まる・障害物は歩きながら回り込む) / 横歩き / 後退 / 1歩。10Hz のスレッド。

    2026-09-04 夕、調査(docs/自動歩行 §6b-15)に基づいて**いちから**書き直した。方針:
      - 指令は「LiDAR の残り距離に比例、下限 cmd_min(0.30)・上限 v_fwd」。下限未満の指令は出さない
        (足踏みだけで進まず、後ろへ流れる)。止めるときはゼロ。
      - 止まったかは内蔵の fsm_mode(GetFsmMode 7002: 0=静止)で判定する。FSM は切り替えない
        (4 は動作中には拒否される。802 は走跑モードで速度指令を受けない)。
      - 距離は LiDAR。オドメトリは足踏みも距離に数えるので、横歩き・後退(LiDAR が測れない)にだけ使う。
      - 学習・較正・最終接近・アンカー保持のような継ぎ足しは持たない。

    io は WalkController が渡す関数の束:
      obstacle()  → ObstacleDetector.update の結果(最新)と、その時刻
      odom()      → (t, x, y, yaw_odom, vx, vy) / None
      yaw()       → IMUヨー[rad](LowState)
      tilt_deg()  → 体幹の傾き[度]
      hb_ok()     → UIハートビートが生きているか
      fsm_mode()  → 0 静止 / 1 動作中 / None(読めない)
      vel(vx,vy,om) / stop(why) / log(str)
    """

    def __init__(self, params, io, log_path=None):
        self.p = dict(params)
        self.io = io
        self.phase = "INIT"
        self.msg = ""
        self.done = False
        self.result = ""
        self.traveled = 0.0          # 経路に沿った前進[m](オドメトリ。表示用)
        self.traveled_base = 0.0
        self.side_traveled = 0.0     # 横歩きした距離[m]
        self.offset = 0.0            # 経路線からの左へのずれ[m]
        self.detours = 0
        self.v = 0.0                 # いま送っている前進指令[m/s]
        self.v_meas = 0.0            # LiDAR の距離の減りから見た実速度[m/s]
        self.step_last = None
        self.step_est = 0.06
        self.steps = 0
        self.stop_info = None        # 停止の記録 {d0, d1, d2, t_settle, fsm_mode}
        self.t_start = time.time()
        mode0 = self.p.get("mode", "both")
        self.need_lidar = (mode0 not in ("side", "back")
                           and not (mode0 == "step" and self.p.get("step_dir", "left") != "fwd"))
        self._stop_req = None
        self._log_path = pathlib.Path(log_path) if log_path else None
        self._rows = []
        self._th = threading.Thread(target=self._run, daemon=True, name="autowalk")
        self._x0 = self._y0 = 0.0
        self._fx = self._fy = 0.0
        self._lx = self._ly = 0.0
        self._yaw_ref = 0.0
        self._path_yaw_ref = 0.0

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
        if self.need_lidar and (obs_t is None or now - obs_t > LIDAR_STALE_S):
            raise _Abort(f"中止({self.phase}): LiDAR途絶({LIDAR_STALE_S:.1f}秒)")
        if not self.need_lidar and (obs_t is None or now - obs_t > LIDAR_STALE_S):
            obs = dict(ok=False, why="LiDAR無し")
        self._obs_t = obs_t
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
        if not self._rows or self._log_path is None:
            self._rows = []
            return
        try:
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

    # ---- 指令の作り方
    def _cmd_for(self, err, cmd_max, k):
        """残り距離 err[m] → 指令[m/s]。err<=0 なら 0、それ以外は k·err を [cmd_min, cmd_max] に収める"""
        if err <= 0.0:
            return 0.0
        return float(min(cmd_max, max(float(self.p["cmd_min"]), k * err)))

    def _slew(self, cur, target, dt):
        """指令の変化を鈍らせる。★ただし 0 と cmd_min の間の値は出さない(足踏みだけで進まず後ろへ流れる帯)。
        目標がゼロなら即ゼロ、動かすなら最低 cmd_min から始める(2026-09-04 レビュー)"""
        cm = float(self.p["cmd_min"])
        sgn = 1.0 if (target if abs(target) > 1e-9 else cur) >= 0 else -1.0
        a, b = abs(cur), abs(target)
        if b < 1e-9:
            return 0.0
        v = min(b, a + float(self.p["slew_up"]) * dt) if b > a else max(b, a - float(self.p["slew_down"]) * dt)
        return sgn * max(cm, v)

    def _side_clear_m(self, obs, sgn):
        """横歩きのときの「その側の空き」[m]。真横(side_free)と斜め前(side_fwd)の近い方。
        ★真横は自分の体の除外(|lat| < self_lat)で見えないことが多い。無いとは言えない(操作者の目が必要)"""
        if not obs.get("ok"):
            return None
        a = obs.get("side_free_l" if sgn > 0 else "side_free_r")
        b = obs.get("side_fwd_l" if sgn > 0 else "side_fwd_r")
        c = [x for x in (a, b) if x is not None]
        return min(c) if c else None

    def _lidar_dist(self, obs):
        """前方の残り距離の元(LiDAR)。帯の中の最近点と、壁の面(幅が壁幅の 8 割以上)の近い方"""
        if not obs.get("ok"):
            return None
        d = obs.get("dist")
        wl = obs.get("wall_len") or 0.0
        wd = obs.get("wall_dist") if wl >= 0.8 * float(self.p["wall_width"]) else None
        cands = [x for x in (d, wd) if x is not None]
        return min(cands) if cands else None

    # ---- 停止(公式の判定 fsm_mode==0 を待つ)
    def _stop_and_settle(self, label):
        """指令ゼロを送り続け、内蔵の fsm_mode が 0(静止)で settle_s 秒続くまで待つ(最長 settle_max)。
        fsm_mode が読めない FW では、LiDAR の距離(壁が見えていれば)またはオドメトリの動きが止まるのを待つ。
        止まってから 3 秒後の距離も記録し、ログに「停止: 指令ゼロ時 d0 → 静止 d1 → 3秒後 d2」を出す。
        ★FSM は切り替えない(4 は動作中に拒否される。802 は走跑モードで速度指令を受けない)"""
        p = self.p
        self.io["vel"](0.0, 0.0, 0.0)
        self.v = 0.0
        need = self.need_lidar
        self.need_lidar = False                     # 止まる間は LiDAR が途切れても中止しない
        od, obs = self._sense()
        d0 = self._lidar_dist(obs)
        s0, e0 = self._pose(od)
        t0 = time.time()
        self._t_fsm_poll = 0.0
        t_still = None
        d_prev, x_prev, t_prev = d0, (s0, e0), t0
        mode = self.io["fsm_mode"]()
        t_settle = None
        while time.time() - t0 < float(p["settle_max"]):
            time.sleep(0.1)
            od, obs = self._sense()
            self.io["vel"](0.0, 0.0, 0.0)
            now = time.time()
            if now - getattr(self, "_t_fsm_poll", 0.0) > 0.3:
                self._t_fsm_poll = now
                mode = self.io["fsm_mode"]()
            s, e = self._pose(od)
            d = self._lidar_dist(obs)
            if mode is not None:
                still = (mode == 0)
            elif d is not None and d_prev is not None and abs(d - d_prev) < 0.5:
                still = abs(d - d_prev) / max(0.05, now - t_prev) < 0.02
            else:
                still = math.hypot(s - x_prev[0], e - x_prev[1]) / max(0.05, now - t_prev) < 0.03
            d_prev, x_prev, t_prev = d, (s, e), now
            if still:
                t_still = t_still or now
                if now - t_still >= float(p["settle_s"]):
                    t_settle = now - t0
                    break
            else:
                t_still = None
            self.msg = f"{label}: 止まるのを待っています(fsm_mode {mode if mode is not None else '-'})"
            self._rec(settle=1, fsm_mode=mode, d=d)
        od, obs = self._sense()
        d1 = self._lidar_dist(obs)
        s1, e1 = self._pose(od)
        # 3 秒後の位置(止めた後に流れないか)
        t2 = time.time()
        while time.time() - t2 < 3.0:
            time.sleep(0.1)
            self._sense()
            self.io["vel"](0.0, 0.0, 0.0)
        od, obs = self._sense()
        d2 = self._lidar_dist(obs)
        s2, e2 = self._pose(od)
        f = lambda x: "-" if x is None else f"{x:.2f}m"          # noqa: E731
        self.stop_info = dict(d0=d0, d1=d1, d2=d2, t_settle=t_settle, fsm_mode=mode)
        self.io["log"](f"停止({label}): 指令ゼロ時 {f(d0)} → 静止 {f(d1)}"
                       f"({'%.1f秒' % t_settle if t_settle is not None else '静止を確認できず %.0f秒' % float(p['settle_max'])}"
                       f"、fsm_mode {mode if mode is not None else '読めない'}) → 3秒後 {f(d2)}"
                       f"  オドメトリ 前後{(s2 - s0) * 100:+.0f}cm 横{(e2 - e0) * 100:+.0f}cm")
        self.need_lidar = need
        return t_settle is not None

    # ---- 前進(壁の手前で止まる。障害物は歩きながら回り込む)
    def _forward(self):
        """戻り値: "wall"(壁・障害物の手前で停止) / "max"(壁なしで最大距離)"""
        p = self.p
        self.phase = "FORWARD"
        self.v = 0.0
        vy = 0.0
        veer = None
        t_phase = time.time()
        t_prev = t_phase
        t_limit = float(p["max_fwd"]) / 0.08 + 20.0
        d_prev = None
        t_dprev = None
        obs_t_prev = None
        hit = 0
        stall_d = None                              # 停滞の見張り: 前回そこそこ動いていたときの距離
        stall_t = time.time()
        while True:
            time.sleep(0.1)
            now = time.time()
            dt = max(0.02, min(0.3, now - t_prev))
            t_prev = now
            od, obs = self._sense()
            if now - t_phase > t_limit:
                raise _Abort(f"中止(FORWARD): 時間切れ({t_limit:.0f}秒)")
            s, e = self._pose(od)
            self.traveled, self.offset = self.traveled_base + s, e
            if not obs.get("ok"):
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                self.msg = f"待機: 障害物の判定不能({obs.get('why')})"
                continue
            if veer is None:                        # ★回り込み中は経路を張り直さない(目標が壊れる)
                self._track_wall_heading(obs, od)
            dist = obs.get("dist")                  # 帯の中の最近点(障害物も壁も)
            ah = obs.get("ahead")
            wl = obs.get("wall_len") or 0.0
            wall_d = obs.get("wall_dist") if wl >= 0.8 * float(p["wall_width"]) else None
            # 実速度(LiDAR の距離の減り)。★点群が更新されたコマだけで測る(同じコマを 2 回数えない)
            d_now = self._lidar_dist(obs)
            fresh_obs = (self._obs_t is not None and self._obs_t != obs_t_prev)
            if fresh_obs:
                if (d_now is not None and d_prev is not None and t_dprev is not None
                        and abs(d_now - d_prev) < 0.5):
                    self.v_meas = 0.7 * self.v_meas + 0.3 * ((d_prev - d_now) / max(0.05, now - t_dprev))
                d_prev, t_dprev = d_now, now
                obs_t_prev = self._obs_t
            # --- 回り込みの開始: 壁でない物体が近づいたら、空いている側へ歩きながら寄せる
            if (veer is None and p.get("avoid", True) and dist is not None and ah is not None
                    and not ah["wall"] and self.detours < MAX_DETOURS
                    and dist <= max(float(p["stop_dist"]) + 0.9, 1.5)):
                cands = [c for c in (ah.get("free_l"), ah.get("free_r")) if c is not None]
                if cands:
                    e_t = min(cands, key=abs)
                    veer = dict(e_t=e + e_t, s_end=s + dist + ah["depth"] + 0.25, stage="out",
                                dir=(1.0 if e_t > 0 else -1.0), moving_e=True,
                                obj_lo=e + ah["lat_lo"], obj_hi=e + ah["lat_hi"],
                                obj_s=s + dist + ah["depth"])
                    self.phase = "DETOUR_OUT"
                    self.io["log"](f"障害物 {dist:.2f}m(横幅{ah['lat_hi'] - ah['lat_lo']:.2f}m・奥行{ah['depth']:.2f}m) — "
                                   f"歩きながら{'左' if e_t > 0 else '右'}へ{abs(e_t):.2f}m寄せて脇を通ります")
                elif now - getattr(self, "_t_noveer", 0.0) > 2.0:
                    self._t_noveer = now
                    self.io["log"](f"障害物 {dist:.2f}m(横幅{ah['lat_hi'] - ah['lat_lo']:.2f}m)に回り込み先が無い"
                                   f"(左 {ah.get('free_l')} / 右 {ah.get('free_r')}) — 手前で止まります")
            # --- 残り距離。回り込み中は「いま避けている物」だけ無視し、それ以外(別の物・壁)は普通に止まる。
            #     ★自分の体を隠す範囲が前方 0.40m なので、緊急の床は 0.45m(それ未満は測れない)
            if veer is not None:
                # 「いま避けている物」かどうか(経路座標での横の重なり)
                same_obj = False
                if ah is not None and dist is not None:
                    lo, hi = e + ah["lat_lo"], e + ah["lat_hi"]
                    ov = min(hi, veer["obj_hi"]) - max(lo, veer["obj_lo"])
                    w = min(max(hi - lo, 1e-3), max(veer["obj_hi"] - veer["obj_lo"], 1e-3))
                    same_obj = (ov > 0.5 * w) and (s + dist <= veer["obj_s"] + 0.6)
                # 横の余白(体の中心から物の端まで)。体の半幅ぶん空いていれば脇を通れる
                if e <= veer["obj_lo"]:
                    clear_e = veer["obj_lo"] - e
                elif e >= veer["obj_hi"]:
                    clear_e = e - veer["obj_hi"]
                else:
                    clear_e = 0.0
                can_pass = (clear_e >= float(p["body_half"])) or (s > veer["obj_s"] + 0.2)
                veer["clear_e"] = clear_e
                cands = [x for x in (wall_d,) if x is not None]
                # 避けている物は、横の余白が取れていれば無視して脇を通る。取れていなければ普通に止まる。
                # 別の物(重なりが無い)は常に見る
                if dist is not None and not (same_obj and can_pass):
                    cands.append(dist)
                d_use = min(cands) if cands else None
            else:
                d_use = min([x for x in (dist, wall_d) if x is not None], default=None)
            err = None if d_use is None else d_use - float(p["stop_dist"])
            # --- 停止: 残りが stop_lead 以下。★点群が更新されたコマで 2 回続けて(1 コマのノイズで止めない)
            if fresh_obs:
                hit = hit + 1 if (err is not None and err <= float(p["stop_lead"])) else 0
            elif err is not None and err > float(p["stop_lead"]):
                hit = 0
            if hit >= 2:
                what = "壁" if (wall_d is not None and (d_use == wall_d)) or (ah and ah["wall"]) else "障害物"
                self.io["log"](f"{what} {d_use:.2f}m(停止距離{p['stop_dist']:.2f}m) — 止めます  前進{self.traveled:.2f}m")
                self.phase = "STOPPING"
                self._stop_and_settle("壁の手前")
                return "wall"
            if self.traveled >= float(p["max_fwd"]):
                self.phase = "STOPPING"
                self._stop_and_settle("最大距離")
                return "max"
            # --- 停滞の見張り: 下限以上の指令を出しているのに 6 秒で 5cm も縮まらない = 歩行が指令に応じていない
            if self.v >= float(p["cmd_min"]) - 1e-6 and d_now is not None:
                # 進んだ(5cm 縮んだ)か、見ている物が変わった(25cm 以上遠くなった。回り込みで脇を抜けた等)なら数え直す
                if stall_d is None or d_now < stall_d - 0.05 or d_now > stall_d + 0.25:
                    stall_d, stall_t = d_now, now
                elif now - stall_t > 6.0:
                    self.io["vel"](0.0, 0.0, 0.0)
                    raise _Abort(f"中止(FORWARD): 6秒 指令{self.v:.2f}を出しても前方の距離が縮まらない"
                                 f"({d_now:.2f}m)。歩行モードが速度指令に応じていない可能性"
                                 "(十字キーで歩けるか確認。docs 自動歩行 §6b-15)")
            else:
                stall_d, stall_t = None, now
            # --- 前進指令: 残り距離に比例(下限 cmd_min・上限 v_fwd)。回り込み中は veer_v 以下
            cmd_max = float(p["v_fwd"]) if veer is None else min(float(p["v_fwd"]), float(p["veer_v"]))
            v_t = self._cmd_for(err, cmd_max, float(p["k_dist"])) if err is not None else cmd_max
            if veer is not None and v_t < float(p["cmd_min"]):
                v_t = float(p["cmd_min"])            # 寄せている間は歩き続ける
            self.v = self._slew(self.v, v_t, dt)
            # --- 横指令: 回り込み中は目標の横位置へ、それ以外は経路線の保持(10cm 超のずれだけ)
            vy_t = 0.0
            if veer is not None:
                rem_e = veer["e_t"] - e
                if veer["stage"] == "out":
                    # 奥行きの見積り(0.5m 上限)より深い物は、見えている間は「通り過ぎる位置」を伸ばす
                    if (same_obj and dist is not None and ah is not None and not ah["wall"]
                            and s < veer["obj_s"]):     # ★まだ通り過ぎていない間だけ(壁では伸ばさない)
                        veer["obj_s"] = max(veer["obj_s"], s + dist + ah["depth"])
                        veer["s_end"] = max(veer["s_end"], veer["obj_s"] + 0.25)
                    sf = obs.get("side_free_l" if veer["dir"] > 0 else "side_free_r")
                    if abs(rem_e) > 0.06 and sf is not None and sf < float(p["side_clear"]):
                        self.io["vel"](0.0, 0.0, 0.0)
                        self.v = 0.0
                        raise _Abort(f"中止(DETOUR_OUT): 回り込む側 {sf:.2f}m に障害物")
                    if abs(rem_e) <= 0.06 and s >= veer["s_end"]:
                        veer["stage"] = "back"
                        veer["e_t"] = 0.0
                        self.phase = "DETOUR_BACK"
                        self.io["log"](f"障害物を通り過ぎました(進み{self.traveled:.2f}m) — 歩きながら元の経路へ戻ります")
                else:
                    if abs(rem_e) <= 0.06:
                        veer = None
                        self.detours += 1
                        self.phase = "FORWARD"
                        self.io["log"](f"元の経路へ戻りました(回り込み{self.detours}回目)")
                if veer is not None:
                    # ヒステリシス: 10cm 超で寄せ始め、3cm 未満で止める(往復を防ぐ)
                    if veer.get("moving_e", True) and abs(rem_e) < 0.03:
                        veer["moving_e"] = False
                    elif not veer.get("moving_e", True) and abs(rem_e) > 0.10:
                        veer["moving_e"] = True
                    if veer["moving_e"]:
                        vy_t = math.copysign(self._cmd_for(abs(rem_e), float(p["v_side"]), float(p["k_side"])), rem_e)
            elif abs(e) > 0.10 and self.v > 0.0:
                vy_t = math.copysign(float(p["cmd_min"]), -e)
            vy = self._slew(vy, vy_t, dt) if vy_t >= vy else max(vy_t, vy - float(p["slew_down"]) * dt)
            om = self._om() if (self.v > 0.0 or abs(vy) > 0.0) else 0.0
            self.io["vel"](self.v, vy, om)
            wa = obs.get("wall_ang")
            self.msg = (f"{'回り込み' if veer is not None else '前進'}"
                        f"{'' if veer is None else '(余白%.2fm)' % veer.get('clear_e', 0.0)}"
                        f" 指令{self.v:.2f} 横{vy:+.2f} 回転{om:+.2f} | 残り "
                        f"{'---' if err is None else f'{err:.2f}m'} 実速度{self.v_meas:.2f}"
                        f"{'' if wa is None else f' 壁の角度{wa:+.0f}°'} | {self.traveled:.2f}m ずれ{e * 100:+.0f}cm")
            self._rec(cx=round(self.v, 3), cy=round(vy, 3), om=round(om, 3), dist=dist, wall_d=wall_d, err=err,
                      v_meas=round(self.v_meas, 3), s=round(self.traveled, 3), e=round(e, 3), n=obs.get("n_obs"),
                      wall_ang=wa, ah_wall=(None if ah is None else bool(ah["wall"])),
                      ah_w=(None if ah is None else round(ah["lat_hi"] - ah["lat_lo"], 2)),
                      free_l=(None if ah is None else ah.get("free_l")), free_r=(None if ah is None else ah.get("free_r")),
                      veer=(None if veer is None else veer["stage"]))

    # ---- 横歩き・後退(オドメトリ。LiDAR では測れない)
    def _move_axis(self, axis, target, label):
        """axis='e'(横。+左) / 's'(経路に沿って。+前)。指令は残りに比例(下限 cmd_min・上限 v_side)。
        残りが tol 以下になったら止めて静止を待つ。行き過ぎても戻さない(往復しない)。
        4 秒送って 2cm も進まなければ中止。戻り値: 到達したか(横に障害物なら False)"""
        p = self.p
        tol = max(0.03, float(p["side_tol"]))
        od, obs = self._sense()
        s0, e0 = self._pose(od)
        x0 = s0 if axis == "s" else e0
        sgn = 1.0 if target >= x0 else -1.0
        t_limit = 15.0 + abs(target - x0) / 0.05
        t0 = time.time()
        t_prev = t0
        c = 0.0
        hist = []
        while True:
            time.sleep(0.1)
            now = time.time()
            dt = max(0.02, min(0.3, now - t_prev))
            t_prev = now
            od, obs = self._sense()
            s, e = self._pose(od)
            self.offset = e
            if axis == "s":
                self.traveled = self.traveled_base + s
            x = s if axis == "s" else e
            rem = (target - x) * sgn
            hist.append((now, x))
            if rem <= tol:
                self.phase = "STOPPING"
                self._stop_and_settle(label)
                return True
            if now - t0 > t_limit:
                self.io["vel"](0.0, 0.0, 0.0)
                raise _Abort(f"中止({self.phase}): 時間切れ({t_limit:.0f}秒、残り{rem * 100:+.0f}cm)")
            if axis == "e":
                sf = self._side_clear_m(obs, sgn)
                if sf is not None and sf < float(p["side_clear"]):
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 横{sf:.2f}mに障害物 — 止めます"
                    return False
            elif sgn > 0:
                d_front = obs.get("dist") if obs.get("ok") else None
                if d_front is not None and d_front < float(p["stop_dist"]) + 0.05:
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 前方{d_front:.2f}m(停止距離{p['stop_dist']:.2f}m) — 前へは出しません"
                    return False
            if c >= float(p["cmd_min"]) - 1e-6 and len(hist) >= 2 and now - hist[0][0] >= 4.0:
                x_old = [xx for tt, xx in hist if tt <= now - 4.0]
                if x_old and abs(x - x_old[-1]) < 0.02:
                    self.io["vel"](0.0, 0.0, 0.0)
                    raise _Abort(f"中止({self.phase}): 4秒送って進み{abs(x - x_old[-1]) * 100:.1f}cm — 歩行モードが"
                                 "速度指令に応じていない(十字キーで歩けるか確認。docs 自動歩行 §6b-15)")
            c_t = self._cmd_for(rem, float(p["v_side"]), float(p["k_side"]))
            c = self._slew(c, c_t, dt)
            cmd = sgn * c
            if axis == "e":
                self.io["vel"](0.0, cmd, self._om())
            else:
                self.io["vel"](cmd, 0.0, self._om())
            self.msg = f"{label}: 指令{cmd:+.2f} 残り{rem * 100:+.0f}cm"
            self._rec(c=round(cmd, 3), x=round(float(x), 3), rem=round(float(rem), 3))

    # ---- 微調整(10cm 以下): 1 歩ずつ
    NUDGE_MAX = 0.10

    def _step_axis(self, axis, target, label, single=False):
        """短い指令(cmd_min × step_on 秒)→ 止めて測る、を繰り返す。1歩の推定を実測で更新。
        3 歩で 1.5cm も進まなければ中止。single=True は 1 歩だけ"""
        p = self.p
        tol = max(0.03, float(p["side_tol"]))
        est = max(0.01, float(self.step_est))
        recent = []
        n = 0
        last_sgn = 0
        while True:
            od, obs = self._sense()
            s, e = self._pose(od)
            self.offset = e
            x = s if axis == "s" else e
            rem = target - x
            if abs(rem) <= tol and not single:
                self._hold(0.5, f"{label}: 到達(残り{rem * 100:+.0f}cm)")
                return True
            if n >= int(p.get("step_max", 20)):
                raise _Abort(f"中止({self.phase}): {n}歩で到達せず(残り{rem * 100:+.0f}cm)")
            sgn = 1.0 if rem > 0 else -1.0
            if n > 0 and sgn != last_sgn:
                self._hold(0.5, f"{label}: 半歩未満の残り{rem * 100:+.0f}cm は追いません")
                return True
            if axis == "e":
                sf = self._side_clear_m(obs, sgn)
                if sf is not None and sf < float(p["side_clear"]):
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 横{sf:.2f}mに障害物 — 止めます"
                    return False
            elif sgn > 0:                            # 前向きの 1 歩は停止距離を守る
                d_front = obs.get("dist") if obs.get("ok") else None
                if d_front is not None and d_front < float(p["stop_dist"]) + 0.05:
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.msg = f"{label}: 前方{d_front:.2f}m(停止距離{p['stop_dist']:.2f}m) — 前へは出しません"
                    return False
            frac = 1.0 if single else min(1.0, abs(rem) / est)
            t_on = max(0.3, float(p["step_on"]) * frac)
            v = sgn * float(p["cmd_min"])
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
                    self.io["vel"](v, 0.0, self._om())
                self.msg = f"{label}: {n}歩目 指令{v:+.2f}×{t_on:.1f}s 残り{(target - (s if axis == 's' else e)) * 100:+.0f}cm"
                self._rec(step=n, v=round(v, 3), t_on=round(t_on, 2), x=round(float(s if axis == "s" else e), 3))
            self._hold(float(p["step_off"]), f"{label}: {n}歩目の着地を待つ")
            od, obs = self._sense()
            s, e = self._pose(od)
            x = s if axis == "s" else e
            d = (x - xa) * sgn
            self.step_last = d
            self.steps = n
            recent = (recent + [d])[-3:]
            if d > 0.005:
                est = d if n == 1 else 0.6 * est + 0.4 * d
                self.step_est = est
            self.io["log"](f"{label}: {n}歩目 {d * 100:+.1f}cm(指令{v:+.2f}×{t_on:.1f}s) 残り{(target - x) * 100:+.1f}cm")
            if n >= 3 and sum(recent) < 0.015:
                self.io["vel"](0.0, 0.0, 0.0)
                raise _Abort(f"中止({self.phase}): 3歩で進み{sum(recent) * 100:.1f}cm — 歩行モードが"
                             "速度指令に応じていない(十字キーで歩けるか確認。docs 自動歩行 §6b-15)")
            if single:
                return True

    def _lateral_to(self, e_target, label):
        od, _obs = self._sense()
        _s, e = self._pose(od)
        if abs(e_target - e) <= self.NUDGE_MAX + 1e-6:
            return self._step_axis("e", e_target, label)
        return self._move_axis("e", e_target, label)

    def _back_to(self, dist, label):
        od, _obs = self._sense()
        s0, _e = self._pose(od)
        self.traveled_base = self.traveled
        if dist <= self.NUDGE_MAX + 1e-6:
            return self._step_axis("s", s0 - dist, label)
        return self._move_axis("s", s0 - dist, label)

    def _step_once(self, direction):
        od, _obs = self._sense()
        s0, e0 = self._pose(od)
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

    # ---- 壁への正対と追従
    def _rebase_path(self, od, yaw_ref_imu):
        s, _e = self._pose(od)
        self.traveled_base = self.traveled_base + s
        yaw_od = od[3] + _wrap(yaw_ref_imu - self.io["yaw"]())
        self._x0, self._y0 = od[1], od[2]
        self._fx, self._fy = math.cos(yaw_od), math.sin(yaw_od)
        self._lx, self._ly = -math.sin(yaw_od), math.cos(yaw_od)
        self._path_yaw_ref = yaw_ref_imu

    def _track_wall_heading(self, obs, od):
        """壁の面の角度 wall_ang を毎コマ見て、目標の向きを壁に垂直へ寄せる(ローパス 0.3)。
        5 度以上変わったら経路線も張り直す。壁幅の 8 割未満の面(箱・机)は追わない"""
        if not self.p.get("wall_track", True) or not obs.get("ok"):
            return
        ang = obs.get("wall_ang")
        wl = obs.get("wall_len") or 0.0
        wd = obs.get("wall_dist")
        if ang is None or wl < 0.8 * float(self.p["wall_width"]) or wd is None or wd > 5.0:
            return
        yaw_now = self.io["yaw"]()
        target = _wrap(yaw_now + math.radians(float(ang)))
        self._yaw_ref = _wrap(self._yaw_ref + 0.3 * _wrap(target - self._yaw_ref))
        if abs(_wrap(self._yaw_ref - self._path_yaw_ref)) > math.radians(5.0):
            self._rebase_path(od, self._yaw_ref)

    def _align_to_wall(self):
        """正面の壁が align_inplace_deg より斜めなら、その場で回転して正対する。戻り値: 回った角度[deg]"""
        p = self.p
        tol = float(p.get("align_tol_deg", 3.0))
        om_max = float(p.get("om_turn", 0.3))
        self._hold(0.4, "正対: 壁の向きを読んでいます")
        od, obs = self._sense()
        ang = obs.get("wall_ang") if obs.get("ok") else None
        if ang is None or abs(ang) <= max(tol, float(p.get("align_inplace_deg", 8.0))):
            return 0.0
        wd = obs.get("wall_dist")
        if wd is not None and wd < float(p["stop_dist"]) + 0.3:
            self.io["log"](f"正面の壁が {ang:+.1f}° 斜めですが {wd:.2f}m と近いので、その場では回しません")
            return 0.0
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
                if miss > 10:
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
            om = math.radians(ang) * 1.2
            om = max(-om_max, min(om_max, om))
            if abs(om) < 0.15:
                om = math.copysign(0.15, om)
            self.io["vel"](0.0, 0.0, om)
            self.msg = f"正対中: 壁の角度 {ang:+.1f}° 回転 {om:+.2f}rad/s"
            self._rec(om=round(om, 3), wall_ang=ang)
        self._hold(0.8, "正対: 静定")
        turned = math.degrees(_wrap(self.io["yaw"]() - yaw0))
        self.io["log"](f"正対しました(回転 {turned:+.1f}°、壁の角度 {ang if ang is None else round(ang, 1)}°)")
        return turned

    # ---- 全体
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
            f"指令 上限{p['v_fwd']:.2f}/下限{p['cmd_min']:.2f}m/s 停止距離{p['stop_dist']:.2f}m "
            f"横{'左' if sdir > 0 else '右'}{p['side_dist']:.2f}m 最大前進{p['max_fwd']:.1f}m "
            f"回り込み{'あり' if p.get('avoid', True) else 'なし'}  点群座標系={obs.get('frame')}")
        if mode == "step":
            self.phase = "STEP"
            self.result = "完了: " + self._step_once(p.get("step_dir", "left"))
            log(f"自動歩行 {self.result}")
            return
        if mode == "back":
            self.phase = "BACK"
            od0, _o = self._sense()
            s_a, _e = self._pose(od0)
            ok = self._back_to(float(p.get("back_dist", 0.05)), "後退")
            od1, _o = self._sense()
            s_b, _e = self._pose(od1)
            moved = s_a - s_b
            self.result = (f"完了: 後ろへ{moved:.2f}m下がりました" if ok
                           else f"中止(BACK): 後退{moved:.2f}m")
            log(f"自動歩行 {self.result}")
            return
        if mode == "side":
            self.phase = "SIDE"
            ok = self._lateral_to(sdir * p["side_dist"], f"横歩き({'左' if sdir > 0 else '右'})")
            self.side_traveled = sdir * self.offset
            if not ok:
                self.result = f"中止(SIDE): 横方向に障害物(横移動{self.side_traveled:.2f}m/{p['side_dist']:.2f}m)"
                log(f"★自動歩行 {self.result}")
                return
            self.result = f"完了: {'左' if sdir > 0 else '右'}へ{self.side_traveled:.2f}m横移動"
            log(f"自動歩行 {self.result}")
            return
        if p.get("align_wall", True):
            turned = self._align_to_wall()
            if abs(turned) > 0.5:
                od, obs = self._sense()
                self._set_path(od)
        how = self._forward()
        if how == "max":
            self.result = f"完了(壁なし): 最大前進距離{p['max_fwd']:.1f}mに到達"
            log(f"自動歩行 {self.result}")
            return
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


class WalkController:
    """コックピット(Engine)から見た唯一の入口。

    robot に要るもの:
      set_velocity(vx, vy, om, duration) -> bool   内蔵歩行への速度指令(同期・有限時間)
      open_lidar() / open_odom()                    .latest()/.enable()/.stop() を持つ読み手
      yaw() -> rad                                  IMUヨー(LowState)
      state() -> (q, dq, quat, gyro, tau)           傾きの計算用
      ensure_walk_mode(log) -> (ok, fsm)            歩行FSMへ
    """

    RANGES = {"v_fwd": (0.3, 0.9), "v_side": (0.3, 0.6), "cmd_min": (0.15, 0.5), "k_dist": (0.3, 2.0),
              "k_side": (0.3, 2.0), "stop_lead": (0.0, 0.6), "slew_up": (0.2, 2.0), "slew_down": (0.2, 2.0),
              "cmd_dur": (0.2, 1.0), "settle_s": (0.3, 3.0), "settle_max": (2.0, 15.0),
              "stop_dist": (0.3, 2.5), "side_dist": (0.02, 3.0), "max_fwd": (0.3, 10.0), "back_dist": (0.02, 0.5),
              "step_on": (0.2, 1.5), "step_off": (0.3, 2.0), "step_max": (1, 60),
              "tele_vx": (0.3, 0.9), "tele_vy": (0.3, 0.6), "tele_om": (0.1, 0.8),
              "half_w": (0.2, 0.6), "h_min": (0.05, 0.5), "h_max": (0.5, 2.5), "side_clear": (0.2, 1.5),
              "self_fwd": (0.1, 0.8), "self_lat": (0.2, 0.8), "yaw_fix_deg": (-360.0, 360.0), "front_offset": (-0.3, 0.5),
              "align_tol_deg": (1.0, 15.0), "align_inplace_deg": (2.0, 45.0), "om_turn": (0.1, 0.6),
              "wall_width": (0.8, 3.0), "detour_margin": (0.02, 0.4), "body_half": (0.15, 0.45), "detour_max": (0.3, 1.5),
              "veer_v": (0.3, 0.9), "side_tol": (0.01, 0.1)}

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
            elif k in ("dry_run", "avoid", "align_wall", "wall_track", "yaw_autocal"):
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
        """歩行 FSM へ入れ(4 → 501、だめなら 200)、静止立位モード(SetBalanceMode 0)にし、センサ読み取りを始める。
        ワーカースレッドで呼ぶこと(RPC)。"""
        ok, fsm = self.robot.ensure_walk_mode(log=self.log)
        self.fsm_id = fsm
        self.ready = bool(ok)
        if ok:
            self.enable_sensors(True)
            self.log(f"歩行モード(FSM {fsm})。[前進][横歩き]、十字キーの手動操作が押せます")
        else:
            self.log(f"★歩行モードへ入れませんでした({fsm})")
        return ok

    def tele(self, vx, vy, om):
        """手動操作(押している間だけ届く)。上限で切って送信スレッドへ渡す。
        ★指令は cmd_min(0.30)未満だと足踏みだけになるので、押されたら下限以上を出す"""
        if not self.ready:
            return False
        if self.auto is not None and not self.auto.done:
            return False                           # 自動歩行中は手動を受けない
        p = self.params
        cm = float(p["cmd_min"])

        def shape(v, lim):
            v = float(np.clip(v, -lim, lim))
            return 0.0 if abs(v) < 0.02 else math.copysign(max(cm, abs(v)), v)
        vx = shape(vx, p["tele_vx"])
        vy = shape(vy, p["tele_vy"])
        om = float(np.clip(om, -p["tele_om"], p["tele_om"]))
        self._tele_t = time.time()
        self.sender.set(vx, vy, om, "tele", dry=bool(p.get("dry_run")))
        return True

    def start_auto(self, overrides=None):
        """自動歩行を始める。overrides でこの1回だけのパラメータ(mode, side_dist 等)を上書き"""
        if not self.ready:
            self.log("★先に[歩行モードへ]を押してください")
            return False
        if self.auto is not None and not self.auto.done:
            self.log("★自動歩行はすでに実行中です")
            return False
        try:
            f_now = self.robot.get_fsm_id()
        except Exception:                          # noqa: BLE001
            f_now = None
        if f_now is not None and f_now not in WALK_FSMS:
            self.log(f"歩行 FSM に居ない(FSM {f_now})ので、歩行モードへ入れ直します")
            if not self.prepare():
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
        io = dict(obstacle=self.obstacle,
                  fsm_mode=(lambda: getattr(self.robot, "get_fsm_mode", lambda: None)()),
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
        self.sender.duration = float(self.params.get("cmd_dur", 0.3))

        self.auto = AutoWalk(params, io, log_path=lp)
        self.auto.start()
        return True

    def stop(self, why="停止"):
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
            "side_fwd_l": obs.get("side_fwd_l"), "side_fwd_r": obs.get("side_fwd_r"),
            "rear_n": obs.get("rear_n"), "rear_dist": obs.get("rear_dist"),
            "rear_h": obs.get("rear_h"),
            "frame": obs.get("frame_id", obs.get("frame")),
            "floor_ok": obs.get("floor_ok"), "floor_h": obs.get("floor_h"),
            "why": obs.get("why", ""),
            "wall_dist": obs.get("wall_dist"), "wall_ang": obs.get("wall_ang"), "wall_len": obs.get("wall_len"),
            "dirs": obs.get("dirs"),
            "yaw_fix_deg": float(self.params.get("yaw_fix_deg", 0.0)),
            "stop_info": (a.stop_info if a is not None else None),
            "v_meas": (round(a.v_meas, 2) if a is not None else None),
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
