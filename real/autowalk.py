#!/usr/bin/env python3
"""自動歩行(前進 → 壁の手前で自然に停止 / 障害物は回り込んで元の経路へ / 横歩きは
細かい足踏み)と、スマホからの手動操作(押している間だけ動く)のための速度送信。

★方策(rt/lowcmd)は一切使わない。Unitree内蔵の歩行制御(FSM 500/501/801/802)に
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
import math
import pathlib
import threading
import time

import numpy as np

# ---------------------------------------------------------------- 既定値
WALK_DEFAULTS = dict(
    v_fwd=0.35,        # 前進の巡航速度[m/s]。G1の歩行は0.3〜0.9で安定。まずは遅く
    v_side=0.15,       # 横歩き(足踏みパルス)の速度[m/s]。横は前進より不安定なので低め
    v_creep=0.12,      # 忍び足の最低速度[m/s]。★内蔵歩行は 0.1m/s 未満では歩かない(2026-09-04 実測の疑い)
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
    avoid=True,        # 壁以外の障害物は回り込んで避ける
    wall_width=1.4,    # 前方の物体の横幅がこれ以上なら「壁」(回り込まない)[m]
    detour_margin=0.15,  # 回り込みで物体の端から取る余裕[m]
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
    step_v=0.20,       # 1歩の指令速度[m/s](内蔵歩行の不感帯より上)
    step_on=0.6,       # 1歩の指令時間[s]
    step_off=1.0,      # 止めて着地と計測を待つ時間[s]
    step_min_on=0.3,   # 残りが小さいときの最短指令時間[s]
    step_est=0.06,     # 1歩の推定移動量[m]。実測で更新
    step_max=20,       # 1回の指示で出す最大歩数
    step_dir="left",   # mode="step"(1歩だけ)の向き: left/right/back/fwd
)
WALK_FSMS = {500, 501, 801, 802}   # 速度指令を受ける内蔵FSM(新FW。旧FWは200)
YAW_KP = 1.6                       # 直進保持のゲイン[(rad/s)/rad](旧コックピット実績値)
YAW_OM_MAX = 0.30                  # 直進保持の補正上限[rad/s]
LAT_KP = 0.6                       # 経路線への横ずれ補正[(m/s)/m]
LAT_VY_MAX = 0.08                  # 同・上限[m/s]
ACC_UP = 0.30                      # 前進の加速上限[m/s²]
ACC_DOWN = 0.60                    # 前進の減速上限[m/s²]
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


def stage_speed(v_max, d_rem, v_creep=0.12):
    """壁までの残り距離(停止距離を引いた値)に応じた前進速度。段階的に落として忍び足で止まる"""
    if d_rem is None:
        return v_max
    if d_rem <= 0.0:
        return 0.0
    if d_rem < 0.25:
        return max(v_creep, 0.15 * v_max)
    if d_rem < 0.6:
        return max(v_creep, 0.35 * v_max)
    if d_rem < 1.0:
        return 0.55 * v_max
    if d_rem < 1.5:
        return 0.80 * v_max
    return v_max


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
            fwd = pts[:, 0] + SENSOR_FWD_OFFSET
            lat = pts[:, 1]
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
                depth = max(0.0, float(np.percentile(fwd[mc], 95)) - d0)
                wall = ((hi - lo) > cfg["wall_width"]) or (lo < -1.0 and hi > 1.0)
                mgn = cfg["detour_margin"]

                def free(e_lo, e_hi):
                    mb = (hm & (fwd > d0 - 0.5) & (fwd < d0 + depth + 0.8)
                          & (lat > e_lo) & (lat < e_hi))
                    return int(mb.sum()) < 6
                eL = hi + hw + mgn
                eR = lo - hw - mgn
                free_l = eL if (not wall and eL <= cfg["detour_max"]
                                and free(hi + 0.03, eL + hw)) else None
                free_r = eR if (not wall and -eR <= cfg["detour_max"]
                                and free(eR - hw, lo - 0.03)) else None
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
        out["ok"] = True
        return out


# ---------------------------------------------------------------- 速度送信
class VelSender:
    """10Hzで速度を送る唯一のスレッド。指令は CMD_HOLD_S 秒で失効しゼロを送る。"""

    def __init__(self, send_fn, log=print):
        self._send = send_fn         # send_fn(vx, vy, om, duration) -> bool
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
                        ok = self._send(0.0, 0.0, 0.0, CMD_HOLD_S)
                        self.n_sent += 1
                        if not ok:
                            self.n_fail += 1
                    self.last_sent = (0.0, 0.0, 0.0)
                    self.last_sent_t = time.time()
                    with self._lock:
                        self._need_stop = max(0, self._need_stop - 1)
                elif moving:
                    if not dry:
                        ok = self._send(cmd[0], cmd[1], cmd[2], CMD_HOLD_S)
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
            self.done = True
            self._flush()

    # ---- 経路線とセンサ
    def _set_path(self, od):
        self._x0, self._y0 = od[1], od[2]
        yaw_od = od[3]
        self._fx, self._fy = math.cos(yaw_od), math.sin(yaw_od)
        self._lx, self._ly = -math.sin(yaw_od), math.cos(yaw_od)
        self._yaw_ref = self.io["yaw"]()

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
        yaw = self.io["yaw"]()
        return float(np.clip(-YAW_KP * _wrap(yaw - self._yaw_ref), -YAW_OM_MAX, YAW_OM_MAX))

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
            v = sgn * float(p["step_v"])
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
            if d > 0.005:
                est = d if n == 1 else 0.6 * est + 0.4 * d
                self.step_est = est
            self.io["log"](f"{label}: {n}歩目 {d * 100:+.1f}cm(指令{v:+.2f}m/s×{t_on:.1f}s)"
                           f" 残り{(target - x) * 100:+.1f}cm 1歩の推定{est * 100:.1f}cm")
            if n >= 3 and net < 0.015 and best < 0.01:
                self.io["vel"](0.0, 0.0, 0.0)
                raise _Abort(f"中止({self.phase}): 3歩で進み{net * 100:.1f}cm — 歩行モードが"
                             "速度指令に応じていない(十字キーで歩けるか確認。docs 自動歩行 §6b-3)")
            if single:
                return True

    def _lateral_to(self, e_target, label):
        """経路線からのずれ e を e_target へ(小刻みステップ)。到達したか(横に障害物なら False)"""
        return self._step_axis("e", e_target, label)

    def _back_to(self, dist, label):
        """後退(椅子との距離を詰める)。経路に沿って dist だけ下がる(小刻みステップ)。
        後ろは LiDAR が見えないので操作者が目で見る前提。"""
        od, _obs = self._sense()
        s0, _e = self._pose(od)
        self.traveled_base = self.traveled
        return self._step_axis("s", s0 - dist, label)

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

    def _forward(self):
        """壁の手前(stop_dist)で止まるか、最大前進距離で止まるまで進む。
        戻り値: "wall"(壁の手前で停止) / "max"(壁なしで最大距離)"""
        p = self.p
        self.phase = "FORWARD"
        self.v = 0.0
        hit = 0
        t_phase = time.time()
        t_prev = t_phase
        t_limit = p["max_fwd"] / max(p["v_fwd"], 0.05) * 2.5 + 20.0
        while True:
            time.sleep(0.1)
            now = time.time()
            dt = max(0.02, min(0.3, now - t_prev))
            t_prev = now
            od, obs = self._sense()
            if now - t_phase > t_limit:
                raise _Abort(f"中止(FORWARD): 時間切れ({t_limit:.0f}秒)")
            s, e = self._pose(od)
            self.traveled, self.offset = s, e
            if not obs.get("ok"):
                # 判定できない(床が取れない等)= 見えていない。安全側で止まって待つ
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                self.msg = f"待機: 障害物の判定不能({obs.get('why')})"
                self._rec(vx=0, dist=None, s=round(s, 3))
                continue
            dist = obs.get("dist")
            ah = obs.get("ahead")
            # --- 回り込み: 壁でない物体が近づいたら、空いている側へ避けて元の経路へ戻る
            if (p.get("avoid", True) and dist is not None and ah is not None
                    and not ah["wall"] and self.detours < MAX_DETOURS
                    and dist <= max(p["stop_dist"] + 0.5, 0.9)):
                cands = [c for c in (ah.get("free_l"), ah.get("free_r")) if c is not None]
                if cands:
                    e_t = min(cands, key=abs)
                    self.io["vel"](0.0, 0.0, 0.0)
                    self.v = 0.0
                    self.io["log"](f"障害物 {dist:.2f}m(横幅{ah['lat_hi'] - ah['lat_lo']:.2f}m"
                                   f"・奥行{ah['depth']:.2f}m) — "
                                   f"{'左' if e_t > 0 else '右'}へ{abs(e_t):.2f}m回り込みます")
                    self._detour(e_t, s + dist + ah["depth"])
                    hit = 0
                    t_prev = time.time()
                    continue
            d_rem = None if dist is None else dist - p["stop_dist"]
            near = d_rem is not None and d_rem <= 0.03
            hit = hit + 1 if near else 0
            if hit >= 2:
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                self.io["log"](f"{'壁' if (ah and ah['wall']) else '障害物'} {dist:.2f}m"
                               f"(停止距離{p['stop_dist']:.2f}m) — 停止します  前進{s:.2f}m")
                return "wall"
            if s >= p["max_fwd"]:
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                return "max"
            # --- 段階的な速度 + 加減速の上限で滑らかに
            v_t = stage_speed(p["v_fwd"], d_rem, p["v_creep"])
            if v_t > self.v:
                self.v = min(v_t, self.v + ACC_UP * dt)
            else:
                self.v = max(v_t, self.v - ACC_DOWN * dt)
            vy = float(np.clip(-LAT_KP * e, -LAT_VY_MAX, LAT_VY_MAX)) if self.v > 0.05 else 0.0
            om = self._om() if self.v > 0.03 else 0.0
            self.io["vel"](self.v, vy, om)
            self.msg = (f"前進 v={self.v:.2f} 横補正{vy:+.2f} 補正{om:+.2f} | 前方 "
                        f"{'---' if dist is None else f'{dist:.2f}m'}"
                        f"{'(壁)' if (ah and ah['wall']) else ''} | {s:.2f}m ずれ{e * 100:+.0f}cm")
            self._rec(vx=round(self.v, 3), vy=round(vy, 3), om=round(om, 3), dist=dist,
                      s=round(s, 3), e=round(e, 3), n=obs.get("n_obs"))

    def _detour(self, e_t, s_end):
        """障害物の横へ出て(DETOUR_OUT) → 奥まで進み(DETOUR_PASS) → 経路線へ戻る(DETOUR_BACK)"""
        p = self.p
        self.phase = "DETOUR_OUT"
        if not self._lateral_to(e_t, "回り込み(横へ)"):
            raise _Abort("中止(DETOUR_OUT): 回り込む側に障害物")
        self.phase = "DETOUR_PASS"
        s_goal = s_end + 0.6
        t0 = time.time()
        t_limit = (s_goal - self.traveled) / max(0.6 * p["v_fwd"], 0.05) * 3.0 + 15.0
        self.v = 0.0
        t_prev = time.time()
        while True:
            time.sleep(0.1)
            now = time.time()
            dt = max(0.02, min(0.3, now - t_prev))
            t_prev = now
            od, obs = self._sense()
            s, e = self._pose(od)
            self.traveled, self.offset = s, e
            if s >= s_goal:
                self.io["vel"](0.0, 0.0, 0.0)
                self.v = 0.0
                break
            if now - t0 > t_limit:
                raise _Abort("中止(DETOUR_PASS): 時間切れ")
            dist = obs.get("dist") if obs.get("ok") else None
            if dist is not None and dist <= p["stop_dist"] + 0.05:
                self.io["vel"](0.0, 0.0, 0.0)
                raise _Abort(f"中止(DETOUR_PASS): 回り込み先にも障害物 {dist:.2f}m")
            v_t = min(0.6 * p["v_fwd"], stage_speed(p["v_fwd"], None if dist is None
                                                    else dist - p["stop_dist"], p["v_creep"]))
            self.v = min(v_t, self.v + ACC_UP * dt) if v_t > self.v else max(v_t, self.v - ACC_DOWN * dt)
            vy = float(np.clip(-LAT_KP * (e - e_t), -LAT_VY_MAX, LAT_VY_MAX))
            self.io["vel"](self.v, vy, self._om())
            self.msg = f"回り込み(通過) v={self.v:.2f} | {s:.2f}/{s_goal:.2f}m ずれ{e * 100:+.0f}cm"
            self._rec(vx=round(self.v, 3), vy=round(vy, 3), s=round(s, 3), e=round(e, 3), dist=dist)
        self.phase = "DETOUR_BACK"
        if not self._lateral_to(0.0, "回り込み(経路へ戻る)"):
            raise _Abort("中止(DETOUR_BACK): 戻る側に障害物")
        self.detours += 1
        self.io["log"](f"回り込み完了({self.detours}回目) — 元の経路へ戻りました(ずれ{self.offset * 100:+.0f}cm)")
        self.phase = "FORWARD"

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
            if not ok:
                self.result = f"中止(SIDE): 横方向に障害物(横移動{self.side_traveled:.2f}m/{p['side_dist']:.2f}m)"
                log(f"★自動歩行 {self.result}")
                return
            self.result = f"完了: {'左' if sdir > 0 else '右'}へ{self.side_traveled:.2f}m横移動"
            log(f"自動歩行 {self.result}")
            return
        how = self._forward()
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

    RANGES = {"v_fwd": (0.05, 0.9), "v_side": (0.05, 0.4), "v_creep": (0.1, 0.25),
              "stop_dist": (0.3, 2.5), "side_dist": (0.02, 3.0),
              "max_fwd": (0.3, 10.0), "half_w": (0.2, 0.6), "h_min": (0.05, 0.5),
              "h_max": (0.5, 2.5), "side_clear": (0.2, 1.5), "wall_width": (0.8, 3.0),
              "detour_margin": (0.05, 0.4), "detour_max": (0.3, 1.5),
              "side_tol": (0.01, 0.1), "pulse_on": (0.2, 1.0), "pulse_off": (0.2, 1.0),
              "step_v": (0.1, 0.4), "step_on": (0.2, 1.5), "step_off": (0.3, 2.0),
              "step_min_on": (0.15, 0.8), "step_est": (0.01, 0.3), "step_max": (1, 60),
              "back_dist": (0.02, 0.5), "self_fwd": (0.1, 0.8), "self_lat": (0.2, 0.8),
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
            elif k in ("dry_run", "avoid"):
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
                odxy = (od[1], od[2]) if od is not None else None
                sdir = 1 if self.params.get("side_dir", "left") == "left" else -1
                r = self.det.update(pts, frame, odxy, self.robot.yaw(), sdir)
                r["age_ms"] = round((time.time() - t) * 1000.0)
                r["frame_id"] = frame
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
            self.enable_sensors(True)
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
        p = self.params
        vx = float(np.clip(vx, -p["tele_vx"] * 0.6, p["tele_vx"]))   # 後退は6割
        vy = float(np.clip(vy, -p["tele_vy"], p["tele_vy"]))
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
            "rear_n": obs.get("rear_n"), "rear_dist": obs.get("rear_dist"),
            "rear_h": obs.get("rear_h"),
            "frame": obs.get("frame_id", obs.get("frame")),
            "floor_ok": obs.get("floor_ok"), "floor_h": obs.get("floor_h"),
            "why": obs.get("why", ""),
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
