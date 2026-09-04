#!/usr/bin/env python3
"""Livox Mid-360 → 点群を /dev/shm へ 10Hz で書く(機体上で動かす。コックピットが読む)。

この機体(FW 1.5.3.8)では Unitree 側の LiDAR 配信(rt/utlidar/*)が無い(2026-09-04 実測:
DDS 91トピック中ゼロ、ドライバのプロセスも無し)。Mid-360 自体は 192.168.123.120 に居て、
機体には Livox SDK2 の共有ライブラリ(/usr/local/lib/liblivox_lidar_sdk_shared.so)が
入っているので、それを ctypes で直接叩いて点群を受ける。

  python3 real/lidar_bridge.py                  # 常駐(start_onboard.sh が起動する)
  python3 real/lidar_bridge.py --seconds 10     # 10秒だけ動かして統計を出す(確認用)
  python3 real/lidar_bridge.py --raw            # 取り付け補正をせず生のセンサ座標で書く

出力(10Hz、原子的に差し替え):
  /dev/shm/g1_lidar.npy   float32 (N,4) = x,y,z[m], reflectivity。座標は下記の「体水平系」
  /dev/shm/g1_lidar.json  {"t":受信時刻, "seq":連番, "n":点数, "frame_id":"livox_level",
                           "hz":生パケット/秒, "mount":{...推定した取り付け...}}

座標系(frame_id="livox_level"): 起動後の最初の数秒で床平面を推定して
  z = 床法線(上)、x = 床が見えている方向(=センサが前下がりに傾いている向き = 機体前方)
  になるように回す。原点はセンサ。ヨーの向きが逆のときは lidar_mount.json の
  yaw_offset_deg に 180 を書けば反転する(コックピットの「前方」の距離が実物と合うかで判断)。
  床が推定できないうちは frame_id="livox_frame"(生のセンサ座標)で書く。

★受信はSDKのCスレッドから Python コールバックに入る(1パケット=96点、約2000回/秒)。
  コールバックでは bytes をコピーして積むだけにし、変換と書き出しは別スレッドで 10Hz。
"""
import argparse
import ctypes
import json
import math
import os
import socket
import struct
import threading
import time

import numpy as np

SDK_SO = "/usr/local/lib/liblivox_lidar_sdk_shared.so"
SHM_NPY = "/dev/shm/g1_lidar.npy"
SHM_JSON = "/dev/shm/g1_lidar.json"
NO_DATA_EXIT_S = 60.0        # 常駐時、この秒数点群が来なければ終了(systemd が立て直す)
CFG_PATH = "/tmp/livox_mid360_cfg.json"
MOUNT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_mount.json")
LIDAR_IP = "192.168.123.120"
HDR = struct.Struct("<BHHHHBBB12sI8s")          # LivoxLidarEthernetPacket の先頭36バイト(pack 1)
PT_HIGH = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("refl", "u1"), ("tag", "u1")])
PT_LOW = np.dtype([("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("refl", "u1"), ("tag", "u1")])


def host_ip_for(lidar_ip):
    """LiDAR と同じサブネットの自分のIP(eth0)を求める"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((lidar_ip, 9))
        return s.getsockname()[0]
    finally:
        s.close()


def write_cfg(host_ip):
    cfg = {
        "lidar_summary_info": {"lidar_type": 8},
        "MID360": {
            "lidar_net_info": {"cmd_data_port": 56100, "push_msg_port": 56200,
                               "point_data_port": 56300, "imu_data_port": 56400,
                               "log_data_port": 56500},
            "host_net_info": {"cmd_data_ip": host_ip, "cmd_data_port": 56101,
                              "push_msg_ip": host_ip, "push_msg_port": 56201,
                              "point_data_ip": host_ip, "point_data_port": 56301,
                              "imu_data_ip": host_ip, "imu_data_port": 56401,
                              "log_data_ip": "", "log_data_port": 56501},
        },
        "lidar_configs": [{"ip": LIDAR_IP, "pcl_data_type": 1, "pattern_mode": 0,
                           "extrinsic_parameter": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                                                   "x": 0, "y": 0, "z": 0}}],
    }
    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, indent=1)
    return CFG_PATH


class Mount:
    """取り付け補正: 床平面から z(上) と x(前=床が見える向き) を決める回転"""

    def __init__(self):
        self.R = None                 # (3,3) raw → level
        self.info = {}
        self.yaw_offset = 0.0
        try:
            with open(MOUNT_PATH) as f:
                self.yaw_offset = math.radians(float(json.load(f).get("yaw_offset_deg", 0.0)))
        except Exception:                          # noqa: BLE001
            pass

    def estimate(self, P):
        """P (N,3) 生座標。RANSACで最大平面(=床)を探し、原点が上側になる法線を取る。
        床の点の方位の平均を前方にする。成功したら True"""
        r = np.linalg.norm(P, axis=1)
        Q = P[(r > 0.4) & (r < 5.0)]
        if len(Q) < 800:
            return False
        rng = np.random.default_rng(0)
        best_n, best_c, best_cnt = None, 0.0, 0
        idx = rng.integers(0, len(Q), size=(400, 3))
        for a, b, c in idx:
            p0, p1, p2 = Q[a], Q[b], Q[c]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n /= nn
            cc = -float(n @ p0)
            if n @ np.zeros(3) + cc < 0:          # 原点が正側(=法線がセンサへ向く=上)になるように
                n, cc = -n, -cc
            dist = np.abs(Q @ n + cc)
            cnt = int((dist < 0.03).sum())
            if cnt > best_cnt:
                best_n, best_c, best_cnt = n, cc, cnt
        if best_n is None or best_cnt < 0.10 * len(Q):
            return False
        # 精密化(内点でLSQ)
        inl = Q[np.abs(Q @ best_n + best_c) < 0.03]
        cen = inl.mean(axis=0)
        u, s, vt = np.linalg.svd(inl - cen, full_matrices=False)
        n = vt[2]
        if n @ (-cen) < 0:                        # 原点(センサ)が平面より上
            n = -n
        h = float(abs(n @ cen))                   # センサの床からの高さ
        if not (0.4 < h < 2.0):
            return False
        # 前方 = 床の点の水平方向の平均
        horiz = inl - np.outer(inl @ n, n)
        fwd = horiz.mean(axis=0)
        fwd -= n * (fwd @ n)
        if np.linalg.norm(fwd) < 0.05:
            return False
        fwd /= np.linalg.norm(fwd)
        # ヨーの上書き(lidar_mount.json)
        if abs(self.yaw_offset) > 1e-6:
            c, s_ = math.cos(self.yaw_offset), math.sin(self.yaw_offset)
            left = np.cross(n, fwd)
            fwd = c * fwd + s_ * left
        left = np.cross(n, fwd)
        self.R = np.stack([fwd, left, n])         # 行 = 新しい軸(旧座標で表現)
        tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
        self.info = dict(height=round(h, 3), tilt_deg=round(tilt, 1),
                         fwd_raw=[round(float(v), 3) for v in fwd],
                         up_raw=[round(float(v), 3) for v in n],
                         n_floor=int(len(inl)), yaw_offset_deg=round(math.degrees(self.yaw_offset), 1))
        return True

    def apply(self, P):
        return P @ self.R.T


class Bridge:
    def __init__(self, raw=False, verbose=True):
        self.raw = raw
        self.verbose = verbose
        self.lock = threading.Lock()
        self.chunks = []              # [(data_type, bytes)]
        self.n_pkt = 0
        self.n_pts = 0
        self.seq = 0
        self.mount = Mount()
        self.mount_ok = False
        self._stop = False
        self.last_write = 0.0
        self.last_n = 0
        self.hz = 0.0
        self.lib = ctypes.CDLL(SDK_SO)
        CB = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_void_p)
        self._cb = CB(self._on_packet)             # ★参照を保持(GCされると落ちる)
        self.lib.LivoxLidarSdkInit.restype = ctypes.c_bool
        self.lib.LivoxLidarSdkInit.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
        self.lib.LivoxLidarSdkStart.restype = ctypes.c_bool
        self.lib.SetLivoxLidarPointCloudCallBack.argtypes = [CB, ctypes.c_void_p]

    def _on_packet(self, handle, dev_type, data, client):
        try:
            hdr = ctypes.string_at(data, HDR.size)
            (_ver, _length, _ti, dot_num, _udp, _fc, dtype, _tt, _rsvd, _crc, _ts) = HDR.unpack(hdr)
            if dtype == 1:
                n = dot_num * PT_HIGH.itemsize
            elif dtype == 2:
                n = dot_num * PT_LOW.itemsize
            else:
                return
            buf = ctypes.string_at(data + HDR.size, n)
            with self.lock:
                self.chunks.append((dtype, buf))
                self.n_pkt += 1
                self.n_pts += dot_num
        except Exception:                          # noqa: BLE001
            pass

    def start(self, host_ip):
        cfg = write_cfg(host_ip)
        # SDK のコンソールログ(Receive Command … を毎秒)を止める。常駐だとログが 1日 20MB 以上育つ
        try:
            self.lib.DisableLivoxSdkConsoleLogger.restype = None
            self.lib.DisableLivoxSdkConsoleLogger()
        except Exception:                      # noqa: BLE001  (古い SDK には無い)
            pass
        ok = self.lib.LivoxLidarSdkInit(cfg.encode(), b"", None)
        if not ok:
            raise RuntimeError("LivoxLidarSdkInit failed")
        self.lib.SetLivoxLidarPointCloudCallBack(self._cb, None)
        if not self.lib.LivoxLidarSdkStart():
            raise RuntimeError("LivoxLidarSdkStart failed")

    def frame(self):
        """溜まったパケットを (N,4) float32 [x,y,z,refl] にして返す(生座標、m)"""
        with self.lock:
            ch, self.chunks = self.chunks, []
        if not ch:
            return None
        parts = []
        for dtype, buf in ch:
            if dtype == 1:
                p = np.frombuffer(buf, dtype=PT_HIGH)
                xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float32) * 0.001
            else:
                p = np.frombuffer(buf, dtype=PT_LOW)
                xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float32) * 0.01
            parts.append(np.concatenate([xyz, p["refl"].astype(np.float32)[:, None]], 1))
        P = np.concatenate(parts)
        P = P[np.all(np.isfinite(P[:, :3]), axis=1)]
        P = P[np.linalg.norm(P[:, :3], axis=1) > 0.05]       # 0点(無効)を除く
        return P

    def run(self, seconds=None):
        t0 = time.time()
        t_prev = t0
        t_good = t0                   # 最後に点群が来た時刻
        n_prev = 0
        acc = []                      # 取り付け推定用に最初の数コマを溜める
        while not self._stop:
            time.sleep(0.1)
            P = self.frame()
            now = time.time()
            if now - t_prev >= 1.0:
                self.hz = (self.n_pkt - n_prev) / (now - t_prev)
                n_prev, t_prev = self.n_pkt, now
            if P is None or len(P) < 10:
                if seconds and now - t0 > seconds:
                    break
                # ★常駐(systemd)のとき: 60秒点群が来なければ終了して立て直してもらう。
                #   LiDAR の電源が後から入った/一度切れた場合に SDK の接続が戻らないことがある
                if not seconds and now - t_good > NO_DATA_EXIT_S:
                    print(f"{NO_DATA_EXIT_S:.0f}秒点群なし(packets={self.n_pkt}) → 終了して再起動を待つ", flush=True)
                    self.exit_code = 2
                    break
                continue
            t_good = now
            frame_id = "livox_frame"
            if not self.raw:
                if not self.mount_ok:
                    acc.append(P[:, :3])
                    if sum(len(a) for a in acc) > 40000 or now - t0 > 3.0:
                        if self.mount.estimate(np.concatenate(acc)):
                            self.mount_ok = True
                            if self.verbose:
                                print(f"取り付け推定: {self.mount.info}", flush=True)
                        acc = acc[-3:]
                if self.mount_ok:
                    P = np.concatenate([self.mount.apply(P[:, :3]), P[:, 3:4]], 1).astype(np.float32)
                    frame_id = "livox_level"
            tmp = SHM_NPY + ".tmp.npy"
            np.save(tmp, P)
            os.replace(tmp, SHM_NPY)
            self.seq += 1
            meta = dict(t=now, seq=self.seq, n=int(len(P)), frame_id=frame_id,
                        hz=round(self.hz, 1), mount=self.mount.info if self.mount_ok else None)
            tmpj = SHM_JSON + ".tmp"
            with open(tmpj, "w") as f:
                json.dump(meta, f)
            os.replace(tmpj, SHM_JSON)
            self.last_write, self.last_n = now, len(P)
            if self.verbose and self.seq % 600 == 1:      # 約1分に1行
                print(f"seq {self.seq} n={len(P)} pkt/s={self.hz:.0f} frame={frame_id}", flush=True)
            if seconds and now - t0 > seconds:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0, help="この秒数で終了(確認用)")
    ap.add_argument("--raw", action="store_true", help="取り付け補正をしない")
    ap.add_argument("--host-ip", default="", help="自分のIP(既定: LiDARへの経路から自動)")
    a = ap.parse_args()
    host = a.host_ip or host_ip_for(LIDAR_IP)
    print(f"host_ip={host} lidar={LIDAR_IP} sdk={SDK_SO}", flush=True)
    b = Bridge(raw=a.raw)
    b.start(host)
    print("SDK started. waiting for points...", flush=True)
    try:
        b.run(seconds=a.seconds or None)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            b.lib.LivoxLidarSdkUninit()
        except Exception:                          # noqa: BLE001
            pass
    print(f"終了: packets={b.n_pkt} points={b.n_pts} seq={b.seq} last_n={b.last_n} "
          f"mount={'OK ' + json.dumps(b.mount.info) if b.mount_ok else '未推定'}", flush=True)
    return getattr(b, "exit_code", 0) or (0 if b.n_pkt > 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
