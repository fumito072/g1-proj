#!/usr/bin/env python3
"""実機ミラー(デジタルツイン)。実機の姿勢をMuJoCoモデルに反映して見せる。

  python3 real/twin.py --iface enp46s0          # ライブ(実機と並走)
  python3 real/twin.py --replay <ログのnpz>      # 走行後の再生(実機不要)

ブラウザで http://localhost:8091 。コックピット(8090)の画面にも埋め込まれる。

★**このプロセスは1バイトも送信しない。**rt/lowstate を購読するだけ。
  コックピットの /state も読むが、GETのみ。

--- なぜ別プロセスなのか(ここは変えないこと) -------------------------------
コックピット本体は 50Hz制御ループ + 500Hz送信スレッドを抱えている。
そこへMuJoCoの描画(OpenGL。GILを握る)を同居させると、送信が途切れうる。
実績として、同一プロセスで rt/lowcmd を publish しながら subscribe して
cyclonedds がデッドロックしている(2026-08-20)。DDSは複数プロセスから
購読できる(watch_live.py が同じ方式)ので、描画は完全に隔離する。
このプロセスが落ちても、固まっても、実機の制御には一切影響しない。

--- 何が見えるか -----------------------------------------------------------
左: **実機の実際の姿勢**   エンコーダ29軸 + IMUの姿勢をFKで再現(物理なし)
右: **方策が意図した姿勢**  参照軌道 ref_q[t]。コックピットの進行に追従

  この2つの差が「方策の意図」と「実機の現実」のずれそのもの。
  さらに数値で3つの誤差を出す:

  追従誤差 |q - target|  … 指令したのに来ていない量。**PDが効いていない**
                           か、負荷に負けているか。関節マッピングの取り違え
                           はここに巨大な値として出る
  参照誤差 |q - ref_q[t]| … 方策が action で意図的に付けたずれを含む
  指令ずれ |target - ref| … 方策が参照からどれだけ離す判断をしたか(≒action)

★これは**運動学のミラーであって物理ツインではない。**姿勢は映すが力は
  映さない。「シムならこうなったはず」を見たいなら、走行後に
  `--replay` で記録を読み直すこと(そちらは記録された指令を使うので、
  実機との差が同じ時間軸で並ぶ)。
"""
import argparse
import json
import os
import pathlib
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]


class Poser:
    """MuJoCoモデルを「関節角+IMU」で決まる姿勢に置いて描画する。

    物理は回さない(mj_kinematics のみ)。足の低い方が床(z=0)に来るように
    体幹の高さだけ決める。水平位置は原点固定 — 実機の絶対位置は分からず、
    脚オドメトリはドリフトするので、見た目に嘘を混ぜない。
    """

    def __init__(self, size=420, scene="plain"):
        import mujoco
        self.mj = mujoco
        # ★既定は「ロボット+床だけ」のシーン。
        #   課題シーン(scene_task.xml)には椅子と段差が原点に置いてあるが、
        #   実機の水平位置は分からない(脚オドメトリはドリフトする)ので、
        #   家具と並べて描くと「椅子に座っている」ように見えてしまう。
        #   ミラーに嘘を混ぜない。家具込みで見たいときだけ --scene task。
        xml = "scene_task.xml" if scene == "task" else "scene_29dof.xml"
        self.scene = scene
        self.m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / xml))
        self.d = mujoco.MjData(self.m)
        self.qadr = []
        for nm in JOINTS:
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                  nm + "_joint")
            if j < 0:
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, nm)
            if j < 0:
                raise KeyError(f"MJCFに関節が無い: {nm}")
            self.qadr.append(int(self.m.jnt_qposadr[j]))
        self.qadr = np.array(self.qadr)
        self.fid = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.size = size
        self._r = None
        self._cam = None
        self.lock = threading.Lock()

    def _renderer(self):
        if self._r is None:
            self._r = self.mj.Renderer(self.m, height=self.size,
                                       width=self.size)
            self._cam = self.mj.MjvCamera()
            self._cam.lookat[:] = [0, 0, 0.55]
            self._cam.distance = 2.4
            self._cam.elevation = -12
        return self._r

    def render(self, q, quat, azimuth=40.0):
        """姿勢を置いて1枚描く。RGB配列を返す"""
        with self.lock:
            d = self.d
            self.mj.mj_resetData(self.m, d)
            d.qpos[0:3] = [0, 0, 1.0]
            d.qpos[3:7] = quat
            d.qpos[self.qadr] = q
            self.mj.mj_kinematics(self.m, d)
            # 低い方の足が床に来るよう体幹の高さを合わせる
            zmin = min(d.xpos[self.fid[0]][2], d.xpos[self.fid[1]][2])
            d.qpos[2] = 1.0 - zmin + 0.03
            self.mj.mj_kinematics(self.m, d)
            r = self._renderer()
            self._cam.azimuth = azimuth
            r.update_scene(d, self._cam)
            return r.render()


def _jpeg(img, quality=78):
    import cv2
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def _label(img, text, sub=""):
    import cv2
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (18, 18, 18), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (8, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
                    cv2.LINE_AA)
    return img


class Twin:
    def __init__(self, iface, cockpit_url, replay=None, azimuth=40.0,
                 scene="plain"):
        self.poser = Poser(scene=scene)
        self.cockpit_url = cockpit_url
        self.azimuth = azimuth
        self.lock = threading.Lock()
        self.q = np.zeros(29)
        self.dq = np.zeros(29)
        self.tau = np.zeros(29)
        self.target = np.zeros(29)
        self.kp = np.zeros(29)
        self.temps = np.zeros(29)
        self.quat = np.array([1.0, 0, 0, 0])
        self.gyro = np.zeros(3)
        self.n_state = 0
        self.last_t = 0.0
        self.src = "?"
        self.cock = {}
        self.pol_name = None
        self.ref_q = None
        self.ref_t = 0
        self.replay = replay
        self.replay_i = 0
        self.replay_n = 0
        self.replay_play = True
        self._pol_cache = {}
        # ★描画は専用スレッド1本だけで行う。
        #   MuJoCoのGLコンテキストは**作ったスレッドに紐づく**。
        #   ThreadingHTTPServer はリクエストごとに別スレッドを作るので、
        #   ハンドラから直接 render() すると、2回目以降は別スレッドから
        #   GLXMakeCurrent することになり X が BadAccess で**プロセスごと
        #   落とす**(Xの致命エラーはPythonで捕まえられない)。実測で再現。
        #   ここで作った1枚を全ハンドラが読むだけにする。
        self._jpeg = b""
        self._render_n = 0
        self._render_ms = 0.0
        self._render_err = ""
        threading.Thread(target=self._render_loop, daemon=True,
                         name="render").start()
        if replay is not None:
            self._load_replay(replay)
            self.src = f"再生 {pathlib.Path(replay).name}"
            threading.Thread(target=self._replay_loop, daemon=True).start()
        else:
            self._init_dds(iface)
            self.src = f"実機 {iface or '(既定NIC)'}"
            threading.Thread(target=self._poll_cockpit, daemon=True).start()

    # ---------------- ライブ(DDS購読。★送信しない)
    def _init_dds(self, iface):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, iface) if iface \
            else ChannelFactoryInitialize(0)
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)

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
            self.n_state += 1
            self.last_t = time.time()

    def _poll_cockpit(self):
        """コックピットの /state?full を読んで、いま何を走らせているかを知る。
        ★GETのみ。コックピットには何も指示しない。落ちていても動き続ける。"""
        while True:
            try:
                with urllib.request.urlopen(self.cockpit_url + "/state?full=1",
                                            timeout=1.0) as r:
                    d = json.load(r)
                with self.lock:
                    self.cock = d
                    if "target" in d:
                        self.target = np.asarray(d["target"], float)
                        self.kp = np.asarray(d["kp"], float)
                    ph = d.get("phases") or []
                    i = d.get("phase_i", -1)
                    nm = ph[i] if 0 <= i < len(ph) else None
                    if nm and nm != self.pol_name:
                        self.pol_name = nm
                        self.ref_q = self._ref(nm)
                    self.ref_t = int(d.get("t", 0))
            except Exception:                      # noqa: BLE001
                with self.lock:
                    self.cock = {}
            time.sleep(0.2)

    def _ref(self, name):
        if name not in self._pol_cache:
            try:
                z = np.load(ROOT / "deploy" / name / "reference.npz")
                self._pol_cache[name] = np.asarray(z["ref_q"], float)
            except Exception:                      # noqa: BLE001
                self._pol_cache[name] = None
        return self._pol_cache[name]

    # ---------------- 再生(記録したnpzを読み直す)
    def _load_replay(self, path):
        d = np.load(path, allow_pickle=True)
        cols = [str(c) for c in d["cols"]]
        rec = d["rec"]
        self._rp = {
            "q": rec[:, [cols.index(f"q{i}") for i in range(29)]],
            "dq": rec[:, [cols.index(f"dq{i}") for i in range(29)]],
            "tau": rec[:, [cols.index(f"tau{i}") for i in range(29)]],
            "target": rec[:, [cols.index(f"target{i}") for i in range(29)]],
            "temps": rec[:, [cols.index(f"temp{i}") for i in range(29)]],
            "quat": rec[:, [cols.index(c) for c in
                            ("quat_w", "quat_x", "quat_y", "quat_z")]],
            "tilt": rec[:, cols.index("tilt_deg")],
        }
        self.replay_n = len(rec)
        self.replay_final = bool(d["final"]) if "final" in d else None
        # run<NN>_<phase>_<方策名>.npz というファイル名から方策名を取る
        nm = "_".join(pathlib.Path(path).stem.split("_")[2:])
        self.pol_name = nm
        self.ref_q = self._ref(nm)

    def _replay_loop(self):
        while True:
            time.sleep(1.0 / 25.0)                 # 実時間の1/2で再生
            with self.lock:
                if self.replay_play and self.replay_n:
                    self.replay_i = (self.replay_i + 1) % self.replay_n
                    self._apply_replay()

    def _apply_replay(self):
        i = min(self.replay_i, self.replay_n - 1)
        r = self._rp
        self.q = r["q"][i].astype(float)
        self.dq = r["dq"][i].astype(float)
        self.tau = r["tau"][i].astype(float)
        self.target = r["target"][i].astype(float)
        self.temps = r["temps"][i].astype(float)
        self.quat = r["quat"][i].astype(float)
        self.ref_t = i
        self.last_t = time.time()
        self.n_state = i

    # ---------------- 出力
    def numbers(self):
        with self.lock:
            q = self.q.copy(); tgt = self.target.copy(); kp = self.kp.copy()
            tau = self.tau.copy(); tp = self.temps.copy()
            quat = self.quat.copy(); ref = self.ref_q
            t = self.ref_t; cock = dict(self.cock)
            age = time.time() - self.last_t
        rq = (ref[min(t, len(ref) - 1)] if ref is not None and len(ref)
              else None)
        e_track = np.abs(q - tgt)
        rows = []
        for i in range(29):
            rows.append({
                "j": i, "name": JOINTS[i],
                "q": round(float(q[i]), 3),
                "target": round(float(tgt[i]), 3),
                "track": round(float(e_track[i]), 3),
                "ref": None if rq is None else round(float(rq[i]), 3),
                "eref": None if rq is None else round(float(abs(q[i] - rq[i])), 3),
                "cmd": None if rq is None else round(float(tgt[i] - rq[i]), 3),
                "tau": round(float(tau[i]), 1),
                "temp": int(tp[i]),
            })
        w, x, y, z = quat
        up_z = 1 - 2 * (x * x + y * y)
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        worst = sorted(rows, key=lambda r: -r["track"])[:6]
        return {
            "src": self.src, "policy": self.pol_name, "t": t,
            "n": 0 if rq is None else len(ref),
            "state_age_ms": round(age * 1000, 1),
            "n_state": self.n_state,
            "tilt": round(tilt, 1),
            "track_max": round(float(e_track.max()), 3),
            "track_max_j": JOINTS[int(e_track.argmax())],
            "track_rms": round(float(np.sqrt(np.mean(e_track ** 2))), 4),
            "has_ref": rq is not None,
            "eref_max": None if rq is None else round(float(np.abs(q - rq).max()), 3),
            "gain_on": bool(kp.max() > 0),
            "rows": rows, "worst": worst,
            "replay": {"i": self.replay_i, "n": self.replay_n,
                       "play": self.replay_play} if self.replay else None,
            "fsm": cock.get("fsm"), "msg": cock.get("msg"),
            "render_n": int(self._render_n),
            "render_ms": round(float(self._render_ms), 1),
            "render_err": self._render_err,
        }

    def frame(self):
        """最新の1枚を返す(描画は専用スレッドが作る)"""
        return self._jpeg

    def _render_loop(self, hz=5.0):
        """★このスレッドだけがMuJoCoを描画する(GLコンテキストの持ち主)"""
        dt = 1.0 / hz
        while True:
            t0 = time.time()
            try:
                self._jpeg = self._render_now()
                self._render_ms = (time.time() - t0) * 1000
                self._render_n += 1
                self._render_err = ""
            except Exception as e:                 # noqa: BLE001
                self._render_err = f"{type(e).__name__}: {e}"
                if self._render_n == 0:
                    print(f"★描画に失敗: {self._render_err}")
                time.sleep(0.5)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def _render_now(self):
        """左=実機の実際の姿勢 / 右=方策が意図した姿勢(参照)"""
        with self.lock:
            q = self.q.copy(); quat = self.quat.copy()
            ref = self.ref_q; t = self.ref_t
        a = self.poser.render(q, quat, self.azimuth)
        d = self.numbers()
        note = "xy NOT estimated" if self.poser.scene == "task" else ""
        a = _label(a, "実機(エンコーダ+IMU)",
                   f"tilt {d['tilt']:.0f}deg  track {d['track_max']:.3f}rad"
                   + (f"   {note}" if note else ""))
        if ref is not None and len(ref):
            rq = ref[min(t, len(ref) - 1)]
            b = self.poser.render(rq, np.array([1.0, 0, 0, 0]), self.azimuth)
            b = _label(b, f"方策の参照 {self.pol_name or ''} [{t}]",
                       f"eref {d['eref_max']:.3f}rad" if d["eref_max"] is not None
                       else "")
        else:
            b = np.full_like(a, 26)
            b = _label(b, "参照なし(方策の実行中に出ます)")
        return _jpeg(np.concatenate([a, b], axis=1))


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>G1 実機ミラー</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#111;color:#eee;font:13px/1.5 "Segoe UI",sans-serif;margin:0;padding:12px}
h1{font-size:15px;margin:0 0 8px}
img{width:100%;max-width:900px;border-radius:8px;display:block}
.lbl{color:#9a9a9a;font-size:11px}
table{border-collapse:collapse;margin-top:10px;font:12px Consolas,monospace}
td,th{padding:2px 8px;border-bottom:1px solid #2a2a2a;text-align:right}
th{color:#9a9a9a;font-weight:400}td:first-child,th:first-child{text-align:left}
.big{font-size:20px;font-weight:700}
.k{color:#9a9a9a;font-size:10px}
.tiles{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}
.tile{background:#1c1c1c;border:1px solid #333;border-radius:8px;padding:6px 12px}
.bad{color:#e34948}.warn{color:#eda100}.ok{color:#1baf7a}
</style></head><body>
<h1>🪞 G1 実機ミラー <span class="lbl" id="src"></span></h1>
<img id="cam">
<div class="tiles">
 <div class="tile"><div class="k">追従誤差 最大</div><div class="big" id="tr">-</div><div class="k" id="trj"></div></div>
 <div class="tile"><div class="k">追従誤差 RMS</div><div class="big" id="trr">-</div></div>
 <div class="tile"><div class="k">参照との差 最大</div><div class="big" id="er">-</div></div>
 <div class="tile"><div class="k">傾き</div><div class="big" id="ti">-</div></div>
 <div class="tile"><div class="k">受信</div><div class="big" id="ag">-</div></div>
</div>
<div class="lbl" id="note"></div>
<table id="tb"><thead><tr><th>関節</th><th>実測 q</th><th>指令 target</th>
<th>追従誤差</th><th>参照 ref</th><th>指令ずれ</th><th>τ</th><th>温度</th></tr></thead>
<tbody></tbody></table>
<script>
async function tick(){
 let d;try{d=await(await fetch('/nums')).json()}catch(e){return}
 document.getElementById('src').textContent=
  `[${d.src}] ${d.policy?d.policy+' '+d.t+'/'+d.n:'(方策の実行なし)'} ${d.fsm?'FSM='+d.fsm:''}`;
 const set=(id,v,cls)=>{const e=document.getElementById(id);e.textContent=v;e.className='big '+(cls||'')};
 set('tr',d.track_max.toFixed(3),d.track_max>0.35?'bad':(d.track_max>0.20?'warn':'ok'));
 document.getElementById('trj').textContent=d.track_max_j;
 set('trr',d.track_rms.toFixed(3));
 set('er',d.eref_max==null?'-':d.eref_max.toFixed(3));
 set('ti',d.tilt.toFixed(0)+'°',d.tilt>25?'warn':'');
 set('ag',d.state_age_ms<200?'OK':'途絶',d.state_age_ms<200?'ok':'bad');
 document.getElementById('note').textContent=
  (d.gain_on?'':'※ゲイン0(標準制御中/待機中)なので追従誤差に意味はありません。')
  +'追従誤差=|実測−指令|。0.35radを超え続けるならPDが負けているか関節マッピングを疑う。';
 const tb=document.getElementById('tb').querySelector('tbody');
 tb.innerHTML=d.worst.map(r=>`<tr><td>${r.name}</td><td>${r.q.toFixed(3)}</td>
  <td>${r.target.toFixed(3)}</td><td class="${r.track>0.35?'bad':(r.track>0.2?'warn':'')}">${r.track.toFixed(3)}</td>
  <td>${r.ref==null?'-':r.ref.toFixed(3)}</td><td>${r.cmd==null?'-':r.cmd.toFixed(3)}</td>
  <td>${r.tau.toFixed(1)}</td><td>${r.temp}</td></tr>`).join('');
 document.getElementById('cam').src='/frame.jpg?t='+Date.now();
}
setInterval(tick,250);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    twin = None

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:                          # noqa: BLE001
            pass

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/nums":
            self._send(json.dumps(self.twin.numbers()).encode(),
                       "application/json")
        elif p == "/frame.jpg":
            self._send(self.twin.frame(), "image/jpeg")
        elif p == "/seek":
            qs = parse_qs(urlparse(self.path).query)
            with self.twin.lock:
                if "i" in qs:
                    self.twin.replay_i = int(qs["i"][0])
                    self.twin._apply_replay()
                if "play" in qs:
                    self.twin.replay_play = qs["play"][0] == "1"
            self._send(b"ok", "text/plain")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

    def log_message(self, *a):
        pass


def _other_renderers():
    """既に走っているMuJoCo描画プロセスを探す。

    この環境ではMuJoCoの描画バックエンドがGLX(glfw)しか使えず
    (EGL/OSMesaは初期化に失敗する)、**同じXディスプレイで2つのプロセスが
    同時にGLコンテキストを作るとX側で BadAccess になって落ちる**。
    Xの致命エラーはPythonで捕まえられないので、起動する前に弾く。
    """
    import subprocess
    me = str(os.getpid())
    out = []
    try:
        ps = subprocess.run(["ps", "-eo", "pid,comm,args", "--no-headers"],
                            capture_output=True, text=True, timeout=5).stdout
    except Exception:                              # noqa: BLE001
        return out
    for line in ps.splitlines():
        f = line.split(None, 2)
        if len(f) < 3 or f[0] == me or not f[1].startswith("python"):
            continue
        args = f[2]
        if "twin.py" in args or ("cockpit.py" in args and "--sim" in args):
            out.append((f[0], args.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="", help="実機NIC名(例 enp46s0)")
    ap.add_argument("--replay", default="", help="走行ログのnpzを再生する")
    ap.add_argument("--cockpit", default="http://localhost:8090")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--azimuth", type=float, default=40.0, help="視点の方位角")
    ap.add_argument("--scene", default="plain", choices=("plain", "task"),
                    help="plain=ロボットと床のみ(既定) / task=椅子と段差も描く"
                         "(★水平位置は推定していないので位置関係は信用しない)")
    ap.add_argument("--force", action="store_true",
                    help="他の描画プロセスがいても起動する(★落ちる可能性)")
    a = ap.parse_args()
    for pid, args in _other_renderers():
        print(f"※MuJoCoを描画する別プロセスが動いています: pid {pid} "
              f"{args[:80]}")
        print("  この環境はGLX(glfw)しか使えません(EGL/OSMesaは初期化に失敗)。"
              "描画が不安定なら片方を止めてください")
    print("★このプロセスは1バイトも送信しません(rt/lowstate の購読のみ)")
    if a.scene == "task":
        print("★--scene task: 椅子と段差を描きますが、実機の水平位置は"
              "推定していません。位置関係は見た目に出ているだけです")
    tw = Twin(a.iface, a.cockpit, a.replay or None, a.azimuth, a.scene)
    Handler.twin = tw
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"実機ミラー: http://localhost:{a.port}   ({tw.src})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n終了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
