#!/usr/bin/env python3
"""方策をシミュレータで走らせて、動画に保存する。**実機には一切繋がらない。**

  python3 real/make_video.py                    # 着座系の方策すべて
  python3 real/make_video.py --policy sit_up_deep_r2
  python3 real/make_video.py --all              # deploy の全方策
  python3 real/make_video.py --slow 2           # 2倍スローで書き出す

出力: video/<方策名>.mp4  (左=側面 / 右=斜め の2画面)

★実機の操縦中には走らせないこと。MuJoCoの物理と描画でCPUを使う。
  2026-08-26 に解析ツールを走らせたまま実機を動かして、制御ループが
  19Hzまで落ちて転倒した実績がある。

--- 何が見えるか -----------------------------------------------------------
方策を**シムの中で閉ループで**回す(実機のログ再生ではない)。
つまり「この方策は、理想的な環境ならどう座るのか」が見える。
画面には次を焼き込む:

  t        経過時間とコマ番号
  tilt     体幹の傾き[度]。参照軌道の最大は18度。実機は完走で20.8〜32.8度、
           転倒は34.6〜39.4度(2026-08-26の実測11本)
  pelvis   骨盤の高さ[m](足裏基準)。開始0.734 → 着座0.454 が参照
  seat     椅子が尻から受けている鉛直力[N]。ロボットは約35kg=350N
  foot     足が段から受けている鉛直力[N]

  ★seat が増えて foot が減る = 体重が座面へ移る = 座れている
    seat が増えたのに後から減る = ずり落ち(実機セッション手順 §6)
"""
import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

VIEWS = (("side", 0.0), ("angled", 48.0))       # (名前, 方位角)


class _writer:
    """H.264(avc1)で書く。**mp4v では VS Code もブラウザも再生できない。**

    Chromium系(VS Codeの組み込みビデオプレビュー vscode.videoPreview、
    ブラウザ、Slack等)がデコードできるのは H.264 / VP8 / VP9 / AV1 で、
    OpenCV既定の mpeg4(mp4v) は**入っていない**。2026-08-26に mp4v で
    書いて「VS Codeで開けない」となったので ffmpeg へ生フレームを流す。
    ffmpeg が無ければ cv2 の mp4v へ落ちる(その場合は再生できない旨を警告)。
    """

    def __init__(self, path, w, h, fps):
        import shutil
        import subprocess
        self.path = path
        self.proc = None
        self.vw = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-preset", "medium", "-crf", "23",
                 "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE)
        else:
            import cv2
            print("  ※ffmpeg が無いので mp4v で書きます"
                  "(VS Code/ブラウザでは再生できません)")
            self.vw = cv2.VideoWriter(str(path),
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, (w, h))
            if not self.vw.isOpened():
                raise RuntimeError("VideoWriter を開けない")

    def write(self, img):
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.vw.write(img)

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.vw.release()


def overlay(img, lines, w=None):
    import cv2
    img = np.ascontiguousarray(img)
    h = 22 * len(lines) + 10
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (16, 16, 16), -1)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)
    return img


def render_policy(name, size=520, slow=1, fps=50, quiet=False):
    import cv2
    import mujoco
    from sim_compare import Sim
    from run_fsm import ACTION_SCALE, ObsBuilder, Policy

    pol = Policy(name)
    ob = ObsBuilder(pol)
    sim = Sim(name)
    z = sim.ref
    ob.reset(est_xy=z["ref_xy_abs"][0][:2])
    sim.place(z["ref_q"][0], z["ref_quat"][0])

    rend = mujoco.Renderer(sim.m, height=size, width=size)
    cams = []
    for _nm, az in VIEWS:
        c = mujoco.MjvCamera()
        # 開始位置と着座位置の中間を見る(この方策は後方へ0.32m移動する)
        c.lookat[:] = [float(z["ref_xy_abs"][0][0]),
                       float((z["ref_xy_abs"][0][1] + z["ref_xy_abs"][-1][1]) / 2),
                       0.55]
        c.distance, c.elevation, c.azimuth = 2.5, -12, az
        cams.append(c)

    out = ROOT / "video"
    out.mkdir(exist_ok=True)
    path = out / f"{name}.mp4"
    vw = _writer(path, size * len(VIEWS), size, fps / max(slow, 1))

    tilts, seats, pelv = [], [], []
    t0 = time.time()
    for t in range(pol.n):
        s = sim.read()
        gyro = sim.d.qvel[3:6].copy()
        obs = ob.build(pol, t, s["q"], s["dq"], s["quat"], gyro)
        a = pol.act(obs)
        ob.last_cmd = a.copy()
        sim.step(pol.ref_q[min(t, pol.n - 1)] + a * ACTION_SCALE,
                 np.asarray(pol.kp, float), np.asarray(pol.kd, float))
        s = sim.read()
        tilts.append(s["tilt"]); seats.append(s["seat_n"]); pelv.append(s["pelvis_z"])
        # --- 描画
        frames = []
        for c in cams:
            rend.update_scene(sim.d, c)
            frames.append(rend.render())
        img = np.concatenate(frames, axis=1)
        tc = ((90, 220, 120) if s["tilt"] < 33 else (80, 80, 235))
        img = overlay(img[:, :, ::-1], [
            (f"{name}   t={t/50:.2f}s  frame {t+1}/{pol.n}", (235, 235, 235)),
            (f"tilt {s['tilt']:5.1f} deg   (ref max 18 / real OK<33 / fall>34)", tc),
            (f"pelvis {s['pelvis_z']:.3f} m   seat {s['seat_n']:6.0f} N   "
             f"foot {s['foot_n']:6.0f} N", (200, 200, 200)),
        ])
        vw.write(img)
    vw.close()
    dt = time.time() - t0
    tilts = np.array(tilts); seats = np.array(seats); pelv = np.array(pelv)
    tail = slice(int(len(pelv) * 0.85), len(pelv))
    res = dict(name=name, n=pol.n, tilt_max=float(tilts.max()),
               tilt_end=float(tilts[-1]),
               pelv0=float(pelv[0]), pelv_end=float(pelv[-1]),
               sink_mm=float((pelv[tail].max() - pelv[-1]) * 1000),
               seat_end=float(seats[tail].mean()), path=path, sec=dt)
    if not quiet:
        print(f"  {name:24s} 最大傾き {res['tilt_max']:5.1f}度 / 終端 {res['tilt_end']:5.1f}度"
              f" / 骨盤 {res['pelv0']:.3f}→{res['pelv_end']:.3f}m"
              f" / 着座後の沈み {res['sink_mm']:4.0f}mm"
              f" / 座面荷重 {res['seat_end']:5.0f}N   ({dt:.0f}秒で書き出し)")
        print(f"  → {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="", help="1つだけ書き出す")
    ap.add_argument("--all", action="store_true", help="deployの全方策")
    ap.add_argument("--slow", type=float, default=1.0, help="2で2倍スロー")
    ap.add_argument("--size", type=int, default=520)
    a = ap.parse_args()

    import subprocess
    try:
        ps = subprocess.run(["ps", "-eo", "comm,args"], capture_output=True,
                            text=True, timeout=5).stdout
        if [l for l in ps.splitlines() if l.startswith("python")
                and "cockpit.py" in l and "--sim" not in l]:
            print("※実機のコックピットが動いています。"
                  "**実機を操縦している最中には実行しないこと**")
    except Exception:                              # noqa: BLE001
        pass

    names = []
    if a.policy:
        names = [a.policy]
    else:
        for d in sorted((ROOT / "deploy").iterdir()):
            if not (d / "policy.pt").exists():
                continue
            if a.all or d.name.startswith("sit"):
                names.append(d.name)
    if not names:
        print("対象の方策が無い")
        return 1
    print(f"シミュレータで {len(names)} 件を動画にします → {ROOT/'video'}/")
    rows = []
    for n in names:
        try:
            rows.append(render_policy(n, size=a.size, slow=a.slow))
        except Exception as e:                     # noqa: BLE001
            print(f"  {n:24s} ★失敗: {type(e).__name__}: {e}")
    if rows:
        print(f"\n{'方策':24s} {'最大傾き':>8s} {'終端傾き':>8s} {'骨盤降下':>8s} "
              f"{'沈み':>6s} {'座面荷重':>8s}")
        for r in sorted(rows, key=lambda x: x["tilt_max"]):
            print(f"{r['name']:24s} {r['tilt_max']:7.1f}° {r['tilt_end']:7.1f}° "
                  f"{r['pelv0']-r['pelv_end']:7.3f}m {r['sink_mm']:5.0f}mm "
                  f"{r['seat_end']:7.0f}N")
        print("\n※最大傾き: 参照軌道は18度。実機の完走は20.8〜32.8度、"
              "転倒は34.6〜39.4度(2026-08-26 実測11本)")
        print("※沈み: 着座後に骨盤が下がった量。0〜6mmが正常、"
              "67mmが「ずり落ち」1本(シム39本中1本)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
