#!/usr/bin/env python3
"""autowalk の机上試験(実機不要・1〜2分)。

  python3 real/test_autowalk.py

見るもの:
  1. 障害物検出: 世界座標(odom)の点群 / センサ座標(傾き10度)の点群 の両方で
     前方距離が幾何どおりに出るか(±0.10m)。壁と障害物の区別、回り込める側
  2. デッドマン: 速度指令が0.5秒更新されなければゼロを送るか
  3. 壁: 前進のみ。段階的に減速して停止距離±0.15mで止まるか(速度が単調に落ちるか)
  4. 回り込み: 幅0.6mの箱を左右どちらかへ避け、元の経路(ずれ±5cm)へ戻って壁の手前で止まるか。
     箱の横を通る間、箱に近づきすぎないか
  5. 横歩き(足踏みパルス): 0.5m と 0.05m の微調整が ±3cm で止まるか
  6. 中止: UIハートビート途絶で自動歩行が止まるか
"""
import math
import pathlib
import sys
import threading
import time
import os
import tempfile
os.environ["G1_WALK_CALIB"] = os.path.join(tempfile.gettempdir(), "g1_walk_calib_test.json")   # 試験は本物の較正ファイルを汚さない

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from autowalk import (ObstacleDetector, VelSender, WalkController,   # noqa: E402
                      WALK_DEFAULTS, SENSOR_FWD_OFFSET)

FAIL = []


def check(cond, msg):
    print(("  [OK]   " if cond else "  [NG]   ") + msg)
    if not cond:
        FAIL.append(msg)


def make_world_cloud(rng, robot_xy, box_front_dist, yaw, box_w=0.6, box_h=1.0, box_lat=0.0):
    """床 + 箱(機体前方 box_front_dist にある前面、横中心 box_lat)を世界座標で作る"""
    x0, y0 = robot_xy
    n = 2000
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = np.sqrt(rng.uniform(0, 1, n)) * 4.0
    floor = np.stack([x0 + rad * np.cos(ang), y0 + rad * np.sin(ang),
                      rng.normal(0, 0.01, n)], 1)
    m = 300 if box_w < 2 else 1200
    lat = rng.uniform(box_lat - box_w / 2, box_lat + box_w / 2, m)
    h = rng.uniform(0.0, box_h, m)
    fwd = np.full(m, box_front_dist) + rng.uniform(0, 0.3, m)   # 奥行き0.3
    c, s = math.cos(yaw), math.sin(yaw)
    face = np.stack([x0 + c * fwd - s * lat, y0 + s * fwd + c * lat, h], 1)
    return np.concatenate([floor, face]).astype(np.float32)


def test_detector():
    print("--- 1. 障害物検出 ---")
    rng = np.random.default_rng(0)
    det = ObstacleDetector(WALK_DEFAULTS)
    # 世界座標: 機体は (3,2)・ヨー0.7rad、前方1.35mに箱(幅0.6)
    pts = make_world_cloud(rng, (3.0, 2.0), 1.35, 0.7)
    r = det.update(pts, "odom", (3.0, 2.0), 0.7, side_dir=1)
    check(r["ok"] and r["frame"] == "world", f"世界座標の点群を体基準へ変換 {r['why']}")
    check(r["dist"] is not None and abs(r["dist"] - 1.35) < 0.10,
          f"世界座標: 前方距離 {r['dist']} (期待1.35±0.10)")
    check(r["floor_ok"] and abs(r["floor_h"]) < 0.05,
          f"世界座標: 床平面 {r['floor_h']} (期待≈0)")
    ah = r["ahead"] or {}
    check(ah and not ah["wall"] and abs((ah["lat_hi"] - ah["lat_lo"]) - 0.6) < 0.15,
          f"幅0.6mの箱を障害物と判定(壁でない) 横幅 {None if not ah else round(ah['lat_hi'] - ah['lat_lo'], 2)}")
    check(ah and ah["free_l"] is not None and ah["free_r"] is not None
          and abs(abs(ah["free_l"]) - 0.63) < 0.1,
          f"回り込み先 左{ah.get('free_l')} / 右{ah.get('free_r')} (期待 ±0.63)")
    check(r["side_free"] is None, f"横の空き {r['side_free']} (期待 None=空き)")
    # 壁(幅4m)
    pw = make_world_cloud(rng, (0.0, 0.0), 1.2, 0.0, box_w=4.0)
    rw = det.update(pw, "odom", (0.0, 0.0), 0.0)
    ahw = rw["ahead"] or {}
    check(ahw and ahw["wall"] and ahw["free_l"] is None and ahw["free_r"] is None,
          f"幅4mを壁と判定(回り込み先なし) wall={ahw.get('wall')}")
    # 障害物なし
    pts2 = make_world_cloud(rng, (0.0, 0.0), 9.0, 0.0)
    r2 = det.update(pts2, "odom", (0.0, 0.0), 0.0)
    check(r2["ok"] and r2["dist"] is None, f"障害物なし: dist={r2['dist']} (期待 None)")
    # 世界座標なのにオドメトリが無い → 判定不能(安全側)
    r3 = det.update(pts, "odom", None, 0.7)
    check(not r3["ok"], "世界座標でオドメトリ無し → 判定不能を返す")
    # センサ座標: センサ高さ1.15m・下向き10度
    body = make_world_cloud(rng, (0.0, 0.0), 1.20, 0.0)
    th = math.radians(10.0)
    P = body.astype(np.float64)
    P[:, 0] -= SENSOR_FWD_OFFSET
    P[:, 2] -= 1.15
    cx, sx = math.cos(th), math.sin(th)
    xs = cx * P[:, 0] - sx * P[:, 2]
    zs = sx * P[:, 0] + cx * P[:, 2]
    sensor = np.stack([xs, P[:, 1], zs], 1).astype(np.float32)
    det2 = ObstacleDetector(WALK_DEFAULTS)
    r4 = det2.update(sensor, "livox_frame", None, None, side_dir=1)
    check(r4["ok"] and r4["frame"] == "sensor", f"センサ座標の点群を扱える {r4['why']}")
    check(r4["floor_ok"] and abs(r4["floor_h"] - 1.15) < 0.08,
          f"センサ座標: 床までの高さ {r4['floor_h']} (期待1.15±0.08)")
    check(r4["dist"] is not None and abs(r4["dist"] - 0.95) < 0.10,
          f"センサ座標(傾き10度): 前方距離 {r4['dist']} (期待0.95±0.10 = 1.20 − つま先0.15 − センサ0.10)")
    # 壁(幅4m)をセンサ座標で: 壁の面フィット・4方向・ヨー補正(2026-09-04 午後)
    bodyw = make_world_cloud(rng, (0.0, 0.0), 1.50, 0.0, box_w=4.0)
    Pw = bodyw.astype(np.float64)
    Pw[:, 0] -= SENSOR_FWD_OFFSET
    Pw[:, 2] -= 1.15
    xs = cx * Pw[:, 0] - sx * Pw[:, 2]
    zs = sx * Pw[:, 0] + cx * Pw[:, 2]
    sensw = np.stack([xs, Pw[:, 1], zs], 1).astype(np.float32)
    det3 = ObstacleDetector(WALK_DEFAULTS)
    r5 = det3.update(sensw, "livox_level", None, None, side_dir=1)
    exp = 1.50 - SENSOR_FWD_OFFSET - WALK_DEFAULTS["front_offset"]
    check(r5["wall_dist"] is not None and abs(r5["wall_dist"] - exp) < 0.10,
          f"壁の面フィット: {r5['wall_dist']} (期待{exp:.2f}±0.10) 角度{r5['wall_ang']} 幅{r5['wall_len']}")
    d = r5.get("dirs") or {}
    check(d.get("front") is not None and abs(d["front"] - (1.50 - SENSOR_FWD_OFFSET)) < 0.15,
          f"4方向: 前 {d.get('front')} (期待{1.50 - SENSOR_FWD_OFFSET:.2f}±0.15) 後 {d.get('back')} 左 {d.get('left')} 右 {d.get('right')}")
    # センサがヨー +30 度回って付いている → yaw_fix_deg=-30 で同じ距離になる
    th2 = math.radians(30.0)
    c2, s2 = math.cos(th2), math.sin(th2)
    rot = np.stack([c2 * sensw[:, 0] - s2 * sensw[:, 1], s2 * sensw[:, 0] + c2 * sensw[:, 1], sensw[:, 2]], 1).astype(np.float32)
    cfg = dict(WALK_DEFAULTS)
    cfg["yaw_fix_deg"] = -30.0
    det4 = ObstacleDetector(cfg)
    r6 = det4.update(rot, "livox_level", None, None, side_dir=1)
    check(r6["wall_dist"] is not None and abs(r6["wall_dist"] - exp) < 0.10 and abs(r6["wall_ang"] or 0) < 6,
          f"ヨー補正 -30度: 壁 {r6['wall_dist']} (期待{exp:.2f}±0.10) 壁の角度 {r6['wall_ang']} (期待≈0)")


def test_deadman():
    print("--- 2. デッドマン ---")
    sent = []
    vs = VelSender(lambda vx, vy, om, d: (sent.append((round(vx, 2), round(vy, 2), round(om, 2))) or True),
                   log=print)
    vs.set(0.3, 0.0, 0.0, "test")
    time.sleep(0.35)
    check(any(s[0] == 0.3 for s in sent), f"指令が送られる {sent[-1] if sent else None}")
    time.sleep(0.9)
    check(vs.last_sent == (0.0, 0.0, 0.0) and (0.0, 0.0, 0.0) in sent,
          f"0.5秒更新が無ければゼロを送る(最後の送信 {vs.last_sent})")
    vs.close()


class _Ticker:
    """Engineの50Hzループの代わりに SimRobot.tick を回し、位置の履歴を残す"""

    def __init__(self, robot):
        self.r = robot
        self.on = True
        self.hist = []               # (t, x, y, v_cmd)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.on:
            t0 = time.time()
            self.r.tick(10)
            self.hist.append((t0, self.r._wx, self.r._wy, self.r._wv[0]))
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))


def _setup(boxes, params, hb_ok=True, log=None):
    from sim_robot import SimRobot
    logs = []

    def _log(s):
        logs.append(s)
        print("    " + s)
    robot = SimRobot()
    robot.standard_mode("stand")
    robot.sim_obstacles(boxes)
    tk = _Ticker(robot)
    wc = WalkController(robot, log=log or _log, hb_ok=lambda: hb_ok, is_sim=True)
    wc.set_params(params)
    if os.environ.get("G1_TEST_LOGDIR"):            # 1 コマごとの記録(walk_*.jsonl)を残したいとき
        wc.log_dir = os.environ["G1_TEST_LOGDIR"]
    check(wc.prepare(), "歩行モードへ(4→501)")
    time.sleep(0.6)
    return robot, tk, wc


def _wait(wc, limit=120):
    t0 = time.time()
    while not wc.auto.done and time.time() - t0 < limit:
        time.sleep(0.2)
    return time.time() - t0


def test_wall():
    print("--- 3. 壁の手前で減速して停止(前進のみ)。止まった後は fsm_mode 0、FSM は歩行のまま ---")
    wall_front = 2.5 - 0.1
    robot, tk, wc = _setup([dict(x=2.5, y=0.0, w=4.0, d=0.2, h=1.0)],
                           dict(v_fwd=0.9, stop_dist=0.6, mode="forward", max_fwd=6.0))
    st = wc.status()
    check(st["dist"] is not None and abs(st["dist"] - wall_front) < 0.12 and st["wall"],
          f"開始時: 壁 {st['dist']}m 壁判定={st['wall']} (期待{wall_front:.2f})")
    check(wc.start_auto(), "前進 開始")
    t0 = time.time()
    while time.time() - t0 < 60 and not wc.auto.done and wc.auto.phase != "STOPPING":
        time.sleep(0.1)
    n_hist_stop = len(tk.hist)
    dt = _wait(wc)
    tk.on = False
    res = wc.auto.result
    si = wc.status().get("stop_info") or {}
    print(f"    結果: {res}  x={robot._wx:.2f} y={robot._wy:.2f}  所要{dt:.1f}秒  停止記録 {si}")
    check(res.startswith("完了"), "壁の手前で完了した")
    check(abs(robot._wx - (wall_front - 0.6)) < 0.25,
          f"停止位置 x={robot._wx:.2f} (期待{wall_front - 0.6:.2f}±0.25。実機は停止記録 d0/d1 で stop_lead を合わせる)")
    check(si.get("t_settle") is not None and si.get("fsm_mode") == 0, f"止まった判定は fsm_mode==0 で取れた: {si}")
    check(robot.get_fsm_id() == 501, f"止まった後も歩行 FSM のまま(いま {robot.get_fsm_id()})")
    v = np.array([h[3] for h in tk.hist[:n_hist_stop]])
    peak = v.max()
    i_pk = int(v.argmax())
    tail = v[i_pk:]
    rises = int(np.sum(np.diff(tail) > 0.03))
    check(peak > 0.28 and rises == 0, f"速度は巡航{peak:.2f}まで上がり(実速度。指令上限0.9で実機相当0.32)、その後は単調に減速(増加{rises}回)")
    sent = wc.sender.last_sent
    check(abs(sent[0]) < 1e-6 and abs(sent[1]) < 1e-6, "終了時に速度ゼロ")
    wc.close()


def _clearance(tk, box):
    """箱の横を通っている間(x が箱の範囲±0.25)の、体の中心から箱の端までの最小距離[m]。通っていなければ None"""
    xs = np.array([h[1] for h in tk.hist]); ys = np.array([h[2] for h in tk.hist])
    near = (xs > box["x"] - box["d"] / 2 - 0.25) & (xs < box["x"] + box["d"] / 2 + 0.25)
    if not near.any():
        return None, None
    dy = ys[near] - box["y"]
    k = int(np.argmin(np.abs(dy)))
    return float(np.abs(dy).min() - box["w"] / 2), float(dy[k])


def test_detour():
    print("--- 4. 障害物の回り込み(幅0.6mの箱 → 左を通る → 元の経路へ → 壁の手前で停止) ---")
    box = dict(x=1.8, y=0.0, w=0.6, d=0.3, h=1.0)
    boxes = [box, dict(x=4.5, y=0.0, w=4.0, d=0.2, h=1.0)]      # 障害物(箱)とその先の壁
    robot, tk, wc = _setup(boxes, dict(v_fwd=0.5, stop_dist=0.6, mode="forward",
                                       max_fwd=8.0, avoid=True))
    st = wc.status()
    check(st["dist"] is not None and st["wall"] is False,
          f"開始時: 障害物 {st['dist']}m 幅{st['width']} 壁判定={st['wall']}")
    check(wc.start_auto(), "前進 開始")
    dt = _wait(wc, 180)
    tk.on = False
    res = wc.auto.result
    print(f"    結果: {res}  x={robot._wx:.2f} y={robot._wy:.2f} 回り込み{wc.auto.detours}回  所要{dt:.1f}秒")
    check(res.startswith("完了") and wc.auto.detours == 1, "回り込み1回で完了した")
    check(abs(robot._wy) < 0.2, f"元の経路へ戻った(横ずれ y={robot._wy:+.3f} 期待±0.2)")
    check(abs(robot._wx - (4.5 - 0.1 - 0.6)) < 0.3, f"壁の手前で停止 x={robot._wx:.2f} (期待3.80±0.3)")
    cl, side = _clearance(tk, box)
    check(cl is not None and 0.05 < cl < 0.45 and side > 0,
          f"箱の左をギリギリで通る: 最小クリアランス {cl}m 側={side} (期待 0.05〜0.45・左=正)")
    wc.close()


def test_detour_wide():
    print("--- 4b. 幅1.3mの箱(横へ1m寄る必要がある。2026-09-07 の実機の物)を左から回り込む ---")
    box = dict(x=2.2, y=0.0, w=1.3, d=0.3, h=1.0)
    boxes = [box, dict(x=4.6, y=0.0, w=6.0, d=0.2, h=1.0),
             dict(x=2.3, y=2.5, w=0.2, d=7.0, h=1.0), dict(x=2.3, y=-2.5, w=0.2, d=7.0, h=1.0)]   # 左右の壁
    robot, tk, wc = _setup(boxes, dict(v_fwd=0.5, stop_dist=0.6, mode="forward", max_fwd=8.0, avoid=True))
    check(wc.start_auto(), "前進 開始")
    dt = _wait(wc, 180)
    tk.on = False
    res = wc.auto.result
    print(f"    結果: {res}  x={robot._wx:.2f} y={robot._wy:.2f} 回り込み{wc.auto.detours}回  所要{dt:.1f}秒")
    check(res.startswith("完了") and wc.auto.detours == 1, "回り込み1回で完了した")
    check(abs(robot._wx - (4.6 - 0.1 - 0.6)) < 0.3, f"壁の手前で停止 x={robot._wx:.2f} (期待3.90±0.3)")
    check(abs(robot._wy) < 0.2, f"元の経路へ戻った(横ずれ y={robot._wy:+.3f} 期待±0.2)")
    cl, side = _clearance(tk, box)
    check(cl is not None and 0.05 < cl < 0.45 and side > 0,
          f"箱の左を通る: 最小クリアランス {cl}m 側={side} (期待 0.05〜0.45・左=正)")
    wc.close()


def test_detour_blocked():
    print("--- 4c. 幅1.3mの箱の両脇が 0.35m しか空いていない(体が通らない) → 手前で止まる ---")
    box = dict(x=2.0, y=0.0, w=1.3, d=0.3, h=1.0)
    boxes = [box, dict(x=2.0, y=1.05, w=0.1, d=6.0, h=1.0), dict(x=2.0, y=-1.05, w=0.1, d=6.0, h=1.0),
             dict(x=4.5, y=0.0, w=2.0, d=0.2, h=1.0)]
    robot, tk, wc = _setup(boxes, dict(v_fwd=0.5, stop_dist=0.6, mode="forward", max_fwd=8.0, avoid=True))
    check(wc.start_auto(), "前進 開始")
    dt = _wait(wc, 120)
    tk.on = False
    res = wc.auto.result
    print(f"    結果: {res}  x={robot._wx:.2f} y={robot._wy:.2f} 回り込み{wc.auto.detours}回  所要{dt:.1f}秒")
    check(res.startswith("完了") and wc.auto.detours == 0, "回り込まずに完了した")
    check(abs(robot._wx - (2.0 - 0.15 - 0.6)) < 0.3, f"箱の手前で停止 x={robot._wx:.2f} (期待1.25±0.3)")
    check(abs(robot._wy) < 0.25, f"横へ逃げていない(y={robot._wy:+.3f})")
    wc.close()


def test_detour_two():
    print("--- 4d. 箱が 2 つ(右寄りと左寄りに互い違い) → 両方を回り込んで壁の手前で止まる ---")
    b1 = dict(x=1.8, y=-0.3, w=0.7, d=0.3, h=1.0)
    b2 = dict(x=3.4, y=0.45, w=0.7, d=0.3, h=1.0)
    boxes = [b1, b2, dict(x=5.6, y=0.0, w=6.0, d=0.2, h=1.0),
             dict(x=2.5, y=2.5, w=0.2, d=8.0, h=1.0), dict(x=2.5, y=-2.5, w=0.2, d=8.0, h=1.0)]
    robot, tk, wc = _setup(boxes, dict(v_fwd=0.5, stop_dist=0.6, mode="forward", max_fwd=8.0, avoid=True))
    check(wc.start_auto(), "前進 開始")
    dt = _wait(wc, 240)
    tk.on = False
    res = wc.auto.result
    print(f"    結果: {res}  x={robot._wx:.2f} y={robot._wy:.2f} 回り込み{wc.auto.detours}回  所要{dt:.1f}秒")
    check(res.startswith("完了") and wc.auto.detours >= 1, "回り込んで完了した")
    check(abs(robot._wx - (5.6 - 0.1 - 0.6)) < 0.3, f"壁の手前で停止 x={robot._wx:.2f} (期待4.90±0.3)")
    check(abs(robot._wy) < 0.2, f"元の経路へ戻った(横ずれ y={robot._wy:+.3f} 期待±0.2)")
    c1, _s1 = _clearance(tk, b1)
    c2, _s2 = _clearance(tk, b2)
    check(c1 is not None and c1 > 0.05 and c2 is not None and c2 > 0.05,
          f"どちらの箱にも当たらない(クリアランス {c1} / {c2})")
    wc.close()


def test_side():
    print("--- 5. 小刻みステップ: 横 0.5m / 右へ5cm / 後ろへ10cm / 1歩 / 歩かないときの中止 ---")
    robot, tk, wc = _setup([dict(x=3.0, y=0.0, w=4.0, d=0.2, h=1.0)],
                           dict(v_side=0.5, mode="side", side_dir="left", side_dist=0.5))
    check(wc.start_auto(), "横歩き 0.5m 開始")
    dt = _wait(wc, 90)
    print(f"    結果: {wc.auto.result}  y={robot._wy:.3f}  {wc.auto.steps}歩 所要{dt:.1f}秒")
    check(wc.auto.result.startswith("完了") and abs(robot._wy - 0.5) < 0.06,
          f"左へ0.50m: y={robot._wy:.3f} (期待0.50±0.06)")
    check(wc.auto.steps == 0, f"0.5m は普通の歩行(小刻みステップ 0 回): 歩数 {wc.auto.steps}")
    y1 = robot._wy
    check(wc.start_auto(dict(side_dir="right", side_dist=0.05)), "微調整 右へ5cm 開始")
    dt = _wait(wc, 60)
    print(f"    結果: {wc.auto.result}  y={robot._wy:.3f}  {wc.auto.steps}歩 所要{dt:.1f}秒")
    check(wc.auto.result.startswith("完了") and abs((y1 - robot._wy) - 0.05) < 0.05,
          f"右へ5cm: 移動 {(y1 - robot._wy) * 100:.1f}cm (期待5±5)")
    x1 = robot._wx
    check(wc.start_auto(dict(mode="back", back_dist=0.10)), "後退 10cm 開始(椅子との距離を詰める)")
    dt = _wait(wc, 60)
    print(f"    結果: {wc.auto.result}  x={robot._wx:.3f}  {wc.auto.steps}歩 所要{dt:.1f}秒")
    check(wc.auto.result.startswith("完了") and abs((x1 - robot._wx) - 0.10) < 0.05,
          f"後ろへ10cm: 移動 {(x1 - robot._wx) * 100:.1f}cm (期待10±5)")
    # 1歩だけ
    y2 = robot._wy
    check(wc.start_auto(dict(mode="step", step_dir="left")), "[左へ1歩] 開始")
    dt = _wait(wc, 30)
    print(f"    結果: {wc.auto.result}  移動 {(robot._wy - y2) * 100:.1f}cm  所要{dt:.1f}秒")
    check(wc.auto.result.startswith("完了") and "1歩" in wc.auto.result
          and 0.008 < (robot._wy - y2) < 0.20, f"1歩で 1〜20cm 動く: {(robot._wy - y2) * 100:.1f}cm")
    st = wc.status()
    check(st.get("step_last_cm") is not None and st["step_last_cm"] > 0.8, f"状態に1歩の移動量 {st.get('step_last_cm')}cm")
    # 指令は受理されるのに歩かない機体(2026-09-04 11:47 の実機) → 3歩で中止して理由を出す
    orig = robot.set_velocity
    robot.set_velocity = lambda vx, vy, om, duration=0.5: True
    try:
        check(wc.start_auto(dict(mode="back", back_dist=0.10)), "歩かない機体で後退 開始(指令は受理される)")
        dt = _wait(wc, 60)
        print(f"    結果: {wc.auto.result}  所要{dt:.1f}秒")
        check(wc.auto.result.startswith("中止") and "応じていない" in wc.auto.result,
              "歩かないときは3歩で中止し『速度指令に応じていない』と出す")
    finally:
        robot.set_velocity = orig
    tk.on = False
    wc.close()


def test_side_wall():
    print("--- 5a. 壁の手前 0.6m で横歩き(2026-09-07 実機: 正面の壁で『横方向に障害物』になり動かなかった) ---")
    # 壁は前面が x=0.65(幅 4m)。右へ 1.0m
    robot, tk, wc = _setup([dict(x=0.75, y=0.0, w=4.0, d=0.2, h=1.0)],
                           dict(v_side=0.5, mode="side", side_dir="right", side_dist=1.0))
    st = wc.status()
    print(f"    開始時: 前方 {st['dist']}m 真横 左{st['side_free_l']}/右{st['side_free_r']} 斜め前 左{st['side_fwd_l']}/右{st['side_fwd_r']}")
    check(wc.start_auto(), "横歩き 右 1.0m 開始")
    dt = _wait(wc, 90)
    print(f"    結果: {wc.auto.result}  y={robot._wy:.3f}  所要{dt:.1f}秒")
    check(wc.auto.result.startswith("完了") and abs(robot._wy + 1.0) < 0.10,
          f"壁の手前でも右へ 1.0m 歩けた: y={robot._wy:+.3f} (期待 -1.00±0.10)")
    wc.close()
    print("--- 5a-2. 右 0.9m に椅子(幅 0.4m)があるときの右へ 1.0m → 椅子の手前で止まる ---")
    robot, tk, wc = _setup([dict(x=0.75, y=0.0, w=4.0, d=0.2, h=1.0), dict(x=0.0, y=-0.9, w=0.4, d=0.4, h=0.9)],
                           dict(v_side=0.5, mode="side", side_dir="right", side_dist=1.0))
    check(wc.start_auto(), "横歩き 右 1.0m 開始(椅子あり)")
    dt = _wait(wc, 90)
    print(f"    結果: {wc.auto.result}  y={robot._wy:.3f}  所要{dt:.1f}秒")
    # 椅子の近い端 y=-0.7。体の半幅 0.25 + 余白 0.10 = 0.35 手前(y≈-0.35)までなら OK
    check(wc.auto.result.startswith("中止") and -0.45 < robot._wy < -0.05,
          f"椅子の手前で止まった: y={robot._wy:+.3f} (期待 -0.45〜-0.05、椅子の端 -0.70)")
    wc.close()


def test_align():
    print("--- 5b. 斜めの壁: 最初に回転して正対してから前進 ---")
    robot, tk, wc = _setup([dict(x=2.5, y=0.0, w=4.0, d=0.2, h=1.0)],
                           dict(v_fwd=0.5, stop_dist=0.6, mode="forward", max_fwd=6.0))
    robot._wyaw = math.radians(20.0)              # 機体が壁に対して 20 度斜めに立っている
    time.sleep(0.6)                               # モックの判定が新しい向きで更新されるのを待つ
    check(wc.start_auto(), "前進 開始(壁が 20° 斜め)")
    dt = _wait(wc, 90)
    yaw_end = math.degrees(robot._wyaw)
    print(f"    結果: {wc.auto.result}  yaw={yaw_end:+.1f}°  x={robot._wx:.2f} y={robot._wy:.2f}  所要{dt:.1f}秒")
    check(abs(yaw_end) < 5.0, f"正対した: 終了時のヨー {yaw_end:+.1f}° (期待 ±5°)")
    check(wc.auto.result.startswith("完了") and abs(robot._wx - (2.5 - 0.6)) < 0.3,
          f"正対後に壁の手前で停止 x={robot._wx:.2f} (期待1.90±0.3)")
    tk.on = False
    wc.close()


def test_turn():
    print("--- 5c. 指定角度の旋回(その場で +90° / −45°、IMU のヨーで止める) ---")
    robot, tk, wc = _setup([dict(x=2.5, y=0.0, w=4.0, d=0.2, h=1.0)], dict(v_fwd=0.5, stop_dist=0.6))
    wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
    for deg in (90.0, -45.0):
        y0 = robot._wyaw
        check(wc.start_auto(dict(mode="turn", turn_deg=deg)), f"旋回 開始({deg:+.0f}°)")
        dt = _wait(wc, 60)
        turned = math.degrees(wrap(robot._wyaw - y0))
        print(f"    結果: {wc.auto.result}  回転 {turned:+.1f}°  所要{dt:.1f}秒")
        check(wc.auto.result.startswith("完了") and abs(turned - deg) <= 4.0,
              f"{deg:+.0f}° 回った(実測 {turned:+.1f}°、許容 ±4°)")
        time.sleep(0.5)
    tk.on = False
    wc.close()


def test_restart():
    print("--- 5c. 止まった後(歩行 FSM のまま)に、もう一度前進できる ---")
    robot, tk, wc = _setup([dict(x=2.0, y=0.0, w=4.0, d=0.2, h=1.0)],
                           dict(v_fwd=0.7, stop_dist=0.6, mode="forward", max_fwd=6.0))
    check(wc.start_auto(), "前進 開始")
    dt = _wait(wc, 90)
    print(f"    結果: {wc.auto.result}  FSM={robot.get_fsm_id()} fsm_mode={robot.get_fsm_mode()}  x={robot._wx:.2f}  所要{dt:.1f}秒")
    check(wc.auto.result.startswith("完了") and robot.get_fsm_id() == 501 and robot.get_fsm_mode() == 0,
          f"止まった後: FSM 501 のまま、fsm_mode 0(いま {robot.get_fsm_id()}/{robot.get_fsm_mode()})")
    robot.sim_obstacles([dict(x=robot._wx + 2.0, y=0.0, w=4.0, d=0.2, h=1.0)])
    time.sleep(0.6)
    check(wc.start_auto(), "2 回目の前進 開始")
    dt = _wait(wc, 90)
    print(f"    結果: {wc.auto.result}  所要{dt:.1f}秒")
    check(wc.auto.result.startswith("完了"), "2 回目も完了")
    tk.on = False
    wc.close()


def test_hb_loss():
    print("--- 6. ハートビート途絶 ---")
    robot, tk, wc = _setup([dict(x=2.0, y=0.0, w=4.0, d=0.2, h=1.0)],
                           dict(v_fwd=0.5, mode="forward"), hb_ok=False)
    check(wc.start_auto(), "開始要求")
    _wait(wc, 10)
    tk.on = False
    res = wc.auto.result
    check("ハートビート" in res, f"ハートビート途絶で中止した: {res}")
    check(wc.sender.last_sent == (0.0, 0.0, 0.0), "中止後は速度ゼロ")
    wc.close()


if __name__ == "__main__":
    test_detector()
    test_deadman()
    test_wall()
    test_detour()
    test_detour_wide()
    test_detour_blocked()
    test_detour_two()
    test_side()
    test_side_wall()
    test_align()
    test_turn()
    test_restart()
    test_hb_loss()
    print("=" * 60)
    if FAIL:
        print(f"NG {len(FAIL)} 件:")
        for f in FAIL:
            print("  - " + f)
        sys.exit(1)
    print("全項目 OK")
