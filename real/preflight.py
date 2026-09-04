#!/usr/bin/env python3
"""現場へ行く前に、このPCで実機投入の前提が揃っているかを全部確かめる。

  python3 real/preflight.py                 # 実機なしで通る検査だけ
  python3 real/preflight.py --iface enp46s0 # ネットワークと残留プロセスも見る

**「フォルダだけで自己完結」は正しくない。**方策(deploy/)とモデル(model/)は
自己完結しているが、Python環境(torch / mujoco / unitree_sdk2py)は現場PCに
入っていなければならない。現場でpip installはしない。ここで先に落とす。

出力の見方: [OK] 合格 / [警告] 進めてよいが把握しておく / [NG] 直すまで実機に出ない
"""
import argparse
import importlib
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

N_OK = [0]
N_WARN = [0]
N_NG = [0]


def ok(msg):
    N_OK[0] += 1
    print(f"  [OK]   {msg}")


def warn(msg):
    N_WARN[0] += 1
    print(f"  [警告] {msg}")


def ng(msg):
    N_NG[0] += 1
    print(f"  [NG]   {msg}")


def head(t):
    print(f"\n--- {t} " + "-" * max(0, 58 - len(t)))


# ---------------------------------------------------------------- 1. 環境
def check_env():
    head("1. Python環境(現場でpip installしないための確認)")
    if sys.version_info < (3, 10):
        ng(f"Python {sys.version_info.major}.{sys.version_info.minor} — 3.10以上が要る")
    else:
        ok(f"Python {sys.version.split()[0]}")
    for m, need in (("numpy", True), ("torch", True), ("mujoco", True),
                    ("unitree_sdk2py", True), ("cv2", False)):
        try:
            mod = importlib.import_module(m)
            ok(f"{m} {getattr(mod, '__version__', '')}".rstrip())
        except Exception as e:                     # noqa: BLE001
            (ng if need else warn)(
                f"{m} が無い: {e}"
                + ("" if need else "(simの映像だけが出なくなる)"))


# ---------------------------------------------------------------- 2. ログ
def check_logs():
    head("2. ログの書き込み先")
    d = ROOT / "logs" / "real"
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / ".preflight_write_test"
        p.write_text("ok", encoding="utf-8")
        p.unlink()
        ok(f"{d} に書ける")
    except Exception as e:                         # noqa: BLE001
        ng(f"{d} に書けない: {e}  → chmod で権限を直すこと"
           f"(実行権が無いディレクトリだとログが1本も残らない)")


# ---------------------------------------------------------------- 3. 方策
def check_deploy():
    head("3. 方策パッケージ(deploy/)")
    import numpy as np
    import torch
    from run_fsm import ACTION_SCALE, CONTROL_HZ

    dirs = [d for d in sorted((ROOT / "deploy").iterdir())
            if (d / "policy.pt").exists()]
    if not dirs:
        ng("deploy/ に方策が1つも無い")
        return []
    ok(f"方策 {len(dirs)}件: " + ", ".join(d.name for d in dirs))

    # 実機で失敗の実績があるもの。**消さない**(比較・再現に要る)。
    # 意図して置いてあるので NG ではなく警告。UIでも選ぶ前に警告が出る。
    risky = {"sit_up_rd_r2": "2026-08-24 の実機で横に倒れた(ゲート未通過)",
             "sit_up_slow_r2": "現行条件の再測定で8回中5回が早落ち(成功率40%)"}
    for d in dirs:
        if d.name in risky:
            warn(f"{d.name} — {risky[d.name]}。"
                 f"UIには出るが、実行前に確認ダイアログで警告する")

    names0 = None
    pols = []
    for d in dirs:
        z = np.load(d / "reference.npz")
        nm = [str(s) for s in z["joint_names"]]
        if names0 is None:
            names0 = nm
        elif nm != names0:
            ng(f"{d.name}: 関節順が他と違う(方策ごとに並びが違うと四肢が暴れる)")
            continue
        bad = []
        for k in ("ref_q", "kp", "kd"):
            v = np.asarray(z[k], dtype=float)
            if not np.all(np.isfinite(v)):
                bad.append(k)
        if bad:
            ng(f"{d.name}: {','.join(bad)} にNaN/Infがある")
            continue
        rq = z["ref_q"]
        v = np.abs(np.diff(rq, axis=0)) * CONTROL_HZ
        vmax = float(v.max())
        meta_ok = (d / "meta.json").exists()
        if not meta_ok:
            ng(f"{d.name}: meta.json が無い")
        pols.append((d.name, rq, z, vmax))
        line = (f"{d.name:16s} {len(rq):4d}コマ/{len(rq)/CONTROL_HZ:4.1f}秒 "
                f"参照の最大関節速度 {vmax:5.1f} rad/s")
        # 安全停止しきい値(VEL_HARD=32)に対する余裕
        from cockpit import VEL_HARD
        if vmax > 0.6 * VEL_HARD:
            warn(line + f" — 停止しきい値{VEL_HARD:.0f}に対し余裕が"
                        f"{VEL_HARD - vmax:.1f}rad/sしかない")
        else:
            ok(line)
    return pols


# ---------------------------------------------------------------- 4. 可動域
def check_limits(pols):
    head("4. 可動域と目標ガード(action飽和時に何が起きるか)")
    import numpy as np
    from real_robot import GUARD_RANGE_MARGIN, _load_joint_limits
    from run_fsm import ACTION_SCALE

    if not pols:
        return
    z0 = pols[0][2]
    names = [str(s) for s in z0["joint_names"]]
    lo, hi = _load_joint_limits(names)
    if lo is None:
        ng("MJCFから可動域を読めない → 目標ガードが無効のまま走ることになる")
        return
    ok(f"MJCFから29関節の可動域を取得(ガード余裕 ±{GUARD_RANGE_MARGIN}rad)")
    worst = {}
    for name, rq, z, _ in pols:
        over = np.maximum(rq + ACTION_SCALE - hi, lo - (rq - ACTION_SCALE))
        for i in range(len(names)):
            m = float(over[:, i].max())
            if m > 0:
                worst[names[i]] = max(worst.get(names[i], 0.0), m)
    if not worst:
        ok("action±1でも全関節が可動域内")
        return
    hardest = max(worst.values())
    warn(f"actionが飽和すると{len(worst)}関節が可動域を出る"
         f"(=機械的ストッパへ押し付ける。シムでも同じことが起きている):")
    for k, v in sorted(worst.items(), key=lambda x: -x[1])[:5]:
        print(f"         {k:22s} 最大 {v:+.3f} rad はみ出す")
    if hardest <= GUARD_RANGE_MARGIN:
        ok(f"はみ出しの最大{hardest:.2f}radはガード余裕{GUARD_RANGE_MARGIN}rad"
           f"の内側 — ガードは正常な指令を書き換えない(数えるだけ)")
    else:
        ng(f"はみ出し{hardest:.2f}radがガード余裕{GUARD_RANGE_MARGIN}radを"
           f"超える — ガードが正常な方策の指令を書き換えてしまう")


# ---------------------------------------------------------------- 5. 観測
def check_obs(pols):
    head("5. 観測構築と推論(MJCFの関節名対応・次元・NaN)")
    import numpy as np
    from run_fsm import ObsBuilder, Policy

    for name, _rq, _z, _v in pols:
        try:
            pol = Policy(name)
            ob = ObsBuilder(pol)
            q, dq = pol.ref_q[0].copy(), np.zeros(29)
            quat = np.array([1.0, 0, 0, 0])
            gyro = np.zeros(3)
            ob.reset(est_xy=pol.ref["ref_xy_abs"][0][:2])
            t0 = time.perf_counter()
            obs = ob.build(pol, 0, q, dq, quat, gyro)
            a = pol.act(obs)
            ms = (time.perf_counter() - t0) * 1000
            dim = int(pol.meta.get("obs_dim", 0))
            msg = (f"{name:16s} 観測{len(obs)}次元 / 出力{len(a)}次元 / "
                   f"{ms:5.1f}ms")
            if dim and len(obs) != dim:
                ng(msg + f" — meta.json の obs_dim={dim} と食い違う")
            elif not (np.all(np.isfinite(obs)) and np.all(np.isfinite(a))):
                ng(msg + " — NaN/Infが出た")
            elif float(np.abs(a).max()) > 1.0 + 1e-6:
                ng(msg + f" — 出力が±1を超える({np.abs(a).max():.3f})")
            else:
                ok(msg)
        except Exception as e:                     # noqa: BLE001
            ng(f"{name}: {type(e).__name__}: {e}")


# ------------------------------------------------------------- 5b. ヨー合わせ
def check_yaw_align(pols):
    """観測に入る「体の向き」がワールド座標なので、実機IMUのヨー原点が
    ずれていると方策は学習していない課題を解かされる。

    2026-08-27の実機13本で、参照とのヨー差と結果は 符号つき相関 r=0.83
    (差 -23度→最大傾き23度で完走 / +90度→膝の左右差1.16rad)。
    シムでIMUのヨー原点だけをずらす試験では、合わせ無しで +90度のとき
    終端傾きが18.5→3.9度・膝左右差0.11→0.70に化けた(=座らずに捻れる)。
    合わせを入れるとどのずれでも結果が一致する。**その一致を守る番人。**
    """
    head("5b. ヨー合わせ(IMUのヨー原点ずれの打ち消し)")
    import numpy as np
    from run_fsm import ObsBuilder, Policy, _yaw_of

    def qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    name = pols[0][0] if pols else "sit_up_dp4_r2"
    try:
        pol = Policy(name)
        ref_quat = np.asarray(pol.ref["ref_quat"][0], dtype=float)
        q, dq, gyro = pol.ref_q[0].copy(), np.zeros(29), np.zeros(3)
        base = None
        worst = 0.0
        for deg in (0.0, -44.0, 42.0, 90.0, 179.0, -179.0):
            rq = np.array([np.cos(np.radians(deg) / 2), 0, 0,
                           np.sin(np.radians(deg) / 2)])
            quat = qmul(rq, ref_quat)
            ob = ObsBuilder(pol)
            off = ob.reset(est_xy=pol.ref["ref_xy_abs"][0][:2],
                           quat=quat, ref_quat=ref_quat)
            # 打ち消せているか: 合わせ後のヨーが参照と一致すること
            resid = abs(np.degrees(_yaw_of(quat) + off - _yaw_of(ref_quat)))
            resid = min(resid % 360.0, 360.0 - resid % 360.0)
            a = pol.act(ob.build(pol, 0, q, dq, quat, gyro))
            if base is None:
                base = a
            worst = max(worst, resid, float(np.abs(a - base).max()) * 1e3)
            if resid > 0.01:
                ng(f"{deg:+6.0f}度のずれを打ち消せていない(残り{resid:.2f}度)")
                return
            if float(np.abs(a - base).max()) > 1e-4:
                ng(f"{deg:+6.0f}度ずらすと出力が変わる"
                   f"(最大{np.abs(a - base).max():.4f})— 合わせが効いていない")
                return
        ok(f"{name}: ±179度までのヨーずれを打ち消し、出力が一致した")
        # 合わせを切ったら**ちゃんと壊れる**こと(試験が空回りしていない証拠)
        ob = ObsBuilder(pol)
        ob.reset(est_xy=pol.ref["ref_xy_abs"][0][:2])
        rq = np.array([np.cos(np.radians(90.0) / 2), 0, 0,
                       np.sin(np.radians(90.0) / 2)])
        a2 = pol.act(ob.build(pol, 0, q, dq, qmul(rq, ref_quat), gyro))
        d = float(np.abs(a2 - base).max())
        if d < 1e-3:
            ng(f"合わせ無しでも出力が変わらない({d:.5f})— "
               f"観測に体の向きが入っていない疑い。試験が意味を成していない")
        else:
            ok(f"合わせ無しなら90度で出力が変わる(最大{d:.3f})— 試験は生きている")
    except Exception as e:                         # noqa: BLE001
        ng(f"{name}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- 6. ガード
def check_guards():
    head("6. 目標ガードの動作(実機に出る指令の最後の関所)")
    import numpy as np
    from real_robot import (GUARD_RANGE_MARGIN, GUARD_STEP_MAX, RealRobot,
                            _load_joint_limits)

    # DDSに触らずガードだけを試す
    r = RealRobot.__new__(RealRobot)
    import threading
    r.lock = threading.Lock()
    r.target_q = np.zeros(29)
    r.kp = np.zeros(29)
    r.kd = np.zeros(29)
    r._estop_latched = False
    r.guard_n_clip = r.guard_n_rate = r.guard_n_nan = r.guard_n_over = 0
    r.guard_over_max = 0.0
    r._nan_streak = 0
    r.q_lo, r.q_hi = _load_joint_limits()
    if r.q_lo is None:
        ng("可動域を読めないためガード検査を飛ばす")
        return

    good, why = r.set_target(np.zeros(29), np.full(29, 100.0),
                             np.full(29, 3.0), latch=True)
    ok("正常な指令は通る") if good else ng(f"正常な指令が拒否された: {why}")

    q = np.zeros(29); q[3] = np.nan
    good, why = r.set_target(q, np.full(29, 100.0), np.full(29, 3.0))
    if not good and r.guard_n_nan == 1:
        ok(f"NaNを拒否し、直前の目標を保持({why})")
    else:
        ng("NaNを含む指令が通ってしまった")

    r.guard_n_over = 0; r.guard_over_max = 0.0
    q = np.zeros(29); q[7] = 99.0                  # 可動域の遥か外(壊れた指令)
    r.set_target(q, np.full(29, 100.0), np.full(29, 3.0), latch=True)
    lim = float(r.q_hi[7] + GUARD_RANGE_MARGIN)
    if abs(float(r.target_q[7]) - lim) < 1e-9:
        ok(f"可動域外を +{GUARD_RANGE_MARGIN}rad の余裕まで丸めた"
           f"({q[7]:.0f} → {r.target_q[7]:.2f})")
    else:
        ng(f"可動域ガードが効いていない({r.target_q[7]})")

    r.set_target(np.zeros(29), np.full(29, 100.0), np.full(29, 3.0),
                 latch=True)
    q = np.zeros(29); q[3] = 2.5                   # 1ステップで2.5rad飛ぶ
    q[3] = min(q[3], float(r.q_hi[3]))             # 可動域ガードと分離して見る
    q[3] = 2.5 if r.q_hi[3] >= 2.5 else q[3]
    r.set_target(q, np.full(29, 100.0), np.full(29, 3.0))
    if abs(float(r.target_q[3]) - GUARD_STEP_MAX) < 1e-9:
        ok(f"1ステップの飛びを {GUARD_STEP_MAX}rad に制限した")
    else:
        ng(f"変化量ガードが効いていない({r.target_q[3]})")

    r._estop_latched = True
    good, why = r.set_target(np.zeros(29), np.full(29, 100.0),
                             np.full(29, 3.0), latch=True)
    ok("E-STOPラッチ中は指令を受け付けない") if not good else \
        ng("E-STOPラッチ中に指令が通ってしまった")
    # ゲインを下げるガードが無いこと(下げると支持力が消えて事故になる)
    r._estop_latched = False
    r.set_target(np.zeros(29), np.full(29, 100.0), np.full(29, 3.0),
                 latch=True)
    ok("ガードはゲインに触らない" if float(r.kp.max()) == 100.0
       else "★ゲインが書き換わった")


# ---------------------------------------------------------------- 6b. UI
def check_ui():
    """コックピットのHTML/JSが壊れていないかを機械的に確かめる。

    ★2026-08-26に**3回**同じ壊し方をした:
      `PAGE` は Python の三重引用符文字列なので、そこへ `\n` と書くと
      **Pythonが解釈して本物の改行になる**。JSの文字列やコメントの途中で
      改行が入ると script 全体が構文エラーになり、**画面が完全に固まる**
      (数値が更新されず、ボタンも効かない)。JSへ改行を渡したいときは
      `\\n` と書くか、`String.fromCharCode(10)` を使う。
    """
    import re
    head("6b. UI(HTML/JS)の健全性")
    src = (HERE / "cockpit.py").read_text(encoding="utf-8")
    # ★2026-09-04 かんたん画面(PAGE_SIMPLE)も同じ検査に掛ける
    for marker in ('PAGE = """', 'PAGE_SIMPLE = """'):
        i = src.index(marker)
        j = src.index('"""', i + len(marker))
        page_src = src[i:j]
        # Python が解釈するエスケープ(バックスラッシュ+n/t/r/x??/u????/0)が
        # 三重引用符の中にあると本物の文字に化けてHTML/JSが壊れる。全部見る。
        bad = []
        for m in re.finditer(r'(?<!\\)\\(n|t|r|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|0)',
                             page_src):
            bad.append((page_src[:m.start()].count("\n") + 1, m.group(0)))
        if bad:
            ng(f"{marker[:-6]} の中に Python が解釈するエスケープがある {bad[:6]} — "
               f"本物の文字に化けてHTML/JSが壊れる。二重にする(\\\\n)か "
               f"String.fromCharCode() を使うこと")
        else:
            ok(f"{marker[:-6]} に危険なエスケープは無い")

    sys.path.insert(0, str(HERE))
    import cockpit                                  # noqa: E402
    for page_name in ("PAGE", "PAGE_SIMPLE"):
        _check_page_js(getattr(cockpit, page_name), page_name)


def _check_page_js(cockpit_page, page_name):
    import re
    js = cockpit_page.split("<script>")[1].split("</script>")[0]
    def code_part(line):
        """行の中で、文字列の外にある // 以降を落とす(http:// を誤検出しない)"""
        q = None
        for k, ch in enumerate(line):
            if q:
                if ch == "\\":
                    q = q                          # 次の1文字はエスケープ
                elif ch == q:
                    q = None
            elif ch in "'\"`":
                q = ch
            elif ch == "/" and k + 1 < len(line) and line[k + 1] == "/":
                return line[:k]
        return line

    # ★インライン属性(onclick等)も見る。<script>の外なので前の検査に掛からない
    inline = []
    for m in re.finditer(r'\son(click|change|keydown)="([^"]*)"', cockpit_page):
        body = m.group(2)
        if "\n" in body:
            inline.append((m.group(1), body[:40].replace("\n", "⏎")))
        elif body.count("'") % 2:
            inline.append((m.group(1), body[:40]))
    if inline:
        ng(f"{page_name}: インライン属性が壊れている(改行 or 引用符の不一致): {inline[:4]}")
    else:
        ok(f"{page_name}: インライン属性(onclick等)は健全")

    lines = js.split("\n")
    broken = []
    for n, l in enumerate(lines, 1):
        code = code_part(l)
        if code.count("'") % 2 or code.count("`") % 2:
            broken.append((n, l.strip()[:50]))
    if broken:
        ng(f"{page_name}: JSの文字列が行内で閉じていない: {broken}")
    else:
        ok(f"{page_name}: JSの文字列リテラルは全て閉じている({len(lines)}行)")
    # コメント行の次の行が「日本語だけ」なら、改行で割れた疑い
    susp = [(n, lines[n].strip()[:40]) for n in range(len(lines) - 1)
            if "//" in lines[n] and n + 1 < len(lines)
            and re.match(r'^[ぁ-んァ-ン一-龥]+$', lines[n + 1].strip())]
    if susp:
        ng(f"{page_name}: コメントが改行で割れている疑い: {susp}")
    else:
        ok(f"{page_name}: コメントが改行で割れていない")
    needs = (("function askRun", "function tick", "setInterval(tick") if page_name == "PAGE"
             else ("function walkGo", "function sitGo", "function tick", "setInterval(tick"))
    for need in needs:
        (ok if need in js else ng)(f"{page_name}: {need} がある" if need in js
                                   else f"★{page_name}: {need} が無い")


# ---------------------------------------------------------------- 6c. 配線
def check_ui_wiring():
    """UIのボタンが**実際にサーバへ届くか**を照合する。

    ★2026-08-27。do_POST の許可リストに入れ忘れると、そのコマンドは
      黙って捨てられる。[補助][ヨー合わせ][打ち切りコマ数]の3つがこれで、
      UIで選んでも何も起きないのに、画面上はセレクトボックスが動くので
      操作者には「効いている」ように見えていた。**半日気づかなかった。**
      押しても効かないボタンは、無いより悪い。
    """
    import re
    head("6c. UIのボタンとサーバの配線")
    src = (HERE / "cockpit.py").read_text(encoding="utf-8")
    # ★2026-09-04 詳細画面(PAGE)とかんたん画面(PAGE_SIMPLE)の両方を見る
    pages = {}
    for marker in ('PAGE = """', 'PAGE_SIMPLE = """'):
        i = src.index(marker)
        j = src.index('"""', i + len(marker))
        pages[marker[:-6]] = src[i:j]
    page = "\n".join(pages.values())
    used = set(re.findall(r"cmd\(\s*'([a-z_]+)'", page))
    m = re.search(r"CMD_ALLOW = \(([^)]*)\)", src, re.S)
    allow = set(re.findall(r'"([a-z_]+)"', m.group(1))) if m else set()
    special = {"estop", "damp", "select", "beat"}  # do_POST で特別扱い
    handled = set(re.findall(r'cmd == "([a-z_]+)"', src))
    missing = sorted(used - allow - special)
    if missing:
        ng(f"UIから呼ばれているのにサーバが受け付けないコマンド: {missing}"
           f" — cockpit.py の CMD_ALLOW に足すこと")
    else:
        ok(f"UIの全コマンド({len(used)}個)がサーバに届く")
    # cmd.startswith("mode_") のような前方一致の分岐も拾う
    prefixes = set(re.findall(r'cmd\.startswith\("([a-z_]+)"\)', src))
    handled |= {c for c in allow if any(c.startswith(x) for x in prefixes)}
    nohandler = sorted(allow - handled)
    if nohandler:
        ng(f"許可されているのに処理する分岐が無い: {nohandler}")
    else:
        ok("許可された全コマンドに処理の分岐がある")
    dead = sorted(allow - used)
    if dead:
        warn(f"サーバは受け付けるがUIから呼ばれていない: {dead}")
    for pname, psrc in pages.items():
        ids = set(re.findall(r'id="([a-zA-Z_]+)"', psrc))
        # getElementById('sel_'+k) のような連結は静的には解決できないので外す
        refs = set(re.findall(r"getElementById\(\s*'([a-zA-Z_]+)'\s*\)", psrc))
        miss = sorted(refs - ids)
        if miss:
            ng(f"{pname}: JSが触るのにHTMLに無いid: {miss} — 画面が固まります")
        else:
            ok(f"{pname}: JSが触るid({len(refs)}個)はすべてHTMLにある")


# ---------------------------------------------------------------- 7. 実機側
def check_field(iface):
    head("7. 現場の環境(残留プロセス・ネットワーク)")
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                             text=True, timeout=5).stdout
        hits = [l for l in out.splitlines()
                if any(k in l for k in ("cockpit.py", "run_fsm.py",
                                        "listen_only.py", "probe_release.py"))
                and "preflight" not in l]
        if hits:
            ng("rt/lowcmd を掴みうるプロセスが残っている — "
               "先に落とすこと(残骸が500Hzで送り続けた実績あり):")
            for h in hits:
                print(f"        {h.strip()}")
        else:
            ok("コックピット類の残留プロセスは無い")
    except Exception as e:                         # noqa: BLE001
        warn(f"プロセス確認に失敗: {e}")
    if not iface:
        warn("--iface 未指定のためネットワーク確認は省略")
        return
    try:
        a = subprocess.run(["ip", "-o", "addr", "show", iface],
                           capture_output=True, text=True, timeout=5).stdout
        if "192.168.123." in a:
            ok(f"{iface} に 192.168.123.x が付いている")
        else:
            ng(f"{iface} に 192.168.123.x が無い — "
               f"sudo ip addr add 192.168.123.222/24 dev {iface}")
    except Exception as e:                         # noqa: BLE001
        ng(f"{iface} が見つからない: {e}")
    try:
        p = subprocess.run(["ping", "-c", "2", "-W", "1", "192.168.123.161"],
                           capture_output=True, text=True, timeout=8)
        ok("実機 192.168.123.161 に到達") if p.returncode == 0 else \
            ng("実機 192.168.123.161 に到達しない — 配線とIPを確認")
    except Exception as e:                         # noqa: BLE001
        ng(f"ping に失敗: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="", help="実機NIC名(例 enp46s0)")
    a = ap.parse_args()
    print("=" * 64)
    print(f"実機投入前チェック  {ROOT}")
    print("=" * 64)
    check_env()
    check_logs()
    pols = check_deploy()
    check_limits(pols)
    check_obs(pols)
    check_yaw_align(pols)
    check_guards()
    check_ui()
    check_ui_wiring()
    check_field(a.iface)
    print("\n" + "=" * 64)
    print(f"OK {N_OK[0]} / 警告 {N_WARN[0]} / NG {N_NG[0]}")
    if N_NG[0]:
        print("★NGが残っています。直すまで実機に出さないこと")
        return 1
    print("実機投入前の検査は通りました。"
          "次は `python3 real/cockpit.py --sim` で手順を通すこと")
    return 0


if __name__ == "__main__":
    sys.exit(main())
