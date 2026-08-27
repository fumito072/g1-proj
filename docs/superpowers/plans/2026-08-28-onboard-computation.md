# 機体側コンピューターでの方策演算 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** G1 の操縦PCで動いている方策推論＋制御ループを機体側コンピューター(Jetson)で動かすための最小変更を実装する（アプローチA）。

**Architecture:** `cockpit.py` を機体側で起動し、DDS をループバック化（`--iface` 無し）、HTTP を `0.0.0.0` で公開して PC ブラウザから遠隔操作する。UI が機体から離れるため、遠隔UIハートビート（3秒途絶で自動 damp）を安全機構として追加する。

**Tech Stack:** Python 3.10+ / numpy / torch（JetPack） / mujoco / unitree_sdk2py（cyclonedds）。変更は `real/cockpit.py`・`real/preflight.py`・新規の配備資産のみ。

**Spec:** `docs/superpowers/specs/2026-08-28-onboard-computation-design.md`

## Global Constraints

- 推論ランタイムは**今回 PyTorch `.pt` のまま**。ONNX 化はしない（次段で独立検証）。
- 既存の安全7層（estop_now／100Hz watchdog／目標ガード／傾き40°／関節速度／LowState断0.2s／送信断50ms）は**変更しない**。追加するのはハートビートのみ。
- ハートビート: 既定 `3.0` 秒、`--heartbeat-sec 0` で無効。未受信（`_last_beat is None`）の間は enforce しない。
- **`real/preflight.py` の 6b/6c 検査は NG 0 を維持すること**（配線チェックを壊さない）。
- HTTP 既定バインドは `0.0.0.0`（対象ネットワークは隔離された 192.168.123.x が前提）。
- テストフレームワーク（pytest 等）は導入しない。検証は `preflight.py`（静的検査）と `--sim`＋curl（スモーク）で行う（本リポジトリの既存手法に合わせる）。
- 物理 E-stop ＋ ガントリーは最上位の安全層のまま。ソフト damp はその代替にならない。

## ファイル構造

| ファイル | 変更 | 責務 |
|---|---|---|
| `real/cockpit.py` | 変更 | ハートビート（Engine・do_POST・PAGE JS・main の `--heartbeat-sec`）と HTTP バインド（`--host`） |
| `real/preflight.py` | 変更 | 6c 配線チェックの `special` 集合に `beat` を追加 |
| `requirements_jetpack.txt` | 新規 | 機体側の Python 依存ピン（torch 以外） |
| `setup_onboard.sh` | 新規 | 機体側のセットアップ（venv・依存・import確認） |

---
### Task 1: 遠隔UIハートビート watchdog

**Files:**
- Modify: `real/cockpit.py:299`（Engine コンストラクタ）、`:350`（状態変数）、`:398-400`（`command` の直後に `beat()` 追加）、`:440-456`（`_watchdog`）、`:2281-2286`（`do_POST`）、`:2246`（PAGE JS）、`:2307-2311`＋`:2333`（main の引数と Engine 生成）
- Modify: `real/preflight.py:456`（`special` 集合）

**Interfaces:**
- Consumes: 既存 `Engine._monitoring()`（custom 制御中に True）、`Engine.estop_now(why)`、`Engine.command(cmd, arg)`。
- Produces:
  - `Engine.__init__(self, robot, is_sim, heartbeat_sec=3.0)`
  - `Engine.beat()` → `self._last_beat = time.time()`（HTTP スレッドから直接呼ぶ）
  - `Engine._last_beat`（`None`=未受信）、`Engine._heartbeat_sec`（`0` で無効）
  - `do_POST` が `c == "beat"` を受理
  - `preflight.check_ui_wiring()` の `special` に `"beat"` を含む

- [ ] **Step 1: Engine コンストラクタにハートビート状態を追加**

`real/cockpit.py` の `def __init__(self, robot, is_sim):`（299行）を:
```python
    def __init__(self, robot, is_sim):
```
に変更:
```python
    def __init__(self, robot, is_sim, heartbeat_sec=3.0):
```

次に 350 行の `self._estop_pending = None       # estop_now が立てる。ループが後始末` の直後に 2 行を挿入:
```python
        self._last_beat = None              # UIハートビートの最終受信時刻(未受信=None)
        self._heartbeat_sec = heartbeat_sec # 0で無効
```

- [ ] **Step 2: `beat()` メソッドを追加**

`command` メソッド（398-400行）の直後、`_pop` の前に挿入:
```python
    def beat(self):
        """UIからのハートビート。HTTPスレッドから直接呼ぶ(キューを経由しない)。"""
        self._last_beat = time.time()
```

- [ ] **Step 3: `_watchdog` にハートビート判定を追加**

`_watchdog`（440-456行）の `if why:` ブロックに `continue` とハートビート分岐を追加。現状:
```python
                why = self._safety()
                if why:
                    self.estop_now(why)
            except Exception:                      # noqa: BLE001
                pass
```
を:
```python
                why = self._safety()
                if why:
                    self.estop_now(why)
                    continue
                if (self._heartbeat_sec > 0 and self._last_beat is not None
                        and time.time() - self._last_beat > self._heartbeat_sec):
                    self.estop_now("UIハートビート途絶")
            except Exception:                      # noqa: BLE001
                pass
```

- [ ] **Step 4: `do_POST` で `beat` を受理**

`do_POST`（2281-2286行）現状:
```python
        if c in ("estop", "damp"):
            # ★キューに載せない。50Hzループが方策読み込みやSDKのRPCで
            #   詰まっていても、ここで即座にdampが出る(2026-08-26)
            why = ("操作者による緊急停止" if c == "estop"
                   else "操作者によるdamp")
            self.engine.estop_now(why)
```
を:
```python
        if c in ("estop", "damp"):
            # ★キューに載せない。50Hzループが方策読み込みやSDKのRPCで
            #   詰まっていても、ここで即座にdampが出る(2026-08-26)
            why = ("操作者による緊急停止" if c == "estop"
                   else "操作者によるdamp")
            self.engine.estop_now(why)
        elif c == "beat":
            self.engine.beat()
```

- [ ] **Step 5: PAGE の JS に毎秒ビートを追加**

`real/cockpit.py` 2246 行の:
```python
setInterval(tick,200);tick();
```
を:
```python
setInterval(tick,200);tick();
setInterval(function(){cmd('beat')},1000);
```

- [ ] **Step 6: preflight の配線チェックに `beat` を反映**

`real/preflight.py` 456 行の:
```python
    special = {"estop", "damp", "select"}          # do_POST で特別扱い
```
を:
```python
    special = {"estop", "damp", "select", "beat"}  # do_POST で特別扱い
```

- [ ] **Step 7: main に `--heartbeat-sec` を追加して Engine へ渡す**

`real/cockpit.py` `main()`（2307-2311行）の:
```python
    ap.add_argument("--port", type=int, default=8090)
```
を:
```python
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--heartbeat-sec", type=float, default=3.0,
                    help="UIハートビートの許容間隔[秒]。0で無効")
```

2333 行の:
```python
    eng = Engine(robot, a.sim)
```
を:
```python
    eng = Engine(robot, a.sim, heartbeat_sec=a.heartbeat_sec)
```

- [ ] **Step 8: 静的検査を通す**

Run: `python real/preflight.py`
Expected: 6c が「UIの全コマンド…がサーバに届く」「許可された全コマンドに処理の分岐がある」で PASS し、NG が増えないこと。特に `missing: {'beat'}` が出ないこと。

- [ ] **Step 9: `--sim` スモークテスト（beat 到達・無効化）**

Run（別端末で `--sim` を起動）: `python real/cockpit.py --sim`
Expected: 起動後、以下で `beat` が通る（HTTP 200）:
```
curl -s -X POST "http://127.0.0.1:8090/cmd?c=beat"
```
さらに `--heartbeat-sec 0` で起動し直した場合も通常動作する（= 無効化が効く）こと:
```
python real/cockpit.py --sim --heartbeat-sec 0
```

- [ ] **Step 10: 手動確認（ハートビート途絶で自動 damp）**

`--sim` で `1. ARM` → `2. START` で `WAIT_CONFIRM` もしくは `RUNNING` にした後、ブラウザのタブを閉じる。
Expected: 約3秒後に `GET /state` の `fsm` が `DAMP`、`msg` に「UIハートビート途絶」が入る。ログ（コンソール）にも `★DAMP: UIハートビート途絶` が出る。

- [ ] **Step 11: コミット**

```bash
git add real/cockpit.py real/preflight.py
git commit -m "feat: UIハートビートwatchdogを追加(遠隔UI途絶で3秒自動damp)"
```

---
### Task 2: HTTP を `0.0.0.0` で公開（`--host` 引数）

**Files:**
- Modify: `real/cockpit.py:2308-2311`（`--host` 引数）、`:2336-2337`（バインドとログ）

**Interfaces:**
- Consumes: Task 1 で `main()` に追加済みの `--heartbeat-sec`（`--host` はその直後に追加する）。
- Produces: `main()` の `--host` 引数（既定 `0.0.0.0`）、`ThreadingHTTPServer((a.host, a.port), Handler)`。

- [ ] **Step 1: `--host` 引数を追加**

Task 1 で追加した `--heartbeat-sec` 引数の直後に挿入:
```python
    ap.add_argument("--host", default="0.0.0.0",
                    help="HTTP待受アドレス(既定0.0.0.0=遠隔可。localhost縛りは127.0.0.1)")
```

- [ ] **Step 2: バインド先と起動ログを変更**

`real/cockpit.py` 2336-2337 行の:
```python
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"コックピット: http://localhost:{a.port}")
```
を:
```python
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"コックピット: http://{a.host}:{a.port}")
```

- [ ] **Step 3: スモークテスト（`--host` が効く）**

Run: `python real/cockpit.py --sim --host 127.0.0.1`
Expected: ログに `http://127.0.0.1:8090` と出て、`curl -s http://127.0.0.1:8090/state` が JSON を返す。
Run: `python real/cockpit.py --sim`（既定）
Expected: ログに `http://0.0.0.0:8090` と出て、同一ホストの `127.0.0.1` からも `/state` に到達できる。

- [ ] **Step 4: コミット**

```bash
git add real/cockpit.py
git commit -m "feat: HTTPを0.0.0.0で公開し--host引数を追加"
```

---
### Task 3: 機体側の配備資産（requirements_jetpack.txt / setup_onboard.sh）とループバックDDS確認

**Files:**
- Create: `requirements_jetpack.txt`
- Create: `setup_onboard.sh`

**Interfaces:**
- Consumes: なし（Task 1/2 と独立）。
- Produces: `requirements_jetpack.txt`（pip ピン）、`setup_onboard.sh`（機体上で実行して `.venv` を作る）。

- [ ] **Step 1: `requirements_jetpack.txt` を作成**

```text
# G1 機体側(Jetson / Ubuntu / ARM64)で cockpit.py を動かすための依存。
# torch は NVIDIA の JetPack 用 index から別途入れる(setup_onboard.sh 参照)。
numpy
mujoco
unitree_sdk2py
cyclonedds==0.10.2
```

- [ ] **Step 2: `setup_onboard.sh` を作成**

```bash
#!/usr/bin/env bash
set -euo pipefail
# G1 機体側(Jetson / Ubuntu / ARM64)でコックピットを動かすためのセットアップ。
# 使い方: bash setup_onboard.sh
cd "$(dirname "$0")"

PY=python3
"$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# ベース依存(numpy/mujoco/unitree_sdk2py/cyclonedds)
pip install -r requirements_jetpack.txt

# Jetson 用 torch は NVIDIA の index から。JetPack 6 を既定とし、
# 入らなければ JetPack 5 を試す。
TORCH_IDX6="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/"
TORCH_IDX5="https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/"
pip install torch --extra-index-url "$TORCH_IDX6" \
  || pip install torch --extra-index-url "$TORCH_IDX5"

# import 確認
"$PY" - <<'PY'
import numpy, mujoco, torch
import unitree_sdk2py  # noqa: F401
print("numpy", numpy.__version__, "/ torch", torch.__version__,
      "/ mujoco", mujoco.__version__)
PY
echo "完了: source .venv/bin/activate して python3 real/cockpit.py"
```

- [ ] **Step 3: ループバック DDS スモークテスト（機体上で実施）**

機体側（Jetson）へ `real/`・`deploy/`・`model/`・`motions/` をコピーし、`.venv` 有効化後に以下を実行する:
```bash
# LowState がループバック DDS で届くか(送信しない・読むだけ)
# ★--iface "" で ChannelFactoryInitialize(0)=ループバックになる(既定は enp46s0 なので必須)
python3 real/listen_only.py --iface ""
# 実機の素性(FW世代・FSM)が読めるか(送信しない・読むだけ)
python3 real/probe_version.py --iface ""
```
Expected: `listen_only.py` で LowState の数値が流れ、`probe_version.py` で FW 世代が判定できること。
届かない場合は CycloneDDS の domain/transport 設定（`CYCLONEDDS_URI`）を追加し、再確認する（spec §10 リスク参照）。

- [ ] **Step 4: コミット**

```bash
git add requirements_jetpack.txt setup_onboard.sh
git commit -m "chore: 機体側セットアップ資産を追加"
```

---

## 検証サマリ（全タスク完了後に実施）

1. `python real/preflight.py` → NG 0（6b/6c 含む）。
2. `python real/cockpit.py --sim` → `http://0.0.0.0:8090` に到達、`beat` が通る、ブラウザ閉鎖で 3 秒自動 damp。
3. 機体上: `listen_only.py` / `probe_version.py` でループバック DDS 到達を確認。
4. 実機投入は既存の COCKPIT.md §6 チェックリストに従う（本計画のスコープ外）。
