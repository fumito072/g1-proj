# 機体側コンピューターでの方策演算 設計（アプローチA）

- 日付: 2026-08-28
- 対象: `g1-proj/`（G1 実機コックピット一式）
- 分類: アーキテクチャ変更（配備トポロジの変更。ただし第1段は漸進的）

## 1. 目的と背景

現状、方策推論・50Hz制御ループ・500Hz送信・安全監視はすべて**操縦PC（Linux）**上で
実行し、有線LAN（192.168.123.222 → 機体 .161）経由の DDS で機体のモーションコントローラと
通信している。

主目的は **遅延・確実性の向上**。制御ループと方策推論を**機体側コンピューター（Jetson / ARM）**で
回すことで、DDS を LAN ホップからループバックへ置き換え、遅延を減らし確実性を上げる。
PC は**監視・UI として残す**。

第1段（本設計）は「まず推論＋制御を機体側へ」を最小変更で実現する。

## 2. 現状アーキテクチャ

`real/cockpit.py` は1プロセスに2役割を同居させている:

| 部分 | 役割 | 移動先 |
|---|---|---|
| `Engine`（cockpit.py:296） | FSM・50Hz制御ループ・安全監視・引き継ぎ・方策実行 | 機体側 |
| `RealRobot`（real_robot.py） | DDSリンク（500Hz送信 / LowState購読）・E-STOP・目標ガード | 機体側 |
| `Policy` / `ObsBuilder`（run_fsm.py） | 方策推論（PyTorch `.pt`）・観測構築（MuJoCo FK） | 機体側 |
| `Handler` / `PAGE`（cockpit.py:2252〜） | HTTPサーバ・ブラウザUI（`GET /state` でtelemetry、`POST ?c=…` でコマンド） | PC側（遠隔ブラウザ） |

既に確認済みの好材料:
- `Engine` と `Handler` の境界は綺麗で、UI は単純な HTTP ポーリング（WebSocket 不要）。
- `ChannelFactoryInitialize(0)`（iface 無し = ループバック）の分岐が既存（real_robot.py:146-149）。
- `deploy/*/meta.json` は `onnx: true / verify_max_err 1.19e-7`（= ONNX は検証済み。今回は不使用）。

## 3. 目標アーキテクチャ（アプローチA）

- **機体側（Jetson）**: 既存の `cockpit.py`＋`real_robot.py`＋`run_fsm.py`＋`deploy/`＋
  `model/`＋`motions/` を配備・起動。DDS はループバック（`--iface` なし）、
  HTTP は `0.0.0.0:8090` で待受。
- **PC側**: ブラウザで `http://<機体IP>:8090` を開く。UI・操作は従来と同一。

## 4. データフロー（ロジック不変・配置のみ変更）

```
モーションコントローラ ⇄(ループバックDDS)⇄ RealRobot ⇄ Engine(50Hz + 方策推論) ⇄ Handler(HTTP)
                                                                                    ↑
                                                          PCブラウザ(遠隔) ──────────┘
```

既存の制御ループ・安全機構・引き継ぎ・ガードはそのまま機体側へ移る。アルゴリズム変更は無い。

## 5. コード変更点

1. **`real/cockpit.py` `main()`**
   - `ThreadingHTTPServer(("127.0.0.1", a.port), Handler)` → `("0.0.0.0", a.port)`。
   - `--host` 引数を追加（既定 `0.0.0.0`。localhost 縛りが必要な場合は `--host 127.0.0.1`）。
   - 起動ログに「接続先: http://<機体IP>:8090」を出す。
2. **DDS（変更不要・確認のみ）**
   - `--iface ""` で `ChannelFactoryInitialize(0)` になる既存分岐を使う。
   - 機体上でループバック DDS がモーションコントローラに到達するかを**スモークテストで実測**する。
     - 到達しない場合の対応は §8 のリスクを参照（CycloneDDS の domain/transport 設定を追加）。
3. **遠隔UIハートビート（新規。§6）**

変更しないもの:
- `RealRobot` / `Policy` / `ObsBuilder` のロジック。推論ランタイムは**今回 PyTorch `.pt` のまま**
  （遅延の主因は LAN/DDS ホップであり、推論自体は中央 1.1ms。ONNX 化は次段で独立検証）。

## 6. 安全設計（★最重要）

既存7層（estop_now／100Hz watchdog／目標ガード／傾き40°／関節速度／LowState断0.2s／送信断50ms）は
そのまま機体側へ移動する。

**追加: 遠隔UIハートビート**
- 目的: UI が機体と離れることで「PC断線・ブラウザ閉鎖・PC死亡」時に誰も E-STOP を押せなくなる
  故障モードを防ぐ。
- 仕様:
  - ブラウザ JS（`PAGE`）に `setInterval` で毎秒 `POST /?c=beat` を追加。
  - `do_POST` で `c == "beat"` を受理し、`Engine.beat()` を呼ぶ（`estop`/`damp` と同様、
    **コマンドキューを経由せず HTTP スレッドから直接** `self._last_beat = time.time()` を更新）。
  - 機体側の watchdog（既存 100Hz `_watchdog`）で、`_monitoring()` が True（= custom 制御中 or
    MOVING/RUNNING/HOLD/WAIT_CONFIRM）かつ**初回 beat 受信済み**かつ
    `time.time() - _last_beat > ハートビート秒（既定 3.0）` なら `estop_now("UIハートビート途絶")`。
  - `--heartbeat-sec` 引数（既定 `3.0`）。`0` で無効（localhost 検証用）。
  - 「初回 beat 受信済み」ガードにより、ページ未オープンでの誤 trip を防ぐ。ページを開いた瞬間から
    監視が効き、閉じると 3 秒で自動 damp する。
- 物理 E-stop ＋ ガントリーは従来どおり最上位の安全層。ソフトの damp はその代替にならない。

## 7. 配備手順

1. `real/`＋`deploy/`＋`model/`＋`motions/` を1フォルダにまとめ、`scp`/`rsync` で機体へ。
2. 機体側に以下を追加:
   - `requirements_jetpack.txt`: Jetson 用 torch（NVIDIA index 指定）・numpy・mujoco・
     `unitree_sdk2py`（同梱SDK）・cyclonedds。
   - `setup_onboard.sh`: venv 作成 → 依存導入 → import 確認（既存 `preflight.py` の環境部を流用）。
3. 起動: `python3 real/cockpit.py`（`--iface` なし）。
4. PC ブラウザで `http://<機体IP>:8090`。

## 8. 検証計画

1. **機体上 `--sim`**: ループバック DDS 以外を全手順リハーサル（既存 §6 の WP8 準拠）。
2. **遠隔 UI 確認**: PC から `http://<機体IP>:8090` 到達、E-STOP、ハートビート断（ブラウザ閉鎖
   or ケーブル抜き）での 3 秒自動 damp を確認。
3. **ループバック DDS スモークテスト**: 機体上で `probe_version.py` / `listen_only.py`（iface 無し）を
   走らせ、LowState が届く・FSM が読めることを実測。
4. **実機**: 既存 §6 チェックリスト（probe_version → preflight NG 0 → 吊りテスト → 接地確認 → 単体 → 通し）。

## 9. スコープ外（次段以降のフォローアップ）

- ONNX Runtime への推論切替（`Policy.act` の差し替え。meta.json の `verify_max_err` で検証可能）。
- アプローチB（機体=ヘッドレスランナー＋PC=ローカルUI分離）。ハートビートが既に入るため移行は容易。
- `twin.py`（実機ミラー）の遠隔化。読み取り専用の DDS 購読なので、PC 側から LAN 購読を維持できる
  可能性が高いが、ループバック化後の domain 到達性は別途確認。
- AprilTag による自己位置（脚オドメトリのドリフト対策）は既存課題のまま。

## 10. リスク・未確定事項

| リスク | 影響 | 対応 |
|---|---|---|
| ループバック DDS がモーションコントローラに届かない | 起動不可 | スモークテストで早期検出。CycloneDDS の domain/transport 設定で補正 |
| 機体側の Jetson に依存（torch 等）が入らない／CPU が遅い | 起動不可／遅延 | `requirements_jetpack.txt` の NVIDIA index。推論は中央1.1msなので性能余裕あり |
| 遠隔 E-STOP がネットワーク遅延を持つ | 停止遅延 | ハートビート自動 damp が担保。物理 E-stop 併用 |
| `--sim` と実機でホスト差異（ARM/x86、パス、SDK ビルド） | 環境不整合 | 配備は自己完結フォルダ。preflight で環境・deploy 整合を確認 |
