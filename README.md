# G1 着座タスク — 実機コックピット

Unitree G1(29DoF)に強化学習の方策で椅子へ座らせるシステム。方策の実行・
安全監視・実機ログの記録をブラウザから行う。

**現在の本番方策: `sit_up_ln23_r2`  /  実行形態: オンボード演算(機体内で完結)**

---

## 2026-09-04 の追加(このPCで実施。★実機未検証)

| 追加 | どこ | 文書 |
|---|---|---|
| **Android(スマホ)から無線で操作** — 1カラム表示・画面下固定の E-STOP/歩行停止・画面消灯防止・ホーム画面追加 | `real/cockpit.py`(PAGE) | [docs/Android無線操作.md](docs/Android無線操作.md) |
| **十字キーの手動歩行**(押している間だけ。0.5秒のデッドマン二重) | `real/autowalk.py` / `real/cockpit.py` | 同上 §4 |
| **自動歩行: 前進 → LiDARで障害物の手前に停止 → 決めた距離だけ横移動** | `real/autowalk.py`(本体) / `real/real_robot.py`(SetVelocity・LiDAR/odom購読・4→802) / `real/sim_robot.py`(運動学モック+合成LiDAR) | [docs/自動歩行_障害物停止_横移動.md](docs/自動歩行_障害物停止_横移動.md) |
| 方策の完了後の既定を **着座(FSM3)→ダンプ** に変更 | `real/cockpit.py` `after_phase` | Android無線操作.md §5 |
| 机上試験 `python3 real/test_autowalk.py`(障害物検出・デッドマン・通し・途絶中止) | `real/test_autowalk.py` | 自動歩行 §8 |
| **着座の「膝が出る」「回転する」の解析** — 実機11本で確認、原因は方策(早落ち・浅い着座・ヨーの癖)、走行ごとの数値化(`real/sit_shape.py`)と脚残差の試験オプション | `real/sit_shape.py` / `real/cockpit.py` | [docs/着座_膝と回転の解析_20260904.md](docs/着座_膝と回転の解析_20260904.md) |
| **通信途絶で damp しない安全設計** — 方策の実行中は最後まで、方策なしのPD保持は内蔵バランスへ返して静止、歩行は速度ゼロ | `real/cockpit.py` / `real/real_robot.py` `return_to_balance` | [docs/通信途絶時の安全設計_20260904.md](docs/通信途絶時の安全設計_20260904.md) |
| **かんたん画面**(既定 `/`。詳細は `/detail`) — ダンプ/スタンドロック/歩行モード、前進(壁の手前で段階減速・障害物は回り込み)、横歩き(足踏み)・5cm・後ろへ5/10cm、LiDAR上面図、着座の事前確認(サーバ側点検+操作者確認+3秒)、E-STOP | `real/cockpit.py` PAGE_SIMPLE | Android無線操作.md §2 |
| **LiDAR ブリッジ** — この FW では rt/utlidar/* が無いので、機体内の Livox SDK2 を ctypes で直接叩いて /dev/shm へ点群を書く(上下逆さ取付を自動補正) | `real/lidar_bridge.py` / `real/start_onboard.sh` | 自動歩行 §6b |
| **深く座る** — 座面接触を検知したら膝・足首の残差を2秒で抜いて参照の深い着座姿勢へ(シム 後退 23→32cm)。既定 ON | `real/cockpit.py` `_contact_update` | 着座解析 §3-0 |
| **方策の整理** — deploy/ は `sit_up_ln23_r2`(本番)・`sit_up_ln21_r2`(A/B対照)・`climb_slow_r2`・`turn_wide_r2` の4つだけに。他19本は `deploy_archive_20260904/` へ退避 | `deploy/` | deploy/README.md |
| 機体への同期+起動 `./deploy_to_robot.sh [wifi] [sync]` | `deploy_to_robot.sh` | |

| 自動起動 | `real/systemd/g1-lidar-bridge.service`・`g1-cockpit.service`、`real/install_autostart.sh` | 機体の電源投入で LiDAR ブリッジ→コックピットが自動で立ち上がる(Jetson 再起動で確認)。`start_onboard.sh` はサービス再起動に化ける。docs/オンボード運用.md §10 |
| 着座の修正(9/4 午後) | `real/cockpit.py` | 接触後フェードは実機で 3/3 崩れた(誤検知で残差が消えロール33°・ヨー90°)→ **既定 OFF**、検知にゲート。完了時に座面に載っている証拠が無ければ自動ダンプを保留(赤帯)。docs/着座_膝と回転の解析_20260904.md §3-0b |
| 小刻みステップ(9/4 午後) | `real/autowalk.py`(`_step_axis`)、`real/sim_robot.py`、かんたん画面 | 後退ナッジが実機で進まなかった(弱いパルスでは内蔵歩行が一歩も出ない)ため、横歩き・5cm・後退・[1歩] を「確実に一歩出る短い指令→止めてオドメトリで測る」方式に統一。1歩の推定を実測で更新、3歩で進まなければ中止。docs/自動歩行_障害物停止_横移動.md §6b-3。実機未検証 |
| 12:36 の転倒の原因と対策 | `real/systemd/*.service`、`real/start_onboard.sh`、`real/cockpit.py` | 再起動後のサービスで制御周期が 23 Hz に落ちガードでダンプ→立位のまま前へ倒れた。jetson_clocks 固定・DDS 環境変数・Nice・走行中の再起動拒否。完了後の既定を元の FSM3 へ戻す。docs/着座_膝と回転の解析_20260904.md §3-0c |
| LiDAR の壁距離の作り直し(9/4 午後) | `real/lidar_bridge.py`、`real/autowalk.py`、かんたん画面 | 前方を生座標 +X 固定に(床が見える向きは後ろを向いていた)、つま先基準、壁の面を直線フィット、前後左右の最近距離、[前後を反転]、前進時の自動ヨー較正。docs/自動歩行 §6b-4 |
| 歩行が始まらない原因と修正(9/4 午後) | `real/real_robot.py`(`ensure_walk_mode`)、`sim_robot.py`、UI 表記 | 802 では SetVelocity が効かない(0.35 m/s×4 秒で 4 cm)。最新 SDK の Start()=SetFsmId(500) に合わせ、歩行は loco 500/501 で行い、802 は着座の UserCtrl 入口だけに。docs/自動歩行 §6b-5。実機未検証 |
| 速度プロファイルと正対(9/4 午後) | `real/autowalk.py`(`speed_profile`, `_move_axis`, `_align_to_wall`)、かんたん画面 | 段階式→距離に応じた連続減速(√(2ad)、応答遅れ込み)、巡航 0.5 m/s と「速さ」選択。横歩き・後退は普通の歩行(10 cm 以下だけ小刻み)。斜めの壁には正対してから前進。docs/自動歩行 §6b-6 |
| 斜めの回り込み・壁追従・最新 SDK の歩き方(9/4 夕) | `real/autowalk.py`(`_forward`, `_track_wall_heading`)、`real_robot.py`(`set_speed_mode`) | 歩きながら斜めに寄せてギリギリ(余白 8 cm)で回り込み、歩きながら戻る。壁の角度を毎コマ測って垂直を保つ。SetSpeedMode / ContinuousGait / duration 2 s。docs/自動歩行 §6b-7 |
実機での検証順は 自動歩行 §6(A 距離表示 → B ドライラン → C 十字キー → D 短距離 → E 既定値)。

## 引き継ぐ人がまず読むもの(この順で)

| # | 文書 | 内容 |
|---|---|---|
| 0 | [docs/Android無線操作.md](docs/Android無線操作.md) | スマホからの無線操作(2026-09-04) |
| 1 | **[docs/オンボード運用.md](docs/オンボード運用.md)** | **起動方法と、実機で嵌まった問題の全記録。まずこれ** |
| 2 | [実機セッション_20260827.md](実機セッション_20260827.md) | 現場手順。特に §0(落下原因)と §7(やってはいけないこと) |
| 3 | **[docs/RTX.md](docs/RTX.md)** | 学習側の全記録。方策の系譜・実測・未解決課題 |
| 4 | [実機開発で学んだこと_20260827.md](実機開発で学んだこと_20260827.md) | 実機一般の教訓 |

---

## 起動

### オンボード(既定)

制御・推論・DDSをすべて機体内で完結させる。ネットワークを渡るのはUIだけ。

```bash
ssh unitree@192.168.179.100 '~/cl-workspace/start_onboard.sh'
```

→ ブラウザで http://192.168.179.100:8090

競合プロセスの停止(§トルク競合)から待受確認までを1本でやる。
**手で `cockpit.py` を直接叩かないこと。** 機体を再起動すると消えるので叩き直す。

### PCから有線で(従来)

```bash
python3 real/cockpit.py --sim              # モックで手順確認(実機不要)
python3 real/cockpit.py --iface enp46s0    # 実機(有線LAN)
```

★**無線でPC制御はできない。** 機体のDDSは eth0 にしかバインドしていない
(詳細は [docs/オンボード運用.md](docs/オンボード運用.md) §1)。

### 実機投入前

```bash
python3 real/preflight.py    # 55項目の自己点検。NG=0 を確認してから実機へ
```

---

## 構成

```
real/       cockpit.py(本体・UI)/ real_robot.py(DDS)/ run_fsm.py(方策と観測構築)
            preflight.py(実機投入前の点検)/ start_onboard.sh(機体側の起動)
            log_view.py・ab_report.py(走行ログの解析)
deploy/     方策 = policy.pt + reference.npz + meta.json
model/      MuJoCoモデル。★シミュレーションには使わず FK計算機として使う
motions/    climb_stand.npz(登りの開始立位)
logs/real/  走行ログ。セッションごとに run<NN>_*.npz / _設定.json / イベント.log
docs/       設計と運用の記録
```

### 方策の既定は自動で最新に追従する

UIの既定方策は、`cockpit.py` の `PATTERN_NOTES` で **`★★最新` の印**が付いた
ものが選ばれる(`default_pattern()`)。新しい方策が来たら**印を移すだけ**でよく、
既定の更新漏れで古い方策のまま実機を回す事故を防いでいる。

---

## 安全上、最初に知っておくこと

1. **本当の安全装置はリモコンのE-STOP**(ハードウェア・ネットワーク非依存)。
   UIのdampボタンは通信が切れれば押せない。ネットワークは安全経路ではない。
2. **`3f_demo.service` を止めてから制御すること。** 機体には `rt/lowcmd` を出せる
   デモサーバが常駐・自動起動有効で入っており、放置すると同じ関節を2つの制御系が
   奪い合ってトルク異常になる。`start_onboard.sh` が自動で止める。
   **機体を再起動すると復活する。**
3. **方策の実走中はLANケーブルを抜く。** 動作中に引っ張られる/引っ掛かるのが
   物理的に危険。オンボードなので制御には影響しない。
4. UserCtrl への進入は **FSM 802(走行)を経由する**。走行制御が動くので、
   押す前にE-STOPを握ること。

---

## 走行ログについて

`logs/real/` は**版管理に含めている**(実験記録もコードと同じ履歴に置く方針)。
1本約500KBのnpzが走行のたびに増えるので、**セッション単位・日単位でまとめて
コミットする**こと。重くなってきたら `.gitignore` へ移して zip 配布に切り替える。

解析は `log_view.py`(1走行の可視化)と `ab_report.py`(A/B比較)。
RTXへ渡すバンドルは README + 走行ログ + 解析スクリプト + 参照 の構成で zip にする。

---

## いま残っている課題

[docs/RTX.md](docs/RTX.md) §8 が正本。実機側の当面の焦点は:

- **`sit_up_ln23_r2` の実機A/B**(ln21と交互に各5本以上)。見るのは
  **開始0.5秒の膝突き出し**と**左ロール**。シムでは膝は改善・左ロールは悪化
- **左ロールの切り分け**。参照の対称化(ln21)でロールは半減したが方策のaction
  非対称はむしろ増えた = **実機側の非対称が主因**。関節ゼロ点・IMU取り付け・
  左右の摩擦差の較正が要る。椅子の左右反転 / 機体180度回転のテストが未実施
- Wi-Fiの安定性。電波が弱いとUIが重くなる(制御には影響しない)
