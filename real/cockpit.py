#!/usr/bin/env python3
"""G1コックピット — deployの方策を実機で走らせる操縦システム。

  python3 real/cockpit.py --sim            # MuJoCoモックで結合試験(実機不要)
  python3 real/cockpit.py --iface enp46s0  # 実機(有線LAN)
ブラウザで http://localhost:8090 を開く。使い方は real/COCKPIT.md。

構成:
  ロボットIF: RealRobot(DDS) / SimRobot(MuJoCoモック、--sim)
  FSMエンジン: 50Hzループ。フェーズ=deployの方策。遷移は操作者確認(既定)
  Webサーバ: 状態JSON・映像(simのみ)・コマンド受付
安全:
  E-STOP(UIの赤ボタン)→即damp / 傾き40度・関節速度超過・受信途絶→自動damp
"""
import argparse
import gc
import json
import pathlib
import platform
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from run_fsm import (Policy, ObsBuilder, quat_to_mat, ACTION_SCALE,   # noqa: E402
                     CONTROL_HZ, TILT_LIMIT_DEG, _yaw_of)
from autowalk import WalkController                                    # noqa: E402
from sit_shape import SagittalFK, shape_metrics, describe as shape_describe  # noqa: E402
import base64                                                          # noqa: E402

DEPLOY = ROOT / "deploy"
VEL_HARD = 32.0            # 全関節の速度ハード上限[rad/s]
# 引き継ぎ直後の「静的な前傾」を開始前に弾くための境目。
# 2026-08-21の実バンク学習で、傾き16.1度と36.9度・角速度|gyro|≒0.04rad/s の
# 2状態だけが 0/10 だった。成功した7状態は傾き12〜20度に戻り方向の角速度〜1rad/s。
# 傾きだけでは決まらない(11.9度で失敗・18.5度で成功の例がある)ので、
# 「大きく傾いているのに動いていない」の組で判定する。
HANDOVER_STATIC_TILT_DEG = 15.0
HANDOVER_STATIC_RATE = 0.25       # rad/s

# --- フェーズ開始直後の追従誤差(★記録のみ。止めない) --------------------
# 2026-08-26 11:09 の実機転倒のあと、これで自動中止できないかを実ログ27本で
# 検証した。**できなかった。**しきい値は完走回の最大0.949radより上、転倒回の
# 1.112radより下でなければならず、失敗の実データが1本しかない状態でその
# 15%の窓に置くのは、完走回を止める側に倒れる(実際 0.80radで検証すると
# 完走7本を誤って止めた。今日完走した deep も 0.949 で引っかかる)。
#
# **数字が足りないうちは止めない。記録して貯める。**
# 走行ごとに run<NN>_設定.json と イベント.log に残るので、失敗回が数本
# 貯まった時点でしきい値を決め直すこと。
#
# 参考(2026-08-26時点の実測。フェーズ開始0.6秒以内の脚腰|q-target|最大):
#     完走 20本: 中央 0.53  最大 0.949
#     転倒  1本: 1.112
# ★2026-08-26 11:09 の転倒について分かっていること:
#
#   - 傾き3.8度・角速度0.01rad/s・脚腰の開始ずれ0.378rad。**どれも8/24の
#     完走回とほぼ同じ値で、開始前の情報では区別できなかった**
#   - 開始姿勢は「しゃがみ」(膝0.855rad。参照は0.549)。[custom]で41秒
#     保持した姿勢をそのまま掴んでいた
#   - 3分前に**同じしゃがみ姿勢から sit_up_deep_r2 は完走している**
#   - 転倒回は足首ピッチが act=-0.99 で飽和し、可動域(-0.87)の外の
#     -0.983 を指令してストッパへ突っ込んでいた。deep は同じ姿勢から
#     もっと穏やかな指令を出していた
#   → 開始前の状態量では区別できない。**手順(純正standから入る)で防ぐ。**
#     8/24にstandから入った7本は全部完走している
START_TRACK_WINDOW_S = 0.6        # 記録する窓[秒]

# --- 制御ループの周期ガード(★これは較正済み。止める) --------------------
# 方策は50Hz(20ms)で学習してある。ループが遅れると、参照は1コマずつ進むのに
# 実時間は遅れる = **動作がスローモーションになる**。重力は遅くならないので
# 釣り合いが崩れる。19Hzで走ると2.6倍のスローモーションになる。
#
# 2026-08-26 の実測(dt_ms列を持つ全走行):
#     完走 : 20.0 / 20.9 / 19.8 ms(中央)  5コマ移動中央の最大 32.4 / 33.0 / 26.2
#     転倒 : 25.6 / 51.9 ms(中央)         5コマ移動中央の最大 46.1 / 76.0
#   → 完走側の最大33.0 と 転倒側の最小46.1 の間で分離できる。40msに置く。
#
# ★これは姿勢から意図を推し量るガードではなく、「制御周期が仕様外」という
#   **定義上おかしい状態**を検出するもの。しきい値は分離点を実測で決めただけ。
# --- 接地チェックの基準値 -------------------------------------------------
# ★足裏に力センサは無い(G1のLowStateに foot_force は含まれない)。
#   代わりに「足首に微小な指令を入れて、どれだけ素直に動くか」で判定する。
#   - 接地していれば、足首を回すと体を傾けることになるので**ほとんど動かない**
#     (指令0.05rad×kp40 = 2N·m。重力トルク20N·m台に対して小さい)
#   - 浮いていれば、足(約0.5kg)だけが軽く回るので**指令どおり動く**
GC_DELTA = 0.05          # 足首ピッチに入れる微小指令[rad] (2.9度)
GC_DUR = 0.25            # 保持時間[秒]
GC_RATIO_GROUNDED = 0.30 # 追従率がこれ未満なら接地とみなす
GC_RATIO_FLOATING = 0.60 # これを超えたら浮いている疑い

# --- 方策出力の立ち上げ(ランプ) -----------------------------------------
# ★2026-08-26 の実機で、開始姿勢を参照へ合わせても(脚腰の残差0.10〜0.20rad)
#   **方策1コマ目の跳びが 0.46〜0.72rad 残った**。
#   target = ref_q[t] + action×0.7 で、方策が1コマ目からいきなり
#   action≈0.65〜1.0 を出すため。左足首ピッチで18〜29N·mの段差になる。
#   実測: 完走した2本は跳び0.459/0.556rad、転倒した3本は0.644/0.718/0.404rad。
#
#   そこで**最初の数コマだけ action を 0 から立ち上げる**。
#     target = ref_q[t] + action × ACTION_SCALE × w(t)
#     w(t) = min(1, (t+1)/ランプコマ数)   ← t=0 では w=0 なので跳びが消える
#
#   ランプ中は方策の補正が弱まるが、開始直後は機体が参照開始姿勢で静止して
#   いるので、参照軌道そのものが妥当な指令になる(参照は最初の1秒で最大
#   1.4rad/s しか動かない)。
#   ★observation の last_cmd には**方策の生出力 a** を入れる(ランプ後の値では
#     ない)。学習時に方策が見ていたのは自分の出力そのものなので。
#   シムで振った実測(2026-08-26):
#     ランプ    deep跳び/最大傾き   slow跳び/最大傾き   rc跳び/最大傾き
#     0.00s     0.568 / 28.3°       0.264 / 19.7°       0.283 / 32.3°
#     0.20s     0.057 / 28.1°       0.026 / 19.7°       0.028 / 31.7°
#     0.30s     0.038 / 28.0°       0.018 / 19.8°       0.019 / 31.1°
#     0.50s     0.023 / 28.8°       0.011 / 19.9°       0.011 / 28.3°
#     0.80s     0.014 / 28.8°       0.007 / 21.1°       0.007 / ★114.7°(転倒)
#   → **跳びは1/15になり、最大傾きは悪化しない。**ただし長すぎると方策の
#     補正が効かなくなって転ぶ(rcが0.8秒で転倒)。0.2〜0.5秒が安全域。
#   ★シムでは最大傾きが改善しない点に注意。ランプが消すのは「開始時の
#     衝撃」であって、傾きの行き過ぎではない。実機では接地やハーネスの
#     応答があるので、シムが表さない効果が出る可能性がある(要実測)。
ACTION_RAMP_S = 0.30              # 立ち上げ時間[秒]。0で無効(従来どおり)

# --- 参照のブレンド(ソフトスタート) ---------------------------------------
# ★2026-08-26 の実機で分かったこと:
#   **PD保持ではどんな姿勢も保持できない。**simで実測すると、参照開始姿勢
#   (膝31度)は1.5秒で98度まで倒れ、ほぼ直立の姿勢でも1秒で29.6度へ流れる。
#   操作者が毎回手で支えていたのはこのため。
#   つまり「開始姿勢へ寄せて → 保持 → 方策開始」という段取りは、
#   **保持している間ずっと支えが要る**設計になっていた。
#
#   そこで、姿勢を保持せずに済ませる:
#     ref_eff[t] = (1-u)·q0 + u·ref_q[t]      u: 0→1 を REF_BLEND_S 秒で
#     target     = ref_eff[t] + action×0.7×w  w: ランプ
#   t=0 では ref_eff = q0 = いまの姿勢なので、**どこから始めても跳びが出ない**。
#   方策は1コマ目から動いているのでバランスも効いている。
#
#   simで12通り×3方策を実測(開始姿勢のずれ 0.00〜0.53rad):
#     跳び      ブレンド無し 0.45〜0.53rad → 0.5秒ブレンド **0.02〜0.04rad**
#     最大傾き  改善しない(ばらつく)。deep+直立22.7度 / slow+直立35.7度
#     1.0秒だと rc+完全直立 で転倒(116度) → **0.5秒が安全域**
#   ★効果は「跳びを消す」ことと「PD保持を経由せずに済む」ことであって、
#     傾きの行き過ぎが直るわけではない。
REF_BLEND_S = 0.50                # 参照を現在姿勢から滑り込ませる秒数。0で無効

LOOP_DT_WINDOW = 5                # 移動中央値を取るコマ数
LOOP_DT_MAX_MS = 40.0             # これを超えたら中止(20msの2倍)

# ログ形式は 2026-08-24 の実機セッションのものに合わせる。
# real/log_view.py と real/ab_report.py がこの形式を前提にしており
# (`rec` 行列 + `cols` 列名 + `run<NN>_設定.json`)、別形式で書くと
# 現場で取ったデータを既存の解析ツールで読めなくなる。
FSM_CODE = {"IDLE": 0, "MOVING": 1, "WAIT_CONFIRM": 2, "RUNNING": 3,
            "HOLD": 4, "DAMP": 5}
REC_COLS = (["t", "fsm", "phase_i", "elapsed_s", "tilt_deg"]
            + [f"q{i}" for i in range(29)]
            + [f"dq{i}" for i in range(29)]
            + [f"tau{i}" for i in range(29)]
            + ["quat_w", "quat_x", "quat_y", "quat_z"]
            + ["gyro_x", "gyro_y", "gyro_z"]
            + [f"target{i}" for i in range(29)]
            + [f"act{i}" for i in range(29)]
            + [f"temp{i}" for i in range(29)]
            # ここから先は2026-08-26に追加した列。既存ツールは名前で引くので
            # 末尾に足す分には影響しない。dt_ms/ms_inferはPythonの尾側の
            # 遅れが実機で出ているかを後から数値で確かめるため
            + ["dt_ms", "ms_infer"]
            # ここから先は2026-08-27に追加。**すべて末尾に足す**ので
            # log_view.py / ab_report.py は今までどおり読める。
            # 再学習(摩擦・動力学の同定)のために、LowStateに載っているのに
            # 記録していなかった量を全部入れる。
            + [f"ddq{i}" for i in range(29)]        # 関節加速度
            + [f"vol{i}" for i in range(29)]        # モータ電圧
            + [f"temp2_{i}" for i in range(29)]     # 温度センサ2(巻線側)
            + [f"mstate{i}" for i in range(29)]     # モータのエラーフラグ
            + ["acc_x", "acc_y", "acc_z"]           # IMU加速度計
            + ["rpy_r", "rpy_p", "rpy_y"]           # IMUの姿勢角
            + ["imu_temp", "tick", "mode_pr", "mode_machine"]
            + [f"mmode{i}" for i in range(29)]      # モータの有効/無効
            + [f"remote{i}" for i in range(40)]     # リモコンのボタン状態
            + ["version0", "version1", "crc",
               "ls_res0", "ls_res1", "ls_res2", "ls_res3"]
            + ["ramp_w", "blend_u"]                 # その時の立ち上げ係数
            + ["wall_t"])                           # 壁時計(動画との突合せ用)
# ★ここに入れない量(意味が未文書 or 量が多い)は、npzの別配列へ入れる:
#     raw_sensor  (T,29,2)  motor_state[i].sensor
#     raw_reserve (T,29,4)  motor_state[i].reserve
#     motor_ext   (T,6,5)   29〜34番のモータ(29dofでは未使用)[mode,q,dq,ddq,tau]
#   recに混ぜると列が倍近くになって log_view/ab_report が読みづらくなるため。
# 立位の姿勢。**[立つ]を押したあと、方策を始める前にここを見る。**
# 2026-08-27の実機32本(シム除外)から、着座に成功した13本
# (終端傾き15〜24度 = シムの18.5に届いた回)の10〜90パーセンタイル。
#
# ★正直に書いておく: 個々の関節と結果の相関は弱い(|r|最大0.32、
#   同一セッション10本の中でも符号が入れ替わる)。**これは合否判定ではない。**
#   「いまの立ち方が、うまくいった回の立ち方と同じかどうか」を現場で
#   目視できるようにするためのもの。操作者の
#   「補助なしで自立できたときは高確率で座れる」という観察を数値で追うための窓。
# ★しゃがみ深さ = 膝角 - 足首角[度](左右平均)。**立つモードで測る。**
# 2026-08-27 実機30本(シム除外・着座成功=終端傾き15〜24度):
#     65度未満  ◎14 / △ 9 = 61%成功
#     65度以上  ◎ 4 / △17 = 19%成功
#     Fisher正確検定 両側 p = 0.0065  (実機44本)
# ★30本時点では 77%対18%(p=0.0024) だったが、44本に増えると上記まで落ちた。
#   しきい値もデータから選んでいるのでp値は楽観側。方向は残っているが
#   **単独で合否を決められる指標ではない。**
# 個々の関節の相関(|r|<0.32)より遥かに強い。線形ではなく**しきい値**として効く。
# 操作者の「補助なしで自立できたときは高確率で座れる」という観察と一致する:
# 支えると機体は沈み、深くしゃがんだ立ち方になる。
# ★NGにはしていない。深くても3本は座れているので、押させないのは行き過ぎ。
CROUCH_TH = 65.0
POSE_KEYS = [
    ("足首ピッチ", "left_ankle_pitch", "right_ankle_pitch", 4, 10),
    ("足首ロール", "left_ankle_roll", "right_ankle_roll", 5, 11),
    ("膝", "left_knee", "right_knee", 3, 9),
    ("股ピッチ", "left_hip_pitch", "right_hip_pitch", 0, 6),
    ("股ロール", "left_hip_roll", "right_hip_roll", 1, 7),
]
POSE_WAIST = [("腰ピッチ", "waist_pitch", 14), ("腰ロール", "waist_roll", 13),
              ("腰ヨー", "waist_yaw", 12)]
GOOD_RANGE = {
    "left_ankle_pitch": (-0.461, -0.359), "right_ankle_pitch": (-0.508, -0.415),
    "left_ankle_roll": (0.019, 0.061), "right_ankle_roll": (-0.058, -0.032),
    "left_knee": (0.551, 0.776), "right_knee": (0.518, 0.751),
    "left_hip_pitch": (-0.246, -0.003), "right_hip_pitch": (-0.231, 0.023),
    "left_hip_roll": (-0.063, -0.004), "right_hip_roll": (-0.037, -0.002),
    "waist_pitch": (-0.065, -0.013), "waist_roll": (0.003, 0.057),
    "waist_yaw": (-0.002, -0.001),
}
ASSIST_LABEL = {"none": "補助なし(自立できていた)",
                "light": "軽く触れていた",
                "hold": "しっかり支えた",
                "?": "★未記入 — 記録してください"}
# UIから受け付けるコマンド。**PAGE の cmd('X') と1対1で対応させること。**
# estop/damp/select だけは do_POST で特別扱いしている(即時実行・引数の分解)。
# preflight 6c がこの対応を照合する。
CMD_ALLOW = (
    "armres", "armmode",
    "arm", "start", "next", "place_sim", "mode", "goto_start",
    "ground_check", "ramp", "blend", "memo", "newsession",
    "stand_user", "fsm_read", "user_run",
    "mode_zero", "mode_damp", "mode_stand", "mode_walk", "mode_sit",
    "mode_seated",
    "custom", "run_task",
    "assist", "yawalign", "stopframe", "afterphase",         # ← 2026-08-27に入れ忘れていた
    # 2026-09-04 内蔵歩行: 自動歩行 / スマホ手動操作。tele と walk_stop は
    # do_POST で**キューを経由せず**直接処理する — 手動操作の遅延と停止の即時性のため。
    # ★このタプルの中のコメントに丸括弧を書かないこと。preflight 6c の正規表現が
    #   最初の閉じ括弧で止まり、それ以降の名前を見落とす
    "walk_ready", "walk_auto", "walk_param", "walk_stop", "tele", "lidar",
    "legres",                                                # 脚腰の残差スケール。試験用
    "walk_go", "sit_check", "sit_go", "hbdrop",              # かんたん画面と途絶の模擬
    "cfade",                                                 # 接触後の膝・足首の残差抜き
    "lidar_flip",                                            # LiDAR の前後を反転。lidar_mount.json に書く
)
CONTACT_FADE_MIN_T = 85           # 接触後フェードの検知を許す最初のコマ。参照の座面到達1.9秒の少し前
SEAT_KT_MAX = 18.0                # 完了時: 終端0.5秒の両膝トルク中央値がこれ以上なら「脚に体重が残っている」
SEAT_BACK_MIN_CM = 3.0            # 完了時: 骨盤の後退がこれ未満なら「座面に載っていない疑い」
REC_SAVE_EVERY = 100              # 途中で落ちても失わないよう逐次保存


# UIのプルダウンに出す短い注記。現場で「どれが本命でどれがアンカーか」を
# 名前だけで思い出せないと、比較の設計ごと崩れる(実機セッション手順 §5)。
# ★ここに無い方策は注記なしで出る。使用可否の判断は手順書が唯一の真実。
PATTERN_NOTES = {
    # --- 着座(これまで使った全部を出す。使用可否は operator が決める)
    # 2026-08-28着。dp4 の**脚腰はビット単位で同一**で、腕にだけ軌道が入った
    # (dp4は腕14関節が全395コマ0.000rad固定。ar1は振れ幅合計10.3rad)。
    # 開始の肘も1.15radで実機の実測1.21radに近い。dp4は0.000で1.2radずれていた。
    # 静止区間も53%→0%。シム実測: 座面荷重 228N→333N(機体35kg=350N)、
    # 左股action +0.307→+0.212。ただし終端傾きは18.5→23.7度と参照(18.0)から離れた
    # 2026-09-02 夕着。ln20の重みを親に、**参照の左右非対称を厳密にゼロ**へ
    # 直して微調整した版(9/2の依頼1「参照のIKに左右対称拘束を」への回答)。
    # 対称面を足の中点から実測(x=+0.034m)し、右脚=左脚の鏡像で強制。
    # 検証: 股ピッチ/膝/足首Pの左右差、股ロール/ヨー/足首Rの左右和、腕の左右和、
    #       腰ロール/ヨー — **全て 0.000000 rad**。前傾相(+20度)は保存。
    # 配布元: 教師97%(プロジェクト最高)・蒸留の劣化ゼロ・ゲート6項目全合格。
    #       シムのロール 平均-3.60度→-0.42度、左右の偏り 左28/右2 → 左18/右12。
    # ★ただしシムのロールは実機の1/3しか再現しない。実機ではA/Bでしか判定できない。
    # ★腕の終端静止は未対応。打ち切り247コマの運用はこの版でも必要(次版ln22で対応予定)。
    # 2026-09-03着。ln22の重みに、実機30走行(9/2)のフレーム0×ロール±2.5度=90状態から
    # 半分のエピソードを開始する「開始バンク」と、足ごと接触摩擦・脚ごとトルク上限の
    # 左右非対称DRを加えて学習した版。ハンドオフ状態からの立て直しを直接学習。
    # 配布元の測定(シム・実機ハンドオフ状態×各20試行):
    #   ② 開始0.5秒の膝突き出し: ln22 完走11/20・膝逸脱16.0度 → ln23 完走20/20・13.1度(★改善)。
    #      「開始直後に前へ倒れる」の原因だった転倒がシム上で消滅。
    #   ① 左ロール: ln22 +0.37度 → ln23 -2.87度と★悪化(バンクに実機の左癖が焼き込まれた仮説)。
    #      鏡像増強版は学習中。実機A/Bで左ロールがどう出るかは要実測。
    # 終端静止を継承し「打ち切り不要(395全部)」。深さは参考値(シム134mm)。
    "sit_up_ln23_r2":         "★★最新 実機開始分布+左右非対称DRで学習。膝突き出し改善(完走20/20)・左ロールは要実測。打ち切り不要",
    "sit_up_ln21_r2":         "参照を厳密に左右対称化(全関節0.000000)。教師97%・ゲート6項目合格",
    # 2026-09-02着の5系統。★配布ゲート6項目に合格したのは ln20 だけ。
    # 8/28夕の指摘「背中から倒れるように座る」への対策として、参照の下降中に
    # 上体を前傾させる相(ピーク+20度・開始1.12秒)を追加したもの。
    # 重みは ar1 の教師と同一で参照を差し替えただけ(学習ステップ0)。
    # 腕の残差は全14関節0.2で reference.npz の action_scale_v に同梱。
    # ★左ロールは5系統どれも未解決(配布元 README_必読.md)。
    "sit_up_ln20_r2":         "★A/Bの対照 前傾相を追加(+20度)。実機14/15完走・平均ロール-10.5度",
    "sit_up_ln20d_r2":        "▲前傾+ゆっくり(下降1.4倍) 教師94%で最高だが深さ99mmで不合格。ゆっくり用",
    "sit_up_ln20s_r2":        "▲ロール対策 効果ゼロ(シムがその領域に入らない)。深さ126mm",
    "sit_up_ln20a_r2":        "▲腕の横方向だけ絞る 3項目不合格・深さ99mm",
    "sit_up_ln20x_r2":        "▲全部入り 3項目不合格・深さ93mmで最も浅い",
    # 2026-08-28着。参照側は明確に良くなっている(waist_pitchの22.5度段差を修正・
    # 手首間隔0.184m一定・終端で腕が静止・下降1.4倍遅い)が、**配布ゲートに不合格**。
    # 配布元の測定(deploy実体・各30試行): 成功 23/30(ar1は27/30)、着座の深さ12mm
    # (ar1は54mm)、下降中の前傾25度(ar1は22度)。手の交差は17/20でほぼ改善せず。
    # 「ar1を主とし比較用に交互で回すこと」と配布元が指示している
    "sit_up_ar4_r2":          "▲ゲート不合格 参照は改善(段差修正/前へならえ)だが成功23/30・着座12mm。比較用",
    "sit_up_ar1_r2":          "★主力 腕に軌道を追加(脚腰はdp4と同一)。成功27/30・着座54mm",
    "sit_up_dp4_r2":          "deepと同じ参照を摩擦較正シムで18万step学習。8/27に◎18本",
    "sit_up_deep_r2":         "深座り(rc比+6.7cm) シム18/20。8/26実機で3本完走",
    "sit_up_rc_r2":           "アンカー 実機較正シムで再学習 95%",
    "sit_up_r2":              "原点アンカー(一番古い基準)",
    "sit_up_rb_r2":           "rc の比較対象 95%(実機開始分布込み)",
    "sit_up_rd_r2":           "❌ 実機で横転(8/24)。ゲート未通過",
    "sit_up_slow_r2":         "❌ 現行条件で8回中5回が早落ち(成功率40%)",
    "sit_lean_r2":            "軽い後傾(-34度) 90%",
    "sit_r2":                 "後傾24度・右ロール17度(記録用。姿勢に癖)",
    "sit_patternA_recline24": "後傾24度・右ロール17度(記録用。姿勢に癖)",
    # --- 登り / 旋回
    "climb_slow_r2":  "慎重に登る版。登りはこれから",
    "climb_r2":       "標準速の登り",
    "turn_wide_r2":   "ワイドスタンス旋回",
    "turn_fine_r2":   "細かい旋回",
}
# ❌ 印のものは実機で失敗の実績がある。UIから消しはしない(比較や再現に要る)
# が、選ぶと確認ダイアログで警告を出す。判断は operator に残す。
PATTERN_WARN = {
    "sit_up_rd_r2": "sit_up_rd_r2 は 2026-08-24 の実機で横に倒れています"
                    "(ゲート未通過の暫定版)。",
    "sit_up_slow_r2": "sit_up_slow_r2 は現行条件の再測定で8回中5回が"
                      "下降途中で早落ちしています(成功率40%)。",
}


def list_patterns():
    out = {"climb": [], "turn": [], "sit": []}
    for d in sorted(DEPLOY.iterdir()):
        if not (d / "policy.pt").exists():
            continue
        n = d.name
        if n.startswith("climb"):
            out["climb"].append(n)
        elif n.startswith(("turn", "bridge")):
            out["turn"].append(n)
        elif n.startswith("sit"):
            out["sit"].append(n)
    return out


def default_pattern(task, fallback="(skip)"):
    """その課題の既定の方策を決める。

    ★PATTERN_NOTES で「★★最新」と印を付けた方策を既定にする。
      新しい方策が来たら PATTERN_NOTES の印を移すだけで既定が追従するので、
      ここのコードを書き換える必要がない(既定の更新漏れで古い方策のまま
      実機を回してしまう事故を防ぐ)。
    印が無ければ deploy へ最後に入ったもの(mtime最新)を既定にする。
    """
    names = list_patterns().get(task, [])
    if not names:
        return fallback
    marked = [n for n in names
              if PATTERN_NOTES.get(n, "").startswith("★★最新")]
    if len(marked) == 1:
        return marked[0]
    if len(marked) > 1:                      # 印の付け替え忘れ。新しい方を採る
        return max(marked, key=lambda n: (DEPLOY / n).stat().st_mtime)
    return max(names, key=lambda n: (DEPLOY / n).stat().st_mtime)


class Engine:
    """FSMエンジン。stateはUIへそのまま出す"""

    # 横方向に効く腕関節(肩ロール・肩ヨー・手首ロール・手首ヨー)。
    # 手の左右位置はここで決まる。前後の釣り合いの錘は肩ピッチ(15,22)と肘(18,25)
    ARM_LAT = (16, 17, 19, 21, 23, 24, 26, 28)
    ARM_ALL = tuple(range(15, 29))

    def __init__(self, robot, is_sim, heartbeat_sec=3.0):
        self.robot = robot
        self.is_sim = is_sim
        self.lock = threading.Lock()
        self.cmd_q = []
        self.fsm = "IDLE"                # IDLE/MOVING/WAIT_CONFIRM/RUNNING/HOLD/DAMP
        self.phase_i = -1
        self.phases = []                 # [(名前, Policy), ...]
        self.t = 0
        self.n = 1
        self.msg = "起動しました。パターンを選んでARMしてください"
        self.step_mode = True            # フェーズ間で操作者確認
        # ヨー合わせ(2026-08-27)。実機でA/Bを取るために切れるようにしてある。
        # 既定はON。切ると2026-08-27 14:13以前と同じ挙動になる
        self.yaw_align = True
        # 腕(15〜28)の残差スケール。1.0 = 従来どおり。
        # ★方策の出力のうち腕だけを縮めて、参照の腕軌道に近づける。
        #   2026-08-28 の実機8本で手が5本クロスした。参照(ar4)は手首間隔を
        #   0.184m一定・左右対称に作り直してあるので、腕の残差を縮めれば
        #   参照どおりの「前へならえ」に寄る。脚腰(0〜14)は触らない
        #   — 触ると釣り合いの取り方そのものが変わってしまう。
        # ★観測に入る last_cmd は縮めない生のまま渡す(学習時と同じ意味にする)。
        #   縮めた値を渡すと方策は「自分は小さく出した」と誤認する。
        # ★2026-09-02より、腕の残差は**方策側の reference.npz が指定**する
        #   (ln20系は腕0.2)。ここは、その上にさらに掛ける現場調整の係数。
        #   既定1.00 = 方策の指定どおり。以前は実機側だけで0.3にしていた
        self.arm_res = 1.0
        # 残差を縮める対象。"all"=腕14関節すべて / "lat"=横方向だけ。
        # ★2026-08-28の実機12本で用量反応が出た:
        #     残差1.0 → 手首間隔 -0.111m(クロス) / 終端傾き29.9度
        #     残差0.5 → 0.107m                  / 8.2度
        #     残差0.3 → 0.135m                  / 6.6度   参照は0.184m / 18.0度
        #   手のクロスは直るが、**着座が浅くなる**。腕は釣り合いの錘でもあるため。
        #   手の左右位置を決めるのは肩ロールと肩ヨー、
        #   前後の釣り合いを担うのは肩ピッチと肘なので、**横方向だけ縮めれば
        #   クロスを直したまま錘を残せる**はず。それが "lat"。
        self.arm_res_mode = "lat"
        # ★2026-09-04 脚腰(0〜14)の残差スケール(試験用・既定1.0=従来どおり)。
        #   実機11本(9/3 ln23)とシムの両方で、方策は参照より0.6〜0.9秒早く座面に乗り、
        #   骨盤が参照の30.5cmに対し13〜28cmしか下がらないまま座るので膝が6〜22cm前へ出る
        #   (docs/着座_膝と回転の解析_20260904.md)。シム(参照開始)では脚残差0.7で
        #   後退17→22cm・膝前方16→9cm・ヨー-12.5→-9.6度と改善したが、残差は
        #   バランスの権限そのものなので、実機開始状態からの試行では0.6で1本転倒した。
        #   **既定では触らない。** A/Bするなら 0.85 から、ハーネス必須。
        self.leg_res = 1.0
        self._fk_shape = None            # 着座の形(sit_shape)のFK。ARM時に作る
        # ★2026-09-04 座面に乗ったあと、膝・足首の残差を抜いて参照の深い着座姿勢へ寄せる。
        #   実機11本+シムで、方策は座面に乗った後も膝を曲げ足首を引く残差を出し続け、
        #   足首〜骨盤の距離が 30→20cm に縮んで浅く座っていた(膝が前に出る)。
        #   学習シーンのシム(参照開始×3シード): 膝+足首の残差を接触後2秒で0へ →
        #   後退 23→32cm(参照30.5)・膝前方 14→3cm・膝角 107→91度・尻荷重 223→327N・
        #   転倒0/3・傾き最大 14→20度。脚腰全部を抜くと傾き最大27度(前傾相が大きく出る)
        #   なので膝・足首だけにする。旧コックピットの「減衰retarget」(ずれを時間で
        #   0へ)を接触後に当てた形。docs/着座_膝と回転の解析_20260904.md §3-4
        #   接触検知は両膝トルクの落ち込み(sit_shape.contact_frame と同じ判定をオンラインで)
        # ★2026-09-04 午後: 実機で 3/3 崩れた(下降の入口で誤検知→残差が消えてロール33°・ヨー90°)。
        #   既定は "off"。docs/着座_膝と回転の解析_20260904.md §3-0b
        self.contact_fade = "off"        # "off"=しない / "0"=残差0へ / "0.3"=0.3まで
        self._kt_low_n = 0               # 接触判定: 低トルクが続いたコマ数
        self._seat_doubt = None          # 完了時に「座面に載っていない疑い」なら理由文
        self.contact_fade_s = 2.0        # 抜くのに掛ける秒数
        self._contact_t = None           # 接触を検知したコマ(走行ごとにリセット)
        self._kt_peak = 0.0              # 降下中の両膝トルク合計のピーク
        self._res_vec = np.ones(29)      # 関節別の残差倍率(接触後に膝・足首が下がる)
        # 1回の走行が終わるたびに要点を1行ぶん積む。UIの[走行の統計]タブが
        # ここを読む。npzを開き直さずに現場でその場で比較できるようにするため。
        # 走らせながらnpzを解析すると制御ループを食う(2026-08-26に19Hzまで
        # 落として転倒させた)ので、走行中に計算済みの値だけを使う。
        self.run_stats = []
        self.stop_frame = 0              # >0 でそのコマ数で打ち切る(0=最後まで)
        # ★2026-08-27。操作者が機体を支えたかどうかは、これまで**どこにも
        #   記録していない最大の未観測変数**。実機23本でどの事前指標も
        #   結果を説明できなかった(開始角速度 r=-0.13 / 開始トルク r=0.14)。
        #   操作者は「立つモードで補助なしに自立できたときは高確率で座れる」と
        #   観察している。数字にするにはここで拾うしかない
        self.assist = "?"
        self.armed = False
        self.action_ramp_s = ACTION_RAMP_S
        self.ref_blend_s = REF_BLEND_S
        self._q0_blend = np.zeros(29)
        # 着座は新しい方策が頻繁に来るので、PATTERN_NOTES の「★★最新」の印から
        # 既定を自動で決める(2026-09-03。従来は sit_up_rc_r2 の固定だった)。
        # climb/turn は方策が安定しているので実績のある既定を据え置く。
        self.sel = {"climb": "climb_slow_r2", "turn": "turn_wide_r2",
                    "sit": default_pattern("sit", "sit_up_rc_r2")}
        # 方策を走り終えたあと、自動で標準モードへ渡すか(2026-09-03)。
        #   "seated"=着座(FSM3)へ渡す(既定) / "sit"=スクワット(FSM2) /
        #   "damp"=ダンプ / "none"=渡さず方策のPDで姿勢維持(従来)
        # 座り終えた姿勢の引き継ぎ先は **着座(FSM3)**。スクワット(FSM2)は
        # 立位でしゃがむモードなので、座った状態から立ち上がろうとしうる。
        # ★2026-09-04 既定を "seated" → "damp" に変更。理由:
        #   - 走行ログ(logs/real 全セッション)に FSM3 への自動移行の実績が1本も無い。
        #     9/3 の全完走回は操作者が 1〜10秒後に手で[ダンプ]を押しており、
        #     椅子に座った姿勢からのダンプは**実機で繰り返し通っている経路**
        #   - 別系統の実機記録(docs/ポータブル版_設計メモ/06 §6.2)では FSM3 は
        #     「床へ倒れ込む着座」で、1.1秒でピッチ −2度 → −27度。椅子の上で
        #     これを掛けると椅子から滑り落ちうる。未検証の経路を既定にしない
        #   FSM3 は選択肢として残す(実機で安全に試せたら既定へ戻す)
        # ★2026-09-04 午後: 既定を元の "seated"(内蔵の着座 FSM3)へ戻した。9/3 14:20〜14:52 の 8 本が
        #   FSM3 自動移行で運用されて問題なし(逆に "damp" は浅く座った機体を前へ崩す)
        self.after_phase = "seated"
        self.obs_b = None
        self.log_dir = None
        self.logs = []
        self.run_i = 0                   # 1セッション中の実行回数(run01, run02…)
        self.single_task = None
        self._rec_rows = []
        self._rec_obs = []
        self._rec_raw = []
        self._rec_path = None
        self.interp = None               # (q0, q_goal, kp, kd, i, steps)
        self._interp_then = "begin"      # 補間の後 "begin"=方策開始 / "hold"=保持
        self.hold_pol = None
        self.hold_t = None
        self.stand = None
        # simは待機中の物理を凍結して置く(実機では操作者が支える)。
        # 凍結しないと、操作を始める前にモックが勝手に崩れる
        self.sim_frozen = bool(is_sim)
        self.busy = False                # ワーカーが実機RPC/方策読込を実行中
        self.busy_what = ""
        self._estop_pending = None       # estop_now が立てる。ループが後始末
        self._last_beat = None              # UIハートビートの最終受信時刻(未受信=None)
        self._heartbeat_sec = heartbeat_sec # 0で無効
        # --- 通信途絶(UIハートビート)への対応(★2026-09-04 改訂)
        #   旧: 途絶8秒で一律 DAMP(kp=0)。立っている機体・座る途中の機体が突然崩れて
        #       転倒した(実機)。ネットワークは安全経路ではないのに、切れた瞬間に
        #       最も危険な操作(脱力)をしていた。
        #   新: 状態で分ける(_ui_loss_plan)。damp にはしない。
        #       方策の実行中(RUNNING)      → 最後まで実行し、完了後は方策で姿勢を保持
        #       方策で保持中(HOLD等)      → そのまま(完了後の自動移行もしない)
        #       方策なしのPD保持(制御権だけ・補間中・立位待機) → 内蔵バランス制御へ返して静止
        #       内蔵歩行(WALK:*)          → 速度ゼロ(autowalk側)。内蔵バランスで静止
        #       内蔵制御中(STD:*)         → 何もしない
        #   物理の異常(傾き40度・関節速度・LowState断・送信断)は従来どおり即DAMP。
        self._ui_lost = False
        self._ui_lost_t = 0.0
        self._ui_lost_what = ""
        self._ui_lost_last = None           # 直近の途絶イベント(UI表示用)
        self._ui_note_t = 0.0
        self._ui_action_t = 0.0
        self._beat_ignore_until = 0.0       # 途絶の模擬(hbdrop。シムでの手順確認用)
        self.sit_gate = None                # 着座前の確認(sit_check)の結果とトークン
        self._want_arm = False           # ワーカーの準備物をループが取り込む
        self._armed_bundle = None
        self._want_begin = None          # 開始するフェーズ番号
        self._nan_frames = 0
        self._dt_hist = []
        self._dt_ms = 0.0
        self._ground = None
        self._ground_t = 0.0
        self._goto_err = None
        self._loop_hist = []
        self._phase_done = False
        self._last_reject = ""
        # ★記録の書き出しは制御ループから外す(_saver スレッド)。
        #   _closing より先にスレッドを起こすと、_saver が未定義属性を読んで
        #   即死する(記録が1本も残らない)。**必ず _closing を先に立てる。**
        self._save_q = {}
        self._save_lock = threading.Lock()
        self._save_ev = threading.Event()
        self._save_n = 0
        self._save_ms = 0.0
        self._closing = False
        # 内蔵歩行(自動歩行・スマホ手動操作)。方策(lowcmd)とは独立の経路で、
        # 内蔵の歩行FSMに SetVelocity を送るだけ。詳細は autowalk.py
        self.walk = WalkController(robot, log=self.log, hb_ok=self._hb_ok,
                                   is_sim=is_sim)
        # ★LiDAR/オドメトリは実機では起動時から読む(2026-09-04 操作者の指示: 自動ON)。
        #   方策の実行中(RUNNING)だけ止め、終わったら再開する(_begin_phase/_end_phase)
        if not is_sim:
            self.walk.enable_sensors(True)
        threading.Thread(target=self._saver, daemon=True, name="saver").start()
        self._th = threading.Thread(target=self._loop, daemon=True,
                                    name="fsm50hz")
        self._th.start()
        # ★安全監視は制御ループから独立させる。ループが詰まっても止まらない
        self._wd = threading.Thread(target=self._watchdog, daemon=True,
                                    name="watchdog")
        self._wd.start()

    def log(self, s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-200:]
            self.msg = s
        # セッションの全操作を「イベント.log」へ追記する。何時何分に何を
        # 押したか・引き継ぎに何秒かかったかは、後から数値で追える唯一の記録
        if self.log_dir is not None:
            try:
                with open(self.log_dir / "イベント.log", "a",
                          encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
            except Exception:                      # noqa: BLE001
                pass

    def command(self, cmd, arg=None):
        with self.lock:
            self.cmd_q.append((cmd, arg))

    def beat(self):
        """UIからのハートビート。HTTPスレッドから直接呼ぶ(キューを経由しない)。"""
        now = time.time()
        if now < self._beat_ignore_until:
            return                                 # 途絶の模擬中(hbdrop)
        if self._ui_lost:
            dur = now - self._ui_lost_t
            self._ui_lost = False
            self._ui_lost_last = {"t": time.strftime("%H:%M:%S"), "dur": round(dur, 1),
                                  "what": self._ui_lost_what}
            self.log(f"UI(通信)復帰: 途絶 {dur:.1f}秒。途絶中の対応: {self._ui_lost_what}")
        self._last_beat = now

    def _pop(self):
        with self.lock:
            return self.cmd_q.pop(0) if self.cmd_q else (None, None)

    def _safety(self):
        """安全条件を1つ返す(問題なければNone)。**副作用なし**。
        50Hzループと監視スレッドの両方から呼ばれる。"""
        q, dq, quat, gyro, tau = self.robot.state()
        if not self.robot.healthy():
            h = self.robot.health_detail()
            if not h["send_hz_ok"]:
                # ★画面は正常なのに指令だけ止まる故障。受信だけ見ていると
                #   見逃す(2026-08-26レビュー指摘)
                return (f"指令の送信が停止({h['send_age_ms']:.0f}ms前"
                        f"{' / ' + h['send_err_msg'] if h['send_err_msg'] else ''})")
            return f"受信途絶({h['state_age_ms']:.0f}ms)"
        up_z = quat_to_mat(quat)[2, 2]
        if up_z < np.cos(np.radians(TILT_LIMIT_DEG)):
            return f"傾き{np.degrees(np.arccos(min(1, max(-1, up_z)))):.0f}度"
        if float(np.abs(dq).max()) > VEL_HARD:
            j = int(np.abs(dq).argmax())
            return f"関節速度超過(j{j} {dq[j]:+.0f}rad/s)"
        return None

    def _monitoring(self):
        """いま安全監視を掛けるべき状態か。

        ★方策を走らせている間だけでは足りない。[custom]で制御権だけ取った
          状態(FSM=IDLE)も、支えているのは当方のPDだけで、傾き・速度・
          通信断の監視が要る(2026-08-26レビュー指摘)。
          標準モード中(STD:*)は内蔵制御が持っているので監視しない。
        """
        if (self.fsm.startswith("STD:") or self.fsm.startswith("WALK:")
                or self.fsm == "DAMP"):
            # WALK:* は内蔵歩行(自動歩行・手動操作)。内蔵バランスが支えている。
            # 歩行中に傾きでdampすると倒れるので、ここでは監視しない
            # (自動歩行側は傾き25度・途絶・ハートビートで**速度ゼロ**にする)
            return False
        if self.fsm in ("MOVING", "RUNNING", "HOLD", "WAIT_CONFIRM"):
            return True
        return bool(getattr(self.robot, "custom_active", False))

    def _watchdog(self):
        """★50Hzループから独立した安全監視(100Hz)。

        メインループはGCの停止・方策の読み込み・SDKのRPCで遅れることがある。
        監視までそこに相乗りしていると、詰まった瞬間に安全系ごと止まる。
        ここは state() を読んで比較するだけで、重い処理もRPCも呼ばない。
        """
        while not self._closing:
            time.sleep(0.01)
            try:
                if self._monitoring():
                    why = self._safety()
                    if why:
                        self.estop_now(why)
                        continue
                # ★UIハートビートの途絶は DAMP しない(2026-09-04)。状態に応じて
                #   「続ける / 保持する / 内蔵バランスへ返す」(_on_ui_lost)。
                if (self._heartbeat_sec > 0 and self._last_beat is not None
                        and time.time() - self._last_beat > self._heartbeat_sec):
                    self._on_ui_lost()
            except Exception:                      # noqa: BLE001
                pass

    # ---------------- 通信途絶(UIハートビート)の扱い ★2026-09-04
    def _ui_loss_plan(self):
        """いまの状態で通信が切れたら何をするか(表示・ログ用の文)"""
        if self.fsm == "RUNNING":
            return "動作(方策)を最後まで実行し、完了後は方策で姿勢を保持(dampしない)"
        if self.fsm in ("HOLD", "WAIT_CONFIRM") and self.hold_pol is not None:
            return "方策で姿勢を保持したまま静止(完了後の自動移行もしない)"
        if self.fsm.startswith("WALK:"):
            return "歩行を止め(速度ゼロ)、内蔵バランスで静止"
        if self.fsm.startswith("STD:") or self.fsm == "DAMP":
            return "内蔵制御のまま(何もしない)"
        if getattr(self.robot, "custom_active", False):
            return "内蔵バランス制御へ返して静止(方策なしのPD保持は数秒で倒れるため。dampはしない)"
        return "何もしない(制御権を持っていない)"

    def _on_ui_lost(self):
        """ハートビート途絶中、監視スレッドから100Hzで呼ばれる(冪等)"""
        now = time.time()
        if not self._ui_lost:
            self._ui_lost = True
            self._ui_lost_t = now
            self._ui_lost_what = self._ui_loss_plan()
            self._ui_note_t = now
            self.log(f"★UI(通信)途絶 {self._heartbeat_sec:.0f}秒: {self._ui_lost_what}")
        if (self.fsm == "RUNNING" or self.fsm.startswith("STD:") or self.fsm == "DAMP"
                or self.fsm.startswith("WALK:")):
            return                                 # 続ける / 内蔵制御 / 自動歩行は自分で止まる
        if self.fsm in ("HOLD", "WAIT_CONFIRM") and self.hold_pol is not None:
            if now - self._ui_note_t > 30.0:       # 保持が長引くときは温度を残す
                self._ui_note_t = now
                tp = np.asarray(getattr(self.robot, "temps", np.zeros(29)))
                self.log(f"UI途絶のまま方策で姿勢維持中({now - self._ui_lost_t:.0f}秒)"
                         f" 温度 最大{tp.max():.0f}度")
            return
        # ここから: 方策なしのPD保持(IDLE / MOVING / 立位待機)で制御権を持っている
        if getattr(self.robot, "custom_active", False) and now - self._ui_action_t > 3.0:
            self._ui_action_t = now
            self._spawn("UI途絶→内蔵バランスへ", self._do_return_to_balance)

    def _do_return_to_balance(self):
        """方策なしのPD保持を内蔵バランス制御へ返す(ワーカー。RPCを含む)"""
        self.interp = None
        self._want_begin = None
        self._interp_then = "begin"
        self.hold_pol = None
        self.hold_t = None
        self.fsm = "IDLE"
        if not getattr(self.robot, "custom_active", False):
            return
        ok, f = self.robot.return_to_balance(log=self.log)
        self.armed = False
        if ok:
            self.fsm = "STD:balance"
            self.log(f"内蔵バランス制御で静止しています(FSM {f})。"
                     f"UIが戻ったら[立つ]/[ダンプ]などで続けてください")
        else:
            self.log("★内蔵制御へ返せませんでした — 現姿勢のPD保持を続けます(バランス無し)。"
                     "リモコンの L2+UP(ロック立位) か L2+B(ダンピング) で引き取ってください")

    # ---------------- 緊急停止(★キューに載せない)
    def estop_now(self, why):
        """どのスレッドからでも直接呼ぶ即時停止。**ブロックしない。**

        旧実装はUIのE-STOPをコマンドキューに積み、50Hzループが拾って
        処理していた。ReleaseMode や方策読み込みでループが詰まると、
        その間E-STOPも安全監視も効かなかった(2026-08-26レビュー指摘)。
        いまはHTTPスレッド/監視スレッドから robot.estop() を直接呼ぶ。
        送信バッファへdampを書くだけなので、500Hz送信スレッドが次の
        パケット(<2ms)で出す。実機のRPCが要る経路(標準モード中)も
        応答を待たずに投げる。
        ★ソフトのdampは物理E-stopの代替にはならない。必ず併用すること。
        """
        with self.lock:
            if self._estop_pending or getattr(self.robot, "_estop_latched",
                                              False):
                return False
            self._estop_pending = why
            self.cmd_q = []                        # 溜まった操作は全部捨てる
        self.robot.estop(why)                      # ← ここで実際に止まる
        w = getattr(self, "walk", None)
        if w is not None:
            w.stop(f"E-STOP: {why}")               # 歩行の速度指令もゼロへ
        self.fsm = "DAMP"
        self.armed = False
        self.log(f"★DAMP: {why}")
        return True

    def _estop_bookkeeping(self):
        """E-STOP後の後始末(記録の保存など)。50Hzループから呼ぶ。
        止めること自体は estop_now が済ませてある。"""
        why, self._estop_pending = self._estop_pending, None
        # 中断した回の記録も必ず残す(final=False で「途中で終わった」印になる)。
        # 失敗した回のログの方が価値が高い。
        # ★ただし**完走済みの回を上書きしない**。_end_phase が final=True を
        #   書いた後にHOLD中のE-STOPで上書きすると、完走した回が「中断」に
        #   化ける(2026-08-26 12:32 の完走回がこれで中断扱いになっていた)。
        if not self._phase_done:
            self._save_rec(final=False)
            self._push_run_stat(f"中断: {why}"[:60])
        self.hold_pol = None
        self.hold_t = None
        self.interp = None
        self._want_begin = None
        gc.enable()
        if not self.robot.custom_active:
            # 標準モード中は robot.estop() が SDK の Damp を投げてある
            self.log("(標準モード中のE-STOP: SDKのDampを送信済み)")

    def _clear_estop(self, why):
        """再実行の直前にだけ、操作者の明示操作でラッチを解除する"""
        if getattr(self.robot, "_estop_latched", False):
            self.robot.clear_estop()
            self.log(f"E-STOPラッチを解除しました({why})")
        self._estop_pending = None

    # ---------------- ブロッキング処理はワーカースレッドへ
    def _spawn(self, what, fn):
        """実機のRPCや方策の読み込みを**50Hzループの外**で走らせる。

        ループの中で呼ぶと、その間だけ安全監視もE-STOPも止まる。
        監視スレッドは独立に動き続けるので、ワーカーが詰まっても停止は効く。
        """
        if self.busy:
            self.log(f"★{self.busy_what} を処理中です。少し待ってください")
            return False
        self.busy = True
        self.busy_what = what

        def run():
            try:
                fn()
            except Exception as e:                 # noqa: BLE001
                self.log(f"★{what}失敗: {e}")
            finally:
                self.busy = False
                self.busy_what = ""
        threading.Thread(target=run, daemon=True, name=what).start()
        return True

    def _handover(self, pol):
        """内蔵制御からの引き継ぎ。**準備が全部終わってから**呼ぶこと。

        順序: 目標=現姿勢で500Hz送信を開始 → 解放 → 指令の到達を実測。
        重い準備(Policy/ObsBuilder/ログ)を制御権取得後にやると、その間は
        PD保持だけでバランスが取れず体幹が傾く(実測: 合計3秒で37度・自動DAMP。
        準備を前へ出して3秒→0.73秒)。所要時間もログに残す。
        ★ワーカースレッドから呼ぶこと(SDKのRPCを含む)。
        """
        t0 = time.time()
        # 内蔵の速度指令をゼロにしてから静定を待つ。ウォーキングFSM(200)は
        # 目標が毎tick変わるので、その途中の姿勢をラッチすると重心移動の
        # 途中で固めることになる(2026-08-24実測: walkから引き継いだ1回だけ
        # 方策開始時の傾き7度。standからの6回は0〜4度)
        if hasattr(self.robot, "stop_move"):
            self.robot.stop_move()
            time.sleep(0.3)
        q0, _, quat0, gyro0, _ = self.robot.state()
        rate0 = float(np.linalg.norm(gyro0[:2]))
        self.log(f"内蔵制御から引き継ぎます(姿勢を保持したまま解放)"
                 f" 角速度{rate0:.2f}rad/s")
        if not self.robot.ensure_custom(kp=pol.kp, kd=pol.kd):
            # ★解放に失敗したら custom_active を立てない(旧実装は無条件に
            #   立てていたので、制御権が無いのに「持っている」と表示された)
            self.log("★引き継ぎに失敗しました。開始を中止します")
            self.fsm = "IDLE"
            return False
        ok, moved, tau, why = self.robot.check_authority()
        if ok:
            self.log(f"指令の到達を確認: OK — 左肩ピッチを+0.060rad指令 → "
                     f"実測{moved:.3f}rad / トルク{tau:.1f}Nm"
                     f"(他の関節は保持したまま)")
        else:
            # ここで damp してはいけない。指令が届いていないなら damp も
            # 届かず、届くなら脱力させることになる。現姿勢保持のまま止める
            self.log(f"★指令が届いていません — {why}。開始を中止します"
                     f"(実測{moved:.3f}rad / トルク{tau:.1f}Nm)")
            self.fsm = "IDLE"
            return False
        q, _, quat, _, _ = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        self.log(f"解放から方策開始まで {time.time() - t0:.2f}秒(傾き{tilt:.0f}度)")
        return True

    # ---------------- 操作(ワーカーで走る本体)
    def _do_arm(self):
        self._clear_estop("ARM")
        phases = []
        for k in ("climb", "turn", "sit"):
            if self.sel[k] != "(skip)":
                phases.append((self.sel[k], Policy(self.sel[k])))
        if not phases:
            self.log("★全て(skip)です。パターンを選んでください")
            return
        obs_b = ObsBuilder(phases[0][1])
        # 開始は自然な両足立位から(climb系の参照fr0は片脚立ちだが、
        # 蒸留済み方策は両足立位スタートでも完走する。10/10で実測)
        sp = ROOT / "motions" / "climb_stand.npz"
        stand = dict(np.load(sp)) if (
            sp.exists() and phases[0][0].startswith("climb")) else None
        log_dir = self._session_dir()
        self._armed_bundle = dict(phases=phases, obs_b=obs_b, stand=stand,
                                  log_dir=log_dir, single_task=None,
                                  begin=None, interp=None)
        self._want_arm = True
        self.log("ARM完了: " + " → ".join(n for n, _ in phases))

    def _do_fsm_read(self):
        """いまのFSM番号を読むだけ(指令は送らない)"""
        f = self.robot.get_fsm_id() if hasattr(self.robot, "get_fsm_id") else None
        NAME = {0: "ゼロトルク", 1: "ダンピング", 2: "スクワット", 3: "着座(床)",
                4: "ロック立位", 200: "運動制御(旧FW)", 500: "歩行",
                501: "歩行(腰3DoF)", 702: "伏せ⇄立ち", 706: "バランススクワット",
                801: "走行", 802: "走行(29dof)", 1000: "UserCtrl(自作制御所有)"}
        self.log(f"FSM = {f} ({NAME.get(f, '不明')})  "
                 f"内蔵制御サービス = {self.robot.current_mode()}")
        self._fsm_id = f

    def _do_custom(self):
        """[カスタム制御へ]。**引き継いだら到達確認まで必ずやる。**

        旧実装は ensure_custom() を呼ぶだけで、指令が届いているかを
        確かめないままFSM=IDLEに戻していた(2026-08-26レビュー指摘)。
        """
        self._clear_estop("カスタム制御へ")
        if hasattr(self.robot, "stop_move"):
            self.robot.stop_move()
            time.sleep(0.3)
        if not self.robot.ensure_custom():
            self.log("★引き継ぎに失敗しました(制御権を取っていません)")
            self.fsm = "IDLE"
            return
        ok, moved, tau, why = self.robot.check_authority()
        # ★2026-08-27 14:23 の事故。到達確認が 実測0.000rad で落ちたのに、
        #   旧実装はログを出すだけで「引き継ぎ完了・現姿勢を保持中」と表示して
        #   IDLE に戻していた。ReleaseMode は既に送ってあるので、
        #   **標準制御は手を離し、こちらの指令も届いていない = 誰も脚を持って
        #   いない**状態になる。実機はここで膝から崩れた。
        #   3日間25回の引き継ぎで初めての失敗(他は全て 実測0.049〜0.075rad)。
        #   黙って進めてはいけない。取り直すか、内蔵ダンピングへ逃がす。
        for _try in range(2):
            if ok:
                break
            self.log(f"★指令が届いていません — {why}"
                     f"(実測{moved:.3f}rad / トルク{tau:.1f}Nm) — 取り直します")
            self.robot.custom_active = False
            if not self.robot.ensure_custom():
                break
            ok, moved, tau, why = self.robot.check_authority()
        if not ok:
            # ここは lowcmd が通っていない。set_damp は内蔵FSMへのRPCなので
            # 別経路で効く。**脱力ではなくダンピング**へ逃がすのが唯一の安全策
            self.log(f"★制御権を取れませんでした({why} / 実測{moved:.3f}rad)。"
                     f"膝から崩れるのでダンプへ逃がします — "
                     f"機体を支えてください。純正standからやり直すこと")
            try:
                self.robot.set_damp()
            except Exception as e:                 # noqa: BLE001
                self.log(f"★ダンプにも失敗: {e} — リモコンのL2+Bで止めてください")
            self.fsm = "DAMP"
            return
        self.log(f"指令の到達を確認: OK(実測{moved:.3f}rad / "
                 f"トルク{tau:.1f}Nm)")
        q, _, quat, _, _ = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        self.fsm = "IDLE"
        self.log(f"カスタム制御へ引き継ぎ(現姿勢を保持中 傾き{tilt:.0f}度)。"
                 f"★このPD保持ではバランスは取れません — "
                 f"支えたまま、速やかに方策を開始するか[ダンプ]へ戻すこと")

    def _do_stand_user(self):
        """[スタンド&カスタム(開発者モード)] — 純正立位→走行(802)→UserCtrl(1000)。

        ★従来の [カスタム制御へ] は ReleaseMode 方式で、解放〜初回指令の間に
          **0.20〜0.50秒 誰もモータを持たない窓**ができる。実機ではここで沈み、
          操作者の支えが要っていた。
          UserCtrl は切替の**前から** rt/user_lowcmd へ流しておけるので、
          この窓が原理的にゼロになる(docs/ポータブル版_設計メモ/09)。

        ★危険: 途中で FSM 802(走行)を通る。静止立位に見えても走行制御が
          動いている状態で、吊った機体が空中で暴れた実績がある(実機13:58)。
          **リモコンE-STOPを握った状態で押すこと。**
        """
        self._clear_estop("スタンド&カスタム")
        if not hasattr(self.robot, "enter_user_ctrl"):
            self.log("★この実機インタフェースは UserCtrl に対応していません")
            return
        ok, why = self.robot.enter_user_ctrl(log=self.log)
        if not ok:
            self.log(f"★UserCtrlへ入れませんでした: {why}")
            self.fsm = "IDLE"
            return
        q, _, quat, _, _ = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        self.fsm = "IDLE"
        self.log(f"★UserCtrl で制御権を取得(傾き{tilt:.0f}度)。"
                 f"脱力窓なし。そのまま [▶ 座る] を押せます")

    def _do_user_run(self, task):
        """[⚡ 立位→UserCtrl→そのまま実行] — 隙間ゼロで方策まで一気に通す。

        ★UserCtrl は**モータの所有権をくれるだけ**で、バランス制御はくれない。
          取った後にやっているのは現姿勢のPD保持だけで、これはどんな姿勢も
          保持できない(sim実測: 参照姿勢は1.5秒で98度、直立でも1秒で29.6度)。
          **バランスを取るのは方策**なので、切替の直後に方策を始めるしかない。

        実機13:09 の実測: 7110→FSM確認→ウィグルの約1秒で、掴んだ姿勢から
        既に 0.204rad(11.7度) ずれていた。docs/ポータブル版_設計メモ/09 も
        「確認秒数_user進入 = 0.0(静的保持を挟まない)」を根拠つきで指示している。

        手順:
          1. 方策とObsBuilderを**先に**読み込む(重い準備を前へ出す)
          2. UserCtrl へ進入(802 → 7110 → FSM1000 → ウィグル)
          3. **そのまま方策開始**(操作者の操作を挟まない)
        """
        name = self.sel[task]
        if name == "(skip)":
            self.log(f"★{task} のパターンが(skip)です")
            return
        if not hasattr(self.robot, "enter_user_ctrl"):
            self.log("★この実機インタフェースは UserCtrl に対応していません")
            return
        self._clear_estop(f"UserCtrl→{name}")
        # 1) 重い準備を先に(制御権を取ってからやるとその間ずっと沈む)
        pol = Policy(name)
        obs_b = ObsBuilder(pol)
        sp = ROOT / "motions" / "climb_stand.npz"
        stand = (dict(np.load(sp)) if task == "climb" and sp.exists() else None)
        log_dir = self._session_dir()
        self.log(f"方策 {name} を読み込みました。UserCtrl へ入ります")
        # 2) UserCtrl 進入(方策のゲインでラッチする)
        ok, why = self.robot.enter_user_ctrl(kp=pol.kp, kd=pol.kd, log=self.log)
        if not ok:
            self.log(f"★UserCtrlへ入れませんでした: {why}")
            self.fsm = "IDLE"
            return
        # 3) 待たずにそのまま開始
        self.sim_frozen = False
        self._armed_bundle = dict(phases=[(name, pol)], obs_b=obs_b,
                                  stand=stand, log_dir=log_dir,
                                  single_task=task, begin=0, interp=None)
        self._want_arm = True
        self.log("★UserCtrl取得 → 待たずに方策を開始します(静的保持を挟まない)")

    # ---------------- 内蔵歩行(自動歩行 / スマホ手動操作) 2026-09-04
    def _hb_ok(self):
        """UIハートビートが生きているか(自動歩行の継続条件)。未接続=True"""
        return (self._heartbeat_sec <= 0 or self._last_beat is None
                or time.time() - self._last_beat < self._heartbeat_sec)

    def _do_walk_ready(self):
        """[歩行モードへ]: ロック立位(4)→走行(802)へ遷移し、LiDAR/オドメトリの
        読み取りを始める。方策の制御権を持ったままでは入らない。"""
        if self.fsm in ("RUNNING", "MOVING") or getattr(self.robot, "custom_active", False):
            self.log("★方策が制御権を持っています。先に[ダンプ]→[立つ]で"
                     "内蔵制御へ戻してください")
            return
        self._clear_estop("歩行モードへ")
        # 歩行の記録先。イベント.log にも歩行の操作・結果を残す(方策を一度も
        # ARM していないセッションでは log_dir が無く、イベントが紙に残らなかった)
        self.walk.log_dir = self._session_dir()
        if self.log_dir is None:
            self.log_dir = self.walk.log_dir
        if self.walk.prepare():
            self.fsm = "WALK:ready"
            self.sim_frozen = False
            self._fsm_id = self.walk.fsm_id
        else:
            self.fsm = "IDLE"

    def _do_walk_auto(self):
        if self.fsm not in ("WALK:ready", "WALK:auto") or not self.walk.ready:
            self.log("★先に[歩行モードへ]を押してください")
            return
        if self.walk.start_auto():
            self.fsm = "WALK:auto"

    def _set_walk_params(self, arg):
        try:
            d = json.loads(arg) if arg else {}
        except Exception:                          # noqa: BLE001
            self.log(f"★歩行パラメータが不正: {arg}")
            return
        ch = self.walk.set_params(d)
        if ch:
            self.log("歩行パラメータ: " + " ".join(ch))

    def walk_stop(self, why="停止"):
        """歩行の速度をゼロにする(dampではない)。HTTPスレッドから直接呼ばれる"""
        self.walk.stop(why)
        if self.fsm == "WALK:auto":
            self.fsm = "WALK:ready"
        self.log(f"歩行停止(速度ゼロ): {why}")

    def tele(self, arg):
        """スマホ十字キー。arg='vx,vy,om'。0.5秒更新が無ければ自動停止"""
        try:
            vx, vy, om = [float(x) for x in str(arg).split(",")]
        except Exception:                          # noqa: BLE001
            return False
        return self.walk.tele(vx, vy, om)

    def _do_walk_go(self, arg):
        """かんたん画面の[前進][横歩き][5cm]。arg=JSON {mode, stop_dist, side_dir, side_dist, avoid}"""
        try:
            d = json.loads(arg) if arg else {}
        except Exception:                          # noqa: BLE001
            self.log(f"★歩行の指定が不正: {arg}")
            return
        if self.fsm not in ("WALK:ready", "WALK:auto") or not self.walk.ready:
            self.log("★先に[歩行モード]を押してください(ロック立位から走行802へ入れます)")
            return
        mode = d.get("mode", "forward")
        keep = {k: v for k, v in d.items() if k in ("stop_dist", "side_dir", "side_dist",
                                                    "avoid", "v_fwd", "dry_run", "step_v")}
        if d.get("nudge"):                         # 5cm の微調整は設定を汚さない
            keep.pop("side_dist", None)
            keep.pop("side_dir", None)
        ch = self.walk.set_params(keep)
        if ch:
            self.log("歩行パラメータ: " + " ".join(ch))
        ov = {"mode": mode}
        if mode == "step":                         # [1歩] ボタン: 向きだけ
            ov["step_dir"] = str(d.get("dir", "left"))
        elif d.get("nudge") and mode == "back":
            ov["back_dist"] = float(d.get("back_dist", 0.05))
        elif d.get("nudge"):
            ov["side_dir"] = d.get("side_dir", "left")
            ov["side_dist"] = float(d.get("side_dist", 0.05))
        if self.walk.start_auto(ov):
            self.fsm = "WALK:auto"

    def _clock_check(self):
        """Jetson の CPU クロックが最大に固定されているか(実機のみ)。[ok, 文] / 判定不能は None"""
        if self.is_sim or platform.machine() != "aarch64":
            return None
        try:
            mn = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq").read()) // 1000
            mx = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq").read()) // 1000
        except Exception:                          # noqa: BLE001
            return None
        if mn >= 1500 and mx >= 1500:
            return [True, f"CPUクロック固定 {mn}MHz(推論が遅くならない)"]
        return [False, f"CPUクロックが固定されていない(min {mn}MHz / max {mx}MHz) — 機体で sudo jetson_clocks "
                       "を実行(自動起動なら再起動で直る)。このままだと 0.2秒で制御周期ガードが働き前へ倒れる"]

    def _do_sit_check(self):
        """[着座(確認へ)]: この位置で座ってよいかを**サーバ側で**点検する。

        結果は sit_gate に入り、30秒以内に同じトークンで sit_go が来たときだけ
        着座を始める(ブラウザ側の確認だけに頼らない)。
        NG(×)が1つでもあれば始めない。△は注意(操作者の判断)。
        """
        items = []
        g = self._go_check()
        for s in g["ng"]:
            if s.startswith("処理中(着座前の確認"):   # この確認自身のワーカーは除く
                continue
            items.append([False, s])
        # ★2026-09-04 12:36: 再起動後に CPU が 729MHz のままで推論が 4〜5 倍遅くなり、0.2 秒で
        #   制御周期ガードがダンプ→立位のまま前へ倒れた。クロック固定(jetson_clocks)を毎回点検する
        ck = self._clock_check()
        if ck is not None:
            items.append(ck)
        for s in g["warn"]:
            items.append([None, s])
        q, dq, quat, gyro, tau = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        items.append([tilt < 10.0, f"傾き {tilt:.1f}度(10度未満)"])
        rms = float(np.sqrt(np.mean(dq ** 2)))
        items.append([rms < 0.10, f"静止(関節速度RMS {rms:.3f}、0.10未満)"])
        f = self.robot.get_fsm_id() if hasattr(self.robot, "get_fsm_id") else None
        self._fsm_id = f
        items.append([(f is None) or (f in (4, 500, 501, 801, 802, 1000)),
                      f"内蔵FSM {f}(立位/歩行/UserCtrlのどれか)" if f is not None
                      else "内蔵FSM 読めず(simなど)"])
        ye = self._yaw_err_deg()
        items.append([None, f"参照との向きのずれ {ye if ye is not None else '-'}度"
                      f"(開始時に自動で合わせる。椅子が真後ろかは目視)"])
        w = self.walk.status()
        if w["sensors"] and w["lidar_age_ms"] is not None and w["lidar_age_ms"] < 1500:
            if w["rear_dist"] is not None:
                items.append([None, f"LiDAR: 後方{w['rear_dist'] * 100:.0f}cmに高さ"
                                    f"{w['rear_h'] * 100:.0f}cmの面(座面?)。参照は踵の2cm後ろから座面"])
            else:
                items.append([None, "LiDAR: 後方に座面らしい面は見えない"
                                    "(頭のLiDARは後ろ下が死角。無いとは言えない)"])
        else:
            items.append([None, "LiDAR未使用 — 椅子の位置は目視で確認"])
        cr = self._crouch_deg(q)
        if cr is not None:
            items.append([None, f"しゃがみ深さ {cr:.0f}度(65度未満なら成功率61%)"])
        ok = all(it[0] is not False for it in items)
        import random
        token = f"{int(time.time())}-{random.randint(1000, 9999)}"
        self.sit_gate = dict(ok=ok, items=items, token=token, t=time.time(),
                             pattern=self.sel["sit"])
        self.log("着座前の確認: " + ("OK(操作者の確認へ)" if ok else "★NGあり") + " / "
                 + " / ".join(("○" if it[0] else ("×" if it[0] is False else "△")) + it[1]
                              for it in items))

    def _do_sit_go(self, arg):
        """確認済みトークンで着座を始める(50Hzループから。実行本体はワーカー)"""
        g = self.sit_gate
        if g is None or not arg or arg != g["token"]:
            self.log("★着座の確認が無い/古い。もう一度[着座(確認)]から")
            return
        if time.time() - g["t"] > 30.0:
            self.sit_gate = None
            self.log("★確認から30秒以上経ちました。もう一度[着座(確認)]から")
            return
        if not g["ok"]:
            self.log("★確認でNGがあるので始めません")
            return
        if self.sel["sit"] != g["pattern"]:
            self.sit_gate = None
            self.log("★確認後に方策が変わりました。もう一度[着座(確認)]から")
            return
        self.sit_gate = None
        self.log("着座を開始します(確認済み)")
        if hasattr(self.robot, "enter_user_ctrl"):
            self._spawn("UserCtrl→sit", lambda: self._do_user_run("sit"))
        else:
            self._spawn("単体実行 sit", lambda: self._do_run_task("sit"))

    def _seat_check(self, name):
        """完了時に座面に載っているかの証拠を見る。疑いがあれば理由文、無ければ None。
        証拠 = 終端0.5秒の両膝トルク中央値が小さい(体重が脚に残っていない)+ 骨盤の後退(sit_shape)。"""
        if not name.startswith("sit"):
            return None
        why = []
        try:
            rows = self._rec_rows[-25:]
            if len(rows) >= 5:
                i3, i9 = REC_COLS.index("tau3"), REC_COLS.index("tau9")
                kt = float(np.median([abs(r[i3]) + abs(r[i9]) for r in rows]))
                if kt >= SEAT_KT_MAX:
                    why.append(f"終端の膝τ合計 {kt:.0f}N·m(脚に体重。座れた回は5〜14)")
            st = self.run_stats[-1] if self.run_stats else {}
            be = st.get("backe")
            if be is not None and float(be) < SEAT_BACK_MIN_CM:
                why.append(f"骨盤の後退 {float(be):+.0f}cm(座れた回は+8〜+28)")
        except Exception:                          # noqa: BLE001
            return None if not why else " / ".join(why)
        return " / ".join(why) if why else None

    def _do_after_phase(self, name):
        """方策の完了後に標準モードへ渡す(2026-09-03)。

        ★椅子に座った姿勢から内蔵モードへ渡すことになる。FSM2(スクワット)は
          立位でしゃがむモードなので、機体が椅子を蹴って立ち上がろうとする
          可能性がある。必ずリモコンのE-STOPを握って見ていること。
        渡す前に少しだけ待つ。方策の最終姿勢がPDで落ち着く前に内蔵制御へ
        渡すと、目標の食い違いぶんだけ動き出しが大きくなる。
        """
        time.sleep(0.5)
        if self.fsm != "HOLD":                     # その間に操作者が介入した
            self.log("完了後の自動移行: 状態が変わったので中止しました")
            return
        if self._ui_lost or not self._hb_ok():
            # ★操作者が見ていない状態で内蔵制御へ渡さない。方策の保持が最も安全
            self.log("UI(通信)途絶中のため、完了後の自動移行は行いません — "
                     "方策で姿勢を保持し続けます(UIが戻ったら手動で)")
            return
        self._do_standard(name)

    def _do_standard(self, name):
        self.armed = False
        self.sim_frozen = False
        # 歩行(自動/手動)は止め、歩行FSMの前提(ready)も下ろす。
        # センサ読み取りは止めない(立位で距離表示を見たいことがある)
        self.walk.stop(f"標準モード {name}")
        self.walk.ready = False
        self._clear_estop(f"標準モード {name}")
        ok = self.robot.standard_mode(name)
        self.fsm = f"STD:{name}" if ok else "DAMP"
        self.log(f"標準モード {name}" + ("" if ok else "(失敗→要確認)"))
        if ok and self.is_sim and name == "stand":
            # ★シムの手順練習: [スタンドロック]で、選択中の着座方策の参照開始位置
            #   (段の上・椅子の前・参照の立位)に置く。実機では内蔵制御が立たせる
            #   ので何もしない。かんたん画面の [スタンドロック]→[着座の確認]→着座 を
            #   シムで通すため(2026-09-04)
            try:
                name_sit = self.sel.get("sit", "(skip)")
                if name_sit != "(skip)":
                    with np.load(DEPLOY / name_sit / "reference.npz") as z:
                        self.robot.place(z["ref_q"][0], z["ref_quat"][0],
                                         z["ref_xy_abs"][0][:2], float(z["ref_z"][0]))
                    self.log(f"(sim) {name_sit} の参照開始位置に立位で配置しました")
            except Exception as e:                 # noqa: BLE001
                self.log(f"(sim) 参照開始位置への配置に失敗: {e}")

    def _do_start(self):
        pol = self.phases[0][1]
        self._clear_estop("START")
        if not self.robot.custom_active:
            if not self._handover(pol):
                return
        self.sim_frozen = False
        q, _, _, _, _ = self.robot.state()
        # 目標は両足立位(保持可能。蒸留方策はここから完走できる)。
        # 立位ファイルが無い課題は参照fr0へ(片脚立ちなら即開始)。
        q_goal = (self.stand["q"] if self.stand is not None else pol.ref_q[0])
        err = float(np.abs(q - q_goal).max())
        if err < 0.30:  # 蒸留方策は開始ずれに頑健。補間(方策なしPD)は最小限に
            self._want_begin = 0
        else:
            steps = int(1.5 * CONTROL_HZ)
            self.interp = [q.copy(), np.asarray(q_goal).copy(),
                           pol.kp, pol.kd, 0, steps]
            self.fsm = "MOVING"
            self.log(f"開始立位へ1.5秒補間(関節差 最大{err:.2f}rad)")

    def _session_dir(self):
        """1セッション = 1フォルダ。run01, run02… が同じ所へ並ぶ。

        以前は [▶ 単体実行] を押すたびに新しいフォルダを作っていたので、
        10本連続で取ると10フォルダに散らばって比較しづらかった。
        [新セッション] を押すまで同じフォルダを使う。
        """
        if getattr(self, "_sess", None) is None:
            self._sess = (ROOT / "logs" / "real"
                          / time.strftime("cockpit_%Y%m%d_%H%M%S"))
            self._sess.mkdir(parents=True, exist_ok=True)
            self.log(f"セッション開始: {self._sess.name}")
        return self._sess

    def _do_ground_check(self):
        """接地を数値で測る。**足首ピッチに微小な指令を入れて反力を見る。**

        なぜ要るか(2026-08-26の実機):
          転倒した回は、右足首が指令どおり0.53rad(30度)スルスル回り、
          足首トルクが+18N·m→0に消えていた。可動域には当たっていない
          (最小-0.43 / 限界-0.87)ので、関節が詰まったのではなく
          **踏む相手がいなかった**。完走した回は同じ指令でも0.16〜0.26radしか
          動かず、トルクを+18〜+22N·m出し続けていた。
          方策は「両足が床を踏んでいる」前提で学習してあるので、体重が
          吊り具に逃げていると足首・膝の指令は物理的に意味を持たない。

        ★他の関節は現姿勢で保持したまま。足首だけ 0.05rad(2.9度) 動かす。
          check_authority と同じ流儀(2026-08-20に脚をkp=0にして膝が
          167度まで沈んだ失敗を繰り返さない)。
        """
        if not self.robot.custom_active:
            self.log("★接地チェックは制御権を持っているときだけ。"
                     "[⤓ 開始姿勢へ] のあとに押してください")
            return
        with self.robot.lock:
            kp = np.asarray(self.robot.kp).copy()
            kd = np.asarray(self.robot.kd).copy()
            tgt0 = np.asarray(self.robot.target_q).copy()
        if float(kp[:15].max()) <= 0:
            self.log("★ゲインが0です(保持していない)")
            return
        # --- 基準(擾乱なし)を0.2秒平均で取る
        qs, ts = [], []
        for _ in range(10):
            q, _dq, _qt, _gy, ta = self.robot.state()
            qs.append(q); ts.append(ta); time.sleep(0.02)
        q0 = np.mean(qs, axis=0); tau0 = np.mean(ts, axis=0)
        # --- 足首ピッチだけ +GC_DELTA
        tgt = tgt0.copy()
        tgt[4] += GC_DELTA
        tgt[10] += GC_DELTA
        self._set_target(tgt, kp, kd, latch=True)
        time.sleep(GC_DUR)
        qs, ts = [], []
        for _ in range(5):
            q, _dq, _qt, _gy, ta = self.robot.state()
            qs.append(q); ts.append(ta); time.sleep(0.02)
        q1 = np.mean(qs, axis=0); tau1 = np.mean(ts, axis=0)
        self._set_target(tgt0, kp, kd, latch=True)      # 元へ戻す(dampしない)
        # --- 判定
        out = []
        for lbl, i in (("左足", 4), ("右足", 10)):
            moved = float(q1[i] - q0[i])
            ratio = abs(moved) / GC_DELTA
            dtau = float(tau1[i] - tau0[i])
            v = ("接地OK" if ratio < GC_RATIO_GROUNDED else
                 "★浮いている疑い" if ratio > GC_RATIO_FLOATING else "△微妙")
            out.append(f"{lbl} 追従率{ratio:.2f}(動き{np.degrees(moved):+.1f}度 "
                       f"τ変化{dtau:+.1f}N) {v}")
        # --- 静的な指標(擾乱なし)も一緒に出す
        knee = abs(float(tau0[3])) + abs(float(tau0[9]))
        ank = float(tau0[4]) + float(tau0[10])
        self.log("接地チェック: " + " / ".join(out))
        self.log(f"  静的指標: 膝トルク合計 {knee:.1f}N·m "
                 f"(今日の完走2本は27.0/30.9・転倒回は33.4) / "
                 f"足首ピッチ合計 {ank:+.1f}N·m (完走 +2.4/+5.9・転倒 +4.9)")
        self._ground_t = time.time()
        self._ground = dict(
            l_ratio=round(abs(float(q1[4] - q0[4])) / GC_DELTA, 2),
            r_ratio=round(abs(float(q1[10] - q0[10])) / GC_DELTA, 2),
            knee=round(knee, 1), ankle=round(ank, 1))

    def _do_goto_start(self, task):
        """選択中の方策の**参照開始姿勢**へゆっくり補間して保持する。

        なぜ要るか(2026-08-26の実機7本から):
          方策は参照開始姿勢から学習してある。実機は[custom]のPD保持で
          しゃがんだ姿勢(膝0.78〜0.85rad。参照は0.55)から始めており、
          1コマ目に 足首+0.86〜0.95rad / 膝-0.42rad の段差が入って
          膝トルクが52〜123N·m跳ねていた。参照軌道自体は最初の1秒で
          最大1.4rad/sしか動かない**ゆっくりした**動きなので、
          急なのは軌道ではなく「開始位置のずれ」。
        ここで先に寄せておけば、その段差が消える。
        """
        name = self.sel[task]
        if name == "(skip)":
            self.log(f"★{task} のパターンが(skip)です")
            return
        self._clear_estop(f"開始姿勢へ {name}")
        pol = Policy(name)
        obs_b = ObsBuilder(pol)
        sp = ROOT / "motions" / "climb_stand.npz"
        stand = (dict(np.load(sp)) if task == "climb" and sp.exists() else None)
        log_dir = self._session_dir()
        if not self.robot.custom_active:
            if not self._handover(pol):
                return
        self.sim_frozen = False            # simの物理を動かす(実機では無関係)
        q, _, _, _, _ = self.robot.state()
        q_goal = np.asarray(stand["q"] if stand is not None else pol.ref_q[0],
                            dtype=float)
        d = float(np.abs(q[:15] - q_goal[:15]).max())
        j = int(np.abs(q[:15] - q_goal[:15]).argmax())
        self.log(f"開始姿勢へ1.5秒で寄せます(いまの脚腰の残差 {d:.3f}rad "
                 f"@ {pol.joint_names[j]})")
        self._interp_then = "hold"
        self._armed_bundle = dict(phases=[(name, pol)], obs_b=obs_b,
                                  stand=stand, log_dir=log_dir,
                                  single_task=task, begin=None,
                                  interp=[q.copy(), q_goal.copy(),
                                          pol.kp, pol.kd, 0,
                                          int(1.5 * CONTROL_HZ)])
        self._want_arm = True

    def _do_run_task(self, task):
        """単体タスク実行: 選択中のパターンを1フェーズだけ走らせる"""
        if self.sel[task] == "(skip)":
            self.log(f"★{task} のパターンが(skip)です")
            return
        self._clear_estop(f"単体実行 {self.sel[task]}")
        # ★[開始姿勢へ]で既に読み込み済みなら、そのまま開始する。
        #   ここで読み直すと1〜2秒、バランスの無いPD保持で待つことになる。
        if (self.armed and self.phases and self.obs_b is not None
                and self.phases[0][0] == self.sel[task]
                and self.single_task == task and self.robot.custom_active):
            self.log(f"準備済みの {self.sel[task]} をそのまま開始します")
            self._want_begin = 0
            return
        # 重い準備は**引き継ぎより前**に全部済ませる。
        # ObsBuilder は MuJoCo モデルを読むので1〜2秒かかり、
        # 制御権を取ってから作ると、その間PD保持だけで
        # バランスが取れず体幹が傾く(実測: 合計3秒で37度傾き、
        # 開始直後に自動DAMP。references/handover.md §3)
        pol = Policy(self.sel[task])
        obs_b = ObsBuilder(pol)
        sp = ROOT / "motions" / "climb_stand.npz"
        stand = (dict(np.load(sp)) if task == "climb" and sp.exists() else None)
        log_dir = self._session_dir()
        # ★準備物はここでは self へ入れない。HOLD中は直前フェーズの方策が
        #   self.obs_b でバランスを取っている。途中で差し替えると、保持中の
        #   方策に別の次元の観測を渡すことになる。差し替えは50Hzループが
        #   フェーズ開始の瞬間にまとめて行う(_want_arm)
        if not self.robot.custom_active:
            if not self._handover(pol):            # 準備が済んでから引き継ぐ
                return
        self.sim_frozen = False
        q, _, quat, _, _ = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        begin, interp = None, None
        if up_z > np.cos(np.radians(20.0)):
            # 直立していれば方策を直接開始する。
            # ハンドオフ後の姿勢は参照と最大0.4rad程度ずれるが、
            # 蒸留方策はその分布で学習済み(バンク)。方策なしの
            # 補間はかえって転倒する(実測)
            begin = 0
        else:
            q_goal = (stand["q"] if stand is not None else pol.ref_q[0])
            interp = [q.copy(), np.asarray(q_goal).copy(),
                      pol.kp, pol.kd, 0, int(1.5 * CONTROL_HZ)]
            self.log(f"単体実行 {self.sel[task]}: 姿勢が崩れている"
                     f"ため開始姿勢へ補間(実機では支えること)")
        # ★差し替えと開始は1つの束にして渡す。別々に渡すと、その隙間の
        #   1コマだけ「保持していた方策が外れて誰も更新しない」状態になる
        self._armed_bundle = dict(phases=[(self.sel[task], pol)], obs_b=obs_b,
                                  stand=stand, log_dir=log_dir,
                                  single_task=task, begin=begin, interp=interp)
        self._want_arm = True

    # ---------------- メインループ(50Hz)
    def _loop(self):
        dt = 1.0 / CONTROL_HZ
        prev = None
        while not self._closing:
            t0 = time.time()
            # ★方策を走らせていなくても常に周期を測る。走らせる**前に**
            #   「いま50Hzで回れているか」を操作者が見られるようにするため。
            #   2026-08-26 11:40 の転倒は19Hzまで落ちていたのが原因で、
            #   その事実は走り終わってログを開くまで誰にも見えなかった。
            now = time.perf_counter()
            if prev is not None:
                self._loop_hist.append((now - prev) * 1000.0)
                if len(self._loop_hist) > 100:
                    self._loop_hist = self._loop_hist[-100:]
            prev = now
            try:
                self._tick()
            except Exception as e:                 # noqa: BLE001
                # ★ここで例外を逃がすと制御スレッドごと死ぬ。画面は残り、
                #   ロボットは最後の目標を保持したまま監視も止まる。
                #   例外は必ず停止に落とす(2026-08-26レビュー指摘)
                self.estop_now(f"制御ループ例外: {type(e).__name__}: {e}")
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def _tick(self):
        # --- E-STOPの後始末(止めること自体は estop_now が済ませてある)
        if self._estop_pending:
            self._estop_bookkeeping()
        # --- 自動歩行が終わったら歩行待機へ戻す(結果は autowalk がログ済み)
        if self.fsm == "WALK:auto" and (self.walk.auto is None
                                        or self.walk.auto.done):
            self.fsm = "WALK:ready"
        # --- ワーカーが用意した方策/ログ先をここでまとめて差し替える
        if self._want_arm:
            b, self._armed_bundle, self._want_arm = self._armed_bundle, None, False
            self.phases = b["phases"]
            self.obs_b = b["obs_b"]
            self.stand = b["stand"]
            self.log_dir = b["log_dir"]
            self.single_task = b["single_task"]
            self.armed = True
            self.phase_i = -1
            self.hold_pol = None
            self.hold_t = None
            self.fsm = "IDLE"
            if b["interp"] is not None:            # 開始姿勢へ補間してから
                self.interp = b["interp"]
                self.fsm = "MOVING"
            elif b["begin"] is not None:           # そのまま方策を開始
                self._want_begin = b["begin"]
        if self._want_begin is not None:
            self.phase_i, self._want_begin = self._want_begin, None
            self._begin_phase()
        # --- コマンド処理(ブロックするものはワーカーへ投げる)
        cmd, arg = self._pop()
        if cmd == "select":
            k, v = arg
            self.sel[k] = v
            self.log(f"パターン選択: {k}={v}")
        elif cmd == "memo" and arg:
            d = self._session_dir()
            try:
                with open(d / "セッションメモ.txt", "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + arg + "\n")
                self.log(f"メモを記録: {arg}")
            except Exception as e:                 # noqa: BLE001
                self.log(f"★メモの保存に失敗: {e}")
        elif cmd == "newsession":
            self._sess = None
            self.run_i = 0
            self.log("次の実行から新しいセッションフォルダを作ります")
        elif cmd == "blend":
            try:
                self.ref_blend_s = max(0.0, min(2.0, float(arg)))
                self.log(f"参照ブレンドを {self.ref_blend_s:.2f}秒 に設定"
                         + ("(無効=従来どおり)" if self.ref_blend_s <= 0 else
                            " — この秒数で現在姿勢から参照へ滑り込ませます"))
            except Exception:                      # noqa: BLE001
                self.log(f"★ブレンドの値が不正: {arg}")
        elif cmd == "ramp":
            try:
                self.action_ramp_s = max(0.0, min(1.0, float(arg)))
                self.log(f"方策出力の立ち上げ(ランプ)を {self.action_ramp_s:.2f}秒 "
                         f"に設定" + ("(無効=従来どおり)" if self.action_ramp_s <= 0 else ""))
            except Exception:                      # noqa: BLE001
                self.log(f"★ランプの値が不正: {arg}")
        elif cmd == "armmode":
            self.arm_res_mode = "lat" if arg == "lat" else "all"
            self.log(f"腕の残差の対象: "
                     + ("横方向のみ(肩ロール/ヨー・手首ロール/ヨー)。"
                        "肩ピッチと肘は縮めないので釣り合いの錘は残る"
                        if self.arm_res_mode == "lat" else "腕14関節すべて"))
        elif cmd == "armres":
            try:
                self.arm_res = max(0.0, min(1.0, float(arg)))
                self.log(f"腕の残差スケール: {self.arm_res:.2f}"
                         + ("(従来どおり)" if self.arm_res >= 0.999 else
                            f" — 腕は参照の{100 * (1 - self.arm_res):.0f}%%ぶん"
                            f"方策の補正を受けなくなります"
                            + ("(完全に参照どおり)" if self.arm_res <= 0.001 else "")))
            except Exception:                      # noqa: BLE001
                self.log(f"★腕の残差スケールが不正: {arg}")
        elif cmd == "cfade":
            if arg in ("off", "0", "0.3"):
                self.contact_fade = arg
                self.log("接触後の膝・足首の残差: " + {
                    "off": "抜かない(従来どおり。浅く座る)",
                    "0": "★2秒で0へ抜く(既定。シムで後退23→32cm)",
                    "0.3": "2秒で0.3まで抜く(控えめ。シムで30cm・傾き最大14度)"}[arg])
            else:
                self.log(f"★接触後の残差の指定が不正: {arg}")
        elif cmd == "legres":
            try:
                self.leg_res = max(0.6, min(1.0, float(arg)))
                self.log(f"★脚腰の残差スケール(試験用): {self.leg_res:.2f}"
                         + ("(従来どおり)" if self.leg_res >= 0.999 else
                            " — 方策のバランス権限を縮めます。ハーネス必須・"
                            "実機未検証。docs/着座_膝と回転の解析_20260904.md"))
            except Exception:                      # noqa: BLE001
                self.log(f"★脚腰の残差スケールが不正: {arg}")
        elif cmd == "stopframe":
            try:
                self.stop_frame = max(0, int(float(arg)))
                self.log(f"打ち切りコマ数: "
                         + (f"{self.stop_frame}コマ({self.stop_frame / CONTROL_HZ:.2f}秒)で止めます"
                            if self.stop_frame else "最後まで走ります(既定)"))
            except Exception:                      # noqa: BLE001
                self.log(f"★打ち切りコマ数が不正: {arg}")
        elif cmd == "afterphase":
            if arg in ("none", "seated", "sit", "damp"):
                self.after_phase = arg
                self.log("方策の完了後: " + {
                    "none": "何もしない(方策のPDで姿勢維持・従来どおり)",
                    "seated": "★着座(FSM3)へ自動で渡します",
                    "sit": "★スクワット(FSM2)へ自動で渡します",
                    "damp": "ダンプへ自動で渡します"}[arg])
            else:
                self.log(f"★完了後の動作が不正: {arg}")
        elif cmd == "assist":
            self.assist = arg or "?"
            self.log(f"補助の記録: {ASSIST_LABEL.get(self.assist, self.assist)}")
        elif cmd == "yawalign":
            self.yaw_align = (arg == "on")
            self.log(f"ヨー合わせ: {'ON(既定)' if self.yaw_align else '★OFF — 14:13以前と同じ挙動'}")
        elif cmd == "mode":
            self.step_mode = (arg == "step")
            self.log(f"進行モード: {'ステップ(各フェーズ前に確認)' if self.step_mode else '自動'}")
        elif cmd == "arm" and self.fsm in ("IDLE", "DAMP", "HOLD"):
            self._spawn("ARM", self._do_arm)
        elif cmd == "place_sim" and self.is_sim and self.armed:
            pol = self.phases[0][1]
            z = pol.ref
            if self.stand is not None:
                s = self.stand
                self.robot.place(s["q"], s["quat"], s["xy"][:2], float(s["z"]))
                self.robot.set_target(s["q"], pol.kp, pol.kd, latch=True)
                self.log("(sim)開始位置に両足立位で配置しました")
            else:
                self.robot.place(pol.ref_q[0], z["ref_quat"][0],
                                 z["ref_xy_abs"][0][:2], float(z["ref_z"][0]))
                self.robot.set_target(pol.ref_q[0], pol.kp, pol.kd, latch=True)
                self.log("(sim)参照開始姿勢で配置しました")
            # 方策なしのPD保持は両足立位でも数秒で倒れる(実測)。
            # simでは待機中は物理を凍結(実機では操作者が支える)
            self.sim_frozen = True
        elif cmd == "start" and self.armed and self.fsm in ("IDLE", "HOLD"):
            self._spawn("START", self._do_start)
        elif cmd == "next" and self.fsm == "WAIT_CONFIRM":
            self._begin_phase()
        elif cmd == "walk_ready":
            self._spawn("歩行モードへ", self._do_walk_ready)
        elif cmd == "walk_auto":
            self._spawn("自動歩行 開始", self._do_walk_auto)
        elif cmd == "walk_param":
            self._set_walk_params(arg)
        elif cmd == "walk_stop":                   # 通常は do_POST が直接処理
            self.walk_stop(arg or "操作者による停止")
        elif cmd == "tele":                        # 通常は do_POST が直接処理
            self.tele(arg)
        elif cmd == "lidar":
            on = (arg == "on")
            self.walk.enable_sensors(on)
            self.log("LiDAR/オドメトリの読み取り: " + ("開始" if on else "停止"))
        elif cmd == "lidar_flip":                  # かんたん画面: [前後を反転]
            try:
                cur = self.walk.mount_yaw_file()
                new = self.walk.set_mount_yaw(cur + 180.0)
                self.log(f"LiDAR の前後を反転しました: yaw_offset_deg {cur:.0f}→{new:.0f}(lidar_mount.json。"
                         "即時に効きます。壁の数字が実物と合うか見てください)")
            except Exception as e:                 # noqa: BLE001
                self.log(f"★LiDAR の反転に失敗: {e}")
        elif cmd == "walk_go":                     # かんたん画面: 前進 / 横歩き / 5cm
            self._spawn("歩行 開始", lambda a=arg: self._do_walk_go(a))
        elif cmd == "sit_check":                   # かんたん画面: 着座前の確認
            self._spawn("着座前の確認", self._do_sit_check)
        elif cmd == "sit_go":
            self._do_sit_go(arg)
        elif cmd == "hbdrop":                      # 通信途絶の模擬(シムでの手順確認用)
            try:
                secs = max(1.0, min(120.0, float(arg or 20)))
            except Exception:                      # noqa: BLE001
                secs = 20.0
            self._beat_ignore_until = time.time() + secs
            if self._last_beat is not None:
                self._last_beat = min(self._last_beat,
                                      time.time() - self._heartbeat_sec - 0.5)
            self.log(f"★通信途絶を模擬します({secs:.0f}秒間 ハートビートを無視)")
        elif cmd and cmd.startswith("mode_"):
            # 標準モード(SDK): zero/damp/stand/walk。方策実行は中断
            name = cmd[5:]
            self._spawn(f"標準モード {name}", lambda n=name: self._do_standard(n))
        elif cmd == "custom":
            self._spawn("カスタム制御へ", self._do_custom)
        elif cmd == "stand_user":
            self._spawn("スタンド&カスタム", self._do_stand_user)
        elif cmd == "user_run" and arg in ("climb", "turn", "sit"):
            g = self._go_check()
            if not g["ok"]:
                self.log("★実行できません: " + " / ".join(g["ng"]))
            else:
                self._spawn(f"UserCtrl→{arg}",
                            lambda a=arg: self._do_user_run(a))
        elif cmd == "fsm_read":
            self._spawn("FSM読み取り", self._do_fsm_read)
        elif cmd == "run_task" and arg in ("climb", "turn", "sit"):
            g = self._go_check()
            if not g["ok"]:
                self.log("★実行できません: " + " / ".join(g["ng"]))
            else:
                self._spawn(f"単体実行 {arg}",
                            lambda a=arg: self._do_run_task(a))
        elif cmd == "ground_check":
            self._spawn("接地チェック", self._do_ground_check)
        elif cmd == "goto_start" and arg in ("climb", "turn", "sit"):
            self._spawn(f"開始姿勢へ {arg}",
                        lambda a=arg: self._do_goto_start(a))

        # --- 安全監視(監視スレッドと二重。どちらが先に見つけてもよい)
        if self._monitoring():
            why = self._safety()
            if why:
                self.estop_now(why)
                self._estop_bookkeeping()
                return
        # --- 状態処理
        if self.fsm == "MOVING" and self.interp:
            q0, qg, kp, kd, i, steps = self.interp
            w = (i + 1) / steps
            w = w * w * (3 - 2 * w)
            # 目標は現在姿勢から始まる(=初期トルク0)ので、kpは最初から
            # フル値でよい。kpをランプすると序盤の支持力が消えて崩れる(実測)
            # ★1コマ目は latch。直前がdamp(目標=ゼロ)だと、現姿勢へ戻すのが
            #   変化量ガードに掛かって何コマもかけて動くことになる
            self._set_target((1 - w) * q0 + w * qg, kp, kd, latch=(i == 0))
            self.interp[4] = i + 1
            if self.interp[4] >= steps:
                self.interp = None
                self.phase_i = 0
                if self._interp_then == "hold":
                    # ★参照開始姿勢へ寄せて、そこでPD保持するだけ。
                    #   方策の1コマ目に入る「段差」を消すための下ごしらえ。
                    #   ここはバランス制御が無いので、支えたまま速やかに
                    #   [▶ 単体実行]するか[ダンプ]へ戻すこと。
                    self._interp_then = "begin"
                    self.fsm = "IDLE"
                    q, _, quat, _, _ = self.robot.state()
                    d = float(np.abs(q[:15] - qg[:15]).max())
                    self._goto_err = d
                    self.log(f"開始姿勢に到達(脚腰の残差 {d:.3f}rad)。"
                             f"★保持中はバランスがありません — "
                             f"支えたまま速やかに [▶ 単体実行] を押すこと")
                    return
                if self._ui_lost:
                    # ★通信途絶中に方策を勝手に始めない。監視スレッドが
                    #   内蔵バランスへ返す(_on_ui_lost)まで現姿勢を保持
                    self.fsm = "IDLE"
                    self.log("補間完了。UI途絶中なので方策は始めず、内蔵バランスへ返します")
                elif self.step_mode and self.stand is not None:
                    self.fsm = "WAIT_CONFIRM"
                    if self.is_sim:
                        self.sim_frozen = True
                    self.log(f"立位で待機中(実機では支えて)。"
                             f"[NEXT]で {self.phases[0][0]} 開始")
                else:
                    self._begin_phase()
        elif self.fsm == "RUNNING":
            name, pol = self.phases[self.phase_i]
            _now = time.perf_counter()
            self._dt_ms = (0.0 if self._rec_prev_t is None
                           else (_now - self._rec_prev_t) * 1000.0)
            q, dq, quat, gyro, tau = self.robot.state()
            _ti = time.perf_counter()
            obs = self.obs_b.build(pol, self.t, q, dq, quat, gyro)
            a = pol.act(obs)
            self._ms_infer = (time.perf_counter() - _ti) * 1000.0
            # ★NaNは指令にしない。観測(IMU/エンコーダ)か方策が壊れた合図で、
            #   そのまま送るとPDの目標がNaNになる。1コマなら直前の目標を
            #   保持して見送り、続くなら止める(2026-08-26レビュー指摘)
            if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(a))):
                self._nan_frames += 1
                self.log(f"★観測/方策出力にNaN({self._nan_frames}コマ連続) — "
                         f"このコマは直前の目標を保持します")
                if self._nan_frames >= 3:
                    self.estop_now("観測/方策出力のNaNが3コマ連続")
                    self._estop_bookkeeping()
                return
            self._nan_frames = 0
            # ★制御周期の監視。方策は50Hz前提。遅れるとスローモーションに
            #   なって釣り合いを失う(2026-08-26に実機2件。上の注記)
            # ★測るのは**ループの周期そのもの**(_loop が反復ごとに記録して
            #   いる値)。以前ここで使っていた「前回の記録から今回の処理開始
            #   まで」は、遅くなると縮む量(=スリープ時間)なので、
            #   周期が遅いときほど小さく出て一度も発火しなかった。
            #   2026-08-26 12:04 の転倒(実測33Hz)で発火しなかったのがこれ。
            recent = self._loop_hist[-LOOP_DT_WINDOW:]
            # ★simモックは論理時間(tick)で物理が進むので、壁時計のループ周期が遅くても
            #   スローモーションにはならない。この周期ガードは実機だけに掛ける
            #   (遅いPCでシムの手順確認が周期ガードで中断されないように。2026-09-04)
            if (not self.is_sim and self.t > LOOP_DT_WINDOW
                    and len(recent) >= LOOP_DT_WINDOW):
                med = float(np.median(recent))
                if med > LOOP_DT_MAX_MS:
                    self.estop_now(
                        f"制御周期が遅い: 直近{LOOP_DT_WINDOW}コマの中央"
                        f"{med:.0f}ms(規定20ms / 限度{LOOP_DT_MAX_MS:.0f}ms)"
                        f" = {1000 / med:.0f}Hz。方策は50Hz前提なので"
                        f"スローモーションになり釣り合いを失う。"
                        f"★PCで他の重い処理(sim_compare/描画/ブラウザ)を"
                        f"止めてから再実行してください")
                    self._estop_bookkeeping()
                    return
            # 開始直後の脚腰の追従誤差を**記録する**(止めはしない。理由は
            # START_TRACK_WINDOW_S の注記)。しきい値を決めるための材料
            if 0 < self.t <= int(START_TRACK_WINDOW_S * CONTROL_HZ):
                _, _tgt, _kp = self._cmd_snapshot()
                if _kp is not None and float(_kp[:15].max()) > 0:
                    terr = float(np.abs(q[:15] - _tgt[:15]).max())
                    if terr > self._early_track:
                        self._early_track = terr
                        self._early_track_j = pol.joint_names[
                            int(np.abs(q[:15] - _tgt[:15]).argmax())]
            self.obs_b.last_cmd = a.copy()      # ★生の出力を観測へ返す
            nr = int(self.action_ramp_s * CONTROL_HZ)
            w = 1.0 if nr <= 0 else min(1.0, (self.t + 1) / nr)
            nb = int(self.ref_blend_s * CONTROL_HZ)
            u = 1.0 if nb <= 0 else min(1.0, (self.t + 1) / nb)
            ref_eff = ((1.0 - u) * self._q0_blend
                       + u * pol.ref_q[min(self.t, pol.n - 1)])
            # ★2026-09-02: 方策ごとの関節別スケール(pol.action_scale)を使う。
            #   ln20系は腕0.2/脚腰0.7が参照npzに同梱されている。
            #   UIの[腕の残差]は、それに**さらに掛ける**補正として働く
            #   (既定1.00なら方策の指定どおり)。
            self._contact_update(name, self.t, tau)
            target = ref_eff + self._scale_res(a) * pol.action_scale * w
            if self.t == 0:
                jm = np.abs(target[:15] - q[:15])
                j = int(jm.argmax())
                self.log(f"方策開始: 1コマ目の跳び {jm.max():.3f}rad "
                         f"({pol.joint_names[j]} / {pol.kp[j]*jm.max():.0f}N·m)"
                         f"  ランプ {self.action_ramp_s:.2f}秒"
                         f" / 参照ブレンド {self.ref_blend_s:.2f}秒"
                         f"  ※完走回の実測は0.459/0.556rad・転倒回は0.404〜0.718")
            self._set_target(target, pol.kp, pol.kd, latch=(self.t == 0))
            self._rec(name, q, dq, quat, gyro, tau, obs, a, target)
            self.t += 1
            _lim = pol.n if self.stop_frame <= 0 else min(self.stop_frame, pol.n)
            if self.t >= _lim:
                # ★参照npzは切らないこと。観測に正規化時刻 t/n と先読み
                #   ref_q[t+k] が入っているので、配列を切ると観測が別物になる。
                #   走行を途中で止めるだけなら、そこまでの観測は通常走行と同一。
                #   sit_up_dp4_r2 は 146コマ(2.92秒)で座り終わって静止しており、
                #   146〜242コマが「背もたれへ18度反る」区間(2026-08-27解析)
                self._end_phase()
        elif (self.fsm in ("HOLD", "WAIT_CONFIRM")
              and getattr(self, "hold_pol", None) is not None):
            # 待機中も直前フェーズの方策で最終コマを維持(バランスあり)
            pol = self.hold_pol
            ht = int(getattr(self, "hold_t", pol.n - 1))
            ht = max(0, min(ht, pol.n - 1))
            q, dq, quat, gyro, tau = self.robot.state()
            obs = self.obs_b.build(pol, ht, q, dq, quat, gyro)
            a = pol.act(obs)
            if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(a))):
                self._nan_frames += 1
                if self._nan_frames >= 3:
                    self.estop_now("保持中の観測/方策出力のNaNが3コマ連続")
                    self._estop_bookkeeping()
                return
            self._nan_frames = 0
            self.obs_b.last_cmd = a.copy()
            self._set_target(pol.ref_q[ht] + self._scale_res(a) * pol.action_scale,
                             pol.kp, pol.kd)
        # simモックは論理時間で進める(壁時計非依存。実機は実時間)
        if self.is_sim and not getattr(self, "sim_frozen", False):
            self.robot.tick(10)

    def _go_check(self):
        """方策を開始してよいかを判定する。

        ★NG(押させない)は、実測で**明確に分離できたものだけ**にする。
          迷う基準で止めると、本当は走れる回まで止めてしまう。
          分離が弱いものは「要注意」にして、押せるが警告を出す。

        2026-08-26 の実機実測(完走3本 / 転倒10本):
          制御ループ  完走 49.7〜50Hz / 転倒 17・23・33・42Hz  → **明確に分離**
          足首τ合計   完走 +2.4/+5.9  / 直近の転倒 −4.4        → 傾向あり(n少)
          膝τ合計     完走 27.0/30.9  / 転倒 25.9〜39.4        → 重なる
          開始ずれ    完走 0.10/0.17  / 転倒 0.10〜0.41        → 重なる
        """
        ng, warn = [], []
        # ★ACアダプタ。2026-08-26 と 2026-08-27 15:23〜15:29 の計5本の転倒が
        #   これ。バッテリー駆動だと CPUガバナが performance でも実クロックが
        #   800MHz〜1.2GHz までしか上がらず(AC時は3.2GHz)、方策の推論が
        #   間に合わずループが19〜22Hzに落ちる。周期ガードがDAMPを掛け、
        #   DAMPは姿勢を保持しないのでそのまま前に倒れる。
        #   ★ループ速度のNGは「走り出してから」しか効かない(待機中の
        #     コックピットは推論していないので50Hz出てしまう)。
        #     電源は走る前に分かるので、ここで止める。
        pw = self._power_state()
        if pw.get("ac") == 0 and not self.is_sim:
            ng.append("★ACアダプタが抜けています — バッテリー駆動では"
                      f"CPUが{pw.get('mhz', 0):.0f}MHzまでしか上がらず、"
                      "制御ループが19〜22Hzに落ちて必ず前に倒れます"
                      "(2026-08-26/27に計5本の転倒実績)")
        elif pw.get("mhz") and pw["mhz"] < 2500 and platform.machine() == "x86_64":
            # 最大コアが2.5GHzに届かない = 本当に上がっていない。
            # ★ただしこれは「いま何も走っていない」ときも起こりうる。
            #   走行中の実測は制御周期(dt_ms)で見ること
            # ★機体(Jetson, aarch64)は最大2GHz級・schedutilで暇なときは730MHzまで
            #   落ちるのが正常なので、この警告はノートPC(x86_64)だけに出す(2026-09-04)
            warn.append(f"CPUの最大コアが{pw['mhz']:.0f}MHzまでしか"
                        f"上がっていない — ガバナ={pw.get('gov', '?')} / "
                        f"power-profiles-daemon を疑う")
        h = self.robot.health_detail()
        lp = self._loop_stat()
        if self.busy:
            ng.append(f"処理中({self.busy_what})")
        if self.fsm in ("RUNNING", "MOVING"):
            ng.append(f"実行中({self.fsm})")
        if self.walk.auto is not None and not self.walk.auto.done:
            ng.append("自動歩行の実行中 — 先に[歩行停止]")
        if h.get("estop_latched"):
            ng.append("E-STOPラッチ中 — [⤓ 開始姿勢へ]で解除")
        if not self.robot.healthy():
            ng.append("受信途絶または送信停止")
        if lp["hz"] and lp["hz"] < 45.0 and not self.is_sim:
            ng.append(f"制御ループ {lp['hz']:.0f}Hz(45Hz未満) "
                      f"— 電源アダプタとCPUガバナを確認")
        q, dq, quat, gyro, tau = self.robot.state()
        if not self.robot.custom_active:
            warn.append("まだ制御権が無い(押せば自動で引き継ぎます)")
        cr = self._crouch_deg(q)
        if cr is not None and cr >= CROUCH_TH:
            warn.append(f"深くしゃがんだ立ち方(しゃがみ深さ{cr:.0f}度 "
                        f"≧ {CROUCH_TH:.0f}度)。この立ち方からの着座成功は "
                        f"4/21(19%) — 浅い立ち方なら 14/23(61%)。"
                        f"支えを完全に離して、自分で立ち切らせてから押すこと")
        rms = float(np.sqrt(np.mean(dq ** 2)))
        if rms > 0.05:
            warn.append(f"関節が動いている(速度RMS {rms:.3f} > 0.05)")
        g = getattr(self, "_ground", None)
        gt = getattr(self, "_ground_t", 0.0)
        if g is None:
            warn.append("接地チェック未実施")
        elif time.time() - gt > 120:
            warn.append(f"接地チェックが古い({time.time() - gt:.0f}秒前)")
        elif max(g["l_ratio"], g["r_ratio"]) > 0.30:
            warn.append(f"接地が怪しい(追従率 左{g['l_ratio']} 右{g['r_ratio']})")
        e = getattr(self, "_goto_err", None)
        if self.ref_blend_s <= 0:
            # ブレンド無効のときだけ、開始姿勢を合わせておく必要がある
            if e is None:
                warn.append("[⤓ 開始姿勢へ]未実施(参照ブレンドが0秒のため)")
            elif e > 0.25:
                warn.append(f"開始姿勢の残差 {e:.2f}rad(> 0.25)")
        ank = float(tau[4]) + float(tau[10])
        if self.robot.custom_active and not (0.0 <= ank <= 8.0):
            warn.append(f"足首τ合計 {ank:+.1f}N·m(完走時は+2.4〜+5.9)")
        knee = abs(float(tau[3])) + abs(float(tau[9]))
        if self.robot.custom_active and not (18.0 <= knee <= 42.0):
            warn.append(f"膝τ合計 {knee:.1f}N·m(完走時は27〜31)")
        return {"ok": not ng, "ng": ng, "warn": warn}

    def _loop_stat(self):
        """制御ループの実測周期。UIに常時出す"""
        h = self._loop_hist
        if len(h) < 10:
            return {"ms": 0.0, "hz": 0.0, "max": 0.0, "ok": True}
        a = np.array(h)
        med = float(np.median(a))
        return {"ms": round(med, 1), "hz": round(1000.0 / max(med, 1e-6), 1),
                "max": round(float(a.max()), 1),
                "ok": bool(med <= LOOP_DT_MAX_MS * 0.75)}

    def _cmd_snapshot(self):
        """いま実機へ出している目標とゲインを読む(読むだけ)"""
        try:
            with self.robot.lock:
                return (None, np.asarray(self.robot.target_q).copy(),
                        np.asarray(self.robot.kp).copy())
        except Exception:                          # noqa: BLE001
            return None, None, None

    def _crouch_deg(self, q):
        """しゃがみ深さ[度] = 膝角 - 足首角(左右平均)。上の CROUCH_TH の注記参照。"""
        try:
            knee = (float(q[3]) + float(q[9])) / 2.0
            ank = (float(q[4]) + float(q[10])) / 2.0
            return round(float(np.degrees(knee - ank)), 1)
        except Exception:                          # noqa: BLE001
            return None

    def _power_state(self, _cache=[0.0, {}]):
        """ACアダプタとCPU実クロック。/state から毎周期呼ばれるので2秒キャッシュ。"""
        now = time.time()
        if now - _cache[0] < 2.0:
            return _cache[1]
        out = {}
        try:
            for f in sorted(pathlib.Path("/sys/class/power_supply").glob("A*/online")):
                out["ac"] = int(f.read_text().strip())
                break
        except Exception:                          # noqa: BLE001
            pass
        try:
            fs = sorted(pathlib.Path("/sys/devices/system/cpu").glob(
                "cpu*/cpufreq/scaling_cur_freq"))
            v = sorted(int(f.read_text().strip()) for f in fs)
            if v:
                # ★中央値を見てはいけない。16コアのうち動いているのは
                #   制御ループの1コアだけで、残りは省電力で800MHzに落ちる。
                #   2026-08-27〜28に、この中央値800MHzを2度「異常」と誤判定した。
                #   意味があるのは**いちばん回っているコア**の周波数。
                out["mhz"] = v[-1] / 1000.0
                out["mhz_med"] = v[len(v) // 2] / 1000.0
            g = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            out["gov"] = g.read_text().strip()
        except Exception:                          # noqa: BLE001
            pass
        _cache[0], _cache[1] = now, out
        return out

    def _pol_scale(self):
        """選んでいる方策が指定する残差スケール。UIで見えないと、
        腕が学習時の何倍で動いているのか現場で分からない。"""
        try:
            name = self.sel.get("sit", "(skip)")
            if name == "(skip)":
                return None
            c = getattr(self, "_ps_cache", None)
            if c is None:
                c = self._ps_cache = {}
            if name not in c:
                # ★npzは開いたままにしない(遅延読み込みのzipハンドルが残ると
                #   別スレッドの読みと交錯して BadZipFile を起こす。2026-09-03)
                with np.load(DEPLOY / name / "reference.npz") as z:
                    if "action_scale_v" in z.files:
                        v = np.asarray(z["action_scale_v"], dtype=float)
                        c[name] = dict(v=True, leg=round(float(v[:15].max()), 2),
                                       arm=round(float(v[15:].max()), 2))
                    else:
                        sc = (float(z["action_scale"])
                              if "action_scale" in z.files else 0.7)
                        c[name] = dict(v=False, leg=round(sc, 2), arm=round(sc, 2))
            return c[name]
        except Exception:                          # noqa: BLE001
            return None

    def _scale_arm(self, a):
        """方策出力の腕成分を縮めた配列を返す。

        脚腰(0〜14)は**既定では触らない**(leg_res=1.0)。leg_res<1 は 2026-09-04 の
        試験用オプションで、操作者が詳細設定から明示的に選んだときだけ効く。
        """
        if self.arm_res >= 0.999 and self.leg_res >= 0.999:
            return a
        out = a.copy()
        if self.arm_res < 0.999:
            idx = (self.ARM_LAT if self.arm_res_mode == "lat" else self.ARM_ALL)
            out[list(idx)] *= self.arm_res
        if self.leg_res < 0.999:
            out[:15] *= self.leg_res
        return out

    CONTACT_FADE_JOINTS = (3, 4, 5, 9, 10, 11)     # 左右の 膝・足首ピッチ・足首ロール

    def _contact_update(self, name, t, tau):
        """座面接触の検知と、接触後の膝・足首の残差倍率(_res_vec)の更新。RUNNINGで毎コマ呼ぶ"""
        if not name.startswith("sit") or self.contact_fade == "off":
            return
        kt = float(abs(tau[3]) + abs(tau[9]))
        if 25 <= t < 100:
            self._kt_peak = max(self._kt_peak, kt)
        if self._contact_t is None:
            # ★実機の膝トルクは下降の入口(0.5〜1.0秒)で符号が変わる途中にゼロ付近を通る。
            #   参照の座面到達(1.9秒)より前では検知せず、低トルクが10コマ続いたときだけ検知する
            low = (t >= CONTACT_FADE_MIN_T and self._kt_peak > 8.0 and kt < 0.4 * self._kt_peak)
            self._kt_low_n = self._kt_low_n + 1 if low else 0
            if self._kt_low_n >= 10:
                self._contact_t = t
                floor = float(self.contact_fade)
                self.log(f"座面接触を検知({t / CONTROL_HZ:.2f}秒、膝τ {kt:.1f}N·m<ピーク"
                         f"{self._kt_peak:.1f}の40%) → 膝・足首の残差を"
                         f"{self.contact_fade_s:.1f}秒で{floor:.1f}へ抜きます(深く座るため)")
            return
        floor = float(self.contact_fade)
        n = max(1, int(self.contact_fade_s * CONTROL_HZ))
        f = max(floor, 1.0 - (t - self._contact_t) / n * (1.0 - floor))
        self._res_vec[list(self.CONTACT_FADE_JOINTS)] = f

    def _scale_res(self, a):
        """腕の縮小(_scale_arm)に加え、接触後の膝・足首の倍率を掛ける"""
        out = self._scale_arm(a)
        if float(self._res_vec.min()) < 0.999:
            out = out * self._res_vec
        return out

    def _pose_rows(self, q):
        """立位の姿勢を、うまくいった回のレンジと並べて返す。
        [名前, 左値, 左判定, 右値, 右判定] の並び(右が無い腰は None)。
        判定: 0=レンジ内 1=外れ小(レンジ幅の半分以内) 2=外れ大"""
        def one(name, i):
            v = float(q[i])
            lo, hi = GOOD_RANGE.get(name, (None, None))
            if lo is None:
                return [round(v, 3), 0, None, None]
            w = max(hi - lo, 1e-3)
            d = (lo - v) if v < lo else ((v - hi) if v > hi else 0.0)
            return [round(v, 3), (0 if d <= 0 else (1 if d <= 0.5 * w else 2)),
                    round(lo, 3), round(hi, 3)]
        rows = []
        for lbl, ln, rn, li, ri in POSE_KEYS:
            rows.append([lbl] + one(ln, li) + one(rn, ri))
        for lbl, nm, i in POSE_WAIST:
            rows.append([lbl] + one(nm, i) + [None, None, None, None])
        return rows

    def _yaw_err_deg(self):
        """いまの体の向きが、選んでいる方策の参照からヨーで何度ずれているか。

        観測には体の向きがワールド座標で入る(run_fsm.ObsBuilder.reset の注記)。
        実機IMUのヨーには絶対基準が無いので、電源を入れ直すとここが動く。
        2026-08-27はこれが+90度まで開いて、座らずに捻れる形で出た。
        """
        # 走行前は phases が空なので、UIで選んでいる着座方策の参照を見る。
        # ★分からないときは 0 ではなく None を返すこと。0を返すと
        #   「ずれていない」と読めてしまい、壊れているのに合格に見える
        try:
            if 0 <= self.phase_i < len(self.phases):
                ref_q = self.phases[self.phase_i][1].ref["ref_quat"][0]
            else:
                name = self.sel.get("sit", "(skip)")
                if name == "(skip)":
                    return None
                ref_q = self._ref_quat0(name)
                if ref_q is None:
                    return None
            _q, _dq, quat, _g, _t = self.robot.state()
            d = np.degrees(_yaw_of(quat) - _yaw_of(ref_q))
            return round(float((d + 180.0) % 360.0 - 180.0), 1)
        except Exception:                          # noqa: BLE001
            return None

    def _ref_quat0(self, name):
        """方策名 → 参照の開始クォータニオン。表示のためだけに毎周期
        Policy() を作ると推論器まで読み込むので、reference.npz だけ引いて覚える"""
        c = getattr(self, "_refq_cache", None)
        if c is None:
            c = self._refq_cache = {}
        if name not in c:
            try:
                z = np.load(DEPLOY / name / "reference.npz")
                c[name] = np.asarray(z["ref_quat"][0], dtype=float)
            except Exception:                      # noqa: BLE001
                c[name] = None
        return c[name]

    def _set_target(self, q, kp, kd, latch=False):
        """robot.set_target のラッパ。**拒否されたら黙らせない。**"""
        ok, why = self.robot.set_target(q, kp, kd, latch=latch)
        if not ok and why and why != self._last_reject:
            self.log(f"★指令が拒否されました: {why}")
        self._last_reject = why if not ok else ""
        return ok

    def _begin_phase(self):
        name, pol = self.phases[self.phase_i]
        q, _dq, quat, gyro, _tau = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        rate = float(np.linalg.norm(gyro[:2]))
        # 立位→方策の引き継ぎでは、目標=現在姿勢のPDが重力に釣り合うまで沈む。
        # 2026-08-21実測: 姿勢チェック時2.5〜2.8度が方策開始時点で12〜19度に育ち、
        # 開始0.4秒後に31〜40度のピーク(中止しきい値40度に対し余裕2〜8度)。
        # ここで開始時点の値を必ず残す。これが無いと後から切り分けられない。
        # 開始姿勢が参照からどれだけ離れているか。**傾きとは別に必ず見る。**
        q_goal = np.asarray(self.stand["q"] if self.stand is not None
                            else pol.ref_q[0], dtype=float)
        derr = np.abs(q - q_goal)
        emax = float(derr.max())
        worst = ", ".join(
            f"{pol.joint_names[i]} {derr[i]:.2f}"
            for i in np.argsort(-derr)[:3])
        self.log(f"引き継ぎ計測: 傾き{tilt:.1f}度 角速度{rate:.2f}rad/s "
                 f"開始姿勢のずれ 最大{emax:.2f}rad ({worst})")
        # 傾きが大きいのに角速度が小さい = 静的な前傾は、重心が支持多角形の外で
        # 静止した「倒れ確定」姿勢で、踏み出し無しでは物理的に回復不能
        # (2026-08-21 §5: 実バンクの該当2状態だけが0/10。除外して初めて合格した)。
        # 動的なトランジェント(戻り方向の角速度つき)とは区別して止める。
        if tilt > HANDOVER_STATIC_TILT_DEG and rate < HANDOVER_STATIC_RATE:
            # **ここで damp してはいけない。** 機体はいま自前PDで立っている。
            # kp=0 にすればその瞬間に崩れる(スキル原則1: 安全のために足した
            # 機構が事故を起こす)。現姿勢の保持を続けたまま開始だけ拒否する。
            self._set_target(q, pol.kp, pol.kd, latch=True)
            self.fsm = "IDLE"
            self.log(f"★開始を拒否: 静的な前傾{tilt:.0f}度"
                     f"(角速度{rate:.2f}rad/s)。現姿勢を保持中 — "
                     f"支えて立て直してから再実行してください")
            return
        self.sim_frozen = False
        self.hold_pol = None
        self.hold_t = None
        self._nan_frames = 0
        self._early_track = 0.0
        self._early_track_j = ""
        self._dt_hist = []
        self._dt_ms = 0.0
        self._contact_t = None                     # 接触後の残差抜きを走行ごとに初期化
        self._kt_low_n = 0
        self._seat_doubt = None
        self._kt_peak = 0.0
        self._res_vec = np.ones(29)
        # ★方策の走行中は LiDAR/オドメトリの読み取りを止める(機体ではGILの
        #   取り合いが制御周期に直接効く。docs/オンボード運用.md §3)。
        #   自動歩行が動いていたら速度ゼロにしてから方策へ
        self.walk.stop("方策開始")
        self.walk.ready = False
        self.walk.enable_sensors(False)
        self._start_pose_err = emax
        self._start_pose_worst = worst
        # ★参照ブレンドの起点。いまの姿勢から参照へ滑り込ませる
        self._q0_blend = np.asarray(q, dtype=float).copy()
        self._phase_done = False
        # ★走行中はGCを止める。1フェーズは7.9秒・395コマなので、その間だけ
        #   回収を止めても増え方は知れている。走り終わったら戻して回収する
        gc.disable()
        # ★ヨー合わせ(2026-08-27)。実測のIMUヨーには絶対基準が無く、電源を
        #   入れ直すたびに原点がずれる。観測には体の向きがワールドの回転行列
        #   として入るので、ずれると方策は「向きを直しながら座る」という
        #   学習していない課題を解かされる。実測13本で 符号つき相関 r=0.83
        _kw = (dict(quat=quat, ref_quat=pol.ref["ref_quat"][0])
               if self.yaw_align else {})
        yaw_off = self.obs_b.reset(est_xy=pol.ref["ref_xy_abs"][0][:2], **_kw)
        self._yaw_off_deg = float(np.degrees(yaw_off))
        if not self.yaw_align:
            self.log(f"★ヨー合わせOFF のまま開始します"
                     f"(実測は参照より{self._yaw_err_deg()}度ずれています)")
        elif abs(self._yaw_off_deg) > 1.0:
            self.log(f"ヨー合わせ: 実測の向きが参照より{-self._yaw_off_deg:+.1f}度"
                     f"ずれていたので参照側の座標系へ揃えました"
                     f"(2026-08-27より。これをしないと方策は向きの修正を"
                     f"同時にやろうとして膝の左右差が開きます)")
        self.t = 0
        self.n = pol.n
        self.start_tilt = tilt
        self.start_rate = rate
        self._rec_rows = []
        self._rec_obs = []
        self._rec_raw = []
        self._rec_prev_t = None
        self._phase_t0 = time.time()
        if self.phase_i == 0:            # 通しの1回 = run。設定を先に残す
            self.run_i += 1
            self._write_run_meta()
            self.log(f"補助: {ASSIST_LABEL.get(self.assist, self.assist)}"
                 f"  ★立つモードで自立できていたかどうかが結果を決めている疑い")
        self._last_route = ("UserCtrl" if getattr(self.robot, "_use_user_topic", False)
                            else "カスタム")
        self.log(f"── run{self.run_i:03d} 開始 ({self.log_dir.name})")
        self._rec_path = (self.log_dir /
                          f"run{self.run_i:03d}_{self.phase_i}_{name}.npz")
        self.fsm = "RUNNING"
        tp = np.asarray(getattr(self.robot, "temps", np.zeros(29)))
        self.log(f"フェーズ開始: {name}({pol.n}コマ / {pol.n / CONTROL_HZ:.1f}秒)"
                 f"  温度 膝 L{tp[3]:.0f}/R{tp[9]:.0f}度"
                 f"  足首 L{tp[4]:.0f}/R{tp[10]:.0f}度"
                 f"  最大{tp.max():.0f}度(j{int(np.argmax(tp))})")

    def _write_run_meta(self):
        """run<NN>_設定.json — 何をどの設定で回したかの唯一の記録。
        ab_report.py がこれと npz を突き合わせて集計する"""
        name0, pol0 = self.phases[0]
        meta = {
            "run": self.run_i,
            "phases": [n for n, _ in self.phases],
            "single_task": self.single_task,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_sim": bool(self.is_sim),
            "step_mode": bool(self.step_mode),
            "control_hz": CONTROL_HZ,
            "action_scale": ACTION_SCALE,
            "action_ramp_s": float(self.action_ramp_s),
            "ref_blend_s": float(self.ref_blend_s),
            "tilt_limit_deg": TILT_LIMIT_DEG,
            "vel_hard": VEL_HARD,
            "kp": [float(x) for x in pol0.kp],
            "kd": [float(x) for x in pol0.kd],
            "joint_names": list(pol0.joint_names),
            "n_frames": int(pol0.n),
            "duration_s": round(pol0.n / CONTROL_HZ, 2),
            "temps_start": [float(x) for x in
                            np.asarray(getattr(self.robot, "temps",
                                               np.zeros(29)))],
            "policy_meta": {n: p.meta for n, p in self.phases},
            # 2026-08-26追加。実機ガードが効いた回かどうかを後から数値で見る
            "guard": dict(getattr(self.robot, "health_detail", dict)()),
            # しきい値を決めるための材料。走行ごとに貯める(上の注記)
            "start_pose_err": round(float(getattr(self, "_start_pose_err", 0)), 3),
            "start_pose_worst": getattr(self, "_start_pose_worst", ""),
            # 2026-08-27追加。開始時のIMUヨーが参照からどれだけずれていたか。
            # ヨー合わせで打ち消しているので走行結果には出ないが、
            # 「合わせが効いた回かどうか」を後から必ず区別できるようにする
            "yaw_offset_deg": round(float(getattr(self, "_yaw_off_deg", 0.0)), 2),
            "yaw_align": bool(self.yaw_align),
            # 立つモードで自立できていたか(操作者の申告)。★事前指標で唯一
            # 結果を説明できそうな候補なので、走行ごとに必ず残す
            "assist": self.assist,
            "stop_frame": int(self.stop_frame),
            "after_phase": self.after_phase,
            "arm_res": float(self.arm_res),
            "arm_res_mode": self.arm_res_mode,
            "leg_res": float(self.leg_res),
            "contact_fade": self.contact_fade,
            "contact_fade_s": float(self.contact_fade_s),
        }
        try:
            (self.log_dir / f"run{self.run_i:03d}_設定.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:                     # noqa: BLE001
            self.log(f"★設定の保存に失敗: {e}")

    def _save_rec(self, final):
        """逐次保存を**依頼するだけ**。実際の書き出しは _saver スレッド。

        ★np.savez_compressed は実測で 100コマ19.7ms / 300コマ31.0ms /
          395コマ43.2ms かかる。50Hzループの中でやると、その回だけ制御が
          1〜2コマ止まる。しかも完走直後(=方策が姿勢を保持している一番
          大事な瞬間)に43msの穴が空いていた。
        行リストは参照のコピーだけ渡す(追記後に中身を書き換えないので安全)。
        """
        if self._rec_path is None or not self._rec_rows:
            return
        with self._save_lock:
            # ★走行ごとに1件ずつ溜める(1枠だと連続実行で前の回が捨てられる)。
            #   同じファイルへの依頼は最新で上書きしてよい。
            self._save_q[str(self._rec_path)] = (
                self._rec_path, list(self._rec_rows), list(self._rec_obs),
                bool(final), list(self._rec_raw))
        self._save_ev.set()

    def _saver(self):
        """記録をディスクへ書く専用スレッド。取りこぼしても最新が必ず残る"""
        while not self._closing:
            self._save_ev.wait(0.5)
            self._save_ev.clear()
            with self._save_lock:
                if not self._save_q:
                    continue
                k = next(iter(self._save_q))
                job = self._save_q.pop(k)
            path, rows, obs, final, raw = job
            t0 = time.perf_counter()
            try:
                r = np.asarray(raw, dtype=np.float32)
                n = len(r)
                np.savez_compressed(
                    path,
                    rec=np.asarray(rows, dtype=np.float32),
                    cols=np.array(REC_COLS),
                    obs=np.asarray(obs, dtype=np.float32),
                    final=np.array(bool(final)),
                    raw_sensor=r[:, :58].reshape(n, 29, 2) if n else r,
                    raw_reserve=r[:, 58:174].reshape(n, 29, 4) if n else r,
                    motor_ext=r[:, 174:204].reshape(n, 6, 5) if n else r)
                self._save_n += 1
                self._save_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:                 # noqa: BLE001
                self.log(f"★記録の保存に失敗: {e}")

    def _drain_saves(self, timeout=5.0):
        """終了前に書き残しを吐き切る"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._save_lock:
                if not self._save_q:
                    return True
            self._save_ev.set()
            time.sleep(0.05)
        return False

    def _push_run_stat(self, how):
        """走行1本の要点を run_stats に積む。**npzを開き直さない。**"""
        try:
            rows = self._rec_rows
            if len(rows) < 5:
                return
            iq = [REC_COLS.index(f"q{i}") for i in range(29)]
            ia = [REC_COLS.index(f"act{i}") for i in range(29)]
            it = REC_COLS.index("tilt_deg")
            idt = REC_COLS.index("dt_ms")
            q0 = np.array([rows[0][k] for k in iq])
            qe = np.array([rows[-1][k] for k in iq])
            act = np.array([[r[k] for k in ia] for r in rows])
            tilt = np.array([r[it] for r in rows])
            dt = np.array([r[idt] for r in rows[1:]])
            name, pol = self.phases[self.phase_i]
            crouch = float(np.degrees(((q0[3] + q0[9]) - (q0[4] + q0[10])) / 2))
            # ★2026-09-04 着座の形(座面接触の時刻 / 骨盤の後退 / 膝の前方 / ヨーと滑り)。
            #   「膝が出る」「回転する」を毎回その場で数値にする(sit_shape.py)。
            #   FKは接触時・終端など数コマだけなので軽い。失敗しても統計は残す。
            shape = {}
            try:
                if name.startswith("sit") and len(rows) >= 60 and self.obs_b is not None:
                    if self._fk_shape is None:
                        self._fk_shape = SagittalFK(self.obs_b.m, self.obs_b.d,
                                                    self.obs_b.qadr)
                    itau = [REC_COLS.index(f"tau{i}") for i in range(29)]
                    iqw = REC_COLS.index("quat_w")
                    itt = REC_COLS.index("t")
                    qq = np.array([[r[k] for k in iq] for r in rows])
                    tau = np.array([[r[k] for k in itau] for r in rows])
                    quat = np.array([r[iqw:iqw + 4] for r in rows])
                    tref = np.array([r[itt] for r in rows]).astype(int)
                    mt = shape_metrics(qq, tau, quat, tref, pol.ref_q,
                                       pol.ref["ref_quat"], self._fk_shape)
                    self.log(shape_describe(mt))
                    cm = lambda v: None if v is None else round(v * 100.0)   # noqa: E731
                    shape = {"tc": mt["tc_s"], "backe": cm(mt["back_e"]),
                             "kneex": cm(mt["kneex_e"]), "kdev": mt["kdev_e"],
                             "yawc": mt["yaw_c"], "yawe": mt["yaw_e"],
                             "slip": mt["slip_e"]}
            except Exception as e:                 # noqa: BLE001
                self.log(f"(着座の形の計算に失敗: {e})")
            self.run_stats.append({
                "t": time.strftime("%H:%M:%S"),
                "run": f"run{self.run_i:03d}", "pol": name, "how": how,
                "n": len(rows), "N": int(pol.n),
                "done": bool(len(rows) >= pol.n - 2),
                # 走る前に決まる量(次に何を変えるかの手掛かり)
                "crouch": round(crouch, 1),
                "assist": self.assist,
                "route": getattr(self, "_last_route", "?"),
                "yaw": round(float(getattr(self, "_yaw_off_deg", 0.0)) * -1, 0),
                "rate0": round(float(self.start_rate), 3),
                "tilt0": round(float(self.start_tilt), 1),
                # 走った結果
                "tmax": round(float(tilt.max()), 1),
                "tend": round(float(tilt[-1]), 1),
                "kdiff": round(float(abs(qe[3] - qe[9])), 2),
                "ahip": round(float(act[:, 0].mean()), 3),
                "asat": round(100.0 * float(np.mean(np.abs(act[:, 15:]) > 0.98)), 1),
                "dt_med": round(float(np.median(dt)), 1),
                "dt_max": round(float(dt.max()), 1),
                "legres": round(float(self.leg_res), 2),
                "cfade": self.contact_fade,
                "tcl": (None if self._contact_t is None else round(self._contact_t / CONTROL_HZ, 2)),
                **shape,
            })
            del self.run_stats[:-200]
        except Exception as e:                     # noqa: BLE001
            self.log(f"(走行統計の記録に失敗: {e})")

    def _end_phase(self):
        name, pol = self.phases[self.phase_i]
        self._phase_done = True
        self._save_rec(final=True)
        self._push_run_stat("完走" if self.t >= pol.n else f"{self.t}コマで打切")
        if self.t < pol.n:
            self.log(f"打ち切り {self.t}コマ({self.t / CONTROL_HZ:.2f}秒)。"
                     f"以降はこのコマの姿勢で保持します"
                     f"(最終コマではなく★打ち切ったコマ。2026-08-28修正)")
        gc.enable()
        # ★列を r[-2] で位置指定していたので、2026-08-26/27 に列を
        #   190→393へ増やしたとき blend_u を指すようになり、
        #   「制御周期: 中央1.0ms (1000Hz)」と嘘を出し続けていた。
        #   実測は 中央20.0ms(50Hz)。**位置ではなく名前で引くこと。**
        _i = REC_COLS.index("dt_ms")
        _dt = (np.array([r[_i] for r in self._rec_rows[1:]])
               if len(self._rec_rows) > 2 else np.zeros(1))
        _j = REC_COLS.index("ms_infer")
        _mi = (np.array([r[_j] for r in self._rec_rows[1:]])
               if len(self._rec_rows) > 2 else np.zeros(1))
        _med = float(np.median(_dt))
        _slow = 100.0 * float(np.mean(_dt > 25.0))
        self.log(f"制御周期: 中央{_med:.1f}ms 最大{_dt.max():.1f}ms "
                 f"({1000 / max(_med, 1e-6):.0f}Hz / 規定50Hz)"
                 f"  25ms超え{_slow:.0f}%  推論 中央{np.median(_mi):.1f}ms"
                 f" 最大{_mi.max():.1f}ms"
                 f"  ※正常時の実測は 中央20.0ms / 25ms超え24%")
        self.log(f"開始直後({START_TRACK_WINDOW_S}秒)の脚腰の追従誤差 最大 "
                 f"{self._early_track:.3f}rad ({self._early_track_j})"
                 f"  ※完走回の実測は最大0.949・転倒回は1.112(2026-08-26時点)")
        tp = np.asarray(getattr(self.robot, "temps", np.zeros(29)))
        self.log(f"フェーズ完走: {name}  温度 膝 L{tp[3]:.0f}/R{tp[9]:.0f}度"
                 f"  足首 L{tp[4]:.0f}/R{tp[10]:.0f}度"
                 f"  最大{tp.max():.0f}度(j{int(np.argmax(tp))})")
        # 待機中も方策を最終コマで動かし続ける(方策は終端の静止保持を
        # 学習済み。素のPD保持は数秒で釣り合いを失う。実測)
        # ★2026-08-28 のバグ。打ち切り(stop_frame)で止めたときも
        #   pol.n-1(=最終コマ)を保持していた。ar1 を245コマで切ると
        #   保持の目標が ref_q[244] → ref_q[394] へ一瞬で飛び、
        #   **腰ピッチが22.5度(0.393rad)ジャンプ**して不安定になった。
        #   保持は「実際に走り終わったコマ」で行うこと。
        self.hold_pol = pol
        self.hold_t = max(0, min(self.t, pol.n) - 1)
        if not self.is_sim:                        # 方策の走行が終わったのでLiDARを再開
            self.walk.enable_sensors(True)
        if self.phase_i + 1 < len(self.phases):
            self.phase_i += 1
            if self.step_mode:
                self.fsm = "WAIT_CONFIRM"
                self.log(f"{name} 完了(方策で姿勢維持中)。"
                         f"[NEXT]で {self.phases[self.phase_i][0]} を開始")
            else:
                self._begin_phase()
        else:
            self.fsm = "HOLD"
            self.log(f"{name} 完了。全フェーズ終了 — 方策で姿勢維持中")
            # ★2026-09-04 午後: 椅子に載らずにしゃがんだまま「完走」し、自動ダンプで前へ倒れた
            #   (11:51 run003: 骨盤が足より4cm前・終端の膝τ23N·m)。座面に載っている証拠が
            #   無ければ自動移行を保留し、方策で保持したまま操作者の目視に任せる
            self._seat_doubt = self._seat_check(name)
            if self._seat_doubt:
                self.log(f"★座面に載っていない疑い: {self._seat_doubt} — 完了後の自動移行を保留。"
                         "方策で保持中。目視で確認して [ダンプ] か [スタンドロック] を押す")
            # 完了後に標準モードへ自動で渡す(操作者が選んだときだけ)。
            # ★50Hzループの中でRPCを呼ばないこと。_spawn でワーカーに出す。
            if self.after_phase in ("seated", "sit", "damp") and not self._seat_doubt:
                nm = {"seated": "着座(FSM3)", "sit": "スクワット(FSM2)",
                      "damp": "ダンプ"}[self.after_phase]
                self.log(f"完了後の自動移行: {nm} へ渡します")
                self._spawn(f"完了後 {nm}",
                            lambda n=self.after_phase: self._do_after_phase(n))

    def _rec(self, name, q, dq, quat, gyro, tau, obs, a, target):
        """1コマを REC_COLS の並びで平坦な行にして積む(2026-08-24形式)。
        温度は摩擦の温度依存をシムの同定と突き合わせるため、
        dt_ms/ms_infer はPythonの遅れを実機で数値確認するために入れてある。"""
        now = time.perf_counter()
        prev = self._rec_prev_t
        self._rec_prev_t = now
        up_z = float(quat_to_mat(quat)[2, 2])
        tilt = float(np.degrees(np.arccos(min(1.0, max(-1.0, up_z)))))
        tp = np.asarray(getattr(self.robot, "temps", np.zeros(29)), dtype=float)
        ex = self.robot.state_full()
        nr = int(self.action_ramp_s * CONTROL_HZ)
        nb = int(self.ref_blend_s * CONTROL_HZ)
        w = 1.0 if nr <= 0 else min(1.0, (self.t + 1) / nr)
        u = 1.0 if nb <= 0 else min(1.0, (self.t + 1) / nb)
        self._rec_rows.append(np.concatenate([
            [float(self.t), float(FSM_CODE.get(self.fsm, -1)),
             float(self.phase_i), time.time() - self._phase_t0, tilt],
            q, dq, tau, quat, gyro, target, a, tp,
            [0.0 if prev is None else (now - prev) * 1000.0,
             float(getattr(self, "_ms_infer", 0.0))],
            ex["ddq"], ex["vol"], ex["temps2"], ex["mstate"],
            ex["accel"], ex["rpy"],
            [ex["imu_temp"], float(ex["tick"]), float(ex["mode_pr"]),
             float(ex["mode_machine"])],
            ex["mmode"], ex["remote"], ex["version"], [ex["crc"]],
            ex["ls_reserve"],
            [w, u, time.time()]]))
        self._rec_raw.append(np.concatenate([
            ex["msensor"].reshape(-1), ex["mreserve"].reshape(-1),
            ex["mot_ext"].reshape(-1)]).astype(np.float32))
        self._rec_obs.append(np.asarray(obs, dtype=np.float32))
        if len(self._rec_rows) % REC_SAVE_EVERY == 0:
            self._save_rec(final=False)

    def snapshot(self, full=False):
        """UIへ出す状態。full=True で29軸の生値も付ける(twin.py 用)。

        ★読むだけ。制御には一切影響しない。twin(実機ミラー)を別プロセスに
          したのは、この process が 50Hz制御 + 500Hz送信を抱えているため。
          MuJoCoの描画を同居させない。
        """
        q, dq, quat, gyro, tau = self.robot.state()
        up_z = float(quat_to_mat(quat)[2, 2])
        extra = {}
        if full:
            with self.robot.lock:
                tgt = np.asarray(self.robot.target_q).copy()
                kp = np.asarray(self.robot.kp).copy()
            # 追従誤差 = 指令したのに来ていない量。「指令は出ているのに
            # 追従していない」を live で見分けられる唯一の量
            extra = {
                "q": [round(float(x), 4) for x in q],
                "dq": [round(float(x), 3) for x in dq],
                "target": [round(float(x), 4) for x in tgt],
                "kp": [round(float(x), 1) for x in kp],
                "tau": [round(float(x), 2) for x in tau],
                "temps": [int(x) for x in
                          np.asarray(getattr(self.robot, "temps",
                                             np.zeros(29)))],
                "quat": [round(float(x), 5) for x in quat],
                "gyro": [round(float(x), 4) for x in gyro],
            }
        with self.lock:
            return {
                "fsm": self.fsm, "msg": self.msg, "armed": self.armed,
                "step_mode": self.step_mode, "sel": dict(self.sel),
                "yaw_align": bool(self.yaw_align),
                "assist": self.assist,
                "stop_frame": int(self.stop_frame),
                "after_phase": self.after_phase,
                "arm_res": float(self.arm_res),
                "arm_res_mode": self.arm_res_mode,
                "leg_res": float(self.leg_res),
                "contact_fade": self.contact_fade,
                "contact_t": (None if self._contact_t is None else round(self._contact_t / CONTROL_HZ, 2)),
                "seat_doubt": self._seat_doubt,
                "pol_scale": self._pol_scale(),
                "run_stats": self.run_stats[-40:],
                "ramp": float(self.action_ramp_s),
                "blend": float(self.ref_blend_s),
                "phases": [n for n, _ in self.phases],
                "phase_i": self.phase_i, "t": self.t, "n": self.n,
                "tilt_deg": float(np.degrees(np.arccos(min(1, max(-1, up_z))))),
                # 2026-08-27追加。IMUのヨー原点は電源を入れ直すたびに変わる。
                # 方策開始時に自動で合わせるので走行には影響しないが、
                # 「いまどれだけずれているか」は現場で見えていないと、
                # 合わせが効いているのかどうか後から切り分けられない
                "yaw_err_deg": self._yaw_err_deg(),
                "pose": self._pose_rows(q),
                "crouch": self._crouch_deg(q),
                "power": self._power_state(),
                "qd_rms": float(np.sqrt(np.mean(dq ** 2))),
                "tau_max": float(np.abs(tau).max()),
                "healthy": bool(self.robot.healthy()),
                "is_sim": self.is_sim,
                "busy": self.busy, "busy_what": self.busy_what,
                "custom": bool(getattr(self.robot, "custom_active", False)),
                "health": self.robot.health_detail(),
                "loop": self._loop_stat(),
                "load": {"knee": round(abs(float(tau[3])) + abs(float(tau[9])), 1),
                         "knee_l": round(float(tau[3]), 1),
                         "knee_r": round(float(tau[9]), 1),
                         "ankle": round(float(tau[4]) + float(tau[10]), 1),
                         "ank_l": round(float(tau[4]), 1),
                         "ank_r": round(float(tau[10]), 1),
                         # 左右の膝の曲がりの差。2026-08-26の実測で完走回は
                         # 走行中0.3〜0.4radに収まり、転倒回は0.96radまで開いた
                         "knee_dq": round(float(q[3] - q[9]), 3)},
                "go": self._go_check(),
                "ground": getattr(self, "_ground", None),
                "fsm_id": getattr(self, "_fsm_id", None),
                "walk": self.walk.status(),
                "hb_sec": float(self._heartbeat_sec),
                "ui_lost": bool(self._ui_lost),
                "ui_lost_s": (round(time.time() - self._ui_lost_t, 1) if self._ui_lost else 0.0),
                "ui_lost_last": self._ui_lost_last,
                "ui_plan": self._ui_loss_plan(),
                "sit_gate": (None if self.sit_gate is None else
                             {"ok": self.sit_gate["ok"], "items": self.sit_gate["items"],
                              "token": self.sit_gate["token"], "pattern": self.sit_gate["pattern"],
                              "age": round(time.time() - self.sit_gate["t"], 1)}),
                "user_ctrl": bool(getattr(self.robot, "_use_user_topic", False)),
                "logs": list(self.logs[-25:]),
                **extra,
            }


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>G1 Cockpit</title>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#111">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<style>
:root{--bg:#111;--card:#1c1c1c;--card2:#242424;--line:#333;--t1:#eee;--t2:#9a9a9a;
 --ok:#1baf7a;--warn:#eda100;--bad:#e34948;--acc:#3987e5}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--t1);font:14px/1.5 "Segoe UI",sans-serif;
 padding:14px;max-width:1500px;margin:auto}
h1{font-size:16px;margin-bottom:8px}
.top{display:grid;grid-template-columns:1fr 320px;gap:12px;align-items:start}
.grid{display:grid;grid-template-columns:minmax(400px,1fr) minmax(460px,1.15fr);gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:12px;color:var(--t2);margin-bottom:8px;letter-spacing:.05em}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 9px}
.tile .k{font-size:10px;color:var(--t2)}.tile .v{font-size:19px;font-weight:700}
select{max-width:330px}
select,button,input{font:inherit;border-radius:8px;border:1px solid var(--line);
 background:var(--card2);color:var(--t1);padding:7px 11px}
button{cursor:pointer}button:disabled{opacity:.32;cursor:default}
button.go{background:var(--acc);border:none;font-weight:700}
button.next{background:var(--ok);border:none;font-weight:700}
#estop{background:var(--bad);border:none;color:#fff;font-size:20px;
 font-weight:900;width:100%;padding:16px;border-radius:12px}
.bar{height:9px;background:#2c2c2c;border-radius:5px;overflow:hidden;margin:6px 0}
.bar>div{height:100%;background:var(--acc)}
.row{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:6px 0}
.sec{border-top:1px solid var(--line);margin-top:12px;padding-top:10px}
.sec:first-of-type{border-top:none;margin-top:0;padding-top:0}
.sec>.st{font-size:11px;color:var(--t2);letter-spacing:.05em;margin-bottom:5px}
.log{font:12px/1.5 Consolas,monospace;white-space:pre-wrap;color:var(--t2);
 height:420px;overflow-y:auto;background:#161616;border-radius:8px;padding:8px}
.state{font-size:24px;font-weight:900}
.lbl{font-size:11px;color:var(--t2)}
.tabs{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px}
.tabs button{padding:6px 12px;font-size:13px;background:#1a1a1a;color:var(--t2)}
.tabs button.on{background:var(--card2);color:var(--t1);border-color:var(--acc);font-weight:700}
.pane{display:none}.pane.on{display:block}
details{margin-top:8px}
details>summary{cursor:pointer;font-size:12px;color:var(--t2);
 padding:5px 0;list-style:none;user-select:none}
details>summary::before{content:"▸ ";color:var(--acc)}
details[open]>summary::before{content:"▾ "}
table.st{border-collapse:collapse;width:100%;font:12px/1.6 ui-monospace,monospace}
table.st th{position:sticky;top:0;background:#1a1a1a;color:var(--t2);
 font-size:10px;font-weight:400;text-align:right;padding:4px 6px;white-space:nowrap}
table.st th:first-child,table.st td:first-child{text-align:left}
table.st td{text-align:right;padding:3px 6px;border-top:1px solid var(--line);white-space:nowrap}
.scroll{max-height:430px;overflow:auto;border:1px solid var(--line);border-radius:8px}
/* ---- 2026-09-04 スマホ(Android)向け。歩行の十字キーと下部固定の停止バー ---- */
.num{width:74px;padding:6px 6px}
.dpad{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;max-width:360px;margin:8px 0;
 touch-action:none;user-select:none;-webkit-user-select:none}
.dpad .dk{padding:14px 4px;font-size:15px;line-height:1.25;touch-action:none;
 user-select:none;-webkit-user-select:none;-webkit-touch-callout:none}
.dpad .dk.on{background:var(--acc)}
#dk_stop{background:#7a1b1b;font-size:22px;font-weight:900}
.fixbar{display:none;position:fixed;left:0;right:0;bottom:0;gap:8px;padding:8px 10px;
 background:rgba(17,17,17,.97);border-top:1px solid var(--line);z-index:30}
.fixbar button{flex:1;padding:16px 8px;font-size:18px;font-weight:900;border:none;border-radius:12px;color:#fff}
@media (max-width:760px){
 body{padding:8px;padding-bottom:100px;font-size:15px;overflow-x:hidden}
 .top,.grid{grid-template-columns:1fr}
 /* グリッドの子は min-width:auto だと中身(長い option や表)の幅まで広がって
    横スクロールになる。0 にして枠に収め、長い語は折り返す */
 .top>div,.grid>.card,.card,.sec,.row{min-width:0;max-width:100%}
 .lbl,.card{overflow-wrap:anywhere}
 .tiles{grid-template-columns:repeat(2,1fr)}
 select{max-width:100%;width:100%}
 .row>select{flex:1 1 160px}
 button{padding:12px 14px;font-size:15px}
 .fixbar{display:flex}
 #estop,#estop_lbl{display:none}
 .log{height:240px}
 .scroll{max-height:300px}
}
</style></head><body>

<h1>&#129302; G1 Cockpit <span id="mode" class="lbl"></span></h1>

<!-- ===== 常時表示: E-STOP と 判定 と 主要数値 ===== -->
<div class="top">
 <div>
  <div class="tiles">
   <div class="tile"><div class="k">FSM</div><div class="v state" id="fsm">-</div></div>
   <div class="tile"><div class="k">制御権</div><div class="v" id="own">-</div></div>
   <div class="tile"><div class="k">傾き</div><div class="v" id="tilt">-</div></div>
   <div class="tile"><div class="k">制御ループ</div><div class="v" id="loop">-</div></div>
  </div>
  <div id="go" style="margin-top:10px;padding:8px 12px;border-radius:8px;
   border:1px solid var(--line);background:#1c1c1c;font-size:13px">-</div>
  <div id="crouch" style="margin-top:10px;padding:9px 12px;border-radius:10px;
   border:1px solid var(--line)"></div>
 </div>
 <div>
  <button id="estop" onclick="cmd('estop')">&#9632; E-STOP</button>
  <div class="lbl" id="estop_lbl" style="margin-top:6px;text-align:center">スペースキーでも止まります</div>
  <div class="card" style="margin-top:10px;padding:9px 11px">
   <div class="lbl" id="pwr">-</div>
  </div>
 </div>
</div>

<div class="grid">

<!-- ================= 操作部 ================= -->
<div class="card">
 <h2>&#9654; 操作</h2>

 <div class="sec"><div class="st">1. ロボットのモード (Unitree標準制御)</div>
  <div class="row">
   <button onclick="cmd('mode_zero')">ゼロトルク</button>
   <button onclick="cmd('mode_damp')">ダンプ</button>
   <button onclick="cmd('mode_stand')">立つ</button>
   <button onclick="cmd('mode_walk')">ウォーキング</button>
   <button onclick="cmd('mode_sit')">座る(スクワット)</button>
   <button onclick="cmd('mode_seated')">着座</button>
   <button onclick="cmd('damp')">damp(方策側)</button>
  </div>
  <div class="lbl">★「ウォーキング」は走行制御(FSM802)。押すと歩き出しうる。
  歩行状態から方策へ引き継ぐ場合はこれで走行にしてから「制御権を取る」→方策を実行する。<br>
  ★「座る(スクワット)」はUnitree標準のFSM2。立位のまましゃがみ込む。押すと実機が下方へ動く。<br>
  ★「着座」はFSM3。方策で座り終えた姿勢を内蔵制御へ渡す先で、完了後の自動移行の既定でもある。</div>
 </div>

 <div class="sec"><div class="st">2. 制御権を取る</div>
  <div class="row">
   <button class="go" style="background:#8a5a1b" onclick="cmd('stand_user')">&#9889; スタンド&amp;カスタム(開発者モード)</button>
   <button onclick="cmd('custom')">カスタム制御へ(従来)</button>
  </div>
  <div class="lbl">どちらも取った後は<b>PD保持だけでバランスは無い</b>。
  速やかに方策を始めるか[ダンプ]へ。★802を通るのでリモコンE-STOPを握ること</div>
 </div>

 <div class="sec"><div class="st">3. 走る前に記録する</div>
  <div class="row">補助
   <select id="sel_assist" onchange="cmd('assist',this.value)">
    <option value="?">&#9733;未記入</option>
    <option value="none">補助なし(自立できていた)</option>
    <option value="light">軽く触れていた</option>
    <option value="hold">しっかり支えた</option></select></div>
  <div class="lbl">★毎回選ぶこと。事前指標で結果を説明できたのは
  「しゃがみ深さ」だけで、支えたかどうかは記録が無い</div>
  <div class="row">メモ
   <input id="memo" placeholder="椅子の距離/座面高/足の位置 など" style="flex:1;min-width:200px"
    onkeydown="if(event.key==='Enter'){cmd('memo',this.value);this.value=''}">
   <button onclick="const e=document.getElementById('memo');cmd('memo',e.value);e.value=''">記録</button>
  </div>
 </div>

 <div class="sec"><div class="st">4. 方策を実行</div>
  <div class="row">着座 <select id="sel_sit" onchange="sel('sit')"></select></div>
  <div class="row">
   <button class="go" id="user_sit" style="background:#1b7a4a" onclick="askUserRun('sit')">&#9889; 立位&rarr;UserCtrl&rarr;そのまま座る</button>
   <button class="go" id="run_sit" onclick="askRun('sit')">&#9654; 座る(制御権を取ってから)</button>
  </div>
  <div class="row">
   <button onclick="cmd('goto_start','sit')">&#10515; 開始姿勢へ</button>
   <button onclick="cmd('ground_check')">&#9878; 接地チェック</button>
   <button class="next" id="next" onclick="cmd('next')">&#9654; NEXT</button>
  </div>
  <div class="row">腕の残差
   <select id="sel_armres" onchange="cmd('armres',this.value)">
    <option value="1.0">1.00 — 既定。方策の指定どおり</option>
    <option value="0.9">0.90</option>
    <option value="0.8">0.80</option>
    <option value="0.7">0.70</option>
    <option value="0.6">0.60</option>
    <option value="0.5">0.50 — 半分</option>
    <option value="0.4">0.40</option>
    <option value="0.3">0.30 — 旧ar1での手のクロス対策値</option>
    <option value="0.2">0.20</option>
    <option value="0.1">0.10</option>
    <option value="0.0">0.00 — 参照の腕軌道そのまま(方策の補正なし)</option>
   </select>
   <select id="sel_armmode" onchange="cmd('armmode',this.value)">
    <option value="lat">★横方向のみ(肩ロール/ヨー・手首)</option>
    <option value="all">腕14関節すべて</option>
   </select></div>
  <div class="lbl" id="polscale" style="margin-bottom:4px"></div>
  <div class="lbl">方策の出力のうち<b>腕だけ</b>を<b>さらに</b>縮める。脚腰は触らない。
   ★2026-09-02以降の方策は参照npzが関節別スケールを持つので、既定1.00でよい。<br>
   <b>実機12本の用量反応:</b> 残差1.0→手首間隔-0.111m(クロス)/終端傾き29.9度、
   0.5→0.107m/8.2度、0.3→0.135m/6.6度。参照は0.184m/18.0度。
   <b>クロスは直るが着座が浅くなる</b>(腕は釣り合いの錘でもあるため)。<br>
   ★<b>横方向のみ</b>なら、手の左右位置を決める肩ロール/ヨーだけを縮め、
   前後の錘である肩ピッチと肘は残せる。
   2026-08-28の実機8本で手が5本クロスした(62%)。原因は参照の手首間隔の余裕
   (2〜10cm)より実機の腕の追従誤差(5〜10cm)が大きいこと。
   ar4の参照は手首間隔0.184m一定・左右対称なので、腕の残差を縮めれば
   「前へならえ」に寄る。<br>
   ★観測に入る値は縮めない(学習時と同じ意味を保つ)ので、方策の判断自体は変わらない。
   ★0.00でも腕は動く(参照の軌道どおりに動く)</div>
  <div class="row">完了後
   <select id="sel_after" onchange="cmd('afterphase',this.value)">
    <option value="seated">★着座(FSM3)へ渡す — 既定</option>
    <option value="sit">スクワット(FSM2)へ渡す</option>
    <option value="damp">ダンプへ渡す</option>
    <option value="none">何もしない — 方策のPDで姿勢維持(従来)</option>
   </select></div>
  <div class="lbl">方策を走り終えたあと、内蔵の標準モードへ自動で渡す。
  0.5秒待ってから渡す(方策の最終姿勢が落ち着く前に渡すと動き出しが大きい)。<br>
  <b>★着座(FSM3)が座り終えた姿勢の引き継ぎ先。スクワット(FSM2)は立位で
  しゃがむモードなので、座った状態から立ち上がろうとしうる。</b><br>
  「何もしない」だと方策のPDが最終姿勢を保持し続ける(モータが発熱するので
  終わったら手動でダンプする)。<b>最初の数回はリモコンのE-STOPを握って見ていること。</b></div>
  <div class="row">打ち切り
   <select id="sel_stop" onchange="cmd('stopframe',this.value)">
    <option value="0">最後まで(方策ごとの全長)</option>
    <option value="285">★285コマ(5.70秒) — ln20d / ln20x 用(全433コマ)</option>
    <option value="247">★247コマ(4.94秒) — ln21 / ln20 / ln20s / ln20a / ar1 用(全395コマ)</option>
    <option value="146">146コマ(2.92秒) — dp4用。背もたれへ反る前で止める</option>
    <option value="115">115コマ(2.30秒) — 座り終わった直後</option>
   </select></div>
  <div class="lbl">参照npzは書き換えず<b>走行を途中で止めるだけ</b>。
  観測には正規化時刻と先読みが入るので、配列を切ると別物になる。<br>
  <b>★ln20系5種と ln21 はすべて静止区間0%で、脚腰が止まった後も腕だけが最後まで動き続ける</b>
  (ar1と同じ)。座った直後に手足が動くのはこれが原因。
  ln21も配布元が「腕の終端静止は未対応。次版ln22で対応予定」と明記している。<br>
  <table style="font-size:10px;border-collapse:collapse;margin:5px 0">
  <tr><td style="padding:1px 8px 1px 0"><b>ln21</b> / ln20 / ln20s / ln20a</td><td style="padding:1px 8px">全395コマ</td>
      <td style="padding:1px 8px">脚腰は245コマ(4.90秒)で停止</td><td><b>→ 247で打ち切り</b></td></tr>
  <tr><td style="padding:1px 8px 1px 0">ln20d / ln20x</td><td style="padding:1px 8px">全433コマ</td>
      <td style="padding:1px 8px">脚腰は283コマ(5.66秒)で停止</td><td><b>→ 285で打ち切り</b></td></tr>
  </table>
  体幹の傾きは打ち切り点の手前(241 / 279コマ)で終端値-18度に到達済み。
  <b>脚腰の軌道は全部走り切ったうえで、その先の腕の動きだけが消える。</b><br>
  ar4は282コマ以降が全関節静止なので打ち切り不要。
  <b>ln23は終端静止を継承しているので打ち切り不要(「最後まで」を選ぶ)。</b><br>
  dp4: 146〜242コマが「背もたれへ18度反る」区間(2026-08-27解析)</div>
 </div>

 <div class="sec"><div class="st">5. 歩行(内蔵制御) — スマホ手動操作 / 自動歩行</div>
  <div class="row">
   <button onclick="cmd('walk_ready')">&#128694; 歩行モードへ(802)</button>
   <button class="go" id="walk_auto" onclick="startAuto()">&#9654; 自動歩行 開始</button>
   <button id="walk_stop" style="background:#7a1b1b;border:none;font-weight:700" onclick="cmd('walk_stop')">&#9632; 歩行停止(速度ゼロ)</button>
  </div>
  <div class="lbl">★[歩行モードへ]はロック立位(4)から走行(802)へ遷移する。走行制御が動くので
  <b>機体を接地させ、リモコンのE-STOPを握って</b>押すこと(吊ったままだと空中で暴れる)。
  [歩行停止]は速度をゼロにするだけで内蔵バランスは生きている。E-STOPはdampなので<b>歩行中に押すと倒れる</b>。</div>
  <div class="row">停止距離 <input id="w_stop" type="number" step="0.05" min="0.3" max="2.5" value="0.60" class="num">m
   &nbsp;横移動 <select id="w_dir"><option value="left">左へ</option><option value="right">右へ</option></select>
   <input id="w_side" type="number" step="0.1" min="0.1" max="3" value="0.5" class="num">m
   &nbsp;速度 <input id="w_v" type="number" step="0.05" min="0.05" max="0.9" value="0.35" class="num">m/s
   &nbsp;最大前進 <input id="w_max" type="number" step="0.5" min="0.3" max="10" value="4" class="num">m
   &nbsp;<label><input id="w_dry" type="checkbox"> ドライラン(速度を送らない)</label>
   <button onclick="sendWalkParams()">適用</button>
   <button onclick="cmd('lidar','on')">LiDAR読取 開始</button>
   <button onclick="cmd('lidar','off')">停止</button></div>
  <div class="lbl">自動歩行 = 前進 &rarr; LiDARで前方コリドー(幅0.7m)に障害物を見たら停止距離の手前で停止 &rarr;
  指定した向きへ横移動 &rarr; 停止。障害物が無ければ最大前進距離で止まる(横移動はしない)。
  ★頭のLiDARは膝より低い物が見えにくい。まず<b>ドライラン</b>で「前方」の距離が実物と合うことを確かめてから。</div>
  <div id="walkst" class="lbl" style="line-height:1.8">-</div>
  <div class="dpad" id="dpad" oncontextmenu="return false">
   <div></div><button class="dk" data-v="1,0,0">&#9650;<br>前</button><div></div>
   <button class="dk" data-v="0,1,0">&#9664;<br>左</button><button class="dk" id="dk_stop" onclick="cmd('walk_stop')">&#9632;</button><button class="dk" data-v="0,-1,0">&#9654;<br>右</button>
   <button class="dk" data-v="0,0,1">&#8630;<br>左旋回</button><button class="dk" data-v="-1,0,0">&#9660;<br>後</button><button class="dk" data-v="0,0,-1">&#8631;<br>右旋回</button>
  </div>
  <div class="lbl">十字キーは<b>押している間だけ</b>動く(離すと0.5秒以内に停止。通信が切れても止まる)。
  &nbsp;画面消灯防止: <span id="ka_st">-</span>
  <video id="ka" src="/keepawake.webm" loop muted playsinline style="width:80px;height:60px;vertical-align:middle;opacity:.5;border-radius:6px"></video></div>
 </div>

 <details><summary>詳細設定 (参照ブレンド / ランプ / ヨー合わせ / 進行)</summary>
  <div class="row">参照ブレンド
   <select id="sel_blend" onchange="cmd('blend',this.value)">
    <option value="0">0秒 — 従来(開始姿勢を合わせる必要あり)</option>
    <option value="0.5" selected>0.5秒 — 既定。どこから始めても跳びなし</option>
    <option value="1.0">1.0秒 — ★simでrcが転倒。使わない</option>
   </select></div>
  <div class="row">方策の立ち上げ
   <select id="sel_ramp" onchange="cmd('ramp',this.value)">
    <option value="0">0秒 — 従来(跳び 0.46〜0.72rad)</option>
    <option value="0.2">0.2秒 — 跳び1/10</option>
    <option value="0.3" selected>0.3秒 — 既定(跳び1/15)</option>
    <option value="0.5">0.5秒 — シムでrcの傾きが最小</option>
    <option value="0.8">0.8秒 — ★シムでrcが転倒。使わない</option>
   </select></div>
  <div class="row">接触後の膝・足首の残差
   <select id="sel_cfade" onchange="cmd('cfade',this.value)">
    <option value="0">2秒で0へ抜く — ★実機 9/4 11時台に 3/3 崩れた。試験用</option>
    <option value="0.3">2秒で0.3まで — 控えめ(シム 30cm・傾き最大14度)</option>
    <option value="off" selected>★しない — 既定。実機で座れている設定(9/3・9/4 10時台)</option>
   </select></div>
  <div class="lbl">座面に乗った(両膝トルクが降下中ピークの40%を切った)あと、方策の膝・足首の残差を
   時間で0へ抜き、参照の深い着座姿勢へ寄せる。方策は座面に乗った後も膝を曲げ足首を引く残差を
   出し続け、足首〜骨盤が 30→20cm に縮んで浅く座っていた(実機11本+シムで確認)。
   旧コックピットの「減衰retarget」を接触後に当てた形。<b>★実機未検証。最初はハーネスで。</b></div>
  <div class="row">脚腰の残差(試験用)
   <select id="sel_legres" onchange="cmd('legres',this.value)">
    <option value="1.0" selected>1.00 — 既定。方策どおり(触らない)</option>
    <option value="0.85">0.85 — ★A/B候補。ハーネス必須</option>
    <option value="0.7">0.70 — シムで後退17→22cm・膝前16→9cm。実機開始状態では要注意</option>
    <option value="0.6">0.60 — ★シムで転倒あり。使わない</option>
   </select></div>
  <div class="lbl">方策の脚腰(0〜14)の残差を縮める。<b>バランスの権限を縮めることになる</b>ので既定では触らない。
   膝が前へ出る/回転する問題の切り分け用(docs/着座_膝と回転の解析_20260904.md)。</div>
  <div class="row">ヨー合わせ
   <select id="sel_yaw" onchange="cmd('yawalign',this.value)">
    <option value="on">ON(既定)</option>
    <option value="off">OFF(14:13以前と同じ)</option></select>
   <span class="lbl">膝の左右差が 1.23&rarr;0.11 に直った分</span></div>
  <div class="row">進行
   <select id="sel_mode" onchange="cmd('mode',this.value)">
    <option value="step">ステップ(各フェーズ前に確認)</option>
    <option value="auto">自動</option></select></div>
  <div class="row"><button onclick="cmd('hbdrop','20')">通信途絶を模擬(20秒)</button>
   <span class="lbl">シムで「途絶時の動作」(動作中は最後まで/立位はバランスへ返す)を確かめる用</span></div>
 </details>

 <details><summary>登り / 旋回 / 通しシーケンス</summary>
  <div class="row">登り <select id="sel_climb" onchange="sel('climb')"></select>
   <button class="go" id="run_climb" onclick="askRun('climb')">&#9654; 登壇</button></div>
  <div class="row">旋回 <select id="sel_turn" onchange="sel('turn')"></select>
   <button class="go" id="run_turn" onclick="askRun('turn')">&#9654; 旋回</button></div>
  <div class="row">
   <button onclick="cmd('arm')">1. ARM(方策読込)</button>
   <button id="place" onclick="cmd('place_sim')" style="display:none">1.5 (sim)配置</button>
   <button class="go" onclick="cmd('start')">2. START</button>
  </div>
 </details>

 <details><summary>セッション / 保守</summary>
  <div class="row">
   <button onclick="cmd('newsession')">新セッション</button>
   <button onclick="cmd('fsm_read')">FSM読取</button>
   <button onclick="cmd('mode_walk')">ウォーキング</button>
  </div>
  <div class="lbl">★ウォーキングは走行制御。押すと歩き出しうる</div>
 </details>
</div>

<!-- ================= 出力部 ================= -->
<div class="card">
 <h2>&#128202; 出力</h2>
 <div class="tabs" id="tabs">
  <button class="on" onclick="tab('t_run')">走行の統計</button>
  <button onclick="tab('t_pose')">立位の姿勢</button>
  <button onclick="tab('t_log')">イベントログ</button>
  <button onclick="tab('t_load')">荷重 / ガード</button>
 </div>

 <div id="t_run" class="pane on">
  <div id="phases" class="lbl">-</div>
  <div class="bar"><div id="prog" style="width:0%"></div></div>
  <div class="lbl" id="tn">-</div>
  <div style="margin:6px 0 10px;font-size:14px" id="msg">-</div>
  <div class="scroll"><table class="st" id="stats"></table></div>
  <div class="lbl" id="statsum" style="margin-top:8px"></div>
 </div>

 <div id="t_pose" class="pane">
  <table id="pose" style="border-collapse:collapse;font:14px/1.7 ui-monospace,monospace;width:100%"></table>
  <div class="lbl" id="posesum" style="margin-top:10px"></div>
  <div class="lbl" style="margin-top:8px">かっこ内は<b>着座に成功した13本</b>の実測レンジ(10〜90%)。
  個々の関節と結果の相関は弱い(|r|最大0.32)。合否ではなく
  「うまくいった回と同じ立ち方か」を見るための窓</div>
 </div>

 <div id="t_log" class="pane"><div class="log" id="log"></div></div>

 <div id="t_load" class="pane">
  <div class="lbl" id="load" style="line-height:2"></div>
  <div class="lbl" id="guard" style="margin-top:10px;line-height:2"></div>
 </div>
</div>
</div>

<!-- スマホでは画面下に固定(親指の届く位置)。PCでは非表示 -->
<div class="fixbar">
 <button style="background:var(--bad)" onclick="cmd('estop')">&#9632; E-STOP</button>
 <button style="background:#7a1b1b" onclick="cmd('walk_stop')">&#9632; 歩行停止</button>
</div>

<script>
let S={sel:{}};
function cmd(c,a){fetch('/cmd?c='+c+(a?('&a='+encodeURIComponent(a)):''),{method:'POST'}).catch(function(){})}
function sel(k){cmd('select',k+':'+document.getElementById('sel_'+k).value)}
function tab(id){
 document.querySelectorAll('.pane').forEach(e=>e.classList.remove('on'));
 document.getElementById(id).classList.add('on');
 const bs=document.querySelectorAll('#tabs button');
 bs.forEach(b=>b.classList.remove('on'));
 const ids=['t_run','t_pose','t_log','t_load'];
 if(bs[ids.indexOf(id)])bs[ids.indexOf(id)].classList.add('on');
}
// 2026-08-27 操作者の指示で確認ダイアログを全廃した。
// ★NG(実行不可)は消していない。ここは確認ではなくブロック。
const NL=String.fromCharCode(10);
function askUserRun(k){
 const g=S.go||{};
 if(g.ng&&g.ng.length){alert('実行できません:'+NL+'・'+g.ng.join(NL+'・'));return}
 cmd('user_run',k);
}
function askRun(k){
 const g=S.go||{};
 if(g.ng&&g.ng.length){alert('実行できません:'+NL+'・'+g.ng.join(NL+'・'));return}
 cmd('run_task',k);
}
const POSECOL=['var(--ok)','var(--warn)','var(--bad)'];
function poseCell(v,st,lo,hi){
 if(v===null||v===undefined)return '<td></td><td></td>';
 const D=180/Math.PI;
 const rng=(lo===null||lo===undefined)?'':'('+(lo*D).toFixed(0)+'〜'+(hi*D).toFixed(0)+')';
 return '<td style="text-align:right;padding:2px 8px;color:'+POSECOL[st]+
   ';font-weight:'+(st?'700':'400')+'">'+(v*D).toFixed(1)+'&deg;</td>'+
   '<td style="text-align:right;padding:2px 10px 2px 0;color:var(--t2);font-size:11px">'+rng+'</td>';
}
function drawPose(rows,d){
 const e=document.getElementById('pose'); if(!e||!rows)return;
 let h='<tr style="color:var(--t2);font-size:11px">'+
   '<th style="text-align:left;padding:3px 14px 3px 0">関節</th>'+
   '<th colspan="2" style="padding:3px 8px;text-align:right">左</th>'+
   '<th colspan="2" style="padding:3px 8px;text-align:right">右</th></tr>';
 let inr=0,tot=0,worst=0;
 for(const r of rows){
  for(const st of [r[2],r[6]]){
   if(st===null||st===undefined)continue;
   tot++; if(st===0)inr++; if(st>worst)worst=st;
  }
  h+='<tr style="border-top:1px solid var(--line)">'+
     '<td style="padding:3px 14px 3px 0;color:var(--t2)">'+r[0]+'</td>'+
     poseCell(r[1],r[2],r[3],r[4])+poseCell(r[5],r[6],r[7],r[8])+'</tr>';
 }
 e.innerHTML=h;
 const su=document.getElementById('posesum');
 if(su){
  let x='成功13本のレンジ内 <b style="color:'+POSECOL[worst]+'">'+inr+'/'+tot+'</b>';
  if(d)x+='　傾き <b>'+d.tilt_deg.toFixed(1)+'&deg;</b>'+
     '　揺れRMS <b>'+d.qd_rms.toFixed(3)+'</b>'+
     '　ヨーずれ <b>'+(d.yaw_err_deg==null?'-':(d.yaw_err_deg>0?'+':'')+d.yaw_err_deg.toFixed(0)+'&deg;')+'</b>';
  su.innerHTML=x;
 }
}
function drawCrouch(d){
 const cr=document.getElementById('crouch'); if(!cr)return;
 if(d.crouch==null){cr.innerHTML='<span class="lbl">しゃがみ深さ -</span>';return}
 const v=d.crouch, deep=(v>=65), col=deep?'var(--bad)':'var(--ok)';
 cr.style.background=deep?'rgba(235,80,80,.10)':'rgba(90,220,120,.08)';
 cr.innerHTML='<div style="font-size:11px;color:var(--t2)">しゃがみ深さ (膝角-足首角) &mdash; 走る前に見る一番大事な数字</div>'+
  '<div style="font-size:28px;font-weight:700;color:'+col+';line-height:1.25">'+
  v.toFixed(0)+'&deg;<span style="font-size:13px;font-weight:400;margin-left:10px">'+
  (deep?'&#9733;深い &mdash; この立ち方からの成功は 4/21 (19%)':'浅い &mdash; この立ち方なら 14/23 (61%)')+'</span></div>'+
  '<div class="lbl">しきい値 65&deg;。実機44本 / Fisher p=0.0065。'+
  (deep?'<b style="color:var(--bad)">支えを完全に離して、自分で立ち切らせてから押すこと</b>':'')+'</div>';
}
// 走行1本ごとの統計。npzを開かずコックピットが走行中に計算した値。
const SCOL=[['t','時刻'],['run','run'],['how','結果'],['n','コマ'],['crouch','しゃがみ'],
 ['assist','補助'],['route','経路'],['rate0','開始揺れ'],['tmax','最大傾'],['tend','終端傾'],
 ['kdiff','膝左右差'],['ahip','左股a'],['asat','腕飽和%'],['dt_med','周期ms'],['dt_max','周期最大'],
 ['tc','接触s'],['backe','後退cm'],['kneex','膝前cm'],['kdev','膝角Δ'],['yawc','ヨー接触'],['yawe','ヨー終端'],['slip','滑り°'],['legres','脚残差'],['cfade','接触後'],['tcl','検知s']];
const AL={'?':'未記入','none':'なし','light':'軽く','hold':'支えた'};
function drawStats(rs){
 const e=document.getElementById('stats'); if(!e)return;
 if(!rs||!rs.length){e.innerHTML='<tr><td class="lbl">まだ走行がありません</td></tr>';return}
 let h='<tr>'+SCOL.map(c=>'<th>'+c[1]+'</th>').join('')+'</tr>';
 for(let i=rs.length-1;i>=0;i--){
  const r=rs[i];
  const seated=(r.tend>=15&&r.tend<=24);
  h+='<tr>'+SCOL.map(c=>{
   let v=r[c[0]], st='';
   if(c[0]==='assist')v=AL[v]||v;
   if(c[0]==='n')v=r.n+'/'+r.N;
   if(c[0]==='how')st='color:'+(r.done?'var(--ok)':'var(--bad)');
   if(c[0]==='crouch')st='color:'+(v>=65?'var(--bad)':'var(--ok)');
   if(c[0]==='tend')st='font-weight:700;color:'+(seated?'var(--ok)':'var(--warn)');
   if(c[0]==='ahip')st='color:'+(Math.abs(v)>0.4?'var(--bad)':'var(--ok)');
   if(c[0]==='kdiff')st='color:'+(v>0.4?'var(--warn)':'var(--t1)');
   if(c[0]==='dt_med')st='color:'+(v>25?'var(--bad)':'var(--t1)');
   if(c[0]==='tc')st='color:'+(v!=null&&v<1.5?'var(--warn)':'var(--ok)');
   if(c[0]==='backe')st='color:'+(v!=null&&v<25?'var(--warn)':'var(--ok)');
   if(c[0]==='kneex')st='color:'+(v!=null&&v>5?'var(--warn)':'var(--ok)');
   if(c[0]==='slip'||c[0]==='yawe')st='color:'+(v!=null&&Math.abs(v)>5?'var(--warn)':'var(--t1)');
   return '<td style="'+st+'">'+(v==null?'-':v)+'</td>';
  }).join('')+'</tr>';
 }
 e.innerHTML=h;
 const n=rs.length, ok=rs.filter(r=>r.tend>=15&&r.tend<=24).length;
 document.getElementById('statsum').innerHTML=
  'この起動から '+n+'本 / 座れた(終端傾き15〜24&deg;) <b style="color:var(--ok)">'+ok+'本</b>'+
  '　参照は終端18.0&deg; / シム 最大22.1&deg;・終端18.5&deg;'+
  '<br>左股a: 良+0.05〜+0.17 / 浅い回+0.78〜+0.81　膝左右差: シム0.11　周期: 正常20.0ms'+
  '<br>着座の形(2026-09-04): 接触は参照1.84s(実機0.9〜1.2s) / 後退は参照31cm(実機13〜28) / 膝前は参照0cm(実機+6〜22) / ヨー・滑りは0が理想。'+
  '実機で早く座るほど後退が足りず膝が前に出る(docs/着座_膝と回転の解析)';
}
function fill(id,arr,cur,skip,notes){const e=document.getElementById(id);
 if(!e||e.dataset.done)return; e.dataset.done=1;
 const items=skip?['(skip)',...arr]:arr;
 e.innerHTML=items.map(x=>{const n=(notes||{})[x];
  return `<option value="${x}" ${x===cur?'selected':''}>${x}${n?'  —  '+n:''}</option>`
 }).join('')}
let META=null;
async function loadMeta(){
 try{const m=await(await fetch('/state?meta=1')).json();
     META={patterns:m.patterns,notes:m.notes,warn:m.warn};}catch(e){}
}
async function tick(){
 if(!META){await loadMeta();if(!META)return}
 let d;try{d=await(await fetch('/state')).json()}catch(e){return}
 d.patterns=META.patterns;d.notes=META.notes;d.warn=META.warn;
 S=d;
 document.getElementById('mode').textContent=d.is_sim?'[SIMモック]':'[実機]';
 const f=document.getElementById('fsm');f.textContent=d.fsm;
 f.style.color=d.fsm==='DAMP'?'var(--bad)':(d.fsm==='RUNNING'?'var(--ok)':'var(--t1)');
 const te=document.getElementById('tilt');
 te.textContent=d.tilt_deg.toFixed(0)+'°';
 te.style.color=d.tilt_deg>25?'var(--warn)':'var(--t1)';
 const h=d.health||{};
 const ow=document.getElementById('own');
 ow.textContent=h.estop_latched?'E-STOP':(d.user_ctrl?'UserCtrl':(d.custom?'方策':'標準'));
 ow.style.color=h.estop_latched?'var(--bad)':(d.custom?'var(--warn)':'var(--t1)');
 const lp=d.loop||{}, le=document.getElementById('loop');
 le.textContent=(lp.hz?lp.hz.toFixed(0):'-')+'Hz';
 le.style.color=lp.ok?'var(--ok)':'var(--bad)';
 le.title='規定50Hz。40msを超えると方策の実行を自動中止します';
 document.getElementById('sel_yaw').value=d.yaw_align?'on':'off';
 document.getElementById('sel_assist').value=d.assist||'?';
 document.getElementById('sel_stop').value=String(d.stop_frame||0);
 document.getElementById('sel_after').value=d.after_phase||'none';
 document.getElementById('sel_armres').value=(d.arm_res===undefined?1:d.arm_res).toFixed(1);
 const lr=document.getElementById('sel_legres'); if(lr){const v=(d.leg_res===undefined?1:d.leg_res); lr.value=(v>=0.999?'1.0':(Math.abs(v-0.85)<0.01?'0.85':(Math.abs(v-0.7)<0.01?'0.7':'0.6')));}
 const cf=document.getElementById('sel_cfade'); if(cf&&d.contact_fade)cf.value=d.contact_fade;
 document.getElementById('sel_armmode').value=d.arm_res_mode||'lat';
 const ps=document.getElementById('polscale');
 if(ps){const p=d.pol_scale;
  ps.innerHTML=!p?'':('方策が指定する残差スケール: 脚腰 <b>'+p.leg.toFixed(2)+
   '</b> / 腕 <b style="color:'+(p.v?'var(--ok)':'var(--warn)')+'">'+p.arm.toFixed(2)+'</b>'+
   (p.v?'（参照npzに同梱）':'<b style="color:var(--warn)">（関節別指定なし＝旧方策）</b>'));}
 drawPose(d.pose,d); drawCrouch(d); drawStats(d.run_stats);
 const pw=document.getElementById('pwr'), p=d.power||{}, ac=(p.ac===0);
 pw.innerHTML='電源 <b style="color:'+(ac?'var(--bad)':'var(--ok)')+'">'+
  (ac?'★AC未接続':'AC接続')+'</b>'+
  (p.mhz?('　CPU最大 <b style="color:'+(p.mhz<2500?'var(--warn)':'var(--ok)')+'">'+
    p.mhz.toFixed(0)+'MHz</b>'+(p.mhz_med?('<span style="color:var(--t2)"> (中央'+
    p.mhz_med.toFixed(0)+'MHz — 暇なコアは800まで落ちるので正常)</span>'):'')):'')+
  (p.gov?('　'+p.gov):'')+
  '<br>受信/送信 <b style="color:'+(d.healthy?'var(--ok)':'var(--bad)')+'">'+
  ((h.state_age_ms<200?'受信OK':'受信途絶')+' / '+(h.send_hz_ok?'送信OK':'送信停止'))+'</b>'+
  '　関節速度RMS '+d.qd_rms.toFixed(2)+
  (ac?'<br><b style="color:var(--bad)">バッテリー駆動では制御ループが19〜22Hzに落ちて必ず前に倒れます</b>':'');
 document.getElementById('guard').innerHTML=
  `ループ ${lp.ms||0}ms(最大${lp.max||0}) / 送信 ${h.send_n||0}パケット`
  +`<br>ガード 可動域${h.guard_clip||0} 変化量${h.guard_rate||0} NaN${h.guard_nan||0}`
  +(lp.ok?'':'<br><b style="color:var(--bad)">★制御ループが遅い — 他の重い処理を止めること</b>')
  +(d.busy?`<br>●処理中: ${d.busy_what}`:'')
  +(h.estop_latched?'<br><b style="color:var(--bad)">★E-STOPラッチ中 — [1. ARM]か[▶ 実行]で解除されます</b>':'');
 const ld=d.load||{}, gr=d.ground;
 const kOK=(ld.knee>=20&&ld.knee<=40), aOK=(ld.ankle>=0&&ld.ankle<=8);
 const dOK=Math.abs(ld.knee_dq||0)<=0.45;
 document.getElementById('load').innerHTML=
  `膝τ <b style="color:${kOK?'var(--ok)':'var(--warn)'}">左 ${ld.knee_l||0} / 右 ${ld.knee_r||0} N·m</b>`
  +` (合計 ${ld.knee||0}。体重が足に乗っていれば20〜40。完走時27.0/30.9)`
  +`<br>足首τ <b style="color:${aOK?'var(--ok)':'var(--warn)'}">左 ${ld.ank_l||0} / 右 ${ld.ank_r||0}</b>`
  +` (合計 ${ld.ankle||0}。完走時+2.4〜+5.9)`
  +`<br>膝の左右差 <b style="color:${dOK?'var(--ok)':'var(--bad)'}">${(ld.knee_dq||0).toFixed(2)} rad</b>`
  +` (走行中: 完走0.3〜0.4 / 転倒0.96)`
  +(gr?`<br>接地チェック 左${gr.l_ratio} 右${gr.r_ratio} `
      +`<b style="color:${(gr.l_ratio<0.3&&gr.r_ratio<0.3)?'var(--ok)':((gr.l_ratio>0.6||gr.r_ratio>0.6)?'var(--bad)':'var(--warn)')}">`
      +`${(gr.l_ratio<0.3&&gr.r_ratio<0.3)?'両足 接地OK':((gr.l_ratio>0.6||gr.r_ratio>0.6)?'★浮いている疑い':'△微妙')}</b>`
      +` (追従率 &lt;0.30=接地 / &gt;0.60=浮き)`:'');
 const g=d.go||{ok:true,ng:[],warn:[]}, ge=document.getElementById('go');
 const dis=(g.ng&&g.ng.length>0);
 ['run_sit','run_climb','run_turn','user_sit'].forEach(id=>{
   const b=document.getElementById(id); if(b)b.disabled=dis;});
 if(dis){
   ge.style.borderColor='var(--bad)'; ge.style.background='#2a1414';
   ge.innerHTML='<b style="color:var(--bad)">■ 実行できません</b>　'+g.ng.map(x=>'・'+x).join('　');
 }else if(g.warn&&g.warn.length){
   ge.style.borderColor='var(--warn)'; ge.style.background='#2a2414';
   ge.innerHTML='<b style="color:var(--warn)">▲ 要注意(押せます)</b>　'+g.warn.map(x=>'・'+x).join('　');
 }else{
   ge.style.borderColor='var(--ok)'; ge.style.background='#142a1e';
   ge.innerHTML='<b style="color:var(--ok)">● 実行してよい状態です</b>　制御ループ・電源・接地・立ち方すべて基準内';
 }
 fill('sel_climb',d.patterns.climb,d.sel.climb,true,d.notes);
 fill('sel_turn',d.patterns.turn,d.sel.turn,true,d.notes);
 fill('sel_sit',d.patterns.sit,d.sel.sit,true,d.notes);
 document.getElementById('phases').textContent=
  d.phases.map((p,i)=>(i===d.phase_i?'▶':'')+p).join('  →  ')||'-';
 document.getElementById('prog').style.width=(100*d.t/Math.max(d.n,1))+'%';
 document.getElementById('tn').textContent=`フェーズ ${d.phase_i+1}/${d.phases.length}  コマ ${d.t}/${d.n}`;
 document.getElementById('msg').textContent=d.msg;
 document.getElementById('log').textContent=d.logs.join('\\n');
 document.getElementById('next').disabled=(d.fsm!=='WAIT_CONFIRM');
 document.getElementById('place').style.display=d.is_sim?'inline-block':'none';
 drawWalk(d.walk||{});
}
// ---- 歩行(内蔵制御): 状態表示 / パラメータ / 十字キー ----
function drawWalk(w){
 const ws=document.getElementById('walkst'); if(!ws)return;
 const p=w.params||{};
 const dist=(w.dist==null?'---':w.dist.toFixed(2)+'m');
 const dcol=(w.dist!=null&&w.dist<=(p.stop_dist||0.6))?'var(--bad)':'var(--ok)';
 const lid=(w.lidar_age_ms==null?'<b style="color:var(--bad)">未受信</b>':
   (w.lidar_age_ms<1500?'OK':'<b style="color:var(--bad)">途絶</b>')+' '+w.lidar_age_ms+'ms');
 const odo=(w.odom_age_ms==null?'<b style="color:var(--bad)">未受信</b>':
   (w.odom_age_ms<800?'OK':'<b style="color:var(--bad)">途絶</b>'));
 ws.innerHTML='歩行 <b>'+(w.ready?'準備OK(FSM '+w.fsm_id+')':'未準備')+'</b>'
  +(w.auto?'&nbsp;<b style="color:var(--acc)">自動歩行中 '+w.phase+'</b>':'')
  +(p.dry_run?'&nbsp;<b style="color:var(--warn)">ドライラン</b>':'')
  +'&nbsp; 前方 <b style="color:'+dcol+'">'+dist+'</b>('+(w.n_obs||0)+'pt)'
  +'&nbsp; 横の空き '+(w.side_free==null?'---':w.side_free.toFixed(2)+'m')
  +'<br>LiDAR '+lid+' 座標系 '+(w.frame||'?')+' 床 '
  +(w.floor_ok?(w.floor_h+'m'):'<b style="color:var(--warn)">未検出</b>')
  +'&nbsp; odom '+odo+(w.odom?' ('+w.odom[0]+', '+w.odom[1]+', '+w.odom[2]+'&deg;)':'')
  +'<br>送信 ('+(w.sent||[]).join(', ')+') 成功'+(w.n_sent||0)+'/失敗'+(w.n_fail||0)
  +'&nbsp; 前進'+(w.traveled||0)+'m 横'+(w.side_traveled||0)+'m'
  +(w.msg?'<br><b>'+w.msg+'</b>':'')
  +(w.why?'<br><span style="color:var(--warn)">'+w.why+'</span>':'');
 const ab=document.getElementById('walk_auto');
 if(ab){ab.disabled=!!w.auto||!w.ready; ab.textContent=w.auto?'自動歩行 実行中…':'▶ 自動歩行 開始';}
}
function walkParams(){
 const v=+document.getElementById('w_v').value;
 return {stop_dist:+document.getElementById('w_stop').value,
  side_dir:document.getElementById('w_dir').value,
  side_dist:+document.getElementById('w_side').value,
  v_fwd:v, tele_vx:v,
  max_fwd:+document.getElementById('w_max').value,
  dry_run:document.getElementById('w_dry').checked};
}
function sendWalkParams(){cmd('walk_param',JSON.stringify(walkParams()))}
function startAuto(){
 const w=S.walk||{};
 if(!w.ready){alert('先に[歩行モードへ]を押してください');return}
 sendWalkParams();
 setTimeout(function(){cmd('walk_auto')},150);
}
// 十字キー: pointerdown で保持、100msごとに tele を送る。離す/外れる/キャンセル/
// ウィンドウのフォーカス喪失で即ゼロ。サーバ側も0.5秒更新が無ければ止める
let TELE=null, TELE_TIMER=null;
function teleStart(e,b){
 e.preventDefault();
 if(b.setPointerCapture){try{b.setPointerCapture(e.pointerId)}catch(x){}}
 const v=b.dataset.v.split(',').map(Number), p=walkParams();
 TELE=[v[0]*p.v_fwd, v[1]*0.20, v[2]*0.45].map(x=>x.toFixed(3)).join(',');
 b.classList.add('on');
 cmd('tele',TELE);
 if(TELE_TIMER)clearInterval(TELE_TIMER);
 TELE_TIMER=setInterval(function(){if(TELE)cmd('tele',TELE)},100);
}
function teleEnd(){
 if(TELE_TIMER){clearInterval(TELE_TIMER);TELE_TIMER=null}
 if(TELE===null)return;
 TELE=null;
 document.querySelectorAll('.dk').forEach(x=>x.classList.remove('on'));
 cmd('tele','0,0,0');
}
document.querySelectorAll('.dk[data-v]').forEach(b=>{
 b.addEventListener('pointerdown',e=>teleStart(e,b));
 ['pointerup','pointercancel','pointerleave','lostpointercapture'].forEach(ev=>b.addEventListener(ev,teleEnd));
});
window.addEventListener('blur',teleEnd);
// ---- 画面消灯防止: Wake Lock API(https/localhost のみ) → だめなら小さな動画を再生 ----
let WL=null;
function setKA(s){const e=document.getElementById('ka_st');if(e)e.textContent=s}
async function keepAwake(){
 if('wakeLock' in navigator){
  try{WL=await navigator.wakeLock.request('screen');
      WL.addEventListener('release',function(){WL=null;setKA('解除(画面をタップで再取得)')});
      setKA('WakeLock有効');return}catch(e){}
 }
 const v=document.getElementById('ka');
 if(v){v.play().then(function(){setKA('動画で維持中(端末のスリープも長めに設定)')})
       .catch(function(){setKA('★無効 — 端末のスリープ設定を長くすること')})}
}
setInterval(tick,200);tick();
setInterval(function(){cmd('beat')},1000);
document.addEventListener('visibilitychange',function(){if(!document.hidden){cmd('beat');keepAwake()}});
document.addEventListener('pointerdown',function(){if(!WL)keepAwake()});
keepAwake();
document.addEventListener('keydown',e=>{
 if(e.key===' '&&e.target.tagName!=='INPUT'){e.preventDefault();cmd('estop')}});
</script></body></html>"""

# ---- スマホ向けの付属物(2026-09-04) ------------------------------------------
# 画面消灯防止用の小さな動画(VP8/webm 160×120・2秒・ループ)。Wake Lock API は
# https か localhost でしか使えないので、http://192.168.179.100:8090 では
# 「再生中の動画」で端末のスリープを抑止する(Androidの挙動に依存する保険。
# 端末側のスリープ時間も長くしておくこと。docs/Android無線操作.md)
KEEPAWAKE_WEBM_B64 = (
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAS1EU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHYTbuMU6uEElTDZ1OsggEbTbuMU6uEHFO7a1OsggSf7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmsirXsYMPQkBNgI1MYXZmNjIuMTIuMTAxV0GNTGF2ZjYyLjEyLjEwMUSJiECfQAAAAAAAFlSua76uAQAAAAAAADXXgQFzxYigMusuWuY9eJyBACK1nIN1bmSIgQCGhVZfVlA4g4EBI+ODhA7msoDghrCBoLqBeBJUw2fYc3OgY8CAZ8iaRaOHRU5DT0RFUkSHjUxhdmY2Mi4xMi4xMDFzc7JjwItjxYigMusuWuY9eGfIoUWjiERVUkFUSU9ORIeTMDA6MDA6MDIuMDAwMDAwMDAwAB9DtnVDIeeBAKNBcIEAAIAQDQCdASqgAHgAAUcIhYWIhYSIAaIQGUXqRIbUByAPiq5IDvEX7fekdcgWKZzYH+zeAv7AP8I/in+T4AH9gIO1at6KyIPdmMAuGOkJf4lEg0+bxMYnsx0hJXETOsk8ziqbZjALhjpCYAqyIPdmMAuF+AD+/43EnfZF9se0mcuRvmQ5URUwro306Kfv//dOJf6CkX+vsYgn+wOk3hYQ0HejWnJfAa5KUUKv09IQY7f+Ye/Ps0AIao9uvEYrxwXaJt9tRXpHvukcv3M/zbWdfe2ytOQ8Wcqp5bHdb26wQ1j8a2WxvVXEg1S4/r8ojTaUGMOTqRzlv22n+hl1/vX8o+n519zRvMb1gX4sT/+FPO/2qt9LfZM1bMNhIoOgYj2CC5MMI0JSZLBcmHeP+fvOMSqUPhrn5H/k3NRQcv/3HQ//C7rM/yxWJKAf8bCgCtCGslNa1FivN/v9Dv3XmP2cVcXm9JP9ORomK96wzvwAo8OBAPoAEQQAARANEADAThu5+54HlyBpsrvOAOu58gJIB0cQDIpbNQBcre/5EkbqwbGi16O8ARBWTshYOeeBtlGttOIAo7KBAfQAsQMAAxANEADAAMsFOH9Btm4UIbIWtK4+g1mePl46rAAP15YAB9OxJTNJ1+R/AKO8gQLuALEDAAMQDRAAwADLAXP/QVCUWFvux3NTjOakzeoLOIWAMnOClv734rU1CI9CfFuhJ4lvPJml507Ao7eBA+gA0QMABxANEAD1YYBbCKafqpZGaLv6Cfx/my2Xo/0FavLMLp2xEnJdo3iYI3vO14XTXrAAo8aBBOIAEQQABxANEADANr9kSrABbCKijqpZXizRprp4VNrLaX6U2XQbp4p1FZyKXDcE4YasCQzdAJkN25MswVhDCrxFWsAAo7WBBdwAEQMABxANEADAAMsBer9BUXTlqOUGoRgUnh7ICsEugRwAMBLpOF5YELUaMBwGLOJ9gKO6gQbWABEDAAcQDRCjAAMsBer9BUXTiiAH33KQSmT3IBMMRz3kjVnAfPECoa5/vVdl+MLBdFQbgSFwABxTu2uRu4+zgQC3iveBAfGCAXjwgQM=")
MANIFEST = json.dumps({
    "name": "G1 Cockpit", "short_name": "G1", "start_url": "/",
    "display": "standalone", "background_color": "#111111",
    "theme_color": "#111111", "lang": "ja",
    "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
}, ensure_ascii=False)
ICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<rect width="64" height="64" rx="12" fill="#111"/>'
            '<text x="32" y="43" font-size="30" text-anchor="middle" fill="#3987e5" '
            'font-family="sans-serif" font-weight="700">G1</text></svg>')

# ---- かんたん画面(2026-09-04)。既定の / で出す。詳細画面は /detail ----------------
# 基本操作(ダンプ/スタンドロック/歩行モード)・前進(壁の手前で自然停止)・横歩き(足踏み)・
# 5cm微調整・着座(サーバ側の確認 → 操作者の確認 → 3秒カウントダウン)・E-STOP。
# ★ここも preflight 6b/6c の検査対象(cmd('X') が CMD_ALLOW にあるか、id が揃っているか)。
PAGE_SIMPLE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>G1 かんたん操作</title>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#111">
<meta name="mobile-web-app-capable" content="yes">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<style>
:root{--bg:#111;--card:#1b1b1b;--line:#333;--t1:#eee;--t2:#9a9a9a;--ok:#1baf7a;--warn:#eda100;--bad:#e34948;--acc:#3987e5}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--t1);font:16px/1.5 "Segoe UI",sans-serif;padding:10px 10px 110px;max-width:720px;margin:auto;overflow-x:hidden}
.hd{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.hd b{font-size:20px}.hd a{color:var(--acc);margin-left:auto;font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:6px 8px;min-width:0}
.tile .k{font-size:11px;color:var(--t2)}.tile .v{font-size:17px;font-weight:700;overflow-wrap:anywhere}
#fsm{font-size:14px;letter-spacing:-.02em}
.judge{margin:8px 0;padding:8px 12px;border-radius:10px;border:1px solid var(--line);font-size:14px;overflow-wrap:anywhere}
.warnbar{margin:8px 0;padding:8px 12px;border-radius:10px;border:1px solid var(--warn);background:#2a2414;font-size:14px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-top:10px}
section h2{font-size:13px;color:var(--t2);letter-spacing:.05em;margin-bottom:8px}
.btns3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.btns2{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:8px}
button{font:inherit;border-radius:12px;border:1px solid var(--line);background:#262626;color:var(--t1);padding:14px 8px;min-height:56px;cursor:pointer;font-weight:700}
button:disabled{opacity:.35}
button.go{background:var(--acc);border:none}
button.stop{background:#7a1b1b;border:none}
button.big{width:100%;font-size:18px;min-height:64px;margin-top:6px}
.row{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:6px 0;font-size:14px}
input[type=number],select{font:inherit;border-radius:8px;border:1px solid var(--line);background:#262626;color:var(--t1);padding:8px}
input[type=number]{width:84px}
.st{font-size:13px;color:var(--t2);margin-top:8px;line-height:1.7;overflow-wrap:anywhere}
.gate{margin-top:10px;padding:10px;border:1px solid var(--acc);border-radius:10px;background:#141c2a}
.gate label{display:block;padding:6px 0;font-size:15px}
.gi{font-size:14px;padding:2px 0}
.log{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;color:var(--t2);background:#161616;border-radius:8px;padding:8px;max-height:200px;overflow:auto}
.fixbar{position:fixed;left:0;right:0;bottom:0;display:flex;gap:8px;padding:8px 10px;background:rgba(17,17,17,.97);border-top:1px solid var(--line);z-index:30}
.fixbar button{flex:1;padding:16px 8px;font-size:18px;font-weight:900;border:none;color:#fff}
.estop{background:var(--bad)}.stopb{background:#7a1b1b}
</style></head><body>
<div class="hd"><b>&#129302; G1</b><span id="mode" class="st" style="margin:0"></span><a href="/detail">詳細画面 &rarr;</a></div>
<div class="tiles">
 <div class="tile"><div class="k">FSM</div><div class="v" id="fsm">-</div></div>
 <div class="tile"><div class="k">傾き</div><div class="v" id="tilt">-</div></div>
 <div class="tile"><div class="k">制御</div><div class="v" id="loop">-</div></div>
 <div class="tile"><div class="k">通信</div><div class="v" id="comm">-</div></div>
</div>
<div id="go" class="judge">-</div>
<div id="uiloss" class="warnbar" hidden></div>
<div id="seatdoubt" hidden style="background:#5a1b1b;color:#ffd6d6;border:1px solid #c33;border-radius:8px;padding:8px 10px;margin:6px 0;font-size:14px"></div>

<section><h2>基本操作</h2>
 <div class="btns3">
  <button onclick="cmd('mode_damp')">ダンプ</button>
  <button onclick="cmd('mode_stand')">スタンドロック</button>
  <button onclick="cmd('walk_ready')">歩行モード</button>
 </div>
 <div class="st">スタンドロック=立位(FSM4)。歩行モード=走行制御(802)へ。押すと歩き出しうるので接地とE-STOPを確認。</div>
</section>

<section><h2>歩く</h2>
 <div class="row">
  <label>壁の手前で止まる距離 <input id="w_stop" type="number" step="0.05" min="0.3" max="2.5" value="0.60"> m</label>
  <label>横歩き <select id="w_dir"><option value="left">左へ</option><option value="right">右へ</option></select>
   <input id="w_side" type="number" step="0.05" min="0.02" max="3" value="0.50"> m</label>
  <label><input id="w_avoid" type="checkbox" checked> 障害物は回り込む</label>
 </div>
 <div class="btns3">
  <button class="go" id="b_fwd" onclick="walkGo('forward')">&#9650; 前進</button>
  <button class="go" id="b_side" onclick="walkGo('side')">&#9664;&#9654; 横歩き</button>
  <button class="stop" onclick="cmd('walk_stop')">&#9632; 歩行停止</button>
 </div>
 <div class="st" style="margin:8px 0 2px">小刻みステップ — 内蔵歩行に短い指令を出して1歩ずつ動き、オドメトリで測る。まず [1歩] で何cm動くかを見る。
  強さ <select id="w_stepv" onchange="cmd('walk_param',JSON.stringify({step_v:+this.value}))"><option value="0.15">弱 0.15</option><option value="0.2" selected>中 0.20</option><option value="0.3">強 0.30</option></select>
  <span id="stepinfo">-</span></div>
 <div class="btns3">
  <button onclick="step1('left')">&#9664; 左へ1歩</button>
  <button onclick="step1('back')">&#9660; 後ろへ1歩</button>
  <button onclick="step1('right')">右へ1歩 &#9654;</button>
 </div>
 <div class="btns2">
  <button onclick="nudge('left')">&#9664; 左へ5cm</button>
  <button onclick="nudge('right')">右へ5cm &#9654;</button>
 </div>
 <div id="walldist" style="margin-top:8px;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:#161616">
  <div class="st" style="margin:0">正面の壁までの距離（つま先から、壁の面を当てて測る） — LiDAR は自動で読んでいます</div>
  <div style="font-size:34px;font-weight:800;line-height:1.2" id="walldist_v">-</div>
  <div class="st" id="walldist_o" style="margin:0;color:var(--warn)"></div>
  <div class="st" id="walldist_s" style="margin:0">-</div>
  <div class="st" id="walldist_d" style="margin:4px 0 0">前 - / 後 - / 左 - / 右 -</div>
  <div class="row" style="margin-top:6px"><button onclick="if(confirm('LiDAR の前後を反転します。正面の壁の数字が「後ろ」の距離と入れ替わります。よいですか？'))cmd('lidar_flip')" style="min-height:40px">⇄ 前後を反転（数字が後ろの壁と合っているとき）</button></div>
 </div>
 <div id="walkst" class="st">-</div>
</section>

<section><h2>座る</h2>
 <div class="row">方策 <select id="sel_sit" onchange="sel('sit')"></select></div>
 <div class="row">椅子との距離を詰める（歩行モードで。後ろは LiDAR が見えないので目で見る）
  <button onclick="nudgeBack(0.05)" style="min-height:44px">▼ 後ろへ5cm</button>
  <button onclick="nudgeBack(0.10)" style="min-height:44px">▼ 後ろへ10cm</button></div>
 <button class="go big" id="sit_btn" onclick="cmd('sit_check')">&#129681; 着座（この位置でよいか確認）</button>
 <div id="sitgate" class="gate" hidden>
  <div id="gate_items"></div>
  <label><input type="checkbox" class="gk" onchange="gateUpd()"> 椅子は真後ろ、座面の前縁が踵の位置にある</label>
  <label><input type="checkbox" class="gk" onchange="gateUpd()"> 機体は自立して静止している（支えていない）</label>
  <label><input type="checkbox" class="gk" onchange="gateUpd()"> 周囲に人がいない。リモコンのE-STOPを握っている</label>
  <div class="btns2"><button class="go" id="gate_go" onclick="sitGo()" disabled>着座を開始（3秒後）</button><button onclick="gateCancel()">やめる</button></div>
  <div id="gate_cd" class="st"></div>
 </div>
 <div id="sitst" class="st">-</div>
</section>

<section><h2>通信が切れたときの動作（いまの状態）</h2><div id="uiplan" class="st">-</div>
 <div class="st">物理E-STOP（リモコン）が最上位。傾き40度・受信断・送信断は従来どおり即ダンプ。</div></section>
<section><h2>ログ</h2><div id="log" class="log"></div>
 <div class="st">画面消灯防止: <span id="ka_st">-</span>
 <video id="ka" src="/keepawake.webm" loop muted playsinline style="width:64px;height:48px;vertical-align:middle;opacity:.5;border-radius:6px"></video></div></section>

<div class="fixbar">
 <button class="estop" onclick="cmd('estop')">&#9632; E-STOP</button>
 <button class="stopb" onclick="cmd('walk_stop')">&#9632; 歩行停止</button>
</div>

<script>
let S={}, META=null, LASTOK=0, GATE_T=null, CD=null;
function cmd(c,a){fetch('/cmd?c='+c+(a?('&a='+encodeURIComponent(a)):''),{method:'POST'}).catch(function(){})}
function sel(k){cmd('select',k+':'+document.getElementById('sel_'+k).value)}
function walkGo(mode){
 const d={mode:mode, stop_dist:+document.getElementById('w_stop').value,
  side_dir:document.getElementById('w_dir').value, side_dist:+document.getElementById('w_side').value,
  avoid:document.getElementById('w_avoid').checked};
 cmd('walk_go',JSON.stringify(d));
}
function nudge(dir){cmd('walk_go',JSON.stringify({mode:'side',nudge:true,side_dir:dir,side_dist:0.05}))}
function nudgeBack(m){cmd('walk_go',JSON.stringify({mode:'back',nudge:true,back_dist:m}))}
function step1(dir){cmd('walk_go',JSON.stringify({mode:'step',dir:dir}))}
function drawWall(w){
 const si=document.getElementById('stepinfo'); if(si) si.textContent='1歩≈'+(w.step_est_cm!=null?w.step_est_cm:'-')+'cm（前回 '+(w.step_last_cm!=null?w.step_last_cm:'-')+'cm、'+(w.steps||0)+'歩）';
 const sv=document.getElementById('w_stepv'); if(sv&&w.params&&w.params.step_v!=null&&document.activeElement!==sv){ const t=String(+w.params.step_v); for(const o of sv.options){ if(String(+o.value)===t) sv.value=o.value; } }
 const v=document.getElementById('walldist_v'), s=document.getElementById('walldist_s'); if(!v)return;
 const p=w.params||{}, sd=p.stop_dist||0.6;
 const lidarOK=(w.lidar_age_ms!=null&&w.lidar_age_ms<1500);
 if(!lidarOK){ v.textContent='LiDAR 未受信'; v.style.color='var(--bad)'; s.textContent='点群が来ていません。機体の lidar_bridge を確認'; return; }
 const o=document.getElementById('walldist_o'), dd=document.getElementById('walldist_d');
 const wd=w.wall_dist, od=w.dist;
 if(wd!=null){ v.textContent=wd.toFixed(2)+' m（壁'+(w.wall_ang!=null&&Math.abs(w.wall_ang)>=8?'、'+Math.abs(w.wall_ang)+'° 斜め':'')+'）';
   const near=(od!=null&&od<wd-0.15);
   const ref=near?od:wd; v.style.color=(ref<=sd)?'var(--bad)':(ref<=sd+0.5?'var(--warn)':'var(--ok)');
   if(o) o.textContent=near?('手前 '+od.toFixed(2)+' m に障害物'+(w.width!=null?'（幅 '+w.width+' m）':'')+' — 止まるのはこちら'):''; }
 else if(od==null){ v.textContent='3m 以内に なし'; v.style.color='var(--ok)'; if(o)o.textContent=''; }
 else { v.textContent=od.toFixed(2)+' m'+(w.wall?'（壁）':(w.width!=null?'（幅 '+w.width+' m の障害物）':'')); v.style.color=(od<=sd)?'var(--bad)':(od<=sd+0.5?'var(--warn)':'var(--ok)'); if(o)o.textContent='壁の面は取れていません（手前の物までの距離）'; }
 if(dd&&w.dirs){ const f=x=>(x==null?'-':x.toFixed(2)); dd.textContent='前 '+f(w.dirs.front)+' / 後 '+f(w.dirs.back)+' / 左 '+f(w.dirs.left)+' / 右 '+f(w.dirs.right)+' m（±20°の最近点、つま先から）'+(w.yaw_fix_deg?'　ヨー補正 '+w.yaw_fix_deg+'°':''); }
 s.textContent='停止距離 '+sd.toFixed(2)+' m　LiDAR '+w.lidar_age_ms+'ms 点'+(w.n_obs||0)+' 床 '+(w.floor_h==null?'-':w.floor_h+'m')
  +(w.mount?'　取付: 高さ'+w.mount.height+'m 傾き'+w.mount.tilt_deg+'°':'')+(w.free_l!=null||w.free_r!=null?'　回り込み可 '+(w.free_l!=null?'左':'')+(w.free_r!=null?'右':''):'');
}
function gateUpd(){
 const g=S.sit_gate, ks=document.querySelectorAll('.gk');
 const all=Array.from(ks).every(x=>x.checked);
 const b=document.getElementById('gate_go'); if(b)b.disabled=!(g&&g.ok&&all&&!CD);
}
function gateCancel(){ if(CD){clearInterval(CD);CD=null;} GATE_T=null; document.getElementById('sitgate').hidden=true;
 document.querySelectorAll('.gk').forEach(x=>x.checked=false); document.getElementById('gate_cd').textContent='';}
function sitGo(){
 const g=S.sit_gate; if(!g||!g.ok)return;
 let n=3; const e=document.getElementById('gate_cd'); e.textContent='3秒後に着座を開始します… [やめる]で中止';
 document.getElementById('gate_go').disabled=true;
 CD=setInterval(function(){ n--; if(n>0){e.textContent=n+'秒後に着座を開始します… [やめる]で中止';return}
  clearInterval(CD);CD=null; e.textContent='開始しました'; cmd('sit_go',g.token);
  setTimeout(gateCancel,1500); },1000);
}
async function loadMeta(){try{const m=await(await fetch('/state?meta=1')).json();META={patterns:m.patterns,notes:m.notes}}catch(e){}}
function fillSit(d){const e=document.getElementById('sel_sit'); if(!e||e.dataset.done||!META)return; e.dataset.done=1;
 e.innerHTML=META.patterns.sit.map(x=>'<option value="'+x+'" '+(x===d.sel.sit?'selected':'')+'>'+x+(META.notes[x]?' — '+META.notes[x].slice(0,40):'')+'</option>').join('');}
async function tick(){
 if(!META){await loadMeta();if(!META)return}
 let d; try{d=await(await fetch('/state')).json()}catch(e){ drawComm(false); return }
 S=d; LASTOK=Date.now(); drawComm(true); fillSit(d);
 document.getElementById('mode').textContent=d.is_sim?'[SIMモック]':'[実機]';
 const f=document.getElementById('fsm'); f.textContent=d.fsm; f.style.color=d.fsm==='DAMP'?'var(--bad)':(d.fsm==='RUNNING'?'var(--ok)':'var(--t1)');
 const te=document.getElementById('tilt'); te.textContent=d.tilt_deg.toFixed(0)+'°'; te.style.color=d.tilt_deg>25?'var(--warn)':'var(--t1)';
 const lp=d.loop||{}, le=document.getElementById('loop'); le.textContent=(lp.hz?lp.hz.toFixed(0):'-')+'Hz'; le.style.color=lp.ok?'var(--ok)':'var(--bad)';
 const g=d.go||{ok:true,ng:[],warn:[]}, ge=document.getElementById('go');
 if(g.ng&&g.ng.length){ge.style.borderColor='var(--bad)';ge.style.background='#2a1414';ge.innerHTML='<b style="color:var(--bad)">■ 実行できません</b> '+g.ng.map(x=>'・'+x).join(' ');}
 else if(g.warn&&g.warn.length){ge.style.borderColor='var(--warn)';ge.style.background='#2a2414';ge.innerHTML='<b style="color:var(--warn)">▲ 要注意</b> '+g.warn.map(x=>'・'+x).join(' ');}
 else{ge.style.borderColor='var(--ok)';ge.style.background='#142a1e';ge.innerHTML='<b style="color:var(--ok)">● 実行してよい状態です</b>';}
 const ul=document.getElementById('uiloss');
 if(d.ui_lost_last){ul.hidden=false; ul.innerHTML='<b>通信が '+d.ui_lost_last.dur+' 秒途絶えていました('+d.ui_lost_last.t+')</b><br>途絶中の対応: '+d.ui_lost_last.what;} else ul.hidden=true; const sd=document.getElementById('seatdoubt'); if(sd){ if(d.seat_doubt){sd.hidden=false; sd.innerHTML='<b>★座面に載っていない疑い</b>: '+d.seat_doubt+'<br>自動ダンプは保留。方策で保持中 — 目視で確認して [ダンプ] か [スタンドロック]';} else sd.hidden=true; }
 document.getElementById('uiplan').textContent=d.ui_plan||'-';
 const w=d.walk||{}, p=w.params||{};
 const dist=(w.dist==null?'なし(3m以内)':w.dist.toFixed(2)+'m'+(w.wall?'（壁）':(w.width!=null?'（幅'+w.width+'m の障害物）':'')));
 document.getElementById('walkst').innerHTML=
  '歩行 <b>'+(w.ready?'準備OK(FSM '+w.fsm_id+')':'未準備 — [歩行モード]を押す')+'</b>'
  +(w.auto?' <b style="color:var(--acc)">実行中 '+w.phase+'</b>':'')
  +'<br>前方 <b style="color:'+((w.dist!=null&&w.dist<=(p.stop_dist||0.6))?'var(--bad)':'var(--ok)')+'">'+dist+'</b>'
  +' 速度 '+(w.v||0)+' m/s　進み '+(w.traveled||0)+'m　ずれ '+((w.offset||0)*100).toFixed(0)+'cm　回り込み '+(w.detours||0)+'回'
  +'<br>LiDAR '+(w.lidar_age_ms==null?'<b style="color:var(--bad)">未受信</b>':(w.lidar_age_ms<1500?'OK':'<b style="color:var(--bad)">途絶</b>'))
  +'　odom '+(w.odom_age_ms==null?'<b style="color:var(--bad)">未受信</b>':(w.odom_age_ms<800?'OK':'<b style="color:var(--bad)">途絶</b>'))
  +(w.msg?'<br>'+w.msg:'');
 const fb=document.getElementById('b_fwd'), sb=document.getElementById('b_side'); if(fb){fb.disabled=!w.ready||w.auto; sb.disabled=!w.ready||w.auto;}
 drawWall(w);
 document.getElementById('sitst').innerHTML=(d.phases&&d.phases.length?('方策 '+d.phases.join(' → ')+'　コマ '+d.t+'/'+d.n+'<br>'):'')+(d.msg||'');
 const gt=d.sit_gate, gp=document.getElementById('sitgate');
 if(gt&&(gt.token!==GATE_T)){GATE_T=gt.token; document.querySelectorAll('.gk').forEach(x=>x.checked=false); document.getElementById('gate_cd').textContent='';}
 if(gt){gp.hidden=false;
  document.getElementById('gate_items').innerHTML='<div class="gi"><b>'+(gt.ok?'点検OK — 下の3つを確認して開始':'★点検NG — 直してからもう一度')+'</b>（'+gt.pattern+'、'+gt.age+'秒前）</div>'
   +gt.items.map(it=>'<div class="gi" style="color:'+(it[0]===false?'var(--bad)':(it[0]===null?'var(--warn)':'var(--ok)'))+'">'+(it[0]===false?'×':(it[0]===null?'△':'○'))+' '+it[1]+'</div>').join('');
  if(gt.age>30){gateCancel();}
  gateUpd();
 } else if(!CD){gp.hidden=true;}
 document.getElementById('log').textContent=(d.logs||[]).slice(-10).join(String.fromCharCode(10));
}
function drawComm(ok){const c=document.getElementById('comm'); if(!c)return;
 const age=(Date.now()-LASTOK)/1000;
 if(ok||age<3){c.textContent='OK';c.style.color='var(--ok)';} else {c.textContent='途絶 '+age.toFixed(0)+'s';c.style.color='var(--bad)';}}
let WL=null;
function setKA(s){const e=document.getElementById('ka_st');if(e)e.textContent=s}
async function keepAwake(){
 if('wakeLock' in navigator){try{WL=await navigator.wakeLock.request('screen');WL.addEventListener('release',function(){WL=null;setKA('解除(タップで再取得)')});setKA('WakeLock有効');return}catch(e){}}
 const v=document.getElementById('ka'); if(v){v.play().then(function(){setKA('動画で維持中(端末のスリープも長めに)')}).catch(function(){setKA('★無効 — 端末のスリープ設定を長くすること')})}
}
setInterval(tick,250);tick();
setInterval(function(){cmd('beat')},1000);
document.addEventListener('visibilitychange',function(){if(!document.hidden){cmd('beat');keepAwake()}});
document.addEventListener('pointerdown',function(){if(!WL)keepAwake()});
keepAwake();
document.addEventListener('keydown',e=>{if(e.key===' '&&e.target.tagName!=='INPUT'){e.preventDefault();cmd('estop')}});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None
    patterns = None
    # ★2026-09-03 HTTP/1.1(Keep-Alive)にする。
    #   既定の HTTP/1.0 はリクエストごとにTCP接続を張り直す。UIは
    #   /state を5回/秒 + beat を1回/秒 = 6接続/秒 も張っていた。
    #   無線(実測ロス10%)ではSYNが落ちるたびに**再送で1秒以上**止まり、
    #   これが「UIが重い」の正体だった。Keep-Aliveなら接続を使い回すので
    #   ハンドシェイクごとの取りこぼしが消える。
    #   _send は必ず Content-Length を送っているので、そのまま切り替えて安全。
    protocol_version = "HTTP/1.1"
    timeout = 15                      # 放置された接続を掴んだままにしない

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/state":
            qs = parse_qs(urlparse(self.path).query)
            full = "full" in qs
            d = self.engine.snapshot(full=full)
            # ★2026-09-03 patterns/notes/warn は起動中まったく変わらないのに
            #   毎tick(5回/秒)送っていた=2.6KB×5/秒の無駄。無線では効く。
            #   クライアントは起動時に ?meta=1 で一度だけ取って持っておく。
            if "meta" in qs:
                d["patterns"] = self.patterns
                d["notes"] = PATTERN_NOTES
                d["warn"] = PATTERN_WARN
            self._send(json.dumps(d).encode(), "application/json")
        elif p == "/frame.jpg" and self.engine.is_sim:
            self._send(self.engine.robot.render_jpeg(), "image/jpeg")
        elif p == "/keepawake.webm":
            self._send(base64.b64decode(KEEPAWAKE_WEBM_B64), "video/webm")
        elif p == "/manifest.webmanifest":
            self._send(MANIFEST.encode(), "application/manifest+json")
        elif p == "/icon.svg":
            self._send(ICON_SVG.encode(), "image/svg+xml")
        elif p in ("/detail", "/full"):
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        else:
            # 既定はかんたん画面(2026-09-04)。詳細画面は /detail
            self._send(PAGE_SIMPLE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        q = parse_qs(urlparse(self.path).query)
        c = q.get("c", [""])[0]
        a = q.get("a", [None])[0]
        if c in ("estop", "damp"):
            # ★キューに載せない。50Hzループが方策読み込みやSDKのRPCで
            #   詰まっていても、ここで即座にdampが出る(2026-08-26)
            why = ("操作者による緊急停止" if c == "estop"
                   else "操作者によるdamp")
            self.engine.estop_now(why)
        elif c == "beat":
            self.engine.beat()
        elif c == "walk_stop":
            # ★キューに載せない。歩行の停止は即時に(速度ゼロ。dampではない)
            self.engine.walk_stop(a or "操作者による停止")
        elif c == "tele":
            # スマホ十字キー(100ms周期)。キューに積むと遅れるので直接渡す
            self.engine.beat()                     # 操作中はUIが生きている証拠
            self.engine.tele(a)
        elif c == "select" and a and ":" in a:
            k, v = a.split(":", 1)
            self.engine.command("select", (k, v))
        elif c in CMD_ALLOW:
            self.engine.command(c, a)
        else:
            # ★2026-08-27。ここは黙って捨てていた。[補助]と[ヨー合わせ]と
            #   [打ち切りコマ数]は許可リストに入れ忘れていて、UIで選んでも
            #   何も起きなかった(操作者には「選び忘れ」に見えていた)。
            #   preflight の 6c が PAGE の cmd('X') を全部照合するようにしたが、
            #   実行時にも気づけるようにログへ出す。
            self.engine.log(f"★UIからの未知のコマンド '{c}' を無視しました"
                            f"(cockpit.py の CMD_ALLOW に足してください)")
        self._send(b"ok", "text/plain")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="MuJoCoモックで結合試験")
    ap.add_argument("--iface", default="", help="実機NIC名(例 enp46s0)")
    ap.add_argument("--port", type=int, default=8090)
    # ★2026-09-03 既定を3.0→8.0へ。UIは1秒ごとにビートを打つので3.0は
    #   「3回連続で取りこぼしたら止める」の意味だった。有線LAN前提の値で、
    #   無線(実測: ロス最大83%・RTT 13〜1406ms)では瞬断のたびに誤ってDAMPに
    #   入る。オンボード運用では制御は機体内で完結していて通信断でも止まらず、
    #   途絶して困るのは「UIから介入できないこと」だけ。本当の安全装置は
    #   リモコンのE-STOP(ハードウェア・ネットワーク非依存)なので、
    #   無線の瞬断を吸収しつつ本当にUIを失えば8秒で止まる値にする。
    #   ★ブラウザは前面に置くこと。背面タブだと setInterval が抑制され
    #     (Chromeは最小1回/分)、この値をいくら延ばしても途絶する。
    ap.add_argument("--heartbeat-sec", type=float, default=8.0,
                    help="UIハートビートの許容間隔[秒]。0で無効"
                         "(既定8.0=無線の瞬断を吸収する値)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="HTTP待受アドレス(既定0.0.0.0=遠隔可。localhost縛りは127.0.0.1)")
    a = ap.parse_args()
    if a.sim:
        from sim_robot import SimRobot
        robot = SimRobot()
        print("SIMモックで起動")
    else:
        from real_robot import RealRobot
        robot = RealRobot(iface=a.iface)
        print("実機に接続しました")
    # 制御ループの尾側の遅れ対策。推論そのものは50Hz予算20msに対して
    # 中央1.1msしか使っていないが、GCの一時停止で数十msの飛びが出る。
    # 50Hzが1コマ遅れても500Hz送信スレッドが直前の目標を送り続けるので
    # 実害は小さいが、GILを握ったまま止まると送信自体が途切れうる。
    #  - freeze(): 起動時までに作った全オブジェクトを恒久世代へ移し、
    #              以降のGCの走査対象から外す(解放はしない=安全)
    #  - set_threshold: 世代0の閾値を上げて走行中の回収頻度を下げる
    # 効いているかは、ログの dt_ms / ms_infer を実機で見て判断する
    import torch
    torch.set_num_threads(1)          # 推論スレッドが送信スレッドを食わないように
    gc.collect()
    gc.freeze()
    gc.set_threshold(100_000, 50, 50)
    eng = Engine(robot, a.sim, heartbeat_sec=a.heartbeat_sec)
    Handler.engine = eng
    Handler.patterns = list_patterns()
    # Keep-Aliveだとブラウザが複数の接続を張ったまま保つので待ち行列を広げる
    ThreadingHTTPServer.request_queue_size = 32
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"コックピット: http://{a.host}:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n中断されました")
    finally:
        # ★set_damp だけでは送信バッファを書き換えるだけで、1パケットも
        #   出ないままプロセスが終わり得た。close() は damp を実際に
        #   送り切ってから送信スレッドを止める
        eng.estop_now("サーバ終了")
        eng.walk.close()                   # 歩行の速度ゼロを送ってから
        eng._drain_saves()                 # ★記録を書き切ってから閉じる
        eng._closing = True
        robot.close()


if __name__ == "__main__":
    main()

