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
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from run_fsm import (Policy, ObsBuilder, quat_to_mat, ACTION_SCALE,   # noqa: E402
                     CONTROL_HZ, TILT_LIMIT_DEG, _yaw_of)

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
    "arm", "start", "next", "place_sim", "mode", "goto_start",
    "ground_check", "ramp", "blend", "memo", "newsession",
    "stand_user", "fsm_read", "user_run",
    "mode_zero", "mode_damp", "mode_stand", "mode_walk",
    "custom", "run_task",
    "assist", "yawalign", "stopframe",         # ← 2026-08-27に入れ忘れていた
)
REC_SAVE_EVERY = 100              # 途中で落ちても失わないよう逐次保存


# UIのプルダウンに出す短い注記。現場で「どれが本命でどれがアンカーか」を
# 名前だけで思い出せないと、比較の設計ごと崩れる(実機セッション手順 §5)。
# ★ここに無い方策は注記なしで出る。使用可否の判断は手順書が唯一の真実。
PATTERN_NOTES = {
    # --- 着座(これまで使った全部を出す。使用可否は operator が決める)
    "sit_up_dp4_r2":          "★新 deepと同じ参照を摩擦較正シムで18万step学習",
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


class Engine:
    """FSMエンジン。stateはUIへそのまま出す"""

    def __init__(self, robot, is_sim):
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
        self.sel = {"climb": "climb_slow_r2", "turn": "turn_wide_r2",
                    "sit": "sit_up_rc_r2"}   # 実機較正シム再学習版を既定に(2026-08-23)
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
        self.stand = None
        # simは待機中の物理を凍結して置く(実機では操作者が支える)。
        # 凍結しないと、操作を始める前にモックが勝手に崩れる
        self.sim_frozen = bool(is_sim)
        self.busy = False                # ワーカーが実機RPC/方策読込を実行中
        self.busy_what = ""
        self._estop_pending = None       # estop_now が立てる。ループが後始末
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
        if self.fsm.startswith("STD:") or self.fsm == "DAMP":
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
                if not self._monitoring():
                    continue
                why = self._safety()
                if why:
                    self.estop_now(why)
            except Exception:                      # noqa: BLE001
                pass

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

    def _do_standard(self, name):
        self.armed = False
        self.sim_frozen = False
        self._clear_estop(f"標準モード {name}")
        ok = self.robot.standard_mode(name)
        self.fsm = f"STD:{name}" if ok else "DAMP"
        self.log(f"標準モード {name}" + ("" if ok else "(失敗→要確認)"))

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
        elif cmd == "stopframe":
            try:
                self.stop_frame = max(0, int(float(arg)))
                self.log(f"打ち切りコマ数: "
                         + (f"{self.stop_frame}コマ({self.stop_frame / CONTROL_HZ:.2f}秒)で止めます"
                            if self.stop_frame else "最後まで走ります(既定)"))
            except Exception:                      # noqa: BLE001
                self.log(f"★打ち切りコマ数が不正: {arg}")
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
                if self.step_mode and self.stand is not None:
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
            if self.t > LOOP_DT_WINDOW and len(recent) >= LOOP_DT_WINDOW:
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
            target = ref_eff + a * ACTION_SCALE * w
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
            q, dq, quat, gyro, tau = self.robot.state()
            obs = self.obs_b.build(pol, pol.n - 1, q, dq, quat, gyro)
            a = pol.act(obs)
            if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(a))):
                self._nan_frames += 1
                if self._nan_frames >= 3:
                    self.estop_now("保持中の観測/方策出力のNaNが3コマ連続")
                    self._estop_bookkeeping()
                return
            self._nan_frames = 0
            self.obs_b.last_cmd = a.copy()
            self._set_target(pol.ref_q[pol.n - 1] + a * ACTION_SCALE,
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
        if pw.get("ac") == 0:
            ng.append("★ACアダプタが抜けています — バッテリー駆動では"
                      f"CPUが{pw.get('mhz', 0):.0f}MHzまでしか上がらず、"
                      "制御ループが19〜22Hzに落ちて必ず前に倒れます"
                      "(2026-08-26/27に計5本の転倒実績)")
        elif pw.get("mhz") and pw["mhz"] < 2000:
            warn.append(f"CPUの実クロックが低い(中央{pw['mhz']:.0f}MHz)"
                        f" — ガバナ={pw.get('gov', '?')}")
        h = self.robot.health_detail()
        lp = self._loop_stat()
        if self.busy:
            ng.append(f"処理中({self.busy_what})")
        if self.fsm in ("RUNNING", "MOVING"):
            ng.append(f"実行中({self.fsm})")
        if h.get("estop_latched"):
            ng.append("E-STOPラッチ中 — [⤓ 開始姿勢へ]で解除")
        if not self.robot.healthy():
            ng.append("受信途絶または送信停止")
        if lp["hz"] and lp["hz"] < 45.0:
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
                out["mhz"] = v[len(v) // 2] / 1000.0
            g = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            out["gov"] = g.read_text().strip()
        except Exception:                          # noqa: BLE001
            pass
        _cache[0], _cache[1] = now, out
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
        self._nan_frames = 0
        self._early_track = 0.0
        self._early_track_j = ""
        self._dt_hist = []
        self._dt_ms = 0.0
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
            })
            del self.run_stats[:-200]
        except Exception as e:                     # noqa: BLE001
            self.log(f"(走行統計の記録に失敗: {e})")

    def _end_phase(self):
        name, pol = self.phases[self.phase_i]
        self._phase_done = True
        self._save_rec(final=True)
        self._push_run_stat("完走" if self.t >= pol.n else f"{self.t}コマで打切")
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
        self.hold_pol = pol
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
                "user_ctrl": bool(getattr(self.robot, "_use_user_topic", False)),
                "logs": list(self.logs[-25:]),
                **extra,
            }


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8">
<title>G1 Cockpit</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
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
  <div class="lbl" style="margin-top:6px;text-align:center">スペースキーでも止まります</div>
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
   <button onclick="cmd('damp')">damp(方策側)</button>
  </div>
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
  <div class="row">打ち切り
   <select id="sel_stop" onchange="cmd('stopframe',this.value)">
    <option value="0">最後まで(395コマ / 7.9秒)</option>
    <option value="146">146コマ(2.92秒) — 背もたれへ反る前で止める</option>
    <option value="115">115コマ(2.30秒) — 座り終わった直後</option>
   </select></div>
  <div class="lbl">参照npzは書き換えず<b>走行を途中で止めるだけ</b>。
  観測には正規化時刻と先読みが入るので、配列を切ると別物になる。
  146〜242コマが「背もたれへ18度反る」区間(2026-08-27解析)</div>
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
  <div class="row">ヨー合わせ
   <select id="sel_yaw" onchange="cmd('yawalign',this.value)">
    <option value="on">ON(既定)</option>
    <option value="off">OFF(14:13以前と同じ)</option></select>
   <span class="lbl">膝の左右差が 1.23&rarr;0.11 に直った分</span></div>
  <div class="row">進行
   <select id="sel_mode" onchange="cmd('mode',this.value)">
    <option value="step">ステップ(各フェーズ前に確認)</option>
    <option value="auto">自動</option></select></div>
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

<script>
let S={sel:{}};
function cmd(c,a){fetch('/cmd?c='+c+(a?('&a='+encodeURIComponent(a)):''),{method:'POST'})}
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
 ['kdiff','膝左右差'],['ahip','左股a'],['asat','腕飽和%'],['dt_med','周期ms'],['dt_max','周期最大']];
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
   return '<td style="'+st+'">'+(v==null?'-':v)+'</td>';
  }).join('')+'</tr>';
 }
 e.innerHTML=h;
 const n=rs.length, ok=rs.filter(r=>r.tend>=15&&r.tend<=24).length;
 document.getElementById('statsum').innerHTML=
  'この起動から '+n+'本 / 座れた(終端傾き15〜24&deg;) <b style="color:var(--ok)">'+ok+'本</b>'+
  '　参照は終端18.0&deg; / シム 最大22.1&deg;・終端18.5&deg;'+
  '<br>左股a: 良+0.05〜+0.17 / 浅い回+0.78〜+0.81　膝左右差: シム0.11　周期: 正常20.0ms';
}
function fill(id,arr,cur,skip,notes){const e=document.getElementById(id);
 if(!e||e.dataset.done)return; e.dataset.done=1;
 const items=skip?['(skip)',...arr]:arr;
 e.innerHTML=items.map(x=>{const n=(notes||{})[x];
  return `<option value="${x}" ${x===cur?'selected':''}>${x}${n?'  —  '+n:''}</option>`
 }).join('')}
async function tick(){
 let d;try{d=await(await fetch('/state')).json()}catch(e){return}
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
 drawPose(d.pose,d); drawCrouch(d); drawStats(d.run_stats);
 const pw=document.getElementById('pwr'), p=d.power||{}, ac=(p.ac===0);
 pw.innerHTML='電源 <b style="color:'+(ac?'var(--bad)':'var(--ok)')+'">'+
  (ac?'★AC未接続':'AC接続')+'</b>'+
  (p.mhz?('　CPU <b style="color:'+(p.mhz<2000?'var(--warn)':'var(--ok)')+'">'+p.mhz.toFixed(0)+'MHz</b>'):'')+
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
}
setInterval(tick,200);tick();
document.addEventListener('keydown',e=>{
 if(e.key===' '&&e.target.tagName!=='INPUT'){e.preventDefault();cmd('estop')}});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None
    patterns = None

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/state":
            full = "full" in parse_qs(urlparse(self.path).query)
            d = self.engine.snapshot(full=full)
            d["patterns"] = self.patterns
            d["notes"] = PATTERN_NOTES
            d["warn"] = PATTERN_WARN
            self._send(json.dumps(d).encode(), "application/json")
        elif p == "/frame.jpg" and self.engine.is_sim:
            self._send(self.engine.robot.render_jpeg(), "image/jpeg")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")

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
    eng = Engine(robot, a.sim)
    Handler.engine = eng
    Handler.patterns = list_patterns()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"コックピット: http://localhost:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n中断されました")
    finally:
        # ★set_damp だけでは送信バッファを書き換えるだけで、1パケットも
        #   出ないままプロセスが終わり得た。close() は damp を実際に
        #   送り切ってから送信スレッドを止める
        eng.estop_now("サーバ終了")
        eng._drain_saves()                 # ★記録を書き切ってから閉じる
        eng._closing = True
        robot.close()


if __name__ == "__main__":
    main()

