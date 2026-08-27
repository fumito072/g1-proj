#!/usr/bin/env python3
"""LocoClient の GET 系APIだけを叩いて現在値を読む。**設定は一切しない。**

立ち高さ(SetStandHeight)の単位・現在値・FSM・バランスモードを確認する。
UIに「立ち高さスライダー」を出す前に、実機が何を返すかを知るためのもの。

  python3 real/probe_loco.py --iface enp46s0
"""
import argparse
import json
import sys

GETS = [
    (7001, "FSM ID"),
    (7002, "FSM MODE"),
    (7003, "バランスモード"),
    (7004, "スイング高さ"),
    (7005, "立ち高さ"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp46s0")
    a = ap.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    ChannelFactoryInitialize(0, a.iface) if a.iface else ChannelFactoryInitialize(0)
    c = LocoClient()
    c.SetTimeout(5.0)
    c.Init()
    print("読み取りのみ(設定は送りません)")
    for api, name in GETS:
        try:
            c._RegistApi(api, 0)
        except Exception:                          # noqa: BLE001
            pass
        try:
            code, data = c._Call(api, "{}")
            v = data
            try:
                v = json.loads(data) if isinstance(data, str) else data
            except Exception:                      # noqa: BLE001
                pass
            print(f"  {name:14s} (api {api}): code={code}  data={v}")
        except Exception as e:                     # noqa: BLE001
            print(f"  {name:14s} (api {api}): 例外 {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
