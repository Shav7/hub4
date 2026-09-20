"""Move the hub through named poses. Usage: python pose.py spider flower_closed neutral [--duration 0.5] [--hold 1.0] [--list]"""
import argparse
import logging
import sys
import time

sys.path.insert(0, "src")
from poses import POSES, HubCalibration
from spider import SpiderConfig, SpiderController, build_hardware_bus

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("poses", nargs="*", help="pose names, in order")
    ap.add_argument("--duration", type=float, default=0.5, help="seconds per transition")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds to hold each pose")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.poses:
        for p in POSES.values():
            print(f"{p.name:15s} {p.description}")
        return
    unknown = [n for n in args.poses if n not in POSES]
    if unknown:
        sys.exit(f"unknown poses: {unknown}. Use --list.")

    config = SpiderConfig()
    calib = HubCalibration()
    bus = build_hardware_bus(config)
    bus.connect(handshake=True)
    ctrl = SpiderController(bus, config)
    try:
        ctrl.enable_torque()
        ctrl.set_motion_profile()
        for name in args.poses:
            targets = calib.pose_to_targets(POSES[name])
            t0 = time.monotonic()
            settled = ctrl.move_smooth(targets, duration_s=args.duration)
            pos = ctrl.read_positions()
            err = max(abs(pos[k] - v) for k, v in targets.items())
            print(f"{name:15s} {time.monotonic()-t0:.2f}s settled={settled} max_err={err} ticks")
            time.sleep(args.hold)
        ctrl.move_smooth(calib.pose_to_targets(POSES["neutral"]), duration_s=args.duration)
    finally:
        ctrl.disable_torque()
        bus.disconnect()


if __name__ == "__main__":
    main()
