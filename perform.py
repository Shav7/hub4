"""Perform choreographed moods on the hardware without a camera.
Usage: python perform.py [mood ...] [--seconds 10] [--seed 1]   (default: cycles every mood)"""
import argparse
import logging
import sys
import time

sys.path.insert(0, "src")
from animation import ANIMATIONS
from choreography import MOODS, Choreographer
from realtime import ContinuousPlayer
from spider import SpiderConfig, SpiderController, build_hardware_bus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("lerobot").setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("moods", nargs="*", default=list(MOODS))
    ap.add_argument("--seconds", type=float, default=10.0, help="time to spend in each mood")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    unknown = [m for m in args.moods if m not in MOODS]
    if unknown:
        sys.exit(f"unknown moods {unknown}; choose from {list(MOODS)}")

    config = SpiderConfig()
    bus = build_hardware_bus(config)
    bus.connect(handshake=True)
    ctrl = SpiderController(bus, config)
    ctrl.enable_torque()
    ctrl.set_motion_profile()
    player = ContinuousPlayer(ctrl, ANIMATIONS, "idle")
    player.start()
    choreo = Choreographer(seed=args.seed)
    try:
        for mood in args.moods:
            choreo.set_mood(mood)
            mood_end = time.monotonic() + args.seconds
            print(f"\n=== {mood.upper()} ===")
            while time.monotonic() < mood_end:
                clip = choreo.next_clip()
                player.set_animation(clip.name, clip.speed, clip.offset)
                print(f"  {clip.name:14s} {clip.scale:5s} x{clip.speed:.2f}  {clip.duration_s:.1f}s")
                time.sleep(min(clip.duration_s, max(0.0, mood_end - time.monotonic())))
                if player.error:
                    raise RuntimeError("player died") from player.error
    except KeyboardInterrupt:
        pass
    finally:
        player.set_animation("idle")
        time.sleep(0.8)
        player.stop()
        player.join(2)
        ctrl.disable_torque()
        bus.disconnect()


if __name__ == "__main__":
    main()
