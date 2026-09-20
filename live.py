"""See -> understand -> move. Webcam frames are detected with YOLOv8n (OpenCV DNN),
labels are semantically matched to an animation, and that animation is choreographed
on the four limbs. Usage: python live.py [--dry-run] [--once] [--camera 0] [--segment 3.0]"""
import argparse
import logging
import sys
import time

sys.path.insert(0, "src")
from animation import ANIMATIONS, AnimationPlayer
from semantics import SemanticMatcher
from vision import Camera, YoloDetector, find_camera_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
for noisy in ("lerobot",):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("live")


def describe(detections) -> str:
    return ", ".join(f"{d.label}:{d.confidence:.2f}" for d in detections[:5]) or "nothing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="vision + semantics only, no servos")
    ap.add_argument("--once", action="store_true", help="one frame, then exit")
    ap.add_argument("--camera", type=int, default=None, help="camera index; default = external camera if present")
    ap.add_argument("--segment", type=float, default=3.0, help="seconds to play each chosen animation before re-looking")
    ap.add_argument("--model", default="models/yolov8n.onnx")
    args = ap.parse_args()

    detector = YoloDetector(args.model)
    matcher = SemanticMatcher({name: a.semantics for name, a in ANIMATIONS.items()})
    camera = Camera(args.camera if args.camera is not None else find_camera_index())

    player = None
    bus = None
    if not args.dry_run:
        from spider import SpiderConfig, SpiderController, build_hardware_bus

        config = SpiderConfig()
        bus = build_hardware_bus(config)
        bus.connect(handshake=True)
        ctrl = SpiderController(bus, config)
        ctrl.enable_torque()
        ctrl.set_motion_profile()
        player = AnimationPlayer(ctrl)

    try:
        while True:
            frame = camera.read()
            t0 = time.monotonic()
            detections = detector.detect(frame)
            chosen, scores = matcher.pick(detections)
            top = sorted(scores.items(), key=lambda s: s[1], reverse=True)[:3]
            log.info("saw [%s] in %.0fms -> %s  (%s)", describe(detections), (time.monotonic() - t0) * 1000, chosen,
                     " ".join(f"{n}={s:.2f}" for n, s in top))
            if player is not None:
                player.play(ANIMATIONS[chosen], args.segment)
            elif not args.once:
                time.sleep(args.segment)
            if args.once:
                break
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        camera.release()
        if player is not None:
            player.play(ANIMATIONS["idle"], 0.6)
            player.controller.disable_torque()
            bus.disconnect()


if __name__ == "__main__":
    main()
