"""Real-time see -> understand -> move.

The camera is analysed continuously (YOLOv8n through OpenCV DNN). Labels are semantically
matched to an animation. A debounced switcher confirms a new choice over several frames,
then the player thread crossfades the four limbs into the new animation without stopping.

Usage: python live.py [--show] [--dry-run] [--camera N] [--confirm 4] [--min-conf 0.5] [--save-frame path]
"""
import argparse
import logging
import sys
import time

sys.path.insert(0, "src")
import cv2

from animation import ANIMATIONS
from realtime import AnimationSwitcher, ContinuousPlayer, SwitchDecision
from semantics import SemanticMatcher
from vision import Camera, Detection, YoloDetector, find_camera_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("lerobot").setLevel(logging.WARNING)
log = logging.getLogger("live")

GREEN, ORANGE, WHITE, RED = (0, 200, 0), (0, 160, 255), (255, 255, 255), (0, 0, 255)


def annotate(frame, detections: list[Detection], decision: SwitchDecision, min_conf: float):
    out = frame.copy()
    for d in detections:
        x0, y0, x1, y1 = d.box_xyxy
        color = GREEN if d.confidence >= min_conf else ORANGE
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(out, f"{d.label} {d.confidence:.0%}", (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(out, f"PLAYING: {decision.animation.upper()}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3, WHITE, 3)
    if decision.candidate:
        bar_w = 240
        cv2.rectangle(out, (out.shape[1] - bar_w - 20, 20), (out.shape[1] - 20, 50), WHITE, 2)
        fill = int(bar_w * decision.progress / decision.needed)
        cv2.rectangle(out, (out.shape[1] - bar_w - 20, 20), (out.shape[1] - bar_w - 20 + fill, 50), ORANGE, -1)
        cv2.putText(out, f"-> {decision.candidate}", (out.shape[1] - bar_w - 20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ORANGE, 2)
    if decision.trigger:
        cv2.putText(out, f"trigger: {decision.trigger.label} {decision.trigger.confidence:.0%}", (12, out.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="vision + switching only, no servos")
    ap.add_argument("--show", action="store_true", help="open a window with detections and status")
    ap.add_argument("--camera", type=int, default=None, help="camera index; default = external camera if present")
    ap.add_argument("--confirm", type=int, default=4, help="consecutive agreeing frames before switching")
    ap.add_argument("--min-conf", type=float, default=0.5, help="detection confidence needed to trigger a switch")
    ap.add_argument("--dwell", type=float, default=1.5, help="minimum seconds an animation plays before switching")
    ap.add_argument("--model", default="models/yolov8n.onnx")
    ap.add_argument("--save-frame", default=None, help="also write the latest annotated frame here")
    ap.add_argument("--log-every", type=float, default=2.0, help="seconds between status log lines when nothing changes")
    args = ap.parse_args()

    detector = YoloDetector(args.model, conf_threshold=0.3)
    matcher = SemanticMatcher({name: a.semantics for name, a in ANIMATIONS.items()}, switch_margin=0.05)
    switcher = AnimationSwitcher(confirm_frames=args.confirm, min_confidence=args.min_conf, min_dwell_s=args.dwell)
    camera = Camera(args.camera if args.camera is not None else find_camera_index())

    player = bus = None
    if not args.dry_run:
        from spider import SpiderConfig, SpiderController, build_hardware_bus

        config = SpiderConfig()
        bus = build_hardware_bus(config)
        bus.connect(handshake=True)
        ctrl = SpiderController(bus, config)
        ctrl.enable_torque()
        ctrl.set_motion_profile()
        player = ContinuousPlayer(ctrl, ANIMATIONS, switcher.current)
        player.start()

    last_log = 0.0
    try:
        while True:
            frame = camera.read()
            detections = detector.detect(frame)
            chosen, _ = matcher.pick(detections)
            decision = switcher.update(chosen, detections)
            if decision.switched and player is not None:
                player.set_animation(decision.animation)
            if player is not None and player.error is not None:
                raise RuntimeError("animation player stopped") from player.error

            now = time.monotonic()
            if decision.switched or now - last_log >= args.log_every:
                seen = ", ".join(f"{d.label} {d.confidence:.0%}" for d in detections[:4]) or "nothing"
                cand = f"  candidate {decision.candidate} {decision.progress}/{decision.needed}" if decision.candidate else ""
                log.info("playing %-7s | sees: %s%s", decision.animation, seen, cand)
                last_log = now

            if args.show or args.save_frame:
                shown = annotate(frame, detections, decision, args.min_conf)
                if args.save_frame:
                    cv2.imwrite(args.save_frame, cv2.resize(shown, (960, 540)))
                if args.show:
                    cv2.imshow("hub4", cv2.resize(shown, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if player is not None:
            player.set_animation("idle")
            time.sleep(0.8)
            player.stop()
            player.join(timeout=2)
            player.controller.disable_torque()
            bus.disconnect()


if __name__ == "__main__":
    main()
