"""Live demo: sweep the gaze in a circle and the four limbs follow. Ctrl-C safe."""
import logging
import math
import sys
import time

sys.path.insert(0, "src")
from spider import SpiderConfig, SpiderController, build_hardware_bus

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(cycles: int = 2, steps: int = 16, dwell_s: float = 0.15) -> None:
    config = SpiderConfig()
    bus = build_hardware_bus(config)
    bus.connect(handshake=True)
    ctrl = SpiderController(bus, config)
    try:
        ctrl.enable_torque()
        neutral = ctrl.go_neutral()
        print("neutral settled:", ctrl.wait_settled(neutral))
        for step in range(cycles * steps):
            angle = 2 * math.pi * step / steps
            targets = ctrl.look_at(math.cos(angle), math.sin(angle))
            time.sleep(dwell_s)
            pos = ctrl.read_positions()
            print(f"gaze {math.degrees(angle):5.0f}°  " + "  ".join(f"{n}:{pos[n]:4d}/{targets[n]:4d}" for n in ctrl.limb_names))
        ctrl.go_neutral()
        time.sleep(0.5)
    finally:
        ctrl.disable_torque()
        bus.disconnect()


if __name__ == "__main__":
    main()
