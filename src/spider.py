"""Four-limb 'spider' controller for daisy-chained Feetech STS3215 servos.

Each limb sits at a fixed angle around a central hub. A normalized 2D gaze
vector (from a vision source) tilts each limb in proportion to how much that
limb points toward the target.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

TICKS_PER_REV = 4096
DEFAULT_PORT = "/dev/tty.usbmodem5A7C1170651"


@dataclass(frozen=True)
class Limb:
    name: str
    motor_id: int
    azimuth_deg: float  # direction the limb points, around the hub


@dataclass
class SpiderConfig:
    port: str = DEFAULT_PORT
    limbs: tuple[Limb, ...] = (
        Limb("front", 1, 0.0),
        Limb("right", 2, 90.0),
        Limb("back", 3, 180.0),
        Limb("left", 4, 270.0),
    )
    center_ticks: int = 2048  # neutral pose for every limb
    max_tilt_ticks: int = 300  # ~26 deg of swing at full gaze
    settle_timeout_s: float = 1.5
    settle_tolerance_ticks: int = 20


class MotorBus(Protocol):
    """Subset of lerobot's FeetechMotorsBus that the controller relies on."""

    def read(self, data_name: str, motor: str, *, normalize: bool = True) -> int: ...
    def write(self, data_name: str, motor: str, value: int, *, normalize: bool = True, num_retry: int = 0) -> None: ...
    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None: ...
    def sync_read(self, data_name: str, *, normalize: bool = True) -> dict[str, int]: ...


def clamp_gaze(gaze_x: float, gaze_y: float) -> tuple[float, float]:
    """Clamp a gaze vector to the unit disc; reject non-finite input."""
    if not (math.isfinite(gaze_x) and math.isfinite(gaze_y)):
        raise ValueError(f"gaze must be finite, got ({gaze_x}, {gaze_y})")
    magnitude = math.hypot(gaze_x, gaze_y)
    if magnitude <= 1.0:
        return gaze_x, gaze_y
    return gaze_x / magnitude, gaze_y / magnitude


def gaze_to_targets(config: SpiderConfig, gaze_x: float, gaze_y: float) -> dict[str, int]:
    """Map a gaze vector to a goal tick per limb.

    A limb pointing straight at the target swings +max_tilt; the opposite limb
    swings -max_tilt; perpendicular limbs stay at center.
    """
    gaze_x, gaze_y = clamp_gaze(gaze_x, gaze_y)
    targets: dict[str, int] = {}
    for limb in config.limbs:
        azimuth_rad = math.radians(limb.azimuth_deg)
        alignment = math.cos(azimuth_rad) * gaze_x + math.sin(azimuth_rad) * gaze_y
        offset = round(alignment * config.max_tilt_ticks)
        targets[limb.name] = _clamp_ticks(config.center_ticks + offset)
    return targets


def _clamp_ticks(ticks: int) -> int:
    return max(0, min(TICKS_PER_REV - 1, ticks))


class SpiderController:
    def __init__(self, bus: MotorBus, config: SpiderConfig | None = None) -> None:
        self.bus = bus
        self.config = config or SpiderConfig()
        self._torque_on = False

    @property
    def limb_names(self) -> list[str]:
        return [limb.name for limb in self.config.limbs]

    def enable_torque(self) -> None:
        for name in self.limb_names:
            self.bus.write("Torque_Enable", name, 1, num_retry=3)
        self._torque_on = True
        logger.info("torque enabled on %s", self.limb_names)

    def disable_torque(self) -> None:
        for name in self.limb_names:
            self.bus.write("Torque_Enable", name, 0, num_retry=5)
        self._torque_on = False
        logger.info("torque disabled")

    def read_positions(self) -> dict[str, int]:
        return self.bus.sync_read("Present_Position", normalize=False)

    def move_to(self, targets: dict[str, int]) -> None:
        if not self._torque_on:
            raise RuntimeError("enable_torque() before moving")
        self.bus.sync_write("Goal_Position", targets, normalize=False)
        logger.debug("goal -> %s", targets)

    def wait_settled(self, targets: dict[str, int]) -> bool:
        """Poll until every limb is within tolerance, or timeout. Returns True if settled."""
        deadline = time.monotonic() + self.config.settle_timeout_s
        while time.monotonic() < deadline:
            positions = self.read_positions()
            if all(
                abs(positions[name] - goal) <= self.config.settle_tolerance_ticks
                for name, goal in targets.items()
            ):
                return True
            time.sleep(0.02)
        logger.warning("limbs did not settle within %.1fs: %s", self.config.settle_timeout_s, self.read_positions())
        return False

    def set_motion_profile(self, acceleration: int = 0, velocity: int = 0) -> None:
        """Servo-side limits: 0 means 'no limit' so the servo tracks our software trajectory tightly."""
        for name in self.limb_names:
            self.bus.write("Acceleration", name, acceleration, num_retry=3)
            self.bus.write("Goal_Velocity", name, velocity, normalize=False, num_retry=3)

    def move_smooth(self, targets: dict[str, int], duration_s: float = 0.6, rate_hz: float = 100.0) -> bool:
        """Minimum-jerk interpolate from current positions to targets, streaming goals at rate_hz."""
        if duration_s <= 0 or rate_hz <= 0:
            raise ValueError("duration_s and rate_hz must be positive")
        from poses import interpolate

        start = self.read_positions()
        t0 = time.monotonic()
        period = 1.0 / rate_hz
        while True:
            elapsed = time.monotonic() - t0
            progress = elapsed / duration_s
            self.move_to(interpolate(start, targets, progress))
            if progress >= 1.0:
                break
            time.sleep(max(0.0, period - ((time.monotonic() - t0) % period)))
        return self.wait_settled(targets)

    def look_at(self, gaze_x: float, gaze_y: float) -> dict[str, int]:
        targets = gaze_to_targets(self.config, gaze_x, gaze_y)
        self.move_to(targets)
        return targets

    def go_neutral(self) -> dict[str, int]:
        return self.look_at(0.0, 0.0)


def build_hardware_bus(config: SpiderConfig):
    """Construct a real FeetechMotorsBus for the configured limbs (imports lerobot lazily)."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {limb.name: Motor(limb.motor_id, "sts3215", MotorNormMode.RANGE_0_100) for limb in config.limbs}
    return FeetechMotorsBus(port=config.port, motors=motors)
