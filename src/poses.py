"""Static pose library and smooth trajectory generation for the four-limb hub.

Angles are degrees relative to each limb's neutral (horizontal) position.
Positive = limb swings "up" (toward the hub's top face) for a limb whose
servo is mounted with the default orientation; use ``sign`` in LimbCalibration
to flip a limb that is physically mirrored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TICKS_PER_DEG = 4096 / 360.0
LIMB_NAMES = ("front", "right", "back", "left")


@dataclass(frozen=True)
class Pose:
    name: str
    angles_deg: dict[str, float]
    description: str = ""

    def __post_init__(self) -> None:
        missing = set(LIMB_NAMES) - set(self.angles_deg)
        if missing:
            raise ValueError(f"pose {self.name!r} missing limbs {sorted(missing)}")


def _uniform(name: str, angle_deg: float, description: str) -> Pose:
    return Pose(name, {limb: angle_deg for limb in LIMB_NAMES}, description)


POSES: dict[str, Pose] = {
    p.name: p
    for p in (
        _uniform("neutral", 0.0, "all limbs horizontal"),
        _uniform("spider", -55.0, "all limbs swept down like legs planted"),
        _uniform("spider_crouch", -80.0, "legs tucked further under"),
        _uniform("flower_closed", 75.0, "petals folded up toward the top"),
        _uniform("flower_open", 35.0, "petals half open"),
        _uniform("flower_bloom", -20.0, "petals splayed past horizontal"),
        Pose("alert", {"front": 40.0, "right": -25.0, "back": -25.0, "left": -25.0}, "front limb raised, rest braced"),
        Pose("lean_left", {"front": 0.0, "right": 40.0, "back": 0.0, "left": -40.0}, "tilt toward the left"),
        Pose("lean_right", {"front": 0.0, "right": -40.0, "back": 0.0, "left": 40.0}, "tilt toward the right"),
        Pose("twist", {"front": 45.0, "right": -45.0, "back": 45.0, "left": -45.0}, "alternating up/down"),
    )
}


@dataclass
class LimbCalibration:
    neutral_ticks: int = 2048
    sign: int = 1  # -1 if this limb's servo is mirrored
    min_deg: float = -95.0
    max_deg: float = 95.0

    def deg_to_ticks(self, angle_deg: float) -> int:
        clamped = max(self.min_deg, min(self.max_deg, angle_deg))
        return int(round(self.neutral_ticks + self.sign * clamped * TICKS_PER_DEG))


@dataclass
class HubCalibration:
    limbs: dict[str, LimbCalibration] = field(
        default_factory=lambda: {limb: LimbCalibration() for limb in LIMB_NAMES}
    )

    def pose_to_targets(self, pose: Pose) -> dict[str, int]:
        return {limb: self.limbs[limb].deg_to_ticks(deg) for limb, deg in pose.angles_deg.items()}


def minimum_jerk(progress: float) -> float:
    """Smooth 0->1 blend with zero velocity and acceleration at both ends."""
    s = max(0.0, min(1.0, progress))
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def interpolate(start: dict[str, int], end: dict[str, int], progress: float) -> dict[str, int]:
    blend = minimum_jerk(progress)
    return {limb: int(round(start[limb] + (end[limb] - start[limb]) * blend)) for limb in end}
