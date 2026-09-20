"""Keyframed animations with per-limb choreography, and a player that streams them to the hub."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from poses import LIMB_NAMES, HubCalibration, Pose, minimum_jerk
from semantics import SemanticProfile
from spider import SpiderController

logger = logging.getLogger(__name__)

# Order around the hub, used for ripple choreography.
RING = ("front", "right", "back", "left")


@dataclass(frozen=True)
class Keyframe:
    angles_deg: dict[str, float]
    duration_s: float  # time to travel *into* this keyframe from the previous one

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("keyframe duration must be positive")
        if set(self.angles_deg) != set(LIMB_NAMES):
            raise ValueError("keyframe must specify every limb")


@dataclass(frozen=True)
class Animation:
    name: str
    keyframes: tuple[Keyframe, ...]
    semantics: SemanticProfile
    loop: bool = True
    limb_phase_s: float = 0.0  # ripple: each successive limb around the ring lags by this much
    scale: str = "macro"  # "macro" = full-body absolute pose; "micro" = small offsets layered on a mood's rest pose

    @property
    def period_s(self) -> float:
        return sum(k.duration_s for k in self.keyframes)

    def angles_at(self, t: float) -> dict[str, float]:
        """Per-limb angle at time t, applying the ring phase lag for choreography."""
        return {
            limb: self._single_limb_angle(limb, t - RING.index(limb) * self.limb_phase_s)
            for limb in LIMB_NAMES
        }

    def _single_limb_angle(self, limb: str, t: float) -> float:
        period = self.period_s
        if self.loop:
            t = t % period
        elif t >= period:
            return self.keyframes[-1].angles_deg[limb]
        if t < 0:  # before this limb has started (phase lag): hold the last keyframe's value
            return self.keyframes[-1].angles_deg[limb]
        elapsed = 0.0
        prev = self.keyframes[-1].angles_deg[limb]
        for key in self.keyframes:
            if t < elapsed + key.duration_s:
                blend = minimum_jerk((t - elapsed) / key.duration_s)
                return prev + (key.angles_deg[limb] - prev) * blend
            elapsed += key.duration_s
            prev = key.angles_deg[limb]
        return prev


def _uniform(angle: float) -> dict[str, float]:
    return {limb: angle for limb in LIMB_NAMES}


def _pairs(a: float, b: float) -> dict[str, float]:
    """Opposite limbs share an angle: front/back = a, right/left = b."""
    return {"front": a, "back": a, "right": b, "left": b}


def _sides(a: float, b: float) -> dict[str, float]:
    """front/back = a, right = b, left = -b (a lean)."""
    return {"front": a, "back": a, "right": b, "left": -b}


def _one(limb: str, angle: float, rest: float = 0.0) -> dict[str, float]:
    return {l: (angle if l == limb else rest) for l in LIMB_NAMES}


_NONE = SemanticProfile("", {})


def _macro(name, keyframes, semantics=_NONE, *, loop=True, phase=0.0):
    return Animation(name, tuple(Keyframe(a, d) for a, d in keyframes), semantics, loop=loop, limb_phase_s=phase, scale="macro")


def _micro(name, keyframes, *, loop=True, phase=0.0):
    return Animation(name, tuple(Keyframe(a, d) for a, d in keyframes), _NONE, loop=loop, limb_phase_s=phase, scale="micro")


ANIMATIONS: dict[str, Animation] = {
    a.name: a
    for a in (
        # ---------- mood anchors (carry the semantics) ----------
        _macro("idle", [(_uniform(-5), 1.6), (_uniform(5), 1.6)], SemanticProfile("idle", {}), phase=0.2),
        _macro("spider", [(_pairs(-60, -35), 0.28), (_pairs(-35, -60), 0.28)],
               SemanticProfile("spider", {"creature": 1.0, "animal": 0.9, "predator": 0.8, "threat": 0.6, "small": 0.4, "sharp": 0.3})),
        _macro("flower", [(_uniform(70), 0.6), (_uniform(0), 0.9), (_uniform(-15), 0.7), (_uniform(-5), 0.7), (_uniform(-15), 0.7), (_uniform(60), 0.9)],
               SemanticProfile("flower", {"flower": 1.0, "plant": 0.9, "nature": 0.7, "calm": 0.6, "growth": 0.5, "open": 0.4}), phase=0.15),
        _macro("greet", [({"front": 50, "right": -35, "back": -10, "left": -5}, 0.35), ({"front": 15, "right": -20, "back": -30, "left": -20}, 0.35),
                         ({"front": 50, "right": -5, "back": -10, "left": -35}, 0.35), ({"front": 15, "right": -20, "back": -30, "left": -20}, 0.35)],
               SemanticProfile("greet", {"human": 1.0, "social": 0.8, "face": 0.6, "attention": 0.5, "friendly": 0.5})),
        _macro("wave", [(_uniform(45), 0.35), (_uniform(-30), 0.55)],
               SemanticProfile("wave", {"motion": 1.0, "playful": 0.8, "toy": 0.6, "round": 0.4, "flight": 0.4, "wind": 0.4}), phase=0.22),
        _macro("rest", [(_uniform(80), 1.2)], SemanticProfile("rest", {"rest": 1.0, "calm": 0.7, "furniture": 0.5, "static": 0.5, "soft": 0.4}), loop=False),
        _macro("scan", [(_uniform(25), 0.5), (_uniform(-10), 0.5)],
               SemanticProfile("scan", {"device": 1.0, "screen": 0.7, "work": 0.6, "handheld": 0.4, "object": 0.3}), phase=0.5),
        # ---------- more macros ----------
        _macro("stretch", [(_uniform(-10), 0.8), (_uniform(75), 1.2), (_uniform(-50), 1.0)], phase=0.1),
        _macro("spin", [(_uniform(50), 0.3), (_uniform(-40), 0.3)], phase=0.15),
        _macro("bow", [({"front": -70, "right": -10, "back": 40, "left": -10}, 0.6), ({"front": -70, "right": -10, "back": 40, "left": -10}, 0.5), (_uniform(0), 0.7)]),
        _macro("pounce", [(_uniform(-75), 0.5), (_uniform(45), 0.25), (_uniform(-55), 0.5)]),
        _macro("prowl", [(_pairs(-65, -25), 0.5), (_pairs(-25, -65), 0.5)]),
        _macro("bloom_hold", [(_uniform(70), 1.2), (_uniform(-25), 1.2), (_uniform(-25), 1.5)], phase=0.2),
        _macro("sway_bloom", [({"front": -15, "right": 10, "back": -15, "left": -40}, 0.9), ({"front": -15, "right": -40, "back": -15, "left": 10}, 0.9)]),
        _macro("peek", [(_uniform(-20), 0.5), (_uniform(55), 0.4), (_uniform(-20), 0.3)], phase=0.4),
        _macro("sweep", [(_uniform(35), 0.7), (_uniform(-25), 0.7)], phase=0.35),
        _macro("curl_breathe", [(_uniform(80), 1.5), (_uniform(68), 1.5)]),
        # ---------- entry gestures (play once when a mood begins) ----------
        _macro("startle", [(_uniform(60), 0.2), (_uniform(-10), 0.5), (_uniform(5), 0.4)], loop=False),
        _macro("crouch_snap", [(_uniform(-75), 0.25), (_pairs(-60, -35), 0.4)], loop=False),
        _macro("unfurl", [(_uniform(75), 0.4), (_uniform(-10), 1.2)], loop=False, phase=0.2),
        _macro("alert_up", [(_uniform(30), 0.3), (_uniform(15), 0.4)], loop=False),
        _macro("settle", [(_uniform(80), 1.2)], loop=False),
        # ---------- micro (offsets around a mood's rest pose) ----------
        _micro("breathe", [(_uniform(-4), 1.8), (_uniform(4), 1.8)], phase=0.15),
        _micro("twitch", [(_one("front", 8), 0.12), (_uniform(0), 0.25), (_uniform(0), 0.8)]),
        _micro("shiver", [(_uniform(-3), 0.08), (_uniform(3), 0.08)]),
        _micro("shiver_soft", [(_uniform(-2), 0.15), (_uniform(2), 0.15)]),
        _micro("tap", [(_one("front", -12), 0.15), (_uniform(0), 0.15), (_one("front", -12), 0.15), (_uniform(0), 0.6)]),
        _micro("tilt", [(_sides(0, 12), 1.2), (_sides(0, -12), 1.2)]),
        _micro("nod", [({"front": 10, "back": -10, "right": 0, "left": 0}, 0.4), ({"front": -10, "back": 10, "right": 0, "left": 0}, 0.4)]),
        _micro("sway", [(_sides(0, 10), 0.8), (_sides(0, -10), 0.8)]),
        _micro("shrug", [({"front": 0, "back": 0, "right": 15, "left": 15}, 0.3), (_uniform(0), 0.5), (_uniform(0), 0.5)]),
        _micro("ripple_micro", [(_uniform(10), 0.3), (_uniform(-10), 0.3)], phase=0.15),
        _micro("petal_flutter", [(_uniform(3), 0.2), (_uniform(-3), 0.2)], phase=0.1),
        _micro("scuttle_step", [(_pairs(-8, 12), 0.15), (_pairs(12, -8), 0.15)]),
    )
}

MACROS = tuple(n for n, a in ANIMATIONS.items() if a.scale == "macro")
MICROS = tuple(n for n, a in ANIMATIONS.items() if a.scale == "micro")


@dataclass
class AnimationPlayer:
    controller: SpiderController
    calibration: HubCalibration = field(default_factory=HubCalibration)
    rate_hz: float = 60.0
    blend_in_s: float = 0.4  # smooth hand-off from wherever the limbs are into the new animation

    def play(self, animation: Animation, duration_s: float) -> None:
        logger.info("play %s for %.1fs", animation.name, duration_s)
        start_ticks = self.controller.read_positions()
        t0 = time.monotonic()
        period = 1.0 / self.rate_hz
        while True:
            t = time.monotonic() - t0
            if t >= duration_s:
                break
            targets = self.calibration.pose_to_targets(Pose(animation.name, animation.angles_at(t)))
            if t < self.blend_in_s:
                blend = minimum_jerk(t / self.blend_in_s)
                targets = {k: int(round(start_ticks[k] + (v - start_ticks[k]) * blend)) for k, v in targets.items()}
            self.controller.move_to(targets)
            time.sleep(max(0.0, period - ((time.monotonic() - t0) % period)))
