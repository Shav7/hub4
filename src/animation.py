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


ANIMATIONS: dict[str, Animation] = {
    a.name: a
    for a in (
        Animation(
            "idle",
            (Keyframe(_uniform(-5.0), 1.6), Keyframe(_uniform(5.0), 1.6)),
            SemanticProfile("idle", {}),
            limb_phase_s=0.2,
        ),
        Animation(
            "spider",  # legs planted, alternating pairs scuttle
            (
                Keyframe(_pairs(-60.0, -35.0), 0.28),
                Keyframe(_pairs(-35.0, -60.0), 0.28),
            ),
            SemanticProfile("spider", {"creature": 1.0, "animal": 0.9, "predator": 0.8, "threat": 0.6, "small": 0.4, "sharp": 0.3}),
        ),
        Animation(
            "flower",  # petals open from folded, breathe, close
            (
                Keyframe(_uniform(70.0), 0.6),
                Keyframe(_uniform(0.0), 0.9),
                Keyframe(_uniform(-15.0), 0.7),
                Keyframe(_uniform(-5.0), 0.7),
                Keyframe(_uniform(-15.0), 0.7),
                Keyframe(_uniform(60.0), 0.9),
            ),
            SemanticProfile("flower", {"flower": 1.0, "plant": 0.9, "nature": 0.7, "calm": 0.6, "growth": 0.5, "open": 0.4}),
            limb_phase_s=0.15,
        ),
        Animation(
            "greet",  # rises to attention, waves the front limb
            (
                Keyframe({"front": 50.0, "right": -20.0, "back": -20.0, "left": -20.0}, 0.45),
                Keyframe({"front": 20.0, "right": -20.0, "back": -20.0, "left": -20.0}, 0.3),
                Keyframe({"front": 50.0, "right": -20.0, "back": -20.0, "left": -20.0}, 0.3),
                Keyframe({"front": 20.0, "right": -20.0, "back": -20.0, "left": -20.0}, 0.3),
            ),
            SemanticProfile("greet", {"human": 1.0, "social": 0.8, "face": 0.6, "attention": 0.5, "friendly": 0.5}),
        ),
        Animation(
            "wave",  # a single crest travelling around the ring
            (Keyframe(_uniform(45.0), 0.35), Keyframe(_uniform(-30.0), 0.55)),
            SemanticProfile("wave", {"motion": 1.0, "playful": 0.8, "toy": 0.6, "round": 0.4, "flight": 0.4, "wind": 0.4}),
            limb_phase_s=0.22,
        ),
        Animation(
            "rest",  # curl up and stay
            (Keyframe(_uniform(80.0), 1.2),),
            SemanticProfile("rest", {"rest": 1.0, "calm": 0.7, "furniture": 0.5, "static": 0.5, "soft": 0.4}),
            loop=False,
        ),
        Animation(
            "scan",  # slow look-around: each limb probes in turn
            (Keyframe(_uniform(25.0), 0.5), Keyframe(_uniform(-10.0), 0.5)),
            SemanticProfile("scan", {"device": 1.0, "screen": 0.7, "work": 0.6, "handheld": 0.4, "object": 0.3}),
            limb_phase_s=0.5,
        ),
    )
}


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
