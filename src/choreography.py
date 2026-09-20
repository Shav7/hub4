"""Choreographer: turns a mood into an expressive, non-repeating stream of clips.

Each mood has a rest pose, an entry gesture, a set of macro (full-body) animations and a
set of micro (subtle) animations. The choreographer alternates macros and micros with
randomised tempo and hold counts, never repeats the same macro back to back when it can
avoid it, and plays the entry gesture whenever the mood changes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from animation import ANIMATIONS, Animation
from poses import LIMB_NAMES


def _u(a: float) -> dict[str, float]:
    return {l: a for l in LIMB_NAMES}


@dataclass(frozen=True)
class Mood:
    name: str
    rest_pose: dict[str, float]  # micro clips are offsets from this
    macros: tuple[str, ...]
    micros: tuple[str, ...]
    entry: str | None = None
    tempo: float = 1.0  # base speed multiplier
    micro_bias: float = 0.6  # probability of a micro burst after a macro


MOODS: dict[str, Mood] = {
    m.name: m
    for m in (
        Mood("idle", _u(0), ("idle", "stretch", "sweep"), ("breathe", "twitch", "tilt", "shiver_soft"), None, 0.8, 0.75),
        Mood("greet", {"front": 30, "right": -20, "back": -20, "left": -20}, ("greet", "bow", "wave"), ("nod", "tap", "sway", "shrug"), "startle", 1.1, 0.5),
        Mood("spider", _u(-50), ("spider", "pounce", "prowl"), ("twitch", "shiver", "tap", "scuttle_step"), "crouch_snap", 1.3, 0.55),
        Mood("flower", _u(-15), ("flower", "bloom_hold", "sway_bloom"), ("breathe", "petal_flutter", "tilt"), "unfurl", 0.8, 0.6),
        Mood("wave", _u(0), ("wave", "spin", "stretch"), ("ripple_micro", "tap", "shrug"), "startle", 1.2, 0.4),
        Mood("scan", _u(10), ("scan", "peek", "sweep"), ("twitch", "tilt", "nod"), "alert_up", 1.0, 0.6),
        Mood("rest", _u(75), ("rest", "curl_breathe"), ("breathe", "shiver_soft"), "settle", 0.6, 0.8),
    )
}


@dataclass(frozen=True)
class Clip:
    animation: Animation
    speed: float
    duration_s: float
    offset: dict[str, float] | None  # rest pose for micro clips, None for macros

    @property
    def name(self) -> str:
        return self.animation.name

    @property
    def scale(self) -> str:
        return self.animation.scale


@dataclass
class Choreographer:
    seed: int | None = None
    mood: Mood = field(default_factory=lambda: MOODS["idle"])
    _rng: random.Random = field(init=False)
    _pending_entry: bool = field(default=False, init=False)
    _last_macro: str | None = field(default=None, init=False)
    _micros_left: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def set_mood(self, name: str) -> None:
        if name not in MOODS:
            raise KeyError(f"unknown mood {name!r}; known: {sorted(MOODS)}")
        if MOODS[name] is self.mood:
            return
        self.mood = MOODS[name]
        self._pending_entry = self.mood.entry is not None
        self._last_macro = None
        self._micros_left = 0

    def next_clip(self) -> Clip:
        if self._pending_entry:
            self._pending_entry = False
            return self._make(self.mood.entry, macro=True, repeats=1)
        if self._micros_left > 0:
            self._micros_left -= 1
            return self._make(self._rng.choice(self.mood.micros), macro=False, repeats=self._rng.randint(1, 3))
        name = self._pick_macro()
        if self._rng.random() < self.mood.micro_bias:
            self._micros_left = self._rng.randint(1, 2)
        return self._make(name, macro=True, repeats=self._rng.randint(1, 3))

    def _pick_macro(self) -> str:
        options = [m for m in self.mood.macros if m != self._last_macro] or list(self.mood.macros)
        # the anchor macro (first in the list) is favoured so the mood stays legible
        weights = [2.0 if m == self.mood.macros[0] else 1.0 for m in options]
        name = self._rng.choices(options, weights)[0]
        self._last_macro = name
        return name

    def _make(self, name: str, *, macro: bool, repeats: int) -> Clip:
        anim = ANIMATIONS[name]
        speed = self.mood.tempo * self._rng.uniform(0.85, 1.2)
        period = anim.period_s / speed
        duration = period * (repeats if anim.loop else 1)
        duration = min(duration, 8.0 if macro else 2.5)
        return Clip(anim, speed, max(duration, 0.3), None if macro else dict(self.mood.rest_pose))
