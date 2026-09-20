"""Real-time layer: a debounced animation switcher and a continuously running player thread."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from animation import Animation
from poses import HubCalibration, Pose, minimum_jerk
from spider import SpiderController
from vision import Detection

logger = logging.getLogger(__name__)


@dataclass
class SwitchDecision:
    animation: str
    switched: bool
    candidate: str | None
    progress: int  # agreeing frames so far for the candidate
    needed: int
    trigger: Detection | None  # strongest detection backing the current/candidate animation


@dataclass
class AnimationSwitcher:
    """Only switches when a new animation has won `confirm_frames` consecutive frames,
    backed by a detection at or above `min_confidence`, and the current one has played
    for at least `min_dwell_s`. Falling back to idle needs a longer `idle_frames` streak."""

    fallback: str = "idle"
    confirm_frames: int = 4
    idle_frames: int = 12
    min_confidence: float = 0.5
    min_dwell_s: float = 1.5
    current: str = field(default="idle", init=False)
    _candidate: str | None = field(default=None, init=False)
    _streak: int = field(default=0, init=False)
    _last_switch: float = field(default=0.0, init=False)

    def update(self, chosen: str, detections: list[Detection], now: float | None = None) -> SwitchDecision:
        now = time.monotonic() if now is None else now
        strongest = max(detections, key=lambda d: d.confidence, default=None)
        confident = strongest is not None and strongest.confidence >= self.min_confidence
        is_idle = chosen == self.fallback

        if chosen == self.current or (not confident and not is_idle):
            self._candidate, self._streak = None, 0
            return SwitchDecision(self.current, False, None, 0, self.confirm_frames, strongest)

        if chosen != self._candidate:
            self._candidate, self._streak = chosen, 0
        self._streak += 1
        needed = self.idle_frames if is_idle else self.confirm_frames
        dwell_ok = now - self._last_switch >= self.min_dwell_s
        if self._streak >= needed and dwell_ok:
            logger.info("switch %s -> %s (%s)", self.current, chosen, _describe(strongest))
            self.current, self._last_switch = chosen, now
            self._candidate, self._streak = None, 0
            return SwitchDecision(self.current, True, None, 0, needed, strongest)
        return SwitchDecision(self.current, False, chosen, self._streak, needed, strongest)


def _describe(detection: Detection | None) -> str:
    return f"{detection.label} {detection.confidence:.2f}" if detection else "nothing"


class ContinuousPlayer(threading.Thread):
    """Streams the current animation to the limbs forever; crossfades on set_animation()."""

    def __init__(self, controller: SpiderController, animations: dict[str, Animation], initial: str,
                 calibration: HubCalibration | None = None, rate_hz: float = 60.0, blend_s: float = 0.5) -> None:
        super().__init__(daemon=True, name="animation-player")
        self.controller = controller
        self.animations = animations
        self.calibration = calibration or HubCalibration()
        self.rate_hz = rate_hz
        self.blend_s = blend_s
        self._lock = threading.Lock()
        self._current = animations[initial]
        self._speed = 1.0
        self._offset: dict[str, float] | None = None
        self._t0 = time.monotonic()
        self._blend_from: dict[str, float] | None = None
        self._last_angles: dict[str, float] | None = None
        self._stop_event = threading.Event()
        self.error: BaseException | None = None

    def set_animation(self, name: str, speed: float = 1.0, offset: dict[str, float] | None = None) -> None:
        """Start a clip: `speed` scales time, `offset` (per-limb degrees) is added for micro clips."""
        if speed <= 0:
            raise ValueError("speed must be positive")
        with self._lock:
            self._blend_from = dict(self._last_angles) if self._last_angles else None
            self._current = self.animations[name]
            self._speed = speed
            self._offset = dict(offset) if offset else None
            self._t0 = time.monotonic()

    @property
    def current_name(self) -> str:
        return self._current.name

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        period = 1.0 / self.rate_hz
        try:
            while not self._stop_event.is_set():
                tick = time.monotonic()
                self.controller.move_to(self.calibration.pose_to_targets(Pose("live", self._angles_now())))
                time.sleep(max(0.0, period - (time.monotonic() - tick)))
        except BaseException as exc:  # surfaced to the main thread via .error
            self.error = exc
            logger.exception("player thread died")

    def _angles_now(self) -> dict[str, float]:
        with self._lock:
            t = time.monotonic() - self._t0
            angles = self._current.angles_at(t * self._speed)
            if self._offset is not None:
                angles = {n: a + self._offset[n] for n, a in angles.items()}
            if self._blend_from is not None:
                if t >= self.blend_s:
                    self._blend_from = None
                else:
                    k = minimum_jerk(t / self.blend_s)
                    angles = {n: self._blend_from[n] + (a - self._blend_from[n]) * k for n, a in angles.items()}
            self._last_angles = angles
            return angles
