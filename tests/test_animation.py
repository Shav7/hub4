import math

import pytest

from animation import ANIMATIONS, RING, Animation, AnimationPlayer, Keyframe
from poses import LIMB_NAMES
from semantics import SemanticProfile
from spider import SpiderConfig, SpiderController
from test_spider import FakeBus


def u(a):
    return {l: a for l in LIMB_NAMES}


def test_keyframe_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        Keyframe(u(0.0), 0.0)


def test_keyframe_rejects_missing_limb():
    with pytest.raises(ValueError):
        Keyframe({"front": 0.0}, 1.0)


def test_period_is_sum_of_durations():
    a = Animation("t", (Keyframe(u(0), 0.5), Keyframe(u(10), 1.5)), SemanticProfile("t", {}))
    assert a.period_s == 2.0


def test_angles_hit_keyframes_at_boundaries():
    a = Animation("t", (Keyframe(u(0), 1.0), Keyframe(u(40), 1.0)), SemanticProfile("t", {}))
    assert math.isclose(a.angles_at(1.0)["front"], 0.0, abs_tol=1e-6)
    assert math.isclose(a.angles_at(2.0 - 1e-9)["front"], 40.0, abs_tol=1e-3)


def test_loop_wraps_time():
    a = Animation("t", (Keyframe(u(0), 1.0), Keyframe(u(40), 1.0)), SemanticProfile("t", {}))
    assert math.isclose(a.angles_at(0.3)["front"], a.angles_at(2.3)["front"], abs_tol=1e-6)


def test_non_loop_holds_final_keyframe():
    a = Animation("t", (Keyframe(u(80), 1.0),), SemanticProfile("t", {}), loop=False)
    assert a.angles_at(5.0)["front"] == 80.0


def test_phase_lag_delays_successive_limbs():
    a = Animation("t", (Keyframe(u(0), 1.0), Keyframe(u(40), 1.0)), SemanticProfile("t", {}), limb_phase_s=0.25)
    t = 1.5
    angles = a.angles_at(t)
    for earlier, later in zip(RING, RING[1:]):
        # each later limb is where the earlier one was 0.25s ago
        assert math.isclose(angles[later], a._single_limb_angle(later, t - 0.25 * RING.index(later)), abs_tol=1e-6)


def test_spider_animation_alternates_pairs():
    s = ANIMATIONS["spider"]
    a = s.angles_at(0.28 - 1e-6)  # end of first keyframe
    assert a["front"] < a["right"]
    b = s.angles_at(0.56 - 1e-6)
    assert b["front"] > b["right"]


def test_all_animations_stay_within_calibration_limits():
    for anim in ANIMATIONS.values():
        for step in range(0, 100):
            for angle in anim.angles_at(anim.period_s * step / 100).values():
                assert -95.0 <= angle <= 95.0


def test_player_streams_frames_and_blends_from_start():
    bus = FakeBus(start=2048)
    ctrl = SpiderController(bus, SpiderConfig())
    ctrl.enable_torque()
    player = AnimationPlayer(ctrl, rate_hz=200, blend_in_s=0.1)
    player.play(ANIMATIONS["rest"], 0.3)
    assert len(bus.goals) >= 20
    assert abs(bus.goals[0]["front"] - 2048) < 50  # started near where it was
    assert bus.goals[-1]["front"] > 2048 + 700  # heading toward +80 deg
