import math

import pytest

from poses import LIMB_NAMES, POSES, HubCalibration, LimbCalibration, Pose, interpolate, minimum_jerk
from spider import SpiderConfig, SpiderController
from test_spider import FakeBus


def test_minimum_jerk_endpoints_and_midpoint():
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0
    assert math.isclose(minimum_jerk(0.5), 0.5)


def test_minimum_jerk_clamps_out_of_range():
    assert minimum_jerk(-1.0) == 0.0
    assert minimum_jerk(2.0) == 1.0


def test_minimum_jerk_monotonic():
    samples = [minimum_jerk(i / 50) for i in range(51)]
    assert all(b >= a for a, b in zip(samples, samples[1:]))


def test_pose_missing_limb_raises():
    with pytest.raises(ValueError):
        Pose("bad", {"front": 0.0})


def test_all_library_poses_have_every_limb():
    for pose in POSES.values():
        assert set(pose.angles_deg) == set(LIMB_NAMES)


def test_deg_to_ticks_neutral_and_90():
    cal = LimbCalibration(neutral_ticks=2048)
    assert cal.deg_to_ticks(0) == 2048
    assert cal.deg_to_ticks(90) == 3072


def test_deg_to_ticks_sign_flip_mirrors():
    assert LimbCalibration(sign=-1).deg_to_ticks(45) == 2048 - 512


def test_deg_to_ticks_clamped_to_limits():
    cal = LimbCalibration(min_deg=-30, max_deg=30)
    assert cal.deg_to_ticks(90) == cal.deg_to_ticks(30)


def test_pose_to_targets_spider_all_below_neutral():
    targets = HubCalibration().pose_to_targets(POSES["spider"])
    assert all(t < 2048 for t in targets.values())


def test_interpolate_endpoints():
    start = {n: 1000 for n in LIMB_NAMES}
    end = {n: 3000 for n in LIMB_NAMES}
    assert interpolate(start, end, 0.0) == start
    assert interpolate(start, end, 1.0) == end


def test_move_smooth_streams_frames_and_ends_on_target():
    bus = FakeBus(start=2048)
    ctrl = SpiderController(bus, SpiderConfig(settle_timeout_s=0.1))
    ctrl.enable_torque()
    targets = {n: 2548 for n in LIMB_NAMES}
    assert ctrl.move_smooth(targets, duration_s=0.1, rate_hz=100) is True
    assert len(bus.goals) >= 5
    assert bus.goals[-1] == targets
    fronts = [g["front"] for g in bus.goals]
    assert fronts == sorted(fronts)


def test_move_smooth_rejects_bad_duration():
    ctrl = SpiderController(FakeBus(), SpiderConfig())
    ctrl.enable_torque()
    with pytest.raises(ValueError):
        ctrl.move_smooth({n: 2048 for n in LIMB_NAMES}, duration_s=0)


def test_set_motion_profile_writes_all_limbs():
    bus = FakeBus()
    ctrl = SpiderController(bus, SpiderConfig())
    ctrl.set_motion_profile(acceleration=50, velocity=0)
    assert all(bus.registers[("Acceleration", n)] == 50 for n in LIMB_NAMES)
