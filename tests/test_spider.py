import math

import pytest

from spider import Limb, SpiderConfig, SpiderController, clamp_gaze, gaze_to_targets


class FakeBus:
    def __init__(self, start: int = 2048):
        self.registers: dict[tuple[str, str], int] = {}
        self.positions = {n: start for n in ("front", "right", "back", "left")}
        self.goals: list[dict[str, int]] = []

    def read(self, data_name, motor, *, normalize=True):
        return self.registers.get((data_name, motor), 0)

    def write(self, data_name, motor, value, *, normalize=True, num_retry=0):
        self.registers[(data_name, motor)] = value

    def sync_write(self, data_name, values, *, normalize=True):
        assert data_name == "Goal_Position"
        self.goals.append(dict(values))
        self.positions.update(values)  # ideal servo: arrives instantly

    def sync_read(self, data_name, *, normalize=True):
        return dict(self.positions)


@pytest.fixture
def config():
    return SpiderConfig(center_ticks=2048, max_tilt_ticks=300, settle_timeout_s=0.1)


def test_clamp_gaze_inside_disc_unchanged():
    assert clamp_gaze(0.3, -0.4) == (0.3, -0.4)


def test_clamp_gaze_outside_disc_normalized():
    x, y = clamp_gaze(3.0, 4.0)
    assert math.isclose(math.hypot(x, y), 1.0)


def test_clamp_gaze_nan_raises():
    with pytest.raises(ValueError):
        clamp_gaze(float("nan"), 0.0)


def test_gaze_to_targets_zero_gaze_all_center(config):
    assert set(gaze_to_targets(config, 0.0, 0.0).values()) == {2048}


def test_gaze_to_targets_front_gaze_moves_front_and_back_opposite(config):
    t = gaze_to_targets(config, 1.0, 0.0)
    assert t["front"] == 2048 + 300
    assert t["back"] == 2048 - 300
    assert t["right"] == 2048 and t["left"] == 2048


def test_gaze_to_targets_diagonal_splits_tilt(config):
    t = gaze_to_targets(config, 1.0, 1.0)  # clamped to unit -> cos45 each
    expected = 2048 + round(300 / math.sqrt(2))
    assert t["front"] == expected and t["right"] == expected


def test_gaze_to_targets_clamped_to_valid_ticks():
    cfg = SpiderConfig(center_ticks=100, max_tilt_ticks=500)
    t = gaze_to_targets(cfg, -1.0, 0.0)
    assert t["front"] == 0


def test_move_to_without_torque_raises(config):
    ctrl = SpiderController(FakeBus(), config)
    with pytest.raises(RuntimeError):
        ctrl.move_to({"front": 2048})


def test_look_at_writes_goals_and_settles(config):
    bus = FakeBus()
    ctrl = SpiderController(bus, config)
    ctrl.enable_torque()
    targets = ctrl.look_at(0.0, 1.0)
    assert bus.goals[-1] == targets
    assert ctrl.wait_settled(targets) is True


def test_wait_settled_times_out_when_stuck(config):
    bus = FakeBus()
    bus.sync_write = lambda *a, **k: None  # servo never moves
    ctrl = SpiderController(bus, config)
    ctrl.enable_torque()
    assert ctrl.wait_settled({"front": 3000}) is False


def test_disable_torque_writes_zero_to_every_limb(config):
    bus = FakeBus()
    ctrl = SpiderController(bus, config)
    ctrl.enable_torque()
    ctrl.disable_torque()
    assert all(bus.registers[("Torque_Enable", n)] == 0 for n in ctrl.limb_names)
