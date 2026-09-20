import time

from animation import ANIMATIONS
from realtime import AnimationSwitcher, ContinuousPlayer
from spider import SpiderConfig, SpiderController
from test_spider import FakeBus
from vision import Detection


def det(label, conf):
    return Detection(label, conf, (0, 0, 10, 10))


def test_switcher_needs_consecutive_confirmations():
    sw = AnimationSwitcher(confirm_frames=3, min_dwell_s=0)
    d1 = sw.update("flower", [det("potted_plant", 0.9)], now=10.0)
    d2 = sw.update("flower", [det("potted_plant", 0.9)], now=10.1)
    assert (d1.switched, d2.switched) == (False, False)
    assert d2.candidate == "flower" and d2.progress == 2
    d3 = sw.update("flower", [det("potted_plant", 0.9)], now=10.2)
    assert d3.switched and d3.animation == "flower"


def test_switcher_streak_resets_when_candidate_changes():
    sw = AnimationSwitcher(confirm_frames=3, min_dwell_s=0)
    sw.update("flower", [det("potted_plant", 0.9)], now=1.0)
    sw.update("flower", [det("potted_plant", 0.9)], now=1.1)
    d = sw.update("spider", [det("cat", 0.9)], now=1.2)
    assert not d.switched and d.candidate == "spider" and d.progress == 1


def test_switcher_ignores_low_confidence():
    sw = AnimationSwitcher(confirm_frames=1, min_confidence=0.5, min_dwell_s=0)
    d = sw.update("flower", [det("potted_plant", 0.4)], now=1.0)
    assert not d.switched and d.candidate is None and d.animation == "idle"


def test_switcher_respects_min_dwell():
    sw = AnimationSwitcher(confirm_frames=1, min_dwell_s=2.0)
    assert sw.update("flower", [det("potted_plant", 0.9)], now=10.0).switched
    assert not sw.update("spider", [det("cat", 0.9)], now=11.0).switched
    assert sw.update("spider", [det("cat", 0.9)], now=12.1).switched


def test_switcher_idle_needs_longer_streak():
    sw = AnimationSwitcher(confirm_frames=1, idle_frames=3, min_dwell_s=0)
    sw.update("flower", [det("potted_plant", 0.9)], now=1.0)
    assert not sw.update("idle", [], now=1.1).switched
    assert not sw.update("idle", [], now=1.2).switched
    assert sw.update("idle", [], now=1.3).switched


def test_switcher_same_animation_is_noop():
    sw = AnimationSwitcher(min_dwell_s=0)
    d = sw.update("idle", [], now=1.0)
    assert not d.switched and d.candidate is None


def test_continuous_player_streams_and_crossfades():
    bus = FakeBus(start=2048)
    ctrl = SpiderController(bus, SpiderConfig())
    ctrl.enable_torque()
    player = ContinuousPlayer(ctrl, ANIMATIONS, "idle", rate_hz=200, blend_s=0.1)
    player.start()
    time.sleep(0.15)
    n_idle = len(bus.goals)
    player.set_animation("rest")  # +80 deg
    time.sleep(0.3)
    player.stop(); player.join(1)
    assert n_idle >= 10 and len(bus.goals) > n_idle
    fronts = [g["front"] for g in bus.goals[n_idle:]]
    assert max(abs(b - a) for a, b in zip(fronts, fronts[1:])) < 150  # no jump at the switch
    assert fronts[-1] > 2048 + 700
    assert player.error is None
