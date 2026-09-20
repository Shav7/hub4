import numpy as np
import pytest

import vision
from vision import COCO_LABELS, Detection, find_camera_index, is_builtin, parse_avfoundation_devices


class FakeCapture:
    def __init__(self, index, usable):
        self.ok = index in usable
    def isOpened(self):
        return self.ok
    def read(self):
        return (self.ok, np.zeros((4, 4, 3), np.uint8) if self.ok else None)
    def release(self):
        pass


def _patch(monkeypatch, usable, named=()):
    monkeypatch.setattr(vision.cv2, "VideoCapture", lambda i: FakeCapture(i, usable))
    monkeypatch.setattr(vision, "list_cameras", lambda: list(named))


LISTING = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] [1] Innomaker-U20CAM-1080p-S1
[AVFoundation indev @ 0x1] [2] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
"""


def test_parse_avfoundation_devices_video_only():
    assert parse_avfoundation_devices(LISTING) == [(0, "FaceTime HD Camera"), (1, "Innomaker-U20CAM-1080p-S1"), (2, "Capture screen 0")]


def test_is_builtin_markers():
    assert is_builtin("FaceTime HD Camera") and is_builtin("Capture screen 0") and is_builtin("Shahvir's iPhone Camera")
    assert not is_builtin("Innomaker-U20CAM-1080p-S1")


def test_find_camera_by_name_picks_usb_even_if_first(monkeypatch):
    _patch(monkeypatch, {0, 1}, named=[(0, "Innomaker-U20CAM-1080p-S1"), (1, "FaceTime HD Camera")])
    assert find_camera_index() == (0, "Innomaker-U20CAM-1080p-S1")


def test_find_camera_by_name_raises_without_external(monkeypatch):
    _patch(monkeypatch, {0}, named=[(0, "FaceTime HD Camera")])
    with pytest.raises(RuntimeError):
        find_camera_index()


def test_find_camera_prefers_external_highest_index(monkeypatch):
    _patch(monkeypatch, {0, 1})
    assert find_camera_index() == (1, "camera 1")


def test_find_camera_builtin_when_requested(monkeypatch):
    _patch(monkeypatch, {0, 1})
    assert find_camera_index(prefer_external=False) == (0, "camera 0")


def test_find_camera_none_raises(monkeypatch):
    _patch(monkeypatch, set())
    with pytest.raises(RuntimeError):
        find_camera_index()


def test_coco_has_80_labels():
    assert len(COCO_LABELS) == 80


def test_detection_area():
    assert Detection("cup", 0.9, (10, 10, 30, 50)).area == 800
