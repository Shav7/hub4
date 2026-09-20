import numpy as np
import pytest

import vision
from vision import COCO_LABELS, Detection, find_camera_index


class FakeCapture:
    def __init__(self, index, usable):
        self.ok = index in usable
    def isOpened(self):
        return self.ok
    def read(self):
        return (self.ok, np.zeros((4, 4, 3), np.uint8) if self.ok else None)
    def release(self):
        pass


def _patch(monkeypatch, usable):
    monkeypatch.setattr(vision.cv2, "VideoCapture", lambda i: FakeCapture(i, usable))


def test_find_camera_prefers_external_highest_index(monkeypatch):
    _patch(monkeypatch, {0, 1})
    assert find_camera_index() == 1


def test_find_camera_builtin_when_requested(monkeypatch):
    _patch(monkeypatch, {0, 1})
    assert find_camera_index(prefer_external=False) == 0


def test_find_camera_none_raises(monkeypatch):
    _patch(monkeypatch, set())
    with pytest.raises(RuntimeError):
        find_camera_index()


def test_coco_has_80_labels():
    assert len(COCO_LABELS) == 80


def test_detection_area():
    assert Detection("cup", 0.9, (10, 10, 30, 50)).area == 800
