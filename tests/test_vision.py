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


def _patch(monkeypatch, usable, config_path=None):
    monkeypatch.setattr(vision.cv2, "VideoCapture", lambda i: FakeCapture(i, usable))
    monkeypatch.setattr(vision, "CAMERA_CONFIG", config_path or vision.Path("/nonexistent/camera.json"))


def test_remembered_camera_wins(monkeypatch, tmp_path):
    _patch(monkeypatch, {0, 1}, tmp_path / "camera.json")
    vision.remember_camera(0, "usb")
    assert find_camera_index() == (0, "usb")


def test_malformed_camera_config_is_ignored(monkeypatch, tmp_path):
    cfg = tmp_path / "camera.json"
    cfg.write_text("{not json")
    _patch(monkeypatch, {0, 1}, cfg)
    assert find_camera_index() == (1, "camera 1")
