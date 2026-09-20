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


class FlakyCapture:
    """Opens fine; first read fails, later reads succeed."""
    reads = 0

    def __init__(self, index):
        pass

    def isOpened(self):
        return True

    def read(self):
        FlakyCapture.reads += 1
        if FlakyCapture.reads == 16:  # first read after the 15-frame warm-up
            return False, None
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        pass


def test_camera_reconnects_after_failed_read(monkeypatch):
    FlakyCapture.reads = 0
    monkeypatch.setattr(vision.cv2, "VideoCapture", FlakyCapture)
    monkeypatch.setattr(vision.time, "sleep", lambda s: None)
    cam = vision.Camera(0, warmup_frames=15, reconnect_timeout_s=5)
    frame = cam.read()  # fails once, reconnects, returns a frame
    assert frame.shape == (4, 4, 3)


LISTING = """[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] [1] Innomaker-U20CAM-1080p-S1
[AVFoundation indev @ 0x1] [2] Capture screen 0
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] MacBook Pro Microphone
"""


def test_parse_avfoundation_devices_video_only():
    assert vision.parse_avfoundation_devices(LISTING) == [
        (0, "FaceTime HD Camera"), (1, "Innomaker-U20CAM-1080p-S1"), (2, "Capture screen 0")]


def test_is_builtin_markers():
    assert vision.is_builtin("FaceTime HD Camera") and vision.is_builtin("Capture screen 0")
    assert not vision.is_builtin("Innomaker-U20CAM-1080p-S1")


class FakeFfmpegCamera:
    def __init__(self, name, **kw):
        self.device_name = name


def test_open_camera_waits_then_raises_when_named_camera_absent(monkeypatch, tmp_path):
    _patch(monkeypatch, {0}, tmp_path / "camera.json")
    monkeypatch.setattr(vision, "list_cameras", lambda: [(0, "FaceTime HD Camera")])
    monkeypatch.setattr(vision.time, "sleep", lambda s: None)
    vision.remember_camera(None, "Innomaker-U20CAM-1080p-S1")
    with pytest.raises(RuntimeError):
        vision.open_camera(wait_s=0.0)


def test_open_camera_prefers_remembered_name_when_connected(monkeypatch, tmp_path):
    _patch(monkeypatch, {0}, tmp_path / "camera.json")
    monkeypatch.setattr(vision, "list_cameras", lambda: [(0, "FaceTime HD Camera"), (1, "Innomaker-U20CAM-1080p-S1")])
    monkeypatch.setattr(vision, "FfmpegCamera", FakeFfmpegCamera)
    vision.remember_camera(None, "Innomaker-U20CAM-1080p-S1")
    cam, label = vision.open_camera()
    assert isinstance(cam, FakeFfmpegCamera) and cam.device_name == "Innomaker-U20CAM-1080p-S1"
    assert "ffmpeg" in label


def test_open_camera_picks_first_external_by_name(monkeypatch):
    _patch(monkeypatch, {0})
    monkeypatch.setattr(vision, "list_cameras", lambda: [(0, "FaceTime HD Camera"), (1, "Innomaker-U20CAM-1080p-S1")])
    monkeypatch.setattr(vision, "FfmpegCamera", FakeFfmpegCamera)
    cam, _ = vision.open_camera()
    assert cam.device_name == "Innomaker-U20CAM-1080p-S1"


def test_open_camera_falls_back_to_opencv_without_external(monkeypatch):
    _patch(monkeypatch, {0})
    monkeypatch.setattr(vision, "list_cameras", lambda: [(0, "FaceTime HD Camera")])
    cam, label = vision.open_camera()
    assert isinstance(cam, vision.Camera) and "guessed" in label


def test_open_camera_explicit_index_wins(monkeypatch):
    _patch(monkeypatch, {0, 1})
    monkeypatch.setattr(vision, "list_cameras", lambda: [(0, "Innomaker-U20CAM-1080p-S1")])
    cam, label = vision.open_camera(index=1)
    assert isinstance(cam, vision.Camera) and label.startswith("camera 1")
