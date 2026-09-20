"""OpenCV-only object detection: YOLOv8n ONNX via cv2.dnn, plus a warmed-up webcam."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COCO_LABELS = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
    "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
    "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
    "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
    "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush"
).split()
assert len(COCO_LABELS) == 80

INPUT_SIZE = 640


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.box_xyxy
        return max(0, x1 - x0) * max(0, y1 - y0)


class YoloDetector:
    def __init__(self, model_path: str | Path, conf_threshold: float = 0.35, nms_threshold: float = 0.45) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.size == 0:
            raise ValueError("expected a non-empty HxWx3 BGR frame")
        blob, scale = _letterbox(frame_bgr)
        self.net.setInput(blob)
        raw = self.net.forward()[0].T  # (8400, 84): cx, cy, w, h, 80 class scores
        return _postprocess(raw, scale, frame_bgr.shape[1], frame_bgr.shape[0], self.conf_threshold, self.nms_threshold)


def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    resized = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return cv2.dnn.blobFromImage(canvas, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True), scale


def _postprocess(raw: np.ndarray, scale: float, width: int, height: int, conf_thr: float, nms_thr: float) -> list[Detection]:
    scores = raw[:, 4:]
    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]
    keep = confidences >= conf_thr
    if not keep.any():
        return []
    boxes_cxcywh = raw[keep, :4] / scale
    confidences, class_ids = confidences[keep], class_ids[keep]
    boxes_xywh = np.column_stack(
        [boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2, boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]]
    )
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confidences.tolist(), conf_thr, nms_thr)
    detections = []
    for i in np.array(indices).flatten():
        x, y, w, h = boxes_xywh[i]
        box = (int(max(0, x)), int(max(0, y)), int(min(width, x + w)), int(min(height, y + h)))
        detections.append(Detection(COCO_LABELS[class_ids[i]], float(confidences[i]), box))
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


class Camera:
    """Webcam wrapper that discards warm-up frames so auto-exposure has settled."""

    def __init__(self, index: int = 0, warmup_frames: int = 15, open_timeout_s: float = 5.0) -> None:
        self.capture = cv2.VideoCapture(index)
        deadline = time.monotonic() + open_timeout_s
        while not self.capture.isOpened() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not self.capture.isOpened():
            raise RuntimeError(f"camera {index} did not open within {open_timeout_s}s")
        for _ in range(warmup_frames):
            self.capture.read()
        logger.info("camera %d ready", index)

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("camera read failed")
        return frame

    def release(self) -> None:
        self.capture.release()
