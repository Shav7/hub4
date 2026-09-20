#!/usr/bin/env sh
# Fetch the YOLOv8n ONNX weights (≈12 MB) used by src/vision.py.
set -e
cd "$(dirname "$0")"
curl -L -o yolov8n.onnx "https://github.com/Hyuto/yolov8-onnxruntime-web/raw/master/public/model/yolov8n.onnx"
echo "saved $(pwd)/yolov8n.onnx"
