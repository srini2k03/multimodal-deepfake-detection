#!/usr/bin/env bash
# Download OpenCV's ResNet-10 SSD face detector (used when face.detector=dnn).
set -euo pipefail
DEST="${1:-models/face}"
mkdir -p "$DEST"
BASE="https://raw.githubusercontent.com/opencv"
curl -fsSL "$BASE/opencv/4.x/samples/dnn/face_detector/deploy.prototxt" -o "$DEST/deploy.prototxt"
curl -fsSL "$BASE/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel" \
  -o "$DEST/res10_300x300_ssd_iter_140000.caffemodel"
echo "Saved face detector to $DEST"
