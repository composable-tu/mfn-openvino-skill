from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 参考 preprocess-face-exp：使用 MediaPipe FaceDetector 的关键点做仿射对齐到 112x112
REFERENCE_PTS = np.array(
    [
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [56.1396, 92.2048],  # 嘴中心
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class AlignResult:
    face_bgr_112: np.ndarray
    detected: bool


class MediaPipeFaceAligner:
    def __init__(self, detector_model_path: str, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "未安装 mediapipe，无法做人脸检测/对齐。请先安装：python -m pip install mediapipe"
            ) from e

        model_path = str(detector_model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"找不到人脸检测模型文件: {model_path}")

        self._mp = mp
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=float(min_detection_confidence),
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def align_bgr(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if image_bgr is None:
            return None
        h, w = image_bgr.shape[:2]

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        )
        detection_result = self._detector.detect(mp_image)
        if not detection_result.detections:
            return None

        # 默认取第一张脸（MediaPipe 通常按置信度排序）
        detection = detection_result.detections[0]
        kps = detection.keypoints

        src_pts = np.array(
            [
                [kps[0].x * w, kps[0].y * h],  # 左眼
                [kps[1].x * w, kps[1].y * h],  # 右眼
                [kps[2].x * w, kps[2].y * h],  # 鼻尖
                [kps[3].x * w, kps[3].y * h],  # 嘴中心
            ],
            dtype=np.float32,
        )

        M, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_PTS)
        if M is None:
            return None

        aligned = cv2.warpAffine(
            image_bgr,
            M,
            (112, 112),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

