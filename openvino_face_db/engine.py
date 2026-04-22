import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from openvino import Core

from .align import MediaPipeFaceAligner


@dataclass(frozen=True)
class EmbedResult:
    embedding: np.ndarray
    inference_time_s: float


def _preprocess_rgb_112(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("image is None")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112))
    x = np.expand_dims(rgb, axis=0).astype(np.uint8)
    return x


class OpenVINOFaceEmbedder:
    """
    OpenVINO IR inference wrapper for face embedding extraction.

    Assumes:
    - input: RGB 112x112, uint8, NCHW? (the existing demo uses NHWC uint8)
    - output: embedding vector, later L2-normalized
    """

    def __init__(
        self,
        model_xml: str,
        device: str = "CPU",
        *,
        face_detector_model: str | None = None,
        align_before_embed: bool = True,
        min_detection_confidence: float = 0.5,
    ):
        self.model_xml = str(model_xml)
        self.device = device
        self.align_before_embed = bool(align_before_embed)
        self._aligner = (
            MediaPipeFaceAligner(
                detector_model_path=str(face_detector_model),
                min_detection_confidence=min_detection_confidence,
            )
            if self.align_before_embed and face_detector_model is not None
            else None
        )

        core = Core()
        model = core.read_model(model=self.model_xml)
        self._compiled = core.compile_model(model=model, device_name=self.device)

    def embed_bgr(self, image_bgr: np.ndarray) -> EmbedResult:
        if self.align_before_embed and self._aligner is not None:
            aligned = self._aligner.align_bgr(image_bgr)
            if aligned is None:
                raise ValueError("未检测到人脸，无法对齐裁剪到 112x112。")
            image_bgr = aligned

        input_data = _preprocess_rgb_112(image_bgr)

        start = time.time()
        outputs = self._compiled([input_data])
        infer_s = time.time() - start

        feat = next(iter(outputs.values()))
        feat = np.asarray(feat).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(feat))
        feat = feat / (norm + 1e-10)
        return EmbedResult(embedding=feat, inference_time_s=infer_s)

    def embed_file(self, image_path: str) -> EmbedResult:
        p = Path(image_path)
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"无法读取图像: {p}")
        return self.embed_bgr(img)

