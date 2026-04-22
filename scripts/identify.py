import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openvino_face_db import FaceEmbeddingDB, OpenVINOFaceEmbedder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify a face from a local DB (OpenVINO).")
    parser.add_argument("--image", required=True, help="Path to a cropped face image.")
    parser.add_argument("--db", default=str(_ROOT / "db"), help="DB folder path.")
    parser.add_argument("--model", default=str(_ROOT / "model" / "openvino" / "model.xml"), help="OpenVINO IR model XML path.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device name (default CPU).")
    parser.add_argument("--threshold", type=float, default=0.35, help="Cosine similarity threshold.")
    parser.add_argument(
        "--face_detector_model",
        default=str(_ROOT / "model" / "mediapipe" / "blaze_face_short_range.tflite"),
        help="MediaPipe FaceDetector TFLite 模型路径（用于检测/对齐裁剪到 112x112）。",
    )
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="关闭人脸检测/对齐裁剪；仅对整张图做 resize 到 112x112（要求输入已是对齐人脸图）。",
    )
    args = parser.parse_args()

    embedder = OpenVINOFaceEmbedder(
        model_xml=args.model,
        device=args.device,
        face_detector_model=None if args.no_align else args.face_detector_model,
        align_before_embed=not args.no_align,
    )
    res = embedder.embed_file(args.image)

    db = FaceEmbeddingDB(args.db)
    result = db.identify(embedding=res.embedding, threshold=args.threshold)

    print(f"passed={str(result.passed).lower()}")
    print(f"best_name={result.name if result.name is not None else ''}")
    print(f"best_score={result.score:.6f}")
    print(f"threshold={args.threshold:.6f}")
    print(f"inference_time_s={res.inference_time_s:.6f}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

