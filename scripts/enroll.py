import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from openvino_face_db import FaceEmbeddingDB, OpenVINOFaceEmbedder  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll a face embedding into a local DB (OpenVINO).")
    parser.add_argument("--name", required=True, help="Person name / label to enroll.")
    parser.add_argument("--image", required=True, help="Path to a cropped face image.")
    parser.add_argument("--db", default=str(_ROOT / "db"), help="DB folder path.")
    parser.add_argument("--model", default=str(_ROOT / "model" / "openvino" / "model.xml"), help="OpenVINO IR model XML path.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device name (default CPU).")
    args = parser.parse_args()

    embedder = OpenVINOFaceEmbedder(model_xml=args.model, device=args.device)
    res = embedder.embed_file(args.image)

    db = FaceEmbeddingDB(args.db)
    out = db.enroll(name=args.name, embedding=res.embedding)

    print(f"enrolled_name={args.name}")
    print(f"embedding_dim={res.embedding.shape[0]}")
    print(f"inference_time_s={res.inference_time_s:.6f}")
    print(f"saved_embedding={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

