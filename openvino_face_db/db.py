import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class IdentifyResult:
    name: Optional[str]
    score: float
    passed: bool


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"embedding shape mismatch: {a.shape} vs {b.shape}")
    return float(np.dot(a, b))


class FaceEmbeddingDB:
    def __init__(self, db_dir: str):
        self.db_dir = Path(db_dir)
        self.emb_dir = self.db_dir / "embeddings"
        self.index_path = self.db_dir / "index.json"

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.emb_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> List[Dict[str, str]]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, entries: List[Dict[str, str]]) -> None:
        self.index_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def enroll(self, name: str, embedding: np.ndarray) -> Path:
        if not name or not name.strip():
            raise ValueError("name 不能为空")
        name = name.strip()

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        # assume already normalized, but keep it safe
        emb = emb / (float(np.linalg.norm(emb)) + 1e-10)

        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip("_-")
        if not safe:
            safe = "person"

        existing = self._load_index()
        next_id = 1
        for e in existing:
            try:
                next_id = max(next_id, int(e.get("id", "0")) + 1)
            except Exception:
                continue

        out_path = self.emb_dir / f"{safe}_{next_id}.npy"
        np.save(str(out_path), emb)

        existing.append({"id": str(next_id), "name": name, "path": str(out_path.relative_to(self.db_dir))})
        self._save_index(existing)
        return out_path

    def list_people(self) -> List[str]:
        entries = self._load_index()
        return sorted({e["name"] for e in entries if "name" in e})

    def identify(self, embedding: np.ndarray, threshold: float = 0.35) -> IdentifyResult:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        emb = emb / (float(np.linalg.norm(emb)) + 1e-10)

        entries = self._load_index()
        if not entries:
            return IdentifyResult(name=None, score=float("-inf"), passed=False)

        best_name: Optional[str] = None
        best_score = float("-inf")

        for e in entries:
            rel = e.get("path")
            if not rel:
                continue
            p = self.db_dir / rel
            if not p.exists():
                continue
            vec = np.load(str(p)).astype(np.float32).reshape(-1)
            score = _cosine_similarity(emb, vec)
            if score > best_score:
                best_score = score
                best_name = e.get("name")

        passed = best_name is not None and best_score >= float(threshold)
        return IdentifyResult(name=best_name if passed else None, score=best_score, passed=passed)

