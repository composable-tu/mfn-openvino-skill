import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import lancedb
import pyarrow as pa


@dataclass(frozen=True)
class IdentifyResult:
    name: Optional[str]
    score: float
    passed: bool


class FaceEmbeddingDB:
    def __init__(self, db_dir: str):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 LanceDB 连接
        self._db = lancedb.connect(str(self.db_dir))

        # 创建或打开表
        self._table_name = "face_embeddings"
        if self._table_name in self._db.table_names():
            self._table = self._db.open_table(self._table_name)
        else:
            # 创建新表，schema: id (str), name (str), vector (array<float>)
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 512)),
            ])
            self._table = self._db.create_table(
                self._table_name,
                schema=schema
            )



    def enroll(self, name: str, embedding: np.ndarray) -> str:
        if not name or not name.strip():
            raise ValueError("name 不能为空")
        name = name.strip()

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        # L2 归一化
        emb = emb / (float(np.linalg.norm(emb)) + 1e-10)

        # 生成唯一 ID
        entry_id = str(uuid.uuid4())
        
        # 插入到 LanceDB
        self._table.add([
            {
                "id": entry_id,
                "name": name,
                "vector": emb.tolist(),
            }
        ])
        
        return entry_id

    def list_people(self) -> List[str]:
        # 从 LanceDB 表中查询所有不重复的 name
        result = self._table.to_pandas()["name"].unique().tolist()
        return sorted(result)

    def identify(self, embedding: np.ndarray, threshold: float = 0.35) -> IdentifyResult:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        emb = emb / (float(np.linalg.norm(emb)) + 1e-10)

        # 检查表是否为空
        if len(self._table) == 0:
            return IdentifyResult(name=None, score=float("-inf"), passed=False)

        # 使用 LanceDB 的向量搜索功能
        results = self._table.search(emb.tolist()).limit(1).to_list()
        
        if not results:
            return IdentifyResult(name=None, score=float("-inf"), passed=False)
        
        best_match = results[0]
        best_score = float(best_match["_distance"])  # LanceDB 返回的是距离，不是相似度
        best_name = best_match.get("name")
        
        # LanceDB 默认使用 L2 距离，需要转换为相似度分数
        # 对于归一化向量，cosine_similarity = 1 - (l2_distance^2 / 2)
        # 但这里我们直接使用距离作为分数（越小越相似）
        # 为了保持接口一致性，我们将距离转换为相似度
        cosine_sim = 1.0 - (best_score * best_score / 2.0)
        
        passed = best_name is not None and cosine_sim >= float(threshold)
        return IdentifyResult(
            name=best_name if passed else None,
            score=cosine_sim,
            passed=passed
        )

