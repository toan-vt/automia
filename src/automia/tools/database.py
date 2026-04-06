from ..common import get_logger
from .embedding import EmbeddingTool
from .bm25 import BM25Tool
import psycopg2
import time
import numpy as np
import faiss
import json
import fcntl
import random
from pathlib import Path

class ExperimentRecord:
    def __init__(self, db_record: dict):
        self.id = db_record["id"] if "id" in db_record else None
        self.idea = db_record["idea"] if "idea" in db_record else None
        self.design_justification = db_record["design_justification"] if "design_justification" in db_record else None
        self.implementation = db_record["implementation"] if "implementation" in db_record else None
        self.auc_score = db_record["auc_score"] if "auc_score" in db_record else None
        self.tpr_1_score = db_record["tpr_1_score"] if "tpr_1_score" in db_record else None
        self.tpr_5_score = db_record["tpr_5_score"] if "tpr_5_score" in db_record else None
        self.analysis_summary = db_record["analysis_summary"] if "analysis_summary" in db_record else None
        self.parent_id = db_record["parent_id"] if "parent_id" in db_record else None
        self.embedding_vector = db_record["embedding_vector"] if "embedding_vector" in db_record else None

    def to_dict(self):
        return {
            "idea": self.idea,
            "design_justification": self.design_justification,
            "implementation": self.implementation,
            "auc_score": self.auc_score,
            "tpr_1_score": self.tpr_1_score,
            "tpr_5_score": self.tpr_5_score,
            "analysis_summary": self.analysis_summary,
            "parent_id": self.parent_id,
            "embedding_vector": self.embedding_vector,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

class DatabaseTool:
    def __init__(self, db_name: str = "mia", table_name: str = None, user: str = "user", password: str = "",
                 backend: str = "postgres", data_dir: str = None, embedding_tool: EmbeddingTool = None):
        self._logger = get_logger("system")
        self._backend = backend

        if table_name is not None:
            self._table_name = table_name
        else:
            self._table_name = f"mia_{time.strftime('%Y%m%d%H%M%S')}"

        if backend == "postgres":
            self._db_name = db_name
            self._user = user
            self._password = password
            self._conn = psycopg2.connect(dbname=self._db_name, user=self._user, password=self._password)
            self._cursor = self._conn.cursor()
            self._create_table()
        elif backend == "file":
            if data_dir is None:
                raise ValueError("data_dir is required when backend='file'")
            self._data_dir = Path(data_dir) / self._table_name
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._lock_path = self._data_dir / ".lock"
            self._lock_path.touch(exist_ok=True)
            self._logger.info(f"Database: File backend initialized at {self._data_dir}")
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'postgres' or 'file'")

        self._embedding_tool = embedding_tool
        self._bm25_tool = BM25Tool(existing_experiments=self.get_all_experiments())

    # ── File backend helpers ───────────────────────────────────────────────────

    def _file_path(self, id: int) -> Path:
        return self._data_dir / f"{id}.json"

    def _read_all_file_records(self) -> list:
        """Read all JSON records sorted by id. No lock needed for reads."""
        records = []
        for p in sorted(self._data_dir.glob("*.json"), key=lambda x: int(x.stem)):
            try:
                records.append(json.loads(p.read_text()))
            except Exception:
                pass
        return records

    def _next_id(self) -> int:
        """Compute next ID from existing files. Must be called with the write lock held."""
        existing = list(self._data_dir.glob("*.json"))
        if not existing:
            return 1
        return max(int(p.stem) for p in existing) + 1

    # ── Table creation (postgres only) ────────────────────────────────────────

    def _create_table(self):
        self._cursor.execute(f"CREATE TABLE IF NOT EXISTS {self._table_name} (id SERIAL PRIMARY KEY, idea TEXT, design_justification TEXT, implementation TEXT, analysis_summary TEXT, auc_score FLOAT, tpr_1_score FLOAT, tpr_5_score FLOAT, combined_score FLOAT, parent_id INT DEFAULT -1, idea_embedding FLOAT[], design_justification_embedding FLOAT[], implementation_embedding FLOAT[], analysis_summary_embedding FLOAT[])")
        self._conn.commit()
        self._logger.info(f"Database: Table {self._table_name} created successfully")

    # ── Insert ─────────────────────────────────────────────────────────────────

    def insert_experiment(self, idea: str, design_justification: str, implementation: str, auc_score: float, tpr_1_score: float, tpr_5_score: float, combined_score: float, analysis_summary: str, parent_id: int = -1):
        if not idea or idea == "":
            self._logger.warning(f"Database: Record not inserted into {self._table_name} because of empty idea")
            return
        if not design_justification or design_justification == "":
            self._logger.warning(f"Database: Record not inserted into {self._table_name} because of empty design justification")
            return
        if not implementation or implementation == "":
            self._logger.warning(f"Database: Record not inserted into {self._table_name} because of empty implementation")
            return
        if not analysis_summary or analysis_summary == "":
            self._logger.warning(f"Database: Record not inserted into {self._table_name} because of empty analysis summary")
            return
        if (auc_score is None) or (tpr_1_score is None) or (tpr_5_score is None) or (combined_score is None):
            self._logger.warning(f"Database: Record not inserted into {self._table_name} because of empty scores")
            return

        idea_embedding = self._embedding_tool.embed(idea)
        design_justification_embedding = self._embedding_tool.embed(design_justification)
        implementation_embedding = self._embedding_tool.embed(implementation)
        analysis_summary_embedding = self._embedding_tool.embed(analysis_summary)

        if self._backend == "postgres":
            self._cursor.execute(f"INSERT INTO {self._table_name} (idea, design_justification, implementation, analysis_summary, auc_score, tpr_1_score, tpr_5_score, combined_score, parent_id, idea_embedding, design_justification_embedding, implementation_embedding, analysis_summary_embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id", (idea, design_justification, implementation, analysis_summary, auc_score, tpr_1_score, tpr_5_score, combined_score, parent_id, idea_embedding, design_justification_embedding, implementation_embedding, analysis_summary_embedding))
            inserted_id = self._cursor.fetchone()[0]
            self._conn.commit()
            self._logger.info(f"Database: Record inserted into {self._table_name} successfully (parent id: {parent_id})")
            self._bm25_tool.update_index(self.get_experiment(inserted_id))
        else:
            with open(self._lock_path, "ab") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    inserted_id = self._next_id()
                    record = {
                        "id": inserted_id,
                        "idea": idea,
                        "design_justification": design_justification,
                        "implementation": implementation,
                        "analysis_summary": analysis_summary,
                        "auc_score": auc_score,
                        "tpr_1_score": tpr_1_score,
                        "tpr_5_score": tpr_5_score,
                        "combined_score": combined_score,
                        "parent_id": parent_id,
                        "idea_embedding": idea_embedding,
                        "design_justification_embedding": design_justification_embedding,
                        "implementation_embedding": implementation_embedding,
                        "analysis_summary_embedding": analysis_summary_embedding,
                    }
                    self._file_path(inserted_id).write_text(json.dumps(record))
                    self._logger.info(f"Database: Record {inserted_id} written to {self._data_dir} (parent id: {parent_id})")
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
            self._bm25_tool.update_index(self.get_experiment(inserted_id))

    # ── Get single ─────────────────────────────────────────────────────────────

    def get_experiment(self, id: int):
        if self._backend == "postgres":
            self._cursor.execute(f"SELECT * FROM {self._table_name} WHERE id = %s", (id,))
            cols = [desc[0] for desc in self._cursor.description]
            return dict(zip(cols, self._cursor.fetchone()))
        else:
            p = self._file_path(id)
            if not p.exists():
                return None
            return json.loads(p.read_text())

    # ── Get all ────────────────────────────────────────────────────────────────

    def get_all_experiments(self):
        if self._backend == "postgres":
            self._cursor.execute(f"SELECT * FROM {self._table_name}")
            cols = [desc[0] for desc in self._cursor.description]
            return [dict(zip(cols, row)) for row in self._cursor.fetchall()]
        else:
            return self._read_all_file_records()

    # ── Get random k ──────────────────────────────────────────────────────────

    def get_random_k_experiments(self, k: int):
        if self._backend == "postgres":
            self._cursor.execute(f"SELECT * FROM {self._table_name} ORDER BY RANDOM() LIMIT %s", (k,))
            cols = [desc[0] for desc in self._cursor.description]
            return [dict(zip(cols, row)) for row in self._cursor.fetchall()]
        else:
            records = self._read_all_file_records()
            return random.sample(records, min(k, len(records)))

    # ── Get by parent_id ──────────────────────────────────────────────────────

    def get_all_experiments_by_parent_id(self, parent_id: int, max_num: int = 10):
        if self._backend == "postgres":
            # sample within the parent cluster to reduce bias from recency
            self._cursor.execute(f"SELECT * FROM {self._table_name} WHERE parent_id = %s ORDER BY RANDOM() LIMIT %s", (parent_id, max_num))
            cols = [desc[0] for desc in self._cursor.description]
            return [dict(zip(cols, row)) for row in self._cursor.fetchall()]
        else:
            records = [r for r in self._read_all_file_records() if r.get("parent_id") == parent_id]
            return random.sample(records, min(max_num, len(records)))

    # ── Get top k ─────────────────────────────────────────────────────────────

    def get_top_k_experiments(self, k: int, score_type: str = "auc_score"):
        if self._backend == "postgres":
            self._cursor.execute(f"SELECT * FROM {self._table_name} ORDER BY {score_type} DESC LIMIT %s", (k,))
            cols = [desc[0] for desc in self._cursor.description]
            return [dict(zip(cols, row)) for row in self._cursor.fetchall()]
        else:
            records = self._read_all_file_records()
            records.sort(key=lambda r: r.get(score_type) or 0, reverse=True)
            return records[:k]

    # ── FAISS nearest neighbors ────────────────────────────────────────────────

    def get_top_k_nearest_neighbors(self, query: str, query_type: str = "idea", k: int = 10):
        embedding_vector = np.asarray(self._embedding_tool.embed(query), dtype=np.float32)

        if self._backend == "postgres":
            self._cursor.execute(f"SELECT id, {query_type}_embedding FROM {self._table_name}")
            rows = self._cursor.fetchall()
        else:
            records = self._read_all_file_records()
            rows = [(r["id"], r.get(f"{query_type}_embedding")) for r in records]

        if not rows:
            return []

        ids, embeddings = [], []
        for _id, embedding in rows:
            if embedding is None:
                continue
            ids.append(_id)
            embeddings.append(embedding)
        if not embeddings:
            return []

        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        query_dim = embedding_vector.shape[0]
        stored_dim = embeddings_np.shape[1]
        if query_dim != stored_dim:
            raise ValueError(
                f"Embedding dimension mismatch: current embedding model produces {query_dim}-dim vectors, "
                f"but '{query_type}_embedding' in the database has {stored_dim}-dim vectors. "
                "Use the same embedding model that was used when populating the DB, or clear/recreate the table."
            )

        faiss.normalize_L2(embeddings_np)
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)
        query_vec = embedding_vector[None, :].copy()
        faiss.normalize_L2(query_vec)
        min_k = min(k, len(embeddings_np))
        distances, indices = index.search(query_vec, min_k)
        top_k_ids = [ids[i] for i in indices[0]]

        if self._backend == "postgres":
            self._cursor.execute(f"SELECT * FROM {self._table_name} WHERE id = ANY(%s)", (top_k_ids,))
            cols = [desc[0] for desc in self._cursor.description]
            return [dict(zip(cols, row)) for row in self._cursor.fetchall()]
        else:
            return [self.get_experiment(id) for id in top_k_ids]

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def get_top_k_bm25(self, query: str, k: int = 10):
        return self._bm25_tool.retrieve(query, k)

    # ── Count ─────────────────────────────────────────────────────────────────

    def get_num_experiments(self):
        if self._backend == "postgres":
            self._cursor.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            row = self._cursor.fetchone()
            return int(row[0]) if row else 0
        else:
            return len(list(self._data_dir.glob("*.json")))
