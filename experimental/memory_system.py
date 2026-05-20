import sqlite3
import json
import os
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("daoti.memory")


class MemorySystem:
    def __init__(self, db_path: str = "", vector_path: str = ""):
        data_dir = os.path.dirname(db_path) if db_path else "data"
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(data_dir, "memory.db")
        self.vector_path = vector_path or os.path.join(data_dir, "vectors")
        self._embedder = None
        self._tfidf_vocab = {}
        self._tfidf_docs = []
        self._db_lock = threading.Lock()
        self._init_db()
        self._init_vectordb()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        for table in ["episodic", "semantic", "procedural", "reflective"]:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding_id TEXT,
                    importance REAL DEFAULT 0.5,
                    timestamp TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)
        self.conn.commit()

    def _init_vectordb(self):
        try:
            import chromadb
            os.makedirs(self.vector_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=self.vector_path
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="daoti_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._has_chroma = True
            logger.info("ChromaDB vector store initialized")
        except Exception as e:
            logger.warning(f"ChromaDB not available ({e}), using TF-IDF fallback")
            self._has_chroma = False
            self.collection = None

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
            logger.info("SentenceTransformer embedder loaded")
        except Exception as e:
            logger.warning(f"SentenceTransformer not available ({e}), using TF-IDF")
            self._embedder = None
        return self._embedder

    def _embed(self, text: str) -> list:
        embedder = self._get_embedder()
        if embedder:
            return embedder.encode([text])[0].tolist()
        return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str) -> list:
        words = text.lower().split()[:1000]
        if not self._tfidf_vocab:
            return [0.0]
        vec = [0.0] * len(self._tfidf_vocab)
        for i, word in enumerate(words):
            if word in self._tfidf_vocab:
                vec[self._tfidf_vocab[word]] += 1.0 / (i + 1)
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            return [v / norm for v in vec]
        return vec

    def _update_tfidf(self, text: str):
        words = set(text.lower().split()[:1000])
        for w in words:
            if w not in self._tfidf_vocab:
                self._tfidf_vocab[w] = len(self._tfidf_vocab)

    def _store(self, table: str, content: str, metadata: dict = None,
               importance: float = 0.5) -> str:
        import uuid
        mem_id = str(uuid.uuid4())[:16]
        meta_json = json.dumps(metadata or {})
        ts = datetime.now(timezone.utc).isoformat()
        embedding = self._embed(content)
        embedding_id = ""
        if self._has_chroma and self.collection and any(embedding):
            try:
                self.collection.add(
                    ids=[mem_id],
                    embeddings=[embedding],
                    documents=[content],
                )
                embedding_id = mem_id
            except Exception as e:
                logger.warning(f"ChromaDB insert failed: {e}")
        self._update_tfidf(content)
        with self._db_lock:
            self.conn.execute(
                f"INSERT INTO {table} (id, content, metadata, embedding_id, importance, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (mem_id, content, meta_json, embedding_id, importance, ts),
            )
            self.conn.commit()
        return mem_id

    def store_episodic(self, content: str, metadata: dict = None,
                       importance: float = 0.5) -> str:
        return self._store("episodic", content, metadata, importance)

    def store_semantic(self, content: str, metadata: dict = None,
                       importance: float = 0.5) -> str:
        return self._store("semantic", content, metadata, importance)

    def store_procedural(self, content: str, metadata: dict = None,
                         importance: float = 0.5) -> str:
        return self._store("procedural", content, metadata, importance)

    def store_reflective(self, content: str, metadata: dict = None,
                         importance: float = 0.5) -> str:
        return self._store("reflective", content, metadata, importance)

    def retrieve_episodic(self, query: str, k: int = 10) -> list:
        return self._retrieve_sql("episodic", query, k)

    def retrieve_semantic(self, query: str, k: int = 10) -> list:
        return self._retrieve_sql("semantic", query, k)

    def _retrieve_sql(self, table: str, query: str, k: int = 10) -> list:
        with self._db_lock:
            cursor = self.conn.execute(
                f"SELECT * FROM {table} WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", k),
            )
            results = []
            for row in cursor:
                self.conn.execute(
                    f"UPDATE {table} SET access_count = access_count + 1 WHERE id = ?",
                    (row["id"],),
                )
                results.append(dict(row))
            self.conn.commit()
        return results

    def retrieve_all_recent(self, k: int = 10) -> list:
        tables = ["episodic", "semantic", "procedural", "reflective"]
        results = []
        with self._db_lock:
            for table in tables:
                cursor = self.conn.execute(
                    f"SELECT *, '{table}' as memory_type FROM {table} ORDER BY timestamp DESC LIMIT ?",
                    (k // len(tables) + 1,),
                )
                results.extend(dict(row) for row in cursor)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:k]

    def get_memory_summary(self) -> str:
        recent = self.retrieve_all_recent(10)
        if not recent:
            return "No memories stored yet."
        lines = []
        for mem in recent[:5]:
            lines.append(f"[{mem.get('memory_type', '?')}] {mem['content'][:200]}")
        return "\n".join(lines)

    def close(self):
        if hasattr(self, "conn"):
            self.conn.close()
