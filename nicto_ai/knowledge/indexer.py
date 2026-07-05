"""
NICTO AI - Vector Indexer
Embeds and indexes crawled content for fast semantic retrieval
"""

import json
import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    """A single indexed entry"""
    id: int
    url: str
    title: str
    text: str
    summary: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class VectorIndexer:
    """
    Embeds text content and provides fast similarity search.

    Uses a lightweight approach:
    - TF-IDF style sparse embeddings (no GPU needed)
    - SQLite for persistent storage
    - numpy for vector operations

    For production, replace with a proper vector database (FAISS, ChromaDB).
    """

    def __init__(self, dim: int = 256, db_path: Optional[str] = None):
        self.dim = dim
        self.db_path = db_path or ":memory:"

        # Vocabulary for embedding
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

        # SQLite storage
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                text TEXT,
                summary TEXT,
                embedding BLOB,
                metadata TEXT
            )
        """)
        self._conn.commit()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        import re
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        # Remove very short/long tokens
        return [t for t in tokens if 2 < len(t) < 30]

    def _embed(self, text: str) -> np.ndarray:
        """
        Create a TF-IDF embedding of fixed dimension.

        Uses hashing trick with multiple hash functions for better distribution.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)

        embedding = np.zeros(self.dim, dtype=np.float32)

        # Count token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        for token, count in token_counts.items():
            # Use two hash functions for better distribution
            idx1 = hash(token) % self.dim
            idx2 = (hash(token) + hash(token) * 31) % self.dim
            # TF: normalized frequency
            tf = count / len(tokens)
            # IDF: use stored value or default
            idf = self._idf.get(token, 1.0)
            # Sign hash for decorrelation
            sign = 1 if hash(token + "sign") % 2 == 0 else -1
            embedding[idx1] += sign * tf * idf
            embedding[idx2] += tf * idf * 0.5

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding

    def _update_idf(self, text: str):
        """Update IDF values with new document."""
        tokens = set(self._tokenize(text))
        self._doc_count += 1
        for token in tokens:
            self._idf[token] = self._idf.get(token, 0) + 1
        # Convert counts to IDF
        for token in self._idf:
            self._idf[token] = np.log(self._doc_count / (1 + self._idf[token]))

    def add(
        self,
        url: str,
        title: str,
        text: str,
        summary: str = "",
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Add a document to the index.

        Args:
            url: Source URL
            title: Page title
            text: Full text content
            summary: Short summary
            metadata: Additional metadata

        Returns:
            Entry ID
        """
        self._update_idf(text)
        embedding = self._embed(text + " " + title)

        cursor = self._conn.execute(
            "INSERT INTO entries (url, title, text, summary, embedding, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (url, title, text[:10000], summary[:1000], embedding.tobytes(), json.dumps(metadata or {})),
        )
        self._conn.commit()
        entry_id = cursor.lastrowid
        logger.debug("Added entry %d: %s", entry_id, title[:50])
        return entry_id

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[int, float, Dict]]:
        """
        Semantic search over indexed content.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of (entry_id, score, metadata) tuples
        """
        query_embedding = self._embed(query)

        # Get all entries
        rows = self._conn.execute(
            "SELECT id, url, title, summary, embedding, metadata FROM entries"
        ).fetchall()

        if not rows:
            return []

        # Compute similarities
        results = []
        for row in rows:
            entry_id, url, title, summary, emb_bytes, meta_str = row
            stored_emb = np.frombuffer(emb_bytes, dtype=np.float32)

            # Cosine similarity (both are L2 normalized)
            score = float(np.dot(query_embedding, stored_emb))

            if score >= min_score:
                meta = json.loads(meta_str) if meta_str else {}
                meta["url"] = url
                meta["title"] = title
                meta["summary"] = summary
                results.append((entry_id, score, meta))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get(self, entry_id: int) -> Optional[Dict]:
        """Get a specific entry by ID."""
        row = self._conn.execute(
            "SELECT url, title, text, summary, metadata FROM entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if not row:
            return None
        url, title, text, summary, meta_str = row
        return {
            "id": entry_id,
            "url": url,
            "title": title,
            "text": text,
            "summary": summary,
            "metadata": json.loads(meta_str) if meta_str else {},
        }

    def count(self) -> int:
        """Get total number of indexed entries."""
        return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def clear(self):
        """Clear all entries."""
        self._conn.execute("DELETE FROM entries")
        self._conn.commit()
        self._vocab.clear()
        self._idf.clear()
        self._doc_count = 0

    def save(self, path: str):
        """Export index to a file."""
        rows = self._conn.execute("SELECT * FROM entries").fetchall()
        data = {
            "dim": self.dim,
            "doc_count": self._doc_count,
            "idf": self._idf,
            "entries": [
                {"url": r[1], "title": r[2], "text": r[3], "summary": r[4],
                 "embedding": r[5].hex() if isinstance(r[5], bytes) else r[5], "metadata": r[6]}
                for r in rows
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load index from a file."""
        with open(path) as f:
            data = json.load(f)
        self.dim = data["dim"]
        self._doc_count = data["doc_count"]
        self._idf = data["idf"]
        self.clear()
        for entry in data["entries"]:
            self._conn.execute(
                "INSERT INTO entries (url, title, text, summary, embedding, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (entry["url"], entry["title"], entry["text"], entry["summary"],
                 bytes.fromhex(entry["embedding"]) if isinstance(entry["embedding"], str) else entry["embedding"], entry["metadata"]),
            )
        self._conn.commit()

    def close(self):
        """Close the database connection."""
        self._conn.close()
