"""Similarity retrieval and refusal logic for complaint RAG."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", PROJECT_ROOT / "data" / "faiss.index"))
CHUNK_MAP_PATH = PROJECT_ROOT / "data" / "chunk_map.json"
RAG_LOG_PATH = PROJECT_ROOT / "monitoring" / "rag_requests.jsonl"
DEFAULT_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger("rag-retriever")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

_INDEX: faiss.Index | None = None
_PAYLOAD: dict[str, Any] | None = None
_MODEL: Any | None = None


def _load_payload() -> dict[str, Any]:
    """Load chunk metadata from disk."""
    if not CHUNK_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNK_MAP_PATH}. Run python src/rag/build_index.py first.")
    return json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))


def load_resources() -> tuple[faiss.Index, dict[str, Any]]:
    """Load FAISS index and chunk map once per process."""
    global _INDEX, _PAYLOAD
    if _INDEX is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Missing {INDEX_PATH}. Run python src/rag/build_index.py first.")
        _INDEX = faiss.read_index(str(INDEX_PATH))
    if _PAYLOAD is None:
        _PAYLOAD = _load_payload()
    return _INDEX, _PAYLOAD


def _embed_query(query: str, embedding_model: str) -> np.ndarray:
    """Embed one query with the same embedding family as the index."""
    global _MODEL
    if embedding_model == MODEL_NAME:
        try:
            from sentence_transformers import SentenceTransformer

            if _MODEL is None:
                _MODEL = SentenceTransformer(MODEL_NAME)
            vector = _MODEL.encode([query], normalize_embeddings=True)
            return np.asarray(vector, dtype="float32")
        except Exception as exc:
            logger.warning("MiniLM query embedding unavailable, falling back to hashing: %s", exc)

    vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm=None)
    sparse = vectorizer.transform([query])
    return normalize(sparse, norm="l2").astype("float32").toarray()


def _passes_filters(chunk: dict[str, str], filters: dict[str, str] | None) -> bool:
    """Return whether a chunk matches optional metadata filters."""
    if not filters:
        return True

    for key in ["product", "company", "issue"]:
        expected = filters.get(key)
        if expected and expected.lower() not in chunk.get(key, "").lower():
            return False

    date_from = filters.get("date_from")
    date_to = filters.get("date_to")
    received = chunk.get("date_received", "")
    if received and (date_from or date_to):
        try:
            received_date = date.fromisoformat(received[:10])
            if date_from and received_date < date.fromisoformat(date_from):
                return False
            if date_to and received_date > date.fromisoformat(date_to):
                return False
        except ValueError:
            return False

    return True


def _log_call(query: str, scores: list[float], latency_ms: float, refused: bool) -> None:
    """Append a JSON-lines retrieval log entry for monitoring."""
    RAG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "query": query[:160],
        "top_scores": scores,
        "latency_ms": latency_ms,
        "refused": refused,
        "timestamp": time.time(),
    }
    with RAG_LOG_PATH.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(entry) + "\n")


def retrieve(
    query: str,
    top_k: int = 5,
    filters: dict[str, str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[dict[str, Any]], bool]:
    """Retrieve complaint chunks and return (chunks, refused)."""
    started = time.perf_counter()
    index, payload = load_resources()
    embedding_model = payload.get("embedding_model", "local-hashing-vectorizer-384")
    vector = _embed_query(query, embedding_model)
    search_k = min(max(top_k * 8, top_k), index.ntotal)
    scores, indices = index.search(vector, search_k)

    chunk_order = payload["chunk_order"]
    chunks_by_id = payload["chunks"]
    results: list[dict[str, Any]] = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        chunk_id = chunk_order[int(index_position)]
        chunk = chunks_by_id[chunk_id]
        if not _passes_filters(chunk, filters):
            continue
        result = {"chunk_id": chunk_id, "score": float(score), **chunk}
        results.append(result)
        if len(results) >= top_k:
            break

    best_score = results[0]["score"] if results else 0.0
    refused = best_score < threshold
    if refused:
        results = []

    latency_ms = (time.perf_counter() - started) * 1000
    top_scores = [float(score) for score in scores[0][:top_k]]
    _log_call(query, top_scores, latency_ms, refused)
    logger.info("retrieve query=%s scores=%s latency_ms=%.2f refused=%s", query[:80], top_scores, latency_ms, refused)
    return results, refused

