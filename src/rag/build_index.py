"""Build a local FAISS index for CFPB complaint retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_PATH = PROJECT_ROOT / "data" / "cfpb_complaints_sample.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "faiss.index"
CHUNK_MAP_PATH = PROJECT_ROOT / "data" / "chunk_map.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SAFE_COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Company": "company",
    "Issue": "issue",
    "Consumer complaint narrative": "narrative",
}


def sanitize_text(text: str) -> str:
    """Remove obvious account-like numbers, emails, phone numbers, and extra whitespace."""
    cleaned = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[email_removed]", str(text))
    cleaned = re.sub(r"\b(?:\+?\d[\d -]{7,}\d)\b", "[number_removed]", cleaned)
    cleaned = re.sub(r"\b\d{5}(?:-\d{4})?\b", "[zip_removed]", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def sentence_aware_chunks(text: str, max_tokens: int = 300, overlap: int = 30) -> list[str]:
    """Split text into roughly token-bounded chunks while preserving sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", sanitize_text(text))
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if current and current_len + len(words) > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap < len(current) else current
            current_len = len(current)
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def load_safe_complaints(limit: int | None = None) -> pd.DataFrame:
    """Load only non-PII columns needed for retrieval metadata and narratives."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run python src/data_pipeline/ingest.py --sample 5000 first.")
    df = pd.read_csv(DATA_PATH, usecols=lambda column: column in SAFE_COLUMNS)
    df = df.rename(columns=SAFE_COLUMNS)
    df = df.dropna(subset=["narrative", "complaint_id", "product", "issue"])
    if "company" not in df.columns:
        df["company"] = "unknown"
    if "date_received" not in df.columns:
        df["date_received"] = ""
    if limit:
        df = df.head(limit)
    return df


def build_chunk_map(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Create chunk metadata keyed by stable chunk IDs."""
    chunk_map: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        for chunk_index, chunk_text in enumerate(sentence_aware_chunks(row["narrative"])):
            raw_id = f"{row['complaint_id']}::{chunk_index}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
            chunk_map[chunk_id] = {
                "text": chunk_text,
                "complaint_id": str(row["complaint_id"]),
                "product": str(row["product"]),
                "company": str(row.get("company", "unknown")),
                "date_received": str(row.get("date_received", "")),
                "issue": str(row["issue"]),
            }
    return chunk_map


def embed_texts(texts: list[str]) -> tuple[np.ndarray, str]:
    """Embed texts with MiniLM when available, otherwise use a deterministic local fallback."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32"), MODEL_NAME
    except Exception as exc:
        print(f"MiniLM embedding unavailable, using local HashingVectorizer fallback. Reason: {exc}")
        vectorizer = HashingVectorizer(n_features=384, alternate_sign=False, norm=None)
        sparse = vectorizer.transform(texts)
        dense = normalize(sparse, norm="l2").astype("float32").toarray()
        return dense, "local-hashing-vectorizer-384"


def persist_index(chunk_map: dict[str, dict[str, str]]) -> tuple[int, float, int, str]:
    """Embed chunks, build a FAISS inner-product index, and persist artifacts."""
    chunk_ids = list(chunk_map)
    texts = [chunk_map[chunk_id]["text"] for chunk_id in chunk_ids]
    started = time.perf_counter()
    embeddings, embedding_model = embed_texts(texts)
    elapsed = time.perf_counter() - started

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    payload = {
        "embedding_model": embedding_model,
        "chunk_order": chunk_ids,
        "chunks": chunk_map,
    }
    CHUNK_MAP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(chunk_ids), elapsed, INDEX_PATH.stat().st_size, embedding_model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build CFPB complaint FAISS index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional complaint row limit for fast local demos.")
    return parser.parse_args()


def main() -> None:
    """Build and persist the complaint retrieval index."""
    args = parse_args()
    print("PII CHECK: loading only safe metadata columns and sanitized complaint narratives.")
    df = load_safe_complaints(limit=args.limit)
    chunk_map = build_chunk_map(df)
    total_chunks, embedding_time, index_size, embedding_model = persist_index(chunk_map)
    print(f"total_chunks={total_chunks} | embedding_time_seconds={embedding_time:.2f} | index_file_size_bytes={index_size} | embedding_model={embedding_model}")


if __name__ == "__main__":
    main()

