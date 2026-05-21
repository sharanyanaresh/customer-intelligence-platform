"""Tests for complaint retrieval and refusal behavior."""

from __future__ import annotations

import json

import faiss
import numpy as np

from src.rag import retrieve as retrieve_module


def install_fake_index(tmp_path, monkeypatch) -> None:
    """Create a tiny FAISS index and chunk map for deterministic retrieval tests."""
    index_path = tmp_path / "faiss.index"
    chunk_map_path = tmp_path / "chunk_map.json"
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="float32")
    index = faiss.IndexFlatIP(4)
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    payload = {
        "embedding_model": "local-hashing-vectorizer-384",
        "chunk_order": ["c1", "c2"],
        "chunks": {
            "c1": {"text": "debt collection payment dispute", "complaint_id": "1", "product": "Debt collection", "company": "A", "date_received": "2024-01-01", "issue": "Attempts to collect debt not owed"},
            "c2": {"text": "credit report inaccurate account", "complaint_id": "2", "product": "Credit reporting", "company": "B", "date_received": "2024-01-02", "issue": "Incorrect information on your report"},
        },
    }
    chunk_map_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(retrieve_module, "INDEX_PATH", index_path)
    monkeypatch.setattr(retrieve_module, "CHUNK_MAP_PATH", chunk_map_path)
    monkeypatch.setattr(retrieve_module, "RAG_LOG_PATH", tmp_path / "rag.jsonl")
    monkeypatch.setattr(retrieve_module, "_INDEX", None)
    monkeypatch.setattr(retrieve_module, "_PAYLOAD", None)
    monkeypatch.setattr(retrieve_module, "_embed_query", lambda query, embedding_model: np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))


def test_retrieve_returns_results(monkeypatch, tmp_path) -> None:
    install_fake_index(tmp_path, monkeypatch)
    chunks, refused = retrieve_module.retrieve("debt payment", top_k=1, threshold=0.1)

    assert refused is False
    assert len(chunks) == 1
    assert chunks[0]["complaint_id"] == "1"
    assert chunks[0]["score"] >= 0.99


def test_retrieve_applies_metadata_filters(monkeypatch, tmp_path) -> None:
    install_fake_index(tmp_path, monkeypatch)
    chunks, refused = retrieve_module.retrieve("debt payment", top_k=1, filters={"product": "Debt collection"}, threshold=0.1)

    assert refused is False
    assert chunks[0]["product"] == "Debt collection"
    assert chunks[0]["issue"] == "Attempts to collect debt not owed"


def test_retrieve_refuses_when_score_below_threshold(monkeypatch, tmp_path) -> None:
    install_fake_index(tmp_path, monkeypatch)
    chunks, refused = retrieve_module.retrieve("unrelated", top_k=1, threshold=1.1)

    assert refused is True
    assert chunks == []
    assert (tmp_path / "rag.jsonl").exists()

