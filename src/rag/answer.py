"""FastAPI complaint intelligence endpoint."""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieve import retrieve


PROMPT_VERSION = "v1.0"
MAX_PROMPT_TOKENS = int(os.getenv("MAX_RAG_PROMPT_TOKENS", "1500"))
GENERATION_WARNING_SECONDS = float(os.getenv("RAG_GENERATION_WARNING_SECONDS", "3.0"))
logger = logging.getLogger("rag-answer")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
SYSTEM_PROMPT = (
    "You are a complaint analyst. Answer ONLY using the evidence provided. "
    "Do not give legal or financial advice. Do not reference information outside the evidence."
)


class ComplaintQuestion(BaseModel):
    """Request body for complaint question answering."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=5)
    product: str | None = None
    company: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    issue: str | None = None


class ComplaintAnswer(BaseModel):
    """Response body for complaint question answering."""

    answer: str
    evidence_ids: list[str]
    evidence_sufficiency_note: str
    prompt_version: str = PROMPT_VERSION
    retrieval_latency_ms: float
    generation_latency_ms: float
    refused: bool


def _filters(payload: ComplaintQuestion) -> dict[str, str]:
    """Build non-empty retrieval filters."""
    return {
        key: value
        for key, value in {
            "product": payload.product,
            "company": payload.company,
            "date_from": payload.date_from,
            "date_to": payload.date_to,
            "issue": payload.issue,
        }.items()
        if value
    }


def _extract_themes(chunks: list[dict]) -> list[str]:
    """Summarize complaint themes from retrieved issue metadata."""
    seen: list[str] = []
    for chunk in chunks:
        issue = chunk.get("issue", "Unknown issue")
        if issue not in seen:
            seen.append(issue)
    return seen[:3]


def _local_evidence_answer(question: str, chunks: list[dict]) -> str:
    """Generate a grounded local answer without paid external LLM calls."""
    themes = _extract_themes(chunks)
    products = sorted({chunk.get("product", "unknown") for chunk in chunks})
    companies = sorted({chunk.get("company", "unknown") for chunk in chunks})
    evidence_note = " ".join(chunk["text"][:240] for chunk in chunks[:2])
    return (
        f"Based on the retrieved complaint evidence, the main themes are: {', '.join(themes)}. "
        f"The evidence is mostly associated with product(s): {', '.join(products[:3])} "
        f"and company record(s): {', '.join(companies[:3])}. "
        f"A representative evidence summary is: {evidence_note}"
    )


def _token_count(text: str) -> int:
    """Count whitespace tokens for a lightweight prompt budget guardrail."""
    return len(text.split())


def _build_guarded_prompt(question: str, chunks: list[dict]) -> tuple[str, list[dict], int]:
    """Build a prompt under the token limit by trimming oldest retrieved chunks first."""
    active_chunks = chunks.copy()
    while active_chunks:
        evidence = "\n".join(f"[{idx + 1}] {chunk['text']}" for idx, chunk in enumerate(active_chunks))
        prompt = f"System: {SYSTEM_PROMPT}\nUser: Evidence:\n{evidence}\n\nQuestion: {question}"
        token_count = _token_count(prompt)
        if token_count <= MAX_PROMPT_TOKENS:
            return prompt, active_chunks, token_count
        active_chunks = active_chunks[:-1]

    prompt = f"System: {SYSTEM_PROMPT}\nUser: Evidence:\n\nQuestion: {question}"
    return prompt, [], _token_count(prompt)


def answer_question(payload: ComplaintQuestion) -> ComplaintAnswer:
    """Retrieve evidence and generate a complaint-grounded answer."""
    retrieval_start = time.perf_counter()
    chunks, refused = retrieve(payload.question, top_k=5, filters=_filters(payload))
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    if refused:
        return ComplaintAnswer(
            answer="Insufficient evidence to answer. Please refine your question or check your filters.",
            evidence_ids=[],
            evidence_sufficiency_note="No chunks crossed threshold.",
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=0.0,
            refused=True,
        )

    generation_start = time.perf_counter()
    _, guarded_chunks, token_count = _build_guarded_prompt(payload.question, chunks)
    answer = _local_evidence_answer(payload.question, guarded_chunks)
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000
    if generation_latency_ms > GENERATION_WARNING_SECONDS * 1000:
        logger.warning("RAG generation exceeded %.1fs with %s prompt tokens", GENERATION_WARNING_SECONDS, token_count)

    return ComplaintAnswer(
        answer=answer,
        evidence_ids=sorted({str(chunk["complaint_id"]) for chunk in guarded_chunks}),
        evidence_sufficiency_note=f"{len(guarded_chunks)} chunks used after token guardrail; {len(chunks)} chunks crossed the retrieval threshold.",
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        refused=False,
    )


app = FastAPI(title="Complaint Intelligence RAG API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Return lightweight RAG API health."""
    index_path = PROJECT_ROOT / "data" / "faiss.index"
    chunk_map_path = PROJECT_ROOT / "data" / "chunk_map.json"
    status = "ok" if index_path.exists() and chunk_map_path.exists() else "degraded"
    return {
        "status": status,
        "index_version": str(index_path),
        "prompt_version": PROMPT_VERSION,
    }


@app.post("/ask-complaints", response_model=ComplaintAnswer)
def ask_complaints(payload: ComplaintQuestion) -> ComplaintAnswer:
    """Answer complaint questions using local retrieved evidence."""
    return answer_question(payload)
