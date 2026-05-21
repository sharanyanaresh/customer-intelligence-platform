"""Summarize RAG retrieval and generation monitoring logs."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "monitoring" / "rag_requests.jsonl"
REPORT_PATH = PROJECT_ROOT / "monitoring" / "rag_monitoring_report.json"


def load_events() -> list[dict]:
    """Load JSON-lines RAG events, skipping malformed lines."""
    if not LOG_PATH.exists():
        return []
    events: list[dict] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def summarize(events: list[dict]) -> dict[str, float | int]:
    """Compute RAG monitoring metrics."""
    total = len(events)
    if total == 0:
        return {
            "request_count": 0,
            "retrieval_hit_rate": 0.0,
            "empty_retrieval_count": 0,
            "average_top_1_similarity": 0.0,
            "refusal_rate": 0.0,
            "avg_token_count": 0.0,
            "avg_generation_latency_ms": 0.0,
            "avg_total_latency_ms": 0.0,
        }

    refused_count = sum(1 for event in events if event.get("refused"))
    top_scores = [event.get("top_scores", [0.0])[0] if event.get("top_scores") else 0.0 for event in events]
    retrieval_latencies = [float(event.get("latency_ms", 0.0)) for event in events]
    generation_latencies = [float(event.get("generation_latency_ms", 0.0)) for event in events]
    token_counts = [len(str(event.get("query", "")).split()) for event in events]

    return {
        "request_count": total,
        "retrieval_hit_rate": round((total - refused_count) / total, 4),
        "empty_retrieval_count": refused_count,
        "average_top_1_similarity": round(sum(top_scores) / total, 4),
        "refusal_rate": round(refused_count / total, 4),
        "avg_token_count": round(sum(token_counts) / total, 2),
        "avg_generation_latency_ms": round(sum(generation_latencies) / total, 2),
        "avg_total_latency_ms": round((sum(retrieval_latencies) + sum(generation_latencies)) / total, 2),
    }


def print_table(summary: dict[str, float | int]) -> None:
    """Print a clean terminal table."""
    print("| metric | value |")
    print("|---|---:|")
    for key, value in summary.items():
        print(f"| {key} | {value} |")


def main() -> None:
    """Write the JSON monitoring report and print a summary table."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events = load_events()
    summary = summarize(events)
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

