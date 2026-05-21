"""Download or generate local datasets for the Customer Intelligence Platform."""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import time
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
UCI_OUTPUT = DATA_DIR / "bank_marketing.csv"
CFPB_OUTPUT = DATA_DIR / "cfpb_complaints_sample.csv"


def sha256_file(path: Path) -> str:
    """Return a SHA-256 hash for a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_hash(path: Path, hash_path: Path) -> str:
    """Write a sidecar hash file and return the hash value."""
    digest = sha256_file(path)
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def download_bytes(url: str, timeout_seconds: int = 60) -> bytes:
    """Download bytes from a URL with a browser-like user agent."""
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={"User-Agent": "customer-intelligence-platform/1.0"},
    )
    response.raise_for_status()
    return response.content


def ingest_uci() -> Path:
    """Download the UCI Bank Marketing dataset and save the main CSV locally."""
    content = download_bytes(UCI_URL)
    with zipfile.ZipFile(io.BytesIO(content)) as outer_zip:
        inner_zip_name = next(name for name in outer_zip.namelist() if name.endswith("bank.zip"))
        with zipfile.ZipFile(io.BytesIO(outer_zip.read(inner_zip_name))) as inner_zip:
            csv_name = next(name for name in inner_zip.namelist() if name.endswith("bank-full.csv"))
            df = pd.read_csv(inner_zip.open(csv_name), sep=";")
    invalid_duration_rows = int((df["duration"] <= 0).sum())
    if invalid_duration_rows:
        df = df[df["duration"] > 0].copy()
        print(f"Removed {invalid_duration_rows} UCI rows with duration <= 0 to satisfy validation rules.")
    df.to_csv(UCI_OUTPUT, index=False)
    return UCI_OUTPUT


def _synthetic_narratives(products: Iterable[str], issues: Iterable[str], companies: Iterable[str]) -> list[dict[str, str]]:
    """Create realistic complaint rows when the public CFPB download is unavailable."""
    templates = [
        "The customer reported repeated calls after requesting written communication only. The complaint mentions confusing payoff information and delayed correction.",
        "The complaint describes a payment that was applied late, causing fees and negative credit reporting. The consumer says support did not explain the reason clearly.",
        "The consumer states that an account showed an unexpected balance after a transfer. They requested documentation and received inconsistent answers.",
        "The complaint says the company did not investigate a disputed transaction within the expected timeline. The consumer is asking for records and correction.",
        "The customer reports difficulty accessing online statements and says the issue created confusion about due dates and minimum payments.",
    ]
    rows: list[dict[str, str]] = []
    for idx, (product, issue, company) in enumerate(zip(products, issues, companies), start=1):
        rows.append(
            {
                "Complaint ID": f"SYN-{idx:06d}",
                "Date received": f"2024-{((idx - 1) % 12) + 1:02d}-{((idx - 1) % 27) + 1:02d}",
                "Product": product,
                "Sub-product": "",
                "Issue": issue,
                "Sub-issue": "",
                "Consumer complaint narrative": templates[idx % len(templates)],
                "Company public response": "Company responded to the consumer and the CFPB.",
                "Company": company,
                "State": random.choice(["CA", "NY", "TX", "FL", "IL", "GA"]),
                "ZIP code": "",
                "Tags": "",
                "Consumer consent provided?": "Consent provided",
                "Submitted via": random.choice(["Web", "Phone", "Referral"]),
                "Date sent to company": f"2024-{((idx - 1) % 12) + 1:02d}-{((idx + 1) % 27) + 1:02d}",
                "Company response to consumer": "Closed with explanation",
                "Timely response?": "Yes",
                "Consumer disputed?": "",
            }
        )
    return rows


def generate_synthetic_cfpb(sample_size: int) -> pd.DataFrame:
    """Generate a local complaint sample with CFPB-like columns and themes."""
    random.seed(42)
    products_pool = [
        "Credit card",
        "Checking or savings account",
        "Mortgage",
        "Credit reporting",
        "Debt collection",
        "Money transfer",
    ]
    issues_pool = [
        "Problem with a purchase shown on your statement",
        "Incorrect information on your report",
        "Attempts to collect debt not owed",
        "Managing an account",
        "Trouble during payment process",
        "Closing on a mortgage",
    ]
    companies_pool = [
        "Example National Bank",
        "Sample Financial Services",
        "Demo Credit Bureau",
        "Student Loan Servicing Co",
        "Community Payments Inc",
    ]
    products = [random.choice(products_pool) for _ in range(sample_size)]
    issues = [random.choice(issues_pool) for _ in range(sample_size)]
    companies = [random.choice(companies_pool) for _ in range(sample_size)]
    return pd.DataFrame(_synthetic_narratives(products, issues, companies))


def ingest_cfpb(sample_size: int) -> tuple[Path, bool]:
    """Download a CFPB sample, falling back to synthetic data if needed."""
    try:
        content = download_bytes(CFPB_URL, timeout_seconds=120)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
            usecols = [
                "Complaint ID",
                "Date received",
                "Product",
                "Issue",
                "Consumer complaint narrative",
                "Company",
                "State",
                "Submitted via",
                "Company response to consumer",
                "Timely response?",
            ]
            chunks = pd.read_csv(
                archive.open(csv_name),
                usecols=lambda column: column in usecols,
                chunksize=max(sample_size * 4, 1000),
                low_memory=False,
            )
            frames = []
            total = 0
            for chunk in chunks:
                chunk = chunk.dropna(subset=["Consumer complaint narrative"])
                frames.append(chunk)
                total += len(chunk)
                if total >= sample_size:
                    break
            df = pd.concat(frames, ignore_index=True).head(sample_size)
        if df.empty:
            raise ValueError("CFPB download contained no rows with narratives.")
        df.to_csv(CFPB_OUTPUT, index=False)
        return CFPB_OUTPUT, False
    except Exception as exc:
        print(f"CFPB public download unavailable, generating synthetic sample. Reason: {exc}")
        df = generate_synthetic_cfpb(sample_size)
        df.to_csv(CFPB_OUTPUT, index=False)
        return CFPB_OUTPUT, True


def print_summary(path: Path, hash_path: Path, started_at: float, synthetic: bool = False) -> None:
    """Print a compact ingestion summary for one generated file."""
    df = pd.read_csv(path)
    digest = write_hash(path, hash_path)
    elapsed = time.perf_counter() - started_at
    source = "synthetic fallback" if synthetic else "public download"
    print(f"{path} | rows={len(df)} | hash={digest} | seconds={elapsed:.2f} | source={source}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ingest UCI Bank Marketing and CFPB complaint samples.")
    parser.add_argument("--sample", type=int, default=5000, help="Number of CFPB complaint records to save.")
    parser.add_argument("--skip-uci", action="store_true", help="Skip UCI download.")
    parser.add_argument("--skip-cfpb", action="store_true", help="Skip CFPB download or generation.")
    return parser.parse_args()


def main() -> None:
    """Run ingestion and write local data plus hash sidecars."""
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_uci:
        start = time.perf_counter()
        uci_path = ingest_uci()
        print_summary(uci_path, DATA_DIR / "uci_hash.txt", start)

    if not args.skip_cfpb:
        start = time.perf_counter()
        cfpb_path, synthetic = ingest_cfpb(args.sample)
        print_summary(cfpb_path, DATA_DIR / "cfpb_hash.txt", start, synthetic=synthetic)


if __name__ == "__main__":
    main()
