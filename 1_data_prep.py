# Name: Apryl Essilfua Poku
# Index Number: 10012300025
# CS4241: Part A - Data Engineering & Preparation


from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_CANDIDATES = [
    DATA_DIR / "Ghana Election_Result.csv",
    DATA_DIR / "Ghana_Election_Result.csv",
]
PDF_PATH = DATA_DIR / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"

CHUNK_WORDS = 500
OVERLAP_WORDS = 50


def resolve_csv_path() -> Path:
    for path in CSV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No CSV found. Tried: {[str(p) for p in CSV_CANDIDATES]}"
    )


def extract_text_from_csv(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    parts: list[str] = []
    parts.append(" ".join(str(c) for c in df.columns))
    for _, row in df.iterrows():
        parts.append(" ".join(str(v) for v in row.values))
    return "\n".join(parts)


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pieces: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pieces.append(t)
    return "\n".join(pieces)


def sliding_window_chunks(
    text: str,
    chunk_size: int = CHUNK_WORDS,
    overlap: int = OVERLAP_WORDS,
) -> list[str]:
    """Split whitespace-separated words into overlapping windows (pure Python)."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(" ".join(words[start:end]))
        if end >= n:
            break
        start += step
    return chunks


def main() -> None:
    csv_path = resolve_csv_path()
    logger.info("Reading CSV: %s", csv_path)
    csv_text = extract_text_from_csv(csv_path)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    logger.info("Reading PDF: %s", PDF_PATH)
    pdf_text = extract_text_from_pdf(PDF_PATH)

    csv_chunks = sliding_window_chunks(csv_text)
    pdf_chunks = sliding_window_chunks(pdf_text)

    logger.info("Chunks created for CSV document: %d", len(csv_chunks))
    logger.info("Chunks created for PDF document: %d", len(pdf_chunks))

    print(f"CSV chunks: {len(csv_chunks)}")
    print(f"PDF chunks: {len(pdf_chunks)}")


if __name__ == "__main__":
    main()
