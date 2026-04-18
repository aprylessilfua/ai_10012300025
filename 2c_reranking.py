# Name: Apryl Essilfua Poku
# Index Number: 10012300025
# CS4241: Part 2c — Cross-encoder re-ranking (pure sentence_transformers; no LangChain)

"""
Re-rank retrieved chunks with a local CrossEncoder. Higher cross-encoder score = more relevant
(ms-marco style logits). Optional retrieval scores (e.g. FAISS distances) are logged as "before".
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _preview(text: str, max_len: int = 72) -> str:
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= max_len else one_line[: max_len - 3] + "..."


class CrossEncoderReranker:
    """Scores (query, chunk) pairs with CrossEncoder and returns chunks sorted by score descending."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER) -> None:
        logger.info("Loading CrossEncoder: %s", model_name)
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: Sequence[str],
        retrieval_scores: Sequence[float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Pair `query` with each chunk, score with the cross-encoder, sort by score (highest first).

        If `retrieval_scores` is provided (same length as `chunks`), logs those as pre-rerank
        scores (e.g. FAISS L2 — lower is closer). Always logs cross-encoder scores before and
        after sorting (retrieval order vs final order).
        """
        chunks_list = [str(c) for c in chunks]
        if not chunks_list:
            logger.info("Rerank skipped: empty chunk list.")
            return []

        if retrieval_scores is not None and len(retrieval_scores) != len(chunks_list):
            raise ValueError("retrieval_scores must be the same length as chunks when provided.")

        for i, c in enumerate(chunks_list):
            if retrieval_scores is not None:
                logger.info(
                    "Before rerank | rank=%d retrieval_score=%s | %s",
                    i + 1,
                    float(retrieval_scores[i]),
                    _preview(c),
                )
            else:
                logger.info("Before rerank | rank=%d (no retrieval_score) | %s", i + 1, _preview(c))

        pairs = [[query, c] for c in chunks_list]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)
        ce_scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1).tolist()

        logger.info("Cross-encoder scores in retrieval order (higher = more relevant): %s", ce_scores)

        combined = list(zip(chunks_list, ce_scores, strict=True))
        combined.sort(key=lambda x: x[1], reverse=True)

        after_scores = [s for _, s in combined]
        logger.info("After rerank | cross-encoder scores (high -> low): %s", after_scores)
        for rank, (chunk, score) in enumerate(combined, start=1):
            logger.info("After rerank | rank=%d ce_score=%s | %s", rank, score, _preview(chunk))

        return combined


def rerank_chunks(
    query: str,
    chunks: Sequence[str],
    retrieval_scores: Sequence[float] | None = None,
    model_name: str = DEFAULT_CROSS_ENCODER,
) -> list[tuple[str, float]]:
    """Convenience: one-shot rerank using a freshly loaded CrossEncoder."""
    return CrossEncoderReranker(model_name=model_name).rerank(
        query, chunks, retrieval_scores=retrieval_scores
    )


if __name__ == "__main__":
    q = "What are Ghana election results?"
    ch = [
        "The 2025 budget outlines fiscal consolidation.",
        "Regional vote totals for Ghana's presidential election.",
        "Cooking tips for jollof rice.",
    ]
    prior = [1.2, 0.5, 2.0]
    rer = CrossEncoderReranker()
    out = rer.rerank(q, ch, retrieval_scores=prior)
    for chunk, score in out:
        print(f"{score:.4f}  {_preview(chunk)}")
