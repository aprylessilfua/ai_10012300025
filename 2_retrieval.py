# Name: Apryl Essilfua Poku
# Index Number: 10012300025
# CS4241: Part B - Custom Retrieval System

"""
Embedding + FAISS IndexFlatL2 storage and top-k retrieval using sentence-transformers only.
No LangChain, LlamaIndex, ChromaDB, or other RAG frameworks.
"""

from __future__ import annotations

import logging
from typing import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_K = 3


class FaissChunkRetriever:
    """
    Embeds text chunks with SentenceTransformer, stores vectors in faiss.IndexFlatL2,
    and retrieves by squared L2 distance (FAISS default for IndexFlatL2; lower = closer).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        dim_fn = getattr(self._model, "get_embedding_dimension", None)
        self._dim = int(dim_fn()) if dim_fn else int(self._model.get_sentence_embedding_dimension())
        self._index = faiss.IndexFlatL2(self._dim)
        self._chunks: list[str] = []

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def index_chunks(self, chunks: Sequence[str]) -> None:
        """Replace the index with embeddings for the given chunk strings."""
        self._chunks = [c for c in chunks if c is not None and str(c).strip() != ""]
        self._index = faiss.IndexFlatL2(self._dim)
        if not self._chunks:
            logger.info("No non-empty chunks to index.")
            return
        vectors = self._model.encode(
            self._chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        mat = np.asarray(vectors, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[1] != self._dim:
            raise ValueError(f"Expected embeddings shape (n, {self._dim}), got {mat.shape}")
        self._index.add(mat)
        logger.info("Indexed %d chunks into FAISS IndexFlatL2 (dim=%d).", len(self._chunks), self._dim)

    def search(self, query: str, k: int = DEFAULT_K) -> list[tuple[str, float]]:
        """
        Embed the query, run top-k retrieval, return (chunk_text, raw_score) pairs.

        raw_score is the value returned by FAISS for IndexFlatL2 (squared L2 distance;
        smaller means more similar under Euclidean distance in embedding space).
        """
        if not self._chunks or self._index.ntotal == 0:
            logger.info("Search skipped: index is empty.")
            return []
        k_eff = min(int(k), len(self._chunks))
        q = self._model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        q = np.asarray(q, dtype=np.float32)
        distances, indices = self._index.search(q, k_eff)
        row_d = distances[0].tolist()
        row_i = indices[0].tolist()
        logger.info(
            "Retrieval raw FAISS scores (IndexFlatL2 squared L2; lower is closer): %s",
            row_d,
        )
        for rank, (idx, dist) in enumerate(zip(row_i, row_d, strict=True), start=1):
            logger.info("  rank=%d index=%d score=%s", rank, int(idx), dist)
        out: list[tuple[str, float]] = []
        for idx, dist in zip(row_i, row_d, strict=True):
            if idx < 0:
                continue
            out.append((self._chunks[int(idx)], float(dist)))
        return out


def embed_and_index(chunks: Sequence[str], model_name: str = DEFAULT_MODEL) -> FaissChunkRetriever:
    """Convenience: build a retriever and index the given chunks."""
    retriever = FaissChunkRetriever(model_name=model_name)
    retriever.index_chunks(chunks)
    return retriever


if __name__ == "__main__":
    demo = FaissChunkRetriever()
    demo.index_chunks(
        [
            "Ghana election results by region.",
            "Budget deficit and fiscal policy measures.",
            "Unrelated text about cooking recipes.",
        ]
    )
    results = demo.search("What was the election outcome?", k=3)
    for chunk, score in results:
        print(score, "->", chunk[:80] + ("..." if len(chunk) > 80 else ""))
