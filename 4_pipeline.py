# Name: Apryl Essilfua Poku
# Index Number: 10012300025
# CS4241: Part D - Full RAG Pipeline Implementation

"""
End-to-end RAG: FAISS retrieval -> cross-encoder rerank -> context window -> strict prompt -> Gemini (google.genai).
Plain procedural Python only (no LangChain / LlamaIndex chaining).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

from google import genai

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
_GEMINI_MODEL_NAME = "gemini-1.5-flash"
_RETRIEVAL_K = 10
_MAX_CONTEXT_WORDS = 1500

_gen_client: genai.Client | None = None
_reranker = None


def _load_sibling(module_qualname: str, file_name: str):
    path = _ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_qualname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_retrieval_mod = _load_sibling("student_retrieval", "2_retrieval.py")
_reranking_mod = _load_sibling("student_reranking", "2c_reranking.py")
_prompt_mod = _load_sibling("student_prompt", "3_prompt_engineering.py")

FaissChunkRetriever = _retrieval_mod.FaissChunkRetriever
manage_context_window = _prompt_mod.manage_context_window
build_strict_prompt = _prompt_mod.build_strict_prompt
CrossEncoderReranker = _reranking_mod.CrossEncoderReranker


def _get_genai_client() -> genai.Client:
    """Build a google.genai client (reads ``GEMINI_API_KEY`` from the environment)."""
    global _gen_client
    if _gen_client is not None:
        return _gen_client
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    client = genai.Client()
    _gen_client = client
    logger.info("google.genai client initialized; model=%s", _GEMINI_MODEL_NAME)
    return client


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def _llm_text(response) -> str:
    """Normalize ``GenerateContentResponse`` (google.genai) to plain text."""
    try:
        t = getattr(response, "text", None)
        if t:
            return str(t).strip()
    except Exception:
        pass
    parts: list[str] = []
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            pt = getattr(part, "text", None)
            if pt:
                parts.append(str(pt))
    return "\n".join(parts).strip() if parts else str(response)


def run_rag_pipeline(query: str, retriever) -> dict[str, object]:
    """
    Manual flow: query -> FAISS -> rerank -> context -> prompt -> Gemini -> answer.

    ``retriever`` must provide ``.search(query: str, k: int) -> list[tuple[str, float]]``
    (e.g. ``FaissChunkRetriever`` from ``2_retrieval.py``).

    Returns a dict with keys: ``answer`` (str), ``retrieval_hits`` (list of (chunk, FAISS score)),
    ``reranked`` (list of (chunk, cross-encoder score)), ``prompt`` (str).
    """
    reranker = _get_reranker()

    # 1) FAISS retrieval
    hits = retriever.search(query, k=_RETRIEVAL_K)
    print("\n========== RETRIEVED CHUNKS (FAISS) ==========")
    if not hits:
        print("(no hits)")
    for i, (chunk, score) in enumerate(hits, start=1):
        print(f"\n--- Hit {i} | FAISS score (squared L2; lower is closer) = {score} ---")
        print(chunk)

    # 2) Re-ranking
    chunks = [c for c, _ in hits]
    faiss_scores = [s for _, s in hits] if hits else []
    reranked = reranker.rerank(query, chunks, retrieval_scores=faiss_scores or None)

    print("\n========== RE-RANKED CHUNKS (Cross-Encoder; higher is better) ==========")
    if not reranked:
        print("(no chunks to rerank)")
    for i, (chunk, score) in enumerate(reranked, start=1):
        print(f"\n--- Rank {i} after rerank | CE score = {score} ---")
        print(chunk)

    # 3) Context window
    ordered_chunks = [c for c, _ in reranked]
    context = manage_context_window(ordered_chunks, max_words=_MAX_CONTEXT_WORDS)

    # 4) Prompt construction
    prompt = build_strict_prompt(query, context)

    print("\n========== FINAL PROMPT SENT TO LLM ==========")
    print(prompt)

    # 5) Gemini (google.genai; client only when calling the API)
    client = _get_genai_client()
    response = client.models.generate_content(
        model=_GEMINI_MODEL_NAME,
        contents=prompt,
    )
    answer = _llm_text(response)

    print("\n========== FINAL LLM RESPONSE ==========")
    print(answer)

    return {
        "answer": answer,
        "retrieval_hits": hits,
        "reranked": reranked,
        "prompt": prompt,
    }


if __name__ == "__main__":
    _data_prep = _load_sibling("student_data_prep", "1_data_prep.py")

    if not _data_prep.PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {_data_prep.PDF_PATH}")

    logger.info("Extracting and chunking PDF via 1_data_prep: %s", _data_prep.PDF_PATH)
    _pdf_text = _data_prep.extract_text_from_pdf(_data_prep.PDF_PATH)
    _pdf_chunks = _data_prep.sliding_window_chunks(_pdf_text)
    logger.info("Indexed PDF chunks: %d", len(_pdf_chunks))

    _retriever = FaissChunkRetriever()
    _retriever.index_chunks(_pdf_chunks)

    run_rag_pipeline(
        "What are the key policy initiatives for 2025?",
        _retriever,
    )
