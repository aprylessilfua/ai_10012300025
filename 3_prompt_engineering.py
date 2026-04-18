# Name: Apryl Essilfua Poku
# Index Number: 10012300025
# CS4241: Part C - Prompt Engineering

"""
Context window packing and strict RAG-style prompts using plain Python strings only.
No LangChain PromptTemplate or similar.
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SEPARATOR = "\n\n"


def _word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def manage_context_window(chunks: list[str], max_words: int = 1500) -> str:
    """
    Combine retrieved chunks in rank order (index 0 = highest rank).

    If the combined text exceeds ``max_words``, whole chunks are dropped from the
    lowest rank (end of the list) until the rest fits. If a single chunk still
    exceeds ``max_words``, it is truncated by word count to fit.
    """
    cleaned = [str(c).strip() for c in chunks if c is not None and str(c).strip()]
    if not cleaned:
        logger.info("manage_context_window: no chunks; returning empty string.")
        return ""
    if max_words <= 0:
        logger.info("manage_context_window: max_words=%s; returning empty string.", max_words)
        return ""

    active = cleaned[:]
    dropped = 0
    while active:
        text = CHUNK_SEPARATOR.join(active)
        wc = _word_count(text)
        if wc <= max_words:
            logger.info(
                "manage_context_window: using %d chunk(s), ~%d words (limit %d); dropped %d low-ranked chunk(s).",
                len(active),
                wc,
                max_words,
                dropped,
            )
            return text
        if len(active) == 1:
            words = active[0].split()
            truncated = " ".join(words[:max_words])
            logger.info(
                "manage_context_window: single chunk truncated from %d to %d words (limit %d).",
                len(words),
                _word_count(truncated),
                max_words,
            )
            return truncated
        active = active[:-1]
        dropped += 1

    logger.info("manage_context_window: all chunks dropped; returning empty string.")
    return ""


def build_strict_prompt(query: str, context: str) -> str:
    """
    Build a strict, context-only answer prompt using an f-string (no template libraries).
    """
    instructions = (
        "You are a strict data assistant. Answer ONLY using the provided context. "
        "If the answer is not in the context, say 'I do not have enough information'. "
        "Do not make up facts."
    )
    prompt = f"""{instructions}

### Context
{context}

### Question
{query}

### Answer
"""
    logger.info("build_strict_prompt: final constructed prompt follows (between markers).")
    logger.info("----- BEGIN PROMPT -----")
    logger.info("%s", prompt)
    logger.info("----- END PROMPT -----")
    return prompt
