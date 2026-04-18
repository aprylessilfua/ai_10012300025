# Name: Apryl Essilfua Poku
# Index Number: 10012300025
# CS4241: Part 2b — Memory-based RAG (pure Python; no LangChain / memory frameworks)

"""
ChatMemory stores recent dialogue and builds a single query string for retrieval
by prepending the last two (user, assistant) turns before the new user message.
"""

from __future__ import annotations


class ChatMemory:
    """Keeps conversational turns as plain tuples in a list; contextualizes new queries for FAISS."""

    def __init__(self) -> None:
        self._turns: list[tuple[str, str]] = []

    @property
    def turns(self) -> list[tuple[str, str]]:
        """Copy of stored (user_input, ai_response) pairs, oldest first."""
        return list(self._turns)

    def add_turn(self, user_input: str, ai_response: str) -> None:
        self._turns.append((str(user_input), str(ai_response)))

    def get_contextualized_query(self, new_user_input: str) -> str:
        """
        Prepend the last two turns (if any) as plain text, then the new user input,
        as one string suitable for embedding / FAISS search.
        """
        new_user_input = str(new_user_input)
        last_two = self._turns[-2:]
        if not last_two:
            return new_user_input
        blocks: list[str] = []
        for user_msg, ai_msg in last_two:
            blocks.append(f"User: {user_msg}\nAssistant: {ai_msg}")
        prior = "\n\n".join(blocks)
        return f"{prior}\n\nUser: {new_user_input}"
