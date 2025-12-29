"""In-memory conversation history manager."""
from __future__ import annotations

from typing import Dict, List, Deque
from collections import deque

# Maximum number of messages to keep in history per chat
MAX_HISTORY_LENGTH = 20

class ConversationManager:
    """Manages chat history for context-aware responses."""

    def __init__(self, max_length: int = MAX_HISTORY_LENGTH) -> None:
        self._histories: Dict[int, Deque[Dict[str, str]]] = {}
        self._max_length = max_length

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        """Add a message to the history for the given chat."""
        if chat_id not in self._histories:
            self._histories[chat_id] = deque(maxlen=self._max_length)

        self._histories[chat_id].append({"role": role, "content": content})

    def get_history(self, chat_id: int) -> List[Dict[str, str]]:
        """Retrieve the conversation history for the given chat."""
        if chat_id not in self._histories:
            return []

        return list(self._histories[chat_id])

    def clear_history(self, chat_id: int) -> None:
        """Clear the history for the given chat."""
        if chat_id in self._histories:
            del self._histories[chat_id]
