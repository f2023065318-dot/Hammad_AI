"""
chat_history.py
---------------
This module manages the conversation history for the chatbot.
It stores past messages and formats them for the LLM.
"""

from typing import List, Dict
from datetime import datetime


class ChatHistory:
    """Manages conversation history between user and assistant."""

    def __init__(self, max_turns: int = 10):
        """
        Initializes chat history.

        Args:
            max_turns: Maximum number of conversation turns to keep in memory.
        """
        self.messages: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_user_message(self, message: str) -> None:
        """Adds a user message to history."""
        self.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })
        self._trim()

    def add_assistant_message(self, message: str) -> None:
        """Adds an assistant message to history."""
        self.messages.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })
        self._trim()

    def _trim(self) -> None:
        """Keeps only the last N turns (user + assistant pairs)."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        """Returns all messages."""
        return self.messages

    def get_recent_context(self, n: int = 3) -> str:
        """
        Returns the last N turns as a formatted string
        for the LLM to understand context.

        Args:
            n: Number of recent turns to include.

        Returns:
            Formatted conversation string.
        """
        if not self.messages:
            return ""

        recent = self.messages[-(n * 2):]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "HammadBot"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clears all conversation history."""
        self.messages = []

    def __len__(self) -> int:
        """Returns number of messages in history."""
        return len(self.messages)


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    history = ChatHistory(max_turns=5)

    # Simulate conversation
    history.add_user_message("Hi, who are you?")
    history.add_assistant_message("I am HammadBot, your personal assistant.")
    history.add_user_message("What is Muhammad Waqas's email?")
    history.add_assistant_message("Muhammad Waqas's email is malikhammad6445@gmail.com.")
    history.add_user_message("What about his projects?")

    print("=" * 60)
    print("Testing Chat History for HammadBot")
    print("=" * 60)

    print(f"\nTotal messages: {len(history)}")
    print("\n--- All Messages ---")
    for msg in history.get_messages():
        print(f"[{msg['role']}]: {msg['content']}")

    print("\n--- Recent Context (last 2 turns) ---")
    print(history.get_recent_context(n=2))

    print("\n--- After Clear ---")
    history.clear()
    print(f"Total messages: {len(history)}")

    print("\n" + "=" * 60)
    print("Chat history test complete.")
    print("=" * 60)