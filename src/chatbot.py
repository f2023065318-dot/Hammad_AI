"""
chatbot.py
----------
This is the main chatbot module for HammadBot.
It combines retrieval, LLM generation, and chat history
into a single conversational interface.

Auto-builds the FAISS index on startup if it doesn't exist
(required for cloud deployment like Streamlit Cloud).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from retriever import get_relevant_chunks, format_context
from chat_history import ChatHistory


# Load environment variables
load_dotenv()


# Chatbot identity
CHATBOT_NAME = "HammadBot"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


# System prompt with personality
SYSTEM_PROMPT = """You are {chatbot_name}, a friendly and professional personal assistant chatbot
for Muhammad Waqas, a Software Engineering student at UMT.

Your job is to answer questions about Muhammad Waqas using ONLY the context provided below.

Rules:
1. Use ONLY the information from the provided context.
2. If the answer is not in the context, say politely:
   "I'm sorry, I don't have that information in my knowledge base."
3. Be concise, friendly, and professional.
4. Answer in the same language the user asks (English or Urdu).
5. Never make up or guess information.
6. If the user greets you, greet back warmly and introduce yourself briefly.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Previous conversation:
{history}
"""


class HammadAIChatbot:
    """Main chatbot class for HammadBot."""

    def __init__(self, max_history_turns: int = 5):
        """
        Initializes the chatbot.

        Args:
            max_history_turns: Number of past turns to remember.
        """
        self.name = CHATBOT_NAME
        self.history = ChatHistory(max_turns=max_history_turns)

        # Auto-build FAISS index if it doesn't exist (for cloud deployment)
        self._ensure_index_exists()

        self.llm = self._init_llm()
        self.prompt = self._build_prompt()
        self.parser = StrOutputParser()

    def _ensure_index_exists(self) -> None:
        """Builds the FAISS index if it doesn't already exist."""
        # Add src/ to path for imports
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        base_dir = src_dir.parent
        index_path = base_dir / "vectorstore" / "faiss_index"

        if not index_path.exists():
            print("[INFO] FAISS index not found. Building automatically...")
            try:
                from document_loader import load_all_documents
                from text_splitter import split_documents
                from embeddings import get_embedding_model
                from vector_db import build_vectorstore

                docs = load_all_documents()
                chunks = split_documents(docs)
                embeddings = get_embedding_model()
                build_vectorstore(chunks, embeddings=embeddings, save=True)
                print("[INFO] FAISS index built successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to build FAISS index: {e}")
                raise
        else:
            print(f"[INFO] FAISS index found at {index_path}")

    def _init_llm(self) -> ChatGroq:
        """Initializes Groq LLM."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please add it to your .env file."
            )

        return ChatGroq(
            model=GROQ_MODEL_NAME,
            temperature=0.3,
            max_tokens=512,
            api_key=api_key,
        )

    def _build_prompt(self) -> ChatPromptTemplate:
        """Builds the chat prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

    def _is_greeting(self, text: str) -> bool:
        """Checks if the message is a simple greeting."""
        greetings = [
            "hi", "hello", "hey", "salam", "assalam", "aoa",
            "assalamualaikum", "good morning", "good evening",
        ]
        text_lower = text.lower().strip()
        return any(g in text_lower for g in greetings) and len(text_lower) < 30

    def _greeting_response(self) -> str:
        """Returns a greeting response."""
        return (
            f"Hello! I'm {self.name}, your personal assistant. "
            f"I can answer questions about Muhammad Waqas — his education, "
            f"projects, skills, and experience. How can I help you today?"
        )

    def respond(self, user_message: str) -> str:
        """
        Main method: takes user input, returns chatbot response.

        Args:
            user_message: User's message.

        Returns:
            Chatbot's response as a string.
        """
        if not user_message or not user_message.strip():
            return "Please type a valid message."

        user_message = user_message.strip()

        # Handle greetings separately (no retrieval needed)
        if self._is_greeting(user_message):
            response = self._greeting_response()
            self.history.add_user_message(user_message)
            self.history.add_assistant_message(response)
            return response

        # Retrieve relevant chunks
        try:
            chunks = get_relevant_chunks(user_message, k=4)
            context = format_context(chunks)
        except Exception as e:
            context = "No context available."
            print(f"[WARN] Retrieval error: {e}")

        # Build prompt inputs
        history_text = self.history.get_recent_context(n=3) or "None"

        prompt_inputs = {
            "chatbot_name": self.name,
            "context": context,
            "history": history_text,
            "question": user_message,
        }

        # Generate response
        try:
            response = (self.prompt | self.llm | self.parser).invoke(prompt_inputs)
            response = response.strip()
        except Exception as e:
            response = f"Sorry, I encountered an error: {e}"

        # Save to history
        self.history.add_user_message(user_message)
        self.history.add_assistant_message(response)

        return response

    def reset(self) -> None:
        """Clears conversation history."""
        self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Returns conversation history."""
        return self.history.get_messages()


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {CHATBOT_NAME} - Personal Assistant Chatbot")
    print("=" * 60)
    print("Type 'exit' to quit, 'reset' to clear history.\n")

    bot = HammadAIChatbot()

    # Interactive mode
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print(f"{CHATBOT_NAME}: Goodbye! Have a great day.")
                break
            if user_input.lower() == "reset":
                bot.reset()
                print(f"{CHATBOT_NAME}: Conversation history cleared.")
                continue

            response = bot.respond(user_input)
            print(f"{CHATBOT_NAME}: {response}\n")
        except KeyboardInterrupt:
            print(f"\n{CHATBOT_NAME}: Goodbye!")
            break