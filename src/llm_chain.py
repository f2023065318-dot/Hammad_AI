"""
llm_chain.py
------------
This module builds the RAG chain: it takes the retrieved context
and the user query, then generates a response using the Groq LLM.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from retriever import get_relevant_chunks, format_context


# Load environment variables (.env file)
load_dotenv()


# Groq model name
GROQ_MODEL_NAME = "openai/gpt-oss-120b"


# System prompt for HammadBot
SYSTEM_PROMPT = """You are HammadBot, a friendly and professional personal assistant chatbot.

Your job is to answer questions about Muhammad Waqas using ONLY the context provided below.

Rules:
1. Use ONLY the information from the provided context.
2. If the answer is not in the context, say: "I'm sorry, I don't have that information in my knowledge base."
3. Be concise, friendly, and professional.
4. Answer in the same language the user asks (English or Urdu).
5. Never make up information.

Context:
{context}
"""


def get_llm() -> ChatGroq:
    """
    Initializes and returns the Groq LLM instance.
    Requires GROQ_API_KEY in the .env file.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0.3,
        max_tokens=512,
        api_key=api_key,
    )
    return llm


def build_prompt() -> ChatPromptTemplate:
    """Builds the chat prompt template."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return prompt


def build_rag_chain():
    """
    Builds the full RAG chain:
    query -> retrieve -> format context -> prompt -> LLM -> answer
    """
    llm = get_llm()
    prompt = build_prompt()
    parser = StrOutputParser()

    def retrieve_and_format(question: str) -> dict:
        """Retrieves relevant chunks and formats them as context."""
        chunks = get_relevant_chunks(question, k=3)
        context = format_context(chunks)
        return {"context": context, "question": question}

    chain = (
        RunnablePassthrough()
        | retrieve_and_format
        | prompt
        | llm
        | parser
    )
    return chain


def ask_question(question: str, chain=None) -> str:
    """
    Asks a question and returns the LLM's answer.

    Args:
        question: User's question.
        chain: Optional pre-built RAG chain.

    Returns:
        LLM's answer as a string.
    """
    if not question or not question.strip():
        return "Please ask a valid question."

    if chain is None:
        chain = build_rag_chain()

    try:
        answer = chain.invoke(question)
        return answer.strip()
    except Exception as e:
        return f"Error: {e}"


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Testing LLM Chain (RAG) for HammadBot")
    print("=" * 60)

    # Build chain once
    chain = build_rag_chain()

    test_questions = [
        "What is Muhammad Waqas's email address?",
        "What projects has he worked on?",
        "What are his technical skills?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        answer = ask_question(q, chain=chain)
        print(f"A: {answer}")

    print("\n" + "=" * 60)
    print("LLM chain test complete.")
    print("=" * 60)