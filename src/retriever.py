"""
retriever.py
------------
This module retrieves the most relevant document chunks
from the FAISS vector store based on a user query.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from vector_db import load_vectorstore, get_retriever


# Default number of chunks to retrieve
DEFAULT_TOP_K = 3


def get_relevant_chunks(
    query: str,
    k: int = DEFAULT_TOP_K,
    retriever: Optional[VectorStoreRetriever] = None,
) -> List[Document]:
    """
    Retrieves the top-k most relevant chunks for a given query.

    Args:
        query: User's question.
        k: Number of top chunks to retrieve.
        retriever: Optional pre-built retriever. If None, loads from disk.

    Returns:
        List of relevant Document chunks.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if retriever is None:
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore, k=k)

    results = retriever.invoke(query)
    return results


def format_context(documents: List[Document]) -> str:
    """
    Formats retrieved documents into a single context string
    that can be passed to the LLM.

    Args:
        documents: List of retrieved Document chunks.

    Returns:
        Formatted context string.
    """
    if not documents:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[Chunk {i} | Source: {source}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "What is Muhammad Waqas's email address?",
        "What projects has he worked on?",
        "What is his education?",
        "What are his technical skills?",
    ]

    print("=" * 60)
    print("Testing Retriever for HammadBot")
    print("=" * 60)

    # Load retriever once
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=3)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        chunks = get_relevant_chunks(query, retriever=retriever)

        for i, chunk in enumerate(chunks, 1):
            print(f"\n  Chunk {i} (Source: {chunk.metadata.get('source')}):")
            print(f"  {chunk.page_content[:150]}...")

    print("\n" + "=" * 60)
    print("Retriever test complete.")
    print("=" * 60)