"""
vector_db.py
------------
This module creates and manages a FAISS vector database
for storing and retrieving document embeddings.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from embeddings import get_embedding_model


# Path to store FAISS index
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"


def build_vectorstore(
    documents: List[Document],
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    save: bool = True,
) -> FAISS:
    """
    Builds a FAISS vector store from a list of documents.

    Args:
        documents: List of chunked LangChain Documents.
        embeddings: Optional embedding model. If None, loads default.
        save: Whether to save the index to disk.

    Returns:
        FAISS vector store instance.
    """
    if not documents:
        raise ValueError("No documents provided to build vector store.")

    if embeddings is None:
        embeddings = get_embedding_model()

    print(f"[INFO] Building FAISS index from {len(documents)} chunks...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    if save:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(FAISS_INDEX_PATH))
        print(f"[INFO] FAISS index saved to: {FAISS_INDEX_PATH}")

    return vectorstore


def load_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    """
    Loads an existing FAISS index from disk.

    Args:
        embeddings: Optional embedding model. If None, loads default.

    Returns:
        FAISS vector store instance.
    """
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Please run build_index.py first."
        )

    if embeddings is None:
        embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[INFO] FAISS index loaded from: {FAISS_INDEX_PATH}")
    return vectorstore


def get_retriever(
    vectorstore: FAISS,
    k: int = 3,
) -> VectorStoreRetriever:
    """
    Returns a retriever from a FAISS vector store.

    Args:
        vectorstore: FAISS vector store instance.
        k: Number of top documents to retrieve.

    Returns:
        VectorStoreRetriever instance.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    from document_loader import load_all_documents
    from text_splitter import split_documents

    # Load and split
    docs = load_all_documents()
    chunks = split_documents(docs)

    # Build vector store
    vs = build_vectorstore(chunks)

    # Test retrieval
    print("\n--- Test Retrieval ---")
    query = "What is Muhammad Waqas's email address?"
    results = vs.similarity_search(query, k=2)

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("Source:", doc.metadata.get("source"))
        print("Content:", doc.page_content[:200])