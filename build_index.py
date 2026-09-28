"""
build_index.py
--------------
This script builds the FAISS vector index from all documents
in the data/ folder. Run this once before starting the chatbot,
or whenever you update your data files.

Usage:
    python build_index.py
"""

import sys
import os
from pathlib import Path

# Add src/ to Python path so we can import our modules
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from document_loader import load_all_documents
from text_splitter import split_documents
from embeddings import get_embedding_model
from vector_db import build_vectorstore


def main():
    """Builds the FAISS vector index from scratch."""
    print("=" * 60)
    print("  Building FAISS Vector Index for HammadBot")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[Step 1/4] Loading documents from data/ folder...")
    documents = load_all_documents()
    print(f"[OK] Loaded {len(documents)} document(s).")

    # Step 2: Split into chunks
    print("\n[Step 2/4] Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"[OK] Created {len(chunks)} chunk(s).")

    # Step 3: Load embedding model
    print("\n[Step 3/4] Loading embedding model...")
    embeddings = get_embedding_model()
    print("[OK] Embedding model loaded.")

    # Step 4: Build and save FAISS index
    print("\n[Step 4/4] Building FAISS index...")
    vectorstore = build_vectorstore(chunks, embeddings=embeddings, save=True)
    print("[OK] FAISS index built and saved successfully.")

    print("\n" + "=" * 60)
    print("  Index build complete!")
    print(f"  Total chunks indexed: {len(chunks)}")
    print(f"  Index saved at: {BASE_DIR / 'vectorstore' / 'faiss_index'}")
    print("=" * 60)
    print("\nYou can now run the chatbot:")
    print("  python src/chatbot.py                (Interactive CLI mode)")
    print("  python -m streamlit run app/streamlit_app.py  (Web UI mode)")


if __name__ == "__main__":
    main()