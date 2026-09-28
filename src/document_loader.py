"""
document_loader.py
------------------
This module loads all files (PDF, TXT) from the data/ folder
and converts them into LangChain Documents.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# Project root directory (Hammad_AI folder)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_pdf(file_path: str) -> List[Document]:
    """Loads a PDF file and returns a list of Documents."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"[OK] PDF loaded: {os.path.basename(file_path)} "
              f"({len(documents)} pages)")
        return documents
    except Exception as e:
        print(f"[ERROR] Failed to load PDF: {file_path} -> {e}")
        return []


def load_txt(file_path: str) -> List[Document]:
    """Loads a TXT file and returns a list of Documents."""
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"[OK] TXT loaded: {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        print(f"[ERROR] Failed to load TXT: {file_path} -> {e}")
        return []


def load_all_documents(data_dir: str = None) -> List[Document]:
    """
    Loads all .pdf and .txt files from the data/ folder.
    Adds source metadata to each document.
    """
    if data_dir is None:
        data_dir = str(DATA_DIR)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    all_documents: List[Document] = []

    # Recursively search for all files
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            extension = file_name.lower().split(".")[-1]

            if extension == "pdf":
                docs = load_pdf(file_path)
            elif extension == "txt":
                docs = load_txt(file_path)
            else:
                print(f"[SKIP] Unsupported file: {file_name}")
                continue

            # Add source metadata to each document
            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["file_path"] = file_path

            all_documents.extend(docs)

    if not all_documents:
        raise ValueError(
            "No documents were loaded. Please add a PDF or TXT file to the data/ folder."
        )

    print(f"\n[INFO] Total {len(all_documents)} documents loaded.")
    return all_documents


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    docs = load_all_documents()
    print("\n--- Sample Document ---")
    print("Source:", docs[0].metadata.get("source"))
    print("Content (first 300 chars):")
    print(docs[0].page_content[:300])