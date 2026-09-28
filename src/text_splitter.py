"""
text_splitter.py
----------------
This module splits large documents into smaller chunks
so they can be embedded and retrieved efficiently.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default chunk settings
CHUNK_SIZE = 500          # Maximum characters per chunk
CHUNK_OVERLAP = 100       # Overlap between chunks (for context continuity)


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks.

    Args:
        documents: List of LangChain Documents to split.
        chunk_size: Maximum size of each chunk (in characters).
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Documents with preserved metadata.
    """
    if not documents:
        raise ValueError("No documents provided to split.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"[INFO] Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    from document_loader import load_all_documents

    docs = load_all_documents()
    chunks = split_documents(docs)

    print("\n--- Sample Chunk ---")
    print("Chunk ID:", chunks[0].metadata.get("chunk_id"))
    print("Source:", chunks[0].metadata.get("source"))
    print("Length:", len(chunks[0].page_content), "characters")
    print("Content:")
    print(chunks[0].page_content)