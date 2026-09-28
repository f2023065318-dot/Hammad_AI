"""
embeddings.py
-------------
This module generates vector embeddings for text chunks
using a Hugging Face sentence-transformer model.
"""

from typing import List

from langchain_huggingface import HuggingFaceEmbeddings


# Model name (small, fast, and good quality)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a HuggingFaceEmbeddings instance.

    The model runs locally (no API key needed) and converts
    text into 384-dimensional vectors.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def embed_texts(texts: List[str], embeddings: HuggingFaceEmbeddings) -> List[List[float]]:
    """
    Generates embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.
        embeddings: HuggingFaceEmbeddings instance.

    Returns:
        List of embedding vectors.
    """
    if not texts:
        raise ValueError("No texts provided for embedding.")

    vectors = embeddings.embed_documents(texts)
    print(f"[INFO] Generated embeddings for {len(vectors)} texts. "
          f"Vector size: {len(vectors[0])}")
    return vectors


# ------------------------------------------
# For testing (when run directly)
# ------------------------------------------
if __name__ == "__main__":
    from document_loader import load_all_documents
    from text_splitter import split_documents

    # Load and split documents
    docs = load_all_documents()
    chunks = split_documents(docs)

    # Generate embeddings
    print("\n[INFO] Loading embedding model (first run may download ~80MB)...")
    embed_model = get_embedding_model()

    texts = [chunk.page_content for chunk in chunks]
    vectors = embed_texts(texts, embed_model)

    print("\n--- Sample Embedding ---")
    print("Text:", texts[0][:80], "...")
    print("Vector (first 5 values):", vectors[0][:5])
    print("Vector length:", len(vectors[0]))