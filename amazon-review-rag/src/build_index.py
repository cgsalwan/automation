"""
build_index.py

Embeds review chunks (from data_prep.py output) and stores them in a
persistent ChromaDB collection, filterable by product/category metadata.
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "amazon_reviews"

# Sentence-transformers model — swap for a different one if you want to
# experiment with retrieval quality tradeoffs.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    chunks = []
    with open(PROCESSED_DIR / "chunks.jsonl") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index():
    chunks = load_chunks()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"Indexed {i + len(batch)} / {len(chunks)}")

    print(f"Done. Collection '{COLLECTION_NAME}' has {collection.count()} items.")


if __name__ == "__main__":
    build_index()
