"""
rag_pipeline.py

Core retrieval + generation logic. Given a question (and optional
category/product filter), retrieves relevant review chunks and prompts
an LLM to answer using only that evidence, citing which reviews it used.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "amazon_reviews"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a product research assistant. You answer questions about products \
using ONLY the customer reviews provided in CONTEXT REVIEWS below. You have no other \
knowledge of these products — do not use anything you know from training data.

RULES:
1. Every factual claim must be followed by the chunk_id(s) it came from, in square \
brackets, e.g. "battery life drains fast after a few months [review_142]." Do this \
inline, not as a list at the end.
2. If fewer than 2 of the retrieved reviews support a claim, either state that the \
evidence is limited ("one review mentions...") or omit the claim if it's too thin to \
be useful.
3. If the retrieved reviews don't contain enough evidence to answer the question at \
all, say so directly: "The available reviews don't discuss this." Do not guess, infer \
beyond what's stated, or fill gaps with general product knowledge.
4. If reviews disagree with each other, represent the disagreement rather than \
averaging it into a vague middle-ground answer. E.g. "Opinions are split: some \
reviewers found the battery lasted all day [review_12, review_45], while others \
reported it draining within a few hours [review_88]."
5. Be concise. 2-4 sentences unless the question genuinely requires more.

EXAMPLE:
CONTEXT REVIEWS:
[review_10] (product: TabPro 8, rating: 2)
Battery died by noon even with light use. Disappointing for the price.

[review_11] (product: TabPro 8, rating: 5)
Battery easily lasts a full workday for me, no complaints.

QUESTION: How's the battery life on the TabPro 8?

ANSWER: Reviews are mixed on battery life. One reviewer found it died by noon under \
light use [review_10], while another reported it lasting a full workday [review_11]. \
With only two reviews, this isn't a strong sample — worth checking more before relying \
on it.
"""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    distance: float


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(question: str, top_k: int = 8, category_filter: str | None = None) -> list[RetrievedChunk]:
    collection = get_collection()

    where = {"category": category_filter} if category_filter else None
    results = collection.query(query_texts=[question], n_results=top_k, where=where)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append(
            RetrievedChunk(
                chunk_id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                distance=results["distances"][0][i],
            )
        )
    return chunks


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[{c.chunk_id}] (product: {c.metadata.get('product_name')}, "
        f"rating: {c.metadata.get('rating')})\n{c.text}"
        for c in chunks
    )
    return f"""{SYSTEM_PROMPT}

CONTEXT REVIEWS:
{context}

QUESTION: {question}

ANSWER:"""


def answer_question(question: str, category_filter: str | None = None, top_k: int = 8):
    """
    Wire this up to your LLM of choice (Mistral-7B via HF, Ollama, or an
    API call). Left as a stub since the model call depends on how you're
    hosting it (local vs API).
    """
    chunks = retrieve(question, top_k=top_k, category_filter=category_filter)
    prompt = build_prompt(question, chunks)

    # TODO: replace with actual LLM call, e.g.:
    # response = llm.invoke(prompt)
    raise NotImplementedError(
        "Wire up your LLM call here (Mistral-7B, or swap in any chat model). "
        "The retrieved `chunks` and assembled `prompt` are ready to use."
    )


if __name__ == "__main__":
    q = "Do people complain about battery life on tablets?"
    result_chunks = retrieve(q, category_filter=None)
    for c in result_chunks:
        print(c.chunk_id, c.metadata.get("product_name"), c.distance)
