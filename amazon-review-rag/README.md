# Amazon Review Q&A — A RAG System for Grounded Product Insights

## Problem
Shoppers and product teams alike drown in review volume. A single popular product can have thousands of reviews, and no one reads them all before answering a question like *"do people say the battery life holds up after 6 months?"* This project builds a retrieval-augmented generation (RAG) system that answers natural-language questions about Amazon products using only the evidence in real customer reviews — with citations back to the source reviews, so answers are verifiable rather than hallucinated.

## Users
- **Shoppers** comparing products who want a synthesized answer instead of skimming 200 reviews.
- **Product/brand teams** who want a fast way to query "what are customers saying about X" without building a full BI pipeline.

## Dataset
[Consumer Reviews of Amazon Products (Datafiniti)](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) — review text, ratings, and product metadata (category, brand, price) across a range of Amazon product categories.

## Approach
1. **Chunking strategy:** Reviews are chunked per-review (not split mid-review) and tagged with product ID, category, and rating as metadata. Rationale: splitting a review across chunks breaks the sentiment/context link between a complaint and its explanation, which matters more here than maximizing retrieval granularity. Documented tradeoff: this caps chunk size at review length, so very long reviews may lose some retrieval precision — noted as a future improvement (e.g., recursive chunking for outlier-length reviews).
2. **Embedding + storage:** Reviews are embedded and stored in ChromaDB, filterable by product/category metadata so retrieval can be scoped (e.g., only search within "tablets").
3. **Retrieval + generation:** LangChain orchestrates retrieval (top-k similarity search, metadata-filtered) and passes retrieved reviews to an LLM (Mistral-7B) with a prompt that requires the model to cite which reviews it drew from and to say "not enough evidence" rather than fabricate an answer.
4. **Evaluation:** A hand-built set of ~15-20 questions with expected-answer characteristics (not just accuracy — also measuring whether citations are correct and whether the model appropriately abstains when evidence is thin).

## Success Metrics
- **Retrieval precision/recall** against the hand-labeled eval set (are the right reviews being surfaced?)
- **Citation accuracy** (does the generated answer's citation actually support the claim?)
- **Abstention rate on out-of-scope questions** (does it say "I don't know" instead of hallucinating when asked about a product/attribute not in the data?)

## Tradeoffs & Next Steps
- **Tradeoff:** Chose per-review chunking for interpretability over maximizing retrieval granularity — see Approach #1.
- **Tradeoff:** Used a smaller open-source model (Mistral-7B) for cost/speed over a larger hosted model; answer quality ceiling is lower but iteration is faster.
- **Next steps:** Add cross-category comparison support (e.g., "compare battery complaints between Brand A and Brand B"), add a lightweight re-ranker on top of similarity search, expand the eval set with adversarial questions to stress-test abstention.

## Demo
A Streamlit app (`src/app.py`) provides a simple interface: type a question, get an answer with linked source reviews.

## Project Structure
```
amazon-review-rag/
├── README.md
├── requirements.txt
├── data/                   # raw + processed data (not committed — see .gitignore)
├── src/
│   ├── data_prep.py        # clean, dedupe, chunk reviews
│   ├── build_index.py      # embed reviews into ChromaDB
│   ├── rag_pipeline.py     # retrieval + generation logic
│   └── app.py               # Streamlit demo
├── eval/
│   └── eval_questions.json # hand-built eval set + scoring script
└── .gitignore
```
