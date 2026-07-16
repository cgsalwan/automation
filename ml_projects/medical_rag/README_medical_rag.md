# Medical Q&A — Retrieval-Augmented Generation (RAG)

A RAG pipeline that answers clinical questions by grounding an LLM's responses in a real medical reference manual, rather than relying on the model's raw (and potentially outdated or hallucinated) knowledge.

## Problem

Healthcare professionals often need fast, reliable answers to clinical questions — symptoms, treatment protocols, drug information — but sifting through thousands of pages of reference material under time pressure isn't practical. This project builds a system that retrieves the relevant passages from a medical manual and uses them to ground an LLM's answer, rather than letting the model answer from memory alone.

## Approach

- **Source data**: a medical reference manual (PDF, 4,000+ pages), loaded and split into ~500-token chunks with overlap to preserve context across chunk boundaries
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, stored in a **ChromaDB** vector store for similarity search
- **LLM**: Mistral-7B-Instruct (quantized GGUF, run locally via `llama-cpp-python`)
- **Three response strategies compared** on the same 5 clinical questions (sepsis protocol, appendicitis treatment, hair loss causes, brain injury treatment, fracture care):

| Approach | Description |
|---|---|
| Plain LLM | Direct question to the model, no context |
| Prompt-engineered LLM | Adds a system prompt establishing a "medical assistant" persona and tone |
| RAG | Retrieves top-k relevant chunks from the manual and includes them as context before generating |

- **Evaluation**: RAG responses are automatically scored on two axes using LLM-as-judge prompts — **groundedness** (is the answer actually supported by the retrieved context, or does it include unsupported claims?) and **relevance** (does the answer address the question asked?)

## Key findings

- Grounding responses in retrieved manual content produced more specific, protocol-aligned answers than the plain or prompt-engineered LLM alone
- The system reliably differentiated between related-but-distinct conditions (e.g., early vs. late-stage sepsis) when the retrieved context supported that distinction
- Automated groundedness/relevance scoring is a practical way to sanity-check RAG output quality without manual review of every response

## Tech stack

Python · LangChain · ChromaDB · Sentence-Transformers · Mistral-7B-Instruct (via llama-cpp-python) · PyMuPDF

## Notes

This notebook expects a local PDF file (`medical_diagnosis_manual.pdf`) and downloads the Mistral-7B-Instruct GGUF model from Hugging Face Hub on first run — expect a large download and a machine with enough RAM/VRAM to run a 7B quantized model.
