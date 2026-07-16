# Restaurant Review Sentiment Analysis — LLM Prompt Engineering

A restaurant review sentiment analyzer built with local, open-source LLMs (Llama-2-13B and Mistral-7B), progressively extending from basic sentiment classification to structured, aspect-level analysis and automated response drafting — entirely through prompt design, no fine-tuning required.

## Problem

Food delivery/aggregator platforms accumulate large volumes of unstructured customer reviews. Manually reading and categorizing them doesn't scale, and it's hard to extract consistent, structured insight (sentiment by aspect, specific liked/disliked features) from free-text reviews at volume. This project builds an LLM-based pipeline to do that automatically.

## Approach

Five progressively more sophisticated prompting stages, each building on the last:

| Stage | What it does |
|---|---|
| 1. Basic sentiment | Classifies each review as Positive / Negative / Neutral |
| 2. Structured output | Same task, but forces the model to return valid JSON for easy downstream parsing |
| 3. Aspect-level sentiment | Breaks sentiment down by Food Quality, Service, and Ambience separately |
| 4. Feature extraction | Adds extraction of specific liked/disliked features per aspect (e.g. "slow service", "well-prepared food") |
| 5. Response drafting | Adds an automatically drafted customer response, tone-adjusted based on the review's sentiment (grateful for positive, inquisitive for neutral, apologetic for negative) |

- **Models compared**: Llama-2-13B-chat and Mistral-7B-Instruct (both quantized GGUF, run locally via `llama-cpp-python`), with prompt formats adapted to each model's expected instruction format
- **Output parsing**: a custom JSON extraction utility handles the common LLM failure mode of near-valid JSON output, with manual correction for the small number of cases it couldn't parse

## Key findings

- Explicitly instructing the model to return structured JSON (stage 2+) made downstream analysis dramatically easier than free-text sentiment output (stage 1), at low cost to output quality
- Aspect-level breakdown surfaced more actionable insight than overall sentiment alone — a review can be net-positive while flagging a real service problem, which single-label sentiment would miss entirely
- A well-specified prompt can carry an LLM through a fairly complex multi-part task (classify → extract → draft response) in a single call, without needing separate models or fine-tuning for each step

## Tech stack

Python · llama-cpp-python · Hugging Face Hub · Llama-2-13B-chat (GGUF) · Mistral-7B-Instruct (GGUF) · pandas
