# Python Fundamentals for Prompt Engineering

A set of practical exercises covering the core Python skills needed before doing serious prompt engineering or NLP work: handling multi-line text, reading structured data into Python, and basic text cleaning.

## What's covered

1. **Storing and editing multi-line text** — using triple-quoted strings and f-string placeholders to build and compose prompt text programmatically
2. **Reading structured data into Python** — parsing JSON (e.g. credential/config files), reading plain text files, and loading tabular data with pandas
3. **Text cleaning** — lowercasing and punctuation stripping on a sample customer-feedback dataset, a common preprocessing step before feeding text into an LLM or NLP pipeline

## Notes

- `sample.json` is created by the script itself, so that part runs standalone.
- `customer_feedback.txt` and `sample_dataset.csv` are expected to already exist in the working directory — swap in your own text file / CSV to run those sections.

## Tech stack

Python · pandas · `json` · `re`
