"""
data_prep.py

Loads the raw Datafiniti Amazon Consumer Reviews CSV, cleans it, and
produces a per-review chunk file ready for embedding.

Download the dataset from:
https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
and place the CSV(s) in data/raw/
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Columns we care about — adjust to match the actual CSV headers once downloaded.
KEEP_COLUMNS = [
    "id",  # product id
    "name",  # product name
    "asins",
    "brand",
    "categories",
    "primaryCategories",
    "reviews.date",
    "reviews.doRecommend",
    "reviews.rating",
    "reviews.text",
    "reviews.title",
    "reviews.username",
]


def load_raw(filename: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename)
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns missing from CSV: {missing}. "
            "Check the actual column names in the downloaded file and update KEEP_COLUMNS."
        )
    return df[KEEP_COLUMNS].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["reviews.text"])
    df = df.drop_duplicates(subset=["id", "reviews.text"])
    df["reviews.text"] = df["reviews.text"].str.strip()
    df = df[df["reviews.text"].str.len() > 10]  # drop near-empty reviews
    return df


def to_chunks(df: pd.DataFrame) -> list[dict]:
    """
    One chunk per review, tagged with metadata for filtered retrieval.
    Per-review chunking preserves the sentiment/context link within a review
    (see README > Approach for the tradeoff discussion).
    """
    chunks = []
    for idx, row in df.iterrows():
        chunks.append(
            {
                "chunk_id": f"review_{idx}",
                "text": f"{row.get('reviews.title', '')}\n{row['reviews.text']}".strip(),
                "metadata": {
                    "product_id": row["id"],
                    "product_name": row["name"],
                    "asin": row.get("asins", "unknown"),
                    "category": row["categories"],
                    "primary_category": row.get("primaryCategories", "unknown"),
                    "brand": row.get("brand", "unknown"),
                    "rating": row["reviews.rating"],
                    "do_recommend": row.get("reviews.doRecommend", None),
                    "date": row.get("reviews.date", "unknown"),
                    "username": row.get("reviews.username", "unknown"),
                },
            }
        )
    return chunks


def main(raw_filename: str = "amazon_reviews.csv"):
    df = load_raw(raw_filename)
    df = clean(df)
    chunks = to_chunks(df)

    out_path = PROCESSED_DIR / "chunks.jsonl"
    import json

    with open(out_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
