# Article Categorization — Word Embeddings

A text classification project that automatically categorizes news articles by topic, comparing two word embedding approaches (Word2Vec vs. GloVe) as feature representations for a Random Forest classifier.

## Problem

Media organizations publish a high volume of articles across many topics, and manual categorization doesn't scale — delays or misclassification directly hurt content discovery and audience engagement. This project builds a model that automatically categorizes articles from their raw text, so new content can be routed and surfaced accurately at publication speed.

## Approach

- **Dataset**: ~4,000 news articles across categories (sports ~53%, news ~40%, plus smaller categories), with text, headline, section, and publish-date metadata
- **Preprocessing**: lowercasing, punctuation/number removal, stopword removal, Porter stemming
- **Feature engineering — two embedding approaches compared**:
  - **Word2Vec**: trained from scratch on the article corpus itself (300-dim vectors)
  - **GloVe**: pretrained Stanford embeddings (100-dim vectors), used as-is without retraining
  - Both approaches average word vectors across each article to get a single document-level feature vector
- **Model**: Random Forest classifier, trained with class weighting to handle category imbalance, then hyperparameter-tuned via GridSearchCV (optimizing for recall)

## Results

| Model | Notes |
|---|---|
| Word2Vec — base Random Forest | Baseline |
| **Word2Vec — tuned Random Forest** | **Best generalizing model** — selected as final |
| GloVe — base Random Forest | Baseline |
| GloVe — tuned Random Forest | Did not outperform the Word2Vec-based model |

- **Final model**: Word2Vec embeddings + tuned Random Forest, achieving ~86% accuracy and ~86% recall on the test set
- Word2Vec (trained on the article corpus itself) outperformed pretrained GloVe embeddings — domain-specific training data captured this corpus's vocabulary and context better than general-purpose pretrained vectors

## Key findings

- The model performs well on categories with more training data (sports, politics, news, business) and noticeably worse on underrepresented categories (entertainment, health) — a direct consequence of class imbalance in the training data, not a modeling flaw
- Collecting more samples for the underrepresented categories is the most direct lever to improve performance further, more so than additional model tuning

## Tech stack

Python · Gensim (Word2Vec, GloVe) · scikit-learn (Random Forest, GridSearchCV) · NLTK · pandas · matplotlib/seaborn
