from __future__ import annotations

import re
import string
from collections import Counter

from nltk.stem import PorterStemmer


_STEMMER = PorterStemmer()


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [
        _STEMMER.stem(token) for token in normalize_answer(prediction).split()
    ]
    truth_tokens = [
        _STEMMER.stem(token) for token in normalize_answer(ground_truth).split()
    ]
    if not prediction_tokens or not truth_tokens:
        return float(prediction_tokens == truth_tokens)
    overlap = Counter(prediction_tokens) & Counter(truth_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def official_locomo_f1(prediction: str, ground_truth: str, category: int) -> float:
    """Match the official LoCoMo QA evaluator for categories 1–4."""
    answer = ground_truth.split(";")[0].strip() if category == 3 else ground_truth
    if category == 1:
        predictions = [item.strip() for item in prediction.split(",")]
        truths = [item.strip() for item in answer.split(",")]
        return sum(
            max(token_f1(candidate, truth) for candidate in predictions)
            for truth in truths
        ) / len(truths)
    if category in (2, 3, 4):
        return token_f1(prediction, answer)
    raise ValueError("This baseline intentionally evaluates LoCoMo categories 1–4")


def answer_token_recall(retrieval_text: str, ground_truth: str) -> float | None:
    truth_tokens = [
        _STEMMER.stem(token) for token in normalize_answer(ground_truth).split()
    ]
    if not truth_tokens:
        return None
    retrieved_tokens = Counter(
        _STEMMER.stem(token) for token in normalize_answer(retrieval_text).split()
    )
    overlap = Counter(truth_tokens) & retrieved_tokens
    return sum(overlap.values()) / len(truth_tokens)
