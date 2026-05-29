from __future__ import annotations


def average_precision_at_k(predictions: list[str], relevant: set[str], k: int = 12) -> float:
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    seen: set[str] = set()
    for rank, item in enumerate(predictions[:k], start=1):
        if item in seen:
            continue
        seen.add(item)
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def precision_at_k(predictions: list[str], relevant: set[str], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    return len(set(predictions[:k]) & relevant) / k


def recall_at_k(predictions: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(predictions[:k]) & relevant) / len(relevant)


def hit_rate_at_k(predictions: list[str], relevant: set[str], k: int = 12) -> float:
    return float(bool(set(predictions[:k]) & relevant))
