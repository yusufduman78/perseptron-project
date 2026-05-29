from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, FUSION_NUMERIC_FEATURES, TABULAR_NUMERIC_FEATURES


def make_cutoff(transactions: pd.DataFrame, validation_days: int) -> pd.Timestamp:
    return transactions["t_dat"].max() - pd.Timedelta(days=validation_days)


def sample_positive_pairs(
    transactions: pd.DataFrame,
    customers: set[str],
    cutoff: pd.Timestamp,
    max_positives: int | None,
    seed: int,
) -> pd.DataFrame:
    positives = transactions[
        transactions["customer_id"].isin(customers) & (transactions["t_dat"] <= cutoff)
    ][["customer_id", "article_id"]].drop_duplicates()
    positives["label"] = 1
    if max_positives and len(positives) > max_positives:
        positives = positives.sample(max_positives, random_state=seed)
    return positives.reset_index(drop=True)


def make_validation_positive_pairs(
    transactions: pd.DataFrame,
    customers: set[str],
    cutoff: pd.Timestamp,
    max_positives: int | None,
    seed: int,
) -> pd.DataFrame:
    positives = transactions[
        transactions["customer_id"].isin(customers) & (transactions["t_dat"] > cutoff)
    ][["customer_id", "article_id"]].drop_duplicates()
    positives["label"] = 1
    if max_positives and len(positives) > max_positives:
        positives = positives.sample(max_positives, random_state=seed)
    return positives.reset_index(drop=True)


def add_negative_pairs(
    positives: pd.DataFrame,
    article_pool: np.ndarray,
    negatives_per_positive: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    neg_customers = np.repeat(positives["customer_id"].to_numpy(), negatives_per_positive)
    neg_articles = rng.choice(article_pool, size=len(neg_customers), replace=True)
    negatives = pd.DataFrame({"customer_id": neg_customers, "article_id": neg_articles, "label": 0})
    data = pd.concat([positives, negatives], ignore_index=True)
    data = data.drop_duplicates(["customer_id", "article_id", "label"])
    return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_history_frame(transactions: pd.DataFrame, customers: set[str], cutoff: pd.Timestamp) -> pd.DataFrame:
    return transactions[
        transactions["customer_id"].isin(customers) & (transactions["t_dat"] <= cutoff)
    ][["customer_id", "article_id"]]


def build_customer_profiles(
    history: pd.DataFrame,
    embeddings: np.ndarray,
    article_to_index: dict[str, int],
) -> tuple[dict[str, int], np.ndarray, np.ndarray, dict[tuple[str, str], int]]:
    history = history[history["article_id"].isin(article_to_index)].copy()
    customer_ids = sorted(history["customer_id"].unique())
    customer_to_index = {customer_id: index for index, customer_id in enumerate(customer_ids)}
    sums = np.zeros((len(customer_ids), embeddings.shape[1]), dtype="float32")
    counts = np.zeros(len(customer_ids), dtype="float32")
    pair_counts: dict[tuple[str, str], int] = {}
    for row in history.itertuples(index=False):
        customer_idx = customer_to_index[row.customer_id]
        article_idx = article_to_index[row.article_id]
        sums[customer_idx] += embeddings[article_idx]
        counts[customer_idx] += 1.0
        key = (row.customer_id, row.article_id)
        pair_counts[key] = pair_counts.get(key, 0) + 1
    return customer_to_index, sums, counts, pair_counts


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def visual_arrays_for_pairs(
    pairs: pd.DataFrame,
    embeddings: np.ndarray,
    article_to_index: dict[str, int],
    customer_to_index: dict[str, int],
    profile_sums: np.ndarray,
    profile_counts: np.ndarray,
    pair_counts: dict[tuple[str, str], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_dim = embeddings.shape[1]
    article_emb = np.zeros((len(pairs), image_dim), dtype="float32")
    profile_emb = np.zeros((len(pairs), image_dim), dtype="float32")
    visual_similarity = np.zeros(len(pairs), dtype="float32")
    visual_history_count = np.zeros(len(pairs), dtype="float32")

    for index, row in enumerate(pairs[["customer_id", "article_id"]].itertuples(index=False)):
        article_idx = article_to_index.get(row.article_id)
        customer_idx = customer_to_index.get(row.customer_id)
        if article_idx is None or customer_idx is None:
            continue
        candidate = np.asarray(embeddings[article_idx], dtype="float32")
        count_to_remove = pair_counts.get((row.customer_id, row.article_id), 0)
        usable_count = max(float(profile_counts[customer_idx] - count_to_remove), 0.0)
        if usable_count > 0:
            profile = (profile_sums[customer_idx] - count_to_remove * candidate) / usable_count
        else:
            profile = np.zeros(image_dim, dtype="float32")
        article_emb[index] = candidate
        profile_emb[index] = profile
        visual_history_count[index] = usable_count
        denom = np.linalg.norm(candidate) * np.linalg.norm(profile)
        visual_similarity[index] = float(np.dot(candidate, profile) / denom) if denom > 0 else 0.0
    return article_emb, profile_emb, visual_similarity, visual_history_count


def merge_metadata(pairs: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    frame = pairs.merge(customers, on="customer_id", how="left").merge(articles, on="article_id", how="left")
    return frame


def fit_tabular_metadata(frame: pd.DataFrame, numeric_features: list[str]) -> dict:
    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_mean": {},
        "numeric_std": {},
        "category_maps": {},
    }
    for column in numeric_features:
        values = pd.to_numeric(frame[column], errors="coerce").astype("float32")
        metadata["numeric_mean"][column] = float(values.mean()) if len(values) else 0.0
        std = float(values.std()) if len(values) else 1.0
        metadata["numeric_std"][column] = std if std > 1e-8 else 1.0
    for column in CATEGORICAL_FEATURES:
        values = frame[column].fillna("UNKNOWN").astype(str)
        categories = ["__UNK__"] + sorted(values.unique().tolist())
        metadata["category_maps"][column] = {value: index for index, value in enumerate(categories)}
    return metadata


def encode_tabular(frame: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    numeric_columns = []
    for column in metadata["numeric_features"]:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(metadata["numeric_mean"][column]).astype("float32")
        values = (values - metadata["numeric_mean"][column]) / metadata["numeric_std"][column]
        numeric_columns.append(values.to_numpy(dtype="float32"))
    numeric = np.stack(numeric_columns, axis=1).astype("float32")

    categorical_columns = []
    for column in metadata["categorical_features"]:
        mapping = metadata["category_maps"][column]
        values = frame[column].fillna("UNKNOWN").astype(str).map(mapping).fillna(0).astype("int64")
        categorical_columns.append(values.to_numpy(dtype="int64"))
    categorical = np.stack(categorical_columns, axis=1).astype("int64")
    return numeric, categorical


def category_sizes(metadata: dict) -> list[int]:
    return [len(metadata["category_maps"][column]) for column in metadata["categorical_features"]]


def metadata_to_jsonable(metadata: dict) -> dict:
    return json.loads(json.dumps(metadata))


def numeric_features_for_model(model_name: str) -> list[str]:
    if model_name == "late_fusion":
        return FUSION_NUMERIC_FEATURES
    return TABULAR_NUMERIC_FEATURES
