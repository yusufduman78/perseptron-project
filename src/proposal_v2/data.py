from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR


def log(message: str) -> None:
    print(message, flush=True)


def _candidate_raw_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("HM_RAW_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(DATA_DIR / "raw")
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for path in kaggle_root.rglob("transactions_train.csv"):
            candidates.append(path.parent)
    return candidates


def resolve_raw_dir(raw_dir: str | Path | None = None) -> Path:
    candidates = [Path(raw_dir)] if raw_dir else _candidate_raw_dirs()
    for candidate in candidates:
        if (candidate / "transactions_train.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find H&M raw data directory. Set HM_RAW_DIR or pass --raw-dir.")


def resolve_images_dir(raw_dir: Path, images_dir: str | Path | None = None) -> Path:
    if images_dir:
        return Path(images_dir)
    for candidate in [raw_dir / "images", DATA_DIR / "images" / "hm_images"]:
        if candidate.exists():
            return candidate
    return raw_dir / "images"


def _find_file_by_name(root: Path, name: str) -> Path | None:
    if not root.exists():
        return None
    for path in root.rglob(name):
        return path
    return None


def resolve_embedding_paths(
    embeddings_path: str | Path | None = None,
    embedding_ids_path: str | Path | None = None,
) -> tuple[Path, Path]:
    if embeddings_path and embedding_ids_path:
        return Path(embeddings_path), Path(embedding_ids_path)

    env_embeddings = os.environ.get("HM_EMBEDDINGS_PATH")
    env_ids = os.environ.get("HM_EMBEDDING_IDS_PATH")
    if env_embeddings and env_ids:
        return Path(env_embeddings), Path(env_ids)

    local_embeddings = DATA_DIR / "embeddings" / "final_kaggle" / "article_image_embeddings_popular.npy"
    local_ids = DATA_DIR / "embeddings" / "final_kaggle" / "article_image_embedding_ids_popular.csv"
    if local_embeddings.exists() and local_ids.exists():
        return local_embeddings, local_ids

    kaggle_root = Path("/kaggle/input")
    embeddings = _find_file_by_name(kaggle_root, "article_image_embeddings_popular.npy")
    ids = _find_file_by_name(kaggle_root, "article_image_embedding_ids_popular.csv")
    if embeddings and ids:
        return embeddings, ids

    raise FileNotFoundError("Could not find EfficientNet embedding cache paths.")


def read_transactions(raw_dir: Path) -> pd.DataFrame:
    transactions = pd.read_csv(raw_dir / "transactions_train.csv", dtype={"article_id": str})
    transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["price"] = transactions["price"].astype("float32")
    transactions["sales_channel_id"] = transactions["sales_channel_id"].astype("int8")
    return transactions


def read_customers(raw_dir: Path) -> pd.DataFrame:
    customers = pd.read_csv(raw_dir / "customers.csv")
    customers["FN"] = customers["FN"].fillna(0).astype("float32")
    customers["Active"] = customers["Active"].fillna(0).astype("float32")
    customers["age"] = customers["age"].fillna(customers["age"].median()).astype("float32")
    for column in ["club_member_status", "fashion_news_frequency"]:
        customers[column] = customers[column].fillna("UNKNOWN").astype(str)
    return customers


def read_articles(raw_dir: Path) -> pd.DataFrame:
    articles = pd.read_csv(raw_dir / "articles.csv", dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)
    for column in articles.columns:
        if articles[column].dtype == "object":
            articles[column] = articles[column].fillna("UNKNOWN").astype(str)
    return articles


def load_core_tables(raw_dir: str | Path | None = None) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved = resolve_raw_dir(raw_dir)
    log(f"Using raw data: {resolved}")
    return resolved, read_transactions(resolved), read_customers(resolved), read_articles(resolved)


def load_embeddings(
    embeddings_path: str | Path | None = None,
    embedding_ids_path: str | Path | None = None,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    emb_path, ids_path = resolve_embedding_paths(embeddings_path, embedding_ids_path)
    log(f"Using embeddings: {emb_path}")
    embeddings = np.load(emb_path, mmap_mode=mmap_mode)
    ids_frame = pd.read_csv(ids_path, dtype={"article_id": str})
    article_ids = ids_frame["article_id"].astype(str).str.zfill(10).tolist()
    article_to_index = {article_id: index for index, article_id in enumerate(article_ids)}
    return embeddings, article_ids, article_to_index


def article_image_path(images_dir: Path, article_id: str) -> Path:
    padded = str(article_id).zfill(10)
    nested = images_dir / padded[:3] / f"{padded}.jpg"
    if nested.exists():
        return nested
    return images_dir / f"{padded}.jpg"
