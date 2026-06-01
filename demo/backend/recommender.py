from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.proposal_v2.features import encode_tabular  # noqa: E402
from src.proposal_v2.models import build_model  # noqa: E402


DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EMBEDDING_DIR = DATA_DIR / "embeddings" / "final_kaggle"
IMAGE_DIR = DATA_DIR / "images" / "hm_images"

ARTIFACT_DIR = PROJECT_DIR / "artifacts" / "final"
V2_MODEL_DIR = ARTIFACT_DIR / "models"
V2_METRICS_DIR = ARTIFACT_DIR / "metrics"
V2_EXPLAIN_DIR = ARTIFACT_DIR / "explainability"

ARTICLES_PATH = RAW_DIR / "articles.csv"
TRANSACTIONS_PATH = RAW_DIR / "transactions_train.csv"
EMBEDDINGS_PATH = EMBEDDING_DIR / "article_image_embeddings_popular.npy"
EMBEDDING_IDS_PATH = EMBEDDING_DIR / "article_image_embedding_ids_popular.csv"

TABULAR_MODEL_PATH = V2_MODEL_DIR / "tabular_only.pt"
IMAGE_MODEL_PATH = V2_MODEL_DIR / "image_history.pt"
FUSION_MODEL_PATH = V2_MODEL_DIR / "late_fusion.pt"
CNN_MODEL_PATH = V2_MODEL_DIR / "image_only_effnet_cnn.pt"

CLASSIFICATION_METRICS_PATH = V2_METRICS_DIR / "proposal_v2_classification_metrics.csv"
RANKING_METRICS_PATH = V2_METRICS_DIR / "proposal_v2_ranking_metrics.csv"
RANKING_SUMMARY_PATH = V2_METRICS_DIR / "proposal_v2_cv_summary.md"
CNN_METRICS_PATH = V2_METRICS_DIR / "proposal_v2_cnn_metrics.csv"
SHAP_SUMMARY_PATH = V2_EXPLAIN_DIR / "proposal_v2_shap_summary.csv"
GRADCAM_DIR = V2_EXPLAIN_DIR / "gradcam_examples"

MODEL_ORDER = ("tabular_only", "image_history", "late_fusion")

DEFAULT_CUSTOMER_PROFILE = {
    "FN": 0,
    "Active": 1,
    "age": 28,
    "club_member_status": "ACTIVE",
    "fashion_news_frequency": "NONE",
}


@dataclass
class Recommendation:
    article_id: str
    score: float
    prod_name: str
    product_type_name: str
    colour_group_name: str
    image_path: str
    visual_similarity: float | None = None
    nearest_history_article_id: str | None = None
    nearest_history_prod_name: str | None = None
    reason_tags: list[str] = field(default_factory=list)


def normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def article_image_path(article_id: str) -> Path:
    padded = str(article_id).zfill(10)
    return IMAGE_DIR / padded[:3] / f"{padded}.jpg"


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def recommendations_to_dict(recommendations: dict[str, list[Recommendation]]) -> dict:
    return {
        model_name: [
            {
                "article_id": item.article_id,
                "score": item.score,
                "prod_name": item.prod_name,
                "product_type_name": item.product_type_name,
                "colour_group_name": item.colour_group_name,
                "visual_similarity": item.visual_similarity,
                "nearest_history_article_id": item.nearest_history_article_id,
                "nearest_history_prod_name": item.nearest_history_prod_name,
                "reason_tags": item.reason_tags,
            }
            for item in items
        ]
        for model_name, items in recommendations.items()
    }


class PerseptronRecommender:
    def __init__(self, device: str | None = None, candidate_limit: int = 800):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.candidate_limit = candidate_limit

        self.articles = self._load_articles()
        self.article_meta = self.articles.set_index("article_id")
        self.article_ids, self.article_to_index = self._load_embedding_ids()
        self.embeddings = normalize_rows(np.load(EMBEDDINGS_PATH, mmap_mode="r"))
        self.popular_articles = self._load_popular_articles()

        self.checkpoints = {
            "tabular_only": load_checkpoint(TABULAR_MODEL_PATH),
            "image_history": load_checkpoint(IMAGE_MODEL_PATH),
            "late_fusion": load_checkpoint(FUSION_MODEL_PATH),
        }
        self.models = {
            name: self._build_loaded_model(checkpoint)
            for name, checkpoint in self.checkpoints.items()
        }

    def _build_loaded_model(self, checkpoint: dict) -> torch.nn.Module:
        model = build_model(checkpoint["model_name"], checkpoint["metadata"], checkpoint["image_dim"]).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _load_articles(self) -> pd.DataFrame:
        articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
        articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)
        for column in articles.columns:
            if articles[column].dtype == "object":
                articles[column] = articles[column].fillna("UNKNOWN").astype(str)
        return articles

    def _load_embedding_ids(self) -> tuple[list[str], dict[str, int]]:
        article_ids = (
            pd.read_csv(EMBEDDING_IDS_PATH, dtype={"article_id": str})["article_id"]
            .astype(str)
            .str.zfill(10)
            .tolist()
        )
        return article_ids, {article_id: index for index, article_id in enumerate(article_ids)}

    def _load_popular_articles(self) -> list[str]:
        transactions = pd.read_csv(TRANSACTIONS_PATH, usecols=["article_id"], dtype={"article_id": str})
        transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
        counts = transactions["article_id"].value_counts()
        counts = counts[counts.index.isin(self.article_to_index)]
        return counts.head(max(self.candidate_limit, 800)).index.tolist()

    def build_candidate_pool(self, history_article_ids: list[str]) -> list[str]:
        history_article_ids = [str(article_id).zfill(10) for article_id in history_article_ids]
        candidates: list[str] = []
        candidates.extend(self.popular_articles[:300])

        history_meta = self.articles[self.articles["article_id"].isin(history_article_ids)]
        if not history_meta.empty:
            product_types = set(history_meta["product_type_no"].dropna().tolist())
            garment_groups = set(history_meta["garment_group_no"].dropna().tolist())
            colours = set(history_meta["colour_group_code"].dropna().tolist())
            similar_meta = self.articles[
                self.articles["product_type_no"].isin(product_types)
                | self.articles["garment_group_no"].isin(garment_groups)
                | self.articles["colour_group_code"].isin(colours)
            ]
            candidates.extend(similar_meta["article_id"].head(500).tolist())

        embedded_history = [article_id for article_id in history_article_ids if article_id in self.article_to_index]
        if embedded_history:
            history_vectors = self.embeddings[[self.article_to_index[article_id] for article_id in embedded_history]]
            profile = normalize_rows(history_vectors.mean(axis=0, keepdims=True))[0]
            scores = np.asarray(self.embeddings @ profile)
            top_indices = np.argsort(-scores)[:700]
            candidates.extend([self.article_ids[index] for index in top_indices])

        unique_candidates = []
        seen = set(history_article_ids)
        for article_id in candidates:
            article_id = str(article_id).zfill(10)
            if article_id in seen or article_id not in self.article_to_index:
                continue
            seen.add(article_id)
            unique_candidates.append(article_id)
            if len(unique_candidates) >= self.candidate_limit:
                break
        return unique_candidates

    def _customer_frame(self, candidate_ids: list[str], customer_profile: dict) -> pd.DataFrame:
        base = pd.DataFrame([customer_profile] * len(candidate_ids))
        base["article_id"] = candidate_ids
        frame = base.merge(self.articles, on="article_id", how="left")
        return frame

    def _visual_features(
        self,
        candidate_ids: list[str],
        history_article_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[str | None, str | None]]]:
        image_dim = int(self.embeddings.shape[1])
        candidate_indices = [self.article_to_index[article_id] for article_id in candidate_ids]
        article_emb = np.asarray(self.embeddings[candidate_indices], dtype="float32")

        embedded_history = [str(article_id).zfill(10) for article_id in history_article_ids if str(article_id).zfill(10) in self.article_to_index]
        nearest: dict[str, tuple[str | None, str | None]] = {}
        if embedded_history:
            history_indices = [self.article_to_index[article_id] for article_id in embedded_history]
            history_vectors = np.asarray(self.embeddings[history_indices], dtype="float32")
            profile = normalize_rows(history_vectors.mean(axis=0, keepdims=True))[0]
            profile_emb = np.repeat(profile.reshape(1, -1), len(candidate_ids), axis=0).astype("float32")
            visual_similarity = np.asarray(article_emb @ profile, dtype="float32")
            history_count = np.full(len(candidate_ids), len(embedded_history), dtype="float32")
            history_scores = article_emb @ history_vectors.T
            best_indices = np.argmax(history_scores, axis=1)
            for row_index, article_id in enumerate(candidate_ids):
                history_id = embedded_history[int(best_indices[row_index])]
                prod_name = None
                if history_id in self.article_meta.index:
                    prod_name = str(self.article_meta.loc[history_id].get("prod_name", ""))
                nearest[article_id] = (history_id, prod_name)
        else:
            profile_emb = np.zeros((len(candidate_ids), image_dim), dtype="float32")
            visual_similarity = np.zeros(len(candidate_ids), dtype="float32")
            history_count = np.zeros(len(candidate_ids), dtype="float32")
            for article_id in candidate_ids:
                nearest[article_id] = (None, None)

        return article_emb, profile_emb, visual_similarity, history_count, nearest

    def _score_model(
        self,
        model_name: str,
        frame: pd.DataFrame,
        article_emb: np.ndarray,
        profile_emb: np.ndarray,
        visual_similarity: np.ndarray,
        visual_history_count: np.ndarray,
    ) -> np.ndarray:
        checkpoint = self.checkpoints[model_name]
        model = self.models[model_name]
        batch_size = 4096
        scores: list[np.ndarray] = []

        numeric = categorical = None
        if model_name in {"tabular_only", "late_fusion"}:
            numeric, categorical = encode_tabular(frame, checkpoint["metadata"])

        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                end = start + batch_size
                if model_name == "tabular_only":
                    logits = model(
                        torch.tensor(numeric[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(categorical[start:end], dtype=torch.long, device=self.device),
                    )
                elif model_name == "image_history":
                    logits = model(
                        torch.tensor(article_emb[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(profile_emb[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(visual_similarity[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(visual_history_count[start:end], dtype=torch.float32, device=self.device),
                    )
                elif model_name == "late_fusion":
                    logits = model(
                        torch.tensor(numeric[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(categorical[start:end], dtype=torch.long, device=self.device),
                        torch.tensor(article_emb[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(profile_emb[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(visual_similarity[start:end], dtype=torch.float32, device=self.device),
                        torch.tensor(visual_history_count[start:end], dtype=torch.float32, device=self.device),
                    )
                else:
                    raise ValueError(model_name)
                scores.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(scores) if scores else np.array([], dtype="float32")

    def _reason_tags(self, row: pd.Series, history_meta: pd.DataFrame, visual_similarity: float) -> list[str]:
        tags = []
        if not history_meta.empty:
            if row.get("product_type_no") in set(history_meta["product_type_no"].dropna().tolist()):
                tags.append("Urun tipi uyumu")
            if row.get("garment_group_no") in set(history_meta["garment_group_no"].dropna().tolist()):
                tags.append("Benzer kategori")
            if row.get("colour_group_code") in set(history_meta["colour_group_code"].dropna().tolist()):
                tags.append("Renk uyumu")
        if visual_similarity >= 0.45:
            tags.append("Gorsel gecmis")
        if not tags:
            tags.append("Aday havuzu secimi")
        return tags[:3]

    def recommend(
        self,
        history_article_ids: list[str],
        customer_profile: dict,
        top_k: int = 8,
    ) -> dict[str, list[Recommendation]]:
        customer_profile = {**DEFAULT_CUSTOMER_PROFILE, **customer_profile}
        history_article_ids = [str(article_id).zfill(10) for article_id in history_article_ids]
        candidates = self.build_candidate_pool(history_article_ids)
        if not candidates:
            return {name: [] for name in MODEL_ORDER}

        frame = self._customer_frame(candidates, customer_profile)
        article_emb, profile_emb, visual_similarity, visual_history_count, nearest = self._visual_features(
            candidates,
            history_article_ids,
        )
        frame["visual_similarity"] = visual_similarity
        frame["visual_history_count"] = visual_history_count
        history_meta = self.articles[self.articles["article_id"].isin(history_article_ids)]

        output: dict[str, list[Recommendation]] = {}
        for model_name in MODEL_ORDER:
            scores = self._score_model(model_name, frame, article_emb, profile_emb, visual_similarity, visual_history_count)
            order = np.argsort(-scores)[:top_k]
            items = []
            for index in order:
                row = frame.iloc[int(index)]
                article_id = str(row["article_id"]).zfill(10)
                nearest_id, nearest_name = nearest.get(article_id, (None, None))
                items.append(
                    Recommendation(
                        article_id=article_id,
                        score=float(scores[int(index)]),
                        prod_name=str(row.get("prod_name", "")),
                        product_type_name=str(row.get("product_type_name", "")),
                        colour_group_name=str(row.get("colour_group_name", "")),
                        image_path=str(article_image_path(article_id)),
                        visual_similarity=float(visual_similarity[int(index)]),
                        nearest_history_article_id=nearest_id,
                        nearest_history_prod_name=nearest_name,
                        reason_tags=self._reason_tags(row, history_meta, float(visual_similarity[int(index)])),
                    )
                )
            output[model_name] = items
        return output
