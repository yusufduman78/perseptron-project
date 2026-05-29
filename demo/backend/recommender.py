from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EMBEDDING_DIR = DATA_DIR / "embeddings" / "final_kaggle"
MODEL_DIR = PROJECT_DIR / "models" / "final_kaggle"
IMAGE_DIR = DATA_DIR / "images" / "hm_images"

ARTICLES_PATH = RAW_DIR / "articles.csv"
TRANSACTIONS_PATH = RAW_DIR / "transactions_train.csv"
EMBEDDINGS_PATH = EMBEDDING_DIR / "article_image_embeddings_popular.npy"
EMBEDDING_IDS_PATH = EMBEDDING_DIR / "article_image_embedding_ids_popular.csv"

TABULAR_MODEL_PATH = MODEL_DIR / "tabular_only_streaming_full.pt"
IMAGE_MODEL_PATH = MODEL_DIR / "image_history_streaming_full.pt"
FUSION_MODEL_PATH = MODEL_DIR / "multimodal_fusion_streaming_full.pt"


DEFAULT_CUSTOMER_PROFILE = {
    "FN": 0,
    "Active": 1,
    "age": 28,
    "club_member_status": "ACTIVE",
    "fashion_news_frequency": "NONE",
}


class TabularOnlyMLP(nn.Module):
    def __init__(self, numeric_dim: int, category_sizes: list[int]):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, min(50, max(4, (size + 1) // 2))) for size in category_sizes]
        )
        categorical_dim = sum(layer.embedding_dim for layer in self.embeddings)
        self.network = nn.Sequential(
            nn.Linear(numeric_dim + categorical_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        x = torch.cat([numeric, *embedded], dim=1)
        return self.network(x).squeeze(1)


class ImageHistoryMLP(nn.Module):
    def __init__(self, image_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(image_dim * 2 + 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        article_emb: torch.Tensor,
        profile_emb: torch.Tensor,
        visual_similarity: torch.Tensor,
        visual_history_count: torch.Tensor,
    ) -> torch.Tensor:
        visual_extra = torch.stack(
            [visual_similarity, torch.log1p(torch.clamp(visual_history_count, min=0.0))],
            dim=1,
        )
        x = torch.cat([article_emb, profile_emb, visual_extra], dim=1)
        return self.network(x).squeeze(1)


class MultimodalFusion(nn.Module):
    def __init__(self, numeric_dim: int, category_sizes: list[int], image_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, min(50, max(4, (size + 1) // 2))) for size in category_sizes]
        )
        categorical_dim = sum(layer.embedding_dim for layer in self.embeddings)
        self.tabular_branch = nn.Sequential(
            nn.Linear(numeric_dim + categorical_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.visual_branch = nn.Sequential(
            nn.Linear(image_dim * 2 + 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        article_emb: torch.Tensor,
        profile_emb: torch.Tensor,
        visual_similarity: torch.Tensor,
        visual_history_count: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        tabular_repr = self.tabular_branch(torch.cat([numeric, *embedded], dim=1))
        visual_extra = torch.stack(
            [visual_similarity, torch.log1p(torch.clamp(visual_history_count, min=0.0))],
            dim=1,
        )
        visual_repr = self.visual_branch(torch.cat([article_emb, profile_emb, visual_extra], dim=1))
        return self.fusion_head(torch.cat([tabular_repr, visual_repr], dim=1)).squeeze(1)


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
    values = values.astype("float32", copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def article_image_path(article_id: str) -> Path:
    padded = str(article_id).zfill(10)
    return IMAGE_DIR / padded[:3] / f"{padded}.jpg"


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


class PerseptronRecommender:
    def __init__(self, device: str | None = None, candidate_limit: int = 800):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.candidate_limit = candidate_limit

        self.articles = self._load_articles()
        self.article_meta = self.articles.set_index("article_id")
        self.article_ids, self.article_to_index = self._load_embedding_ids()
        self.embeddings = normalize_rows(np.load(EMBEDDINGS_PATH, mmap_mode="r"))
        self.popular_articles = self._load_popular_articles()

        self.tabular_checkpoint = load_checkpoint(TABULAR_MODEL_PATH)
        self.image_checkpoint = load_checkpoint(IMAGE_MODEL_PATH)
        self.fusion_checkpoint = load_checkpoint(FUSION_MODEL_PATH)

        image_dim = int(self.embeddings.shape[1])
        self.tabular_model = TabularOnlyMLP(
            len(self.tabular_checkpoint["metadata"]["numeric_features"]),
            self.tabular_checkpoint["metadata"]["category_sizes"],
        ).to(self.device)
        self.tabular_model.load_state_dict(self.tabular_checkpoint["model_state_dict"])
        self.tabular_model.eval()

        self.image_model = ImageHistoryMLP(image_dim).to(self.device)
        self.image_model.load_state_dict(self.image_checkpoint["model_state_dict"])
        self.image_model.eval()

        self.fusion_model = MultimodalFusion(
            len(self.fusion_checkpoint["metadata"]["numeric_features"]),
            self.fusion_checkpoint["metadata"]["category_sizes"],
            image_dim,
        ).to(self.device)
        self.fusion_model.load_state_dict(self.fusion_checkpoint["model_state_dict"])
        self.fusion_model.eval()

    def _load_articles(self) -> pd.DataFrame:
        articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
        articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)
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
        transactions = pd.read_csv(
            TRANSACTIONS_PATH,
            usecols=["article_id"],
            dtype={"article_id": str},
        )
        transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
        counts = transactions["article_id"].value_counts()
        counts = counts[counts.index.isin(self.article_to_index)]
        return counts.head(500).index.tolist()

    def build_candidate_pool(self, history_article_ids: list[str]) -> list[str]:
        history_article_ids = [article_id.zfill(10) for article_id in history_article_ids]
        candidates = []
        candidates.extend(self.popular_articles[:300])

        history_meta = self.articles[self.articles["article_id"].isin(history_article_ids)]
        if not history_meta.empty:
            product_types = set(history_meta["product_type_no"].dropna().tolist())
            garment_groups = set(history_meta["garment_group_no"].dropna().tolist())
            similar_meta = self.articles[
                self.articles["product_type_no"].isin(product_types)
                | self.articles["garment_group_no"].isin(garment_groups)
            ]
            candidates.extend(similar_meta["article_id"].head(500).tolist())

        embedded_history = [article_id for article_id in history_article_ids if article_id in self.article_to_index]
        if embedded_history:
            history_vectors = self.embeddings[[self.article_to_index[article_id] for article_id in embedded_history]]
            profile = normalize_rows(history_vectors.mean(axis=0, keepdims=True))[0]
            scores = self.embeddings @ profile
            top_indices = np.argsort(-scores)[:500]
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

    def build_profile(self, history_article_ids: list[str]) -> tuple[np.ndarray, float]:
        valid_ids = [article_id.zfill(10) for article_id in history_article_ids if article_id.zfill(10) in self.article_to_index]
        if not valid_ids:
            profile = np.zeros(self.embeddings.shape[1], dtype="float32")
            return profile, 0.0
        vectors = self.embeddings[[self.article_to_index[article_id] for article_id in valid_ids]]
        profile = normalize_rows(vectors.mean(axis=0, keepdims=True))[0]
        return profile.astype("float32"), float(len(valid_ids))

    def make_scoring_frame(self, candidate_ids: list[str], customer_profile: dict) -> pd.DataFrame:
        customer_values = {**DEFAULT_CUSTOMER_PROFILE, **customer_profile}
        rows = []
        for article_id in candidate_ids:
            row = {"article_id": article_id}
            row.update(customer_values)
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame = frame.merge(self.articles, on="article_id", how="left")
        return frame

    def encode_tabular(self, frame: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
        frame = frame.copy()
        for column in metadata["numeric_features"]:
            if column not in frame.columns:
                frame[column] = 0.0
        for column in metadata["categorical_features"]:
            if column not in frame.columns:
                frame[column] = "__MISSING__"

        numeric_values = frame[metadata["numeric_features"]].copy()
        for col in metadata["numeric_features"]:
            numeric_values[col] = pd.to_numeric(numeric_values[col], errors="coerce")
            numeric_values[col] = numeric_values[col].fillna(0)
        numeric = numeric_values.astype("float32").to_numpy()
        numeric = (numeric - metadata["numeric_mean"]) / metadata["numeric_std"]

        categorical_arrays = []
        for col in metadata["categorical_features"]:
            values = frame[col].fillna("__MISSING__").astype(str)
            mapping = metadata["category_maps"][col]
            categorical_arrays.append(values.map(mapping).fillna(0).astype("int64").to_numpy())
        return numeric.astype("float32"), np.stack(categorical_arrays, axis=1).astype("int64")

    def visual_arrays(self, candidate_ids: list[str], profile: np.ndarray, history_count: float):
        article_indices = [self.article_to_index[article_id] for article_id in candidate_ids]
        article_embeddings = self.embeddings[article_indices].astype("float32")
        profile_embeddings = np.repeat(profile.reshape(1, -1), len(candidate_ids), axis=0).astype("float32")
        visual_similarity = np.sum(article_embeddings * profile_embeddings, axis=1).astype("float32")
        visual_history_count = np.full(len(candidate_ids), history_count, dtype="float32")
        return article_embeddings, profile_embeddings, visual_similarity, visual_history_count

    def score_tabular(self, frame: pd.DataFrame) -> np.ndarray:
        numeric, categorical = self.encode_tabular(frame, self.tabular_checkpoint["metadata"])
        with torch.no_grad():
            logits = self.tabular_model(
                torch.tensor(numeric, dtype=torch.float32, device=self.device),
                torch.tensor(categorical, dtype=torch.long, device=self.device),
            )
        return torch.sigmoid(logits).cpu().numpy()

    def score_image(self, candidate_ids: list[str], profile: np.ndarray, history_count: float) -> np.ndarray:
        article_emb, profile_emb, visual_similarity, visual_history_count = self.visual_arrays(
            candidate_ids, profile, history_count
        )
        with torch.no_grad():
            logits = self.image_model(
                torch.tensor(article_emb, dtype=torch.float32, device=self.device),
                torch.tensor(profile_emb, dtype=torch.float32, device=self.device),
                torch.tensor(visual_similarity, dtype=torch.float32, device=self.device),
                torch.tensor(visual_history_count, dtype=torch.float32, device=self.device),
            )
        return torch.sigmoid(logits).cpu().numpy()

    def score_fusion(
        self,
        frame: pd.DataFrame,
        candidate_ids: list[str],
        profile: np.ndarray,
        history_count: float,
    ) -> np.ndarray:
        article_emb, profile_emb, visual_similarity, visual_history_count = self.visual_arrays(
            candidate_ids, profile, history_count
        )
        fusion_frame = frame.copy()
        fusion_frame["visual_similarity"] = visual_similarity
        fusion_frame["visual_history_count"] = visual_history_count
        numeric, categorical = self.encode_tabular(fusion_frame, self.fusion_checkpoint["metadata"])
        with torch.no_grad():
            logits = self.fusion_model(
                torch.tensor(numeric, dtype=torch.float32, device=self.device),
                torch.tensor(categorical, dtype=torch.long, device=self.device),
                torch.tensor(article_emb, dtype=torch.float32, device=self.device),
                torch.tensor(profile_emb, dtype=torch.float32, device=self.device),
                torch.tensor(visual_similarity, dtype=torch.float32, device=self.device),
                torch.tensor(visual_history_count, dtype=torch.float32, device=self.device),
            )
        return torch.sigmoid(logits).cpu().numpy()

    def format_recommendations(
        self,
        candidate_ids: list[str],
        scores: np.ndarray,
        top_k: int,
        history_article_ids: list[str] | None = None,
        profile: np.ndarray | None = None,
    ) -> list[Recommendation]:
        order = np.argsort(-scores)[:top_k]
        results = []
        history_article_ids = [str(article_id).zfill(10) for article_id in (history_article_ids or [])]
        history_meta = self.articles[self.articles["article_id"].isin(history_article_ids)]
        history_product_types = set(history_meta["product_type_no"].dropna().astype(str))
        history_garment_groups = set(history_meta["garment_group_no"].dropna().astype(str))
        history_colours = set(history_meta["colour_group_name"].dropna().astype(str))
        embedded_history = [article_id for article_id in history_article_ids if article_id in self.article_to_index]
        history_vectors = None
        if embedded_history:
            history_vectors = self.embeddings[[self.article_to_index[article_id] for article_id in embedded_history]]

        for index in order:
            article_id = candidate_ids[int(index)]
            meta = self.article_meta.loc[article_id]
            visual_similarity = None
            nearest_history_article_id = None
            nearest_history_prod_name = None
            nearest_similarity = None
            if profile is not None and article_id in self.article_to_index:
                article_vector = self.embeddings[self.article_to_index[article_id]]
                visual_similarity = float(np.dot(article_vector, profile))
                if history_vectors is not None:
                    similarities = history_vectors @ article_vector
                    nearest_index = int(np.argmax(similarities))
                    nearest_similarity = float(similarities[nearest_index])
                    nearest_history_article_id = embedded_history[nearest_index]
                    nearest_meta = self.article_meta.loc[nearest_history_article_id]
                    nearest_history_prod_name = str(nearest_meta.get("prod_name", ""))

            reason_tags = []
            if str(meta.get("product_type_no", "")) in history_product_types:
                reason_tags.append("Aynı ürün tipi")
            if str(meta.get("garment_group_no", "")) in history_garment_groups:
                reason_tags.append("Benzer giyim grubu")
            if str(meta.get("colour_group_name", "")) in history_colours:
                reason_tags.append("Renk eşleşmesi")
            if nearest_similarity is not None and nearest_similarity >= 0.55:
                reason_tags.append("Yakın görsel geçmiş")
            elif visual_similarity is not None and visual_similarity >= 0.30:
                reason_tags.append("Görsel profil uyumu")
            if article_id in self.popular_articles[:60]:
                reason_tags.append("Popüler ürün")
            if not reason_tags:
                reason_tags.append("Aday havuzu seçimi")

            results.append(
                Recommendation(
                    article_id=article_id,
                    score=float(scores[int(index)]),
                    prod_name=str(meta.get("prod_name", "")),
                    product_type_name=str(meta.get("product_type_name", "")),
                    colour_group_name=str(meta.get("colour_group_name", "")),
                    image_path=str(article_image_path(article_id)),
                    visual_similarity=visual_similarity,
                    nearest_history_article_id=nearest_history_article_id,
                    nearest_history_prod_name=nearest_history_prod_name,
                    reason_tags=reason_tags[:3],
                )
            )
        return results

    def recommend(
        self,
        history_article_ids: list[str],
        customer_profile: dict | None = None,
        top_k: int = 12,
    ) -> dict[str, list[Recommendation]]:
        customer_profile = customer_profile or {}
        history_article_ids = [str(article_id).zfill(10) for article_id in history_article_ids]
        candidate_ids = self.build_candidate_pool(history_article_ids)
        if not candidate_ids:
            raise ValueError("No scoreable candidate articles could be built from the provided history.")

        profile, history_count = self.build_profile(history_article_ids)
        frame = self.make_scoring_frame(candidate_ids, customer_profile)

        tabular_scores = self.score_tabular(frame)
        image_scores = self.score_image(candidate_ids, profile, history_count)
        fusion_scores = self.score_fusion(frame, candidate_ids, profile, history_count)

        return {
            "tabular_only": self.format_recommendations(
                candidate_ids, tabular_scores, top_k, history_article_ids, profile
            ),
            "image_history": self.format_recommendations(
                candidate_ids, image_scores, top_k, history_article_ids, profile
            ),
            "late_fusion": self.format_recommendations(
                candidate_ids, fusion_scores, top_k, history_article_ids, profile
            ),
        }


def recommendations_to_dict(recommendations: dict[str, list[Recommendation]]) -> dict[str, list[dict]]:
    return {
        model_name: [item.__dict__ for item in model_recommendations]
        for model_name, model_recommendations in recommendations.items()
    }
