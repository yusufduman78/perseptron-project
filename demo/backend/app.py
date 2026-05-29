from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommender import (
    ARTICLES_PATH,
    DEFAULT_CUSTOMER_PROFILE,
    EMBEDDING_IDS_PATH,
    EMBEDDINGS_PATH,
    FUSION_MODEL_PATH,
    IMAGE_DIR,
    IMAGE_MODEL_PATH,
    TRANSACTIONS_PATH,
    TABULAR_MODEL_PATH,
    PerseptronRecommender,
    Recommendation,
)


app = FastAPI(title="Perseptron Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")


class CustomerProfile(BaseModel):
    FN: int = DEFAULT_CUSTOMER_PROFILE["FN"]
    Active: int = DEFAULT_CUSTOMER_PROFILE["Active"]
    age: int = DEFAULT_CUSTOMER_PROFILE["age"]
    club_member_status: str = DEFAULT_CUSTOMER_PROFILE["club_member_status"]
    fashion_news_frequency: str = DEFAULT_CUSTOMER_PROFILE["fashion_news_frequency"]


class RecommendRequest(BaseModel):
    history_article_ids: list[str] = Field(default_factory=list)
    customer_profile: CustomerProfile = Field(default_factory=CustomerProfile)
    top_k: int = 8
    candidate_limit: int = 500


DEMO_SCENARIOS = [
    {
        "id": "black_basics",
        "title": "Black Basics",
        "subtitle": "Jersey ve knitwear ağırlıklı siyah günlük stil.",
        "history_article_ids": ["0108775015", "0118458028", "0118458038", "0120129001"],
        "customer_profile": {
            **DEFAULT_CUSTOMER_PROFILE,
            "age": 24,
            "fashion_news_frequency": "Regularly",
        },
    },
    {
        "id": "dress_style",
        "title": "Dress Style",
        "subtitle": "Elbise odaklı, renk ve siluet çeşitliliği olan geçmiş.",
        "history_article_ids": ["0192460006", "0202017055", "0212629031", "0212629032"],
        "customer_profile": {
            **DEFAULT_CUSTOMER_PROFILE,
            "FN": 1,
            "age": 31,
            "fashion_news_frequency": "Monthly",
        },
    },
    {
        "id": "denim_casual",
        "title": "Denim Casual",
        "subtitle": "Trousers/denim tabanlı rahat hafta sonu alışverişi.",
        "history_article_ids": ["0156289011", "0212766041", "0212766042", "0212766043"],
        "customer_profile": {
            **DEFAULT_CUSTOMER_PROFILE,
            "age": 27,
        },
    },
    {
        "id": "bright_knit",
        "title": "Bright Knit",
        "subtitle": "Knitwear ağırlıklı, daha renkli ve sıcak bir profil.",
        "history_article_ids": ["0192460006", "0216081011", "0244267001", "0244267017"],
        "customer_profile": {
            **DEFAULT_CUSTOMER_PROFILE,
            "FN": 1,
            "age": 35,
            "fashion_news_frequency": "Regularly",
        },
    },
]


def image_url(article_id: str) -> str:
    padded = str(article_id).zfill(10)
    if not article_image_path(padded).exists():
        return f"/api/image-placeholder/{padded}.svg"
    return f"/images/{padded[:3]}/{padded}.jpg"


def article_image_path(article_id: str) -> Path:
    padded = str(article_id).zfill(10)
    return IMAGE_DIR / padded[:3] / f"{padded}.jpg"


def has_article_image(article_id: str) -> bool:
    return article_image_path(article_id).exists()


def recommendation_to_payload(item: Recommendation) -> dict:
    nearest_history = None
    if item.nearest_history_article_id:
        nearest_history = {
            "article_id": item.nearest_history_article_id,
            "prod_name": item.nearest_history_prod_name,
            "image_url": image_url(item.nearest_history_article_id),
        }
    return {
        "article_id": item.article_id,
        "score": item.score,
        "prod_name": item.prod_name,
        "product_type_name": item.product_type_name,
        "colour_group_name": item.colour_group_name,
        "image_url": image_url(item.article_id),
        "explanation": {
            "visual_similarity": item.visual_similarity,
            "nearest_history": nearest_history,
            "reason_tags": item.reason_tags,
        },
    }


def article_to_payload(article_id: str) -> dict | None:
    padded = str(article_id).zfill(10)
    articles = load_articles()
    rows = articles[articles["article_id"] == padded]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "article_id": padded,
        "prod_name": str(row.get("prod_name", "")),
        "product_type_name": str(row.get("product_type_name", "")),
        "colour_group_name": str(row.get("colour_group_name", "")),
        "garment_group_name": str(row.get("garment_group_name", "")),
        "image_url": image_url(padded),
    }


def build_comparison(recommendations: dict[str, list[Recommendation]]) -> dict:
    ids_by_model = {
        model_name: [item.article_id for item in items]
        for model_name, items in recommendations.items()
    }
    counts = Counter(article_id for ids in ids_by_model.values() for article_id in ids)
    payload_by_id = {
        item.article_id: recommendation_to_payload(item)
        for items in recommendations.values()
        for item in items
    }
    shared_items = []
    for article_id, count in counts.items():
        if count <= 1:
            continue
        item = payload_by_id[article_id]
        item["shared_by"] = [model for model, ids in ids_by_model.items() if article_id in ids]
        shared_items.append(item)

    unique_by_model = {
        model_name: [payload_by_id[article_id] for article_id in ids if counts[article_id] == 1]
        for model_name, ids in ids_by_model.items()
    }
    overlap_matrix = {}
    model_names = list(ids_by_model)
    for first_index, first_model in enumerate(model_names):
        for second_model in model_names[first_index + 1 :]:
            overlap_matrix[f"{first_model}__{second_model}"] = len(
                set(ids_by_model[first_model]) & set(ids_by_model[second_model])
            )

    return {
        "ids_by_model": ids_by_model,
        "shared_items": shared_items,
        "unique_by_model": unique_by_model,
        "overlap_matrix": overlap_matrix,
    }


@lru_cache(maxsize=1)
def load_articles() -> pd.DataFrame:
    articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)
    return articles


@lru_cache(maxsize=1)
def popular_article_ids() -> list[str]:
    transactions = pd.read_csv(TRANSACTIONS_PATH, usecols=["article_id"], dtype={"article_id": str})
    transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
    return transactions["article_id"].value_counts().head(400).index.tolist()


@lru_cache(maxsize=4)
def get_recommender(candidate_limit: int = 500) -> PerseptronRecommender:
    return PerseptronRecommender(candidate_limit=candidate_limit)


@app.get("/api/health")
def health() -> dict:
    checkpoint_files = {
        "tabular_only": TABULAR_MODEL_PATH.exists(),
        "image_history": IMAGE_MODEL_PATH.exists(),
        "late_fusion": FUSION_MODEL_PATH.exists(),
    }
    data_files = {
        "articles": ARTICLES_PATH.exists(),
        "transactions": TRANSACTIONS_PATH.exists(),
        "embeddings": EMBEDDINGS_PATH.exists(),
        "embedding_ids": EMBEDDING_IDS_PATH.exists(),
        "images_dir": IMAGE_DIR.exists(),
    }
    cuda_available = torch.cuda.is_available()
    return {
        "status": "ok" if all(checkpoint_files.values()) and all(data_files.values()) else "degraded",
        "device": {
            "cuda_available": cuda_available,
            "name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        },
        "checkpoints": checkpoint_files,
        "data": data_files,
    }


@app.get("/api/customer-defaults")
def customer_defaults() -> dict:
    return DEFAULT_CUSTOMER_PROFILE


@app.get("/api/demo-scenarios")
def demo_scenarios() -> dict:
    scenarios = []
    for scenario in DEMO_SCENARIOS:
        items = [
            item
            for item in (article_to_payload(article_id) for article_id in scenario["history_article_ids"])
            if item is not None
        ]
        scenarios.append({**scenario, "items": items})
    return {"scenarios": scenarios}


@app.get("/api/image-placeholder/{article_id}.svg")
def image_placeholder(article_id: str) -> Response:
    padded = str(article_id).zfill(10)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="360" height="480" viewBox="0 0 360 480">
  <rect width="360" height="480" fill="#eef1f4"/>
  <rect x="52" y="54" width="256" height="340" rx="18" fill="#ffffff" stroke="#cbd3df" stroke-width="3"/>
  <path d="M138 144c10 26 74 26 84 0l46 26-27 63-25-10v112H144V223l-25 10-27-63 46-26z" fill="#d9e2ec" stroke="#9aa7b8" stroke-width="4" stroke-linejoin="round"/>
  <text x="180" y="424" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#52606d">Görsel bulunamadı</text>
  <text x="180" y="452" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#829ab1">{padded}</text>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/catalog")
def catalog(limit: int = 60, q: str = "") -> dict:
    articles = load_articles()
    popular = popular_article_ids()
    if q:
        query = q.lower()
        subset = articles[
            articles["prod_name"].fillna("").str.lower().str.contains(query)
            | articles["product_type_name"].fillna("").str.lower().str.contains(query)
            | articles["colour_group_name"].fillna("").str.lower().str.contains(query)
        ].copy()
    else:
        subset = articles[articles["article_id"].isin(popular)].copy()
        subset["popular_rank"] = subset["article_id"].map({article_id: i for i, article_id in enumerate(popular)})
        subset = subset.sort_values("popular_rank")

    items = []
    for row in subset.itertuples(index=False):
        article_id = row.article_id
        if not has_article_image(article_id):
            continue
        items.append(
            {
                "article_id": article_id,
                "prod_name": getattr(row, "prod_name", ""),
                "product_type_name": getattr(row, "product_type_name", ""),
                "colour_group_name": getattr(row, "colour_group_name", ""),
                "garment_group_name": getattr(row, "garment_group_name", ""),
                "image_url": image_url(article_id),
            }
        )
        if len(items) >= limit:
            break
    return {"items": items}


@app.post("/api/recommend")
def recommend(request: RecommendRequest) -> dict:
    recommender = get_recommender(request.candidate_limit)
    profile = request.customer_profile.model_dump()
    recommendations = recommender.recommend(
        history_article_ids=request.history_article_ids,
        customer_profile=profile,
        top_k=request.top_k,
    )
    return {
        "models": {
            model_name: [recommendation_to_payload(item) for item in items]
            for model_name, items in recommendations.items()
        },
        "comparison": build_comparison(recommendations),
        "request_context": {
            "history_count": len(request.history_article_ids),
            "scoreable_history_count": len(
                [
                    article_id
                    for article_id in request.history_article_ids
                    if str(article_id).zfill(10) in recommender.article_to_index
                ]
            ),
            "candidate_limit": request.candidate_limit,
            "device": str(recommender.device),
        },
    }


def main() -> None:
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(Path(__file__).parent))


if __name__ == "__main__":
    main()
