from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(os.environ.get("PERSEPTRON_PROJECT_DIR", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports" / "proposal_v2"
MODELS_DIR = PROJECT_DIR / "models" / "proposal_v2"

DEFAULT_SEED = 42
DEFAULT_N_FOLDS = 5
DEFAULT_VALIDATION_DAYS = 7

CUSTOMER_NUMERIC_FEATURES = ["FN", "Active", "age"]
ARTICLE_NUMERIC_FEATURES = [
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
]
VISUAL_NUMERIC_FEATURES = ["visual_similarity", "visual_history_count"]
TABULAR_NUMERIC_FEATURES = CUSTOMER_NUMERIC_FEATURES + ARTICLE_NUMERIC_FEATURES
FUSION_NUMERIC_FEATURES = TABULAR_NUMERIC_FEATURES + VISUAL_NUMERIC_FEATURES

CUSTOMER_CATEGORICAL_FEATURES = ["club_member_status", "fashion_news_frequency"]
ARTICLE_CATEGORICAL_FEATURES = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_code",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
]
CATEGORICAL_FEATURES = CUSTOMER_CATEGORICAL_FEATURES + ARTICLE_CATEGORICAL_FEATURES

MODEL_NAMES = ("tabular_only", "image_history", "late_fusion")


@dataclass(frozen=True)
class V2Defaults:
    n_folds: int = DEFAULT_N_FOLDS
    validation_days: int = DEFAULT_VALIDATION_DAYS
    seed: int = DEFAULT_SEED
    top_k: int = 12
    precision_k: int = 10
    candidate_limit: int = 5000
    visual_neighbors: int = 3000
    co_purchase_per_item: int = 300
    hybrid_weights: tuple[float, ...] = (0.25, 0.45, 0.65)
    negatives_per_positive: int = 1
    train_batch_size: int = 4096
    epochs: int = 3
    learning_rate: float = 1e-3


def ensure_v2_dirs() -> None:
    for path in [
        REPORTS_DIR,
        REPORTS_DIR / "folds",
        REPORTS_DIR / "gradcam_examples",
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
