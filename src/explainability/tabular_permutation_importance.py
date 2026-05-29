from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[2]
SANDBOX_DIR = PROJECT_DIR / "data" / "processed" / "sandbox"
EMBEDDINGS_DIR = PROJECT_DIR / "data" / "embeddings" / "local_sandbox"
MODELS_DIR = PROJECT_DIR / "models" / "local_experiments"
REPORTS_DIR = PROJECT_DIR / "reports"

VISUAL_FEATURES_PATH = SANDBOX_DIR / "visual_similarity_features.csv"
CUSTOMERS_PATH = SANDBOX_DIR / "sandbox_customers.csv"
ARTICLES_PATH = SANDBOX_DIR / "sandbox_articles.csv"
EMBEDDING_IDS_PATH = EMBEDDINGS_DIR / "article_image_embedding_ids.csv"
DEFAULT_CHECKPOINT_PATH = MODELS_DIR / "controlled_tabular_only_mlp.pt"
DEFAULT_FEATURE_OUTPUT = REPORTS_DIR / "tabular_permutation_importance.csv"
DEFAULT_GROUP_OUTPUT = REPORTS_DIR / "tabular_feature_group_importance.csv"

CUSTOMER_NUMERIC_FEATURES = {"FN", "Active", "age"}
CUSTOMER_CATEGORICAL_FEATURES = {"club_member_status", "fashion_news_frequency"}


class TabularDataset(Dataset):
    def __init__(self, numeric_values: np.ndarray, categorical_values: np.ndarray, labels: np.ndarray) -> None:
        self.numeric_values = torch.tensor(numeric_values, dtype=torch.float32)
        self.categorical_values = torch.tensor(categorical_values, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.numeric_values[index], self.categorical_values[index], self.labels[index]


class TabularMLP(nn.Module):
    def __init__(self, numeric_dim: int, category_sizes: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, min(50, max(4, (size + 1) // 2))) for size in category_sizes]
        )
        categorical_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
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

    def forward(self, numeric_values: torch.Tensor, categorical_values: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(categorical_values[:, index]) for index, embedding in enumerate(self.embeddings)]
        x = torch.cat([numeric_values, *embedded], dim=1)
        return self.network(x).squeeze(1)


def log(message: str) -> None:
    print(message, flush=True)


def load_checkpoint(path: Path) -> dict:
    log(f"Loading checkpoint: {path}")
    # Local project checkpoint, trusted source. weights_only=False is needed for metadata with numpy arrays.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_article_universe() -> set[str]:
    ids = pd.read_csv(EMBEDDING_IDS_PATH, dtype={"article_id": str})["article_id"].astype(str).str.zfill(10)
    return set(ids.tolist())


def load_controlled_data(metadata: dict, max_rows: int, random_seed: int) -> pd.DataFrame:
    log("Loading controlled tabular rows.")
    embedded_article_ids = load_article_universe()

    needed_columns = ["customer_id", "article_id", "label"]
    base = pd.read_csv(VISUAL_FEATURES_PATH, dtype={"article_id": str}, usecols=needed_columns)
    base["article_id"] = base["article_id"].astype(str).str.zfill(10)
    base = base[base["article_id"].isin(embedded_article_ids)].copy()

    customers = pd.read_csv(CUSTOMERS_PATH)
    articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

    data = base.merge(customers, on="customer_id", how="left")
    data = data.merge(articles, on="article_id", how="left")

    if max_rows > 0 and len(data) > max_rows:
        data = data.groupby("label", group_keys=False).sample(
            frac=max_rows / len(data),
            random_state=random_seed,
        )
        data = data.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    required = metadata["numeric_features"] + metadata["categorical_features"] + ["label"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for tabular importance: {missing}")

    log(f"Controlled rows loaded: {len(data):,}")
    return data


def encode_with_metadata(data: pd.DataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    numeric_frame = data[numeric_features].copy()
    for column in numeric_features:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
        numeric_frame[column] = numeric_frame[column].fillna(numeric_frame[column].median())

    numeric = numeric_frame.astype("float32").to_numpy()
    numeric = (numeric - metadata["numeric_mean"]) / metadata["numeric_std"]

    categorical_arrays = []
    for column in categorical_features:
        values = data[column].fillna("__MISSING__").astype(str)
        categories = metadata["category_maps"][column]
        mapping = {value: index + 1 for index, value in enumerate(categories)}
        categorical_arrays.append(values.map(mapping).fillna(0).astype("int64").to_numpy())

    categorical = np.stack(categorical_arrays, axis=1)
    labels = data["label"].astype("float32").to_numpy()
    return numeric.astype("float32"), categorical.astype("int64"), labels


def predict_probabilities(
    model: nn.Module,
    numeric: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = TabularDataset(numeric, categorical, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    probabilities = []
    y_true = []
    with torch.no_grad():
        for numeric_batch, categorical_batch, label_batch in loader:
            numeric_batch = numeric_batch.to(device)
            categorical_batch = categorical_batch.to(device)
            logits = model(numeric_batch, categorical_batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(label_batch.numpy())

    return np.concatenate(y_true), np.concatenate(probabilities)


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype("int64")
    return {
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def evaluate_arrays(
    model: nn.Module,
    numeric: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    y_true, probabilities = predict_probabilities(model, numeric, categorical, labels, device, batch_size)
    return compute_metrics(y_true, probabilities)


def feature_group(feature_name: str, kind: str) -> str:
    if kind == "numeric" and feature_name in CUSTOMER_NUMERIC_FEATURES:
        return "customer_numeric"
    if kind == "categorical" and feature_name in CUSTOMER_CATEGORICAL_FEATURES:
        return "customer_categorical"
    if kind == "numeric":
        return "article_numeric"
    return "article_categorical"


def permutation_importance(
    model: nn.Module,
    numeric: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
    baseline: dict[str, float],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    for index, feature in enumerate(numeric_features):
        perturbed_numeric = numeric.copy()
        perturbed_numeric[:, index] = rng.permutation(perturbed_numeric[:, index])
        metrics = evaluate_arrays(model, perturbed_numeric, categorical, labels, device, batch_size)
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "feature_group": feature_group(feature, "numeric"),
                "baseline_auc_roc": baseline["auc_roc"],
                "permuted_auc_roc": metrics["auc_roc"],
                "auc_drop": baseline["auc_roc"] - metrics["auc_roc"],
                "baseline_accuracy": baseline["accuracy"],
                "permuted_accuracy": metrics["accuracy"],
                "accuracy_drop": baseline["accuracy"] - metrics["accuracy"],
            }
        )

    for index, feature in enumerate(categorical_features):
        perturbed_categorical = categorical.copy()
        perturbed_categorical[:, index] = rng.permutation(perturbed_categorical[:, index])
        metrics = evaluate_arrays(model, numeric, perturbed_categorical, labels, device, batch_size)
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "feature_group": feature_group(feature, "categorical"),
                "baseline_auc_roc": baseline["auc_roc"],
                "permuted_auc_roc": metrics["auc_roc"],
                "auc_drop": baseline["auc_roc"] - metrics["auc_roc"],
                "baseline_accuracy": baseline["accuracy"],
                "permuted_accuracy": metrics["accuracy"],
                "accuracy_drop": baseline["accuracy"] - metrics["accuracy"],
            }
        )

    feature_importance = pd.DataFrame(rows).sort_values("auc_drop", ascending=False)
    group_importance = (
        feature_importance.groupby("feature_group", as_index=False)
        .agg(
            total_auc_drop=("auc_drop", "sum"),
            mean_auc_drop=("auc_drop", "mean"),
            total_accuracy_drop=("accuracy_drop", "sum"),
            feature_count=("feature", "count"),
        )
        .sort_values("total_auc_drop", ascending=False)
    )
    return feature_importance, group_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute permutation importance for the controlled tabular-only MLP model."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--max-validation-rows", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--group-output", type=Path, default=DEFAULT_GROUP_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    config = checkpoint.get("config", {})
    random_seed = int(config.get("random_seed", args.seed))
    validation_size = float(config.get("validation_size", 0.2))
    max_rows = int(config.get("max_rows", 0))

    data = load_controlled_data(metadata, max_rows=max_rows, random_seed=random_seed)
    numeric, categorical, labels = encode_with_metadata(data, metadata)

    indices = np.arange(len(labels))
    _, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_seed,
        stratify=labels,
    )

    if args.max_validation_rows > 0 and len(val_idx) > args.max_validation_rows:
        rng = np.random.default_rng(args.seed)
        val_idx = rng.choice(val_idx, size=args.max_validation_rows, replace=False)

    val_numeric = numeric[val_idx]
    val_categorical = categorical[val_idx]
    val_labels = labels[val_idx]
    log(f"Validation rows used for importance: {len(val_labels):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Evaluation device: {device}")

    model = TabularMLP(len(metadata["numeric_features"]), metadata["category_sizes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    baseline = evaluate_arrays(model, val_numeric, val_categorical, val_labels, device, args.batch_size)
    log(f"Baseline on sampled validation: auc={baseline['auc_roc']:.4f} | acc={baseline['accuracy']:.4f}")

    feature_importance, group_importance = permutation_importance(
        model,
        val_numeric,
        val_categorical,
        val_labels,
        metadata,
        baseline,
        device,
        args.batch_size,
        seed=args.seed,
    )

    args.feature_output.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(args.feature_output, index=False)
    group_importance.to_csv(args.group_output, index=False)

    log(f"Saved feature importance: {args.feature_output}")
    log(f"Saved group importance: {args.group_output}")
    log("Top features by AUC drop:")
    log(feature_importance.head(10)[["feature", "feature_type", "feature_group", "auc_drop"]].to_string(index=False))


if __name__ == "__main__":
    main()
