import csv
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
MODELS_DIR = PROJECT_DIR / "models" / "local_experiments"
REPORTS_DIR = PROJECT_DIR / "reports"

VISUAL_FEATURES_PATH = SANDBOX_DIR / "visual_similarity_features.csv"
CUSTOMERS_PATH = SANDBOX_DIR / "sandbox_customers.csv"
ARTICLES_PATH = SANDBOX_DIR / "sandbox_articles.csv"

MODEL_OUTPUT_PATH = MODELS_DIR / "visual_tabular_mlp.pt"
RESULTS_PATH = REPORTS_DIR / "phase2_baseline_results.csv"

RANDOM_SEED = 42
BATCH_SIZE = 4096
EPOCHS = 5
LEARNING_RATE = 1e-3
VALIDATION_SIZE = 0.2

NUMERIC_FEATURES = [
    "FN",
    "Active",
    "age",
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
    "visual_similarity",
    "visual_history_count",
]

CATEGORICAL_FEATURES = [
    "club_member_status",
    "fashion_news_frequency",
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


class TabularDataset(Dataset):
    def __init__(self, numeric_values: np.ndarray, categorical_values: np.ndarray, labels: np.ndarray) -> None:
        self.numeric_values = torch.tensor(numeric_values, dtype=torch.float32)
        self.categorical_values = torch.tensor(categorical_values, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.numeric_values[index], self.categorical_values[index], self.labels[index]


class VisualTabularMLP(nn.Module):
    def __init__(self, numeric_dim: int, category_sizes: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size, min(50, max(4, (size + 1) // 2)))
                for size in category_sizes
            ]
        )
        embedding_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
        input_dim = numeric_dim + embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, numeric_values: torch.Tensor, categorical_values: torch.Tensor) -> torch.Tensor:
        embedded = [
            embedding(categorical_values[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        x = torch.cat([numeric_values, *embedded], dim=1)
        return self.network(x).squeeze(1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset() -> pd.DataFrame:
    print("Loading visual features and metadata...")
    visual_features = pd.read_csv(VISUAL_FEATURES_PATH, dtype={"article_id": str})
    customers = pd.read_csv(CUSTOMERS_PATH)
    articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})

    data = visual_features.merge(customers, on="customer_id", how="left")
    data = data.merge(articles, on="article_id", how="left")
    return data


def prepare_features(data: pd.DataFrame):
    data = data.copy()

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())

    numeric_values = data[NUMERIC_FEATURES].astype("float32").to_numpy()
    numeric_mean = numeric_values.mean(axis=0)
    numeric_std = numeric_values.std(axis=0)
    numeric_std[numeric_std == 0] = 1.0
    numeric_values = (numeric_values - numeric_mean) / numeric_std

    categorical_arrays = []
    category_maps = {}
    category_sizes = []
    for column in CATEGORICAL_FEATURES:
        values = data[column].fillna("__MISSING__").astype(str)
        categories = pd.Categorical(values)
        codes = categories.codes.astype("int64") + 1
        categorical_arrays.append(codes)
        category_maps[column] = list(categories.categories)
        category_sizes.append(len(categories.categories) + 1)

    categorical_values = np.stack(categorical_arrays, axis=1)
    labels = data["label"].astype("float32").to_numpy()

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_maps": category_maps,
        "category_sizes": category_sizes,
    }
    return numeric_values, categorical_values, labels, metadata


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for numeric_values, categorical_values, batch_labels in loader:
            numeric_values = numeric_values.to(device)
            categorical_values = categorical_values.to(device)
            logits = model(numeric_values, categorical_values)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.numpy())

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels)
    y_pred = (y_prob >= 0.5).astype("int64")
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def append_results(row: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    data = load_dataset()
    print(f"Merged rows: {len(data):,}")
    print("Label distribution:")
    print(data["label"].value_counts().sort_index().to_string())

    numeric_values, categorical_values, labels, metadata = prepare_features(data)

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_dataset = TabularDataset(numeric_values[train_idx], categorical_values[train_idx], labels[train_idx])
    val_dataset = TabularDataset(numeric_values[val_idx], categorical_values[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = VisualTabularMLP(
        numeric_dim=len(NUMERIC_FEATURES),
        category_sizes=metadata["category_sizes"],
    ).to(device)

    positive_count = float(labels[train_idx].sum())
    negative_count = float(len(train_idx) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_auc = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for numeric_batch, categorical_batch, label_batch in train_loader:
            numeric_batch = numeric_batch.to(device)
            categorical_batch = categorical_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(numeric_batch, categorical_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_loader, device)
        mean_loss = float(np.mean(losses))
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={mean_loss:.4f} | "
            f"val_auc={metrics['auc_roc']:.4f} | "
            f"val_acc={metrics['accuracy']:.4f}"
        )

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "metadata": metadata,
                "validation_metrics": metrics,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "validation_size": VALIDATION_SIZE,
                    "random_seed": RANDOM_SEED,
                },
            }

    torch.save(best_state, MODEL_OUTPUT_PATH)
    print(f"Saved best model: {MODEL_OUTPUT_PATH}")

    append_results(
        {
            "phase": "phase3",
            "model": "visual_similarity_tabular_mlp",
            "rows": len(data),
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "auc_roc": round(best_state["validation_metrics"]["auc_roc"], 6),
            "accuracy": round(best_state["validation_metrics"]["accuracy"], 6),
            "artifact": str(MODEL_OUTPUT_PATH),
        }
    )
    print(f"Updated report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()


