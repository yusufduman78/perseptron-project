import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
SANDBOX_DIR = PROJECT_DIR / "data" / "processed" / "sandbox"
EMBEDDINGS_DIR = PROJECT_DIR / "data" / "embeddings" / "local_sandbox"
MODELS_DIR = PROJECT_DIR / "models" / "local_experiments"
REPORTS_DIR = PROJECT_DIR / "reports"

VISUAL_FEATURES_PATH = SANDBOX_DIR / "visual_similarity_features.csv"
TRANSACTIONS_PATH = SANDBOX_DIR / "sandbox_transactions.csv"
CUSTOMERS_PATH = SANDBOX_DIR / "sandbox_customers.csv"
ARTICLES_PATH = SANDBOX_DIR / "sandbox_articles.csv"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "article_image_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "article_image_embedding_ids.csv"

MODEL_OUTPUT_PATH = MODELS_DIR / "multimodal_fusion.pt"
RESULTS_PATH = REPORTS_DIR / "phase2_baseline_results.csv"

RANDOM_SEED = 42
BATCH_SIZE = int(4096)
EPOCHS = int(5)
LEARNING_RATE = float(1e-3)
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


class FusionDataset(Dataset):
    def __init__(
        self,
        numeric_values: np.ndarray,
        categorical_values: np.ndarray,
        article_indices: np.ndarray,
        customer_indices: np.ndarray,
        pair_counts: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.numeric_values = torch.tensor(numeric_values, dtype=torch.float32)
        self.categorical_values = torch.tensor(categorical_values, dtype=torch.long)
        self.article_indices = torch.tensor(article_indices, dtype=torch.long)
        self.customer_indices = torch.tensor(customer_indices, dtype=torch.long)
        self.pair_counts = torch.tensor(pair_counts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.numeric_values[index],
            self.categorical_values[index],
            self.article_indices[index],
            self.customer_indices[index],
            self.pair_counts[index],
            self.labels[index],
        )


class MultimodalFusion(nn.Module):
    def __init__(self, numeric_dim: int, category_sizes: list[int], image_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size, min(50, max(4, (size + 1) // 2)))
                for size in category_sizes
            ]
        )
        categorical_dim = sum(embedding.embedding_dim for embedding in self.embeddings)

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
        numeric_values: torch.Tensor,
        categorical_values: torch.Tensor,
        article_embeddings: torch.Tensor,
        customer_profiles: torch.Tensor,
        visual_similarity: torch.Tensor,
        visual_history_count: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical_values[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        tabular_input = torch.cat([numeric_values, *embedded], dim=1)
        tabular_repr = self.tabular_branch(tabular_input)

        visual_extra = torch.stack(
            [visual_similarity, torch.log1p(torch.clamp(visual_history_count, min=0.0))],
            dim=1,
        )
        visual_input = torch.cat([article_embeddings, customer_profiles, visual_extra], dim=1)
        visual_repr = self.visual_branch(visual_input)

        fused = torch.cat([tabular_repr, visual_repr], dim=1)
        return self.fusion_head(fused).squeeze(1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def load_embeddings():
    article_ids = pd.read_csv(IDS_PATH, dtype={"article_id": str})["article_id"].tolist()
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    embeddings = l2_normalize(embeddings)
    article_to_index = {article_id: index for index, article_id in enumerate(article_ids)}
    return article_ids, embeddings, article_to_index


def load_dataset(article_to_index: dict[str, int]):
    print("Loading fusion rows and metadata...")
    data = pd.read_csv(VISUAL_FEATURES_PATH, dtype={"article_id": str})
    customers = pd.read_csv(CUSTOMERS_PATH)
    articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
    transactions = pd.read_csv(TRANSACTIONS_PATH, dtype={"article_id": str})

    data = data[data["article_id"].isin(article_to_index)].copy()
    data = data.merge(customers, on="customer_id", how="left")
    data = data.merge(articles, on="article_id", how="left")

    pair_counts = (
        transactions.groupby(["customer_id", "article_id"])
        .size()
        .reset_index(name="pair_purchase_count")
    )
    data = data.merge(pair_counts, on=["customer_id", "article_id"], how="left")
    data["pair_purchase_count"] = data["pair_purchase_count"].fillna(0).astype("float32")

    customer_ids = sorted(transactions["customer_id"].unique())
    customer_to_index = {customer_id: index for index, customer_id in enumerate(customer_ids)}
    data = data[data["customer_id"].isin(customer_to_index)].copy()
    data["article_index"] = data["article_id"].map(article_to_index).astype("int64")
    data["customer_index"] = data["customer_id"].map(customer_to_index).astype("int64")

    return data, transactions, customer_ids, customer_to_index


def build_customer_profile_sums(
    transactions: pd.DataFrame,
    embeddings: np.ndarray,
    article_to_index: dict[str, int],
    customer_to_index: dict[str, int],
):
    image_dim = embeddings.shape[1]
    profile_sums = np.zeros((len(customer_to_index), image_dim), dtype="float32")
    profile_counts = np.zeros(len(customer_to_index), dtype="float32")

    usable = transactions[
        transactions["article_id"].isin(article_to_index)
        & transactions["customer_id"].isin(customer_to_index)
    ][["customer_id", "article_id"]]

    for row in tqdm(usable.itertuples(index=False), total=len(usable), desc="Customer visual sums"):
        customer_index = customer_to_index[row.customer_id]
        article_index = article_to_index[row.article_id]
        profile_sums[customer_index] += embeddings[article_index]
        profile_counts[customer_index] += 1.0

    return profile_sums, profile_counts


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
    article_indices = data["article_index"].astype("int64").to_numpy()
    customer_indices = data["customer_index"].astype("int64").to_numpy()
    pair_counts = data["pair_purchase_count"].astype("float32").to_numpy()

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_maps": category_maps,
        "category_sizes": category_sizes,
    }
    return numeric_values, categorical_values, article_indices, customer_indices, pair_counts, labels, metadata


def make_visual_batch(
    article_indices: torch.Tensor,
    customer_indices: torch.Tensor,
    pair_counts: torch.Tensor,
    article_embedding_tensor: torch.Tensor,
    profile_sum_tensor: torch.Tensor,
    profile_count_tensor: torch.Tensor,
):
    article_embeddings = article_embedding_tensor[article_indices]
    profile_sums = profile_sum_tensor[customer_indices] - (pair_counts.unsqueeze(1) * article_embeddings)
    history_counts = profile_count_tensor[customer_indices] - pair_counts
    safe_counts = torch.clamp(history_counts, min=1.0).unsqueeze(1)
    profiles = profile_sums / safe_counts
    profiles = torch.nn.functional.normalize(profiles, p=2, dim=1)
    profiles = torch.where(history_counts.unsqueeze(1) > 0, profiles, torch.zeros_like(profiles))
    visual_similarity = torch.sum(profiles * article_embeddings, dim=1)
    return article_embeddings, profiles, visual_similarity, history_counts


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    article_embedding_tensor: torch.Tensor,
    profile_sum_tensor: torch.Tensor,
    profile_count_tensor: torch.Tensor,
    criterion=None,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    losses = []
    probabilities = []
    labels_out = []

    desc = "Train" if training else "Validation"
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            numeric_batch, categorical_batch, article_idx, customer_idx, pair_count, label_batch = batch
            numeric_batch = numeric_batch.to(device, non_blocking=True)
            categorical_batch = categorical_batch.to(device, non_blocking=True)
            article_idx = article_idx.to(device, non_blocking=True)
            customer_idx = customer_idx.to(device, non_blocking=True)
            pair_count = pair_count.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)

            article_embeddings, profiles, visual_similarity, history_counts = make_visual_batch(
                article_idx,
                customer_idx,
                pair_count,
                article_embedding_tensor,
                profile_sum_tensor,
                profile_count_tensor,
            )

            if training:
                optimizer.zero_grad()
            logits = model(
                numeric_batch,
                categorical_batch,
                article_embeddings,
                profiles,
                visual_similarity,
                history_counts,
            )

            if training:
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            else:
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                labels_out.append(label_batch.cpu().numpy())

    if training:
        return {"loss": float(np.mean(losses))}

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels_out)
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

    article_ids, embeddings, article_to_index = load_embeddings()
    data, transactions, customer_ids, customer_to_index = load_dataset(article_to_index)
    print(f"Fusion rows: {len(data):,}")
    print("Label distribution:")
    print(data["label"].value_counts().sort_index().to_string())

    profile_sums, profile_counts = build_customer_profile_sums(
        transactions,
        embeddings,
        article_to_index,
        customer_to_index,
    )

    numeric_values, categorical_values, article_indices, customer_indices, pair_counts, labels, metadata = prepare_features(data)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_dataset = FusionDataset(
        numeric_values[train_idx],
        categorical_values[train_idx],
        article_indices[train_idx],
        customer_indices[train_idx],
        pair_counts[train_idx],
        labels[train_idx],
    )
    val_dataset = FusionDataset(
        numeric_values[val_idx],
        categorical_values[val_idx],
        article_indices[val_idx],
        customer_indices[val_idx],
        pair_counts[val_idx],
        labels[val_idx],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    article_embedding_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    profile_sum_tensor = torch.tensor(profile_sums, dtype=torch.float32, device=device)
    profile_count_tensor = torch.tensor(profile_counts, dtype=torch.float32, device=device)

    model = MultimodalFusion(
        numeric_dim=len(NUMERIC_FEATURES),
        category_sizes=metadata["category_sizes"],
        image_dim=embeddings.shape[1],
    ).to(device)

    positive_count = float(labels[train_idx].sum())
    negative_count = float(len(train_idx) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_auc = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            article_embedding_tensor,
            profile_sum_tensor,
            profile_count_tensor,
            criterion=criterion,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            article_embedding_tensor,
            profile_sum_tensor,
            profile_count_tensor,
        )
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"val_auc={val_metrics['auc_roc']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "metadata": metadata,
                "validation_metrics": val_metrics,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "validation_size": VALIDATION_SIZE,
                    "random_seed": RANDOM_SEED,
                    "image_dim": embeddings.shape[1],
                },
            }

    torch.save(best_state, MODEL_OUTPUT_PATH)
    print(f"Saved best model: {MODEL_OUTPUT_PATH}")

    append_results(
        {
            "phase": "phase3",
            "model": "multimodal_late_fusion",
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


