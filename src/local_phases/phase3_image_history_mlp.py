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
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "article_image_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "article_image_embedding_ids.csv"

MODEL_OUTPUT_PATH = MODELS_DIR / "image_history_mlp.pt"
RESULTS_PATH = REPORTS_DIR / "phase2_baseline_results.csv"

RANDOM_SEED = 42
BATCH_SIZE = 4096
EPOCHS = 5
LEARNING_RATE = 1e-3
VALIDATION_SIZE = 0.2


class ImageHistoryDataset(Dataset):
    def __init__(
        self,
        article_indices: np.ndarray,
        customer_indices: np.ndarray,
        pair_counts: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.article_indices = torch.tensor(article_indices, dtype=torch.long)
        self.customer_indices = torch.tensor(customer_indices, dtype=torch.long)
        self.pair_counts = torch.tensor(pair_counts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.article_indices[index],
            self.customer_indices[index],
            self.pair_counts[index],
            self.labels[index],
        )


class ImageHistoryMLP(nn.Module):
    def __init__(self, image_dim: int) -> None:
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
        article_embeddings: torch.Tensor,
        customer_profiles: torch.Tensor,
        visual_similarity: torch.Tensor,
        visual_history_count: torch.Tensor,
    ) -> torch.Tensor:
        visual_extra = torch.stack(
            [visual_similarity, torch.log1p(torch.clamp(visual_history_count, min=0.0))],
            dim=1,
        )
        x = torch.cat([article_embeddings, customer_profiles, visual_extra], dim=1)
        return self.network(x).squeeze(1)


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
    return embeddings, article_to_index


def load_dataset(article_to_index: dict[str, int]):
    print("Loading image-history rows...")
    data = pd.read_csv(VISUAL_FEATURES_PATH, dtype={"article_id": str})
    transactions = pd.read_csv(TRANSACTIONS_PATH, dtype={"article_id": str})

    data = data[data["article_id"].isin(article_to_index)].copy()
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
    return data, transactions, customer_to_index


def build_customer_profile_sums(transactions, embeddings, article_to_index, customer_to_index):
    profile_sums = np.zeros((len(customer_to_index), embeddings.shape[1]), dtype="float32")
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


def make_visual_batch(article_indices, customer_indices, pair_counts, article_tensor, profile_sum_tensor, profile_count_tensor):
    article_embeddings = article_tensor[article_indices]
    profile_sums = profile_sum_tensor[customer_indices] - pair_counts.unsqueeze(1) * article_embeddings
    history_counts = profile_count_tensor[customer_indices] - pair_counts
    safe_counts = torch.clamp(history_counts, min=1.0).unsqueeze(1)
    profiles = profile_sums / safe_counts
    profiles = torch.nn.functional.normalize(profiles, p=2, dim=1)
    profiles = torch.where(history_counts.unsqueeze(1) > 0, profiles, torch.zeros_like(profiles))
    visual_similarity = torch.sum(profiles * article_embeddings, dim=1)
    return article_embeddings, profiles, visual_similarity, history_counts


def evaluate(model, loader, device, article_tensor, profile_sum_tensor, profile_count_tensor):
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for article_idx, customer_idx, pair_count, label_batch in tqdm(loader, desc="Validation", leave=False):
            article_idx = article_idx.to(device, non_blocking=True)
            customer_idx = customer_idx.to(device, non_blocking=True)
            pair_count = pair_count.to(device, non_blocking=True)
            article_embeddings, profiles, visual_similarity, history_counts = make_visual_batch(
                article_idx,
                customer_idx,
                pair_count,
                article_tensor,
                profile_sum_tensor,
                profile_count_tensor,
            )
            logits = model(article_embeddings, profiles, visual_similarity, history_counts)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(label_batch.numpy())

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

    embeddings, article_to_index = load_embeddings()
    data, transactions, customer_to_index = load_dataset(article_to_index)
    print(f"Image-history rows: {len(data):,}")
    print("Label distribution:")
    print(data["label"].value_counts().sort_index().to_string())

    profile_sums, profile_counts = build_customer_profile_sums(
        transactions,
        embeddings,
        article_to_index,
        customer_to_index,
    )

    labels = data["label"].astype("float32").to_numpy()
    article_indices = data["article_index"].astype("int64").to_numpy()
    customer_indices = data["customer_index"].astype("int64").to_numpy()
    pair_counts = data["pair_purchase_count"].astype("float32").to_numpy()

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_dataset = ImageHistoryDataset(article_indices[train_idx], customer_indices[train_idx], pair_counts[train_idx], labels[train_idx])
    val_dataset = ImageHistoryDataset(article_indices[val_idx], customer_indices[val_idx], pair_counts[val_idx], labels[val_idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    article_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    profile_sum_tensor = torch.tensor(profile_sums, dtype=torch.float32, device=device)
    profile_count_tensor = torch.tensor(profile_counts, dtype=torch.float32, device=device)

    model = ImageHistoryMLP(image_dim=embeddings.shape[1]).to(device)
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
        for article_idx, customer_idx, pair_count, label_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            article_idx = article_idx.to(device, non_blocking=True)
            customer_idx = customer_idx.to(device, non_blocking=True)
            pair_count = pair_count.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)

            article_embeddings, profiles, visual_similarity, history_counts = make_visual_batch(
                article_idx,
                customer_idx,
                pair_count,
                article_tensor,
                profile_sum_tensor,
                profile_count_tensor,
            )
            optimizer.zero_grad()
            logits = model(article_embeddings, profiles, visual_similarity, history_counts)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate(model, val_loader, device, article_tensor, profile_sum_tensor, profile_count_tensor)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={float(np.mean(losses)):.4f} | "
            f"val_auc={metrics['auc_roc']:.4f} | "
            f"val_acc={metrics['accuracy']:.4f}"
        )
        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "validation_metrics": metrics,
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
            "model": "image_history_mlp",
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


