import csv
import os
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

RESULTS_PATH = REPORTS_DIR / "phase3_controlled_ablation_results.csv"

RANDOM_SEED = int(os.getenv("CONTROLLED_ABLATION_SEED", "42"))
BATCH_SIZE = int(os.getenv("CONTROLLED_ABLATION_BATCH_SIZE", "4096"))
EPOCHS = int(os.getenv("CONTROLLED_ABLATION_EPOCHS", "5"))
LEARNING_RATE = float(os.getenv("CONTROLLED_ABLATION_LR", "1e-3"))
VALIDATION_SIZE = float(os.getenv("CONTROLLED_ABLATION_VAL_SIZE", "0.2"))
MAX_ROWS = int(os.getenv("CONTROLLED_ABLATION_MAX_ROWS", "0"))

TABULAR_NUMERIC_FEATURES = [
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
]

FUSION_NUMERIC_FEATURES = TABULAR_NUMERIC_FEATURES + [
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
    def __init__(self, numeric_values, categorical_values, labels):
        self.numeric_values = torch.tensor(numeric_values, dtype=torch.float32)
        self.categorical_values = torch.tensor(categorical_values, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.numeric_values[index], self.categorical_values[index], self.labels[index]


class ImageDataset(Dataset):
    def __init__(self, article_indices, customer_indices, pair_counts, labels):
        self.article_indices = torch.tensor(article_indices, dtype=torch.long)
        self.customer_indices = torch.tensor(customer_indices, dtype=torch.long)
        self.pair_counts = torch.tensor(pair_counts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.article_indices[index],
            self.customer_indices[index],
            self.pair_counts[index],
            self.labels[index],
        )


class FusionDataset(Dataset):
    def __init__(self, numeric_values, categorical_values, article_indices, customer_indices, pair_counts, labels):
        self.numeric_values = torch.tensor(numeric_values, dtype=torch.float32)
        self.categorical_values = torch.tensor(categorical_values, dtype=torch.long)
        self.article_indices = torch.tensor(article_indices, dtype=torch.long)
        self.customer_indices = torch.tensor(customer_indices, dtype=torch.long)
        self.pair_counts = torch.tensor(pair_counts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.numeric_values[index],
            self.categorical_values[index],
            self.article_indices[index],
            self.customer_indices[index],
            self.pair_counts[index],
            self.labels[index],
        )


class TabularMLP(nn.Module):
    def __init__(self, numeric_dim, category_sizes):
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

    def forward(self, numeric_values, categorical_values):
        embedded = [embedding(categorical_values[:, index]) for index, embedding in enumerate(self.embeddings)]
        x = torch.cat([numeric_values, *embedded], dim=1)
        return self.network(x).squeeze(1)


class ImageHistoryMLP(nn.Module):
    def __init__(self, image_dim):
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

    def forward(self, article_embeddings, customer_profiles, visual_similarity, visual_history_count):
        visual_extra = torch.stack(
            [visual_similarity, torch.log1p(torch.clamp(visual_history_count, min=0.0))],
            dim=1,
        )
        x = torch.cat([article_embeddings, customer_profiles, visual_extra], dim=1)
        return self.network(x).squeeze(1)


class MultimodalFusion(nn.Module):
    def __init__(self, numeric_dim, category_sizes, image_dim):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, min(50, max(4, (size + 1) // 2))) for size in category_sizes]
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

    def forward(self, numeric_values, categorical_values, article_embeddings, customer_profiles, visual_similarity, visual_history_count):
        embedded = [embedding(categorical_values[:, index]) for index, embedding in enumerate(self.embeddings)]
        tabular_input = torch.cat([numeric_values, *embedded], dim=1)
        tabular_repr = self.tabular_branch(tabular_input)
        visual_extra = torch.stack(
            [visual_similarity, torch.log1p(torch.clamp(visual_history_count, min=0.0))],
            dim=1,
        )
        visual_input = torch.cat([article_embeddings, customer_profiles, visual_extra], dim=1)
        visual_repr = self.visual_branch(visual_input)
        return self.fusion_head(torch.cat([tabular_repr, visual_repr], dim=1)).squeeze(1)


def log(message):
    print(message, flush=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(values, eps=1e-8):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def load_embeddings():
    article_ids = pd.read_csv(IDS_PATH, dtype={"article_id": str})["article_id"].tolist()
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    embeddings = l2_normalize(embeddings)
    article_to_index = {article_id: index for index, article_id in enumerate(article_ids)}
    return embeddings, article_to_index


def load_controlled_rows(article_to_index):
    log("Loading common Phase 3 rows and metadata...")
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

    if MAX_ROWS > 0 and len(data) > MAX_ROWS:
        data = data.groupby("label", group_keys=False).sample(
            frac=MAX_ROWS / len(data),
            random_state=RANDOM_SEED,
        )
        data = data.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

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


def prepare_tabular_arrays(data, numeric_features):
    data = data.copy()
    for column in numeric_features:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())

    numeric_values = data[numeric_features].astype("float32").to_numpy()
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
        categorical_arrays.append(categories.codes.astype("int64") + 1)
        category_maps[column] = list(categories.categories)
        category_sizes.append(len(categories.categories) + 1)

    metadata = {
        "numeric_features": numeric_features,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_maps": category_maps,
        "category_sizes": category_sizes,
    }
    return numeric_values, np.stack(categorical_arrays, axis=1), metadata


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


def compute_metrics(labels, probabilities):
    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels)
    y_pred = (y_prob >= 0.5).astype("int64")
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def train_tabular(model_name, data, train_idx, val_idx, labels, device):
    log(f"\n=== Training {model_name} ===")
    numeric_values, categorical_values, metadata = prepare_tabular_arrays(data, TABULAR_NUMERIC_FEATURES)
    train_dataset = TabularDataset(numeric_values[train_idx], categorical_values[train_idx], labels[train_idx])
    val_dataset = TabularDataset(numeric_values[val_idx], categorical_values[val_idx], labels[val_idx])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = TabularMLP(len(TABULAR_NUMERIC_FEATURES), metadata["category_sizes"]).to(device)
    return train_simple_model(model_name, model, train_loader, val_loader, labels[train_idx], device, metadata)


def train_simple_model(model_name, model, train_loader, val_loader, train_labels, device, metadata):
    positive_count = float(train_labels.sum())
    negative_count = float(len(train_labels) - positive_count)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / max(positive_count, 1.0)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_auc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for numeric_batch, categorical_batch, label_batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{EPOCHS}"):
            numeric_batch = numeric_batch.to(device, non_blocking=True)
            categorical_batch = categorical_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(numeric_batch, categorical_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        probabilities = []
        val_labels = []
        with torch.no_grad():
            for numeric_batch, categorical_batch, label_batch in tqdm(val_loader, desc=f"{model_name} validation", leave=False):
                numeric_batch = numeric_batch.to(device, non_blocking=True)
                categorical_batch = categorical_batch.to(device, non_blocking=True)
                logits = model(numeric_batch, categorical_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(label_batch.numpy())
        metrics = compute_metrics(val_labels, probabilities)
        log(f"{model_name} | epoch {epoch}/{EPOCHS} | loss={np.mean(losses):.4f} | val_auc={metrics['auc_roc']:.4f} | val_acc={metrics['accuracy']:.4f}")

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_state = {"model_state_dict": model.state_dict(), "metadata": metadata, "validation_metrics": metrics}

    return best_state


def train_image(model_name, article_indices, customer_indices, pair_counts, labels, train_idx, val_idx, device, article_tensor, profile_sum_tensor, profile_count_tensor, image_dim):
    log(f"\n=== Training {model_name} ===")
    train_dataset = ImageDataset(article_indices[train_idx], customer_indices[train_idx], pair_counts[train_idx], labels[train_idx])
    val_dataset = ImageDataset(article_indices[val_idx], customer_indices[val_idx], pair_counts[val_idx], labels[val_idx])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    model = ImageHistoryMLP(image_dim).to(device)
    return train_visual_model(model_name, model, train_loader, val_loader, labels[train_idx], device, article_tensor, profile_sum_tensor, profile_count_tensor)


def train_visual_model(model_name, model, train_loader, val_loader, train_labels, device, article_tensor, profile_sum_tensor, profile_count_tensor):
    positive_count = float(train_labels.sum())
    negative_count = float(len(train_labels) - positive_count)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / max(positive_count, 1.0)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_auc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for article_idx, customer_idx, pair_count, label_batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{EPOCHS}"):
            article_idx = article_idx.to(device, non_blocking=True)
            customer_idx = customer_idx.to(device, non_blocking=True)
            pair_count = pair_count.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            article_embeddings, profiles, visual_similarity, history_counts = make_visual_batch(
                article_idx, customer_idx, pair_count, article_tensor, profile_sum_tensor, profile_count_tensor
            )
            optimizer.zero_grad()
            logits = model(article_embeddings, profiles, visual_similarity, history_counts)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate_visual_model(model_name, model, val_loader, device, article_tensor, profile_sum_tensor, profile_count_tensor)
        log(f"{model_name} | epoch {epoch}/{EPOCHS} | loss={np.mean(losses):.4f} | val_auc={metrics['auc_roc']:.4f} | val_acc={metrics['accuracy']:.4f}")
        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_state = {"model_state_dict": model.state_dict(), "validation_metrics": metrics}
    return best_state


def evaluate_visual_model(model_name, model, loader, device, article_tensor, profile_sum_tensor, profile_count_tensor):
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for article_idx, customer_idx, pair_count, label_batch in tqdm(loader, desc=f"{model_name} validation", leave=False):
            article_idx = article_idx.to(device, non_blocking=True)
            customer_idx = customer_idx.to(device, non_blocking=True)
            pair_count = pair_count.to(device, non_blocking=True)
            article_embeddings, profiles, visual_similarity, history_counts = make_visual_batch(
                article_idx, customer_idx, pair_count, article_tensor, profile_sum_tensor, profile_count_tensor
            )
            logits = model(article_embeddings, profiles, visual_similarity, history_counts)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(label_batch.numpy())
    return compute_metrics(labels, probabilities)


def train_fusion(model_name, data, train_idx, val_idx, article_indices, customer_indices, pair_counts, labels, device, article_tensor, profile_sum_tensor, profile_count_tensor, image_dim):
    log(f"\n=== Training {model_name} ===")
    numeric_values, categorical_values, metadata = prepare_tabular_arrays(data, FUSION_NUMERIC_FEATURES)
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
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    model = MultimodalFusion(len(FUSION_NUMERIC_FEATURES), metadata["category_sizes"], image_dim).to(device)

    positive_count = float(labels[train_idx].sum())
    negative_count = float(len(train_idx) - positive_count)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative_count / max(positive_count, 1.0)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_auc = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{EPOCHS}"):
            numeric_batch, categorical_batch, article_idx, customer_idx, pair_count, label_batch = batch
            numeric_batch = numeric_batch.to(device, non_blocking=True)
            categorical_batch = categorical_batch.to(device, non_blocking=True)
            article_idx = article_idx.to(device, non_blocking=True)
            customer_idx = customer_idx.to(device, non_blocking=True)
            pair_count = pair_count.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            article_embeddings, profiles, visual_similarity, history_counts = make_visual_batch(
                article_idx, customer_idx, pair_count, article_tensor, profile_sum_tensor, profile_count_tensor
            )
            optimizer.zero_grad()
            logits = model(numeric_batch, categorical_batch, article_embeddings, profiles, visual_similarity, history_counts)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        metrics = evaluate_fusion_model(model_name, model, val_loader, device, article_tensor, profile_sum_tensor, profile_count_tensor)
        log(f"{model_name} | epoch {epoch}/{EPOCHS} | loss={np.mean(losses):.4f} | val_auc={metrics['auc_roc']:.4f} | val_acc={metrics['accuracy']:.4f}")
        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_state = {"model_state_dict": model.state_dict(), "metadata": metadata, "validation_metrics": metrics}

    return best_state


def evaluate_fusion_model(model_name, model, loader, device, article_tensor, profile_sum_tensor, profile_count_tensor):
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{model_name} validation", leave=False):
            numeric_batch, categorical_batch, article_idx, customer_idx, pair_count, label_batch = batch
            numeric_batch = numeric_batch.to(device, non_blocking=True)
            categorical_batch = categorical_batch.to(device, non_blocking=True)
            article_idx = article_idx.to(device, non_blocking=True)
            customer_idx = customer_idx.to(device, non_blocking=True)
            pair_count = pair_count.to(device, non_blocking=True)
            article_embeddings, profiles, visual_similarity, history_counts = make_visual_batch(
                article_idx, customer_idx, pair_count, article_tensor, profile_sum_tensor, profile_count_tensor
            )
            logits = model(numeric_batch, categorical_batch, article_embeddings, profiles, visual_similarity, history_counts)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(label_batch.numpy())
    return compute_metrics(labels, probabilities)


def save_result(model_name, state, row_count, train_count, val_count):
    output_path = MODELS_DIR / f"controlled_{model_name}.pt"
    torch.save(
        {
            **state,
            "config": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "validation_size": VALIDATION_SIZE,
                "random_seed": RANDOM_SEED,
                "max_rows": MAX_ROWS,
            },
        },
        output_path,
    )
    row = {
        "experiment": "phase3_controlled_ablation",
        "model": model_name,
        "rows": row_count,
        "train_rows": train_count,
        "validation_rows": val_count,
        "epochs": EPOCHS,
        "auc_roc": round(state["validation_metrics"]["auc_roc"], 6),
        "accuracy": round(state["validation_metrics"]["accuracy"], 6),
        "artifact": str(output_path),
    }
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    log(f"Saved {model_name}: {output_path}")


def main():
    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    embeddings, article_to_index = load_embeddings()
    data, transactions, customer_to_index = load_controlled_rows(article_to_index)
    labels = data["label"].astype("float32").to_numpy()
    article_indices = data["article_index"].astype("int64").to_numpy()
    customer_indices = data["customer_index"].astype("int64").to_numpy()
    pair_counts = data["pair_purchase_count"].astype("float32").to_numpy()

    log(f"Controlled rows: {len(data):,}")
    log("Label distribution:")
    log(data["label"].value_counts().sort_index().to_string())

    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    log(f"Common split: train={len(train_idx):,} | validation={len(val_idx):,}")

    profile_sums, profile_counts = build_customer_profile_sums(
        transactions, embeddings, article_to_index, customer_to_index
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Training device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    article_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    profile_sum_tensor = torch.tensor(profile_sums, dtype=torch.float32, device=device)
    profile_count_tensor = torch.tensor(profile_counts, dtype=torch.float32, device=device)

    runs = [
        (
            "tabular_only_mlp",
            lambda: train_tabular("tabular_only_mlp", data, train_idx, val_idx, labels, device),
        ),
        (
            "image_history_mlp",
            lambda: train_image(
                "image_history_mlp",
                article_indices,
                customer_indices,
                pair_counts,
                labels,
                train_idx,
                val_idx,
                device,
                article_tensor,
                profile_sum_tensor,
                profile_count_tensor,
                embeddings.shape[1],
            ),
        ),
        (
            "multimodal_late_fusion",
            lambda: train_fusion(
                "multimodal_late_fusion",
                data,
                train_idx,
                val_idx,
                article_indices,
                customer_indices,
                pair_counts,
                labels,
                device,
                article_tensor,
                profile_sum_tensor,
                profile_count_tensor,
                embeddings.shape[1],
            ),
        ),
    ]

    for model_name, runner in runs:
        state = runner()
        save_result(model_name, state, len(data), len(train_idx), len(val_idx))

    log(f"\nControlled ablation complete: {RESULTS_PATH}")


if __name__ == "__main__":
    main()


