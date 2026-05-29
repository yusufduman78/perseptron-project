"""
Kaggle full-scale pipeline for the H&M multimodal recommendation project.

Run strategy:
1. Keep FAST_RUN=True for the first Kaggle smoke test.
2. If the notebook finishes, switch FAST_RUN=False for the full experiment.
3. The code writes model checkpoints and submission.csv into /kaggle/working.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


# =============================================================================
# 1. CONFIG
# =============================================================================

FAST_RUN = True
RANDOM_SEED = 42

WORK_DIR = Path("/kaggle/working")


def log(message: str) -> None:
    print(message, flush=True)


def get_device(stage: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        log(f"[{stage}] CUDA available: True | GPU: {name} | allocated={allocated:.2f}GB | reserved={reserved:.2f}GB")
    else:
        log(f"[{stage}] CUDA available: False | running on CPU")
    return device


def find_hm_input_dir() -> Path:
    search_roots = [Path("/kaggle/input"), Path("/kaggle/input/competitions")]
    candidates = []
    for root in tqdm(search_roots, desc="Searching Kaggle input roots"):
        if not root.exists():
            continue
        for path in tqdm(list(root.iterdir()), desc=f"Scanning {root}", leave=False):
            if path.is_dir() and (path / "transactions_train.csv").exists():
                candidates.append(path)
            if path.is_dir():
                for child in tqdm(list(path.iterdir()), desc=f"Scanning {path.name}", leave=False):
                    if child.is_dir() and (child / "transactions_train.csv").exists():
                        candidates.append(child)
    if not candidates:
        available = "\n".join(str(path) for path in Path("/kaggle/input").iterdir())
        raise FileNotFoundError(
            "Could not find H&M input directory with transactions_train.csv. "
            f"Available /kaggle/input entries:\n{available}"
        )
    return candidates[0]


INPUT_DIR = find_hm_input_dir()
log(f"Using input directory: {INPUT_DIR}")

TRANSACTIONS_PATH = INPUT_DIR / "transactions_train.csv"
CUSTOMERS_PATH = INPUT_DIR / "customers.csv"
ARTICLES_PATH = INPUT_DIR / "articles.csv"
IMAGES_DIR = INPUT_DIR / "images"
SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"

EMBEDDINGS_PATH = WORK_DIR / "article_image_embeddings_popular.npy"
EMBEDDING_IDS_PATH = WORK_DIR / "article_image_embedding_ids_popular.csv"
MODEL_PATH = WORK_DIR / "multimodal_fusion.pt"
SUBMISSION_PATH = WORK_DIR / "submission.csv"

EMBEDDING_CACHE_CANDIDATES = [
    (
        WORK_DIR / "article_image_embeddings_full.npy",
        WORK_DIR / "article_image_embedding_ids_full.csv",
    ),
    (
        WORK_DIR / "article_image_embeddings_popular.npy",
        WORK_DIR / "article_image_embedding_ids_popular.csv",
    ),
    (
        WORK_DIR / "article_image_embeddings.npy",
        WORK_DIR / "article_image_embedding_ids.csv",
    ),
]

IMAGE_SIZE = 160
EMBED_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 4096
NUM_WORKERS = 0
EPOCHS = 2 if FAST_RUN else 5
LEARNING_RATE = 1e-3
VALIDATION_SIZE = 0.15

MAX_CUSTOMERS = 25000 if FAST_RUN else None
MAX_ARTICLES_FOR_EMBEDDING = 30000 if FAST_RUN else None
NEGATIVES_PER_POSITIVE = 2
POPULAR_CANDIDATES = 80
LAST_7D_CANDIDATES = 80
LAST_30D_CANDIDATES = 80
GLOBAL_CANDIDATES = 80
SIMILAR_CANDIDATES = 20
RECENT_PURCHASE_CANDIDATES = 24
SIMILAR_PRODUCT_TYPE_CANDIDATES = 16
SIMILAR_GARMENT_GROUP_CANDIDATES = 16
CUSTOMER_CANDIDATE_LIMIT = 180
MAX_TRAIN_ROWS = 500000 if FAST_RUN else 4000000
INFERENCE_CUSTOMER_BATCH_SIZE = 1024 if FAST_RUN else 2048
INFERENCE_SCORE_BATCH_SIZE = 8192

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


# =============================================================================
# 2. UTILS
# =============================================================================


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def article_image_path(article_id: str) -> Path:
    padded = str(article_id).zfill(10)
    return IMAGES_DIR / padded[:3] / f"{padded}.jpg"


def reduce_transactions_memory(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
    transactions["price"] = transactions["price"].astype("float32")
    transactions["sales_channel_id"] = transactions["sales_channel_id"].astype("int8")
    return transactions


# =============================================================================
# 3. DATA LOADING
# =============================================================================


def load_raw_data():
    log("Loading raw H&M CSV files...")
    log(f"Reading transactions: {TRANSACTIONS_PATH}")
    transactions = pd.read_csv(TRANSACTIONS_PATH, dtype={"article_id": str})
    log(f"Transactions loaded: {len(transactions):,}")
    log(f"Reading customers: {CUSTOMERS_PATH}")
    customers = pd.read_csv(CUSTOMERS_PATH)
    log(f"Customers loaded: {len(customers):,}")
    log(f"Reading articles: {ARTICLES_PATH}")
    articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
    log(f"Articles loaded: {len(articles):,}")
    log("Optimizing dtypes...")
    transactions = reduce_transactions_memory(transactions)
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)

    if FAST_RUN and MAX_CUSTOMERS:
        log(f"FAST_RUN enabled: keeping top {MAX_CUSTOMERS:,} active customers.")
        active_customers = transactions["customer_id"].value_counts()
        sampled_customers = active_customers.head(MAX_CUSTOMERS).index
        transactions = transactions[transactions["customer_id"].isin(sampled_customers)].copy()
        customers = customers[customers["customer_id"].isin(sampled_customers)].copy()
        used_articles = transactions["article_id"].unique()
        articles = articles[articles["article_id"].isin(used_articles)].copy()

    log(f"Transactions: {len(transactions):,}")
    log(f"Customers: {len(customers):,}")
    log(f"Articles: {len(articles):,}")
    return transactions, customers, articles


# =============================================================================
# 4. IMAGE EMBEDDINGS
# =============================================================================


class ArticleImageDataset(Dataset):
    def __init__(self, article_ids: list[str], transform) -> None:
        self.article_ids = article_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.article_ids)

    def __getitem__(self, index: int):
        article_id = self.article_ids[index]
        path = article_image_path(article_id)
        image = Image.open(path).convert("RGB")
        return article_id, self.transform(image)


class EfficientNetEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def collate_image_batch(batch):
    article_ids, images = zip(*batch)
    return list(article_ids), torch.stack(images)


def load_cached_image_embeddings():
    for embeddings_path, ids_path in EMBEDDING_CACHE_CANDIDATES:
        if embeddings_path.exists() and ids_path.exists():
            log(f"Loading cached image embeddings: {embeddings_path}")
            ids = pd.read_csv(ids_path, dtype={"article_id": str})["article_id"].astype(str).str.zfill(10).tolist()
            embeddings = np.load(embeddings_path).astype("float32")
            if len(ids) != len(embeddings):
                raise ValueError(
                    f"Embedding cache mismatch: {embeddings_path} has {len(embeddings):,} rows "
                    f"but {ids_path} has {len(ids):,} ids."
                )
            embeddings = l2_normalize(embeddings)
            log(f"Loaded cached embeddings: {embeddings.shape}")
            log(f"Loaded cached embedding ids: {len(ids):,}")
            return ids, embeddings, {article_id: i for i, article_id in enumerate(ids)}
    return None


def build_or_load_image_embeddings(articles: pd.DataFrame, transactions: pd.DataFrame | None = None):
    cached = load_cached_image_embeddings()
    if cached is not None:
        return cached

    log("Checking which article images exist...")
    article_set = set(articles["article_id"].drop_duplicates().tolist())
    if transactions is not None:
        log("Ordering article image scan by transaction popularity...")
        popular_order = transactions["article_id"].value_counts().index.tolist()
        article_candidates = [article_id for article_id in popular_order if article_id in article_set]
        candidate_set = set(article_candidates)
        remaining = [article_id for article_id in article_set if article_id not in candidate_set]
        article_candidates.extend(sorted(remaining))
    else:
        article_candidates = sorted(article_set)

    available_ids = []
    for article_id in tqdm(article_candidates, desc="Image existence scan"):
        if article_image_path(article_id).exists():
            available_ids.append(article_id)
            if FAST_RUN and MAX_ARTICLES_FOR_EMBEDDING and len(available_ids) >= MAX_ARTICLES_FOR_EMBEDDING:
                break

    if FAST_RUN and MAX_ARTICLES_FOR_EMBEDDING:
        log(f"FAST_RUN enabled: limiting embeddings to {MAX_ARTICLES_FOR_EMBEDDING:,} articles.")
        available_ids = available_ids[:MAX_ARTICLES_FOR_EMBEDDING]

    log(f"Extracting image embeddings for {len(available_ids):,} articles...")
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ArticleImageDataset(available_ids, transform)
    device = get_device("image embedding extraction")
    loader = DataLoader(
        dataset,
        batch_size=EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        collate_fn=collate_image_batch,
    )

    model = EfficientNetEmbeddingModel().to(device)
    model.eval()

    all_ids = []
    all_embeddings = []
    with torch.no_grad():
        for batch_ids, images in tqdm(loader, desc="Image embeddings"):
            images = images.to(device, non_blocking=True)
            embeddings = model(images).cpu().numpy().astype("float32")
            all_ids.extend(batch_ids)
            all_embeddings.append(embeddings)

    matrix = l2_normalize(np.concatenate(all_embeddings, axis=0))
    log("Saving image embedding cache...")
    np.save(EMBEDDINGS_PATH, matrix)
    pd.DataFrame({"article_id": all_ids}).to_csv(EMBEDDING_IDS_PATH, index=False)
    log(f"Saved embeddings: {matrix.shape}")
    return all_ids, matrix, {article_id: i for i, article_id in enumerate(all_ids)}


# =============================================================================
# 5. TRAINING PAIRS AND VISUAL PROFILES
# =============================================================================


def build_customer_visual_sums(transactions, embeddings, article_to_index):
    log("Filtering transactions to embedded articles...")
    transactions = transactions[transactions["article_id"].isin(article_to_index)].copy()
    log("Building customer index for visual profiles...")
    customer_ids = sorted(transactions["customer_id"].unique())
    customer_to_index = {customer_id: i for i, customer_id in enumerate(customer_ids)}
    log(f"Customers with visual history: {len(customer_ids):,}")
    profile_sums = np.zeros((len(customer_ids), embeddings.shape[1]), dtype="float32")
    profile_counts = np.zeros(len(customer_ids), dtype="float32")

    for row in tqdm(transactions[["customer_id", "article_id"]].itertuples(index=False), total=len(transactions), desc="Customer profiles"):
        customer_index = customer_to_index[row.customer_id]
        article_index = article_to_index[row.article_id]
        profile_sums[customer_index] += embeddings[article_index]
        profile_counts[customer_index] += 1

    return customer_ids, customer_to_index, profile_sums, profile_counts


def build_training_pairs(transactions, article_to_index):
    log("Building positive and negative training pairs...")
    log("Filtering positives to embedded articles...")
    transactions = transactions[transactions["article_id"].isin(article_to_index)].copy()
    log("Dropping duplicate positive customer-article pairs...")
    positives = transactions[["customer_id", "article_id"]].drop_duplicates().copy()
    positives["label"] = 1
    log(f"Positive pairs: {len(positives):,}")

    article_pool = np.array(list(article_to_index.keys()))
    rng = np.random.default_rng(RANDOM_SEED)
    log(f"Sampling negatives: {NEGATIVES_PER_POSITIVE} per positive.")
    neg_customers = np.repeat(positives["customer_id"].values, NEGATIVES_PER_POSITIVE)
    neg_articles = rng.choice(article_pool, size=len(neg_customers), replace=True)
    negatives = pd.DataFrame({"customer_id": neg_customers, "article_id": neg_articles, "label": 0})
    log(f"Negative pairs: {len(negatives):,}")

    log("Combining and shuffling training pairs...")
    data = pd.concat([positives, negatives], ignore_index=True)
    data = data.drop_duplicates(["customer_id", "article_id", "label"])
    if MAX_TRAIN_ROWS and len(data) > MAX_TRAIN_ROWS:
        log(f"Limiting training rows to {MAX_TRAIN_ROWS:,}.")
        data = data.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)
    data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    log(f"Training pairs: {len(data):,}")
    return data


def attach_metadata_and_indices(data, customers, articles, transactions, article_to_index, customer_to_index):
    log("Counting historical purchases per customer-article pair...")
    pair_counts = (
        transactions.groupby(["customer_id", "article_id"])
        .size()
        .reset_index(name="pair_purchase_count")
    )
    log("Merging customer metadata...")
    data = data.merge(customers, on="customer_id", how="left")
    log("Merging article metadata...")
    data = data.merge(articles, on="article_id", how="left")
    log("Merging pair purchase counts...")
    data = data.merge(pair_counts, on=["customer_id", "article_id"], how="left")
    log("Mapping article/customer indices...")
    data["pair_purchase_count"] = data["pair_purchase_count"].fillna(0).astype("float32")
    data["article_index"] = data["article_id"].map(article_to_index).astype("int64")
    data["customer_index"] = data["customer_id"].map(customer_to_index).astype("int64")
    log(f"Training rows after metadata merge: {len(data):,}")
    return data


def add_visual_scalar_features(data, embeddings, profile_sums, profile_counts, verbose=True):
    if verbose:
        log("Computing scalar visual features: candidate/profile cosine similarity...")
    article_embeddings = embeddings[data["article_index"].values]
    sums = profile_sums[data["customer_index"].values] - (
        data["pair_purchase_count"].values[:, None] * article_embeddings
    )
    counts = profile_counts[data["customer_index"].values] - data["pair_purchase_count"].values
    safe_counts = np.maximum(counts, 1.0)[:, None]
    profiles = sums / safe_counts
    profiles = l2_normalize(profiles)
    profiles[counts <= 0] = 0
    data["visual_similarity"] = np.sum(profiles * article_embeddings, axis=1).astype("float32")
    data["visual_history_count"] = counts.astype("float32")
    if verbose:
        log("Visual scalar features ready.")
    return data


# =============================================================================
# 6. MODEL
# =============================================================================


class FusionDataset(Dataset):
    def __init__(self, numeric, categorical, article_idx, customer_idx, pair_counts, labels):
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.categorical = torch.tensor(categorical, dtype=torch.long)
        self.article_idx = torch.tensor(article_idx, dtype=torch.long)
        self.customer_idx = torch.tensor(customer_idx, dtype=torch.long)
        self.pair_counts = torch.tensor(pair_counts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.numeric[index],
            self.categorical[index],
            self.article_idx[index],
            self.customer_idx[index],
            self.pair_counts[index],
            self.labels[index],
        )


class MultimodalFusion(nn.Module):
    def __init__(self, numeric_dim, category_sizes, image_dim):
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

    def forward(self, numeric, categorical, article_emb, profile_emb, visual_similarity, visual_history_count):
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        tabular_repr = self.tabular_branch(torch.cat([numeric, *embedded], dim=1))
        visual_extra = torch.stack(
            [visual_similarity, torch.log1p(torch.clamp(visual_history_count, min=0))],
            dim=1,
        )
        visual_repr = self.visual_branch(torch.cat([article_emb, profile_emb, visual_extra], dim=1))
        return self.fusion_head(torch.cat([tabular_repr, visual_repr], dim=1)).squeeze(1)


def prepare_tabular_arrays(data):
    log("Preparing tabular numeric features...")
    data = data.copy()
    for column in tqdm(NUMERIC_FEATURES, desc="Numeric features"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].fillna(data[column].median())
    numeric = data[NUMERIC_FEATURES].astype("float32").to_numpy()
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0)
    std[std == 0] = 1
    numeric = (numeric - mean) / std

    categorical_arrays = []
    category_sizes = []
    category_maps = {}
    log("Encoding categorical features...")
    for column in tqdm(CATEGORICAL_FEATURES, desc="Categorical features"):
        categories = pd.Categorical(data[column].fillna("__MISSING__").astype(str))
        categorical_arrays.append(categories.codes.astype("int64") + 1)
        category_sizes.append(len(categories.categories) + 1)
        category_maps[column] = list(categories.categories)

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "numeric_mean": mean,
        "numeric_std": std,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_sizes": category_sizes,
        "category_maps": category_maps,
    }
    return numeric, np.stack(categorical_arrays, axis=1), metadata


def prepare_inference_tabular_arrays(data, metadata):
    data = data.copy()
    numeric = data[metadata["numeric_features"]].fillna(0).astype("float32").to_numpy()
    numeric = (numeric - metadata["numeric_mean"]) / metadata["numeric_std"]

    categorical_arrays = []
    for column in metadata["categorical_features"]:
        mapping = {value: index + 1 for index, value in enumerate(metadata["category_maps"][column])}
        codes = data[column].fillna("__MISSING__").astype(str).map(mapping).fillna(0).astype("int64")
        categorical_arrays.append(codes.to_numpy())

    return numeric, np.stack(categorical_arrays, axis=1)


def make_visual_batch(article_idx, customer_idx, pair_counts, article_tensor, profile_sum_tensor, profile_count_tensor):
    article_emb = article_tensor[article_idx]
    sums = profile_sum_tensor[customer_idx] - pair_counts.unsqueeze(1) * article_emb
    counts = profile_count_tensor[customer_idx] - pair_counts
    profiles = sums / torch.clamp(counts, min=1).unsqueeze(1)
    profiles = torch.nn.functional.normalize(profiles, p=2, dim=1)
    profiles = torch.where(counts.unsqueeze(1) > 0, profiles, torch.zeros_like(profiles))
    visual_similarity = torch.sum(profiles * article_emb, dim=1)
    return article_emb, profiles, visual_similarity, counts


def train_model(data, embeddings, profile_sums, profile_counts):
    log("Preparing arrays for fusion training...")
    numeric, categorical, metadata = prepare_tabular_arrays(data)
    labels = data["label"].astype("float32").to_numpy()
    article_idx = data["article_index"].astype("int64").to_numpy()
    customer_idx = data["customer_index"].astype("int64").to_numpy()
    pair_counts = data["pair_purchase_count"].astype("float32").to_numpy()

    log("Creating train/validation split...")
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    train_ds = FusionDataset(numeric[train_idx], categorical[train_idx], article_idx[train_idx], customer_idx[train_idx], pair_counts[train_idx], labels[train_idx])
    val_ds = FusionDataset(numeric[val_idx], categorical[val_idx], article_idx[val_idx], customer_idx[val_idx], pair_counts[val_idx], labels[val_idx])
    log(f"Train rows: {len(train_ds):,} | Validation rows: {len(val_ds):,}")

    device = get_device("fusion training")
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")

    log("Moving embedding/profile tensors to device...")
    article_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    profile_sum_tensor = torch.tensor(profile_sums, dtype=torch.float32, device=device)
    profile_count_tensor = torch.tensor(profile_counts, dtype=torch.float32, device=device)
    log("Initializing multimodal fusion model...")
    model = MultimodalFusion(len(NUMERIC_FEATURES), metadata["category_sizes"], embeddings.shape[1]).to(device)

    pos = float(labels[train_idx].sum())
    neg = float(len(train_idx) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_auc = -1
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True):
            numeric_b, cat_b, article_b, customer_b, pair_b, labels_b = [x.to(device, non_blocking=True) for x in batch]
            article_emb, profiles, sim, hist = make_visual_batch(article_b, customer_b, pair_b, article_tensor, profile_sum_tensor, profile_count_tensor)
            optimizer.zero_grad()
            logits = model(numeric_b, cat_b, article_emb, profiles, sim, hist)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        probs = []
        ys = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=True):
                numeric_b, cat_b, article_b, customer_b, pair_b, labels_b = [x.to(device, non_blocking=True) for x in batch]
                article_emb, profiles, sim, hist = make_visual_batch(article_b, customer_b, pair_b, article_tensor, profile_sum_tensor, profile_count_tensor)
                logits = model(numeric_b, cat_b, article_emb, profiles, sim, hist)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(labels_b.cpu().numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(probs))
        log(f"Epoch {epoch}/{EPOCHS} | loss={np.mean(losses):.4f} | val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {"model_state_dict": model.state_dict(), "metadata": metadata, "auc_roc": auc}

    torch.save(best_state, MODEL_PATH)
    model.load_state_dict(best_state["model_state_dict"])
    log(f"Saved model: {MODEL_PATH} | best_auc={best_auc:.4f}")
    return model, metadata, best_auc


# =============================================================================
# 7. CANDIDATES AND SUBMISSION
# =============================================================================


def top_articles_for_window(transactions, min_date, limit, article_to_index=None):
    if min_date is not None:
        frame = transactions[transactions["t_dat"] >= min_date]
    else:
        frame = transactions

    counts = frame["article_id"].value_counts()
    if article_to_index is not None:
        embedded_articles = set(article_to_index)
        counts = counts[counts.index.isin(embedded_articles)]
    return counts.head(limit).index.tolist()


def build_popular_candidates(transactions, article_to_index=None):
    log("Building time-aware popularity candidates...")
    transactions = transactions.copy()
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    max_date = transactions["t_dat"].max()
    last_7d_start = max_date - pd.Timedelta(days=7)
    last_30d_start = max_date - pd.Timedelta(days=30)

    last_7d = top_articles_for_window(transactions, last_7d_start, LAST_7D_CANDIDATES, article_to_index)
    last_30d = top_articles_for_window(transactions, last_30d_start, LAST_30D_CANDIDATES, article_to_index)
    global_popular = top_articles_for_window(transactions, None, GLOBAL_CANDIDATES, article_to_index)

    candidates = list(dict.fromkeys(last_7d + last_30d + global_popular))
    candidates = candidates[:POPULAR_CANDIDATES]

    log(f"Last-7d candidates: {len(last_7d):,}")
    log(f"Last-30d candidates: {len(last_30d):,}")
    log(f"Global candidates: {len(global_popular):,}")
    log(f"Blended popularity candidates ready: {len(candidates):,}")
    return candidates


def build_customer_recent_candidates(transactions, article_to_index):
    log("Building customer recent-purchase candidates...")
    embedded_articles = set(article_to_index)
    recent_transactions = transactions[transactions["article_id"].isin(embedded_articles)].copy()
    recent_transactions = recent_transactions.sort_values(["customer_id", "t_dat"], ascending=[True, False])
    recent_candidates = (
        recent_transactions.groupby("customer_id")["article_id"]
        .apply(lambda values: list(dict.fromkeys(values.tolist()))[:RECENT_PURCHASE_CANDIDATES])
        .to_dict()
    )
    log(f"Customers with recent embedded purchase candidates: {len(recent_candidates):,}")
    return recent_candidates


def build_metadata_similarity_candidates(transactions, articles, article_to_index):
    log("Building metadata-similar candidates from product_type and garment_group...")
    embedded_articles = set(article_to_index)
    embedded_articles_df = articles[articles["article_id"].isin(embedded_articles)].copy()
    embedded_transactions = transactions[transactions["article_id"].isin(embedded_articles)].copy()

    article_popularity = embedded_transactions["article_id"].value_counts()
    embedded_articles_df["popularity"] = embedded_articles_df["article_id"].map(article_popularity).fillna(0)

    product_type_top = (
        embedded_articles_df.sort_values(["product_type_no", "popularity"], ascending=[True, False])
        .groupby("product_type_no")["article_id"]
        .apply(lambda values: values.head(SIMILAR_PRODUCT_TYPE_CANDIDATES).tolist())
        .to_dict()
    )
    garment_group_top = (
        embedded_articles_df.sort_values(["garment_group_no", "popularity"], ascending=[True, False])
        .groupby("garment_group_no")["article_id"]
        .apply(lambda values: values.head(SIMILAR_GARMENT_GROUP_CANDIDATES).tolist())
        .to_dict()
    )

    article_meta = embedded_articles_df.set_index("article_id")[["product_type_no", "garment_group_no"]].to_dict("index")
    recent_transactions = embedded_transactions.sort_values(["customer_id", "t_dat"], ascending=[True, False])

    customer_candidates = {}
    for customer_id, values in tqdm(
        recent_transactions.groupby("customer_id")["article_id"],
        desc="Metadata similar candidates",
    ):
        candidates = []
        for article_id in list(dict.fromkeys(values.tolist()))[:RECENT_PURCHASE_CANDIDATES]:
            meta = article_meta.get(article_id)
            if not meta:
                continue
            candidates.extend(product_type_top.get(meta["product_type_no"], []))
            candidates.extend(garment_group_top.get(meta["garment_group_no"], []))
            if len(candidates) >= SIMILAR_PRODUCT_TYPE_CANDIDATES + SIMILAR_GARMENT_GROUP_CANDIDATES:
                break
        customer_candidates[customer_id] = list(dict.fromkeys(candidates))

    log(f"Customers with metadata-similar candidates: {len(customer_candidates):,}")
    return customer_candidates


def create_simple_submission(customers, popular_articles):
    # First full Kaggle pass uses a robust popularity fallback. In the next pass
    # we replace this with model-scored candidates.
    log("Creating starter submission from popularity fallback...")
    prediction = " ".join(popular_articles[:12])
    submission = pd.DataFrame({"customer_id": customers["customer_id"], "prediction": prediction})
    if SAMPLE_SUBMISSION_PATH.exists():
        log("Aligning with sample_submission customer IDs...")
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        submission = sample[["customer_id"]].merge(submission, on="customer_id", how="left")
        submission["prediction"] = submission["prediction"].fillna(prediction)
    submission.to_csv(SUBMISSION_PATH, index=False)
    log(f"Saved starter submission: {SUBMISSION_PATH}")


def score_rows(model, metadata, data, embeddings, profile_sums, profile_counts):
    numeric, categorical = prepare_inference_tabular_arrays(data, metadata)
    article_idx = data["article_index"].astype("int64").to_numpy()
    customer_idx = data["customer_index"].astype("int64").to_numpy()
    pair_counts = data["pair_purchase_count"].astype("float32").to_numpy()

    device = next(model.parameters()).device
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        if not getattr(score_rows, "_device_logged", False):
            log(f"[submission scoring] Using GPU tensors | allocated={allocated:.2f}GB | reserved={reserved:.2f}GB")
            score_rows._device_logged = True
    else:
        if not getattr(score_rows, "_device_logged", False):
            log("[submission scoring] Model is on CPU")
            score_rows._device_logged = True
    article_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
    profile_sum_tensor = torch.tensor(profile_sums, dtype=torch.float32, device=device)
    profile_count_tensor = torch.tensor(profile_counts, dtype=torch.float32, device=device)

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(data), INFERENCE_SCORE_BATCH_SIZE):
            end = min(start + INFERENCE_SCORE_BATCH_SIZE, len(data))
            numeric_b = torch.tensor(numeric[start:end], dtype=torch.float32, device=device)
            cat_b = torch.tensor(categorical[start:end], dtype=torch.long, device=device)
            article_b = torch.tensor(article_idx[start:end], dtype=torch.long, device=device)
            customer_b = torch.tensor(customer_idx[start:end], dtype=torch.long, device=device)
            pair_b = torch.tensor(pair_counts[start:end], dtype=torch.float32, device=device)

            article_emb, profiles, sim, hist = make_visual_batch(
                article_b,
                customer_b,
                pair_b,
                article_tensor,
                profile_sum_tensor,
                profile_count_tensor,
            )
            logits = model(numeric_b, cat_b, article_emb, profiles, sim, hist)
            scores.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(scores)


def create_model_scored_submission(
    model,
    metadata,
    customers,
    articles,
    transactions,
    embeddings,
    article_to_index,
    customer_to_index,
    profile_sums,
    profile_counts,
    popular_articles,
    customer_recent_candidates,
    customer_similar_candidates,
):
    log("Creating model-scored submission from popularity candidates...")
    fallback_prediction = " ".join(popular_articles[:12])

    if SAMPLE_SUBMISSION_PATH.exists():
        sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        submission_customers = sample["customer_id"].tolist()
    else:
        submission_customers = customers["customer_id"].tolist()

    scoreable_customers = [customer_id for customer_id in submission_customers if customer_id in customer_to_index]
    scoreable_candidates = [article_id for article_id in popular_articles if article_id in article_to_index]
    log(f"Scoreable customers: {len(scoreable_customers):,}")
    log(f"Global candidate articles: {len(scoreable_candidates):,}")

    if not scoreable_customers or not scoreable_candidates:
        log("No scoreable customers/candidates found. Falling back to popularity submission.")
        create_simple_submission(customers, popular_articles)
        return

    all_candidate_articles = set(scoreable_candidates)
    for customer_id in scoreable_customers:
        all_candidate_articles.update(customer_recent_candidates.get(customer_id, []))
        all_candidate_articles.update(customer_similar_candidates.get(customer_id, []))
    all_candidate_articles = [article_id for article_id in all_candidate_articles if article_id in article_to_index]
    log(f"Total unique candidate articles for scoring: {len(all_candidate_articles):,}")

    articles_meta = articles[articles["article_id"].isin(all_candidate_articles)].copy()
    customer_meta = customers[customers["customer_id"].isin(scoreable_customers)].copy()
    pair_counts = (
        transactions[
            transactions["customer_id"].isin(scoreable_customers)
            & transactions["article_id"].isin(all_candidate_articles)
        ]
        .groupby(["customer_id", "article_id"])
        .size()
        .reset_index(name="pair_purchase_count")
    )

    predictions = {}
    for start in tqdm(range(0, len(scoreable_customers), INFERENCE_CUSTOMER_BATCH_SIZE), desc="Scoring submission customers"):
        batch_customers = scoreable_customers[start:start + INFERENCE_CUSTOMER_BATCH_SIZE]
        rows = []
        for customer_id in batch_customers:
            customer_candidates = []
            customer_candidates.extend(customer_recent_candidates.get(customer_id, []))
            customer_candidates.extend(customer_similar_candidates.get(customer_id, []))
            customer_candidates.extend(scoreable_candidates)
            customer_candidates = [
                article_id
                for article_id in dict.fromkeys(customer_candidates)
                if article_id in article_to_index
            ][:CUSTOMER_CANDIDATE_LIMIT]
            for article_id in customer_candidates:
                rows.append((customer_id, article_id))

        grid = pd.DataFrame(rows, columns=["customer_id", "article_id"])
        grid = grid.merge(customer_meta, on="customer_id", how="left")
        grid = grid.merge(articles_meta, on="article_id", how="left")
        grid = grid.merge(pair_counts, on=["customer_id", "article_id"], how="left")
        grid["pair_purchase_count"] = grid["pair_purchase_count"].fillna(0).astype("float32")
        grid["article_index"] = grid["article_id"].map(article_to_index).astype("int64")
        grid["customer_index"] = grid["customer_id"].map(customer_to_index).astype("int64")
        grid = add_visual_scalar_features(grid, embeddings, profile_sums, profile_counts, verbose=False)
        grid["score"] = score_rows(model, metadata, grid, embeddings, profile_sums, profile_counts)

        top = (
            grid.sort_values(["customer_id", "score"], ascending=[True, False])
            .groupby("customer_id")["article_id"]
            .apply(lambda values: " ".join(values.head(12)))
        )
        predictions.update(top.to_dict())

    submission = pd.DataFrame({"customer_id": submission_customers})
    submission["prediction"] = submission["customer_id"].map(predictions).fillna(fallback_prediction)
    submission.to_csv(SUBMISSION_PATH, index=False)
    log(f"Saved model-scored submission: {SUBMISSION_PATH}")


def main():
    log("Starting Kaggle multimodal pipeline...")
    log(f"FAST_RUN={FAST_RUN} | EPOCHS={EPOCHS} | MAX_TRAIN_ROWS={MAX_TRAIN_ROWS}")
    get_device("startup check")
    set_seed(RANDOM_SEED)
    transactions, customers, articles = load_raw_data()
    article_ids, embeddings, article_to_index = build_or_load_image_embeddings(articles, transactions)
    customer_ids, customer_to_index, profile_sums, profile_counts = build_customer_visual_sums(
        transactions,
        embeddings,
        article_to_index,
    )

    train_pairs = build_training_pairs(transactions, article_to_index)
    train_pairs = train_pairs[train_pairs["customer_id"].isin(customer_to_index)].copy()
    train_data = attach_metadata_and_indices(
        train_pairs,
        customers,
        articles,
        transactions,
        article_to_index,
        customer_to_index,
    )
    train_data = add_visual_scalar_features(train_data, embeddings, profile_sums, profile_counts)
    model, metadata, best_auc = train_model(train_data, embeddings, profile_sums, profile_counts)

    popular_articles = build_popular_candidates(transactions, article_to_index)
    customer_recent_candidates = build_customer_recent_candidates(transactions, article_to_index)
    customer_similar_candidates = build_metadata_similarity_candidates(transactions, articles, article_to_index)
    create_model_scored_submission(
        model,
        metadata,
        customers,
        articles,
        transactions,
        embeddings,
        article_to_index,
        customer_to_index,
        profile_sums,
        profile_counts,
        popular_articles,
        customer_recent_candidates,
        customer_similar_candidates,
    )
    log("Pipeline complete.")


if __name__ == "__main__":
    main()
