from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import (
    DEFAULT_SEED,
    DEFAULT_VALIDATION_DAYS,
    MODEL_NAMES,
    MODELS_DIR,
    REPORTS_DIR,
    V2Defaults,
    ensure_v2_dirs,
)
from .data import load_core_tables, load_embeddings, log
from .features import (
    add_negative_pairs,
    build_customer_profiles,
    build_history_frame,
    encode_tabular,
    fit_tabular_metadata,
    make_cutoff,
    make_validation_positive_pairs,
    merge_metadata,
    metadata_to_jsonable,
    numeric_features_for_model,
    sample_positive_pairs,
    visual_arrays_for_pairs,
)
from .models import build_model


class PairDataset(Dataset):
    def __init__(
        self,
        numeric: np.ndarray | None,
        categorical: np.ndarray | None,
        article_emb: np.ndarray | None,
        profile_emb: np.ndarray | None,
        visual_similarity: np.ndarray | None,
        visual_history_count: np.ndarray | None,
        labels: np.ndarray,
    ) -> None:
        self.numeric = torch.tensor(numeric, dtype=torch.float32) if numeric is not None else None
        self.categorical = torch.tensor(categorical, dtype=torch.long) if categorical is not None else None
        self.article_emb = torch.tensor(article_emb, dtype=torch.float32) if article_emb is not None else None
        self.profile_emb = torch.tensor(profile_emb, dtype=torch.float32) if profile_emb is not None else None
        self.visual_similarity = (
            torch.tensor(visual_similarity, dtype=torch.float32) if visual_similarity is not None else None
        )
        self.visual_history_count = (
            torch.tensor(visual_history_count, dtype=torch.float32) if visual_history_count is not None else None
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"label": self.labels[index]}
        if self.numeric is not None:
            item["numeric"] = self.numeric[index]
            item["categorical"] = self.categorical[index]
        if self.article_emb is not None:
            item["article_emb"] = self.article_emb[index]
            item["profile_emb"] = self.profile_emb[index]
            item["visual_similarity"] = self.visual_similarity[index]
            item["visual_history_count"] = self.visual_history_count[index]
        return item


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_model(model_name: str, model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if model_name == "tabular_only":
        return model(batch["numeric"].to(device), batch["categorical"].to(device))
    if model_name == "image_history":
        return model(
            batch["article_emb"].to(device),
            batch["profile_emb"].to(device),
            batch["visual_similarity"].to(device),
            batch["visual_history_count"].to(device),
        )
    if model_name == "late_fusion":
        return model(
            batch["numeric"].to(device),
            batch["categorical"].to(device),
            batch["article_emb"].to(device),
            batch["profile_emb"].to(device),
            batch["visual_similarity"].to(device),
            batch["visual_history_count"].to(device),
        )
    raise ValueError(model_name)


def evaluate(model_name: str, model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true: list[float] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = forward_model(model_name, model, batch, device)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(probabilities.tolist())
            y_true.extend(batch["label"].numpy().tolist())
    labels = np.array(y_true)
    probabilities = np.array(y_prob)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            logits = forward_model(model_name, model, batch, device)
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model_name, model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
        history.append(metrics)
        log(f"{model_name} epoch {epoch}: loss={metrics['loss']:.4f} auc={metrics['auc_roc']:.4f} acc={metrics['accuracy']:.4f}")
    return {"history": history, "final": history[-1] if history else {}}


def prepare_fold_data(args: argparse.Namespace) -> tuple[dict, dict]:
    raw_dir, transactions, customers, articles = load_core_tables(args.raw_dir)
    embeddings, article_ids, article_to_index = load_embeddings(args.embeddings_path, args.embedding_ids_path, mmap_mode=None)

    folds = pd.read_csv(args.folds_csv)
    cutoff = make_cutoff(transactions, args.validation_days)
    val_customers = set(folds.loc[folds["fold_id"] == args.fold_id, "customer_id"])
    train_customers = set(folds.loc[folds["fold_id"] != args.fold_id, "customer_id"])
    if train_customers & val_customers:
        raise RuntimeError("Customer leakage detected between train and validation.")

    article_pool = np.array([article_id for article_id in article_ids if article_id in set(articles["article_id"])])
    train_positive = sample_positive_pairs(transactions, train_customers, cutoff, args.max_train_positives, args.seed)
    val_positive = make_validation_positive_pairs(transactions, val_customers, cutoff, args.max_val_positives, args.seed)
    train_pairs = add_negative_pairs(train_positive, article_pool, args.negatives_per_positive, args.seed)
    val_pairs = add_negative_pairs(val_positive, article_pool, args.negatives_per_positive, args.seed + 1000)

    train_history = build_history_frame(transactions, train_customers, cutoff)
    val_history = build_history_frame(transactions, val_customers, cutoff)
    train_customer_to_index, train_sums, train_counts, train_pair_counts = build_customer_profiles(
        train_history, embeddings, article_to_index
    )
    val_customer_to_index, val_sums, val_counts, val_pair_counts = build_customer_profiles(
        val_history, embeddings, article_to_index
    )

    train_pairs = train_pairs[
        train_pairs["article_id"].isin(article_to_index) & train_pairs["customer_id"].isin(train_customer_to_index)
    ].reset_index(drop=True)
    val_pairs = val_pairs[
        val_pairs["article_id"].isin(article_to_index) & val_pairs["customer_id"].isin(val_customer_to_index)
    ].reset_index(drop=True)

    train_article_emb, train_profile_emb, train_sim, train_hist = visual_arrays_for_pairs(
        train_pairs, embeddings, article_to_index, train_customer_to_index, train_sums, train_counts, train_pair_counts
    )
    val_article_emb, val_profile_emb, val_sim, val_hist = visual_arrays_for_pairs(
        val_pairs, embeddings, article_to_index, val_customer_to_index, val_sums, val_counts, val_pair_counts
    )

    train_frame = merge_metadata(train_pairs, customers, articles)
    val_frame = merge_metadata(val_pairs, customers, articles)
    for frame, sim, hist in [(train_frame, train_sim, train_hist), (val_frame, val_sim, val_hist)]:
        frame["visual_similarity"] = sim
        frame["visual_history_count"] = hist

    context = {
        "raw_dir": str(raw_dir),
        "fold_id": args.fold_id,
        "cutoff": str(cutoff.date()),
        "train_customers": len(train_customers),
        "validation_customers": len(val_customers),
        "train_rows": len(train_pairs),
        "validation_rows": len(val_pairs),
        "image_dim": int(embeddings.shape[1]),
        "article_ids": article_ids,
    }
    arrays = {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_frame": train_frame,
        "val_frame": val_frame,
        "train_article_emb": train_article_emb,
        "train_profile_emb": train_profile_emb,
        "train_sim": train_sim,
        "train_hist": train_hist,
        "val_article_emb": val_article_emb,
        "val_profile_emb": val_profile_emb,
        "val_sim": val_sim,
        "val_hist": val_hist,
    }
    return context, arrays


def train_one_requested_model(model_name: str, args: argparse.Namespace, context: dict, arrays: dict) -> dict:
    numeric_features = numeric_features_for_model(model_name)
    if model_name == "image_history":
        metadata = {"numeric_features": [], "categorical_features": [], "category_maps": {}}
        train_numeric = train_categorical = val_numeric = val_categorical = None
    else:
        metadata = fit_tabular_metadata(arrays["train_frame"], numeric_features)
        train_numeric, train_categorical = encode_tabular(arrays["train_frame"], metadata)
        val_numeric, val_categorical = encode_tabular(arrays["val_frame"], metadata)

    train_dataset = PairDataset(
        train_numeric,
        train_categorical,
        None if model_name == "tabular_only" else arrays["train_article_emb"],
        None if model_name == "tabular_only" else arrays["train_profile_emb"],
        None if model_name == "tabular_only" else arrays["train_sim"],
        None if model_name == "tabular_only" else arrays["train_hist"],
        arrays["train_pairs"]["label"].to_numpy(dtype="float32"),
    )
    val_dataset = PairDataset(
        val_numeric,
        val_categorical,
        None if model_name == "tabular_only" else arrays["val_article_emb"],
        None if model_name == "tabular_only" else arrays["val_profile_emb"],
        None if model_name == "tabular_only" else arrays["val_sim"],
        None if model_name == "tabular_only" else arrays["val_hist"],
        arrays["val_pairs"]["label"].to_numpy(dtype="float32"),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name, metadata, context["image_dim"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = train_model(model_name, model, train_loader, val_loader, device, args.epochs, args.learning_rate)

    checkpoint = {
        "model_name": model_name,
        "fold_id": args.fold_id,
        "state_dict": model.state_dict(),
        "metadata": metadata_to_jsonable(metadata),
        "image_dim": context["image_dim"],
        "context": context,
        "metrics": result,
    }
    output_path = args.model_dir / f"{model_name}_fold{args.fold_id}.pt"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    log(f"Saved checkpoint: {output_path}")
    row = {
        "fold_id": args.fold_id,
        "model": model_name,
        "auc_roc": result["final"].get("auc_roc"),
        "accuracy": result["final"].get("accuracy"),
        "train_rows": context["train_rows"],
        "validation_rows": context["validation_rows"],
        "checkpoint": str(output_path),
    }
    return row


def parse_models(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = set(values) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return values


def main() -> None:
    defaults = V2Defaults()
    parser = argparse.ArgumentParser(description="Train proposal v2 tabular, image-history, and late-fusion models.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--embedding-ids-path", type=Path, default=None)
    parser.add_argument("--folds-csv", type=Path, default=REPORTS_DIR / "proposal_v2_fold_splits.csv")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--models", default="tabular_only,image_history,late_fusion")
    parser.add_argument("--validation-days", type=int, default=defaults.validation_days)
    parser.add_argument("--negatives-per-positive", type=int, default=defaults.negatives_per_positive)
    parser.add_argument("--max-train-positives", type=int, default=200_000)
    parser.add_argument("--max-val-positives", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.train_batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=REPORTS_DIR / "proposal_v2_classification_metrics.csv")
    parser.add_argument("--context-json", type=Path, default=None)
    args = parser.parse_args()

    ensure_v2_dirs()
    set_seed(args.seed)
    context, arrays = prepare_fold_data(args)
    rows = []
    for model_name in parse_models(args.models):
        rows.append(train_one_requested_model(model_name, args, context, arrays))

    metrics = pd.DataFrame(rows)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.metrics_csv.exists():
        previous = pd.read_csv(args.metrics_csv)
        metrics = pd.concat([previous, metrics], ignore_index=True)
        metrics = metrics.drop_duplicates(["fold_id", "model"], keep="last")
    metrics.to_csv(args.metrics_csv, index=False)
    context_path = args.context_json or (REPORTS_DIR / "folds" / f"fold{args.fold_id}_training_context.json")
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    log(f"Saved metrics: {args.metrics_csv}")
    log(f"Saved context: {context_path}")


if __name__ == "__main__":
    main()
