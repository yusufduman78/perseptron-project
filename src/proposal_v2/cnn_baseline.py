from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import DEFAULT_SEED, MODELS_DIR, REPORTS_DIR, ensure_v2_dirs
from .data import article_image_path, load_core_tables, log, resolve_images_dir
from .features import add_negative_pairs, make_cutoff, make_validation_positive_pairs, sample_positive_pairs
from .models import EfficientNetBinaryClassifier


class ArticleImagePairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, images_dir: Path, transform) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        row = self.pairs.iloc[index]
        path = article_image_path(self.images_dir, row["article_id"])
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(float(row["label"]), dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())
    labels = np.array(y_true)
    probs = np.array(y_prob)
    return {
        "auc_roc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(labels, probs >= 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train proposal v2 EfficientNet-B0 image-only CNN baseline.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--folds-csv", type=Path, default=REPORTS_DIR / "proposal_v2_fold_splits.csv")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--max-train-positives", type=int, default=20_000)
    parser.add_argument("--max-val-positives", type=int, default=5_000)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=REPORTS_DIR / "proposal_v2_cnn_metrics.csv")
    args = parser.parse_args()

    ensure_v2_dirs()
    torch.manual_seed(args.seed)
    raw_dir, transactions, _, articles = load_core_tables(args.raw_dir)
    images_dir = resolve_images_dir(raw_dir, args.images_dir)
    folds = pd.read_csv(args.folds_csv)
    cutoff = make_cutoff(transactions, args.validation_days)
    val_customers = set(folds.loc[folds["fold_id"] == args.fold_id, "customer_id"])
    train_customers = set(folds.loc[folds["fold_id"] != args.fold_id, "customer_id"])
    article_pool = articles["article_id"].astype(str).to_numpy()

    train_positive = sample_positive_pairs(transactions, train_customers, cutoff, args.max_train_positives, args.seed)
    val_positive = make_validation_positive_pairs(transactions, val_customers, cutoff, args.max_val_positives, args.seed)
    train_pairs = add_negative_pairs(train_positive, article_pool, args.negatives_per_positive, args.seed)
    val_pairs = add_negative_pairs(val_positive, article_pool, args.negatives_per_positive, args.seed + 1000)
    train_pairs = train_pairs[train_pairs["article_id"].map(lambda value: article_image_path(images_dir, value).exists())]
    val_pairs = val_pairs[val_pairs["article_id"].map(lambda value: article_image_path(images_dir, value).exists())]

    model = EfficientNetBinaryClassifier(train_backbone=args.train_backbone)
    transform = model.weights.transforms()
    train_loader = DataLoader(ArticleImagePairDataset(train_pairs, images_dir, transform), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArticleImagePairDataset(val_pairs, images_dir, transform), batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in tqdm(train_loader, desc=f"image_only_effnet_cnn epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device)
        metrics.update({"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan")})
        history.append(metrics)
        log(f"cnn epoch {epoch}: loss={metrics['loss']:.4f} auc={metrics['auc_roc']:.4f} acc={metrics['accuracy']:.4f}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.model_dir / f"image_only_effnet_cnn_fold{args.fold_id}.pt"
    torch.save(
        {
            "model_name": "image_only_effnet_cnn",
            "fold_id": args.fold_id,
            "state_dict": model.state_dict(),
            "train_backbone": args.train_backbone,
            "metrics": {"history": history, "final": history[-1] if history else {}},
        },
        checkpoint_path,
    )
    row = {
        "fold_id": args.fold_id,
        "model": "image_only_effnet_cnn",
        "auc_roc": history[-1]["auc_roc"],
        "accuracy": history[-1]["accuracy"],
        "train_rows": len(train_pairs),
        "validation_rows": len(val_pairs),
        "checkpoint": str(checkpoint_path),
    }
    output = pd.DataFrame([row])
    if args.metrics_csv.exists():
        previous = pd.read_csv(args.metrics_csv)
        output = pd.concat([previous, output], ignore_index=True).drop_duplicates(["fold_id", "model"], keep="last")
    output.to_csv(args.metrics_csv, index=False)
    log(f"Saved CNN checkpoint: {checkpoint_path}")
    log(f"Saved CNN metrics: {args.metrics_csv}")


if __name__ == "__main__":
    main()
