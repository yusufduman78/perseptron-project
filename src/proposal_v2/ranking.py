from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODELS_DIR, REPORTS_DIR, V2Defaults, ensure_v2_dirs
from .data import load_core_tables, load_embeddings, log
from .features import (
    build_customer_profiles,
    build_history_frame,
    encode_tabular,
    make_cutoff,
    merge_metadata,
    visual_arrays_for_pairs,
)
from .metrics import average_precision_at_k, hit_rate_at_k, precision_at_k, recall_at_k
from .models import build_model
from .train import forward_model


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.to_dict(orient="records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def load_checkpoint_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_name"], checkpoint["metadata"], checkpoint["image_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def score_pairs(
    checkpoint: dict,
    model: torch.nn.Module,
    pairs: pd.DataFrame,
    frame: pd.DataFrame,
    visual_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model_name = checkpoint["model_name"]
    article_emb, profile_emb, visual_similarity, visual_history_count = visual_arrays
    if model_name == "tabular_only":
        numeric, categorical = encode_tabular(frame, checkpoint["metadata"])
    elif model_name == "late_fusion":
        numeric, categorical = encode_tabular(frame, checkpoint["metadata"])
    else:
        numeric = categorical = None

    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            end = start + batch_size
            batch = {"label": torch.zeros(end - start)}
            if model_name in {"tabular_only", "late_fusion"}:
                batch["numeric"] = torch.tensor(numeric[start:end], dtype=torch.float32)
                batch["categorical"] = torch.tensor(categorical[start:end], dtype=torch.long)
            if model_name in {"image_history", "late_fusion"}:
                batch["article_emb"] = torch.tensor(article_emb[start:end], dtype=torch.float32)
                batch["profile_emb"] = torch.tensor(profile_emb[start:end], dtype=torch.float32)
                batch["visual_similarity"] = torch.tensor(visual_similarity[start:end], dtype=torch.float32)
                batch["visual_history_count"] = torch.tensor(visual_history_count[start:end], dtype=torch.float32)
            logits = forward_model(model_name, model, batch, device)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def build_global_popularity(history: pd.DataFrame, limit: int) -> list[str]:
    return history["article_id"].value_counts().head(limit).index.astype(str).tolist()


def build_co_purchase(history: pd.DataFrame, max_per_item: int) -> dict[str, list[str]]:
    pairs = history.groupby("customer_id")["article_id"].apply(lambda values: list(dict.fromkeys(values))).tolist()
    counts: dict[str, dict[str, int]] = {}
    for basket in pairs:
        unique = basket[-50:]
        for item in unique:
            target = counts.setdefault(item, {})
            for other in unique:
                if other != item:
                    target[other] = target.get(other, 0) + 1
    return {
        item: [other for other, _ in sorted(others.items(), key=lambda kv: kv[1], reverse=True)[:max_per_item]]
        for item, others in counts.items()
    }


def candidate_pool_for_customer(
    customer_id: str,
    history_items: list[str],
    profile: np.ndarray,
    embeddings: np.ndarray,
    article_ids: list[str],
    global_popular: list[str],
    co_purchase: dict[str, list[str]],
    candidate_limit: int,
    visual_neighbors: int,
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(global_popular[: min(candidate_limit, 1000)])
    for item in history_items[-20:]:
        candidates.extend(co_purchase.get(item, [])[:50])
    if np.linalg.norm(profile) > 0:
        scores = np.asarray(embeddings @ (profile / max(np.linalg.norm(profile), 1e-8)))
        top_indices = np.argpartition(-scores, min(visual_neighbors, len(scores) - 1))[:visual_neighbors]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        candidates.extend(article_ids[index] for index in top_indices)

    seen: set[str] = set(history_items)
    output = []
    for article_id in candidates:
        if article_id in seen:
            continue
        seen.add(article_id)
        output.append(article_id)
        if len(output) >= candidate_limit:
            break
    return output


def heuristic_scores(pairs: pd.DataFrame, frame: pd.DataFrame, visual_similarity: np.ndarray) -> np.ndarray:
    popularity = frame.groupby("article_id")["article_id"].transform("count").to_numpy(dtype="float32")
    if popularity.max() > 0:
        popularity = popularity / popularity.max()
    visual = np.nan_to_num(visual_similarity, nan=0.0).astype("float32")
    visual = (visual + 1.0) / 2.0
    return 0.6 * popularity + 0.4 * visual


def main() -> None:
    defaults = V2Defaults()
    parser = argparse.ArgumentParser(description="Run proposal v2 customer-level MAP@12 ranking evaluation.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--embedding-ids-path", type=Path, default=None)
    parser.add_argument("--folds-csv", type=Path, default=REPORTS_DIR / "proposal_v2_fold_splits.csv")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--validation-days", type=int, default=defaults.validation_days)
    parser.add_argument("--candidate-limit", type=int, default=defaults.candidate_limit)
    parser.add_argument("--visual-neighbors", type=int, default=defaults.visual_neighbors)
    parser.add_argument("--co-purchase-per-item", type=int, default=defaults.co_purchase_per_item)
    parser.add_argument("--sample-customers", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--precision-k", type=int, default=defaults.precision_k)
    parser.add_argument("--hybrid-weights", default="0.25,0.45,0.65")
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--output-csv", type=Path, default=REPORTS_DIR / "proposal_v2_ranking_metrics.csv")
    parser.add_argument("--summary-md", type=Path, default=REPORTS_DIR / "proposal_v2_cv_summary.md")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ensure_v2_dirs()
    _, transactions, customers, articles = load_core_tables(args.raw_dir)
    embeddings, article_ids, article_to_index = load_embeddings(args.embeddings_path, args.embedding_ids_path, mmap_mode=None)
    folds = pd.read_csv(args.folds_csv)
    cutoff = make_cutoff(transactions, args.validation_days)
    val_customers = set(folds.loc[folds["fold_id"] == args.fold_id, "customer_id"])
    val_tx = transactions[transactions["customer_id"].isin(val_customers)].copy()
    history = val_tx[val_tx["t_dat"] <= cutoff]
    truth = val_tx[val_tx["t_dat"] > cutoff]
    eligible_customers = sorted(set(history["customer_id"]) & set(truth["customer_id"]))
    if args.sample_customers:
        eligible_customers = eligible_customers[: args.sample_customers]

    customer_to_index, profile_sums, profile_counts, pair_counts = build_customer_profiles(
        history, embeddings, article_to_index
    )
    global_popular = build_global_popularity(transactions[transactions["t_dat"] <= cutoff], args.candidate_limit)
    co_purchase = build_co_purchase(transactions[transactions["t_dat"] <= cutoff], args.co_purchase_per_item)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths = {
        name: args.model_dir / f"{name}_fold{args.fold_id}.pt"
        for name in ["tabular_only", "image_history", "late_fusion"]
    }
    models = {
        name: load_checkpoint_model(path, device)
        for name, path in checkpoint_paths.items()
        if path.exists()
    }
    if not models:
        raise FileNotFoundError(f"No checkpoints found for fold {args.fold_id} in {args.model_dir}")

    rows = []
    hybrid_weights = [float(value.strip()) for value in args.hybrid_weights.split(",") if value.strip()]
    for customer_id in eligible_customers:
        customer_history = history.loc[history["customer_id"] == customer_id, "article_id"].astype(str).tolist()
        relevant = set(truth.loc[truth["customer_id"] == customer_id, "article_id"].astype(str))
        customer_idx = customer_to_index.get(customer_id)
        if customer_idx is None or not relevant:
            continue
        profile = profile_sums[customer_idx] / max(profile_counts[customer_idx], 1.0)
        candidates = candidate_pool_for_customer(
            customer_id,
            customer_history,
            profile,
            embeddings,
            article_ids,
            global_popular,
            co_purchase,
            args.candidate_limit,
            args.visual_neighbors,
        )
        if not candidates:
            continue
        pairs = pd.DataFrame({"customer_id": customer_id, "article_id": candidates})
        frame = merge_metadata(pairs, customers, articles)
        visual = visual_arrays_for_pairs(
            pairs, embeddings, article_to_index, customer_to_index, profile_sums, profile_counts, pair_counts
        )
        frame["visual_similarity"] = visual[2]
        frame["visual_history_count"] = visual[3]

        scored: dict[str, np.ndarray] = {}
        for name, (checkpoint, model) in models.items():
            scored[name] = score_pairs(checkpoint, model, pairs, frame, visual, device, args.batch_size)

        if "late_fusion" in scored:
            heuristics = heuristic_scores(pairs, frame, visual[2])
            for weight in hybrid_weights:
                scored[f"late_fusion_hybrid_w{int(weight * 100):03d}"] = (1 - weight) * scored["late_fusion"] + weight * heuristics

        for model_name, scores in scored.items():
            order = np.argsort(-scores)
            predictions = [candidates[index] for index in order[: args.top_k]]
            rows.append(
                {
                    "fold_id": args.fold_id,
                    "customer_id": customer_id,
                    "model": model_name,
                    "map_at_12": average_precision_at_k(predictions, relevant, args.top_k),
                    "precision_at_10": precision_at_k(predictions, relevant, args.precision_k),
                    "recall_at_10": recall_at_k(predictions, relevant, args.precision_k),
                    "hit_rate_at_12": hit_rate_at_k(predictions, relevant, args.top_k),
                    "candidate_recall": len(set(candidates) & relevant) / len(relevant),
                    "candidate_count": len(candidates),
                    "ground_truth_count": len(relevant),
                    "predictions": " ".join(predictions),
                }
            )

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists():
        previous = pd.read_csv(args.output_csv)
        output = pd.concat([previous[previous["fold_id"] != args.fold_id], output], ignore_index=True)
    output.to_csv(args.output_csv, index=False)
    summary = (
        output.groupby("model", as_index=False)
        .agg(
            customers_evaluated=("customer_id", "nunique"),
            map_at_12=("map_at_12", "mean"),
            precision_at_10=("precision_at_10", "mean"),
            recall_at_10=("recall_at_10", "mean"),
            hit_rate_at_12=("hit_rate_at_12", "mean"),
            candidate_recall=("candidate_recall", "mean"),
        )
        .sort_values("map_at_12", ascending=False)
    )
    args.summary_md.write_text(
        "# Proposal v2 Ranking Summary\n\n"
        f"- Fold evaluated: {args.fold_id}\n"
        f"- Candidate limit: {args.candidate_limit}\n"
        f"- Top-k: {args.top_k}\n\n"
        + frame_to_markdown(summary)
        + "\n",
        encoding="utf-8",
    )
    log(f"Saved ranking rows: {args.output_csv}")
    log(f"Saved summary: {args.summary_md}")


if __name__ == "__main__":
    main()
