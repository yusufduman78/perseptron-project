from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import DEFAULT_N_FOLDS, DEFAULT_SEED, DEFAULT_VALIDATION_DAYS, REPORTS_DIR, ensure_v2_dirs
from .data import load_core_tables, log


def build_customer_folds(
    transactions: pd.DataFrame,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
    min_history: int = 2,
) -> pd.DataFrame:
    customer_counts = transactions.groupby("customer_id").size()
    eligible_customers = customer_counts[customer_counts >= min_history].index.to_numpy()
    eligible_customers = np.array(sorted(eligible_customers))

    fold_rows: list[dict] = []
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (_, val_idx) in enumerate(splitter.split(eligible_customers)):
        for customer_id in eligible_customers[val_idx]:
            fold_rows.append({"customer_id": customer_id, "fold_id": fold_id})
    return pd.DataFrame(fold_rows)


def validate_fold_protocol(
    transactions: pd.DataFrame,
    folds: pd.DataFrame,
    validation_days: int = DEFAULT_VALIDATION_DAYS,
) -> dict:
    cutoff = transactions["t_dat"].max() - pd.Timedelta(days=validation_days)
    summary = {
        "cutoff": str(cutoff.date()),
        "validation_days": validation_days,
        "folds": [],
    }
    for fold_id in sorted(folds["fold_id"].unique()):
        val_customers = set(folds.loc[folds["fold_id"] == fold_id, "customer_id"])
        train_customers = set(folds.loc[folds["fold_id"] != fold_id, "customer_id"])
        intersection = train_customers & val_customers
        val_tx = transactions[transactions["customer_id"].isin(val_customers)]
        history = val_tx[val_tx["t_dat"] <= cutoff]
        truth = val_tx[val_tx["t_dat"] > cutoff]
        summary["folds"].append(
            {
                "fold_id": int(fold_id),
                "train_customers": len(train_customers),
                "validation_customers": len(val_customers),
                "customer_intersection": len(intersection),
                "validation_history_rows": int(len(history)),
                "validation_truth_rows": int(len(truth)),
                "validation_truth_customers": int(truth["customer_id"].nunique()),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create proposal v2 customer-level 5-fold splits.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--validation-days", type=int, default=DEFAULT_VALIDATION_DAYS)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "proposal_v2_fold_splits.csv")
    parser.add_argument("--summary", type=Path, default=REPORTS_DIR / "proposal_v2_fold_protocol_summary.json")
    args = parser.parse_args()

    ensure_v2_dirs()
    _, transactions, _, _ = load_core_tables(args.raw_dir)
    folds = build_customer_folds(transactions, args.n_folds, args.seed, args.min_history)
    summary = validate_fold_protocol(transactions, folds, args.validation_days)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Saved folds: {args.output}")
    log(f"Saved protocol summary: {args.summary}")
    for row in summary["folds"]:
        if row["customer_intersection"] != 0:
            raise RuntimeError(f"Fold {row['fold_id']} has customer leakage.")


if __name__ == "__main__":
    main()
