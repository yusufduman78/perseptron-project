from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import REPORTS_DIR, ensure_v2_dirs
from .data import log


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


def summarize(values: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    aggregations = {}
    for column in metric_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    return values.groupby("model", as_index=False).agg(**aggregations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate proposal v2 fold metrics into report-ready summaries.")
    parser.add_argument("--classification-csv", type=Path, default=REPORTS_DIR / "proposal_v2_classification_metrics.csv")
    parser.add_argument("--ranking-csv", type=Path, default=REPORTS_DIR / "proposal_v2_ranking_metrics.csv")
    parser.add_argument("--output-md", type=Path, default=REPORTS_DIR / "proposal_v2_cv_summary.md")
    args = parser.parse_args()

    ensure_v2_dirs()
    sections = ["# Proposal v2 CV Summary\n"]
    if args.classification_csv.exists():
        classification = pd.read_csv(args.classification_csv)
        sections.append("## Classification Metrics\n")
        sections.append(frame_to_markdown(summarize(classification, ["auc_roc", "accuracy"])))
        sections.append("")
    if args.ranking_csv.exists():
        ranking = pd.read_csv(args.ranking_csv)
        per_fold = (
            ranking.groupby(["fold_id", "model"], as_index=False)
            .agg(
                map_at_12=("map_at_12", "mean"),
                precision_at_10=("precision_at_10", "mean"),
                recall_at_10=("recall_at_10", "mean"),
                hit_rate_at_12=("hit_rate_at_12", "mean"),
                candidate_recall=("candidate_recall", "mean"),
            )
        )
        sections.append("## Ranking Metrics\n")
        sections.append(
            frame_to_markdown(
                summarize(
                    per_fold,
                    ["map_at_12", "precision_at_10", "recall_at_10", "hit_rate_at_12", "candidate_recall"],
                )
            )
        )
        sections.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n\n".join(sections), encoding="utf-8")
    log(f"Saved aggregate summary: {args.output_md}")


if __name__ == "__main__":
    main()
