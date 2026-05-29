from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEMO_BACKEND_DIR = PROJECT_DIR / "demo" / "backend"
if str(DEMO_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_BACKEND_DIR))

from recommender import DEFAULT_CUSTOMER_PROFILE, PerseptronRecommender  # noqa: E402


RAW_DIR = PROJECT_DIR / "data" / "raw"
REPORTS_DIR = PROJECT_DIR / "reports"
TRANSACTIONS_PATH = RAW_DIR / "transactions_train.csv"
CUSTOMERS_PATH = RAW_DIR / "customers.csv"

METRIC_OUTPUT_PATH = REPORTS_DIR / "ranking_metrics_validation.csv"
SUMMARY_OUTPUT_PATH = REPORTS_DIR / "ranking_metrics_validation_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final Perseptron models with time-based validation ranking metrics "
            "(MAP@12, Precision@10, Recall@10)."
        )
    )
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--sample-customers", type=int, default=500)
    parser.add_argument("--candidate-limit", type=int, default=2000)
    parser.add_argument(
        "--candidate-mode",
        choices=["legacy", "improved"],
        default="improved",
        help="legacy uses the demo recommender candidate pool; improved adds time-aware and metadata-rich candidates.",
    )
    parser.add_argument("--visual-neighbors", type=int, default=1200)
    parser.add_argument("--co-purchase-per-item", type=int, default=250)
    parser.add_argument("--co-purchase-max-history-per-customer", type=int, default=100)
    parser.add_argument(
        "--disable-co-purchase",
        action="store_true",
        help="Disable customer-history co-purchase candidate expansion.",
    )
    parser.add_argument(
        "--hybrid-heuristic-weight",
        type=float,
        default=0.45,
        help="Weight of recency/co-purchase/visual heuristic in the late_fusion_hybrid reranker.",
    )
    parser.add_argument(
        "--hybrid-heuristic-weights",
        default=None,
        help=(
            "Comma-separated hybrid weights for a sweep, e.g. 0.25,0.45,0.65. "
            "Overrides --hybrid-heuristic-weight and evaluates all weights in one pass."
        ),
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--precision-k", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cpu or cuda.")
    parser.add_argument("--output-csv", type=Path, default=METRIC_OUTPUT_PATH)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_OUTPUT_PATH)
    return parser.parse_args()


def get_hybrid_weights(args: argparse.Namespace) -> list[float]:
    if not args.hybrid_heuristic_weights:
        return [args.hybrid_heuristic_weight]

    weights = []
    for raw_value in args.hybrid_heuristic_weights.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        weight = float(raw_value)
        if weight < 0 or weight > 1:
            raise ValueError(f"Hybrid heuristic weights must be between 0 and 1: {weight}")
        weights.append(weight)
    if not weights:
        raise ValueError("--hybrid-heuristic-weights did not contain any valid values.")
    return weights


def format_hybrid_weight(weight: float) -> str:
    return f"w{int(round(weight * 100)):03d}"


def hybrid_model_name(weight: float, total_weights: int) -> str:
    if total_weights == 1:
        return "late_fusion_hybrid"
    return f"late_fusion_hybrid_{format_hybrid_weight(weight)}"


class ImprovedCandidateBuilder:
    def __init__(
        self,
        recommender: PerseptronRecommender,
        train: pd.DataFrame,
        candidate_limit: int,
        visual_neighbors: int,
        target_history_articles: set[str] | None = None,
        co_purchase_per_item: int = 250,
        co_purchase_max_history_per_customer: int = 100,
    ) -> None:
        self.recommender = recommender
        self.train = train
        self.candidate_limit = candidate_limit
        self.visual_neighbors = visual_neighbors
        self.co_purchase_per_item = co_purchase_per_item
        self.co_purchase_max_history_per_customer = co_purchase_max_history_per_customer
        self.embedded_articles = set(recommender.article_to_index)
        self.article_popularity = train["article_id"].value_counts()
        self.article_popularity = self.article_popularity[self.article_popularity.index.isin(self.embedded_articles)]
        self.article_rank = {article_id: rank for rank, article_id in enumerate(self.article_popularity.index)}
        self.global_popular = self.article_popularity.head(2500).index.tolist()

        max_train_date = train["t_dat"].max()
        self.last_7d_popular = self._popular_since(max_train_date - pd.Timedelta(days=7), 1200)
        self.last_30d_popular = self._popular_since(max_train_date - pd.Timedelta(days=30), 1800)
        self.last_7d_rank = {article_id: rank for rank, article_id in enumerate(self.last_7d_popular)}
        self.last_30d_rank = {article_id: rank for rank, article_id in enumerate(self.last_30d_popular)}
        self.group_indexes = self._build_group_indexes()
        self.co_purchase_index = self._build_co_purchase_index(target_history_articles or set())
        self.group_columns = [
            "product_type_no",
            "department_no",
            "section_no",
            "garment_group_no",
            "colour_group_code",
        ]
        self.article_meta = (
            self.recommender.articles[
                self.recommender.articles["article_id"].isin(self.embedded_articles)
            ]
            .set_index("article_id")
            .reindex(columns=self.group_columns)
            .to_dict("index")
        )

    def _popular_since(self, min_date: pd.Timestamp, limit: int) -> list[str]:
        frame = self.train[self.train["t_dat"] >= min_date]
        counts = frame["article_id"].value_counts()
        counts = counts[counts.index.isin(self.embedded_articles)]
        return counts.head(limit).index.tolist()

    def _build_group_indexes(self) -> dict[str, dict[object, list[str]]]:
        print("Building metadata-aware candidate indexes...")
        articles = self.recommender.articles.copy()
        articles = articles[articles["article_id"].isin(self.embedded_articles)].copy()
        articles["popularity_rank"] = articles["article_id"].map(self.article_rank).fillna(10**9)
        group_columns = [
            "product_type_no",
            "department_no",
            "section_no",
            "garment_group_no",
            "colour_group_code",
        ]
        indexes: dict[str, dict[object, list[str]]] = {}
        for column in group_columns:
            if column not in articles.columns:
                continue
            sorted_articles = articles.sort_values(["popularity_rank", "article_id"])
            indexes[column] = (
                sorted_articles.groupby(column, dropna=True)["article_id"]
                .apply(lambda values: values.head(700).tolist())
                .to_dict()
            )
        return indexes

    def _build_co_purchase_index(self, target_articles: set[str]) -> dict[str, list[str]]:
        target_articles = {
            str(article_id).zfill(10)
            for article_id in target_articles
            if str(article_id).zfill(10) in self.embedded_articles
        }
        if not target_articles or self.co_purchase_per_item <= 0:
            print("Co-purchase candidate index disabled or empty.")
            return {}

        print(f"Building co-purchase candidate index for {len(target_articles):,} target history articles...")
        target_customers = self.train.loc[
            self.train["article_id"].isin(target_articles),
            "customer_id",
        ].unique()
        frame = self.train[self.train["customer_id"].isin(target_customers)]
        print(f"Co-purchase source customers: {len(target_customers):,}")
        print(f"Co-purchase source transactions: {len(frame):,}")

        counters = {article_id: Counter() for article_id in target_articles}
        for _, values in tqdm(frame.groupby("customer_id", sort=False)["article_id"], desc="Co-purchase baskets"):
            items = list(dict.fromkeys(values.tail(self.co_purchase_max_history_per_customer).tolist()))
            target_hits = [article_id for article_id in items if article_id in target_articles]
            if not target_hits:
                continue
            embedded_items = [article_id for article_id in items if article_id in self.embedded_articles]
            for target in target_hits:
                counters[target].update(article_id for article_id in embedded_items if article_id != target)

        index = {
            article_id: [co_article for co_article, _ in counter.most_common(self.co_purchase_per_item)]
            for article_id, counter in counters.items()
            if counter
        }
        print(f"Co-purchase indexes with candidates: {len(index):,}")
        return index

    @staticmethod
    def _add_unique(target: list[str], seen: set[str], values: list[str], limit: int) -> None:
        for article_id in values:
            article_id = str(article_id).zfill(10)
            if article_id in seen:
                continue
            seen.add(article_id)
            target.append(article_id)
            if len(target) >= limit:
                return

    def build(self, history_article_ids: list[str]) -> list[str]:
        history_article_ids = [str(article_id).zfill(10) for article_id in history_article_ids]
        seen = set(history_article_ids)
        candidates: list[str] = []

        # Time-aware popularity is intentionally first; H&M has strong short-term trend effects.
        self._add_unique(candidates, seen, self.last_7d_popular[:600], self.candidate_limit)

        for article_id in reversed(history_article_ids):
            if len(candidates) >= self.candidate_limit:
                break
            self._add_unique(
                candidates,
                seen,
                self.co_purchase_index.get(article_id, [])[: self.co_purchase_per_item],
                self.candidate_limit,
            )

        self._add_unique(candidates, seen, self.last_30d_popular[:800], self.candidate_limit)

        history_meta = self.recommender.articles[
            self.recommender.articles["article_id"].isin(history_article_ids)
        ]
        for column, per_group in self.group_indexes.items():
            if len(candidates) >= self.candidate_limit:
                break
            if column not in history_meta.columns:
                continue
            group_values = history_meta[column].dropna().tolist()
            # Recent history gets repeated values naturally; dict keeps first occurrence.
            for value in dict.fromkeys(reversed(group_values)):
                self._add_unique(candidates, seen, per_group.get(value, [])[:450], self.candidate_limit)
                if len(candidates) >= self.candidate_limit:
                    break

        embedded_history = [
            article_id for article_id in history_article_ids if article_id in self.recommender.article_to_index
        ]
        if embedded_history and len(candidates) < self.candidate_limit:
            profile, _ = self.recommender.build_profile(embedded_history)
            scores = self.recommender.embeddings @ profile
            top_indices = np.argpartition(-scores, min(self.visual_neighbors, len(scores) - 1))[
                : self.visual_neighbors
            ]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
            visual_candidates = [self.recommender.article_ids[int(index)] for index in top_indices]
            self._add_unique(candidates, seen, visual_candidates, self.candidate_limit)

        if len(candidates) < self.candidate_limit:
            self._add_unique(candidates, seen, self.global_popular, self.candidate_limit)

        return candidates

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = values.astype("float32")
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value <= min_value:
            return np.zeros_like(values, dtype="float32")
        return (values - min_value) / (max_value - min_value)

    def score_candidates(self, candidate_ids: list[str], history_article_ids: list[str]) -> np.ndarray:
        candidate_ids = [str(article_id).zfill(10) for article_id in candidate_ids]
        history_article_ids = [str(article_id).zfill(10) for article_id in history_article_ids]
        candidate_pos = {article_id: index for index, article_id in enumerate(candidate_ids)}
        scores = np.zeros(len(candidate_ids), dtype="float32")

        for index, article_id in enumerate(candidate_ids):
            if article_id in self.last_7d_rank:
                scores[index] += 0.70 / np.log2(self.last_7d_rank[article_id] + 2.0)
            if article_id in self.last_30d_rank:
                scores[index] += 0.35 / np.log2(self.last_30d_rank[article_id] + 2.0)
            if article_id in self.article_rank:
                scores[index] += 0.15 / np.log2(self.article_rank[article_id] + 2.0)

        for age, history_article in enumerate(reversed(history_article_ids[-80:])):
            age_weight = 1.0 / (1.0 + age * 0.04)
            for rank, co_article in enumerate(self.co_purchase_index.get(history_article, [])[: self.co_purchase_per_item]):
                position = candidate_pos.get(co_article)
                if position is not None:
                    scores[position] += 1.10 * age_weight / (1.0 + rank / 30.0)

        history_meta = [
            self.article_meta[article_id]
            for article_id in history_article_ids[-60:]
            if article_id in self.article_meta
        ]
        if history_meta:
            recent_group_values: dict[str, set[object]] = {column: set() for column in self.group_columns}
            for meta in history_meta:
                for column in self.group_columns:
                    value = meta.get(column)
                    if pd.notna(value):
                        recent_group_values[column].add(value)
            for index, article_id in enumerate(candidate_ids):
                meta = self.article_meta.get(article_id)
                if not meta:
                    continue
                for column in self.group_columns:
                    if meta.get(column) in recent_group_values[column]:
                        scores[index] += 0.08

        embedded_history = [
            article_id for article_id in history_article_ids if article_id in self.recommender.article_to_index
        ]
        embedded_candidates = [
            self.recommender.article_to_index[article_id]
            for article_id in candidate_ids
            if article_id in self.recommender.article_to_index
        ]
        if embedded_history and embedded_candidates:
            profile, _ = self.recommender.build_profile(embedded_history)
            candidate_indices = np.array(
                [self.recommender.article_to_index[article_id] for article_id in candidate_ids],
                dtype=np.int64,
            )
            visual_scores = self.recommender.embeddings[candidate_indices] @ profile
            scores += 0.35 * self._normalize(visual_scores)

        return self._normalize(scores)


def average_precision_at_k(predictions: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    seen = set()
    for rank, article_id in enumerate(predictions[:k], start=1):
        if article_id in seen:
            continue
        seen.add(article_id)
        if article_id in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def precision_at_k(predictions: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(predictions[:k]) & relevant) / k


def recall_at_k(predictions: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(predictions[:k]) & relevant) / len(relevant)


def hit_rate_at_k(predictions: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(predictions[:k]) & relevant))


def load_transactions() -> pd.DataFrame:
    print(f"Loading transactions: {TRANSACTIONS_PATH}")
    transactions = pd.read_csv(
        TRANSACTIONS_PATH,
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"article_id": str},
    )
    transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    print(f"Transactions loaded: {len(transactions):,}")
    return transactions


def load_customer_profiles(customer_ids: list[str]) -> dict[str, dict]:
    print(f"Loading customer metadata: {CUSTOMERS_PATH}")
    customers = pd.read_csv(CUSTOMERS_PATH)
    customers = customers[customers["customer_id"].isin(customer_ids)].copy()
    profiles: dict[str, dict] = {}
    for row in customers.itertuples(index=False):
        profile = DEFAULT_CUSTOMER_PROFILE.copy()
        for key in profile:
            if hasattr(row, key):
                value = getattr(row, key)
                if pd.notna(value):
                    profile[key] = value
        profiles[row.customer_id] = profile
    return profiles


def build_time_split(
    transactions: pd.DataFrame,
    validation_days: int,
    embedded_articles: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    max_date = transactions["t_dat"].max()
    cutoff = max_date - pd.Timedelta(days=validation_days)
    train = transactions[transactions["t_dat"] < cutoff].copy()
    validation = transactions[transactions["t_dat"] >= cutoff].copy()

    train = train[train["article_id"].isin(embedded_articles)].copy()
    validation = validation[validation["article_id"].isin(embedded_articles)].copy()

    print(f"Max transaction date: {max_date.date()}")
    print(f"Validation cutoff: {cutoff.date()} | validation_days={validation_days}")
    print(f"Train transactions with embeddings: {len(train):,}")
    print(f"Validation transactions with embeddings: {len(validation):,}")
    return train, validation, cutoff


def sample_evaluation_customers(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    sample_customers: int,
    min_history: int,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    print("Building customer histories and validation ground truth...")
    history_counts = train.groupby("customer_id")["article_id"].size()
    validation_counts = validation.groupby("customer_id")["article_id"].size()
    eligible = sorted(
        set(history_counts[history_counts >= min_history].index)
        & set(validation_counts[validation_counts > 0].index)
    )
    if not eligible:
        raise ValueError("No eligible validation customers found. Try lowering --min-history.")

    rng = np.random.default_rng(seed)
    if sample_customers and len(eligible) > sample_customers:
        selected = sorted(rng.choice(eligible, size=sample_customers, replace=False).tolist())
    else:
        selected = eligible

    selected_set = set(selected)
    train_selected = train[train["customer_id"].isin(selected_set)]
    validation_selected = validation[validation["customer_id"].isin(selected_set)]

    histories: dict[str, list[str]] = {}
    for customer_id, values in train_selected.groupby("customer_id")["article_id"]:
        # Preserve recency order and remove duplicates so the profile is based on actual previous products.
        histories[customer_id] = list(dict.fromkeys(values.tail(100).tolist()))

    ground_truth: dict[str, set[str]] = {}
    for customer_id, values in validation_selected.groupby("customer_id")["article_id"]:
        ground_truth[customer_id] = set(values.tolist())

    print(f"Eligible customers: {len(eligible):,}")
    print(f"Selected customers: {len(selected):,}")
    return histories, ground_truth


def override_popularity_with_train(recommender: PerseptronRecommender, train: pd.DataFrame) -> None:
    counts = train["article_id"].value_counts()
    counts = counts[counts.index.isin(recommender.article_to_index)]
    recommender.popular_articles = counts.head(500).index.tolist()
    print(f"Train-only popularity candidates loaded: {len(recommender.popular_articles):,}")


def evaluate_models(
    recommender: PerseptronRecommender,
    histories: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    customer_profiles: dict[str, dict],
    top_k: int,
    precision_k: int,
    hybrid_heuristic_weights: list[float],
    candidate_builder: ImprovedCandidateBuilder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    skipped = 0
    per_customer_rows = []

    for customer_id in tqdm(sorted(ground_truth), desc="Evaluating validation customers"):
        history = histories.get(customer_id, [])
        relevant = ground_truth.get(customer_id, set())
        if not history or not relevant:
            skipped += 1
            continue

        try:
            if candidate_builder is not None:
                candidate_ids = candidate_builder.build(history)
            else:
                candidate_ids = recommender.build_candidate_pool(history)
            if not candidate_ids:
                skipped += 1
                continue
            candidate_set = set(candidate_ids)
            candidate_recall = len(candidate_set & relevant) / len(relevant)

            profile, history_count = recommender.build_profile(history)
            frame = recommender.make_scoring_frame(
                candidate_ids,
                customer_profiles.get(customer_id, DEFAULT_CUSTOMER_PROFILE),
            )

            scores_by_model = {
                "tabular_only": recommender.score_tabular(frame),
                "image_history": recommender.score_image(candidate_ids, profile, history_count),
                "late_fusion": recommender.score_fusion(frame, candidate_ids, profile, history_count),
            }
            if candidate_builder is not None and any(weight > 0 for weight in hybrid_heuristic_weights):
                heuristic_scores = candidate_builder.score_candidates(candidate_ids, history)
                fusion_scores = ImprovedCandidateBuilder._normalize(scores_by_model["late_fusion"])
                for weight in hybrid_heuristic_weights:
                    if weight <= 0:
                        continue
                    scores_by_model[hybrid_model_name(weight, len(hybrid_heuristic_weights))] = (
                        (1.0 - weight) * fusion_scores
                        + weight * heuristic_scores
                    )
                scores_by_model["candidate_heuristic"] = heuristic_scores
        except Exception as exc:
            skipped += 1
            print(f"Skipping customer {customer_id}: {exc}")
            continue

        for model_name, scores in scores_by_model.items():
            order = np.argsort(-scores)[:top_k]
            predictions = [candidate_ids[int(index)] for index in order]

            ap = average_precision_at_k(predictions, relevant, top_k)
            precision = precision_at_k(predictions, relevant, precision_k)
            recall = recall_at_k(predictions, relevant, precision_k)
            hit_rate = hit_rate_at_k(predictions, relevant, top_k)

            prefix = model_name
            metric_sums[(prefix, "map_at_12")] += ap
            metric_sums[(prefix, "precision_at_10")] += precision
            metric_sums[(prefix, "recall_at_10")] += recall
            metric_sums[(prefix, "hit_rate_at_12")] += hit_rate
            metric_sums[(prefix, "candidate_recall")] += candidate_recall
            metric_counts[prefix] += 1

            per_customer_rows.append(
                {
                    "customer_id": customer_id,
                    "model": model_name,
                    "map_at_12": ap,
                    "precision_at_10": precision,
                    "recall_at_10": recall,
                    "hit_rate_at_12": hit_rate,
                    "candidate_recall": candidate_recall,
                    "history_count": len(history),
                    "relevant_count": len(relevant),
                    "candidate_count": len(candidate_ids),
                    "top_12": " ".join(predictions),
                    "relevant_articles": " ".join(sorted(relevant)),
                }
            )

    summary_rows = []
    hybrid_names = sorted(
        name
        for name in metric_counts
        if name == "late_fusion_hybrid" or name.startswith("late_fusion_hybrid_")
    )
    model_order = [
        "tabular_only",
        "image_history",
        "late_fusion",
        *hybrid_names,
        "candidate_heuristic",
    ]
    for model_name in [name for name in model_order if metric_counts[name] > 0]:
        count = metric_counts[model_name]
        summary_rows.append(
            {
                "model": model_name,
                "customers_evaluated": count,
                "map_at_12": metric_sums[(model_name, "map_at_12")] / count,
                "precision_at_10": metric_sums[(model_name, "precision_at_10")] / count,
                "recall_at_10": metric_sums[(model_name, "recall_at_10")] / count,
                "hit_rate_at_12": metric_sums[(model_name, "hit_rate_at_12")] / count,
                "candidate_recall": metric_sums[(model_name, "candidate_recall")] / count,
            }
        )

    print(f"Skipped customers: {skipped:,}")
    return pd.DataFrame(summary_rows), pd.DataFrame(per_customer_rows)


def write_summary_markdown(
    path: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    cutoff: pd.Timestamp,
    per_customer: pd.DataFrame,
) -> None:
    def project_relative(path_value: Path) -> str:
        try:
            return str(path_value.resolve().relative_to(PROJECT_DIR))
        except ValueError:
            return str(path_value)

    def markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in frame.itertuples(index=False):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# Validation Ranking Evaluation",
        "",
        "Bu dosya, final Perseptron modellerinin zamana dayalı validation ayrımı üzerinde top-k öneri kalitesini özetler.",
        "",
        "## Kurulum",
        "",
        f"- Validation penceresi: son {args.validation_days} gün",
        f"- Cutoff tarihi: {cutoff.date()}",
        f"- Örneklenen müşteri sayısı: {args.sample_customers}",
        f"- Candidate limit: {args.candidate_limit}",
        f"- Candidate mode: {args.candidate_mode}",
        f"- Visual neighbors: {args.visual_neighbors}",
        f"- Co-purchase per item: {0 if args.disable_co_purchase else args.co_purchase_per_item}",
        f"- Co-purchase max history/customer: {args.co_purchase_max_history_per_customer}",
        f"- Hybrid heuristic weight(s): {', '.join(str(weight) for weight in get_hybrid_weights(args))}",
        f"- Top-k: {args.top_k}",
        f"- Precision/Recall k: {args.precision_k}",
        "",
        "Cutoff öncesi işlemler müşteri geçmişi, cutoff sonrası işlemler ground truth olarak kullanılmıştır.",
        "Yalnızca EfficientNet embedding evreninde bulunan ürünler değerlendirmeye alınmıştır.",
        "Improved candidate mode; son dönem popülerliği, metadata grupları, co-purchase adayları ve görsel embedding komşularını birlikte kullanır.",
        "",
        "## Sonuçlar",
        "",
        markdown_table(summary),
        "",
        "## Not",
        "",
        "Bu değerlendirme Kaggle public leaderboard'ın birebir karşılığı değildir. Ama proposal'da belirtilen MAP@12, Precision@10 ve Recall@10 gibi ranking metriklerini, final modellerin aynı aday havuzu üzerinde karşılaştırılması için kullanır.",
        "",
        f"Detaylı müşteri-model satırları: `{project_relative(args.output_csv)}`",
        f"Toplam detay satırı: {len(per_customer):,}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    hybrid_weights = get_hybrid_weights(args)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading final recommender and checkpoints...")
    recommender = PerseptronRecommender(device=args.device, candidate_limit=args.candidate_limit)
    print(f"Device: {recommender.device}")

    transactions = load_transactions()
    train, validation, cutoff = build_time_split(
        transactions,
        validation_days=args.validation_days,
        embedded_articles=set(recommender.article_to_index),
    )
    override_popularity_with_train(recommender, train)

    histories, ground_truth = sample_evaluation_customers(
        train,
        validation,
        sample_customers=args.sample_customers,
        min_history=args.min_history,
        seed=args.seed,
    )
    candidate_builder = None
    if args.candidate_mode == "improved":
        target_history_articles = {
            article_id
            for history in histories.values()
            for article_id in history
        }
        candidate_builder = ImprovedCandidateBuilder(
            recommender,
            train,
            candidate_limit=args.candidate_limit,
            visual_neighbors=args.visual_neighbors,
            target_history_articles=target_history_articles,
            co_purchase_per_item=0 if args.disable_co_purchase else args.co_purchase_per_item,
            co_purchase_max_history_per_customer=args.co_purchase_max_history_per_customer,
        )
    customer_profiles = load_customer_profiles(sorted(ground_truth))

    summary, per_customer = evaluate_models(
        recommender,
        histories,
        ground_truth,
        customer_profiles,
        top_k=args.top_k,
        precision_k=args.precision_k,
        hybrid_heuristic_weights=hybrid_weights,
        candidate_builder=candidate_builder,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    per_customer.to_csv(args.output_csv, index=False)
    write_summary_markdown(args.summary_md, summary, args, cutoff, per_customer)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved per-customer metrics: {args.output_csv}")
    print(f"Saved summary: {args.summary_md}")


if __name__ == "__main__":
    main()
