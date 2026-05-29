from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
EMBEDDING_DIR = PROJECT_DIR / "data" / "embeddings" / "final_kaggle"
IMAGE_DIR = PROJECT_DIR / "data" / "images" / "hm_images"
REPORTS_DIR = PROJECT_DIR / "reports"

TRANSACTIONS_PATH = RAW_DIR / "transactions_train.csv"
ARTICLES_PATH = RAW_DIR / "articles.csv"
EMBEDDINGS_PATH = EMBEDDING_DIR / "article_image_embeddings_popular.npy"
EMBEDDING_IDS_PATH = EMBEDDING_DIR / "article_image_embedding_ids_popular.csv"

DEFAULT_CSV_OUTPUT = REPORTS_DIR / "visual_similarity_examples.csv"
DEFAULT_HTML_OUTPUT = REPORTS_DIR / "visual_similarity_examples.html"


ARTICLE_META_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "index_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]


def log(message: str) -> None:
    print(message, flush=True)


def normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = values.astype("float32", copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def article_image_path(article_id: str) -> Path:
    padded = str(article_id).zfill(10)
    return IMAGE_DIR / padded[:3] / f"{padded}.jpg"


def image_src(article_id: str) -> str:
    path = article_image_path(article_id)
    if path.exists():
        return path.as_uri()
    return ""


def load_embedding_index() -> tuple[list[str], dict[str, int]]:
    log(f"Loading embedding ids: {EMBEDDING_IDS_PATH}")
    article_ids = (
        pd.read_csv(EMBEDDING_IDS_PATH, dtype={"article_id": str})["article_id"]
        .astype(str)
        .str.zfill(10)
        .tolist()
    )
    return article_ids, {article_id: index for index, article_id in enumerate(article_ids)}


def load_articles() -> pd.DataFrame:
    log(f"Loading article metadata: {ARTICLES_PATH}")
    articles = pd.read_csv(
        ARTICLES_PATH,
        dtype={"article_id": str},
        usecols=lambda column: column in ARTICLE_META_COLUMNS,
    )
    articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)
    return articles


def load_transactions(article_to_index: dict[str, int], max_transactions: int | None) -> pd.DataFrame:
    nrows = None if max_transactions is None else max_transactions
    if nrows is None:
        log(f"Loading all transactions: {TRANSACTIONS_PATH}")
    else:
        log(f"Loading first {nrows:,} transactions for fast explainability examples.")

    transactions = pd.read_csv(
        TRANSACTIONS_PATH,
        usecols=["t_dat", "customer_id", "article_id"],
        dtype={"article_id": str},
        nrows=nrows,
    )
    transactions["article_id"] = transactions["article_id"].astype(str).str.zfill(10)
    transactions = transactions[transactions["article_id"].isin(article_to_index)].copy()
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    log(f"Embedded transactions available: {len(transactions):,}")
    return transactions


def load_recommendation_candidates(path: Path | None, sample_size: int) -> pd.DataFrame | None:
    if path is None:
        return None

    log(f"Loading recommendation candidates: {path}")
    recommendations = pd.read_csv(path)
    if not {"customer_id", "prediction"}.issubset(recommendations.columns):
        raise ValueError("Recommendation CSV must contain customer_id and prediction columns.")

    recommendations = recommendations.head(sample_size).copy()
    recommendations["candidate_article_id"] = (
        recommendations["prediction"].fillna("").astype(str).str.split().str[0].astype(str).str.zfill(10)
    )
    return recommendations[["customer_id", "candidate_article_id"]]


def choose_examples_from_history(
    transactions: pd.DataFrame,
    article_to_index: dict[str, int],
    sample_size: int,
    min_history: int,
) -> pd.DataFrame:
    log("Choosing example customers from purchase history.")
    rows = []
    sorted_transactions = transactions.sort_values(["customer_id", "t_dat"])

    for customer_id, group in sorted_transactions.groupby("customer_id", sort=False):
        unique_articles = list(dict.fromkeys(group["article_id"].tolist()))
        if len(unique_articles) < min_history + 1:
            continue
        candidate = unique_articles[-1]
        if candidate not in article_to_index:
            continue
        rows.append({"customer_id": customer_id, "candidate_article_id": candidate})
        if len(rows) >= sample_size:
            break

    examples = pd.DataFrame(rows)
    log(f"Selected customers: {len(examples):,}")
    return examples


def build_explanations(
    examples: pd.DataFrame,
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    embeddings: np.ndarray,
    article_to_index: dict[str, int],
    top_k: int,
) -> pd.DataFrame:
    log("Building visual similarity explanations.")
    article_meta = articles.set_index("article_id")
    history_by_customer = (
        transactions.sort_values(["customer_id", "t_dat"])
        .groupby("customer_id")["article_id"]
        .apply(lambda values: list(dict.fromkeys(values.tolist())))
        .to_dict()
    )

    output_rows = []
    for example_number, row in enumerate(examples.itertuples(index=False), start=1):
        customer_id = row.customer_id
        candidate_id = str(row.candidate_article_id).zfill(10)
        history_ids = [
            article_id
            for article_id in history_by_customer.get(customer_id, [])
            if article_id != candidate_id and article_id in article_to_index
        ]

        if not history_ids or candidate_id not in article_to_index:
            continue

        candidate_vector = normalize_rows(embeddings[[article_to_index[candidate_id]]])[0]
        history_indices = [article_to_index[article_id] for article_id in history_ids]
        history_vectors = normalize_rows(embeddings[history_indices])
        similarities = history_vectors @ candidate_vector
        top_positions = np.argsort(-similarities)[:top_k]

        candidate_meta = article_meta.loc[candidate_id] if candidate_id in article_meta.index else {}
        for rank, position in enumerate(top_positions, start=1):
            history_id = history_ids[int(position)]
            history_meta = article_meta.loc[history_id] if history_id in article_meta.index else {}
            output_rows.append(
                {
                    "example_id": example_number,
                    "customer_id": customer_id,
                    "candidate_article_id": candidate_id,
                    "candidate_prod_name": candidate_meta.get("prod_name", ""),
                    "candidate_product_type": candidate_meta.get("product_type_name", ""),
                    "candidate_colour": candidate_meta.get("colour_group_name", ""),
                    "candidate_image_path": str(article_image_path(candidate_id)),
                    "similar_history_rank": rank,
                    "history_article_id": history_id,
                    "history_prod_name": history_meta.get("prod_name", ""),
                    "history_product_type": history_meta.get("product_type_name", ""),
                    "history_colour": history_meta.get("colour_group_name", ""),
                    "history_image_path": str(article_image_path(history_id)),
                    "cosine_similarity": float(similarities[int(position)]),
                    "explanation": (
                        f"Candidate {candidate_id} is visually similar to customer history item "
                        f"{history_id} with cosine similarity {similarities[int(position)]:.4f}."
                    ),
                }
            )

    explanations = pd.DataFrame(output_rows)
    log(f"Explanation rows: {len(explanations):,}")
    return explanations


def html_cell_image(article_id: str, title: str, subtitle: str) -> str:
    src = image_src(article_id)
    safe_title = html.escape(title or article_id)
    safe_subtitle = html.escape(subtitle or "")
    safe_article_id = html.escape(article_id)
    if src:
        image_markup = f'<img src="{src}" alt="{safe_article_id}">'
    else:
        image_markup = '<div class="missing-image">No image</div>'
    return f"""
    <div class="product">
        {image_markup}
        <div class="article-id">{safe_article_id}</div>
        <div class="title">{safe_title}</div>
        <div class="subtitle">{safe_subtitle}</div>
    </div>
    """


def write_html_report(explanations: pd.DataFrame, output_path: Path) -> None:
    log(f"Writing HTML report: {output_path}")
    sections = []
    for example_id, group in explanations.groupby("example_id", sort=True):
        first = group.iloc[0]
        candidate_card = html_cell_image(
            first["candidate_article_id"],
            first["candidate_prod_name"],
            f'{first["candidate_product_type"]} / {first["candidate_colour"]}',
        )
        history_cards = []
        for history in group.itertuples(index=False):
            history_cards.append(
                f"""
                <div class="match">
                    {html_cell_image(
                        history.history_article_id,
                        history.history_prod_name,
                        f"{history.history_product_type} / {history.history_colour}",
                    )}
                    <div class="score">Similarity: {history.cosine_similarity:.4f}</div>
                </div>
                """
            )

        sections.append(
            f"""
            <section class="example">
                <h2>Example {int(example_id)}</h2>
                <p class="customer">Customer: {html.escape(first["customer_id"])}</p>
                <div class="layout">
                    <div>
                        <h3>Candidate / recommended item</h3>
                        {candidate_card}
                    </div>
                    <div>
                        <h3>Most visually similar history items</h3>
                        <div class="history-grid">
                            {''.join(history_cards)}
                        </div>
                    </div>
                </div>
            </section>
            """
        )

    page = f"""
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Perseptron Visual Similarity Explanations</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    header {{
      padding: 28px 36px;
      background: #ffffff;
      border-bottom: 1px solid #d9dee5;
    }}
    h1, h2, h3, p {{
      margin-top: 0;
    }}
    .example {{
      padding: 28px 36px;
      border-bottom: 1px solid #d9dee5;
      background: #ffffff;
    }}
    .customer {{
      color: #536171;
      font-size: 13px;
      word-break: break-all;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 220px 1fr;
      gap: 28px;
      align-items: start;
    }}
    .history-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 16px;
    }}
    .product {{
      border: 1px solid #d9dee5;
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }}
    .product img, .missing-image {{
      width: 100%;
      aspect-ratio: 3 / 4;
      object-fit: cover;
      background: #eef1f5;
      border-radius: 4px;
      display: grid;
      place-items: center;
      color: #7b8794;
      font-size: 13px;
    }}
    .article-id {{
      margin-top: 8px;
      font-family: Consolas, monospace;
      font-size: 12px;
      color: #52606d;
    }}
    .title {{
      margin-top: 4px;
      font-weight: 700;
      font-size: 14px;
    }}
    .subtitle {{
      margin-top: 3px;
      color: #616e7c;
      font-size: 12px;
    }}
    .score {{
      margin-top: 6px;
      font-size: 13px;
      font-weight: 700;
    }}
    @media (max-width: 760px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Visual Similarity Explanations</h1>
    <p>Bu rapor, aday ürünün müşterinin geçmişindeki hangi ürünlere görsel olarak benzediğini gösterir.</p>
  </header>
  {''.join(sections)}
</body>
</html>
"""
    output_path.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual similarity explanations from H&M image embeddings and customer history."
    )
    parser.add_argument("--sample-size", type=int, default=8, help="Number of customers/examples to explain.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of similar history items per candidate.")
    parser.add_argument("--min-history", type=int, default=3, help="Minimum previous unique purchases per customer.")
    parser.add_argument(
        "--max-transactions",
        type=int,
        default=2_000_000,
        help="Read only the first N transactions for quick local examples. Ignored when --full-transactions is set.",
    )
    parser.add_argument(
        "--full-transactions",
        action="store_true",
        help="Use all transactions. Slower, but closer to the final training setting.",
    )
    parser.add_argument(
        "--recommendations-csv",
        type=Path,
        default=None,
        help="Optional submission-like CSV with customer_id and prediction columns. First prediction is explained.",
    )
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--html-output", type=Path, default=DEFAULT_HTML_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    article_ids, article_to_index = load_embedding_index()
    log(f"Embedding article ids: {len(article_ids):,}")

    log(f"Loading embeddings with memory map: {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    log(f"Embedding matrix shape: {embeddings.shape}")

    articles = load_articles()
    max_transactions = None if args.full_transactions else args.max_transactions
    transactions = load_transactions(article_to_index, max_transactions=max_transactions)

    examples = load_recommendation_candidates(args.recommendations_csv, args.sample_size)
    if examples is None:
        examples = choose_examples_from_history(
            transactions,
            article_to_index,
            sample_size=args.sample_size,
            min_history=args.min_history,
        )
    else:
        examples = examples[examples["candidate_article_id"].isin(article_to_index)].copy()
        log(f"Recommendation examples with embeddings: {len(examples):,}")

    explanations = build_explanations(
        examples,
        transactions,
        articles,
        embeddings,
        article_to_index,
        top_k=args.top_k,
    )

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    explanations.to_csv(args.csv_output, index=False)
    log(f"Saved CSV explanations: {args.csv_output}")

    write_html_report(explanations, args.html_output)
    log("Visual similarity explainability step complete.")


if __name__ == "__main__":
    main()
