from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender import DEFAULT_CUSTOMER_PROFILE, PerseptronRecommender, recommendations_to_dict


DEFAULT_HISTORY = [
    "0108775015",
    "0108775044",
    "0521269001",
    "0666448006",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test live recommendations from the three final Perseptron models.")
    parser.add_argument(
        "--history",
        nargs="*",
        default=DEFAULT_HISTORY,
        help="Article ids representing previous purchases.",
    )
    parser.add_argument("--age", type=int, default=DEFAULT_CUSTOMER_PROFILE["age"])
    parser.add_argument("--fn", type=int, default=DEFAULT_CUSTOMER_PROFILE["FN"])
    parser.add_argument("--active", type=int, default=DEFAULT_CUSTOMER_PROFILE["Active"])
    parser.add_argument("--club-member-status", default=DEFAULT_CUSTOMER_PROFILE["club_member_status"])
    parser.add_argument("--fashion-news-frequency", default=DEFAULT_CUSTOMER_PROFILE["fashion_news_frequency"])
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    customer_profile = {
        "FN": args.fn,
        "Active": args.active,
        "age": args.age,
        "club_member_status": args.club_member_status,
        "fashion_news_frequency": args.fashion_news_frequency,
    }

    print("Loading recommender and model checkpoints...")
    recommender = PerseptronRecommender(candidate_limit=args.candidate_limit)
    print(f"Device: {recommender.device}")
    print("History article ids:", ", ".join(args.history))
    print("Customer profile:", customer_profile)

    recommendations = recommender.recommend(
        history_article_ids=args.history,
        customer_profile=customer_profile,
        top_k=args.top_k,
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(recommendations_to_dict(recommendations), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON recommendations: {args.json_output}")

    for model_name, items in recommendations.items():
        print(f"\n=== {model_name} ===")
        for rank, item in enumerate(items, start=1):
            print(
                f"{rank:02d}. {item.article_id} | score={item.score:.4f} | "
                f"{item.prod_name} | {item.product_type_name} | {item.colour_group_name}"
            )


if __name__ == "__main__":
    main()
