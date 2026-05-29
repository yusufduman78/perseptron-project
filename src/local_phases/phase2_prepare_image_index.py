import os
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_DIR / "data" / "images" / "hm_images"
SANDBOX_DIR = PROJECT_DIR / "data" / "processed" / "sandbox"
OUTPUT_PATH = SANDBOX_DIR / "image_index.csv"


def main() -> None:
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    rows = []
    for image_path in IMAGES_DIR.rglob("*.jpg"):
        article_id = image_path.stem.zfill(10)
        rows.append(
            {
                "article_id": article_id,
                "image_path": str(image_path),
                "image_size_bytes": os.path.getsize(image_path),
            }
        )

    image_index = pd.DataFrame(rows).drop_duplicates("article_id")
    image_index = image_index.sort_values("article_id").reset_index(drop=True)
    image_index.to_csv(OUTPUT_PATH, index=False)

    articles_path = SANDBOX_DIR / "sandbox_articles.csv"
    if articles_path.exists():
        articles = pd.read_csv(articles_path, dtype={"article_id": str})
        sandbox_article_count = articles["article_id"].nunique()
        matched = articles.merge(image_index[["article_id"]], on="article_id", how="inner")
        matched_count = matched["article_id"].nunique()
        coverage = matched_count / sandbox_article_count if sandbox_article_count else 0.0
        print(f"Sandbox article coverage: {matched_count:,}/{sandbox_article_count:,} ({coverage:.2%})")

    print(f"Indexed images: {len(image_index):,}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


