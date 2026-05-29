from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
SANDBOX_DIR = PROJECT_DIR / "data" / "processed" / "sandbox"
EMBEDDINGS_DIR = PROJECT_DIR / "data" / "embeddings" / "local_sandbox"

MODEL_DATA_PATH = SANDBOX_DIR / "sandbox_model_data.csv"
TRANSACTIONS_PATH = SANDBOX_DIR / "sandbox_transactions.csv"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "article_image_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "article_image_embedding_ids.csv"
OUTPUT_PATH = SANDBOX_DIR / "visual_similarity_features.csv"


def l2_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def main() -> None:
    print("Loading image embeddings...")
    article_ids = pd.read_csv(IDS_PATH, dtype={"article_id": str})["article_id"].tolist()
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    embeddings = l2_normalize(embeddings)
    article_to_index = {article_id: index for index, article_id in enumerate(article_ids)}
    embedding_dim = embeddings.shape[1]
    print(f"Embeddings: {embeddings.shape}")

    print("Loading transactions and model rows...")
    transactions = pd.read_csv(TRANSACTIONS_PATH, dtype={"article_id": str})
    model_data = pd.read_csv(MODEL_DATA_PATH, dtype={"article_id": str})

    transactions = transactions[transactions["article_id"].isin(article_to_index)].copy()
    model_data = model_data[model_data["article_id"].isin(article_to_index)].copy()

    print("Building customer visual profile sums...")
    customer_ids = sorted(transactions["customer_id"].unique())
    customer_to_index = {customer_id: index for index, customer_id in enumerate(customer_ids)}
    profile_sums = np.zeros((len(customer_ids), embedding_dim), dtype="float32")
    profile_counts = np.zeros(len(customer_ids), dtype="int32")

    for row in tqdm(transactions[["customer_id", "article_id"]].itertuples(index=False), total=len(transactions), desc="Profiles"):
        customer_index = customer_to_index[row.customer_id]
        article_index = article_to_index[row.article_id]
        profile_sums[customer_index] += embeddings[article_index]
        profile_counts[customer_index] += 1

    print("Scoring model rows with leave-candidate-out visual similarity...")
    rows = []
    for row in tqdm(model_data[["customer_id", "article_id", "label"]].itertuples(index=False), total=len(model_data), desc="Visual features"):
        customer_index = customer_to_index.get(row.customer_id)
        article_index = article_to_index.get(row.article_id)

        if customer_index is None or article_index is None:
            similarity = 0.0
            history_count = 0
        else:
            profile_sum = profile_sums[customer_index].copy()
            history_count = int(profile_counts[customer_index])

            # Avoid leaking the positive target item into the user's visual profile.
            if history_count > 0:
                profile_sum -= embeddings[article_index]
                history_count -= 1

            if history_count <= 0:
                similarity = 0.0
            else:
                profile_vector = profile_sum / max(history_count, 1)
                profile_norm = np.linalg.norm(profile_vector)
                if profile_norm == 0:
                    similarity = 0.0
                else:
                    profile_vector = profile_vector / profile_norm
                    similarity = float(np.dot(profile_vector, embeddings[article_index]))

        rows.append(
            {
                "customer_id": row.customer_id,
                "article_id": row.article_id,
                "label": int(row.label),
                "visual_similarity": similarity,
                "visual_history_count": history_count,
            }
        )

    features = pd.DataFrame(rows)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved visual features: {OUTPUT_PATH}")
    print(features[["visual_similarity", "visual_history_count"]].describe().to_string())


if __name__ == "__main__":
    main()


