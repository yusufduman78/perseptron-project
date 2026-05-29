import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
SANDBOX_DIR = PROJECT_DIR / "data" / "processed" / "sandbox"
EMBEDDINGS_DIR = PROJECT_DIR / "data" / "embeddings" / "local_sandbox"
os.environ.setdefault("TORCH_HOME", str(PROJECT_DIR / ".torch_cache"))

IMAGE_INDEX_PATH = SANDBOX_DIR / "image_index.csv"
SANDBOX_ARTICLES_PATH = SANDBOX_DIR / "sandbox_articles.csv"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "article_image_embeddings.npy"
IDS_PATH = EMBEDDINGS_DIR / "article_image_embedding_ids.csv"

BATCH_SIZE = int(os.getenv("PHASE3_EMBED_BATCH_SIZE", "256"))
IMAGE_SIZE = int(os.getenv("PHASE3_EMBED_IMAGE_SIZE", "160"))


class ArticleImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return row["article_id"], self.transform(image)


class EfficientNetEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def collate_batch(batch):
    article_ids, images = zip(*batch)
    return list(article_ids), torch.stack(images)


def main() -> None:
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    image_index = pd.read_csv(IMAGE_INDEX_PATH, dtype={"article_id": str})
    sandbox_articles = pd.read_csv(SANDBOX_ARTICLES_PATH, dtype={"article_id": str})
    frame = sandbox_articles[["article_id"]].merge(image_index, on="article_id", how="inner")
    frame = frame[frame["image_path"].map(lambda path: Path(path).exists())].copy()
    frame = frame.drop_duplicates("article_id").sort_values("article_id").reset_index(drop=True)

    print(f"Articles with images to embed: {len(frame):,}")
    print(f"Batch size: {BATCH_SIZE}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = ArticleImageDataset(frame, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_batch,
    )

    print(f"Embedding device: {device}")
    model = EfficientNetEmbeddingModel().to(device)
    model.eval()

    all_ids = []
    all_embeddings = []
    with torch.no_grad():
        for article_ids, images in tqdm(loader, desc="Extracting image embeddings"):
            images = images.to(device, non_blocking=True)
            embeddings = model(images).cpu().numpy().astype("float32")
            all_ids.extend(article_ids)
            all_embeddings.append(embeddings)

    embedding_matrix = np.concatenate(all_embeddings, axis=0)
    np.save(EMBEDDINGS_PATH, embedding_matrix)
    pd.DataFrame({"article_id": all_ids}).to_csv(IDS_PATH, index=False)

    print(f"Embedding shape: {embedding_matrix.shape}")
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Saved ids: {IDS_PATH}")


if __name__ == "__main__":
    main()


