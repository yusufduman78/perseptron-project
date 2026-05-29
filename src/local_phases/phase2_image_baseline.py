import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
SANDBOX_DIR = PROJECT_DIR / "data" / "processed" / "sandbox"
MODELS_DIR = PROJECT_DIR / "models" / "local_experiments"
REPORTS_DIR = PROJECT_DIR / "reports"
os.environ.setdefault("TORCH_HOME", str(PROJECT_DIR / ".torch_cache"))

MODEL_DATA_PATH = SANDBOX_DIR / "sandbox_model_data.csv"
IMAGE_INDEX_PATH = SANDBOX_DIR / "image_index.csv"
MODEL_OUTPUT_PATH = MODELS_DIR / "image_effnet_b0.pt"
RESULTS_PATH = REPORTS_DIR / "phase2_baseline_results.csv"

RANDOM_SEED = 42
BATCH_SIZE = int(os.getenv("PHASE2_IMAGE_BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("PHASE2_IMAGE_EPOCHS", "2"))
MAX_ROWS = int(os.getenv("PHASE2_IMAGE_MAX_ROWS", "50000"))
LEARNING_RATE = float(os.getenv("PHASE2_IMAGE_LR", "0.001"))
VALIDATION_SIZE = 0.2
IMAGE_SIZE = int(os.getenv("PHASE2_IMAGE_SIZE", "160"))


class ImagePurchaseDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label


class EfficientNetBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features

        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).squeeze(1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset() -> pd.DataFrame:
    print("Loading model rows and image index...")
    model_data = pd.read_csv(MODEL_DATA_PATH, dtype={"article_id": str})
    image_index = pd.read_csv(IMAGE_INDEX_PATH, dtype={"article_id": str})

    data = model_data.merge(image_index[["article_id", "image_path"]], on="article_id", how="inner")
    data = data[data["image_path"].map(lambda p: Path(p).exists())].copy()
    print(f"Rows with image: {len(data):,}/{len(model_data):,} ({len(data) / len(model_data):.2%})")

    if MAX_ROWS and len(data) > MAX_ROWS:
        positive = data[data["label"] == 1]
        negative = data[data["label"] == 0]
        pos_target = min(len(positive), MAX_ROWS // 3)
        neg_target = min(len(negative), MAX_ROWS - pos_target)
        data = pd.concat(
            [
                positive.sample(pos_target, random_state=RANDOM_SEED),
                negative.sample(neg_target, random_state=RANDOM_SEED),
            ],
            ignore_index=True,
        )
        data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"Local image baseline sample: {len(data):,} rows")

    return data


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)
        for images, batch_labels in progress:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.numpy())

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels)
    y_pred = (y_prob >= 0.5).astype("int64")
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def append_results(row: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    data = load_dataset()
    print("Label distribution:")
    print(data["label"].value_counts().sort_index().to_string())

    train_frame, val_frame = train_test_split(
        data,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=data["label"],
    )

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ImagePurchaseDataset(train_frame, transform)
    val_dataset = ImagePurchaseDataset(val_frame, transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = EfficientNetBaseline().to(device)
    positive_count = float(train_frame["label"].sum())
    negative_count = float(len(train_frame) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_auc = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        mean_loss = float(np.mean(losses))
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"loss={mean_loss:.4f} | "
            f"val_auc={metrics['auc_roc']:.4f} | "
            f"val_acc={metrics['accuracy']:.4f}"
        )

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "validation_metrics": metrics,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "validation_size": VALIDATION_SIZE,
                    "random_seed": RANDOM_SEED,
                    "max_rows": MAX_ROWS,
                    "image_size": IMAGE_SIZE,
                },
            }

    torch.save(best_state, MODEL_OUTPUT_PATH)
    print(f"Saved best model: {MODEL_OUTPUT_PATH}")

    append_results(
        {
            "phase": "phase2",
            "model": "image_effnet_b0",
            "rows": len(data),
            "train_rows": len(train_frame),
            "validation_rows": len(val_frame),
            "auc_roc": round(best_state["validation_metrics"]["auc_roc"], 6),
            "accuracy": round(best_state["validation_metrics"]["accuracy"], 6),
            "artifact": str(MODEL_OUTPUT_PATH),
        }
    )
    print(f"Updated report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()


