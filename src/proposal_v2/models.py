from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def embedding_dim(size: int) -> int:
    return min(50, max(4, int(size**0.25 * 8)))


class TabularOnlyMLP(nn.Module):
    def __init__(self, numeric_dim: int, category_sizes: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim(size)) for size in category_sizes]
        )
        cat_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
        self.net = nn.Sequential(
            nn.Linear(numeric_dim + cat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [emb(categorical[:, idx]) for idx, emb in enumerate(self.embeddings)]
        x = torch.cat([numeric, *embedded], dim=1)
        return self.net(x).squeeze(1)


class ImageHistoryMLP(nn.Module):
    def __init__(self, image_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim * 2 + 2, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        article_embedding: torch.Tensor,
        profile_embedding: torch.Tensor,
        visual_similarity: torch.Tensor,
        visual_history_count: torch.Tensor,
    ) -> torch.Tensor:
        visual_extra = torch.stack([visual_similarity, visual_history_count], dim=1)
        x = torch.cat([article_embedding, profile_embedding, visual_extra], dim=1)
        return self.net(x).squeeze(1)


class MultimodalLateFusion(nn.Module):
    def __init__(self, numeric_dim: int, category_sizes: list[int], image_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim(size)) for size in category_sizes]
        )
        cat_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
        self.tabular_branch = nn.Sequential(
            nn.Linear(numeric_dim + cat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.visual_branch = nn.Sequential(
            nn.Linear(image_dim * 2 + 2, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        article_embedding: torch.Tensor,
        profile_embedding: torch.Tensor,
        visual_similarity: torch.Tensor,
        visual_history_count: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [emb(categorical[:, idx]) for idx, emb in enumerate(self.embeddings)]
        tabular = self.tabular_branch(torch.cat([numeric, *embedded], dim=1))
        visual_extra = torch.stack([visual_similarity, visual_history_count], dim=1)
        visual = self.visual_branch(torch.cat([article_embedding, profile_embedding, visual_extra], dim=1))
        return self.fusion_head(torch.cat([tabular, visual], dim=1)).squeeze(1)


class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self, train_backbone: bool = False) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.weights = weights
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 1))
        if not train_backbone:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).squeeze(1)


def build_model(model_name: str, metadata: dict, image_dim: int) -> nn.Module:
    category_sizes = [len(metadata["category_maps"][column]) for column in metadata["categorical_features"]]
    numeric_dim = len(metadata["numeric_features"])
    if model_name == "tabular_only":
        return TabularOnlyMLP(numeric_dim, category_sizes)
    if model_name == "image_history":
        return ImageHistoryMLP(image_dim)
    if model_name == "late_fusion":
        return MultimodalLateFusion(numeric_dim, category_sizes, image_dim)
    raise ValueError(f"Unknown model: {model_name}")
