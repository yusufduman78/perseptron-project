from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from .config import MODELS_DIR, REPORTS_DIR, ensure_v2_dirs
from .data import article_image_path, load_core_tables, log, resolve_images_dir
from .features import encode_tabular
from .models import EfficientNetBinaryClassifier
from .models import build_model
from .train import prepare_fold_data


def feature_group(feature: str) -> str:
    if feature in {"FN", "Active", "age", "club_member_status", "fashion_news_frequency"}:
        return "customer_metadata"
    if feature in {"visual_similarity", "visual_history_count"}:
        return "visual_branch"
    return "article_metadata"


def run_tabular_shap(args: argparse.Namespace) -> None:
    checkpoint_path = args.tabular_checkpoint or MODELS_DIR / f"tabular_only_fold{args.fold_id}.pt"
    if not checkpoint_path.exists():
        pd.DataFrame(
            [
                {
                    "feature": "SHAP_NOT_RUN",
                    "feature_group": "missing_checkpoint",
                    "mean_abs_shap": np.nan,
                    "status": f"checkpoint_not_found:{checkpoint_path}",
                }
            ]
        ).to_csv(args.shap_output, index=False)
        log(f"SHAP skipped; checkpoint not found: {checkpoint_path}")
        return

    shap_module = None
    try:
        import shap as shap_module  # type: ignore
    except Exception as exc:
        log(f"SHAP dependency not available; falling back to permutation importance: {exc}")

    prep_args = argparse.Namespace(
        raw_dir=args.raw_dir,
        embeddings_path=args.embeddings_path,
        embedding_ids_path=args.embedding_ids_path,
        folds_csv=args.folds_csv,
        fold_id=args.fold_id,
        validation_days=args.validation_days,
        max_train_positives=args.shap_background_rows,
        max_val_positives=args.shap_explain_rows,
        negatives_per_positive=1,
        seed=args.seed,
    )
    _, arrays = prepare_fold_data(prep_args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    frame = arrays["val_frame"].head(args.shap_explain_rows).copy()
    numeric, categorical = encode_tabular(frame, metadata)
    encoded = np.concatenate([numeric, categorical.astype("float32")], axis=1)
    feature_names = metadata["numeric_features"] + metadata["categorical_features"]
    background = encoded[: min(args.shap_background_rows, len(encoded))]
    explain = encoded[: min(args.shap_explain_rows, len(encoded))]

    model = build_model("tabular_only", metadata, checkpoint["image_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    numeric_dim = len(metadata["numeric_features"])

    def predict_fn(values: np.ndarray) -> np.ndarray:
        numeric_values = torch.tensor(values[:, :numeric_dim], dtype=torch.float32)
        categorical_values = torch.tensor(np.rint(values[:, numeric_dim:]).clip(min=0), dtype=torch.long)
        with torch.no_grad():
            return torch.sigmoid(model(numeric_values, categorical_values)).numpy()

    if shap_module is not None:
        explainer = shap_module.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(explain, nsamples=args.shap_nsamples)
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[0]
        mean_abs = np.abs(shap_array).mean(axis=0)
        rows = [
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "mean_abs_shap": float(value),
                "status": "shap",
            }
            for feature, value in zip(feature_names, mean_abs)
        ]
    else:
        labels = arrays["val_pairs"].head(len(explain))["label"].to_numpy()
        baseline = roc_auc_score(labels, predict_fn(explain)) if len(np.unique(labels)) > 1 else 0.5
        rng = np.random.default_rng(args.seed)
        rows = []
        for index, feature in enumerate(feature_names):
            perturbed = explain.copy()
            perturbed[:, index] = rng.permutation(perturbed[:, index])
            score = roc_auc_score(labels, predict_fn(perturbed)) if len(np.unique(labels)) > 1 else 0.5
            rows.append(
                {
                    "feature": feature,
                    "feature_group": feature_group(feature),
                    "mean_abs_shap": float(max(baseline - score, 0.0)),
                    "status": "permutation_fallback",
                }
            )
    output = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    output.to_csv(args.shap_output, index=False)
    log(f"Saved SHAP summary: {args.shap_output}")


def simple_gradcam_heatmap(model: EfficientNetBinaryClassifier, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    gradients = []
    activations = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.backbone.features[-1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    model.zero_grad(set_to_none=True)
    score = model(image_tensor.to(device).unsqueeze(0))
    score.backward()
    handle_f.remove()
    handle_b.remove()

    acts = activations[0]
    grads = gradients[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / max(cam.max(), 1e-8)
    return cam


def run_gradcam_examples(
    checkpoint_path: Path,
    raw_dir: Path | None,
    images_dir: Path | None,
    output_dir: Path,
    max_examples: int,
    device_name: str | None,
) -> None:
    raw_dir, transactions, _, _ = load_core_tables(raw_dir)
    resolved_images = resolve_images_dir(raw_dir, images_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = EfficientNetBinaryClassifier(train_backbone=checkpoint.get("train_backbone", False))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    transform = model.weights.transforms()
    output_dir.mkdir(parents=True, exist_ok=True)

    article_ids = transactions["article_id"].astype(str).str.zfill(10).drop_duplicates().head(max_examples * 10)
    saved = 0
    for article_id in article_ids:
        path = article_image_path(resolved_images, article_id)
        if not path.exists():
            continue
        original = Image.open(path).convert("RGB")
        tensor = transform(original)
        heatmap = simple_gradcam_heatmap(model, tensor, device)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(original)
        axes[0].set_title(article_id)
        axes[0].axis("off")
        axes[1].imshow(original)
        axes[1].imshow(heatmap, cmap="jet", alpha=0.45, extent=(0, original.width, original.height, 0))
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{article_id}_gradcam.png", dpi=150)
        plt.close(fig)
        saved += 1
        if saved >= max_examples:
            break
    log(f"Saved {saved} Grad-CAM examples to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run proposal v2 SHAP/permutation explainability and Grad-CAM examples.")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--embeddings-path", type=Path, default=None)
    parser.add_argument("--embedding-ids-path", type=Path, default=None)
    parser.add_argument("--folds-csv", type=Path, default=REPORTS_DIR / "proposal_v2_fold_splits.csv")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--tabular-checkpoint", type=Path, default=None)
    parser.add_argument("--cnn-checkpoint", type=Path, default=None)
    parser.add_argument("--shap-output", type=Path, default=REPORTS_DIR / "proposal_v2_shap_summary.csv")
    parser.add_argument("--shap-background-rows", type=int, default=80)
    parser.add_argument("--shap-explain-rows", type=int, default=40)
    parser.add_argument("--shap-nsamples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradcam-dir", type=Path, default=REPORTS_DIR / "gradcam_examples")
    parser.add_argument("--max-gradcam-examples", type=int, default=12)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ensure_v2_dirs()
    run_tabular_shap(args)
    checkpoint = args.cnn_checkpoint or MODELS_DIR / "image_only_effnet_cnn_fold0.pt"
    if checkpoint.exists():
        run_gradcam_examples(checkpoint, args.raw_dir, args.images_dir, args.gradcam_dir, args.max_gradcam_examples, args.device)
    else:
        log(f"Grad-CAM skipped; CNN checkpoint not found: {checkpoint}")


if __name__ == "__main__":
    main()
