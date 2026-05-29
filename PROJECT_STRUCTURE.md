# Perseptron Project Structure

This folder is organized around the final H&M multimodal recommendation project.

## Main Folders

- `data/raw/`
  - Original H&M CSV files: `articles.csv`, `customers.csv`, `transactions_train.csv`.
- `data/images/hm_images/`
  - Full H&M product image folder.
- `data/embeddings/final_kaggle/`
  - Final 105,100-row Kaggle embedding cache used by the full-training notebooks.
  - `article_image_embeddings_popular.npy`
  - `article_image_embedding_ids_popular.csv`
- `data/embeddings/local_sandbox/`
  - Older 52,833-row local/sandbox embedding cache used by local phase scripts.
- `data/processed/sandbox/`
  - Local sandbox/intermediate CSV files from earlier phase experiments.
- `models/final_kaggle/`
  - Final Kaggle-trained model checkpoints and result CSV files.
- `models/local_experiments/`
  - Earlier local/sandbox model checkpoints.
- `notebooks/kaggle/`
  - Final Kaggle notebooks for tabular-only, image-history, and late-fusion training.
- `notebooks/archive/`
  - Older/generated notebook artifacts kept for reference.
- `reports/`
  - Result tables, final report draft, and explainability outputs.
- `src/local_phases/`
  - Local phase scripts used during development and ablation experiments.
- `src/kaggle/`
  - Kaggle pipeline source script.
- `src/explainability/`
  - Visual similarity and tabular permutation-importance scripts.
- `src/utils/`
  - Helper scripts.
- `docs/`
  - Proposal and project documents.
- `demo/`
  - FastAPI backend, React/Vite frontend, and terminal inference tester for the live recommendation demo.

## Main Report

- `reports/FINAL_REPORT.md`

## Final Kaggle Results

The consolidated full-training results are in:

- `reports/kaggle_full_training_results.csv`

Final checkpoints are in:

- `models/final_kaggle/tabular_only_streaming_full.pt`
- `models/final_kaggle/image_history_streaming_full.pt`
- `models/final_kaggle/multimodal_fusion_streaming_full.pt`
