# Kaggle v2 Runbook

Bu dosya, proposal-aligned v2 deneylerini Kaggle'da hangi sirayla
calistiracagini anlatir.

## 0. Kaggle Inputlari

Kaggle notebook inputlarinda sunlar olmali:

- H&M competition/raw dataset: `transactions_train.csv`, `customers.csv`, `articles.csv`, `images/`
- EfficientNet embedding cache dataset:
  - `article_image_embeddings_popular.npy`
  - `article_image_embedding_ids_popular.csv`
- Bu GitHub branchinin kodlari. En pratik yol repo zip'ini Kaggle Dataset olarak eklemek veya notebooka dosyalari yuklemek.

## 1. Smoke Training

Notebook:

`notebooks/kaggle/proposal_v2_training.ipynb`

Ayarlar:

```python
FAST_RUN = True
FOLD_ID = 0
RUN_SPLITS = True
RUN_MLP_MODELS = True
RUN_CNN_BASELINE = True
```

Beklenen ciktilar:

- `reports/proposal_v2/proposal_v2_fold_splits.csv`
- `models/proposal_v2/tabular_only_fold0.pt`
- `models/proposal_v2/image_history_fold0.pt`
- `models/proposal_v2/late_fusion_fold0.pt`
- `models/proposal_v2/image_only_effnet_cnn_fold0.pt`

## 2. Smoke Ranking

Notebook:

`notebooks/kaggle/proposal_v2_ranking.ipynb`

Ayarlar:

```python
FAST_RUN = True
FOLD_ID = 0
```

Beklenen ciktilar:

- `reports/proposal_v2/proposal_v2_ranking_metrics.csv`
- `reports/proposal_v2/proposal_v2_cv_summary.md`

## 3. Full Fold Kosulari

Training notebookunda:

```python
FAST_RUN = False
FOLD_ID = 0  # sonra 1, 2, 3, 4
RUN_SPLITS = FOLD_ID == 0
RUN_MLP_MODELS = True
RUN_CNN_BASELINE = FOLD_ID == 0
```

Her fold bittikten sonra ranking notebookunu ayni `FOLD_ID` ile calistir.

## 4. Explainability

Notebook:

`notebooks/kaggle/proposal_v2_explainability.ipynb`

Bu notebook CNN baseline checkpointinden Grad-CAM ornekleri uretir ve tabular
checkpoint uzerinden SHAP summary olusturur. Kaggle ortaminda `shap` yoksa ayni
dosyada permutation fallback uretilir; raporda bu durum acikca belirtilir.

## 5. Indirilecek Dosyalar

Kaggle'dan su klasorleri indir:

- `reports/proposal_v2/`
- `models/proposal_v2/`

Sonra yerelde ayni pathlere koyacagiz ve rapor/demo v2'yi final sonuclarla
guncelleyecegiz.
