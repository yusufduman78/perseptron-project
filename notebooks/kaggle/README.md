# Kaggle Notebook Sirasi

Notebooklar ayri ayri calistirilir. Her notebook sonunda `Save Version` yap,
sonraki notebookta onceki notebook outputunu `Add Data` ile ekle.

Tum notebooklar H&M raw data inputuna ihtiyac duyar:

- `transactions_train.csv`
- `customers.csv`
- `articles.csv`
- `images/`

`02`, `03`, `05`, `06` ayrica embedding inputuna ihtiyac duyar:

- `article_image_embeddings_popular.npy`
- `article_image_embedding_ids_popular.csv`

## 1. Fold Hazirla

Calistir:

- `proposal_v2_00_folds.ipynb`

Urettigi dosyalar:

- `reports/proposal_v2/proposal_v2_fold_splits.csv`
- `reports/proposal_v2/proposal_v2_fold_protocol_summary.json`

Sonraki notebooklara aktar:

- `01_tabular_only`
- `02_image_history`
- `03_late_fusion`
- `04_image_only_cnn`
- `05_ranking_map12`
- `06_explainability`

## 2. Tabular Model

Calistir:

- `proposal_v2_01_tabular_only.ipynb`

Gerekli onceki output:

- `00_folds` outputu

Urettigi dosyalar:

- `models/proposal_v2/tabular_only_fold0.pt`
- `reports/proposal_v2/proposal_v2_classification_metrics.csv`

Sonraki notebooklara aktar:

- `05_ranking_map12`
- `06_explainability`

Fold degistirirsen dosya adi su sekilde olur:

- `models/proposal_v2/tabular_only_fold{FOLD_ID}.pt`

## 3. Image-History Model

Calistir:

- `proposal_v2_02_image_history.ipynb`

Gerekli onceki output:

- `00_folds` outputu

Urettigi dosyalar:

- `models/proposal_v2/image_history_fold0.pt`
- `reports/proposal_v2/proposal_v2_classification_metrics.csv`

Sonraki notebooka aktar:

- `05_ranking_map12`

Fold degistirirsen dosya adi su sekilde olur:

- `models/proposal_v2/image_history_fold{FOLD_ID}.pt`

## 4. Late-Fusion Model

Calistir:

- `proposal_v2_03_late_fusion.ipynb`

Gerekli onceki output:

- `00_folds` outputu

Urettigi dosyalar:

- `models/proposal_v2/late_fusion_fold0.pt`
- `reports/proposal_v2/proposal_v2_classification_metrics.csv`

Sonraki notebooka aktar:

- `05_ranking_map12`

Fold degistirirsen dosya adi su sekilde olur:

- `models/proposal_v2/late_fusion_fold{FOLD_ID}.pt`

## 5. Image-Only CNN

Calistir:

- `proposal_v2_04_image_only_cnn.ipynb`

Gerekli onceki output:

- `00_folds` outputu

Urettigi dosyalar:

- `models/proposal_v2/image_only_effnet_cnn_fold0.pt`
- `reports/proposal_v2/proposal_v2_cnn_metrics.csv`

Sonraki notebooka aktar:

- `06_explainability`

Fold degistirirsen dosya adi su sekilde olur:

- `models/proposal_v2/image_only_effnet_cnn_fold{FOLD_ID}.pt`

## 6. Ranking / MAP@12

Calistir:

- `proposal_v2_05_ranking_map12.ipynb`

Gerekli onceki outputlar:

- `00_folds` outputu
- `01_tabular_only` outputu
- `02_image_history` outputu
- `03_late_fusion` outputu

Urettigi dosyalar:

- `reports/proposal_v2/proposal_v2_ranking_metrics.csv`
- `reports/proposal_v2/proposal_v2_cv_summary.md`

Final rapor/demo icin sakla.

Not: Fold 1, 2, 3, 4 ranking calistirirken onceki
`proposal_v2_ranking_metrics.csv` dosyasini da input olarak ekle. Boylece CSV
foldlari biriktirir. Eklemezsen sadece o foldun sonucunu yazar.

## 7. Explainability

Calistir:

- `proposal_v2_06_explainability.ipynb`

Gerekli onceki outputlar:

- `00_folds` outputu
- `01_tabular_only` outputu
- `04_image_only_cnn` outputu

Urettigi dosyalar:

- `reports/proposal_v2/proposal_v2_shap_summary.csv`
- `reports/proposal_v2/gradcam_examples/*.png`

Final rapor/demo icin sakla.

## Kisa Ozet

Minimum smoke sirasi:

1. `00_folds`
2. `01_tabular_only`
3. `02_image_history`
4. `03_late_fusion`
5. `05_ranking_map12`
6. `04_image_only_cnn`
7. `06_explainability`

Her adimda onceki gerekli notebook outputlarini `Add Data` ile sonraki notebooka
ekle.
