# Kaggle Notebook Calistirma Rehberi

Bu dokuman iki ayri akis anlatir:

1. Smoke test: notebooklarin calistigini hizli kontrol etmek icin.
2. Full test: final rapor icin kullanilacak fold kosulari icin.

Notebooklar self-contained yapidadir. Kod notebook icindedir; repo icindeki
Python modullerini calisma aninda import etmez.

## Ortak Kural

Kaggle'da her notebookun output alani ayridir. Bir notebook bittikten sonra:

1. Notebooku calistir.
2. `Save Version` yap.
3. Sonraki notebookta `Add Data` ile onceki notebook outputunu ekle.

Notebooklar onceki outputlari otomatik olarak su klasorlerden arar:

- `reports/proposal_v2/`
- `models/proposal_v2/`

Tum notebooklar H&M raw data inputuna ihtiyac duyar:

- `transactions_train.csv`
- `customers.csv`
- `articles.csv`
- `images/`

`02`, `03`, `05`, `06` ayrica embedding inputuna ihtiyac duyar:

- `article_image_embeddings_popular.npy`
- `article_image_embedding_ids_popular.csv`

## Notebook Dosyalari

| No | Notebook | Gorev |
| --- | --- | --- |
| 00 | `proposal_v2_00_folds.ipynb` | Customer-level fold dosyasi uretir |
| 01 | `proposal_v2_01_tabular_only.ipynb` | Tabular-only MLP egitir |
| 02 | `proposal_v2_02_image_history.ipynb` | Image-history MLP egitir |
| 03 | `proposal_v2_03_late_fusion.ipynb` | Multimodal late-fusion model egitir |
| 04 | `proposal_v2_04_image_only_cnn.ipynb` | Image-only EfficientNet-B0 CNN egitir |
| 05 | `proposal_v2_05_ranking_map12.ipynb` | MAP@12 ranking evaluation calistirir |
| 06 | `proposal_v2_06_explainability.ipynb` | SHAP/permutation ve Grad-CAM uretir |

# Smoke Test Akisi

Smoke test amaci tam sonuc almak degil, notebooklarin path, input, output ve
model akisinin calistigini gormektir.

Smoke testte notebook parametreleri:

- `FAST_RUN = True`
- `FOLD_ID = 0`
- `SAMPLE_CUSTOMERS = 100` ranking notebookunda zaten default gelir

## Smoke 1 - Fold Dosyasini Uret

Calistir:

- `proposal_v2_00_folds.ipynb`

Parametre:

- `RAW_DIR = None`
- `N_FOLDS = 5`
- `RANDOM_SEED = 42`
- `MIN_HISTORY = 2`
- `VALIDATION_DAYS = 7`

Beklenen output:

- `reports/proposal_v2/proposal_v2_fold_splits.csv`
- `reports/proposal_v2/proposal_v2_fold_protocol_summary.json`

Kontrol:

- Fold summary tablosunda `customer_intersection` her fold icin `0` olmali.

Sonraki adimlarda `00_folds` notebook outputunu `Add Data` ile ekle.

## Smoke 2 - Tabular-Only Model

Calistir:

- `proposal_v2_01_tabular_only.ipynb`

Eklenmesi gereken inputlar:

- H&M raw data
- `00_folds` outputu

Parametre:

- `MODEL_NAME = 'tabular_only'`
- `FAST_RUN = True`
- `FOLD_ID = 0`

Beklenen output:

- `models/proposal_v2/tabular_only_fold0.pt`
- `reports/proposal_v2/proposal_v2_classification_metrics.csv`
- `reports/proposal_v2/folds/fold0_tabular_only_context.json`

Sonraki adimlara aktar:

- `tabular_only_fold0.pt`
- `proposal_v2_classification_metrics.csv`

Bu outputlar `05_ranking_map12` ve `06_explainability` icin gerekir.

## Smoke 3 - Image-History Model

Calistir:

- `proposal_v2_02_image_history.ipynb`

Eklenmesi gereken inputlar:

- H&M raw data
- Embedding cache
- `00_folds` outputu

Parametre:

- `MODEL_NAME = 'image_history'`
- `FAST_RUN = True`
- `FOLD_ID = 0`

Beklenen output:

- `models/proposal_v2/image_history_fold0.pt`
- `reports/proposal_v2/proposal_v2_classification_metrics.csv`
- `reports/proposal_v2/folds/fold0_image_history_context.json`

Sonraki adima aktar:

- `image_history_fold0.pt`

Bu output `05_ranking_map12` icin gerekir.

## Smoke 4 - Late-Fusion Model

Calistir:

- `proposal_v2_03_late_fusion.ipynb`

Eklenmesi gereken inputlar:

- H&M raw data
- Embedding cache
- `00_folds` outputu

Parametre:

- `MODEL_NAME = 'late_fusion'`
- `FAST_RUN = True`
- `FOLD_ID = 0`

Beklenen output:

- `models/proposal_v2/late_fusion_fold0.pt`
- `reports/proposal_v2/proposal_v2_classification_metrics.csv`
- `reports/proposal_v2/folds/fold0_late_fusion_context.json`

Sonraki adima aktar:

- `late_fusion_fold0.pt`

Bu output `05_ranking_map12` icin gerekir.

## Smoke 5 - Ranking / MAP@12

Calistir:

- `proposal_v2_05_ranking_map12.ipynb`

Eklenmesi gereken inputlar:

- H&M raw data
- Embedding cache
- `00_folds` outputu
- `01_tabular_only` outputu
- `02_image_history` outputu
- `03_late_fusion` outputu

Parametre:

- `FAST_RUN = True`
- `FOLD_ID = 0`
- `SAMPLE_CUSTOMERS = 100`

Gerekli model dosyalari:

- `models/proposal_v2/tabular_only_fold0.pt`
- `models/proposal_v2/image_history_fold0.pt`
- `models/proposal_v2/late_fusion_fold0.pt`

Beklenen output:

- `reports/proposal_v2/proposal_v2_ranking_metrics.csv`
- `reports/proposal_v2/proposal_v2_cv_summary.md`

Kontrol:

- Summary tabloda en az `tabular_only`, `image_history`, `late_fusion`
  satirlari gorunmeli.
- `late_fusion_hybrid_w025`, `late_fusion_hybrid_w045`,
  `late_fusion_hybrid_w065` satirlari gorunebilir; bunlar ana model degil,
  reranking deneyidir.

## Smoke 6 - Image-Only CNN

Calistir:

- `proposal_v2_04_image_only_cnn.ipynb`

Eklenmesi gereken inputlar:

- H&M raw data
- `00_folds` outputu

Parametre:

- `FAST_RUN = True`
- `FOLD_ID = 0`
- `TRAIN_BACKBONE = False`

Beklenen output:

- `models/proposal_v2/image_only_effnet_cnn_fold0.pt`
- `reports/proposal_v2/proposal_v2_cnn_metrics.csv`

Sonraki adima aktar:

- `image_only_effnet_cnn_fold0.pt`

Bu output `06_explainability` icin gerekir.

## Smoke 7 - Explainability

Calistir:

- `proposal_v2_06_explainability.ipynb`

Eklenmesi gereken inputlar:

- H&M raw data
- Embedding cache
- `00_folds` outputu
- `01_tabular_only` outputu
- `04_image_only_cnn` outputu

Parametre:

- `FOLD_ID = 0`
- `SHAP_BACKGROUND_ROWS = 80`
- `SHAP_EXPLAIN_ROWS = 40`
- `MAX_GRADCAM_EXAMPLES = 12`

Gerekli model dosyalari:

- `models/proposal_v2/tabular_only_fold0.pt`
- `models/proposal_v2/image_only_effnet_cnn_fold0.pt`

Beklenen output:

- `reports/proposal_v2/proposal_v2_shap_summary.csv`
- `reports/proposal_v2/gradcam_examples/*.png`

# Full Test Akisi

Full test final rapora konacak daha ciddi deney akisini uretir.

Full testte genel parametre:

- `FAST_RUN = False`

Fold kosulari icin `FOLD_ID` degerleri:

- `0`
- `1`
- `2`
- `3`
- `4`

## Full 1 - Fold Dosyasini Bir Kere Uret

Calistir:

- `proposal_v2_00_folds.ipynb`

Beklenen output:

- `reports/proposal_v2/proposal_v2_fold_splits.csv`
- `reports/proposal_v2/proposal_v2_fold_protocol_summary.json`

Bu output tum full notebooklara aktarilacak.

## Full 2 - Tabular-Only Fold 0-4

Notebook:

- `proposal_v2_01_tabular_only.ipynb`

Her fold icin ayarla:

- `FAST_RUN = False`
- `FOLD_ID = 0`, sonra `1`, sonra `2`, sonra `3`, sonra `4`

Her fold icin gerekli input:

- H&M raw data
- `00_folds` outputu
- Onceki tabular fold outputu, eger `proposal_v2_classification_metrics.csv`
  tek dosyada biriksin istiyorsan

Her folddan beklenen checkpoint:

- `models/proposal_v2/tabular_only_fold0.pt`
- `models/proposal_v2/tabular_only_fold1.pt`
- `models/proposal_v2/tabular_only_fold2.pt`
- `models/proposal_v2/tabular_only_fold3.pt`
- `models/proposal_v2/tabular_only_fold4.pt`

Ayrica guncellenen metrik dosyasi:

- `reports/proposal_v2/proposal_v2_classification_metrics.csv`

Bu outputlari ranking ve explainability icin sakla.

## Full 3 - Image-History Fold 0-4

Notebook:

- `proposal_v2_02_image_history.ipynb`

Her fold icin ayarla:

- `FAST_RUN = False`
- `FOLD_ID = 0`, sonra `1`, sonra `2`, sonra `3`, sonra `4`

Her fold icin gerekli input:

- H&M raw data
- Embedding cache
- `00_folds` outputu
- Onceki classification metrics outputu, eger CSV tek dosyada biriksin istiyorsan

Her folddan beklenen checkpoint:

- `models/proposal_v2/image_history_fold0.pt`
- `models/proposal_v2/image_history_fold1.pt`
- `models/proposal_v2/image_history_fold2.pt`
- `models/proposal_v2/image_history_fold3.pt`
- `models/proposal_v2/image_history_fold4.pt`

Ayrica guncellenen metrik dosyasi:

- `reports/proposal_v2/proposal_v2_classification_metrics.csv`

Bu outputlari ranking icin sakla.

## Full 4 - Late-Fusion Fold 0-4

Notebook:

- `proposal_v2_03_late_fusion.ipynb`

Her fold icin ayarla:

- `FAST_RUN = False`
- `FOLD_ID = 0`, sonra `1`, sonra `2`, sonra `3`, sonra `4`

Her fold icin gerekli input:

- H&M raw data
- Embedding cache
- `00_folds` outputu
- Onceki classification metrics outputu, eger CSV tek dosyada biriksin istiyorsan

Her folddan beklenen checkpoint:

- `models/proposal_v2/late_fusion_fold0.pt`
- `models/proposal_v2/late_fusion_fold1.pt`
- `models/proposal_v2/late_fusion_fold2.pt`
- `models/proposal_v2/late_fusion_fold3.pt`
- `models/proposal_v2/late_fusion_fold4.pt`

Ayrica guncellenen metrik dosyasi:

- `reports/proposal_v2/proposal_v2_classification_metrics.csv`

Bu outputlari ranking icin sakla.

## Full 5 - Ranking Fold 0-4

Notebook:

- `proposal_v2_05_ranking_map12.ipynb`

Her fold icin ayarla:

- `FAST_RUN = False`
- `FOLD_ID = 0`, sonra `1`, sonra `2`, sonra `3`, sonra `4`
- `SAMPLE_CUSTOMERS = None`

Her fold icin gerekli input:

- H&M raw data
- Embedding cache
- `00_folds` outputu
- Ayni fold icin tabular checkpoint
- Ayni fold icin image-history checkpoint
- Ayni fold icin late-fusion checkpoint
- Onceki ranking outputu, eger `proposal_v2_ranking_metrics.csv` tek dosyada
  biriksin istiyorsan

Fold 0 icin gerekli model dosyalari:

- `models/proposal_v2/tabular_only_fold0.pt`
- `models/proposal_v2/image_history_fold0.pt`
- `models/proposal_v2/late_fusion_fold0.pt`

Fold 1 icin gerekli model dosyalari:

- `models/proposal_v2/tabular_only_fold1.pt`
- `models/proposal_v2/image_history_fold1.pt`
- `models/proposal_v2/late_fusion_fold1.pt`

Ayni mantik fold 2, 3, 4 icin devam eder.

Beklenen output:

- `reports/proposal_v2/proposal_v2_ranking_metrics.csv`
- `reports/proposal_v2/proposal_v2_cv_summary.md`

Final rapor icin ana tablo buradan cikacak.

## Full 6 - CNN Baseline

Notebook:

- `proposal_v2_04_image_only_cnn.ipynb`

Onerilen minimum full kosu:

- `FAST_RUN = False`
- `FOLD_ID = 0`
- `TRAIN_BACKBONE = False`

Gerekli input:

- H&M raw data
- `00_folds` outputu

Beklenen output:

- `models/proposal_v2/image_only_effnet_cnn_fold0.pt`
- `reports/proposal_v2/proposal_v2_cnn_metrics.csv`

Not: CNN pahaliysa sadece fold 0 yeterli tutulabilir. Bu model ana recommender
degil; image-only baseline ve Grad-CAM kaynagi olarak raporlanir.

## Full 7 - Explainability

Notebook:

- `proposal_v2_06_explainability.ipynb`

Onerilen minimum full kosu:

- `FOLD_ID = 0`

Gerekli input:

- H&M raw data
- Embedding cache
- `00_folds` outputu
- `tabular_only_fold0.pt`
- `image_only_effnet_cnn_fold0.pt`

Beklenen output:

- `reports/proposal_v2/proposal_v2_shap_summary.csv`
- `reports/proposal_v2/gradcam_examples/*.png`

Final raporun explainability bolumu icin bu dosyalari sakla.

# Finalde Indirilecekler

Kaggle'dan indirilecek klasorler:

- `reports/proposal_v2/`
- `models/proposal_v2/`

Final rapor icin ozellikle gerekli dosyalar:

- `reports/proposal_v2/proposal_v2_classification_metrics.csv`
- `reports/proposal_v2/proposal_v2_ranking_metrics.csv`
- `reports/proposal_v2/proposal_v2_cv_summary.md`
- `reports/proposal_v2/proposal_v2_shap_summary.csv`
- `reports/proposal_v2/proposal_v2_cnn_metrics.csv`
- `reports/proposal_v2/gradcam_examples/*.png`
