# notebooks/kaggle

Bu klasordeki V2 notebooklari Kaggle'da ayri ayri calistirilmek icin
hazirlandi. Notebooklar self-contained yapidadir: deney kodu notebook hucrelerinin
icindedir, repo icindeki Python modullerine calisma aninda bagli degildir.

## Notebook Sirasi

1. `proposal_v2_00_folds.ipynb`
   - Customer-level 5-fold split.
2. `proposal_v2_01_tabular_only.ipynb`
   - Tabular-only MLP baseline.
3. `proposal_v2_02_image_history.ipynb`
   - EfficientNet embedding history MLP.
4. `proposal_v2_03_late_fusion.ipynb`
   - Tabular + visual late fusion.
5. `proposal_v2_04_image_only_cnn.ipynb`
   - Image-only EfficientNet-B0 CNN baseline.
6. `proposal_v2_05_ranking_map12.ipynb`
   - MAP@12 evaluation and hybrid reranking sweep.
7. `proposal_v2_06_explainability.ipynb`
   - SHAP/permutation importance and Grad-CAM outputs.

Buyuk input data, embedding cache ve egitilmis model dosyalari GitHub'a
eklenmeyecek.

## Kaggle Inputlari

Her notebook icin Kaggle inputlarinda sunlar bulunmali:

- H&M raw data:
  - `transactions_train.csv`
  - `customers.csv`
  - `articles.csv`
  - `images/`
- Visual embedding cache gereken notebooklar icin (`02`, `03`, `05`, `06`):
  - `article_image_embeddings_popular.npy`
  - `article_image_embedding_ids_popular.csv`

Notebooklar bu dosyalari `/kaggle/input` altinda otomatik arar. Path farkliysa
parametre hucrelerindeki `RAW_DIR`, `EMBEDDINGS_PATH`, `EMBEDDING_IDS_PATH` veya
`IMAGES_DIR` degiskenleri elle verilebilir.

## Kaggle Output Aktarma Mantigi

Kaggle'da her notebookun `/kaggle/working` alani ayridir. Bu yuzden bir
notebookun urettigi dosyalar sonraki notebooka otomatik gecmez.

Pratik akis:

1. Onceki notebooku calistir.
2. `Save Version` ile outputlari kaydet.
3. Sonraki notebookta `Add Data` ile onceki notebook outputunu input olarak ekle.
4. Notebook basindaki restore hucreleri `/kaggle/input/.../reports/proposal_v2`
   ve `/kaggle/input/.../models/proposal_v2` klasorlerini otomatik olarak
   writable `/kaggle/working` alanina kopyalar.

Notebooklar outputlari su pathlere yazar:

- `/kaggle/working/reports/proposal_v2/`
- `/kaggle/working/models/proposal_v2/`

## Notebook Bagimliliklari

| Notebook | Urettigi ana dosyalar | Sonraki kullanan notebooklar |
| --- | --- | --- |
| `proposal_v2_00_folds.ipynb` | `reports/proposal_v2/proposal_v2_fold_splits.csv`, `reports/proposal_v2/proposal_v2_fold_protocol_summary.json` | `01`, `02`, `03`, `04`, `05`, `06` |
| `proposal_v2_01_tabular_only.ipynb` | `models/proposal_v2/tabular_only_fold{FOLD_ID}.pt`, `reports/proposal_v2/proposal_v2_classification_metrics.csv` | `05`, `06` |
| `proposal_v2_02_image_history.ipynb` | `models/proposal_v2/image_history_fold{FOLD_ID}.pt`, `reports/proposal_v2/proposal_v2_classification_metrics.csv` | `05` |
| `proposal_v2_03_late_fusion.ipynb` | `models/proposal_v2/late_fusion_fold{FOLD_ID}.pt`, `reports/proposal_v2/proposal_v2_classification_metrics.csv` | `05` |
| `proposal_v2_04_image_only_cnn.ipynb` | `models/proposal_v2/image_only_effnet_cnn_fold{FOLD_ID}.pt`, `reports/proposal_v2/proposal_v2_cnn_metrics.csv` | `06` |
| `proposal_v2_05_ranking_map12.ipynb` | `reports/proposal_v2/proposal_v2_ranking_metrics.csv`, `reports/proposal_v2/proposal_v2_cv_summary.md` | final report/demo |
| `proposal_v2_06_explainability.ipynb` | `reports/proposal_v2/proposal_v2_shap_summary.csv`, `reports/proposal_v2/gradcam_examples/*.png` | final report/demo |

## Pratik Calisma Sirasi

Smoke kosu icin:

1. `00_folds` calistir, outputunu sakla.
2. `01_tabular_only`, `02_image_history`, `03_late_fusion` notebooklarina
   `00_folds` outputunu input olarak ekle.
3. `04_image_only_cnn` notebookuna `00_folds` outputunu input olarak ekle.
4. `05_ranking_map12` notebookuna `00_folds`, `01`, `02`, `03` outputlarini
   input olarak ekle.
5. `06_explainability` notebookuna `00_folds`, `01_tabular_only` ve
   `04_image_only_cnn` outputlarini input olarak ekle.

Full kosuda fold 0-4 icin ayni mantik gecerlidir. `05_ranking_map12` her foldda
cumulative `proposal_v2_ranking_metrics.csv` dosyasini buyutebilir; bunun icin
onceki ranking outputunu da yeni ranking notebookuna input olarak ekle. Bunu
yapmazsan her ranking kosusu sadece kendi fold sonucunu yazar; fold CSV'leri
sonradan yerelde veya ayri bir Kaggle notebookunda birlestirilmelidir.
