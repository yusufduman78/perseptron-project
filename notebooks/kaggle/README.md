# notebooks/kaggle

V2 Kaggle notebooklari burada tutulacak. Her notebook tek bir deney adimini
temsil eder; Kaggle'da ayri ayri acilip calistirilmesi hedeflenir.

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

Kaggle kosulari manuel yapilacak. Buyuk input data, embedding cache ve egitilmis
model dosyalari GitHub'a eklenmeyecek.

## Kaggle Output Aktarma Mantigi

Kaggle'da her notebook ayri calistigi icin bir notebookun `/kaggle/working`
altinda urettigi dosyalar sonraki notebooka otomatik gecmez. Bir sonraki
notebookun onceki ciktilari kullanabilmesi icin su yollardan birini kullan:

1. Onceki notebooku calistir, `Save Version` ile outputlari sakla.
2. Sonraki notebookta `Add Data` ile onceki notebook outputunu input olarak ekle.
3. Input altinda gelen `reports/proposal_v2/` ve `models/proposal_v2/`
   klasorlerini calismaya baslamadan once proje klasorune kopyala.

Alternatif olarak tum ara ciktilari tek bir Kaggle Dataset'e yukleyip sonraki
notebooklara o dataset'i input olarak ekleyebilirsin. Hangi yolu secersen sec,
notebooklar varsayilan olarak dosyalari proje icindeki su klasorlerde arar:

- `reports/proposal_v2/`
- `models/proposal_v2/`

Kaggle inputlari read-only oldugu icin onceki outputlar `/kaggle/input/...`
altinda gorunurse bunlari writable proje klasorune kopyalamak gerekir. Bu
hazirlik hucreleri notebooklarin basina eklendi; gerekirse ayni mantik su
sekildedir:

```python
from pathlib import Path
import os
import shutil

SOURCE_PROJECT_DIR = Path("/kaggle/input/perseptron-project")
WORK_PROJECT_DIR = Path("/kaggle/working/perseptron_project_work")
shutil.copytree(SOURCE_PROJECT_DIR, WORK_PROJECT_DIR, dirs_exist_ok=True)

os.environ["PERSEPTRON_PROJECT_DIR"] = str(WORK_PROJECT_DIR)

for input_root in Path("/kaggle/input").glob("*"):
    if input_root == SOURCE_PROJECT_DIR:
        continue
    for relative in ["reports/proposal_v2", "models/proposal_v2"]:
        source = input_root / relative
        target = WORK_PROJECT_DIR / relative
        if source.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
```

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
3. `05_ranking_map12` notebookuna `00_folds`, `01`, `02`, `03` outputlarini
   input olarak ekle.
4. `04_image_only_cnn` notebookuna `00_folds` outputunu input olarak ekle.
5. `06_explainability` notebookuna `00_folds`, `01_tabular_only` ve
   `04_image_only_cnn` outputlarini input olarak ekle.

Full kosuda fold 0-4 icin ayni mantik gecerlidir. `05_ranking_map12` her foldda
cumulative `proposal_v2_ranking_metrics.csv` dosyasini buyutebilir; bunun icin
onceki ranking outputunu da yeni ranking notebookuna input olarak ekle. Bunu
yapmazsan her ranking kosusu sadece kendi fold sonucunu yazar; bu durumda fold
CSV'lerini sonradan yerelde veya ayri bir Kaggle notebookunda birlestirmek gerekir.
