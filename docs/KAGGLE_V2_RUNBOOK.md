# Kaggle v2 Runbook

Bu dosya, proposal-aligned v2 deneylerini Kaggle'da hangi sirayla
calistiracagini anlatir. Notebooklar tek bir toplu launcher olarak degil, her
deney kendi dosyasinda olacak sekilde ayrildi. Bu sayede Kaggle'da notebooklari
tek tek acip calistirabilir, hatayi veya sonucu ilgili deney uzerinden
takip edebilirsin.

## 0. Kaggle Inputlari

Kaggle notebook inputlarinda sunlar olmali:

- H&M competition/raw dataset:
  - `transactions_train.csv`
  - `customers.csv`
  - `articles.csv`
  - `images/`
- EfficientNet embedding cache dataset:
  - `article_image_embeddings_popular.npy`
  - `article_image_embedding_ids_popular.csv`

Notebooklar self-contained yapidadir; `src/` altindaki Python dosyalarina
calisma aninda ihtiyac duymaz. Kaggle input pathin farkli olursa notebook
basindaki `RAW_DIR`, `EMBEDDINGS_PATH`, `EMBEDDING_IDS_PATH` veya `IMAGES_DIR`
degiskenlerinden elle duzeltebilirsin.
Notebooklarin birbirine hangi outputlari tasidigi ayrica
`notebooks/kaggle/README.md` icinde tablo olarak yazildi.

## 1. Notebook Sirasi

1. `notebooks/kaggle/proposal_v2_00_folds.ipynb`
   - Customer-level 5-fold split uretir.
   - Train/validation customer intersection kontrolunu raporlar.
2. `notebooks/kaggle/proposal_v2_01_tabular_only.ipynb`
   - Metadata tabanli MLP baseline egitir.
3. `notebooks/kaggle/proposal_v2_02_image_history.ipynb`
   - EfficientNet embedding history profili ile visual-history MLP egitir.
4. `notebooks/kaggle/proposal_v2_03_late_fusion.ipynb`
   - Tabular branch + visual branch late-fusion modelini egitir.
5. `notebooks/kaggle/proposal_v2_04_image_only_cnn.ipynb`
   - Proposal'a daha yakin image-only EfficientNet-B0 CNN baseline egitir.
   - Grad-CAM bu modelden uretilecek.
6. `notebooks/kaggle/proposal_v2_05_ranking_map12.ipynb`
   - MAP@12, Precision@10, Recall@10 ve hybrid reranking sweep calistirir.
7. `notebooks/kaggle/proposal_v2_06_explainability.ipynb`
   - Tabular SHAP/permutation importance ve CNN Grad-CAM ciktilarini uretir.

## 2. Ilk Smoke Kosusu

Once tum notebooklari fold 0 ve hizli mod ile deneriz:

```python
FAST_RUN = True
FOLD_ID = 0
```

Smoke sirasinda beklenen ana ciktilar:

- `reports/proposal_v2/proposal_v2_fold_splits.csv`
- `models/proposal_v2/tabular_only_fold0.pt`
- `models/proposal_v2/image_history_fold0.pt`
- `models/proposal_v2/late_fusion_fold0.pt`
- `models/proposal_v2/image_only_effnet_cnn_fold0.pt`
- `reports/proposal_v2/proposal_v2_ranking_metrics.csv`
- `reports/proposal_v2/proposal_v2_cv_summary.md`

Bu kosu basariliysa full ayarlara gecilir.

## 3. Full Fold Kosulari

Full kosuda:

```python
FAST_RUN = False
```

Onerilen sira:

1. `proposal_v2_00_folds.ipynb` bir kez calistirilir.
2. `proposal_v2_01_tabular_only.ipynb` fold 0, 1, 2, 3, 4 icin ayri ayri calistirilir.
3. `proposal_v2_02_image_history.ipynb` fold 0, 1, 2, 3, 4 icin ayri ayri calistirilir.
4. `proposal_v2_03_late_fusion.ipynb` fold 0, 1, 2, 3, 4 icin ayri ayri calistirilir.
5. `proposal_v2_05_ranking_map12.ipynb` her fold icin calistirilir.

CNN baseline pahali olursa once sadece fold 0 icin calistir:

```python
FOLD_ID = 0
FAST_RUN = False
```

Sure kalirsa `proposal_v2_04_image_only_cnn.ipynb` diger foldlara da
genisletilebilir. Rapor tarafinda bu durum "image-only CNN sanity baseline /
Grad-CAM source" olarak anlatilacak.

## 4. Explainability

`proposal_v2_06_explainability.ipynb` icin en az su checkpointler hazir olmali:

- `models/proposal_v2/tabular_only_fold0.pt`
- `models/proposal_v2/image_only_effnet_cnn_fold0.pt`

Notebook tabular checkpoint uzerinden SHAP summary uretmeye calisir. Kaggle
ortaminda `shap` kullanilamazsa ayni dosyada permutation fallback uretilir; raporda
bu durum acikca belirtilir.

CNN checkpoint varsa Grad-CAM ornekleri su klasore yazilir:

- `reports/proposal_v2/gradcam_examples/`

## 5. Indirilecek Dosyalar

Kaggle'dan su klasorleri indir:

- `reports/proposal_v2/`
- `models/proposal_v2/`

Sonra yerelde ayni pathlere koyacagiz ve rapor/demo v2'yi final sonuclarla
guncelleyecegiz.
