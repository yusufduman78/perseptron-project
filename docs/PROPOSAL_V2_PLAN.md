# Perseptron v2 Proposal Alignment Plan

## Amac

Bu branchte proje, eski full-training/AUC odakli akistan ayrilip proposal'daki
deney tasarimina tam yaklastirilacak.

Ana hedef:

- Customer-level 5-fold validation.
- MAP@12 ana metrik.
- EfficientNet-B0 visual model + MLP tabular model + late fusion.
- SHAP ve Grad-CAM explainability.
- Rapor ve demo dilinin bu yeni deney duzenine gore kurulmasi.

## Deney Aileleri

1. `tabular_only_mlp`
   - Musteri ve urun metadata baseline.
2. `image_only_effnet_cnn`
   - EfficientNet-B0 image classifier / visual CNN sanity baseline.
   - Grad-CAM bu model uzerinden uretilecek.
3. `image_history_mlp`
   - EfficientNet-B0 embedding + musteri gorsel gecmis profili.
4. `multimodal_late_fusion`
   - Tabular branch + visual branch.
   - Ana proposal modeli.

`late_fusion_hybrid` egitimli ana model degildir; sadece post-ranking/reranking
deneyi olarak ele alinabilir.

## Validation

- `N_FOLDS = 5`
- `RANDOM_SEED = 42`
- Split customer-level olacak.
- Her fold icin train ve validation customer setleri kesismeyecek.
- Validation tarafinda cutoff oncesi islemler history, cutoff sonrasi islemler
  ground truth olacak.

## Ranking Ayarlari

- `top_k = 12`
- `precision_k = 10`
- `candidate_limit = 5000`
- `visual_neighbors = 3000`
- `co_purchase_per_item = 300`
- `hybrid_weights = [0.25, 0.45, 0.65]`

## Beklenen Ciktilar

- `reports/proposal_v2/proposal_v2_cv_summary.md`
- `reports/proposal_v2/proposal_v2_classification_metrics.csv`
- `reports/proposal_v2/proposal_v2_ranking_metrics.csv`
- `reports/proposal_v2/proposal_v2_hybrid_reranking_summary.md`
- `reports/proposal_v2/proposal_v2_shap_summary.csv`
- `reports/proposal_v2/gradcam_examples/`
- `models/proposal_v2/`
- `notebooks/kaggle/proposal_v2_00_folds.ipynb`
- `notebooks/kaggle/proposal_v2_01_tabular_only.ipynb`
- `notebooks/kaggle/proposal_v2_02_image_history.ipynb`
- `notebooks/kaggle/proposal_v2_03_late_fusion.ipynb`
- `notebooks/kaggle/proposal_v2_04_image_only_cnn.ipynb`
- `notebooks/kaggle/proposal_v2_05_ranking_map12.ipynb`
- `notebooks/kaggle/proposal_v2_06_explainability.ipynb`

## Uygulama Sirasi

1. V2 klasor ve script iskeletini kur.
2. Customer-level 5-fold split scriptini yaz.
3. Kaggle notebooklarini deney bazinda ayir ve aciklamali hale getir.
4. Fold 0 smoke training + ranking calistir.
5. Fold 0 full training + ranking calistir.
6. Fold 1-4 kosularini tamamla.
7. SHAP ve Grad-CAM ciktilarini uret.
8. Demo v2'yi yeni metrik ve explanation ciktisina bagla.
9. Final report v2'yi proposal hikayesine gore yaz.

## Kabul Kriterleri

- 5 fold icin train/validation customer intersection `0`.
- Ana sonuc tablosunda MAP@12 mean/std var.
- Precision@10 ve Recall@10 raporlanmis.
- SHAP ve Grad-CAM bolumleri gercek ciktilara dayaniyor.
- Demo v2, proposal model ailelerini ve explanation ciktisini gosteriyor.
