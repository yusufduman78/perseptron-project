# Validation Protokol Notları

Bu not, final rapor yazımında proposal ile uygulanan deney düzeni arasındaki farkları doğru konumlandırmak için hazırlanmıştır.

## Proposal'daki Plan

Proposal'da 5-fold customer-level çapraz doğrulama ve MAP@12 odaklı top-12 ranking değerlendirmesi hedeflenmişti.

## Uygulanan Final Düzen

Final full-training deneyleri Kaggle üzerinde streaming/chunk eğitim düzeniyle yapıldı. Üç ana model aynı pozitif müşteri-ürün çifti evreninde karşılaştırıldı:

- Tabular-only MLP
- Image-history MLP
- Multimodal late fusion

Ana full-training metrikleri AUC-ROC ve accuracy oldu. MAP@12, Precision@10 ve Recall@10 ise sonradan ayrı bir time-based validation ranking evaluation olarak eklendi.

## Neden 5-Fold Full CV Yapılmadı?

Full H&M veri ölçeğinde üç model için 5-fold CV yaklaşık 15 büyük eğitim koşusu anlamına gelir. Her koşu 27 milyondan fazla pozitif müşteri-ürün çifti, negatif örnekleme, EfficientNet embedding cache kullanımı ve streaming eğitim gerektirir. Bu nedenle final teslim için 5-fold CV yerine:

- full-training validation,
- controlled/local ablation,
- time-based ranking evaluation

birlikte kullanıldı.

## Ranking Evaluation Yorumu

`src/evaluation/ranking_metrics_validation.py`, cutoff öncesi işlemleri müşteri geçmişi ve cutoff sonrası işlemleri ground truth olarak kullanır. Bu değerlendirme Kaggle public leaderboard'ın birebir aynısı değildir; proposal'da istenen MAP@12 / Precision@10 / Recall@10 gibi liste metriklerini final modellerin aynı aday havuzu üzerinde karşılaştırmak için kullanılır.

Önemli sınırlılık: bu ranking evaluation final checkpointleri kullanır. Final checkpointler full historical training ile üretildiği için bu deney strict unseen-training test değil, final modellerin aynı time-based aday havuzu üzerindeki davranış karşılaştırmasıdır.

## Late-Fusion Validation Split Düzeltmesi

Audit sırasında `notebooks/kaggle/perseptron_late_fusion.ipynb` içinde validation pozitifleri örneklendikten sonra index resetlenerek train tarafında yanlış satırların düşülme riski olduğu görüldü. Notebook güncellendi ve artık `val_indices` doğrudan kullanılarak train/validation pozitif çiftleri ayrık oluşturuluyor.

Bu nedenle late-fusion full-training sonucu Kaggle üzerinde düzeltilmiş notebookla yeniden çalıştırıldı. Yeni sonuç AUC-ROC `0.9341` ve accuracy `0.8504` olarak kilitlendi; önceki sonuç artık superseded kabul edilmelidir.
