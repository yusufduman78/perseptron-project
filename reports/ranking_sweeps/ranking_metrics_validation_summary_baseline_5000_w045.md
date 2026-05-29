# Validation Ranking Evaluation

Bu dosya, final Perseptron modellerinin zamana dayalı validation ayrımı üzerinde top-k öneri kalitesini özetler.

## Kurulum

- Validation penceresi: son 7 gün
- Cutoff tarihi: 2020-09-15
- Örneklenen müşteri sayısı: 500
- Candidate limit: 5000
- Candidate mode: improved
- Visual neighbors: 3000
- Co-purchase per item: 300
- Co-purchase max history/customer: 100
- Hybrid heuristic weight: 0.45
- Top-k: 12
- Precision/Recall k: 10

Cutoff öncesi işlemler müşteri geçmişi, cutoff sonrası işlemler ground truth olarak kullanılmıştır.
Yalnızca EfficientNet embedding evreninde bulunan ürünler değerlendirmeye alınmıştır.
Improved candidate mode; son dönem popülerliği, metadata grupları, co-purchase adayları ve görsel embedding komşularını birlikte kullanır.

## Sonuçlar

| model | customers_evaluated | map_at_12 | precision_at_10 | recall_at_10 | hit_rate_at_12 | candidate_recall |
| --- | --- | --- | --- | --- | --- | --- |
| tabular_only | 500 | 0.001155 | 0.001200 | 0.003933 | 0.018000 | 0.474761 |
| image_history | 500 | 0.003236 | 0.002400 | 0.007221 | 0.030000 | 0.474761 |
| late_fusion | 500 | 0.002427 | 0.002600 | 0.008488 | 0.032000 | 0.474761 |
| late_fusion_hybrid | 500 | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.474761 |
| candidate_heuristic | 500 | 0.005322 | 0.003600 | 0.010971 | 0.032000 | 0.474761 |

## Not

Bu değerlendirme Kaggle public leaderboard'ın birebir karşılığı değildir. Ama proposal'da belirtilen MAP@12, Precision@10 ve Recall@10 gibi ranking metriklerini, final modellerin aynı aday havuzu üzerinde karşılaştırılması için kullanır.

Detaylı müşteri-model satırları: `reports\ranking_metrics_validation.csv`
Toplam detay satırı: 2,500
