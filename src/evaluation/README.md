# evaluation

Bu klasör, final modellerin öneri listesi kalitesini ölçmek için kullanılan değerlendirme scriptlerini içerir.

## Ranking Evaluation

`ranking_metrics_validation.py`, H&M transaction verisini zamana göre ikiye ayırır:

- Cutoff tarihinden önceki işlemler müşteri geçmişi olarak kullanılır.
- Cutoff tarihinden sonraki işlemler validation ground truth olarak kullanılır.

Script, demo backend'deki final checkpoint yükleme ve skorlama kodunu kullanarak aynı müşteri geçmişi için üç modelin önerilerini üretir:

- Tabular-only
- Image-history
- Multimodal late fusion

Ardından öneri listeleri şu metriklerle ölçülür:

- `MAP@12`
- `Precision@10`
- `Recall@10`
- `HitRate@12`
- `CandidateRecall`

Bu değerlendirme Kaggle leaderboard'ın birebir aynısı değildir; fakat proposal'da geçen top-k ranking metriklerini proje içindeki final modeller üzerinde ölçmek için daha doğru bir validation düzeni sağlar.

Örnek hızlı çalışma:

```powershell
python src\evaluation\ranking_metrics_validation.py --sample-customers 200 --candidate-limit 500
```

Daha ciddi rapor çıktısı için:

```powershell
python src\evaluation\ranking_metrics_validation.py --sample-customers 1000 --candidate-limit 800
```

Hybrid reranking ağırlıklarını aynı veri geçişinde karşılaştırmak için:

```powershell
python src\evaluation\ranking_metrics_validation.py --sample-customers 500 --candidate-limit 5000 --candidate-mode improved --visual-neighbors 3000 --co-purchase-per-item 300 --co-purchase-max-history-per-customer 100 --hybrid-heuristic-weights 0.25,0.45,0.65 --output-csv reports\ranking_sweeps\ranking_metrics_validation_multiweight_5000.csv --summary-md reports\ranking_sweeps\ranking_metrics_validation_summary_multiweight_5000.md
```

Çıktılar:

- `reports/ranking_metrics_validation.csv`
- `reports/ranking_metrics_validation_summary.md`
