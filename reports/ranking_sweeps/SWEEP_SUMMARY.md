# Ranking Sweep Summary

Bu dosya candidate limit ve hybrid heuristic weight sweep sonuclarini tek yerde ozetler.

Onemli not: Bu sweep, duzeltilmis late-fusion Kaggle rerun tamamlanmadan once uretilmistir. Final late-fusion checkpointi artik AUC-ROC `0.9341` ve accuracy `0.8504` sonucuna sahiptir; bu nedenle sweep destekleyici/stale olarak okunmali ve zaman kalirsa yeni checkpoint ile yeniden uretilmelidir.

## En Iyi Satir

- Candidate limit: 3000
- Model: `late_fusion_hybrid_w025`
- MAP@12: 0.005346
- Precision@10: 0.003200
- Recall@10: 0.010019
- CandidateRecall: 0.434896

## Tum Sonuclar

| candidate_limit | model | MAP@12 | Precision@10 | Recall@10 | HitRate@12 | CandidateRecall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3000 | tabular_only | 0.001164 | 0.001400 | 0.004933 | 0.018000 | 0.434896 |
| 3000 | image_history | 0.003268 | 0.002800 | 0.008388 | 0.030000 | 0.434896 |
| 3000 | late_fusion | 0.002531 | 0.002800 | 0.010488 | 0.032000 | 0.434896 |
| 3000 | late_fusion_hybrid_w025 | 0.005346 | 0.003200 | 0.010019 | 0.032000 | 0.434896 |
| 3000 | late_fusion_hybrid_w045 | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.434896 |
| 3000 | late_fusion_hybrid_w065 | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.434896 |
| 3000 | candidate_heuristic | 0.005322 | 0.003600 | 0.010971 | 0.032000 | 0.434896 |
| 5000 | tabular_only | 0.001171 | 0.001200 | 0.003933 | 0.018000 | 0.475379 |
| 5000 | image_history | 0.003236 | 0.002400 | 0.007221 | 0.030000 | 0.475379 |
| 5000 | late_fusion | 0.002427 | 0.002600 | 0.008488 | 0.032000 | 0.475379 |
| 5000 | late_fusion_hybrid_w025 | 0.005346 | 0.003200 | 0.010019 | 0.032000 | 0.475379 |
| 5000 | late_fusion_hybrid_w045 | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.475379 |
| 5000 | late_fusion_hybrid_w065 | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.475379 |
| 5000 | candidate_heuristic | 0.005322 | 0.003600 | 0.010971 | 0.032000 | 0.475379 |
| 10000 | tabular_only | 0.001124 | 0.001200 | 0.003933 | 0.018000 | 0.538416 |
| 10000 | image_history | 0.003223 | 0.002400 | 0.007221 | 0.030000 | 0.538416 |
| 10000 | late_fusion | 0.002420 | 0.002600 | 0.008488 | 0.032000 | 0.538416 |
| 10000 | late_fusion_hybrid_w025 | 0.005346 | 0.003200 | 0.010019 | 0.032000 | 0.538416 |
| 10000 | late_fusion_hybrid_w045 | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.538416 |
| 10000 | late_fusion_hybrid_w065 | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.538416 |
| 10000 | candidate_heuristic | 0.005322 | 0.003600 | 0.010971 | 0.032000 | 0.538416 |

## Yorum

- Candidate limit arttikca CandidateRecall yukseldi: 3000 icin yaklasik 0.435, 5000 icin 0.475, 10000 icin 0.538.
- Buna ragmen MAP@12 ve HitRate@12 neredeyse sabit kaldi. Bu durum dogru urunlerin aday havuzuna daha sik girdigini, fakat top-12 siralamaya tasinmasinin hala zor oldugunu gosterir.
- `late_fusion_hybrid_w025` MAP@12 acisindan en yuksek satirdir; `w045/w065` Precision@10 ve Recall@10 tarafinda kucuk avantaj verir.
- `candidate_heuristic` modelden bagimsiz reranking sinyallerinin H&M tipi top-k probleminde guclu oldugunu gosterir; ana egitimli model olarak sunulmamalidir.
