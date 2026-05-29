# Perseptron v2 - Proposal Aligned Branch

Bu branch, eski tamamlanmis Perseptron projesinin uzerine yama yapmak icin degil,
projeyi proposal'a tam uyumlu bicimde yeniden kurmak icin acildi.

Eski proje snapshot'i `main` branchinde durur. Bu branchte eski raporlar,
eski checkpointler, eski demo ve eski pipeline dosyalari bilerek kaldirildi.

## Ana Hedef

H&M Personalized Fashion Recommendations veri setinde su hipotezi proposal'a
uygun sekilde test etmek:

> EfficientNet-B0 gorsel temsilleri ve MLP tabular metadata branch'i late fusion
> ile birlestirildiginde, image-only ve tabular-only baseline'lara gore MAP@12
> merkezli onerme performansi artar.

## V2 Ana Ciktilari

- Customer-level 5-fold validation.
- MAP@12 merkezli ranking evaluation.
- Tabular-only, image-only/image-history ve multimodal late-fusion karsilastirmasi.
- SHAP tabular explainability.
- Grad-CAM visual CNN explainability.
- Proposal'a uyumlu final rapor ve demo.

## Branch Kurali

- `main`: eski bitmis proje snapshot'i.
- `report-polish`: proposal-aligned v2 calisma branchi.
- Buyuk raw data ve embedding cache repoya eklenmez.
- V2 checkpointleri gerekiyorsa Git LFS ile takip edilir.

Ilk uygulama plani: `docs/PROPOSAL_V2_PLAN.md`
