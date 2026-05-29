# models/local_experiments

Bu klasör local geliştirme ve sandbox veri seti üzerinde eğitilmiş eski model checkpointlerini içerir.

## İçerik

Bu klasördeki modeller final Kaggle full-training modelleri değildir. Daha küçük local veriyle model mimarilerini test etmek, ablation mantığını kurmak ve pipeline hatalarını ayıklamak için kullanılmıştır.

Örnek checkpointler:

- `tabular_mlp.pt`
- `image_history_mlp.pt`
- `multimodal_fusion.pt`
- `controlled_tabular_only_mlp.pt`
- `controlled_image_history_mlp.pt`
- `controlled_multimodal_late_fusion.pt`

## Ne Zaman Kullanılır?

Bu dosyalar final rapor sonucunun ana kanıtı değildir. Ancak geliştirme sürecini göstermek, local ablation sonuçlarını yeniden üretmek veya daha küçük veri üzerinde hızlı denemeler yapmak için saklanmıştır.

Final model karşılaştırması için `models/final_kaggle/` kullanılmalıdır.

