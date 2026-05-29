# H&M Moda Urunleri Icin Multimodal Oneri Sistemi

**Yusuf Duman**  
Perseptron Projesi

## Abstract

Bu calisma, H&M Personalized Fashion Recommendations veri seti uzerinde moda urunleri icin multimodal bir onerme sistemi gelistirmeyi hedefler. Temel problem, bir musterinin belirli bir urunu satin alma olasiligini, yalnizca musteri ve urun metaverisiyle degil, urun fotograflarindan cikarilan gorsel temsillerle birlikte modellemektir. Bu amacla uc model karsilastirilmistir: tabular-only MLP, image-history MLP ve multimodal late fusion MLP. Urun gorselleri ImageNet uzerinde onceden egitilmis EfficientNet-B0 modeliyle 1280 boyutlu embeddinglere donusturulmus, musterinin gecmis satin alimlarindan gorsel tercih profili olusturulmus ve bu gorsel sinyaller tabular musteri/urun metaverisiyle late fusion mimarisinde birlestirilmistir. Final full-training deneylerinde tabular-only model AUC-ROC 0.8550 ve accuracy 0.7636, image-history model AUC-ROC 0.9237 ve accuracy 0.8297, multimodal late fusion model ise AUC-ROC 0.9341 ve accuracy 0.8504 elde etmistir. Ek olarak MAP@12, Precision@10 ve Recall@10 metrikleriyle time-based ranking evaluation yapilmis; post-ranking hybrid reranking sinyallerinin top-k siralamada onemli oldugu gosterilmistir.

## 1. Introduction

Moda onerme sistemleri klasik urun onerme problemlerinden farkli olarak guclu bir gorsel bilesen tasir. Kullanici tercihleri yalnizca urun kategorisi, fiyat, musteri yasi veya satis gecmisiyle degil; urunun rengi, kesimi, dokusu, formu ve onceki satin alimlarla olan gorsel benzerligiyle de iliskilidir. Ayni kategoriye ait iki urun, ornegin iki farkli gomlek veya iki farkli pantolon, metadata acisindan benzer gorunse bile kullanici icin tamamen farkli stil sinyalleri tasiyabilir.

Bu projede H&M Personalized Fashion Recommendations veri seti kullanilarak multimodal bir moda onerme modeli gelistirilmistir. Ana hipotez sudur: EfficientNet-B0 ile cikarilan urun gorsel embeddingleri, musterinin gecmis satin alimlarindan olusturulan gorsel tercih profili ve tabular musteri/urun metaverisi late fusion mimarisinde birlestirildiginde, yalnizca tabular veya yalnizca gorsel gecmis kullanan modellere gore daha guclu bir satin alma skoru uretir.

Calismanin temel katkisi uc egitimli modelin ayni problem uzerinde karsilastirilmasidir. Tabular-only MLP gorsel bilgi kullanmadan metadata tabanli baseline olusturur. Image-history MLP urun gorsellerinden cikarilan embeddingleri ve musteri gorsel gecmisini kullanir. Multimodal late fusion ise tabular branch ve visual branch ciktisini birlestirerek ana hipotezi test eder.

## 2. Related Work

Moda onerme sistemleri uzun suredir collaborative filtering, matrix factorization, content-based filtering ve derin ogrenme tabanli yaklasimlarla ele alinmaktadir. Geleneksel yontemler kullanici-urun etkilesimlerinden yararlanirken, moda alaninda urun gorselinin karar surecinde kritik bir rol oynamasi gorsel temsilleri onemli hale getirmistir. CNN, ResNet, EfficientNet ve benzeri modeller urun fotograflarindan renk, form, desen ve kategori sinyalleri cikarabilmektedir.

Multimodal recommender systems literaturu, gorsel, metinsel ve tabular sinyallerin birlestirilmesinin veri seyrekliği ve semantik temsil sorunlarini azaltabilecegini gostermektedir. Late fusion yaklasimi, farkli modalitelerin once kendi temsil uzaylarinda islenip daha sonra birlestirilmesine dayanir. Bu calismada da gorsel embeddingler ile tabular metadata ayrik branchlerde islenmis ve son katmanda birlestirilmistir.

Proposal asamasinda SHAP ve Grad-CAM gibi aciklanabilirlik yontemleri planlanmistir. Ancak final uygulamada EfficientNet-B0 nihai siniflandirici olarak fine-tune edilmemis, sabit feature extractor olarak kullanilmistir. Bu nedenle gorsel aciklanabilirlik icin embedding uzayinda benzer gecmis urunlerin gosterilmesi, Grad-CAM'e gore mimariyle daha uyumlu bir yontem olarak secilmistir. Tabular tarafta ise PyTorch embedding katmanlari ve streaming preprocessing zinciri nedeniyle SHAP yerine permutation importance kullanilmistir.

## 3. Methods

### 3.1 Veri ve On Isleme

Projede H&M Personalized Fashion Recommendations veri seti kullanilmistir. Ana veri kaynaklari `transactions_train.csv`, `customers.csv`, `articles.csv` ve urun gorselleridir. Transactions dosyasi pozitif musteri-urun satin alma ciftlerini ve musteri gecmisini olusturmak icin kullanilmistir. Customers ve articles dosyalari tabular model girdilerini saglamistir. Urun gorselleri ise EfficientNet-B0 ile embeddinglere donusturulmustur.

Full-training asamasinda 27,194,909 pozitif musteri-urun cifti kullanilmistir. Negatif ornekler, musterinin satin almadigi urunlerden orneklenerek ikili siniflandirma duzeni kurulmustur. Veri buyuklugu nedeniyle egitimler Kaggle GPU ortaminda streaming/chunk yaklasimiyla yapilmistir.

### 3.2 Gorsel Embedding ve Musteri Gorsel Profili

Urun fotograflari egitim sirasinda her epoch tekrar CNN'den gecirilmemistir. Bunun yerine EfficientNet-B0 modeli sabit feature extractor olarak kullanilmis ve 105,100 urun icin 1280 boyutlu embedding cache olusturulmustur. Bu embeddingler daha sonra image-history ve late-fusion modellerinde kullanilmistir.

Musteri gorsel profili, musterinin gecmiste satin aldigi urunlerin embeddinglerinin ortalamasi/toplami uzerinden temsil edilir. Aday urun skorlanirken aday urun embeddingi, musteri gorsel profil embeddingi, aday-profil cosine similarity degeri ve gorsel gecmis uzunlugu modele verilir.

### 3.3 Tabular-Only MLP

Tabular-only model, musteri ve urun metaverisini kullanarak satin alma olasiligi skoru uretir. Sayisal ozellikler normalize edilir; kategorik ozellikler embedding katmanlariyla temsil edilir. Bu model gorsel bilgi olmadan metadata sinyalinin gucunu olcmek icin baseline olarak kullanilir.

### 3.4 Image-History MLP

Image-history model tabular metaveri kullanmadan gorsel sinyale odaklanir. Aday urun embeddingi, musteri gorsel profil embeddingi, cosine similarity ve gorsel gecmis sayisi MLP girdisi olarak kullanilir. Bu model urun gorsellerinin ve musteri gorsel gecmisinin tek basina ne kadar ayirt edici oldugunu olcmek icin tasarlanmistir.

### 3.5 Multimodal Late Fusion

Multimodal late fusion model iki branch icerir. Tabular branch musteri ve urun metaverisini isler. Visual branch aday urun embeddingi, musteri gorsel profili, visual similarity ve history count sinyallerini isler. Iki branch'in temsilleri concatenate edilerek fusion head'e verilir ve final satin alma skoru uretilir.

## 4. Experimental Design

Final egitimler Kaggle GPU ortaminda gerceklestirilmistir. Veri tek seferde bellekte tutulmadigi icin streaming full-training kullanilmistir. Her model 3 epoch egitilmis, pozitif ornekler gercek satin alma ciftlerinden, negatif ornekler satin alinmamis urunlerden olusturulmustur.

Ana full-training metrikleri AUC-ROC ve accuracy'dir. AUC-ROC, modelin pozitif ve negatif musteri-urun ciftlerini ayirma basarisini; accuracy ise destekleyici siniflandirma performansini gosterir.

Proposal'da 5-fold customer-level cross validation planlanmis olsa da full veri olceginde uc model icin 5-fold CV yaklasik 15 buyuk egitim kosusu anlamina gelmektedir. Bu nedenle final teslim icin full-training validation, controlled/local ablation ve time-based ranking evaluation birlikte kullanilmistir. Bu secim, proje kapsaminda hesaplama maliyeti ile deneysel tutarlilik arasinda yapilan metodolojik bir adaptasyondur.

Ranking evaluation icin cutoff oncesi islemler musteri gecmisi, cutoff sonrasi islemler ground truth olarak kullanilmistir. Bu degerlendirme Kaggle public leaderboard'in birebir aynisi degildir; ancak proposal'da belirtilen MAP@12, Precision@10 ve Recall@10 metriklerini final modellerin ayni aday havuzu uzerindeki davranisini olcmek icin kullanir.

## 5. Experimental Results

### 5.1 Full-Training Sonuclari

Final sonuclar Kaggle full-training deneylerine dayanmaktadir. Duzeltilmis late-fusion rerun 30 Mayis 2026'da tamamlanmis ve final checkpoint projeye aktarilmistir.

| Model | Egitim duzeni | Pozitif cift | Validation pozitif | AUC-ROC | Accuracy |
|---|---|---:|---:|---:|---:|
| Tabular-only MLP | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.8550 | 0.7636 |
| Image-history MLP | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.9237 | 0.8297 |
| Multimodal late fusion | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.9341 | 0.8504 |

Image-history modelin tabular-only modelden belirgin sekilde yuksek sonuc vermesi, moda onerilerinde gorsel gecmis sinyalinin guclu oldugunu gostermektedir. Multimodal late fusion modeli tabular-only modele gore AUC-ROC'ta 0.0791, image-history modele gore 0.0104 puan daha yuksek sonuc vermis ve accuracy acisindan da en yuksek degeri elde etmistir. Bu sonuc, gorsel sinyalin tabular metaveriyle birlikte kullanilmasinin nihai katki sagladigini destekler.

### 5.2 Controlled Ablation

Local gelistirme surecinde daha kucuk ve kontrollu bir ablation deneyi yapilmistir. Bu deney ana final sonuc degil, model tasarimini destekleyen yardimci bir analizdir.

| Model | Satir | Train | Validation | Epoch | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Tabular-only MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.7219 | 0.6658 |
| Image-history MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.7883 | 0.7261 |
| Multimodal late fusion | 1,014,202 | 811,361 | 202,841 | 3 | 0.9611 | 0.8411 |

Controlled ablation full-training ile dogrudan ayni olcekte yorumlanmamalidir. Daha kucuk ve kontrollu veri evreni validation dagilimini daha kolay hale getirmis olabilir.

### 5.3 Ranking Evaluation ve Hybrid Reranking

Time-based ranking evaluation, son 7 gunu validation penceresi olarak kullanmistir. Candidate generation asamasinda son donem populerligi, metadata gruplari, co-purchase adaylari ve gorsel embedding komsulari birlikte kullanilmistir.

Mevcut checkpointlerle uretilen sweep sonucuna gore candidate limit arttikca CandidateRecall artmistir: 3000 icin 0.434896, 5000 icin 0.475379 ve 10000 icin 0.538416. Buna ragmen MAP@12 ve HitRate@12 neredeyse sabit kalmistir. Bu bulgu, dogru urunlerin aday havuzuna daha sik girdigini fakat top-12 siralamaya tasinmasinin hala zor oldugunu gosterir.

Hybrid reranking sonucu ana model olarak yorumlanmamalidir. `late_fusion_hybrid`, yeni egitilmis bir checkpoint degil; late-fusion skorunu recency, co-purchase, metadata benzerligi ve visual similarity sinyalleriyle yeniden siralayan post-ranking stratejisidir.

### 5.4 Explainability Sonuclari

Gorsel aciklanabilirlik icin aday urun embeddingi ile musterinin gecmis urun embeddingleri arasinda cosine similarity hesaplanmistir. Bu yontem, onerilen urunun musterinin gecmisindeki hangi urunlere gorsel olarak benzedigini gosterir.

Tabular tarafta permutation importance analizi yapilmistir. En guclu feature grubu urun kategorik ozellikleri olmustur. `section_name`, `department_name`, `product_type_name`, `age`, `product_code` ve `colour_group_name` one cikan ozelliklerdir. Bu sonuc, tabular modelin ozellikle urun bolumu, departmani, tipi, renk grubu ve musteri yasi gibi sinyallerden yararlandigini gostermektedir.

## 6. Conclusions

Bu proje, H&M moda urunleri uzerinde tabular metaveri ve urun gorsellerini birlestiren multimodal onerme hipotezini test etmistir. Tabular-only model metadata sinyalinden anlamli performans uretmis, image-history model gorsel gecmisin moda onerileri icin cok guclu bir sinyal oldugunu gostermis, multimodal late fusion model ise iki sinyali birlestirerek en yuksek AUC-ROC ve accuracy degerlerine ulasmistir.

Calismanin en onemli dersi, moda onerilerinde musterinin ne aldigi kadar, aldigi urunlerin nasil gorundugunun da onemli oldugudur. Ranking evaluation sonuclari ayrica top-k onerme problemlerinde model skorunun tek basina yeterli olmadigini; candidate generation ve reranking katmanlarinin kritik rol oynadigini gostermistir.

Etik ve pratik acidan, moda onerme sistemlerinde seffaflik ve aciklanabilirlik onemlidir. Bu projede gorsel benzerlik aciklamalari ve tabular permutation importance kullanilarak modelin hangi sinyallerden yararlandigi daha anlasilir hale getirilmistir. Bununla birlikte musteri davranis verisiyle calisan sistemlerde gizlilik, adil temsil ve demografik gruplar arasinda performans farklari ileriki calismalarda daha ayrintili incelenmelidir.

## 7. Bibliography

[1] Chakraborty, S., Hoque, M. S., Rahman Jeem, N., Biswas, M. C., Bardhan, D., & Lobaton, E. (2021). Fashion recommendation systems, models and methods: A review. Informatics, 8(3), 49.

[2] Liu, Q., Hu, J., Xiao, Y., Zhao, X., Gao, J., Wang, W., Li, Q., & Tang, J. (2024). Multimodal recommender systems: A survey. ACM Computing Surveys.

[3] Huang, B., Lu, Q., Huang, S., Wang, X., & Yang, H. (2024). Multi-modal clothing recommendation model based on large model and VAE enhancement. arXiv preprint arXiv:2410.02219.

[4] Zhang, C., Ji, X., & Cai, L. (2025). Clothing recommendation with multimodal feature fusion: Price sensitivity and personalization optimization. Applied Sciences, 15(8), 4591.

[5] Kalashi, K., & Teimourpour, B. (2025). A hybrid multimodal deep learning framework for intelligent fashion recommendation. arXiv preprint arXiv:2511.07573.

[6] Kaggle. H&M Personalized Fashion Recommendations. https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
