# Pipeline ve Model Diyagramları

Bu dosya final rapor ve sunuma eklenebilecek Mermaid diyagramlarını içerir.

## 1. Genel Proje Pipeline'ı

```mermaid
flowchart LR
    A["H&M CSV verileri"] --> B["transactions_train.csv"]
    A --> C["customers.csv"]
    A --> D["articles.csv"]
    E["Ürün görselleri"] --> F["EfficientNet-B0"]
    F --> G["Image-only CNN baseline"]
    F --> H["1280 boyutlu ürün embeddingleri"]

    B --> I["Pozitif müşteri-ürün çiftleri"]
    I --> J["Negatif örnekleme"]
    C --> K["Müşteri metaverisi"]
    D --> L["Ürün metaverisi"]
    H --> M["Müşteri görsel profili"]

    K --> N["Tabular-only MLP"]
    L --> N

    H --> O["Image-history baseline"]
    M --> O

    K --> P["Late fusion tabular branch"]
    L --> P
    H --> Q["Late fusion visual branch"]
    M --> Q
    P --> R["Fusion head"]
    Q --> R

    G --> S["CNN AUC + Grad-CAM"]
    N --> T["Classification / Ranking"]
    O --> T
    R --> T
```

## 2. Model Ayrımı

```mermaid
flowchart TB
    subgraph Cnn["image_only_effnet_cnn"]
        C1["Ürün görseli"] --> C2["EfficientNet-B0"]
        C2 --> C3["Binary classifier"]
        C3 --> C4["Image-only AUC + Grad-CAM"]
    end

    subgraph Tab["tabular_only"]
        T1["Müşteri metaverisi"] --> T3["Tabular MLP"]
        T2["Ürün metaverisi"] --> T3
        T3 --> T4["Satın alma skoru"]
    end

    subgraph Img["image_history"]
        I1["Aday ürün embeddingi"] --> I4["Visual-history MLP"]
        I2["Müşteri görsel profili"] --> I4
        I3["Cosine similarity + geçmiş uzunluğu"] --> I4
        I4 --> I5["Ranking skoru"]
    end

    subgraph Fusion["late_fusion"]
        F1["Müşteri + ürün metaverisi"] --> F3["Tabular branch"]
        F2["Aday embedding + görsel profil"] --> F4["Visual branch"]
        F3 --> F5["Fusion head"]
        F4 --> F5
        F5 --> F6["Multimodal skor"]
    end
```

## 3. Demo Inference Akışı

```mermaid
sequenceDiagram
    participant U as Kullanıcı
    participant UI as React demo
    participant API as FastAPI backend
    participant M as V2 checkpointleri
    participant D as Metadata ve embeddingler

    U->>UI: Hazır senaryo veya geçmiş ürün seçer
    UI->>API: /api/recommend
    API->>D: Ürün metadata ve embeddingleri okur
    API->>API: Aday ürün havuzu oluşturur
    API->>M: tabular_only skorları
    API->>M: image_history skorları
    API->>M: late_fusion skorları
    M-->>API: Sıralı öneriler
    API-->>UI: models + comparison + request_context
    UI-->>U: Üç model önerisi ve model insights
```

## 4. Sunum Mesajı

Classification tarafında CNN görsel baseline olarak vardır. Demo/ranking tarafında ise müşteri geçmişi gerektiği için `image_history` kullanılır. `late_fusion` iki sinyali birleştiren ana modeldir; `late_fusion_hybrid_*` yalnızca post-ranking deneyidir.
