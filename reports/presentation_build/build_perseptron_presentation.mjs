import fs from "node:fs/promises";
import path from "node:path";

import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  panel,
  text,
  image,
  fill,
  fixed,
  hug,
  wrap,
  grow,
  fr,
} from "@oai/artifact-tool";

const projectRoot = "C:/Users/Yusuf/Desktop/perseptron_project";
const outDir = `${projectRoot}/reports`;
const previewDir = `${projectRoot}/reports/presentation_previews`;
const pptxPath = `${outDir}/perseptron_multimodal_recommendation_presentation.pptx`;

const W = 1920;
const H = 1080;

const C = {
  bg: "#090910",
  bg2: "#11111B",
  panel: "#171622",
  panel2: "#211A2C",
  text: "#F8F4EF",
  muted: "#B9B3C6",
  soft: "#7C748D",
  line: "#2B2938",
  pink: "#F472B6",
  pink2: "#F9A8D4",
  gold: "#FBBF24",
  teal: "#5EEAD4",
  blue: "#93C5FD",
  green: "#34D399",
  red: "#FCA5A5",
};

const assets = {
  denim: `${projectRoot}/data/images/hm_images/070/0706016001.jpg`,
  tank: `${projectRoot}/data/images/hm_images/056/0565379001.jpg`,
  blackTop: `${projectRoot}/data/images/hm_images/075/0759871002.jpg`,
  trousers: `${projectRoot}/data/images/hm_images/039/0399256001.jpg`,
  gradcam: `${projectRoot}/reports/proposal_v2/gradcam_examples/0505221004_gradcam.png`,
};

async function imageDataUrl(filePath) {
  const bytes = await fs.readFile(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const mime = ext === ".png" ? "image/png" : "image/jpeg";
  return `data:${mime};base64,${bytes.toString("base64")}`;
}

const imgData = Object.fromEntries(
  await Promise.all(Object.entries(assets).map(async ([key, value]) => [key, await imageDataUrl(value)])),
);

const deck = Presentation.create({
  slideSize: { width: W, height: H },
});
const slideRefs = [];

const titleStyle = { fontFace: "Georgia", fontSize: 68, bold: true, color: C.text };
const h2Style = { fontFace: "Aptos Display", fontSize: 46, bold: true, color: C.text };
const bodyStyle = { fontFace: "Aptos", fontSize: 27, color: C.muted };
const smallStyle = { fontFace: "Aptos", fontSize: 18, color: C.soft };
const labelStyle = { fontFace: "Aptos", fontSize: 16, bold: true, color: C.pink, letterSpacing: 1.6 };

function tx(value, options = {}) {
  return text(value, {
    width: options.width ?? fill,
    height: hug,
    style: options.style ?? bodyStyle,
    name: options.name,
  });
}

function root(child, fillColor = C.bg) {
  return panel(
    {
      name: "slide-bg",
      width: fill,
      height: fill,
      fill: fillColor,
      padding: { x: 72, y: 58 },
    },
    child,
  );
}

function add(slideNode) {
  const slide = deck.slides.add();
  slide.compose(slideNode, {
    frame: { left: 0, top: 0, width: W, height: H },
    baseUnit: 8,
  });
  slideRefs.push(slide);
  return slide;
}

function footer(n, label = "Perseptron | H&M Multimodal Recommendation") {
  return row(
    { width: fill, height: hug, align: "center", justify: "between" },
    [
      tx(label, { style: { ...smallStyle, fontSize: 15 } }),
      tx(String(n).padStart(2, "0"), {
        width: fixed(50),
        style: { ...smallStyle, fontSize: 15, bold: true, color: C.pink2 },
      }),
    ],
  );
}

function titleBlock(eyebrow, title, subtitle) {
  return column(
    { width: fill, height: hug, gap: 18 },
    [
      tx(eyebrow.toUpperCase(), { style: labelStyle }),
      tx(title, { style: h2Style }),
      subtitle ? tx(subtitle, { width: wrap(1220), style: bodyStyle }) : null,
    ].filter(Boolean),
  );
}

function bulletList(items, accent = C.pink) {
  return column(
    { width: fill, height: hug, gap: 18 },
    items.map((item) =>
      row(
        { width: fill, height: hug, gap: 14, align: "start" },
        [
          text("•", {
            width: fixed(26),
            height: hug,
            style: { fontFace: "Aptos", fontSize: 30, bold: true, color: accent },
          }),
          tx(item, { style: { ...bodyStyle, fontSize: 25, color: C.text } }),
        ],
      ),
    ),
  );
}

function stat(label, value, color = C.teal) {
  return column(
    { width: fill, height: hug, gap: 8 },
    [
      tx(label.toUpperCase(), { style: { ...smallStyle, bold: true, letterSpacing: 1.1 } }),
      tx(value, { style: { fontFace: "Aptos Display", fontSize: 42, bold: true, color } }),
    ],
  );
}

function softPanel(child, opts = {}) {
  return panel(
    {
      width: opts.width ?? fill,
      height: opts.height ?? hug,
      fill: opts.fill ?? C.panel,
      line: opts.line ?? { color: C.line, width: 1 },
      borderRadius: 20,
      padding: opts.padding ?? { x: 28, y: 24 },
    },
    child,
  );
}

function simpleTable(headers, rows, widths) {
  return column(
    { width: hug, height: hug, gap: 2 },
    [
      row(
        { width: fill, height: hug, gap: 0 },
        headers.map((h, i) =>
          panel(
            { width: fixed(widths[i]), height: hug, fill: C.panel2, padding: { x: 14, y: 12 } },
            tx(h, { style: { ...smallStyle, bold: true, color: C.pink2 } }),
          ),
        ),
      ),
      ...rows.map((r, idx) =>
        row(
          { width: fill, height: hug, gap: 0 },
          r.map((cell, i) =>
            panel(
              {
                width: fixed(widths[i]),
                height: hug,
                fill: idx % 2 === 0 ? "#12121C" : "#171622",
                padding: { x: 14, y: 13 },
              },
              tx(cell, {
                style: {
                  ...bodyStyle,
                  fontSize: i === 0 ? 22 : 24,
                  bold: i > 1,
                  color: i > 1 ? C.teal : C.text,
                },
              }),
            ),
          ),
        ),
      ),
    ],
  );
}

function scoreBar(model, auc, accuracy, color) {
  const aucWidth = Math.round(auc * 580);
  const accWidth = Math.round(accuracy * 580);
  return column(
    { width: fixed(760), height: hug, gap: 8 },
    [
      row(
        { width: fill, height: hug, justify: "between", align: "center" },
        [
          tx(model, { style: { ...bodyStyle, fontSize: 24, color: C.text, bold: true } }),
          tx(`AUC ${auc.toFixed(4)} | Acc ${accuracy.toFixed(4)}`, {
            width: fixed(280),
            style: { ...smallStyle, fontSize: 18, color },
          }),
        ],
      ),
      panel(
        { width: fixed(620), height: fixed(13), fill: "#262333", borderRadius: 7, padding: 0 },
        panel({ width: fixed(aucWidth), height: fill, fill: color, borderRadius: 7 }),
      ),
      panel(
        { width: fixed(620), height: fixed(9), fill: "#1A1824", borderRadius: 5, padding: 0 },
        panel({ width: fixed(accWidth), height: fill, fill: C.soft, borderRadius: 5 }),
      ),
    ],
  );
}

function productImage(dataUrl, width = 260, height = 360) {
  return panel(
    {
      width: fixed(width),
      height: fixed(height),
      fill: "#20202C",
      borderRadius: 24,
      padding: 0,
    },
    image({
      dataUrl,
      width: fill,
      height: fill,
      fit: "cover",
      alt: "H&M product image",
    }),
  );
}

add(
  root(
    row(
      { width: fill, height: fill, gap: 44, align: "center" },
      [
        column(
          { width: grow(1.05), height: fill, gap: 26, justify: "center" },
          [
            tx("PERSEPTRON FINAL SUNUMU", { style: labelStyle }),
            tx("H&M Moda Ürünleri İçin Multimodal Öneri Sistemi", {
              width: wrap(920),
              style: { ...titleStyle, fontSize: 74, lineHeight: 0.95 },
            }),
            tx("Tabular metadata, ürün görselleri ve müşteri görsel geçmişiyle late fusion tabanlı satın alma skorlama modeli.", {
              width: wrap(850),
              style: { ...bodyStyle, fontSize: 29 },
            }),
            row(
              { width: fill, height: hug, gap: 18 },
              [
                stat("Ana sonuç", "Late fusion lider", C.teal),
                stat("Final AUC", "0.8063", C.pink2),
                stat("Best AUC", "0.8121", C.gold),
              ],
            ),
          ],
        ),
        row(
          { width: grow(0.95), height: fill, gap: 22, align: "center", justify: "center" },
          [
            productImage(imgData.denim, 250, 410),
            column(
              { width: fixed(260), height: hug, gap: 22 },
              [productImage(imgData.tank, 260, 330), productImage(imgData.blackTop, 260, 330)],
            ),
          ],
        ),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Problem", "Moda önerisi sadece kategori problemi değildir.", "Aynı kategori içindeki ürünler renk, siluet ve stil açısından çok farklı olabilir."),
        row(
          { width: fill, height: fill, gap: 40, align: "center" },
          [
            column(
              { width: grow(1), height: hug, gap: 22 },
              [
                bulletList([
                  "Tabular metadata ürünü tarif eder; görsel embedding ürünün stilini temsil eder.",
                  "Müşterinin geçmiş alışverişindeki görsel karakter yeni öneriler için ek sinyaldir.",
                  "Amaç tek bir leaderboard optimizasyonu değil, multimodal katkıyı ölçen uçtan uca bir sistemdir.",
                ], C.gold),
              ],
            ),
            row(
              { width: grow(0.9), height: hug, gap: 18, justify: "center" },
              [productImage(imgData.denim, 245, 350), productImage(imgData.trousers, 245, 350), productImage(imgData.tank, 245, 350)],
            ),
          ],
        ),
        footer(2),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Hipotez", "Görsel sinyal tabular bağlamla birleştiğinde değer katar.", "Final sürümde classification ve ranking karşılaştırmaları bilinçli olarak ayrı tutuldu."),
        grid(
          { width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 },
          [
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Classification", { style: { ...h2Style, fontSize: 34 } }),
              tx("tabular_only vs image_only_effnet_cnn vs late_fusion", { style: { ...bodyStyle, fontSize: 24 } }),
              tx("CNN burada image-only baseline ve Grad-CAM kaynağıdır.", { style: { ...smallStyle, color: C.gold, bold: true } }),
            ])),
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Ranking / Demo", { style: { ...h2Style, fontSize: 34 } }),
              tx("tabular_only vs image_history vs late_fusion", { style: { ...bodyStyle, fontSize: 24 } }),
              tx("image_history CNN değil, müşteri görsel geçmiş baseline'ıdır.", { style: { ...smallStyle, color: C.teal, bold: true } }),
            ])),
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Hybrid", { style: { ...h2Style, fontSize: 34 } }),
              tx("late_fusion_hybrid_* sadece reranking/post-ranking deneyidir.", { style: { ...bodyStyle, fontSize: 24 } }),
              tx("Ana model olarak sunulmaz.", { style: { ...smallStyle, color: C.pink2, bold: true } }),
            ]), { fill: "#241629", line: { color: "#6D3B71", width: 1 } }),
          ],
        ),
        footer(3),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Data & pipeline", "H&M veri kaynakları tek öneri probleminde birleşti.", "Müşteri profili, ürün metaverisi, işlem geçmişi ve katalog görselleri aynı pipeline içinde kullanıldı."),
        simpleTable(
          ["Veri", "Projedeki rolü"],
          [
            ["transactions_train.csv", "Pozitif müşteri-ürün çiftleri ve müşteri geçmişi"],
            ["customers.csv", "Müşteri metaverisi"],
            ["articles.csv", "Ürün metaverisi"],
            ["images/", "EfficientNet-B0 ile CNN baseline ve embedding çıkarımı"],
          ],
          [430, 940],
        ),
        row(
          { width: fill, height: hug, gap: 28 },
          [
            stat("Embedding boyutu", "1280", C.teal),
            stat("Validation", "time-based", C.gold),
            stat("Ortam", "Kaggle GPU", C.blue),
          ],
        ),
        footer(4),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 30 },
      [
        titleBlock("Model ayrımı", "CNN, image_history ve late fusion aynı şey değildir.", "Bu ayrım final raporun en kritik savunma noktasıdır."),
        simpleTable(
          ["Model", "Girdi", "Rol"],
          [
            ["tabular_only", "Müşteri + ürün metaverisi", "Metadata baseline"],
            ["image_only_effnet_cnn", "Ürün görseli", "Image-only baseline + Grad-CAM"],
            ["image_history", "Aday embedding + müşteri görsel profili", "Ranking/demo visual-history baseline"],
            ["late_fusion", "Tabular branch + visual branch", "Ana multimodal model"],
          ],
          [360, 650, 470],
        ),
        softPanel(
          tx("Final demo üç recommendation modelini gösterir: tabular_only, image_history, late_fusion. CNN rapor/sunumda image-only baseline olarak anlatılır.", {
            style: { ...bodyStyle, fontSize: 28, color: C.text, bold: true },
          }),
          { fill: "#241629", line: { color: "#6D3B71", width: 1 } },
        ),
        footer(5),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Experimental design", "Final sonuçlar V2 Kaggle full-run artifactlerinden geldi.", "Notebooklar ayrı çalıştırıldı; fold/split, üç MLP modeli, CNN baseline, ranking ve explainability ayrı adımlara bölündü."),
        grid(
          { width: fill, height: hug, columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 22 },
          [
            softPanel(stat("Epoch", "3", C.pink2)),
            softPanel(stat("Split", "time-based", C.teal)),
            softPanel(stat("Ranking", "MAP@12", C.gold)),
            softPanel(stat("Explainability", "SHAP + Grad-CAM", C.blue)),
          ],
        ),
        bulletList([
          "Full 5-fold CV teslim süresi ve GPU kotası nedeniyle final full-run validation'a indirildi.",
          "Karşılaştırmalar aynı split ve aynı pipeline üzerinde tutuldu.",
          "Hybrid varyantlar ana model değil, destekleyici reranking deneyidir.",
        ], C.teal),
        footer(6),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 24 },
      [
        titleBlock("Classification results", "Late fusion en yüksek doğrulama performansını verdi.", "Görsel sinyal tek başına tabular baseline'ı geçmedi; tabular bağlamla birleşince katkı verdi."),
        row(
          { width: fill, height: grow(1), gap: 34, align: "center" },
          [
            column(
              { width: fixed(780), height: hug, gap: 28 },
              [
                scoreBar("tabular_only", 0.7841, 0.7150, C.blue),
                scoreBar("image_only_effnet_cnn", 0.7513, 0.6864, C.teal),
                scoreBar("late_fusion", 0.8063, 0.7207, C.pink2),
                tx("Kalın bar AUC-ROC, ince bar accuracy değerini gösterir.", {
                  style: { ...smallStyle, fontSize: 19, color: C.soft },
                }),
              ],
            ),
            column({ width: grow(1), height: hug, gap: 18 }, [
              stat("Late fusion final AUC", "0.8063", C.pink2),
              stat("Late fusion best AUC", "0.8121", C.gold),
              stat("Image-history AUC", "0.7480", C.teal),
            ]),
          ],
        ),
        footer(7),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 32 },
      [
        titleBlock("Ranking results", "Ana ranking lideri late fusion oldu.", "MAP@12 düşük mutlak değerli; yorum aynı aday havuzu üzerindeki göreli sıralama farkına dayanır."),
        simpleTable(
          ["Model", "MAP@12", "Precision@10", "Recall@10"],
          [
            ["late_fusion", "0.001262", "0.000889", "0.003256"],
            ["tabular_only", "0.000713", "0.000858", "0.002702"],
            ["image_history", "0.000653", "0.000764", "0.002404"],
          ],
          [430, 240, 280, 260],
        ),
        softPanel(
          tx("En iyi hybrid reranking MAP@12 0.000961 ile ana late_fusion sonucunun altında kaldı; bu yüzden hybrid ana model olarak sunulmadı.", {
            style: { ...bodyStyle, fontSize: 27, color: C.text, bold: true },
          }),
          { fill: "#12121C" },
        ),
        footer(8),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Explainability", "Tabular taraf SHAP, görsel taraf Grad-CAM ile desteklendi.", "Demo içinde ek olarak visual similarity ve metadata-match chipleri gösterilir."),
        grid(
          { width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 28 },
          [
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("SHAP summary", { style: { ...h2Style, fontSize: 34, color: C.teal } }),
              tx("Öne çıkan özellikler: age, section_no, product_group_name, colour_group_code, FN.", { style: { ...bodyStyle, fontSize: 25 } }),
              tx("Rol: tabular metadata kararını yorumlamak.", { style: { ...smallStyle, fontSize: 20, color: C.text } }),
            ])),
            softPanel(column({ width: fill, height: hug, gap: 18 }, [
              tx("Grad-CAM", { style: { ...h2Style, fontSize: 34, color: C.pink2 } }),
              productImage(imgData.gradcam, 500, 300),
              tx("Rol: image-only EfficientNet CNN baseline'ını görsel olarak kontrol etmek.", { style: { ...smallStyle, fontSize: 19, color: C.text } }),
            ])),
          ],
        ),
        footer(9),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Demo", "Final checkpointler canlı öneri arayüzüne bağlandı.", "Kullanıcı geçmiş ürün seçer; backend aynı aday havuzunu üç modelle skorlar."),
        row(
          { width: fill, height: grow(1), gap: 30, align: "center" },
          [
            column({ width: grow(1), height: hug, gap: 18 }, [
              bulletList([
                "Frontend: light fashion marketplace UI.",
                "Backend: FastAPI inference servisi.",
                "Endpointler: health, catalog, demo-scenarios, recommend, metrics, explainability.",
                "Çıktı: üç model önerisi, overlap/unique insights, explanation chipleri.",
              ], C.pink2),
            ]),
            softPanel(column({ width: fill, height: hug, gap: 18 }, [
              tx("Demo akışı", { style: { ...h2Style, fontSize: 32 } }),
              tx("Senaryo seçimi -> style bag -> üç model skoru -> öneri gridleri -> final evidence paneli", {
                style: { ...bodyStyle, fontSize: 27, color: C.text },
              }),
            ]), { width: fixed(670), fill: "#111827", line: { color: "#334155", width: 1 } }),
          ],
        ),
        footer(10),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Limitations", "Kalan sapmalar metodolojik adaptasyon olarak ele alındı.", "Final teslim, daha savunulabilir ve çalıştırılabilir bir sistem üretmeye odaklandı."),
        bulletList([
          "Full 5-fold CV yerine tek final full-run validation protokolü kullanıldı.",
          "CNN demo recommender değil; image-only baseline ve Grad-CAM kaynağı.",
          "Ranking metrikleri aday havuzu recall ve kısa validation penceresinden etkileniyor.",
          "Candidate generation ileride daha güçlü sequence/collaborative yöntemlerle geliştirilebilir.",
        ], C.gold),
        footer(11),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 38, justify: "center" },
      [
        tx("Sonuç", { style: labelStyle }),
        tx("Görsel bilgi tek başına değil, tabular bağlamla birleştiğinde en iyi sonucu verdi.", {
          width: wrap(1500),
          style: { ...titleStyle, fontSize: 66, lineHeight: 1.02 },
        }),
        row(
          { width: fill, height: hug, gap: 28 },
          [
            stat("En iyi model", "late_fusion", C.pink2),
            stat("Final AUC", "0.8063", C.teal),
            stat("MAP@12", "0.001262", C.gold),
          ],
        ),
        tx("Bu sonuç, H&M moda öneri probleminde multimodal late fusion hipotezini hem classification hem ranking hattında destekler.", {
          width: wrap(1320),
          style: { ...bodyStyle, fontSize: 29, color: C.text },
        }),
        footer(12),
      ],
    ),
  ),
);

await fs.mkdir(previewDir, { recursive: true });

const pptxBlob = await PresentationFile.exportPptx(deck);
await pptxBlob.save(pptxPath);

for (let i = 0; i < slideRefs.length; i += 1) {
  const slide = slideRefs[i];
  const pngBlob = await deck.export({ slide, format: "png" });
  const bytes = Buffer.from(await pngBlob.arrayBuffer());
  const name = `slide_${String(i + 1).padStart(2, "0")}.png`;
  await fs.writeFile(path.join(previewDir, name), bytes);
}

const summary = {
  pptxPath,
  previewDir,
  slideCount: slideRefs.length,
};

await fs.writeFile(`${previewDir}/build_summary.json`, JSON.stringify(summary, null, 2), "utf8");
console.log(JSON.stringify(summary, null, 2));
