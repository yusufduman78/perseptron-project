import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  Check,
  ChevronRight,
  Heart,
  Layers3,
  Loader2,
  Search,
  Shirt,
  ShoppingBag,
  SlidersHorizontal,
  Sparkles,
  Star,
  Trash2,
  UserRound,
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const MODEL_ORDER = ['tabular_only', 'image_history', 'late_fusion']

const MODEL_META = {
  tabular_only: {
    title: 'Tabular',
    eyebrow: 'Metadata model',
    detail: 'Müşteri ve ürün metadata sinyali.',
    metric: 'AUC 0.8550',
  },
  image_history: {
    title: 'Image History',
    eyebrow: 'Visual model',
    detail: 'Geçmiş ürünlerden görsel stil profili.',
    metric: 'AUC 0.9237',
  },
  late_fusion: {
    title: 'Late Fusion',
    eyebrow: 'Main model',
    detail: 'Metadata ve görsel geçmişi birlikte skorlar.',
    metric: 'AUC 0.9341 / Acc 0.8504',
  },
}

const REASON_LABELS = {
  'Aynı ürün tipi': 'Ürün tipi uyumu',
  'Benzer giyim grubu': 'Benzer kategori',
  'Renk eşleşmesi': 'Renk uyumu',
  'Yakın görsel geçmiş': 'Görsel geçmiş',
  'Görsel profil uyumu': 'Stil profili',
  'Popüler ürün': 'Popüler',
  'Aday havuzu seçimi': 'Aday seçimi',
}

function productImage(url = '') {
  return `${API_BASE}${url}`
}

function productPrice(articleId) {
  const seed = Number(String(articleId).slice(-4)) || 42
  return 249 + (seed % 56) * 10
}

function formatScore(score) {
  return Number.isFinite(score) ? score.toFixed(4) : '0.0000'
}

function formatSimilarity(value) {
  if (!Number.isFinite(value)) return '0.00'
  return value.toFixed(2)
}

function reasonLabel(tag) {
  return REASON_LABELS[tag] || tag
}

function ProductCard({ item, selected, onClick, compact = false }) {
  return (
    <button className={`product-card ${selected ? 'is-selected' : ''} ${compact ? 'compact' : ''}`} onClick={onClick}>
      <span className="product-image">
        <img src={productImage(item.image_url)} alt={item.prod_name} loading="lazy" />
        <span className="select-mark">{selected ? <Check size={14} /> : <Heart size={14} />}</span>
      </span>
      <span className="product-info">
        <strong>{item.prod_name}</strong>
        <small>{item.product_type_name}</small>
        {!compact && (
          <span className="product-meta">
            <span>{item.colour_group_name}</span>
            <b>₺{productPrice(item.article_id)}</b>
          </span>
        )}
      </span>
    </button>
  )
}

function ScenarioChips({ scenarios, activeScenario, onSelect }) {
  return (
    <div className="scenario-chips" aria-label="Hazır stiller">
      {scenarios.map((scenario) => (
        <button
          className={`scenario-chip ${activeScenario === scenario.id ? 'active' : ''}`}
          key={scenario.id}
          onClick={() => onSelect(scenario)}
        >
          <span>{scenario.title}</span>
          <small>{scenario.items?.length || 0} ürün</small>
        </button>
      ))}
    </div>
  )
}

function Header({ query, setQuery, onSearch, selectedCount, health }) {
  const apiReady = health?.status === 'ok'

  return (
    <header className="site-header">
      <div className="top-note">Perseptron Atelier / H&M multimodal recommendation demo</div>
      <div className="main-nav">
        <a className="brand" href="#top" aria-label="Perseptron Atelier">
          <span>PX</span>
          <strong>Perseptron Atelier</strong>
        </a>
        <nav className="category-nav" aria-label="Kategoriler">
          <a href="#catalog">New in</a>
          <a href="#catalog">Clothing</a>
          <a href="#catalog">Knitwear</a>
          <a href="#recommendations">Recommended</a>
        </nav>
        <form className="header-search" onSubmit={onSearch}>
          <Search size={17} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ürün, kategori veya renk ara"
          />
        </form>
        <div className="header-actions">
          <button title={apiReady ? 'API hazır' : 'API kontrol ediliyor'} className={apiReady ? 'ready' : ''}>
            <Activity size={18} />
          </button>
          <button title="Profil">
            <UserRound size={18} />
          </button>
          <button title="Style bag">
            <ShoppingBag size={18} />
            <b>{selectedCount}</b>
          </button>
        </div>
      </div>
    </header>
  )
}

function Hero({ heroProducts, selectedCount, onRun, recommendLoading }) {
  return (
    <section className="editorial-hero" id="top">
      <div className="hero-copy">
        <p className="section-kicker">Curated by models</p>
        <h1>Geçmiş alışverişten yeni sezon seçkisi.</h1>
        <p>
          Birkaç ürünü geçmiş alışveriş gibi seç; sistem aynı aday havuzunu üç final modelle skorlayıp
          kişiselleştirilmiş bir vitrin oluşturur.
        </p>
        <div className="hero-actions">
          <button className="primary-button" onClick={onRun} disabled={recommendLoading || selectedCount === 0}>
            {recommendLoading ? <Loader2 className="spin" size={17} /> : <Sparkles size={17} />}
            Önerileri getir
          </button>
          <a className="text-link" href="#catalog">
            Ürün seç <ArrowRight size={15} />
          </a>
        </div>
      </div>
      <div className="hero-products" aria-label="Editoryal ürün vitrini">
        {heroProducts.map((item, index) => (
          <article className={`hero-look look-${index + 1}`} key={item.article_id}>
            <img src={productImage(item.image_url)} alt={item.prod_name} loading="lazy" />
            <span>{item.product_type_name}</span>
          </article>
        ))}
      </div>
    </section>
  )
}

function StyleBag({ selected, profile, setProfile, onRemove, onRun, recommendLoading }) {
  return (
    <aside className="style-bag" aria-label="Style bag">
      <div className="bag-head">
        <span>Style bag</span>
        <strong>{selected.length} geçmiş ürün</strong>
      </div>

      <div className="bag-preview">
        {selected.length === 0 && (
          <div className="empty-bag">
            <Shirt size={28} />
            <span>Katalogdan ürün seç veya hazır stil yükle.</span>
          </div>
        )}
        {selected.slice(-4).map((item) => (
          <img key={item.article_id} src={productImage(item.image_url)} alt={item.prod_name} loading="lazy" />
        ))}
      </div>

      <div className="selected-list">
        {selected.map((item) => (
          <div className="selected-row" key={item.article_id}>
            <ProductCard item={item} selected compact onClick={() => onRemove(item)} />
            <button className="remove-button" onClick={() => onRemove(item)} title="Geçmişten çıkar">
              <Trash2 size={15} />
            </button>
          </div>
        ))}
      </div>

      <div className="profile-panel">
        <div className="panel-title">
          <SlidersHorizontal size={17} />
          <span>Müşteri profili</span>
        </div>
        <label>
          Yaş
          <input
            type="number"
            min="16"
            max="90"
            value={profile.age}
            onChange={(event) => setProfile({ ...profile, age: Number(event.target.value) })}
          />
        </label>
        <div className="switch-row">
          <label>
            <input
              type="checkbox"
              checked={profile.FN === 1}
              onChange={(event) => setProfile({ ...profile, FN: event.target.checked ? 1 : 0 })}
            />
            Fashion news
          </label>
          <label>
            <input
              type="checkbox"
              checked={profile.Active === 1}
              onChange={(event) => setProfile({ ...profile, Active: event.target.checked ? 1 : 0 })}
            />
            Aktif müşteri
          </label>
        </div>
        <label>
          Üyelik
          <select
            value={profile.club_member_status}
            onChange={(event) => setProfile({ ...profile, club_member_status: event.target.value })}
          >
            <option>ACTIVE</option>
            <option>PRE-CREATE</option>
            <option>LEFT CLUB</option>
          </select>
        </label>
        <label>
          Haber sıklığı
          <select
            value={profile.fashion_news_frequency}
            onChange={(event) => setProfile({ ...profile, fashion_news_frequency: event.target.value })}
          >
            <option>NONE</option>
            <option>Regularly</option>
            <option>Monthly</option>
          </select>
        </label>
      </div>

      <button className="primary-button wide" onClick={onRun} disabled={recommendLoading || selected.length === 0}>
        {recommendLoading ? <Loader2 className="spin" size={17} /> : <Sparkles size={17} />}
        Önerileri getir
      </button>
    </aside>
  )
}

function RecommendationTile({ item }) {
  const explanation = item.explanation || {}
  const tags = explanation.reason_tags || []
  const nearest = explanation.nearest_history

  return (
    <article className="recommendation-tile">
      <div className="recommendation-image">
        <img src={productImage(item.image_url)} alt={item.prod_name} loading="lazy" />
      </div>
      <div className="recommendation-body">
        <strong>{item.prod_name}</strong>
        <span>{item.product_type_name} / {item.colour_group_name}</span>
        <div className="recommendation-meta">
          <b>Skor {formatScore(item.score)}</b>
          <small>₺{productPrice(item.article_id)}</small>
        </div>
        <div className="reason-list">
          {tags.slice(0, 3).map((tag) => (
            <span key={tag}>{reasonLabel(tag)}</span>
          ))}
        </div>
        {nearest && (
          <p className="nearest-copy">
            Benzer geçmiş: <b>{nearest.prod_name}</b> / görsel {formatSimilarity(explanation.visual_similarity)}
          </p>
        )}
      </div>
    </article>
  )
}

function ModelInsights({ comparison, context }) {
  if (!comparison) {
    return (
      <aside className="model-insights empty">
        <Layers3 size={22} />
        <span>Öneri üretildiğinde model kesişimleri burada görünecek.</span>
      </aside>
    )
  }

  const overlap = comparison.overlap_matrix || {}
  const sharedItems = comparison.shared_items || []
  const lateFusionUnique = comparison.unique_by_model?.late_fusion || []

  return (
    <aside className="model-insights">
      <div className="insight-head">
        <p className="section-kicker">Model insights</p>
        <h3>Öneri karşılaştırması</h3>
      </div>
      <div className="insight-stats">
        <div>
          <span>Tabular x Image</span>
          <strong>{overlap.tabular_only__image_history ?? 0}</strong>
        </div>
        <div>
          <span>Image x Fusion</span>
          <strong>{overlap.image_history__late_fusion ?? 0}</strong>
        </div>
        <div>
          <span>Tabular x Fusion</span>
          <strong>{overlap.tabular_only__late_fusion ?? 0}</strong>
        </div>
      </div>
      <div className="insight-list">
        <span>Ortak öneriler</span>
        {sharedItems.slice(0, 3).map((item) => (
          <small key={`shared-${item.article_id}`}>{item.product_type_name} / {item.colour_group_name}</small>
        ))}
        {sharedItems.length === 0 && <small>Bu koşuda modeller ayrıştı.</small>}
      </div>
      <div className="insight-list">
        <span>Late Fusion'a özel</span>
        {lateFusionUnique.slice(0, 3).map((item) => (
          <small key={`fusion-${item.article_id}`}>{item.product_type_name} / {item.colour_group_name}</small>
        ))}
        {lateFusionUnique.length === 0 && <small>Late Fusion önerileri diğer modellerle örtüşüyor.</small>}
      </div>
      <div className="context-note">
        <BadgeCheck size={15} />
        <span>{context?.device || 'device'} / {context?.history_count || 0} geçmiş / aday {context?.candidate_limit || 0}</span>
      </div>
    </aside>
  )
}

function Recommendations({ recommendations, activeModel, setActiveModel, comparison, requestContext, recommendLoading }) {
  if (recommendLoading) {
    return (
      <section className="recommendations-section" id="recommendations">
        <div className="loading-panel">
          <Loader2 className="spin" size={24} />
          <strong>Modeller ürünleri skorluyor.</strong>
          <span>İlk çalıştırmada checkpoint ve embedding yüklemesi biraz sürebilir.</span>
        </div>
      </section>
    )
  }

  if (!recommendations) {
    return (
      <section className="recommendations-section" id="recommendations">
        <div className="empty-results">
          <Star size={26} />
          <strong>Henüz öneri vitrini oluşturulmadı.</strong>
          <span>Bir hazır stil seç veya katalogdan ürün ekleyip önerileri getir.</span>
        </div>
      </section>
    )
  }

  const items = recommendations[activeModel] || []
  const activeMeta = MODEL_META[activeModel]

  return (
    <section className="recommendations-section" id="recommendations">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Recommended for this style</p>
          <h2>{activeMeta.title} seçkisi</h2>
          <span>{activeMeta.detail}</span>
        </div>
        <div className="model-tabs" role="tablist" aria-label="Model seçimi">
          {MODEL_ORDER.map((modelKey) => (
            <button
              key={modelKey}
              className={activeModel === modelKey ? 'active' : ''}
              onClick={() => setActiveModel(modelKey)}
            >
              <span>{MODEL_META[modelKey].title}</span>
              <small>{MODEL_META[modelKey].eyebrow}</small>
            </button>
          ))}
        </div>
      </div>
      <div className="recommendation-layout">
        <div className="recommendation-grid">
          {items.map((item) => (
            <RecommendationTile key={`${activeModel}-${item.article_id}`} item={item} />
          ))}
        </div>
        <ModelInsights comparison={comparison} context={requestContext} />
      </div>
    </section>
  )
}

function App() {
  const [catalog, setCatalog] = useState([])
  const [selected, setSelected] = useState([])
  const [query, setQuery] = useState('')
  const [profile, setProfile] = useState({
    FN: 0,
    Active: 1,
    age: 28,
    club_member_status: 'ACTIVE',
    fashion_news_frequency: 'NONE',
  })
  const [recommendations, setRecommendations] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [requestContext, setRequestContext] = useState(null)
  const [scenarios, setScenarios] = useState([])
  const [activeScenario, setActiveScenario] = useState('')
  const [activeModel, setActiveModel] = useState('late_fusion')
  const [health, setHealth] = useState(null)
  const [catalogLoading, setCatalogLoading] = useState(true)
  const [recommendLoading, setRecommendLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    loadCatalog('')
    loadScenarios()
    loadHealth()
  }, [])

  async function loadHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`)
      const data = await response.json()
      setHealth(data)
    } catch {
      setHealth({ status: 'offline' })
    }
  }

  async function loadScenarios() {
    try {
      const response = await fetch(`${API_BASE}/api/demo-scenarios`)
      const data = await response.json()
      setScenarios(data.scenarios || [])
    } catch {
      setScenarios([])
    }
  }

  async function loadCatalog(search) {
    setCatalogLoading(true)
    setError('')
    const params = new URLSearchParams({ limit: '72' })
    if (search.trim()) params.set('q', search.trim())
    try {
      const response = await fetch(`${API_BASE}/api/catalog?${params}`)
      if (!response.ok) throw new Error('catalog failed')
      const data = await response.json()
      setCatalog(data.items || [])
    } catch {
      setError('Katalog yüklenemedi. Backend çalışıyor mu?')
    } finally {
      setCatalogLoading(false)
    }
  }

  function submitSearch(event) {
    event.preventDefault()
    loadCatalog(query)
    window.setTimeout(() => document.getElementById('catalog')?.scrollIntoView({ behavior: 'smooth' }), 80)
  }

  function applyScenario(scenario) {
    setActiveScenario(scenario.id)
    setSelected(scenario.items || [])
    setProfile(scenario.customer_profile || profile)
    setRecommendations(null)
    setComparison(null)
    setRequestContext(null)
    setError('')
  }

  function toggleProduct(product) {
    setActiveScenario('')
    setSelected((current) => {
      const exists = current.some((item) => item.article_id === product.article_id)
      if (exists) return current.filter((item) => item.article_id !== product.article_id)
      return [...current, product].slice(-8)
    })
  }

  async function runRecommendation() {
    if (selected.length === 0) {
      setError('Öneri üretmek için en az bir geçmiş ürün seç.')
      return
    }
    setRecommendLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          history_article_ids: selected.map((item) => item.article_id),
          customer_profile: profile,
          top_k: 8,
          candidate_limit: 500,
        }),
      })
      if (!response.ok) throw new Error('recommend failed')
      const data = await response.json()
      setRecommendations(data.models)
      setComparison(data.comparison)
      setRequestContext(data.request_context)
      setActiveModel('late_fusion')
      window.setTimeout(() => document.getElementById('recommendations')?.scrollIntoView({ behavior: 'smooth' }), 80)
    } catch {
      setError('Öneriler üretilemedi. Backend loglarını kontrol et.')
    } finally {
      setRecommendLoading(false)
    }
  }

  const selectedIds = useMemo(() => new Set(selected.map((item) => item.article_id)), [selected])
  const heroProducts = useMemo(() => {
    const source = selected.length ? selected : catalog
    return source.slice(0, 4)
  }, [catalog, selected])

  return (
    <main className="marketplace-app">
      <Header
        query={query}
        setQuery={setQuery}
        onSearch={submitSearch}
        selectedCount={selected.length}
        health={health}
      />
      <Hero
        heroProducts={heroProducts}
        selectedCount={selected.length}
        onRun={runRecommendation}
        recommendLoading={recommendLoading}
      />

      <section className="scenario-section">
        <div className="section-heading compact">
          <div>
            <p className="section-kicker">Quick start</p>
            <h2>Hazır stiller</h2>
          </div>
          <span>Sunum sırasında tek tıkla temiz demo akışı.</span>
        </div>
        <ScenarioChips scenarios={scenarios} activeScenario={activeScenario} onSelect={applyScenario} />
      </section>

      {error && <div className="error-banner">{error}</div>}

      <section className="shop-layout">
        <div className="catalog-area" id="catalog">
          <div className="section-heading">
            <div>
              <p className="section-kicker">Shop the catalog</p>
              <h2>Ürün geçmişi seç</h2>
              <span>{catalog.length} ürün gösteriliyor</span>
            </div>
            <div className="catalog-tools">
              <span>Gerçek H&M görselleri</span>
              <ChevronRight size={16} />
            </div>
          </div>
          {catalogLoading ? (
            <div className="loading-panel catalog-loading">
              <Loader2 className="spin" size={24} />
              <strong>Katalog yükleniyor.</strong>
            </div>
          ) : (
            <div className="catalog-grid">
              {catalog.map((item) => (
                <ProductCard
                  key={item.article_id}
                  item={item}
                  selected={selectedIds.has(item.article_id)}
                  onClick={() => toggleProduct(item)}
                />
              ))}
            </div>
          )}
        </div>
        <StyleBag
          selected={selected}
          profile={profile}
          setProfile={setProfile}
          onRemove={toggleProduct}
          onRun={runRecommendation}
          recommendLoading={recommendLoading}
        />
      </section>

      <Recommendations
        recommendations={recommendations}
        activeModel={activeModel}
        setActiveModel={setActiveModel}
        comparison={comparison}
        requestContext={requestContext}
        recommendLoading={recommendLoading}
      />
    </main>
  )
}

export default App
