// RNIA App shell — sidebar + topbar + view router
// Mirrors the Claude Design prototype layout pixel-for-pixel
import { useState, useEffect, useMemo, Fragment } from 'react';
import { fetchSummary, fetchNews } from './api';
import { Icon } from './rnia/Icon';
import { Overview } from './rnia/Overview';
import { NewsFeed } from './rnia/NewsFeed';
import { Analyze } from './rnia/Analyze';
import {
  normalizeArticle, buildPulse, buildStanceRadar,
  normalizeEventDistribution, normalizeStanceDistribution,
  EVENT_TYPES,
} from './rnia/data';

// ─── Navigation config ──────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: 'overview',    label: 'Overview',       icon: 'grid',      section: 'main' },
  { id: 'feed',        label: 'News Feed',      icon: 'feed',      section: 'main', badge: null },
  { id: 'analyze',     label: 'Analyze',        icon: 'sparkles',  section: 'main' },
  { id: 'watchlist',   label: 'Watchlist',       icon: 'bookmark',  section: 'personal', badge: null },
];

const SECTIONS = [
  { id: 'main',     label: 'Intelligence' },
  { id: 'personal', label: 'Personal' },
];

// ─── Stub view for future pages ─────────────────────────────────────────────
const StubView = ({ title, sub, features, mark }) => (
  <div className="view">
    <div className="view-head">
      <div>
        <h1 className="view-title">{title}</h1>
        <div className="view-sub">{sub}</div>
      </div>
    </div>
    <div className="stub-card">
      <div className="stub-mark"><Icon name={mark} size={28} /></div>
      <h2 className="stub-title">Designed, not yet wired</h2>
      <p className="stub-sub">
        The shell is laid out and ready for backend integration. Below is the planned feature set.
      </p>
      <div className="stub-feats">
        {features.map((f, i) => (
          <span key={i} className="badge badge-event" style={{ '--ev-color': '#10B981', padding: '6px 12px', fontSize: 12 }}>{f}</span>
        ))}
      </div>
    </div>
  </div>
);

const STUBS = {};

// ─── Default summary when the backend is unreachable ────────────────────────
const FALLBACK_SUMMARY = {
  total_articles: 0,
  average_impact_score: 0,
  stance_distribution: { bullish: 0, bearish: 0, neutral: 0 },
  event_distribution: EVENT_TYPES.reduce((a, e) => { a[e.id] = 0; return a; }, {}),
};

// ─── App ────────────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView]         = useState('overview');
  const [theme, setTheme]       = useState('light');
  const [loading, setLoading]   = useState(true);
  const [apiOk, setApiOk]       = useState(false);

  // Raw backend data
  const [rawSummary, setRawSummary] = useState(null);
  const [rawArticles, setRawArticles] = useState([]);

  // Watchlist state
  const [watchlist, setWatchlist] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('rnia_watchlist') || '[]');
    } catch {
      return [];
    }
  });

  useEffect(() => {
    localStorage.setItem('rnia_watchlist', JSON.stringify(watchlist));
  }, [watchlist]);

  const toggleWatchlist = (id) => {
    setWatchlist(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  // ─── Theme sync ───────────────────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);
  }, [theme]);

  // ─── Data fetch ───────────────────────────────────────────────────────────
  const loadData = async () => {
    setLoading(true);
    try {
      const [s, n] = await Promise.all([fetchSummary(), fetchNews()]);
      setRawSummary(s);
      setRawArticles(n);
      setApiOk(true);
    } catch {
      setApiOk(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, []);

  // ─── Normalized data ──────────────────────────────────────────────────────
  const articles = useMemo(
    () => rawArticles.map((a, i) => normalizeArticle(a, i)),
    [rawArticles]
  );

  const summary = useMemo(() => {
    if (!rawSummary) return FALLBACK_SUMMARY;
    return {
      ...rawSummary,
      event_distribution: normalizeEventDistribution(rawSummary.event_distribution),
      stance_distribution: normalizeStanceDistribution(rawSummary.stance_distribution),
    };
  }, [rawSummary]);

  const pulseData = useMemo(() => buildPulse(articles), [articles]);
  const stanceRadar = useMemo(() => buildStanceRadar(articles, summary.event_distribution), [articles, summary]);

  // Badge for feed nav
  const feedBadge = summary.total_articles
    ? summary.total_articles >= 1000
      ? `${(summary.total_articles / 1000).toFixed(1)}k`
      : String(summary.total_articles)
    : articles.length > 0
      ? String(articles.length)
      : null;

  // ─── View renderer ────────────────────────────────────────────────────────
  const renderView = () => {
    switch (view) {
      case 'overview':
        return (
          <Overview
            summary={summary}
            articles={articles}
            pulseData={pulseData}
            stanceRadar={stanceRadar}
            onJump={setView}
          />
        );
      case 'feed':
        return (
          <NewsFeed
            articles={articles}
            watchlist={watchlist}
            onToggleWatchlist={toggleWatchlist}
            total={summary.total_articles || articles.length}
          />
        );
      case 'watchlist': {
        const saved = articles.filter(a => watchlist.includes(a.id));
        return (
          <NewsFeed
            articles={saved}
            watchlist={watchlist}
            onToggleWatchlist={toggleWatchlist}
            total={saved.length}
            title={<>Saved <em>articles</em></>}
            sub="Your personal watchlist of high-impact news."
          />
        );
      }
      case 'analyze':
        return <Analyze />;
      default: {
        const Stub = STUBS[view];
        return Stub ? <Stub /> : <Overview summary={summary} articles={articles} pulseData={pulseData} stanceRadar={stanceRadar} onJump={setView} />;
      }
    }
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* TOPBAR */}
      <div className="topbar">
        <div className="topbar-left">
          <div className="logo">
            <div className="logo-mark"><span>R</span></div>
            <span>RNIA</span>
          </div>
        </div>
        <div className="search">
          <Icon name="search" size={14} />
          <input placeholder={`Search ${summary.total_articles?.toLocaleString() || ''} articles, tickers, events…`} />
          <span className="kbd">⌘K</span>
        </div>
        <div className="topbar-right">
          <div className={`live-pill ${apiOk ? '' : 'offline'}`}>
            <span className="live-dot" />
            {apiOk ? 'Live · connected' : 'Offline'}
          </div>
          <button className="icon-btn" title="Refresh" onClick={loadData} disabled={loading}>
            <Icon name="refresh" size={15} />
          </button>
          <button className="icon-btn" title="Notifications">
            <Icon name="bell" size={15} />
          </button>
          <button
            className="icon-btn"
            title="Toggle theme"
            onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
          >
            <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={15} />
          </button>
          <div className="avatar">RT</div>
        </div>
      </div>

      {/* SIDEBAR */}
      <aside className="sidebar">
        {SECTIONS.map(sec => (
          <Fragment key={sec.id}>
            <div className="nav-section">{sec.label}</div>
            {NAV_ITEMS.filter(n => n.section === sec.id).map(n => (
              <button
                key={n.id}
                className={`nav-item ${view === n.id ? 'active' : ''}`}
                onClick={() => setView(n.id)}
              >
                <Icon name={n.icon} size={15} />
                <span>{n.label}</span>
                {(n.id === 'feed' ? feedBadge : n.id === 'watchlist' && watchlist.length > 0 ? String(watchlist.length) : n.badge) && (
                  <span className="nav-badge">{n.id === 'feed' ? feedBadge : n.id === 'watchlist' && watchlist.length > 0 ? String(watchlist.length) : n.badge}</span>
                )}
              </button>
            ))}
          </Fragment>
        ))}
        <div className="sidebar-footer">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ width: 6, height: 6, background: apiOk ? 'var(--emerald)' : 'var(--bear)', borderRadius: '50%' }} />
            <span style={{ color: 'var(--ink-2)', fontSize: 11.5 }}>{apiOk ? 'API connected' : 'API offline'}</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--ink-3)' }}>
            v0.5.0 · {articles.length} articles loaded
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main">
        <div key={view}>{renderView()}</div>
      </main>
    </div>
  );
}
