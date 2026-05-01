import { useMemo, useState, Fragment } from 'react';
import { Icon } from './Icon';
import { EventBadge, ScoreBreakdown, StanceBadge } from './widgets';
import { EVENT_TYPES, SOURCES, credColor, timeAgo } from './data';
import { exportArticlesCSV } from './exportCSV';

export const NewsFeed = ({ articles, watchlist = [], onToggleWatchlist = () => {}, total = null, title = <>News <em>feed</em></>, sub = <>Live, filterable, multi-select. Press <span className="kbd">/</span> to search · <span className="kbd">J</span>/<span className="kbd">K</span> to navigate</> }) => {
  const [eventFilter, setEventFilter]   = useState('all');
  const [stanceFilter, setStanceFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [minImpact, setMinImpact]       = useState(0);
  const [search, setSearch]             = useState('');
  const [expanded, setExpanded]         = useState(null);
  const [sort, setSort]                 = useState('impact');
  const [visible, setVisible]           = useState(40);

  const filtered = useMemo(() => {
    let arr = [...articles];
    if (eventFilter  !== 'all') arr = arr.filter(a => a.event_type === eventFilter);
    if (stanceFilter !== 'all') arr = arr.filter(a => a.stance === stanceFilter);
    if (sourceFilter !== 'all') arr = arr.filter(a => a.source === sourceFilter);
    if (minImpact > 0) arr = arr.filter(a => a.impact_score >= minImpact);
    if (search) {
      const s = search.toLowerCase();
      arr = arr.filter(a => a.headline.toLowerCase().includes(s));
    }
    if (sort === 'impact') arr.sort((a, b) => b.impact_score - a.impact_score);
    if (sort === 'recent') arr.sort((a, b) => (b.timestamp_ms || 0) - (a.timestamp_ms || 0));
    return arr;
  }, [articles, eventFilter, stanceFilter, sourceFilter, minImpact, search, sort]);

  const reset = () => {
    setEventFilter('all'); setStanceFilter('all'); setSourceFilter('all');
    setMinImpact(0); setSearch('');
  };

  const handleOpenOriginal = (a) => {
    if (a.url) window.open(a.url, '_blank', 'noopener,noreferrer');
    else alert('Original link not available.');
  };

  const handleExportCSV = () => {
    const stamp = new Date().toISOString().slice(0, 10);
    exportArticlesCSV(filtered, `rnia-feed-${stamp}.csv`);
  };

  const totalCount = total ?? articles.length;

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1 className="view-title">{title}</h1>
          <div className="view-sub">{sub}</div>
        </div>
        <div className="view-actions">
          <button className="btn" onClick={handleExportCSV}><Icon name="download" size={14} />Export CSV</button>
        </div>
      </div>

      <div className="filter-rail">
        <select className="filter-pill" value={eventFilter} onChange={e => setEventFilter(e.target.value)}>
          <option value="all">All events</option>
          {EVENT_TYPES.map(e => <option key={e.id} value={e.id}>{e.label}</option>)}
        </select>
        <select className="filter-pill" value={stanceFilter} onChange={e => setStanceFilter(e.target.value)}>
          <option value="all">All stances</option>
          <option value="bullish">↑ Bullish</option>
          <option value="bearish">↓ Bearish</option>
          <option value="neutral">─ Neutral</option>
        </select>
        <select className="filter-pill" value={sourceFilter} onChange={e => setSourceFilter(e.target.value)}>
          <option value="all">All sources</option>
          {SOURCES.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
        <div className="filter-pill" style={{ gap: 10 }}>
          <span style={{ fontSize: 11, color: 'var(--ink-3)' }}>Impact ≥</span>
          <input
            type="range" min="0" max="1" step="0.05"
            value={minImpact}
            onChange={e => setMinImpact(+e.target.value)}
            style={{ width: 80, accentColor: 'var(--teal-bright)' }}
          />
          <span className="mono" style={{ fontSize: 11 }}>{minImpact.toFixed(2)}</span>
        </div>
        <div className="filter-search">
          <Icon name="search" size={13} style={{ color: 'var(--ink-3)' }} />
          <input
            placeholder="Search headlines…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
        <div className="filter-meta">
          {filtered.length.toLocaleString()} of {totalCount.toLocaleString()} · sort:{' '}
          <span
            style={{ color: 'var(--ink)', cursor: 'pointer' }}
            onClick={() => setSort(sort === 'impact' ? 'recent' : 'impact')}
          >
            {sort === 'impact' ? 'Impact ↓' : 'Recent ↓'}
          </span>
        </div>
      </div>

      <div className="feed-table">
        <div className="feed-th">
          <div>Impact</div>
          <div>Headline</div>
          <div>Event</div>
          <div>Stance</div>
          <div>Source</div>
          <div>Published</div>
          <div></div>
        </div>
        {filtered.length === 0 ? (
          <div style={{ padding: 56, textAlign: 'center', color: 'var(--ink-2)' }}>
            <div style={{ marginBottom: 8, opacity: 0.5 }}><Icon name="search" size={32} /></div>
            <div>No matches. Try resetting filters.</div>
            <div style={{ marginTop: 14, display: 'flex', gap: 6, justifyContent: 'center' }}>
              <button className="filter-pill" onClick={reset}>Clear all filters</button>
            </div>
          </div>
        ) : filtered.slice(0, visible).map(a => {
          const inWatchlist = watchlist.includes(a.id);
          return (
          <Fragment key={a.id}>
            <div
              className={`feed-row ${expanded === a.id ? 'expanded' : ''}`}
              onClick={() => setExpanded(expanded === a.id ? null : a.id)}
            >
              <div className="feed-impact">
                <span
                  className="v impact-grad"
                  style={{
                    background: 'var(--grad-num)', WebkitBackgroundClip: 'text',
                    backgroundClip: 'text', color: 'transparent',
                  }}
                >
                  {a.impact_score.toFixed(2)}
                </span>
                <div className="bar"><div className="bar-fill" style={{ width: `${a.impact_score * 100}%` }} /></div>
              </div>
              <div className="feed-headline-cell" title={a.headline}>{a.headline}</div>
              <div><EventBadge id={a.event_type} label={a.event_label} /></div>
              <div><StanceBadge stance={a.stance} /></div>
              <div className="feed-source">
                <span className="cred-dot" style={{ background: credColor(a.credibility) }} />
                {a.source_name}
              </div>
              <div className="feed-time">{timeAgo(a.timestamp)}</div>
              <div className="feed-chev"><Icon name="chevron" size={14} /></div>
            </div>
            {expanded === a.id && (
              <div className="feed-expand">
                <div>
                  <div className="feed-expand-headline">{a.headline}</div>
                  <div className="feed-expand-explanation">
                    {a.explanation || 'No explanation available for this article.'}
                  </div>
                  <div className="feed-expand-actions">
                    <button className={`btn ${inWatchlist ? 'active' : ''}`} onClick={() => onToggleWatchlist(a.id)}>
                      <Icon name="bookmark" size={13} fill={inWatchlist ? 'currentColor' : 'none'} />
                      {inWatchlist ? 'Remove from watchlist' : 'Add to watchlist'}
                    </button>
                    <button className="btn" onClick={() => handleOpenOriginal(a)}>
                      <Icon name="arrowRight" size={13} />Open original
                    </button>
                  </div>
                </div>
                <ScoreBreakdown a={a} />
              </div>
            )}
          </Fragment>
        )})}
      </div>

      {filtered.length > visible && (
        <div style={{ textAlign: 'center', padding: 18, color: 'var(--ink-3)', fontSize: 12 }}>
          Showing {visible} of {filtered.length.toLocaleString()} ·{' '}
          <span
            style={{ color: 'var(--teal-bright)', cursor: 'pointer' }}
            onClick={() => setVisible(v => v + 40)}
          >
            Load more
          </span>
        </div>
      )}
    </div>
  );
};
