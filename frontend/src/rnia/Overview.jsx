import { useMemo, useState } from 'react';
import { Icon } from './Icon';
import {
  EventBadge, KpiTile, PulseChart, ScoreBreakdown, Sparkline, StanceBadge,
  EVENT_COLORS, STANCE_COLORS,
} from './widgets';
import {
  EVENT_TYPES, dominantEventLabel, timeAgo, useCountUp,
} from './data';
import { exportArticlesCSV } from './exportCSV';

const RANGE_OPTIONS = [
  { id: 'today', label: 'Today',     hours: 24 },
  { id: 'week',  label: 'This week', hours: 24 * 7 },
  { id: 'month', label: 'This month', hours: 24 * 30 },
  { id: 'all',   label: 'All time',  hours: Infinity },
];

export const Overview = ({ summary, articles, pulseData, stanceRadar, onJump }) => {
  const [rangeId, setRangeId] = useState('today');
  const range = RANGE_OPTIONS.find(r => r.id === rangeId) || RANGE_OPTIONS[0];

  const filteredArticles = useMemo(() => {
    if (!Number.isFinite(range.hours)) return articles;
    return articles.filter(a => (a.hours_ago ?? Infinity) <= range.hours);
  }, [articles, range.hours]);

  const cycleRange = () => {
    const idx = RANGE_OPTIONS.findIndex(r => r.id === rangeId);
    setRangeId(RANGE_OPTIONS[(idx + 1) % RANGE_OPTIONS.length].id);
  };

  const handleExportCSV = () => {
    const stamp = new Date().toISOString().slice(0, 10);
    exportArticlesCSV(filteredArticles, `rnia-${range.id}-${stamp}.csv`);
  };

  const stanceTotal = (summary.stance_distribution.bullish || 0)
                    + (summary.stance_distribution.bearish || 0)
                    + (summary.stance_distribution.neutral || 0);
  const eventTotal = Object.values(summary.event_distribution || {}).reduce((a, b) => a + b, 0);

  const top = dominantEventLabel(summary.event_distribution);
  const topCount = summary.event_distribution[top.id] || 0;
  const topPctNum = eventTotal ? (topCount / eventTotal) * 100 : 0;

  const calculatedAvgImpact = useMemo(() => {
    if (!filteredArticles || filteredArticles.length === 0) return 0;
    const sorted = [...filteredArticles].sort((a, b) => (a.hours_ago || 0) - (b.hours_ago || 0));
    const recent = sorted.filter(a => a.hours_ago <= 24);
    const target = recent.length > 0 ? recent : sorted.slice(0, 40);
    if (target.length === 0) return 0;
    return target.reduce((sum, a) => sum + (a.impact_score || 0), 0) / target.length;
  }, [filteredArticles]);

  const sparkVals = pulseData.map(p => p.volume || 0);
  const peak  = Math.max(...pulseData.map(p => p.impact || 0));
  const trough = Math.min(...pulseData.map(p => p.impact || 0).filter(Boolean));
  const volumeSum = pulseData.reduce((a, b) => a + (b.volume || 0), 0);

  const topMovers = useMemo(() =>
    [...filteredArticles].sort((a, b) => b.impact_score - a.impact_score).slice(0, 5),
  [filteredArticles]);

  const totalArt    = useCountUp(summary.total_articles, 900, 0);
  const totalAvg    = useCountUp(calculatedAvgImpact, 1000, 2);
  const bullishCnt  = useCountUp(summary.stance_distribution.bullish || 0, 900, 0);
  const topPct      = useCountUp(topPctNum, 900, 1);

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1 className="view-title">Command <em>deck</em></h1>
          <div className="view-sub">
            Real-time intelligence across {summary.total_articles.toLocaleString()} articles · {EVENT_TYPES.length} event classes
          </div>
        </div>
        <div className="view-actions">
          <button className="btn btn-ghost" onClick={cycleRange} title="Click to cycle time range">
            <Icon name="filter" size={14} />{range.label}
          </button>
          <button className="btn" onClick={handleExportCSV}><Icon name="download" size={14} />Export CSV</button>
          <button className="btn btn-primary" onClick={() => onJump('analyze')}>
            <Icon name="sparkles" size={14} />New Report
          </button>
        </div>
      </div>

      {/* KPI ROW */}
      <div className="kpi-grid stagger">
        {/* Articles ingested */}
        <div className="card card-hover kpi" onClick={() => onJump('feed')}>
          <div className="kpi-head">
            <span className="card-label">Articles ingested</span>
            <Icon name="layers" size={16} className="kpi-icon" />
          </div>
          <div
            className="kpi-value tnum impact-grad"
            style={{
              background: 'var(--grad-num)', WebkitBackgroundClip: 'text',
              backgroundClip: 'text', color: 'transparent', display: 'inline-block',
            }}
          >
            {totalArt}
          </div>
          <div className="kpi-meta">
            <span className="delta-up"><Icon name="arrowUp" size={11} />live ingest</span>
            <span style={{ color: 'var(--ink-3)' }}>·</span>
            <span style={{ color: 'var(--ink-3)' }}>last 24h</span>
          </div>
          <div className="kpi-spark">
            <Sparkline values={sparkVals} w={200} h={32} color="#10B981" />
          </div>
        </div>

        {/* Avg impact (hero gradient tile) */}
        <div
          className="card card-hover kpi kpi-hero"
          onClick={() => onJump('feed')}
          style={{ position: 'relative', zIndex: 1 }}
        >
          <div className="kpi-head" style={{ position: 'relative', zIndex: 2, alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span className="card-label">Avg impact score</span>
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.6)', marginTop: 2, letterSpacing: '0.02em' }}>of last 24 hours</span>
            </div>
            <Icon name="pulse" size={16} className="kpi-icon" />
          </div>
          <div className="kpi-value tnum" style={{ position: 'relative', zIndex: 2 }}>{totalAvg}</div>
          <div className="kpi-meta" style={{ position: 'relative', zIndex: 2 }}>
            <span style={{ color: 'rgba(255,255,255,0.95)' }}>
              <Icon name="pulse" size={11} /> weighted score
            </span>
            <span style={{ color: 'rgba(255,255,255,0.6)' }}>·</span>
            <span style={{ color: 'rgba(255,255,255,0.7)' }}>0–1 scale</span>
          </div>
          <div style={{ marginTop: 14, position: 'relative', zIndex: 2 }}>
            <div style={{ height: 6, background: 'rgba(255,255,255,0.15)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                width: `${(calculatedAvgImpact || 0) * 100}%`,
                height: '100%',
                background: 'linear-gradient(90deg, rgba(255,255,255,0.5), white)',
                borderRadius: 3,
              }}/>
            </div>
          </div>
        </div>

        {/* Stance split */}
        <div className="card card-hover kpi" onClick={() => onJump('feed')}>
          <div className="kpi-head">
            <span className="card-label">Market stance</span>
            <Icon name="trending" size={16} className="kpi-icon" />
          </div>
          <div className="kpi-value tnum" style={{ display: 'flex', alignItems: 'baseline', gap: 8, fontSize: 30 }}>
            <span style={{ color: 'var(--bull)' }}>{bullishCnt}</span>
            <span style={{ fontSize: 11, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)' }}>bullish</span>
          </div>
          <div className="kpi-stance-bars">
            {[
              { k: 'bullish', l: 'Bullish', sym: '↑' },
              { k: 'bearish', l: 'Bearish', sym: '↓' },
              { k: 'neutral', l: 'Neutral', sym: '─' },
            ].map(s => {
              const v = summary.stance_distribution[s.k] || 0;
              return (
                <div className="stance-row" key={s.k}>
                  <span style={{ color: STANCE_COLORS[s.k], fontFamily: 'var(--font-mono)', fontSize: 11 }}>{s.sym}</span>
                  <div className="stance-track">
                    <div className="stance-fill" style={{
                      width: `${stanceTotal ? (v / stanceTotal) * 100 : 0}%`,
                      background: STANCE_COLORS[s.k],
                    }}/>
                  </div>
                  <span className="mono">{v.toLocaleString()}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Top catalyst */}
        <div className="card card-hover kpi" onClick={() => onJump('events')}>
          <div className="kpi-head">
            <span className="card-label">Top catalyst</span>
            <Icon name="radio" size={16} className="kpi-icon" />
          </div>
          <div className="kpi-value" style={{ fontSize: 28, lineHeight: 1.05 }}>
            {top.label.split(' ')[0]}<br/>
            <em style={{ fontStyle: 'italic', color: 'var(--teal-bright)' }}>
              {top.label.split(' ').slice(1).join(' ') || 'class'}
            </em>
          </div>
          <div className="kpi-meta">
            <span className="mono" style={{ color: 'var(--ink)' }}>{topPct}%</span>
            <span style={{ color: 'var(--ink-3)' }}>·</span>
            <span>{topCount.toLocaleString()} articles</span>
          </div>
          <div className="kpi-bar-track" style={{ marginTop: 14 }}>
            <div className="kpi-bar-fill" style={{ width: `${topPctNum}%` }}/>
          </div>
        </div>
      </div>

      {/* MID ROW */}
      <div className="mid-row">
        <div className="card pulse-card">
          <div className="pulse-head">
            <div>
              <div className="pulse-title">
                Impact <em style={{ fontStyle: 'italic', color: 'var(--teal-bright)' }}>Pulse</em>
              </div>
              <div className="pulse-sub">Mean impact score, hourly · last 24h · stance overlay</div>
            </div>
            <div className="pulse-stats">
              <div className="pulse-stat">
                <div className="l">Peak</div>
                <div
                  className="v impact-grad"
                  style={{
                    background: 'var(--grad-num)', WebkitBackgroundClip: 'text',
                    backgroundClip: 'text', color: 'transparent',
                  }}
                >
                  {peak.toFixed(2)}
                </div>
              </div>
              <div className="pulse-stat">
                <div className="l">Trough</div>
                <div className="v">{Number.isFinite(trough) ? trough.toFixed(2) : '0.00'}</div>
              </div>
              <div className="pulse-stat">
                <div className="l">Volume</div>
                <div className="v">{volumeSum}</div>
              </div>
            </div>
          </div>
          <div className="pulse-chart">
            <PulseChart data={pulseData} />
          </div>

          <div className="event-ribbon">
            <div className="event-ribbon-head">
              <span>Event distribution</span>
              <span style={{ color: 'var(--ink-3)' }}>hover for detail</span>
            </div>
            <div className="ribbon">
              {EVENT_TYPES.map(e => {
                const count = summary.event_distribution[e.id] || 0;
                const pct = eventTotal ? (count / eventTotal) * 100 : 0;
                if (pct < 0.3) return null;
                return (
                  <div
                    key={e.id}
                    className="ribbon-seg"
                    style={{ width: `${pct}%`, background: EVENT_COLORS[e.id] }}
                    title={`${e.label}: ${count} (${pct.toFixed(1)}%)`}
                  >
                    {pct > 8 ? `${pct.toFixed(0)}%` : ''}
                  </div>
                );
              })}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 8, fontSize: 11, color: 'var(--ink-2)' }}>
              {EVENT_TYPES.map(e => (
                <span key={e.id} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 8, height: 8, background: EVENT_COLORS[e.id], borderRadius: 2 }} />
                  {e.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        <div className="card movers-card">
          <div className="movers-head">
            <div>
              <div className="pulse-title">Top movers</div>
              <div className="pulse-sub">Highest impact, last 24h</div>
            </div>
            <button className="btn btn-ghost" style={{ fontSize: 11 }} onClick={() => onJump('feed')}>
              View all<Icon name="arrowRight" size={12} />
            </button>
          </div>
          {topMovers.length === 0 && (
            <div style={{ padding: 24, textAlign: 'center', color: 'var(--ink-3)', fontSize: 12 }}>
              No articles yet.
            </div>
          )}
          {topMovers.map((a) => (
            <div key={a.id} className="mover-row" onClick={() => onJump('feed')}>
              <div
                className="mover-score impact-grad"
                style={{
                  background: 'var(--grad-num)', WebkitBackgroundClip: 'text',
                  backgroundClip: 'text', color: 'transparent',
                }}
              >
                {a.impact_score.toFixed(2)}
              </div>
              <div>
                <div className="mover-headline">{a.headline}</div>
                <div className="mover-meta">
                  <EventBadge id={a.event_type} label={a.event_label} />
                  <StanceBadge stance={a.stance} />
                  <span className="dot">·</span>
                  <span>{a.source_name}</span>
                  <span className="dot">·</span>
                  <span className="mono">{timeAgo(a.timestamp)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* STANCE RADAR */}
      <div className="card radar-card">
        <div className="radar-head">
          <div>
            <div className="pulse-title">
              Market stance <em style={{ fontStyle: 'italic', color: 'var(--teal-bright)' }}>radar</em>
            </div>
            <div className="pulse-sub">Per-event bullish · bearish · neutral split, ranked by volume</div>
          </div>
          <div style={{ display: 'flex', gap: 14, fontSize: 11, color: 'var(--ink-2)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span className="radar-dot" style={{ background: STANCE_COLORS.bullish }} />Bullish
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span className="radar-dot" style={{ background: STANCE_COLORS.bearish }} />Bearish
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span className="radar-dot" style={{ background: STANCE_COLORS.neutral }} />Neutral
            </span>
          </div>
        </div>
        <div>
          {stanceRadar.map((r, i) => (
            <div key={r.id} className="radar-row" style={{ borderTop: i === 0 ? 0 : '1px solid var(--border)' }}>
              <div className="radar-grid" onClick={() => onJump('events')}>
                <div className="radar-label">
                  <span className="radar-dot" style={{ background: EVENT_COLORS[r.id] }} />
                  {r.label}
                </div>
                <div className="radar-vol mono">{(r.total || 0).toLocaleString()}</div>
                <div className="radar-bar">
                  <div className="radar-seg"
                    style={{ width: `${r.bullish * 100}%`, background: STANCE_COLORS.bullish }}
                    title={`${(r.bullish * 100).toFixed(0)}% bullish`} />
                  <div className="radar-seg"
                    style={{ width: `${r.neutral * 100}%`, background: STANCE_COLORS.neutral }}
                    title={`${(r.neutral * 100).toFixed(0)}% neutral`} />
                  <div className="radar-seg"
                    style={{ width: `${r.bearish * 100}%`, background: STANCE_COLORS.bearish }}
                    title={`${(r.bearish * 100).toFixed(0)}% bearish`} />
                </div>
                <div className="radar-mat mono">materiality {r.materiality.toFixed(2)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
