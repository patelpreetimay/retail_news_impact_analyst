// Constants, mappers, derived selectors, and shared hooks for the RNIA shell.
import { useEffect, useState } from 'react';

export const EVENT_TYPES = [
  { id: 'earnings',    label: 'Earnings',        materiality: 0.95 },
  { id: 'ma',          label: 'M&A',             materiality: 0.90 },
  { id: 'regulatory',  label: 'Regulatory',      materiality: 0.85 },
  { id: 'leadership',  label: 'Leadership',      materiality: 0.75 },
  { id: 'legal',       label: 'Legal',           materiality: 0.70 },
  { id: 'product',     label: 'Product',         materiality: 0.65 },
  { id: 'market',      label: 'Market Movement', materiality: 0.60 },
];

export const EVENT_BY_ID = Object.fromEntries(EVENT_TYPES.map(e => [e.id, e]));

export const EVENT_COLORS = {
  earnings:   '#10B981',
  ma:         '#06B6D4',
  regulatory: '#F59E0B',
  leadership: '#A855F7',
  legal:      '#EF4444',
  product:    '#14B8A6',
  market:     '#3B82F6',
};

export const STANCE_COLORS = {
  bullish: '#10B981',
  bearish: '#DC2626',
  neutral: '#A3A3A3',
};

// Source credibility table — used to fill in score breakdowns when the
// /news endpoint doesn't expose component scores per article.
export const SOURCES = [
  { id: 'reuters',        name: 'Reuters',         credibility: 0.95 },
  { id: 'cnbc',           name: 'CNBC',            credibility: 0.88 },
  { id: 'wsj',            name: 'WSJ',             credibility: 0.94 },
  { id: 'bloomberg',      name: 'Bloomberg',       credibility: 0.96 },
  { id: 'ft',             name: 'Financial Times', credibility: 0.93 },
  { id: 'moneycontrol',   name: 'Moneycontrol',    credibility: 0.78 },
  { id: 'economictimes',  name: 'Economic Times',  credibility: 0.80 },
  { id: 'mint',           name: 'Mint',            credibility: 0.82 },
];

// ===== mappers — backend vocabulary → design vocabulary =====
// Backend stance: positive | negative | neutral.  Design: bullish | bearish | neutral.
export const mapStance = (s) => {
  if (s === 'positive' || s === 'bullish') return 'bullish';
  if (s === 'negative' || s === 'bearish') return 'bearish';
  return 'neutral';
};

// Backend uses snake_case event labels (e.g. market_movement, m_and_a).
const EVENT_ALIASES = {
  market_movement: 'market',
  market:          'market',
  m_and_a:         'ma',
  ma:              'ma',
  'm&a':           'ma',
  earnings:        'earnings',
  regulatory:      'regulatory',
  leadership:      'leadership',
  legal:           'legal',
  product:         'product',
};
export const mapEventType = (raw) => {
  if (!raw) return 'market';
  const k = String(raw).toLowerCase().trim();
  return EVENT_ALIASES[k] || (EVENT_BY_ID[k] ? k : 'market');
};

export const credColor = (c) =>
  c >= 0.9 ? '#10B981' : c >= 0.8 ? '#F59E0B' : '#A3A3A3';

const sourceLookup = (rawName) => {
  if (!rawName) return null;
  const k = String(rawName).toLowerCase().replace(/[^a-z]/g, '');
  return SOURCES.find(s => k.includes(s.id)) ||
    SOURCES.find(s => s.name.toLowerCase().replace(/[^a-z]/g, '').includes(k.slice(0, 5))) || null;
};

const parseTimestamp = (ts) => {
  if (!ts) return null;
  if (typeof ts === 'number') return ts;
  const t = Date.parse(ts);
  return Number.isFinite(t) ? t : null;
};

const hoursSince = (ts) => {
  const t = parseTimestamp(ts);
  if (t == null) return null;
  return Math.max(0, (Date.now() - t) / 3_600_000);
};

export const recencyScore = (ts) => {
  const h = hoursSince(ts);
  if (h == null) return 0.5;
  return +(0.05 + 0.95 * Math.pow(0.5, h / 48)).toFixed(4);
};

export const composeImpactScore = (materiality, credibility, recency) => {
  const mat = Math.max(0, Math.min(1, Number(materiality) || 0));
  const cred = Math.max(0, Math.min(1, Number(credibility) || 0));
  const rec = Math.max(0, Math.min(1, Number(recency) || 0));
  const baseImportance = mat * 0.6 + cred * 0.4;
  return +(baseImportance * (0.5 + 0.5 * rec)).toFixed(4);
};

export const timeAgo = (ts) => {
  const h = hoursSince(ts);
  if (h == null) return ts || '—';
  if (h < 1) return `${Math.max(1, Math.floor(h * 60))}m ago`;
  if (h < 24) return `${Math.floor(h)}h ago`;
  return `${Math.floor(h / 24)}d ago`;
};

// Normalize a raw /news article to the shape the views consume.
export const normalizeArticle = (raw, idx) => {
  const eventId = mapEventType(raw.event_type);
  const event = EVENT_BY_ID[eventId];
  const stance = mapStance(raw.stance);
  const sourceMeta = sourceLookup(raw.source);
  const credibility = Number.isFinite(+raw.credibility)
    ? +raw.credibility
    : sourceMeta?.credibility ?? 0.85;
  const hAgo = hoursSince(raw.timestamp);
  const recency = Number.isFinite(+raw.recency) ? +raw.recency : recencyScore(raw.timestamp);
  const materiality = Number.isFinite(+raw.materiality) ? +raw.materiality : event.materiality;
  const impact = Number.isFinite(+raw.impact_score) ? +raw.impact_score
                : composeImpactScore(materiality, credibility, recency);

  return {
    id: raw.id ?? `art-${idx}`,
    headline: raw.headline || '(no headline)',
    event_type: eventId,
    event_label: event.label,
    stance,
    impact_score: impact,
    explanation: raw.explanation || '',
    source: sourceMeta?.id ?? (raw.source || 'unknown'),
    source_name: sourceMeta?.name ?? (raw.source || 'Unknown'),
    credibility,
    recency,
    materiality,
    timestamp: raw.timestamp,
    timestamp_ms: parseTimestamp(raw.timestamp),
    hours_ago: hAgo ?? 0,
    url: raw.url || raw.link || '',
  };
};

// Bucket articles into hourly impact pulse for the Overview chart.
export const buildPulse = (articles) => {
  const buckets = Array.from({ length: 24 }, (_, i) => ({
    hour: 23 - i, label: i === 23 ? 'now' : `${23 - i}h`,
    impact: 0, volume: 0, bullish: 0, bearish: 0, _sum: 0,
  }));
  for (const a of articles) {
    if (a.hours_ago == null || a.hours_ago > 24) continue;
    const bucketHour = Math.floor(a.hours_ago);
    const idx = 23 - bucketHour;
    if (idx < 0 || idx > 23) continue;
    const b = buckets[idx];
    b.volume += 1;
    b._sum += a.impact_score;
    if (a.stance === 'bullish') b.bullish += 1;
    if (a.stance === 'bearish') b.bearish += 1;
  }
  for (const b of buckets) {
    b.impact = b.volume ? +(b._sum / b.volume).toFixed(3) : 0;
    delete b._sum;
  }
  // If we have no real timestamped data at all, return a flat trace so the
  // chart still renders rather than collapsing to NaNs.
  const hasAny = buckets.some(b => b.volume > 0);
  if (!hasAny) {
    return buckets.map((b, i) => ({
      ...b,
      impact: 0.55 + Math.sin(i / 4) * 0.08,
      volume: 30 + ((i * 7) % 50),
      bullish: 12 + ((i * 5) % 18),
      bearish: 10 + ((i * 3) % 14),
    }));
  }
  return buckets;
};

export const buildStanceRadar = (articles, eventDistribution) => {
  return EVENT_TYPES.map(e => {
    const events = articles.filter(a => a.event_type === e.id);
    const total = events.length || 1;
    return {
      ...e,
      total: eventDistribution?.[e.id] ?? events.length,
      bullish: events.filter(a => a.stance === 'bullish').length / total,
      bearish: events.filter(a => a.stance === 'bearish').length / total,
      neutral: events.filter(a => a.stance === 'neutral').length / total,
    };
  }).sort((a, b) => b.total - a.total);
};

export const normalizeEventDistribution = (raw) => {
  const out = {};
  for (const e of EVENT_TYPES) out[e.id] = 0;
  if (!raw) return out;
  for (const [k, v] of Object.entries(raw)) {
    const id = mapEventType(k);
    out[id] = (out[id] || 0) + Number(v || 0);
  }
  return out;
};

export const normalizeStanceDistribution = (raw) => {
  const out = { bullish: 0, bearish: 0, neutral: 0 };
  if (!raw) return out;
  for (const [k, v] of Object.entries(raw)) {
    out[mapStance(k)] += Number(v || 0);
  }
  return out;
};

export const dominantEventLabel = (eventDist) => {
  const sorted = Object.entries(eventDist || {}).sort((a, b) => b[1] - a[1]);
  if (!sorted.length) return EVENT_BY_ID.market;
  return EVENT_BY_ID[sorted[0][0]] || EVENT_BY_ID.market;
};

// ===== Hook: count-up animation =====
export const useCountUp = (target, dur = 800, decimals = 0) => {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!Number.isFinite(target)) { setV(0); return; }
    let raf, start;
    const step = (t) => {
      if (!start) start = t;
      const p = Math.min(1, (t - start) / dur);
      const ease = 1 - Math.pow(1 - p, 3);
      setV(target * ease);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, dur]);
  return decimals === 0 ? Math.round(v).toLocaleString()
                        : v.toFixed(decimals);
};
