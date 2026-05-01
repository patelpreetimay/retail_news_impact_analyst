import { useEffect, useRef, useState, useId } from 'react';
import { EVENT_COLORS, STANCE_COLORS, useCountUp } from './data';
import { Icon } from './Icon';

// ===== Stance + Event badges =====
export const StanceBadge = ({ stance }) => {
  const cls = stance === 'bullish' ? 'badge-bullish'
            : stance === 'bearish' ? 'badge-bearish'
            : 'badge-neutral';
  const label = stance.charAt(0).toUpperCase() + stance.slice(1);
  const arrow = stance === 'bullish' ? '↑' : stance === 'bearish' ? '↓' : '─';
  return (
    <span className={`badge ${cls}`}>
      <span style={{ fontSize: 10 }}>{arrow}</span>{label}
    </span>
  );
};

export const EventBadge = ({ id, label }) => (
  <span className="badge badge-event" style={{ '--ev-color': EVENT_COLORS[id] || '#3B82F6' }}>
    {label}
  </span>
);

// ===== Sparkline =====
export const Sparkline = ({ values, w = 120, h = 32, color = '#10B981' }) => {
  const id = useId().replace(/:/g, '');
  if (!values || values.length < 2) return <svg width={w} height={h}/>;
  const max = Math.max(...values), min = Math.min(...values);
  const range = max - min || 1;
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * w},${h - ((v - min) / range) * (h - 4) - 2}`)
    .join(' ');
  const area = `0,${h} ${pts} ${w},${h}`;
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#${id})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
};

// ===== Impact pulse area chart with stance dots =====
export const PulseChart = ({ data, w = 720, h = 220 }) => {
  const pad = { l: 36, r: 16, t: 16, b: 28 };
  const W = w - pad.l - pad.r;
  const H = h - pad.t - pad.b;
  const yMin = 0.3, yMax = 0.95;
  const x = (i) => pad.l + (i / (data.length - 1)) * W;
  const y = (v) => pad.t + H - ((v - yMin) / (yMax - yMin)) * H;

  const pts = data.map((d, i) => `${x(i)},${y(d.impact || yMin)}`).join(' ');
  const area = `${pad.l},${pad.t + H} ${pts} ${pad.l + W},${pad.t + H}`;

  const [progress, setProgress] = useState(0);
  useEffect(() => {
    let raf, start;
    const step = (t) => {
      if (!start) start = t;
      const p = Math.min(1, (t - start) / 900);
      setProgress(1 - Math.pow(1 - p, 3));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [data]);

  const yTicks = [0.4, 0.55, 0.7, 0.85];

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: '100%', height: h }}>
        <defs>
          <linearGradient id="pulse-grad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#10B981" stopOpacity="0.42" />
            <stop offset="60%" stopColor="#06B6D4" stopOpacity="0.14" />
            <stop offset="100%" stopColor="#06B6D4" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="pulse-line" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#0F6B5C" />
            <stop offset="50%" stopColor="#10B981" />
            <stop offset="100%" stopColor="#06B6D4" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2.5" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <clipPath id="pulse-clip">
            <rect x={pad.l} y={pad.t} width={W * progress} height={H + 6} />
          </clipPath>
        </defs>

        {yTicks.map(t => (
          <g key={t}>
            <line x1={pad.l} x2={pad.l + W} y1={y(t)} y2={y(t)}
                  stroke="currentColor" strokeOpacity="0.08" strokeDasharray="2 4" />
            <text x={pad.l - 8} y={y(t) + 3} textAnchor="end"
                  fontSize="9" fill="currentColor" fillOpacity="0.45"
                  fontFamily="JetBrains Mono">
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        {data.filter((_, i) => i % 4 === 0).map((d, i) => (
          <text key={i} x={x(i * 4)} y={h - 8} textAnchor="middle"
                fontSize="9" fill="currentColor" fillOpacity="0.45"
                fontFamily="JetBrains Mono">
            {d.label}
          </text>
        ))}

        <g clipPath="url(#pulse-clip)">
          <polygon points={area} fill="url(#pulse-grad)" />
          <polyline points={pts} fill="none" stroke="url(#pulse-line)"
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    filter="url(#glow)" />
        </g>

        {data.map((d, i) => {
          const opacity = (i / data.length) * progress;
          const cx = x(i), cy = y(d.impact || yMin);
          if (d.bullish > d.bearish) {
            return <circle key={i} cx={cx} cy={cy} r="3" fill="#10B981" opacity={opacity * 0.9} />;
          }
          if (d.bearish > d.bullish + 3) {
            return <circle key={i} cx={cx} cy={cy} r="3" fill="#DC2626" opacity={opacity * 0.9} />;
          }
          return <circle key={i} cx={cx} cy={cy} r="2" fill="#A3A3A3" opacity={opacity * 0.6} />;
        })}
      </svg>
    </div>
  );
};

// ===== KPI tile =====
export const KpiTile = ({ label, value, decimals = 0, sub, icon, hero, onClick, children, valueAs }) => {
  const animated = useCountUp(value, 800, decimals);
  return (
    <div className={`card card-hover kpi ${hero ? 'kpi-hero' : ''}`} onClick={onClick}>
      <div className="kpi-head">
        <span className="card-label">{label}</span>
        {icon && <Icon name={icon} size={16} className="kpi-icon" />}
      </div>
      {valueAs ? (
        <div className="kpi-value tnum">{valueAs}</div>
      ) : (
        <div
          className="kpi-value tnum"
          style={hero ? {} : {
            background: 'var(--grad-num)',
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
            color: 'transparent',
            display: 'inline-block',
          }}
        >
          {animated}
        </div>
      )}
      {sub && <div className="kpi-meta">{sub}</div>}
      {children}
    </div>
  );
};

// ===== Score breakdown: base importance with recency decay =====
export const ScoreBreakdown = ({ a }) => (
  <div className="breakdown">
    <div className="breakdown-head">
      <span className="breakdown-title">Score breakdown</span>
      <span
        className="breakdown-total impact-grad"
        style={{
          background: 'var(--grad-num)',
          WebkitBackgroundClip: 'text',
          backgroundClip: 'text',
          color: 'transparent',
        }}
      >
        {a.impact_score.toFixed(2)}
      </span>
    </div>
    <div className="bd-row">
      <div className="bd-label">Credibility<span className="w">base weight 0.40</span></div>
      <div className="bd-bar"><div className="bd-fill cred" style={{ width: `${a.credibility * 100}%` }} /></div>
      <div className="bd-value">{a.credibility.toFixed(2)}</div>
    </div>
    <div className="bd-row">
      <div className="bd-label">Recency<span className="w">live decay</span></div>
      <div className="bd-bar"><div className="bd-fill recency" style={{ width: `${a.recency * 100}%` }} /></div>
      <div className="bd-value">{a.recency.toFixed(2)}</div>
    </div>
    <div className="bd-row">
      <div className="bd-label">Materiality<span className="w">base weight 0.60</span></div>
      <div className="bd-bar"><div className="bd-fill materiality" style={{ width: `${a.materiality * 100}%` }} /></div>
      <div className="bd-value">{a.materiality.toFixed(2)}</div>
    </div>
    <div className="bd-formula">
      ({a.materiality.toFixed(2)} x 0.6 + {a.credibility.toFixed(2)} x 0.4) x (0.5 + {a.recency.toFixed(2)} x 0.5) ={' '}
      <span style={{ color: 'var(--teal-bright)' }}>{a.impact_score.toFixed(3)}</span>
    </div>
  </div>
);

export { STANCE_COLORS, EVENT_COLORS };
