// Rich, theme-matched PDF brief export.
// Opens a new window with a fully styled print-ready document and triggers print.
// Both the news feed row export and the Analyze view's "Export PDF brief" button
// route through here so the output stays consistent.

import { EVENT_COLORS, EVENT_BY_ID } from './data';

const THEME = {
  ink:        '#0A0A0A',
  ink2:       '#525252',
  ink3:       '#A3A3A3',
  paper:      '#FAFAF7',
  surface:    '#FFFFFF',
  surface2:   '#F5F5F0',
  hairline:   '#E5E5E0',
  tealDeep:   '#0E4D45',
  teal:       '#0F6B5C',
  tealBright: '#14B8A6',
  cyan:       '#06B6D4',
  bull:       '#10B981',
  bear:       '#DC2626',
  neutral:    '#A3A3A3',
  amber:      '#F59E0B',
  gradHero:   'linear-gradient(135deg, #0E4D45 0%, #0F6B5C 35%, #10B981 75%, #06B6D4 100%)',
  gradNum:    'linear-gradient(180deg, #0E4D45 0%, #10B981 100%)',
};

const escapeHtml = (s) => String(s ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

const stanceMeta = (stance) => {
  if (stance === 'bullish') return { color: THEME.bull,    arrow: '↑', label: 'Bullish', pos: 88 };
  if (stance === 'bearish') return { color: THEME.bear,    arrow: '↓', label: 'Bearish', pos: 12 };
  return                          { color: THEME.neutral, arrow: '─', label: 'Neutral', pos: 50 };
};

const fmtPct = (v) => `${Math.round((Number(v) || 0) * 100)}%`;
const fmt2   = (v) => (Number(v) || 0).toFixed(2);

const formulaText = (mat, cred, rec, impact) =>
  `(${fmt2(mat)} × 0.6 + ${fmt2(cred)} × 0.4) × (0.5 + ${fmt2(rec)} × 0.5) = ${(Number(impact) || 0).toFixed(3)}`;

// LaTeX strings consumed by KaTeX at runtime (rendered into the placeholders
// in the impact-card; ASCII fallbacks ship inline so offline previews still read).
const formulaDefsTeX = () =>
  `M = \\text{materiality},\\quad C = \\text{credibility},\\quad R = \\text{recency}`;

const formulaDerivationTeX = (mat, cred, rec, impact) =>
  `\\begin{aligned}` +
  `I &= \\underbrace{(M \\cdot 0.6 + C \\cdot 0.4)}_{\\text{importance}} ` +
  `\\;\\cdot\\; ` +
  `\\underbrace{(0.5 + 0.5\\,R)}_{\\text{recency factor}} \\\\[4pt]` +
  `&= (${fmt2(mat)} \\cdot 0.6 + ${fmt2(cred)} \\cdot 0.4) ` +
  `\\;\\cdot\\; ` +
  `(0.5 + 0.5 \\cdot ${fmt2(rec)}) \\\\[2pt]` +
  `&= \\mathbf{${(Number(impact) || 0).toFixed(3)}}` +
  `\\end{aligned}`;

// Render a top-K probability stack (event or stance), each row a colored bar.
const topKBlock = (rows, colorOf, labelOf) => {
  if (!rows?.length) return '';
  return `
    <div class="topk">
      ${rows.map((t, i) => {
        const pct = Math.round((t.prob || 0) * 100);
        const isTop = i === 0;
        return `
          <div class="topk-row ${isTop ? 'top' : ''}">
            <div class="topk-label">${escapeHtml(labelOf(t))}</div>
            <div class="topk-track"><div class="topk-fill" style="width:${pct}%;background:${colorOf(t)};opacity:${isTop ? 1 : 0.42};"></div></div>
            <div class="topk-pct">${pct}%</div>
          </div>`;
      }).join('')}
    </div>`;
};

const matchedTerms = (terms) => {
  if (!terms?.length) return '';
  return `<div class="matched">Matched terms · <span class="mono">${escapeHtml(terms.slice(0, 8).join(', '))}</span></div>`;
};

// Build the document body. `a` accepts either a feed article or an analyze result.
const buildHTML = (a) => {
  const eventColor = EVENT_COLORS[a.event_type] || THEME.tealBright;
  const ev = EVENT_BY_ID[a.event_type] || { label: a.event_label || 'Market' };
  const stance = stanceMeta(a.stance);
  const impactPct = Math.round((Number(a.impact_score) || 0) * 100);
  const credibility = Number(a.credibility) || 0;
  const recency     = Number(a.recency)     || 0;
  const materiality = Number(a.materiality) || 0;

  const headline = a.headline || a.title || 'Custom analysis brief';
  const sourceName = a.source_name || a.source || 'RNIA Analyze';
  const dateStr = a.timestamp_ms
    ? new Date(a.timestamp_ms).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
    : new Date().toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });

  const evConfPct = a.event_confidence  != null ? fmtPct(a.event_confidence)  : null;
  const stConfPct = a.stance_confidence != null ? fmtPct(a.stance_confidence) : null;

  const eventTopK = topKBlock(
    a.event_top_k,
    (t) => EVENT_COLORS[t.id] || THEME.ink3,
    (t) => (EVENT_BY_ID[t.id]?.label) || t.id,
  );
  const stanceTopK = topKBlock(
    a.stance_top_k,
    (t) => t.id === 'bullish' ? THEME.bull : t.id === 'bearish' ? THEME.bear : THEME.neutral,
    (t) => t.id.charAt(0).toUpperCase() + t.id.slice(1),
  );

  const evMatched = matchedTerms(a.matched_keywords?.event);
  const stMatched = matchedTerms(a.matched_keywords?.stance);

  return `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>${escapeHtml(headline)} · RNIA brief</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg" crossorigin="anonymous"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: #FFFFFF;
    color: ${THEME.ink};
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .page {
    max-width: 760px;
    margin: 0 auto;
    padding: 0;
  }

  /* === Hero strip === */
  .hero {
    background: ${THEME.gradHero};
    color: #FAFAF7;
    padding: 12px 36px;
    display: flex; align-items: center; justify-content: space-between;
    letter-spacing: 0.02em;
  }
  .hero-mark {
    font-family: 'Fraunces', Georgia, 'Times New Roman', serif;
    font-style: italic; font-weight: 600; font-size: 18px;
    letter-spacing: -0.01em;
  }
  .hero-mark span { font-style: normal; font-weight: 700; }
  .hero-tag {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.18em;
    opacity: 0.85;
  }

  /* === Article header === */
  .article {
    padding: 20px 36px 0 36px;
  }
  .eyebrow {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.18em;
    color: ${THEME.ink3};
    margin-bottom: 8px;
    display: flex; gap: 12px; align-items: center;
  }
  .eyebrow .dot { width: 3px; height: 3px; border-radius: 50%; background: ${THEME.ink3}; }
  h1.headline {
    font-family: 'Fraunces', Georgia, 'Times New Roman', serif;
    font-weight: 600;
    font-size: 24px;
    line-height: 1.18;
    letter-spacing: -0.015em;
    margin: 0 0 12px 0;
    color: ${THEME.ink};
  }

  /* === Status row (event / stance / impact) === */
  .status {
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
    padding-bottom: 14px;
    border-bottom: 1px solid ${THEME.hairline};
  }
  .chip {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 11px;
    border-radius: 999px;
    font-size: 12px; font-weight: 500;
    border: 1px solid ${THEME.hairline};
    background: ${THEME.surface};
    color: ${THEME.ink};
  }
  .chip-event {
    border-left: 3px solid ${eventColor};
    padding-left: 10px;
  }
  .chip-event .swatch {
    width: 8px; height: 8px; border-radius: 2px; background: ${eventColor};
  }
  .chip-stance.bullish { color: ${THEME.bull}; border-color: rgba(16,185,129,0.32); background: rgba(16,185,129,0.08); }
  .chip-stance.bearish { color: ${THEME.bear}; border-color: rgba(220,38,38,0.32); background: rgba(220,38,38,0.08); }
  .chip-stance.neutral { color: ${THEME.ink2}; }
  .chip-stance .arrow { font-size: 11px; }
  .chip-impact {
    margin-left: auto;
    background: ${THEME.surface2};
    border-color: ${THEME.hairline};
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-feature-settings: 'tnum';
    font-weight: 500;
  }
  .chip-impact .v {
    background: ${THEME.gradNum};
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-weight: 700; font-size: 13px;
  }

  /* === Section === */
  .section { padding: 14px 36px 0 36px; }
  .section-eyebrow {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.18em;
    color: ${THEME.ink3};
    margin-bottom: 5px;
  }
  .section-headline {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600; font-size: 16.5px;
    line-height: 1.3;
    color: ${THEME.ink};
    margin: 0 0 4px 0;
  }
  .section-headline em {
    font-style: italic;
    font-weight: 600;
    color: ${THEME.tealBright};
  }
  .section-headline em.bull { color: ${THEME.bull}; }
  .section-headline em.bear { color: ${THEME.bear}; }
  .section-headline em.neutral { color: ${THEME.ink2}; }
  .section-meta {
    font-size: 11px; color: ${THEME.ink2}; margin-bottom: 6px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-feature-settings: 'tnum';
  }

  /* Top-K rows (event distribution / stance distribution) */
  .topk { margin-top: 6px; }
  .topk-row {
    display: grid;
    grid-template-columns: 110px 1fr 38px;
    align-items: center;
    gap: 8px;
    margin-bottom: 3px;
    font-size: 11px;
    color: ${THEME.ink3};
  }
  .topk-row.top { color: ${THEME.ink}; font-weight: 600; }
  .topk-track {
    height: 6px; border-radius: 4px;
    background: ${THEME.surface2};
    overflow: hidden;
  }
  .topk-fill { height: 100%; border-radius: 4px; }
  .topk-pct {
    text-align: right;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-feature-settings: 'tnum';
    color: ${THEME.ink2};
  }
  .matched {
    margin-top: 8px;
    font-size: 10.5px;
    color: ${THEME.ink3};
  }
  .matched .mono {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    color: ${THEME.ink2};
  }

  /* Stance gauge */
  .gauge {
    position: relative;
    height: 7px;
    border-radius: 4px;
    background: linear-gradient(90deg, ${THEME.bear} 0%, #E8C4B0 35%, ${THEME.neutral} 50%, #B8E5D2 65%, ${THEME.bull} 100%);
    margin: 8px 0 4px 0;
  }
  .gauge-marker {
    position: absolute; top: 50%;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: #FFFFFF;
    border: 2px solid ${stance.color};
    transform: translate(-50%, -50%);
    box-shadow: 0 0 0 2px #FFFFFF;
  }
  .gauge-labels {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.14em;
    color: ${THEME.ink3};
  }

  /* Impact score block */
  .impact-card {
    margin-top: 8px;
    padding: 14px 16px 12px 16px;
    background: ${THEME.surface};
    border: 1px solid ${THEME.hairline};
    border-radius: 10px;
  }
  .impact-top {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 10px;
  }
  .impact-big {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 700;
    font-size: 40px;
    line-height: 1;
    letter-spacing: -0.02em;
    background: ${THEME.gradNum};
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .impact-of {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px; color: ${THEME.ink3};
    font-feature-settings: 'tnum';
  }

  .breakdown { display: grid; gap: 6px; }
  .br-row {
    display: grid;
    grid-template-columns: 110px 1fr 42px;
    align-items: center;
    gap: 10px;
    font-size: 11px;
  }
  .br-name {
    color: ${THEME.ink};
    font-weight: 500;
    line-height: 1.2;
  }
  .br-name .sub {
    display: block;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.14em;
    color: ${THEME.ink3}; font-weight: 400;
    margin-top: 1px;
  }
  .br-track {
    height: 7px; border-radius: 4px;
    background: ${THEME.surface2};
    overflow: hidden;
  }
  .br-fill { height: 100%; border-radius: 4px; }
  .br-fill.cred { background: linear-gradient(90deg, ${THEME.tealDeep}, ${THEME.teal}); }
  .br-fill.rec  { background: linear-gradient(90deg, ${THEME.teal}, ${THEME.bull}); }
  .br-fill.mat  { background: linear-gradient(90deg, ${THEME.bull}, ${THEME.cyan}); }
  .br-val {
    text-align: right;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-feature-settings: 'tnum';
    color: ${THEME.ink};
    font-weight: 500;
  }
  .derivation {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed ${THEME.hairline};
    text-align: center;
  }
  .derivation-label {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.18em;
    color: ${THEME.ink3};
    margin-bottom: 6px;
  }
  .tex-defs, .tex-eq {
    color: ${THEME.ink};
    font-feature-settings: 'tnum';
  }
  .tex-defs { font-size: 11px;   margin-bottom: 4px; }
  .tex-eq   { font-size: 12.5px; line-height: 1.4; }
  .tex-fallback {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: ${THEME.ink2};
    white-space: pre-wrap;
  }
  /* KaTeX rendered output — match theme ink and tint the bold result green */
  .katex { color: ${THEME.ink}; }
  .katex .mathbf, .katex .boldsymbol { color: ${THEME.bull}; }
  /* Hide fallback ASCII once KaTeX finishes rendering */
  .tex-rendered .tex-fallback { display: none; }

  /* Why-it-matters callout */
  .callout {
    margin-top: 6px;
    padding: 11px 14px;
    background: ${THEME.surface};
    border: 1px solid ${THEME.hairline};
    border-left: 3px solid ${THEME.tealBright};
    border-radius: 6px;
    font-size: 12px; line-height: 1.5;
    color: ${THEME.ink};
  }
  .callout::before {
    content: '“';
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic; font-size: 22px;
    color: ${THEME.tealBright};
    line-height: 0.6;
    margin-right: 3px;
    vertical-align: -6px;
  }

  /* Footer */
  .footer {
    margin: 14px 36px 0 36px;
    padding: 10px 0 14px 0;
    border-top: 1px solid ${THEME.hairline};
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.16em;
    color: ${THEME.ink3};
  }
  .footer a { color: ${THEME.teal}; text-decoration: none; }

  /* Print rules — preserve gradients & background fills, single page */
  @page { size: A4; margin: 0; }
  @media print {
    html, body { background: #FFFFFF; }
    .page { max-width: 100%; }
    .no-print { display: none !important; }
    /* Avoid orphaned section breaks across pages */
    .section, .impact-card, .callout, .footer { page-break-inside: avoid; }
  }
</style>
</head>
<body>
<div class="page">
  <header class="hero">
    <div class="hero-mark"><span>RNIA</span> · <em>retail news impact analyst</em></div>
    <div class="hero-tag">Impact brief</div>
  </header>

  <div class="article">
    <div class="eyebrow">
      <span>${escapeHtml(sourceName)}</span>
      <span class="dot"></span>
      <span>${escapeHtml(dateStr)}</span>
      <span class="dot"></span>
      <span>${escapeHtml(ev.label)}</span>
    </div>
    <h1 class="headline">${escapeHtml(headline)}</h1>
    <div class="status">
      <span class="chip chip-event"><span class="swatch"></span>${escapeHtml(ev.label)}</span>
      <span class="chip chip-stance ${escapeHtml(a.stance || 'neutral')}"><span class="arrow">${stance.arrow}</span>${stance.label}</span>
      <span class="chip chip-impact">Impact <span class="v">${fmt2(a.impact_score)}</span> / 1.00 · ${impactPct}%</span>
    </div>
  </div>

  <section class="section">
    <div class="section-eyebrow">Event classification</div>
    <h2 class="section-headline">This is a <em>${escapeHtml(ev.label)}</em> article.</h2>
    <div class="section-meta">${evConfPct ? `Confidence ${evConfPct} · ` : ''}materiality weight ${fmt2(materiality)}</div>
    ${eventTopK}
    ${evMatched}
  </section>

  <section class="section">
    <div class="section-eyebrow">Reporting stance</div>
    <h2 class="section-headline">Tone reads as <em class="${escapeHtml(a.stance || 'neutral')}">${stance.label.toLowerCase()}</em> ${stance.arrow}</h2>
    <div class="section-meta">${stConfPct ? `Confidence ${stConfPct}` : `Polarity inferred from reporting language`}</div>
    <div class="gauge"><div class="gauge-marker" style="left:${stance.pos}%;"></div></div>
    <div class="gauge-labels"><span>Bearish</span><span>Neutral</span><span>Bullish</span></div>
    ${stanceTopK}
    ${stMatched}
  </section>

  <section class="section">
    <div class="section-eyebrow">Impact score</div>
    <div class="impact-card">
      <div class="impact-top">
        <div class="impact-big">${fmt2(a.impact_score)}</div>
        <div class="impact-of">/ 1.00</div>
      </div>
      <div class="breakdown">
        <div class="br-row">
          <div class="br-name">Credibility<span class="sub">base weight 0.40</span></div>
          <div class="br-track"><div class="br-fill cred" style="width:${Math.round(credibility * 100)}%;"></div></div>
          <div class="br-val">${fmt2(credibility)}</div>
        </div>
        <div class="br-row">
          <div class="br-name">Recency<span class="sub">live decay</span></div>
          <div class="br-track"><div class="br-fill rec" style="width:${Math.round(recency * 100)}%;"></div></div>
          <div class="br-val">${fmt2(recency)}</div>
        </div>
        <div class="br-row">
          <div class="br-name">Materiality<span class="sub">base weight 0.60</span></div>
          <div class="br-track"><div class="br-fill mat" style="width:${Math.round(materiality * 100)}%;"></div></div>
          <div class="br-val">${fmt2(materiality)}</div>
        </div>
      </div>
      <div class="derivation">
        <div class="derivation-label">Score derivation</div>
        <div class="tex-defs" id="tex-defs" data-tex="${escapeHtml(formulaDefsTeX())}">
          <span class="tex-fallback">M = materiality, C = credibility, R = recency</span>
        </div>
        <div class="tex-eq" id="tex-eq" data-tex="${escapeHtml(formulaDerivationTeX(materiality, credibility, recency, a.impact_score))}">
          <span class="tex-fallback">I = (M · 0.6 + C · 0.4) · (0.5 + 0.5 R)
  = ${escapeHtml(formulaText(materiality, credibility, recency, a.impact_score))}</span>
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-eyebrow">Why it matters</div>
    <div class="callout">${escapeHtml(a.explanation || 'No explanation available for this article.')}</div>
  </section>

  <footer class="footer">
    <span>Generated ${escapeHtml(new Date().toLocaleString())}</span>
    <span>${a.url ? `<a href="${escapeHtml(a.url)}">${escapeHtml(a.url.replace(/^https?:\/\//, '').slice(0, 64))}</a>` : 'rnia.local'}</span>
  </footer>
</div>
<script>
  // Render LaTeX with KaTeX, then wait for web fonts so neither the serif
  // headline nor the math glyphs flash fallbacks in the PDF.
  (function () {
    function fire() { try { window.focus(); window.print(); } catch (e) {} }
    function renderTeX() {
      if (!window.katex) return;
      ['tex-defs', 'tex-eq'].forEach(function (id) {
        var el = document.getElementById(id);
        if (!el || !el.dataset.tex) return;
        try {
          window.katex.render(el.dataset.tex, el, {
            throwOnError: false,
            displayMode: id === 'tex-eq',
            output: 'html',
          });
          el.classList.add('tex-rendered');
        } catch (e) { /* keep ASCII fallback visible */ }
      });
    }
    function ready() {
      renderTeX();
      var fontsReady = (document.fonts && document.fonts.ready && typeof document.fonts.ready.then === 'function')
        ? document.fonts.ready
        : new Promise(function (r) { setTimeout(r, 350); });
      fontsReady.then(function () { setTimeout(fire, 150); });
    }
    if (document.readyState === 'complete') ready();
    else window.addEventListener('load', ready);
  })();
</script>
</body>
</html>`;
};

export const exportArticlePDF = (article) => {
  const win = window.open('', '_blank');
  if (!win) {
    alert('Pop-up blocked — allow pop-ups to export the PDF brief.');
    return;
  }
  win.document.open();
  win.document.write(buildHTML(article || {}));
  win.document.close();
};
