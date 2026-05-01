import { useEffect, useRef, useState } from 'react';
import { analyzeArticle } from '../api';
import { Icon } from './Icon';
import { ScoreBreakdown } from './widgets';
import { EVENT_BY_ID, EVENT_COLORS, composeImpactScore, mapEventType, mapStance } from './data';
import { exportArticlePDF } from './exportPDF';

const SAMPLE_SNIPPETS = {
  earnings: "NVIDIA reported Q4 revenue of $22.1 billion, up 265% year-over-year, smashing analyst estimates of $20.4 billion. Data center revenue alone hit $18.4 billion as enterprise demand for H100 GPUs continued to outpace supply. The company guided next-quarter revenue of $24 billion plus or minus 2%, well above Wall Street consensus. CFO Colette Kress said gross margins should expand to 77% on improving product mix.",
  ma: "Reuters has learned that Apple is in advanced talks to acquire London-based AI startup Perplexity for approximately $4.2 billion, according to people familiar with the matter. The deal would mark Apple's largest acquisition since Beats and signal a strategic pivot toward generative search. Both companies declined to comment, but two sources said terms could be finalized within six weeks pending board approval.",
  regulatory: "The U.S. Securities and Exchange Commission today opened a formal investigation into Tesla's accounting practices around its Full Self-Driving revenue recognition, sources told Bloomberg. The probe focuses on whether the company prematurely booked deferred revenue in 2024. Tesla shares fell 6.4% in after-hours trading. The SEC subpoena requests internal communications and audit working papers dating back to Q1 2023.",
};

const STEPS = [
  { title: 'Classify event', desc: 'Match against the 7-class taxonomy: Earnings, M&A, Regulatory, Leadership, Legal, Product, Market.' },
  { title: 'Detect stance',  desc: 'Score the reporting tone — bullish, bearish, or neutral — independent of impact.' },
  { title: 'Compose impact', desc: 'Blend materiality and credibility, then apply live recency decay into a single 0-1 score.' },
  { title: 'Explain in plain English', desc: 'Generate a transparent, readable rationale linking score to evidence.' },
];

export const Analyze = () => {
  const [text, setText]       = useState('');
  const [phase, setPhase]     = useState('idle');   // idle | working | result | error
  const [stepIdx, setStepIdx] = useState(0);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);
  const stepTimer = useRef(null);

  // tick the visible step animation while the request is in flight
  useEffect(() => {
    if (phase !== 'working') return;
    setStepIdx(0);
    let i = 0;
    const tick = () => {
      i = Math.min(i + 1, STEPS.length - 1);
      setStepIdx(i);
      stepTimer.current = setTimeout(tick, 380);
    };
    stepTimer.current = setTimeout(tick, 320);
    return () => clearTimeout(stepTimer.current);
  }, [phase]);

  const onAnalyze = async () => {
    const wc = text.split(/\s+/).filter(Boolean).length;
    if (wc < 5) return;
    setPhase('working');
    setResult(null);
    setError(null);
    try {
      const raw = await analyzeArticle(text);
      // Map backend → design vocabulary, fill in missing components.
      const eventId = mapEventType(raw.event_type);
      const event = EVENT_BY_ID[eventId];
      const stance = mapStance(raw.stance);
      const credibility = Number.isFinite(+raw.credibility) ? +raw.credibility : 0.95;
      const recency     = Number.isFinite(+raw.recency)     ? +raw.recency     : 0.92;
      const materiality = Number.isFinite(+raw.materiality) ? +raw.materiality : event.materiality;
      const impact = Number.isFinite(+raw.impact_score)
        ? +raw.impact_score
        : composeImpactScore(materiality, credibility, recency);

      // let the step animation finish marking the last step done before swapping
      setStepIdx(STEPS.length);
      setTimeout(() => {
        setResult({
          event_type: eventId,
          event_label: event.label,
          stance,
          impact_score: impact,
          credibility, recency, materiality,
          explanation: raw.explanation || '',
          relevance: raw.relevance,
          event_confidence: Number.isFinite(+raw.event_confidence) ? +raw.event_confidence : null,
          stance_confidence: Number.isFinite(+raw.stance_confidence) ? +raw.stance_confidence : null,
          event_top_k:  Array.isArray(raw.event_top_k)  ? raw.event_top_k  : [],
          stance_top_k: Array.isArray(raw.stance_top_k) ? raw.stance_top_k : [],
          matched_keywords:    raw.matched_keywords    || { event: [], stance: [] },
          matched_article:     !!raw.matched_article,
          matched_article_id:  raw.matched_article_id || null,
        });
        setPhase('result');
      }, 280);
    } catch (e) {
      const detail = e?.response?.data?.detail || e?.message || 'Analysis failed.';
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail));
      setPhase('error');
    }
  };

  const wordCount = text.split(/\s+/).filter(Boolean).length;
  const charCount = text.length;
  const readMin = Math.max(1, Math.round(wordCount / 220));

  // Build a feed-shaped object from the analyzer state so the shared PDF
  // export gets a headline, source, and timestamp to render.
  const buildExportArticle = () => {
    if (!result) return null;
    const firstSentence = (text.match(/[^.!?\n]+[.!?]?/) || [''])[0].trim();
    const headline = firstSentence
      ? (firstSentence.length > 120 ? firstSentence.slice(0, 117) + '…' : firstSentence)
      : 'Custom analysis brief';
    return {
      ...result,
      headline,
      source_name: 'Pasted text · RNIA Analyze',
      timestamp_ms: Date.now(),
      url: '',
    };
  };
  const stanceLeft = result
    ? (result.stance === 'bullish' ? 88 : result.stance === 'bearish' ? 12 : 50)
    : 50;

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1 className="view-title">Analyze <em>any headline</em></h1>
          <div className="view-sub">Paste an article or excerpt — RNIA will classify, score, and explain in seconds.</div>
        </div>
      </div>

      <div className="analyze-grid stagger">
        <div className="card analyze-input-card">
          <div className="snippet-row">
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 10, textTransform: 'uppercase',
              letterSpacing: '0.1em', color: 'var(--ink-3)', marginRight: 4,
            }}>
              Try:
            </span>
            <button className="snippet-chip" onClick={() => setText(SAMPLE_SNIPPETS.earnings)}>Earnings beat</button>
            <button className="snippet-chip" onClick={() => setText(SAMPLE_SNIPPETS.ma)}>M&A rumor</button>
            <button className="snippet-chip" onClick={() => setText(SAMPLE_SNIPPETS.regulatory)}>Regulatory probe</button>
          </div>
          <textarea
            className="analyze-textarea"
            placeholder={'Paste a financial news article, press release, or excerpt…\n\nMinimum 5 words. Works best with full sentences and concrete details.'}
            value={text}
            onChange={e => setText(e.target.value)}
          />
          <div className="analyze-meta">
            <span>{wordCount} words · {charCount} chars · ~{readMin} min read</span>
            <span style={{ color: wordCount < 5 && wordCount > 0 ? 'var(--amber)' : 'var(--ink-3)' }}>
              {wordCount < 5 && wordCount > 0 ? 'Needs ≥5 words' : 'Ready to analyze'}
            </span>
          </div>
          <button
            className="btn btn-primary"
            style={{ width: '100%', marginTop: 14, justifyContent: 'center', padding: '11px 16px', fontSize: 13.5 }}
            onClick={onAnalyze}
            disabled={phase === 'working' || wordCount < 5}
          >
            {phase === 'working' ? 'Analyzing…' : <><Icon name="sparkles" size={14} />Analyze</>}
          </button>
        </div>

        <div className="card analyze-output-card">
          {phase === 'idle' && (
            <div className="analyze-empty" style={{ position: 'relative', zIndex: 1 }}>
              <div className="analyze-empty-title">How RNIA reads the news</div>
              <div className="analyze-empty-sub">Four transparent steps. No black boxes.</div>
              <div className="steps">
                {STEPS.map((s, i) => (
                  <div className="step-card" key={i}>
                    <div className="step-num">{i + 1}</div>
                    <div>
                      <div className="step-title">{s.title}</div>
                      <div className="step-desc">{s.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {phase === 'working' && (
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div className="analyze-empty-title">Analyzing…</div>
              <div className="analyze-empty-sub">Working through the pipeline.</div>
              <div style={{ marginTop: 18 }}>
                {STEPS.map((s, i) => {
                  const done = i < stepIdx;
                  const active = i === stepIdx;
                  return (
                    <div key={i} className={`work-step ${done ? 'done' : ''} ${active ? 'active' : ''}`}>
                      <div className="work-check">
                        {done
                          ? <Icon name="check" size={12} />
                          : active ? <div className="work-spin" />
                          : <span className="mono" style={{ fontSize: 10 }}>{i + 1}</span>}
                      </div>
                      <div className="work-label">{s.title}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {phase === 'error' && (
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div className="analyze-empty-title" style={{ color: 'var(--bear)' }}>Analysis failed</div>
              <div className="callout" style={{ borderLeftColor: 'var(--bear)', marginTop: 14 }}>
                {error}
              </div>
              <button className="btn" style={{ marginTop: 14 }} onClick={() => setPhase('idle')}>
                <Icon name="refresh" size={13} />Try again
              </button>
            </div>
          )}

          {phase === 'result' && result && (
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div className="result-section" style={{ paddingTop: 0 }}>
                <div className="result-eyebrow">Event classification</div>
                <div className="result-headline">
                  This is an <em>{result.event_label}</em> article.
                </div>
                <div style={{ marginTop: 10, fontSize: 12.5, color: 'var(--ink-2)' }}>
                  Confidence {result.event_confidence != null ? `${(result.event_confidence * 100).toFixed(0)}%` : '—'} · materiality weight {result.materiality.toFixed(2)}
                </div>

                {result.event_top_k.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    {result.event_top_k.map((t, i) => {
                      const ev = EVENT_BY_ID[t.id] || { label: t.id };
                      const pct = (t.prob * 100).toFixed(0);
                      const isTop = i === 0;
                      return (
                        <div key={t.id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, fontSize: 11.5 }}>
                          <div style={{ width: 92, color: isTop ? 'var(--ink-1)' : 'var(--ink-3)', fontWeight: isTop ? 600 : 400 }}>
                            {ev.label}
                          </div>
                          <div style={{ flex: 1, height: 6, background: 'var(--bg-3)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{
                              width: `${pct}%`, height: '100%',
                              background: EVENT_COLORS[t.id] || 'var(--ink-3)',
                              opacity: isTop ? 1 : 0.4,
                            }} />
                          </div>
                          <div style={{ width: 36, textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--ink-2)' }}>
                            {pct}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {result.matched_keywords.event.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{
                      fontSize: 10.5,
                      fontFamily: 'var(--font-mono)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.08em',
                      color: 'var(--ink-3)',
                      marginBottom: 8,
                    }}>
                      Why this classification? · {result.matched_keywords.event.length} term{result.matched_keywords.event.length === 1 ? '' : 's'} matched
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {result.matched_keywords.event.map((term, i) => {
                        const color = EVENT_COLORS[result.event_type] || 'var(--ink-2)';
                        return (
                          <span key={`ev-${i}`} style={{
                            display: 'inline-block',
                            padding: '4px 10px',
                            fontSize: 11,
                            fontFamily: 'var(--font-mono)',
                            background: `color-mix(in srgb, ${color} 14%, transparent)`,
                            color: color,
                            borderRadius: 999,
                            border: `1px solid color-mix(in srgb, ${color} 38%, transparent)`,
                            lineHeight: 1.4,
                          }}>
                            {term}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div className="result-section">
                <div className="result-eyebrow">Reporting stance</div>
                <div className="result-headline" style={{ marginBottom: 4 }}>
                  Tone reads as{' '}
                  <em style={{
                    color: result.stance === 'bullish' ? 'var(--bull)'
                         : result.stance === 'bearish' ? 'var(--bear)'
                         : 'var(--neutral)',
                  }}>
                    {result.stance}
                  </em>
                  {result.stance === 'bullish' ? ' ↑' : result.stance === 'bearish' ? ' ↓' : ' ─'}
                </div>
                <div style={{ marginTop: 4, fontSize: 12.5, color: 'var(--ink-2)' }}>
                  Confidence {result.stance_confidence != null ? `${(result.stance_confidence * 100).toFixed(0)}%` : '—'}
                </div>
                <div className="gauge">
                  <div
                    className="gauge-marker"
                    style={{
                      left: `${stanceLeft}%`,
                      borderColor:
                        result.stance === 'bullish' ? 'var(--bull)'
                        : result.stance === 'bearish' ? 'var(--bear)'
                        : 'var(--neutral)',
                    }}
                  />
                </div>
                <div className="gauge-labels">
                  <span>Bearish</span><span>Neutral</span><span>Bullish</span>
                </div>

                {result.stance_top_k.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    {result.stance_top_k.map((t, i) => {
                      const pct = (t.prob * 100).toFixed(0);
                      const color = t.id === 'bullish' ? 'var(--bull)'
                                  : t.id === 'bearish' ? 'var(--bear)'
                                                       : 'var(--neutral)';
                      const isTop = i === 0;
                      return (
                        <div key={t.id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, fontSize: 11.5 }}>
                          <div style={{ width: 92, color: isTop ? 'var(--ink-1)' : 'var(--ink-3)', fontWeight: isTop ? 600 : 400, textTransform: 'capitalize' }}>
                            {t.id}
                          </div>
                          <div style={{ flex: 1, height: 6, background: 'var(--bg-3)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: color, opacity: isTop ? 1 : 0.4 }} />
                          </div>
                          <div style={{ width: 36, textAlign: 'right', fontFamily: 'var(--font-mono)', color: 'var(--ink-2)' }}>
                            {pct}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {result.matched_keywords.stance.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{
                      fontSize: 10.5,
                      fontFamily: 'var(--font-mono)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.08em',
                      color: 'var(--ink-3)',
                      marginBottom: 8,
                    }}>
                      Why this stance? · {result.matched_keywords.stance.length} term{result.matched_keywords.stance.length === 1 ? '' : 's'} matched
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {result.matched_keywords.stance.map((term, i) => {
                        const color = result.stance === 'bullish' ? 'var(--bull)'
                                    : result.stance === 'bearish' ? 'var(--bear)'
                                    : 'var(--neutral)';
                        return (
                          <span key={`st-${i}`} style={{
                            display: 'inline-block',
                            padding: '4px 10px',
                            fontSize: 11,
                            fontFamily: 'var(--font-mono)',
                            background: `color-mix(in srgb, ${color} 14%, transparent)`,
                            color: color,
                            borderRadius: 999,
                            border: `1px solid color-mix(in srgb, ${color} 38%, transparent)`,
                            lineHeight: 1.4,
                          }}>
                            {term}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div className="result-section">
                <div className="result-eyebrow">Impact score</div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 4, margin: '6px 0 14px' }}>
                  <span className="result-impact-big">{result.impact_score.toFixed(2)}</span>
                  <span className="result-impact-of">/ 1.00</span>
                </div>
                <ScoreBreakdown a={result} />
              </div>

              <div className="result-section">
                <div className="result-eyebrow">Why it matters</div>
                <div className="callout">{result.explanation || 'No explanation provided.'}</div>
              </div>

              <div style={{ display: 'flex', gap: 8, marginTop: 18, flexWrap: 'wrap' }}>
                <button className="btn"><Icon name="copy" size={13} />Copy as Markdown</button>
                <button className="btn" onClick={() => { const ex = buildExportArticle(); if (ex) exportArticlePDF(ex); }}><Icon name="download" size={13} />Export PDF brief</button>
                <button className="btn"><Icon name="layers" size={13} />Compare another</button>
                <button className="btn btn-primary" style={{ marginLeft: 'auto' }} onClick={() => setPhase('idle')}>
                  <Icon name="plus" size={13} />Analyze another
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
