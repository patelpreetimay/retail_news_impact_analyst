# Pipeline Algorithm Specification

> Canonical reference for the relevance / classification / impact-scoring algorithm.
>
> **History:** the gold seed dataset (~5,400 articles) was originally labelled by
> Gemini 3.1 Pro / 2.5 Flash. The runtime is now **ML-only** (TF-IDF + Logistic
> Regression for classifiers, Ridge regressors for sub-scores). Gemini is no
> longer called at runtime.
>
> This document describes **what the system does** and **why**. The deterministic
> rules and the four-sub-score impact formula are contracts that any future
> re-implementation MUST preserve.

The pipeline is a hybrid: a strict, deterministic Python rule-engine for
**Relevance gating** and **Impact calculation**, with trained ML models
(originally seeded by an LLM rubric) for the contextual **Classification**.

---

## 1. Relevance Scoring — Deterministic Rule-Engine

**Implementation:** [pipeline/run.py](../pipeline/run.py) — see `_keyword_filter()`

Relevance is evaluated entirely with a fast, zero-cost deterministic algorithm.
**No LLM is used here** — this saves tokens and time, and acts as a hard gate
that filters out noise before the LLM ever sees it.

### Decision rules (applied in order)

| # | Rule | Action |
|---|------|--------|
| 1 | **Junk Stub Check** — `is_junk_stub == 1` (scraper failed / returned garbage HTML) | `relevance = 0`, reason = `junk_stub` |
| 2 | **Body Length Check** — article text shorter than **300 characters** | `relevance = 0`, reason = `body_too_short` |
| 3 | **Keyword Matching** — combine headline + body, run case-insensitive **whole-word boundary** regex (`\b…\b`) against the curated financial vocabulary in [pipeline/keywords.yaml](../pipeline/keywords.yaml) | ≥ 1 match → `relevance = 1` (proceeds to Stage 2). 0 matches → `relevance = 0`, reason = `no_keyword_match` |

The `\b` word-boundary requirement is critical: it prevents short tokens like
`"fed"` from matching inside `"federation"`.

---

## 2. Classification — Trained ML Models

**Runtime implementation:** [pipeline/run.py](../pipeline/run.py) calls
[models/event_classifier.py](../models/event_classifier.py),
[models/stance_detector.py](../models/stance_detector.py), and
[models/sub_score_regressor.py](../models/sub_score_regressor.py).

**Original seed:** the gold dataset was labelled with `gemini-2.5-flash` via the
now-archived V2 pipeline; the rubric (closed-vocabulary JSON output) shaped what
the trained models learned. Today articles that pass the relevance gate are
classified entirely by the ML models — no LLM call at runtime.

### Event Type — exactly one of 10 buckets

```
Earnings
Leadership_Change
Regulatory_Action
Mergers_Acquisitions
Legal_Action
Product_Announcement
Macroeconomic_Geopolitical
Market_Sentiment_Investor_Action
Other
Unclassified
```

### Stance — causal market direction (NOT journalistic tone)

| Value | Definition |
|-------|------------|
| `BULLISH` | Makes the target more valuable / risk-on |
| `BEARISH` | Makes the target less valuable / risk-off |
| `NEUTRAL` | Factual reporting with no clear directional implication |
| `MIXED` | **Strict rule** — only allowed when the text **explicitly** states both a positive and negative outcome for the same entity (e.g. "Profits soared, but the CEO abruptly resigned"). Not for ambiguity. |
| `UNCLASSIFIED` | Article too thin to call |

The strictness on `MIXED` is deliberate: without it the model defaults to
`MIXED` whenever it is unsure, which destroys the signal.

---

## 3. Impact Scoring — Hybrid: ML Sub-scores + Deterministic Math

**Implementation:** [models/sub_score_regressor.py](../models/sub_score_regressor.py)
(see `compute_impact_score()`); driven from
[pipeline/run.py](../pipeline/run.py).

Impact is broken into four continuous sub-scores in `[0.0, 1.0]`. Four trained
Ridge regressors **grade** the qualitative metrics; the **final impact score is
calculated by the Python script** from those grades. This isolates the
arithmetic from any model and keeps the formula auditable.

### The four ML-graded sub-scores

| Sub-score | Range | Anchors |
|-----------|-------|---------|
| **Materiality** | 0.0 – 1.0 | How much could this event move the price? `0` = curiosity; `1.0` = massive cash-flow impact |
| **Market Linkage** | 0.0 – 1.0 | How directly does this connect to a listed equity / index? `0` = generic story; `1.0` = specifically names a ticker |
| **Time Sensitivity** | 0.0 – 1.0 | How quickly will this affect price? `0` = years/months out; `1.0` = same-day / next-day |
| **Credibility** | 0.0 – 1.0 | How reliable is the source? `0` = anonymous rumour; `1.0` = wire service / regulator |

### Deterministic formula (Python-side)

```python
impact_score = (
    0.4 * materiality
  + 0.3 * market_linkage
  + 0.2 * time_sensitivity
  + 0.1 * credibility
)
```

**Weight rationale:** Materiality dominates because nothing matters if the
event doesn't move money. Market Linkage is next — even a huge event is noise
if it doesn't touch a tradable instrument. Time Sensitivity decays the score
for slow-burn stories. Credibility is the final tie-breaker; weighted lowest
because the upstream relevance filter and source whitelist already remove
most low-credibility content.

---

## 4. Escalation / Safety Protocol — Quality Control

**Implementation:** [pipeline/run.py](../pipeline/run.py)
(see `_escalation_reason()`)

Before a row is committed as `done`, it must pass an internal safety check.
A row that fails any check is marked `processing_status = 'needs_escalation'`
and held back from the dashboard. (Originally these were routed to Gemini Pro /
Claude for re-labelling; today they're simply flagged for review.)

### Escalation triggers

| Trigger | Threshold | Why |
|---------|-----------|-----|
| **Low Confidence** | `confidence_event < 0.6` OR `confidence_stance < 0.6` | The model is hedging; don't trust it |
| **Arithmetic Hallucination** | `abs(model_impact_score - python_impact_score) > 0.1` | The model attempted its own arithmetic; the divergence proves it can't be trusted with the formula |
| **Inconsistent Sub-scores** | `materiality >= 0.85 AND market_linkage <= 0.20`, or vice-versa | The model is internally confused — high impact with no market hook (or the inverse) is incoherent |
| **High-Stakes Flag** | `stance == 'BEARISH'` AND `event_type IN ('Regulatory_Action', 'Legal_Action')` | These are reputation-destroying claims; require extra scrutiny before publishing |

---

## Summary — Why this shape

| Concern | Mechanism |
|---------|-----------|
| Cost | Deterministic Stage 1 keeps ~30% of articles out of the classifier entirely; ML inference at runtime is free |
| Hallucination on math | Python computes the final score from sub-scores; the model never owns the final number |
| Hallucination on labels | Strict closed vocabulary baked into training (7 canonical event types, 3 runtime stances) |
| Drift over time | Idempotent: `article_id` PK on `v2_analyses` means re-runs upsert; no double-counting |
| Reputational risk | High-stakes BEARISH-on-regulatory rows auto-escalate before display |

Any future swap (e.g. a different model family) MUST keep the same output
contract — same canonical event types, same stances, same four sub-scores, same
Python-side formula, same escalation triggers — so that downstream consumers
(the dashboard at [backend/app.py](../backend/app.py), the evaluation harness
under [evaluation/](../evaluation/), and the live DB schema of `v2_analyses` in
`data/rnia.db`) remain unchanged.
