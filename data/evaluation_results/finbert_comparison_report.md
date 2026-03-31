# FinBERT vs RNIA LR Stance Detector — Comparison Report

## Setup

- **Task:** 3-class stance detection (bullish / bearish / neutral)
- **Test set:** 487 held-out samples (same 20% split as `evaluate_models.py`, `random_state=42`)
- **Baseline:** TF-IDF + Logistic Regression (RNIA V3 stance detector)
- **Comparator:** `ProsusAI/finbert` zero-shot (no fine-tuning), label-mapped:
  `positive→bullish`, `negative→bearish`, `neutral→neutral`

## Results

| Metric                | TF-IDF + LR (RNIA) | FinBERT (zero-shot) | Δ (FinBERT − LR) |
|-----------------------|-------------------:|--------------------:|-----------------:|
| Accuracy              | 0.8439        | 0.4887        | -0.3552      |
| Precision (weighted)  | 0.8435        | 0.5719        | -0.2716      |
| Recall (weighted)     | 0.8439        | 0.4887        | -0.3552      |
| **F1 (weighted)**     | **0.8435**    | **0.5068**    | **-0.3367**  |
| F1 (macro)            | 0.8295        | 0.4811        | -0.3483      |
| Latency (ms / sample) | 1.26        | 39.48        | 31.3× slower    |
| Interpretable?        | Yes (LR coefficients) | No (transformer black box) | — |
| Memory footprint      | ~30 MB              | ~440 MB                  | ~14× larger      |
| Runtime dependency    | scikit-learn        | torch + transformers     | Heavier stack    |

## Per-class metrics

### TF-IDF + LR (RNIA)
```
              precision    recall  f1-score   support

     bearish       0.87      0.85      0.86       154
     bullish       0.85      0.88      0.87       235
     neutral       0.78      0.74      0.76        98

    accuracy                           0.84       487
   macro avg       0.83      0.83      0.83       487
weighted avg       0.84      0.84      0.84       487

```

### FinBERT (zero-shot)
```
              precision    recall  f1-score   support

     bearish       0.61      0.64      0.63       154
     bullish       0.69      0.41      0.51       235
     neutral       0.23      0.44      0.30        98

    accuracy                           0.49       487
   macro avg       0.51      0.50      0.48       487
weighted avg       0.57      0.49      0.51       487

```

## Verdict

**The RNIA LR ensemble outperforms generic FinBERT by 33.7% F1.** Domain-tuned simpler models can beat large pretrained models when the task distribution differs from the pretraining corpus. This validates the design decision to train task-specific models on curated financial-news data.

## Why we did NOT integrate FinBERT into the V3 pipeline

V3's design principle is **ML-only at runtime — fast, lightweight, offline,
interpretable**. Integrating FinBERT would:

1. Conflict with the unified TF-IDF feature space shared by event / stance / sub-score models
2. Increase per-article latency by ~31× (breaks dashboard real-time UX)
3. Require GPU at inference for acceptable speed (proposal hardware: 8 GB RAM, no GPU)
4. Eliminate per-prediction interpretability (LR coefficients vs transformer attention)

This comparison study confirms our architectural choice is principled, not
accidental.

## Reproducibility

```bash
python evaluation/finbert_comparison.py
```

- Test split is deterministic (`random_state=42`)
- FinBERT weights pinned to `ProsusAI/finbert`
- Outputs written to `data/evaluation_results/`
