# Fare Watch

An airline revenue-management pipeline that answers two questions at scale: **which of thousands of daily-repricing flights need a human's attention, and how should their fares change** — built on real Expedia fare data, with every claim checked against a baseline before it's trusted.

This started as a response to a case-style business problem (an airline pricing team that can only manually review a small share of its flights each day) and grew into a full, honestly-validated pipeline: real data engineering at scale, multiple forecasting approaches compared head-to-head, a non-circular classification model, and a randomized experiment that actually tests whether the pricing policy helps — including reporting a null result and explaining why, rather than hiding it.

## What it actually does

1. **Forecasts** what a flight's fare and seat-availability *should* look like, given its route, cabin, timing, and — where real booking-curve data exists — its own recent price history
2. **Flags** flights whose actual numbers deviate meaningfully from that expectation, and **diagnoses why** (underpriced, overpriced, a genuine demand surge, or weak demand/competitor pressure)
3. **Recommends** a specific fare change, sized off how far out of line the flight is
4. **Validates** that recommendation with a randomized A/B test rather than assuming it works

## Key results

| Component | Method | Result |
|---|---|---|
| Fare forecasting (best) | Random Forest + real per-flight booking-curve lag features | **$14.12 MAE** — 84% better than a flat-mean baseline, 74% better than context-only features |
| Route-level seasonal forecasting | SARIMA(2,1,2)(0,1,1)₇ on a real 217-day daily series | 34% better MAE than seasonal-naive, on a genuine 30-day holdout |
| Mispricing diagnosis | Random Forest classifier, raw features only (no residual leakage) | 73% accuracy vs. 67% majority-class baseline |
| Pricing-policy validation | Randomized A/B test, Welch's t-test + power analysis | No significant revenue lift (p=0.75) — a real, explained finding, not an underpowered study (see below) |
| Independent cross-check | Real aircraft-type → seat-capacity join | Confirms the diagnosis direction: 98.6% estimated load factor for "selling fast" flags vs. 93.4% for "selling slow" |

## The honest part: a null result, and why it matters

The A/B test found that applying the recommended fare change produced **no statistically significant revenue difference** versus doing nothing — and tracing *why* was more valuable than a clean positive result would have been. Under a stated demand-elasticity model, raising a price always looks like it hurts single-sale revenue and cutting it always looks like it helps, so a policy that issues roughly equal numbers of raises and cuts cancels out in aggregate. That's not a flaw in the diagnosis — it reveals that a **static, single-point elasticity model is the wrong lens for a capacity-constrained, perishable-inventory problem**. The real reason to raise a fare on a fast-selling flight isn't to extract more from today's sale — it's to avoid selling the last few seats too cheaply before higher-value demand arrives later in the booking window. That's a dynamic, multi-period bid-price problem (the same class of technique behind real airline revenue-management systems), not something a one-shot A/B test on a point-price change can fully validate. Documented rather than smoothed over.

## Two datasets, compared head-to-head

| | 250K-row sample | Full 84M-row source (31GB) |
|---|---|---|
| Search-date depth | 2 dates only | Streamed and filtered down to **1,085 real flights tracked across ~39 searches each** (up to 60) over a genuine ~7-month window |
| What it enables | Cross-sectional comparison ("how does this flight compare to similar ones") | Genuine booking-curve history ("what was *this exact flight* doing last week") |
| Fare-prediction result | 33% better than baseline (neural network) / 55% (gradient boosting) | **74% better than the cross-sectional approach** using real lag features |

Processing the full source (84M rows) took **~3.5 minutes** streamed in 2M-row chunks — never loaded fully into memory. A second finding worth knowing: `seatsRemaining` turned out to be a coarsely bucketed marketing display, not live inventory — confirmed two ways (8.2% of real same-flight search pairs show it *increasing* over time, and 19% of flights show the identical value across every one of their ~39 real searches spanning months). This is why booking-curve lag features help fare prediction enormously but don't help seats-remaining prediction at all — the ceiling there is data quality, not model choice.

## Pipeline

| Script | Phase | What it does |
|---|---|---|
| `load_data.py` | 1 | Clean and validate the raw Expedia sample |
| `eda.py` | 2 | Exploratory analysis — caught a noisy pooled days-to-departure curve, fixed by adding route/weekday controls |
| `priority_model.py` | 3–4 | Multi-task neural network (entity embeddings, shared trunk) predicting fare + seats-remaining; residual z-scores drive the anomaly flag and diagnosis |
| `fare_policy.py` | 5 | Bid-price-style fare-change sizing; real aircraft-type → seat-capacity join for an independent validation check |
| `explainability.py` / `explain_queue.py` | 6 | SHAP explanations for the model's predictions, including the specific flights shown in the priority queue |
| `diagnosis_classifier.py` | 9 | Random Forest classifier predicting the diagnosis label from raw features only — deliberately withholds the residual that generated the label, so the accuracy number is real, not circular |
| `daily_briefing.py` | 7 | Turns the priority queue into a plain-English briefing via the Anthropic API (Claude Haiku), with a deterministic fallback when no API key is set |
| `ab_test.py` | 8 | Randomized A/B test of the fare policy: Welch's t-test, power analysis, and a guardrail check against disproportionately raising price-sensitive fares |
| `build_route_timeseries.py` | — | Streams the full 31GB source into real daily route-level fare series |
| `sarima_forecast.py` | — | SARIMA seasonal forecasting on a real 217-day series (ATL–LAX) |
| `extract_longitudinal.py` | — | Streams the full source again, this time keeping every raw row for 5 well-covered routes — the real booking-curve extraction |
| `compare_datasets.py` | — | The head-to-head comparison: cross-sectional-only features vs. + real per-flight lag features |

## Data

Real Expedia fare quotes via Kaggle: [`dilwong/flightprices`](https://www.kaggle.com/datasets/dilwong/flightprices) (full source, CC BY 4.0) and the [`justinmitchel/flightprices-min`](https://www.kaggle.com/datasets/justinmitchel/flightprices-min) sample. Raw data is **not committed** (the full source alone is 31GB) — `.gitignore` excludes `data/raw/` entirely; re-download it via the Kaggle CLI (see below).

## Setup

```bash
pip install -r requirements.txt

# Kaggle API credentials required (~/.kaggle/kaggle.json) — see
# https://github.com/Kaggle/kaggle-api#api-credentials
mkdir -p data/raw
kaggle datasets download -d justinmitchel/flightprices-min -p data/raw --unzip

# Optional, for the SARIMA / longitudinal-comparison scripts (31GB):
mkdir -p data/raw/full
kaggle datasets download -d dilwong/flightprices -p data/raw/full --unzip
```

Run the pipeline in order:
```bash
cd src
python load_data.py && python eda.py
python priority_model.py && python fare_policy.py
python explainability.py && python explain_queue.py
python diagnosis_classifier.py
python ab_test.py
python daily_briefing.py           # set ANTHROPIC_API_KEY to use the real LLM path

# With the full 31GB source downloaded:
python build_route_timeseries.py && python sarima_forecast.py
python extract_longitudinal.py && python compare_datasets.py
```

## What's real, assumed, and out of scope

| | Status |
|---|---|
| Fare, seats-remaining, competitor prices, route/cabin/timing | Real (scraped Expedia data) |
| Aircraft seat capacity | Real aircraft type, joined to published typical seat counts (approximate — not the airline's exact configuration) |
| Fare-change magnitude (the ±% sizing rule) | A stated heuristic, explicitly not proven revenue-optimal (see the A/B test finding above) |
| True unconstrained demand | Structurally unobservable — the classic revenue-management censoring problem, not a gap specific to this data |
| Airline's actual internal bid-price/inventory system | Not public anywhere — proprietary to every airline |

## Tech stack

Python · pandas · scikit-learn · TensorFlow/Keras · statsmodels (SARIMA) · SHAP · SciPy (hypothesis testing) · Anthropic API

## Live dashboard

An interactive walkthrough of every result above (with real hover tooltips, not static screenshots) is published as a Claude Artifact. Artifacts are private by default — ask the repo owner for the current link, or [publish your own](https://claude.ai/code/artifacts) from `reports/dashboard.html`.
