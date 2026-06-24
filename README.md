# Market Signal Pipeline

A large-scale stock market signal detection pipeline built on a Microsoft Azure + Databricks stack. Ingests daily OHLCV data for the S&P 500 from Polygon.io, transforms it through a Bronze → Silver → Gold medallion architecture on Azure Data Lake Storage Gen2 with Delta Lake, engineers 18 rolling technical features in PySpark, and trains a LightGBM binary classifier to detect pre-breakout patterns across the full ticker universe.

The core question: **can you systematically identify, across 500 tickers simultaneously, the structural price and volume signatures that precede a stock breaking above its recent resistance ceiling?**

> **Note:** This is a signal detection pipeline, not a prediction system. The model identifies historical patterns that preceded breakouts and flags when current conditions match those patterns. It does not predict prices or provide investment advice.

---

## The Problem

Most retail traders enter positions *after* a stock has already made its move. By the time a ticker is trending on social media, appearing in screeners, and getting picked up by financial news, the structural setup has already resolved. The breakout happened yesterday.

This pipeline attempts to identify the *precursor* — the period of price compression, volume accumulation, and relative strength that tends to appear before a significant move, before it becomes obvious.

### The Signal

A **breakout** is defined as: a stock's closing price exceeding its rolling 20-day high within the next 5 trading days, starting from a position below that resistance ceiling.

The model labels the *ignition point* (the setup day when conditions align) rather than the breakout days themselves. A stock riding momentum above its 20-day high is not a signal — the setup that preceded it is.

---

## Architecture

```
[Polygon.io API — free tier]
  Daily OHLCV for 500 S&P 500 tickers + SPY
        │
        ▼  Python ingestion script (EC2, post-market daily)
[Azure Data Lake Storage Gen2 — Bronze]
  Raw parquet, Hive-style date partitions
  _metadata.json per partition (pull timestamp, ticker counts)
        │
        ▼  PySpark (Databricks Free Edition)
[ADLS Gen2 — Silver]
  Delta tables: cleaned OHLCV, forward-filled gaps, quality log
  Schema-enforced, deduplicated, adjusted-price validated
        │
        ▼  PySpark (Databricks Free Edition)
[ADLS Gen2 — Gold]
  Delta tables: 18 engineered features + breakout label
  Partitioned by date, lookahead-corrected window functions
        │
        ▼  LightGBM inference (EC2)
  signal_score per ticker per day
        │
        ▼  Apache Airflow DAG (Docker, EC2)
  Orchestrates all stages with gatekeeper checks
        │
        ▼  data/signal_summary.json → GitHub → deanslist.dev
  Daily signal feed, top-ranked tickers, market context
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| Data source | Polygon.io (free tier, daily OHLCV) |
| Cloud storage | Azure Data Lake Storage Gen2 |
| Table format | Delta Lake |
| Compute | Databricks Free Edition (PySpark / Apache Spark) |
| Orchestration | Apache Airflow (Docker Compose, AWS EC2) |
| ML | LightGBM |
| Cloud hosting | AWS EC2 (Airflow + ingestion script) |
| Portfolio output | Static Astro site (deanslist.dev), fed by signal_summary.json |

---

## Feature Engineering

All features are computed in PySpark using window functions partitioned by ticker and ordered by date. All rolling aggregations use `rowsBetween(-N, -1)` — explicitly excluding the current row — to prevent lookahead bias.

| Feature | Description |
|---------|-------------|
| `return_5d` | 5-day price return |
| `return_20d` | 20-day price return |
| `vol_20d` | 20-day realized volatility (std of daily returns) |
| `vol_60d` | 60-day realized volatility |
| `vol_ratio` | vol_20d / vol_60d — values <1.0 indicate compression |
| `prior_high_20d` | Max close over prior 20 trading days (excludes today) |
| `dist_from_20d_high` | Distance below resistance as a fraction of resistance |
| `prior_high_52w` | Max close over prior 252 trading days |
| `dist_from_52w_high` | Distance below 52-week high |
| `vol_ratio_5d_20d` | 5-day avg volume / 20-day avg volume |
| `volume_trend_10d` | Linear slope of volume over prior 10 days |
| `rel_strength_5d` | Stock 5-day return minus SPY 5-day return |
| `rel_strength_20d` | Stock 20-day return minus SPY 20-day return |
| `day_of_week` | 0=Monday, 4=Friday |
| `month` | 1–12 |
| `is_earnings_season` | 1 during weeks 2–5 of Jan/Apr/Jul/Oct |

The 20-day high is defined over the *prior* 20 days, excluding today. If today's close were included, `dist_from_20d_high` would evaluate to exactly 0.0 on any day the stock closes at a new high — providing no signal and introducing information about today's outcome into a feature meant to represent conditions before today.

---

## Label Definition

```
label = 1  if:
    close_today < prior_high_20d          (stock is below resistance)
    AND
    max(close_{t+1} ... close_{t+5}) > prior_high_20d   (crosses above within 5 days)

label = 0  otherwise
label = null  for the most recent 5 rows per ticker (no future data available — inference only)
```

This labels the **ignition point**, not the continuation. Days where the stock is already above its 20-day high and riding momentum are labeled 0 — the model is not being trained to chase breakouts in progress.

---

## Model

**Algorithm:** LightGBM binary classifier

**Objective:** `binary` (breakout vs. no breakout)

**Class imbalance:** True breakouts represent roughly 15–25% of rows. `is_unbalance: True` is set so the model doesn't default to predicting 0 for every row.

### Validation Strategy

Financial time series have temporal autocorrelation — adjacent dates share overlapping rolling windows. Standard random k-fold cross-validation allows training data to contain dates adjacent to validation dates, which inflates metrics. Walk-forward validation enforces a strict temporal boundary.

A 20-day purge gap is applied between the end of each training window and the start of its validation window, eliminating any overlap caused by 20-day rolling features.

```
Fold 1: Train 2021      → Validate 2022
Fold 2: Train 2021-22   → Validate 2023
Fold 3: Train 2021-23   → Validate 2024
Test:   Train 2021-24   → Test 2025  ← touched once, final evaluation only
```

Hyperparameters are tuned against mean validation AUC across all three folds. The test set result is the single number reported below and in `signal_summary.json` — it was not used during tuning.

### Results

| Metric | Value |
|--------|-------|
| Val AUC Fold 1 (2022) | — |
| Val AUC Fold 2 (2023) | — |
| Val AUC Fold 3 (2024) | — |
| Mean Val AUC | — |
| **Test AUC (2025)** | — |

*Results will be populated after initial training run.*

Variation in AUC across folds is expected and informative — it reflects the model's sensitivity to different market regimes (2022 rate-hike bear market vs. 2023–2024 bull market). A model that performs consistently across all three folds is more trustworthy than one that excels in a single regime.

---

## Silver Layer: Data Quality

Three quality mechanisms in the Silver transform:

**Backbone join:** The transform generates a full `(ticker, date)` backbone from `config/sp500_tickers.csv` before joining Bronze data. Tickers missing entirely from a given day's API response are exposed as null rows and handled explicitly — not silently dropped.

**Forward-fill:** Missing close prices are filled from the last known value in the Silver history. Volume is set to 0 for filled rows, distinguishing them from actual trading days.

**Completeness check:** If fewer than 450 tickers return from Polygon on a given day, a warning row is written to `_quality_log` in the Silver layer. The pipeline continues but the anomaly is recorded.

---

## Airflow DAG

```
ingest_polygon
      │
validate_bronze       ← fails DAG if <450 tickers in _metadata.json
      │
transform_silver      ← DatabricksSubmitRunOperator → Databricks Free Edition
      │
transform_gold        ← DatabricksSubmitRunOperator → Databricks Free Edition
      │
run_inference         ← LightGBM on EC2, scores latest rows per ticker
      │
publish_signals       ← writes signal_summary.json, git commit + push
```

Airflow runs in Docker Compose on EC2. It submits PySpark notebooks to Databricks via the REST API using `DatabricksSubmitRunOperator` — Airflow is the orchestrator, Databricks is the compute target. They are separate systems.

Scheduled daily at 6:00 PM ET (post-market close).

---

## Output

`data/signal_summary.json` is committed to this repo on every pipeline run. The portfolio site at [deanslist.dev](https://deanslist.dev) fetches it at build time and rebuilds nightly via GitHub Actions.

```json
{
  "meta": {
    "generated_at": "2026-06-24T22:00:00Z",
    "trading_date": "2026-06-24",
    "total_tickers_scanned": 498,
    "breakout_signals_detected": 14
  },
  "model_meta": {
    "test_auc": 0.00,
    "training_window": "2021-01-01 to 2025-12-31",
    "features_used": 18
  },
  "top_signals": [
    {
      "ticker": "AAPL",
      "signal_score": 0.892,
      "metrics": {
        "close": 185.40,
        "dist_from_20d_high": -0.005,
        "volume_ratio_5d": 2.4,
        "rel_strength_spy_5d": 0.035
      }
    }
  ],
  "market_context": {
    "spy_trend_20d": 0.015,
    "average_volume_z_score": 1.1
  }
}
```

---

## Project Structure

```
market-signal-pipeline/
├── config/
│   └── sp500_tickers.csv          # Master ticker list (fixed at project start)
├── data/
│   └── signal_summary.json        # Daily output (read by deanslist.dev)
├── scripts/
│   ├── ingest_polygon.py          # Bronze: Polygon API → ADLS Gen2
│   ├── submit_databricks.py       # Submits PySpark notebook to Databricks
│   ├── run_inference.py           # LightGBM inference on latest Gold rows
│   └── publish_signals.py         # Writes JSON, git commit + push
├── notebooks/
│   ├── silver_transform.py        # PySpark: Bronze → Silver
│   └── gold_features.py           # PySpark: Silver → Gold (features + label)
├── ml/
│   ├── train.py                   # LightGBM training (run manually)
│   ├── features.py                # FEATURE_COLS, feature construction helpers
│   └── models/
│       └── lgbm_model.pkl         # Serialized trained model
├── dags/
│   └── market_signal_pipeline.py  # Airflow DAG definition
├── docker-compose.yml             # Airflow local setup
└── .env.example                   # Required environment variables
```

---

## Setup

### Prerequisites
- Python 3.12+
- Docker + Docker Compose (for Airflow)
- Azure account (free tier sufficient)
- Databricks Free Edition account
- Polygon.io account (free tier)
- AWS EC2 instance (or any always-on Linux host)

### Environment Variables

```bash
cp .env.example .env
# Fill in:
# POLYGON_API_KEY
# ADLS_ACCOUNT_NAME
# ADLS_ACCOUNT_KEY
# DATABRICKS_HOST
# DATABRICKS_TOKEN
# AZURE_TENANT_ID
# AZURE_CLIENT_ID
# AZURE_CLIENT_SECRET
```

### First-Time Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run historical backfill (pulls 5 years of daily OHLCV — run once)
python scripts/ingest_polygon.py --backfill --start 2021-01-01 --end 2025-12-31

# 3. Run Silver and Gold transforms on full history
python scripts/submit_databricks.py --notebook /notebooks/silver_transform --full-history
python scripts/submit_databricks.py --notebook /notebooks/gold_features --full-history

# 4. Train the initial model
python ml/train.py

# 5. Start Airflow
docker-compose up -d
```

---

## Honest Limitations

**Ticker universe is static.** The S&P 500 composition changes over time — stocks are added and removed. This pipeline fixes the universe at project start, introducing mild survivorship bias. Dynamic composition handling is out of scope.

**5 years of history.** Polygon's free tier provides approximately 5 years of daily data. The training window covers 2021–2024, which includes the 2022 rate-hike bear market, the 2023–2024 bull run, and the post-COVID normalization period — three meaningfully different regimes — but excludes longer historical cycles.

**Daily bars, not intraday.** Minute-level data is cost-prohibitive on the free tier. Daily OHLCV is sufficient for a 5-day forward label and all engineered features, but intraday patterns (opening range, volume profile shape) are not captured.

**No transaction costs or slippage.** The signal score is a raw model output, not a backtested return. Converting signals to a trading strategy would require modeling execution costs, which this pipeline does not attempt.
