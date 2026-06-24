# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the full daily pipeline (via Airflow — do not invoke stages manually in production)
# Individual stage scripts for development/debugging:

# Bronze ingestion
python scripts/ingest_polygon.py --date 2026-06-24

# Silver transform (submit to Databricks via REST API, or run locally for dev)
python scripts/submit_databricks.py --notebook /notebooks/silver_transform --date 2026-06-24

# Gold feature engineering
python scripts/submit_databricks.py --notebook /notebooks/gold_features --date 2026-06-24

# Inference (runs locally on EC2)
python scripts/run_inference.py --date 2026-06-24

# Publish signals
python scripts/publish_signals.py --date 2026-06-24

# Airflow (Docker on EC2)
docker-compose up -d          # start Airflow
docker-compose down           # stop Airflow
airflow dags trigger market_signal_pipeline  # manual trigger

# Model training (run manually, not part of nightly DAG)
python ml/train.py
```

**Important:** Never invoke Silver or Gold transforms directly against the production Delta tables
without going through Airflow. The DAG enforces stage dependencies and gatekeeper checks that
prevent bad data from propagating downstream.

## Architecture

Bronze → Silver → Gold medallion pipeline on Azure Data Lake Storage Gen2 (Delta Lake),
with PySpark transforms running on Databricks Free Edition, orchestrated by Airflow on EC2.

### Data Flow

```
[Polygon.io API — free tier]
  Daily OHLCV bars for S&P 500 tickers + SPY
        │
        ▼ scripts/ingest_polygon.py (Python, EC2)
[ADLS Gen2 — Bronze]
  /bronze/ohlcv/date=YYYY-MM-DD/raw_ohlcv.parquet
  /bronze/ohlcv/date=YYYY-MM-DD/_metadata.json
  /bronze/spy/date=YYYY-MM-DD/raw_spy.parquet
        │
        ▼ notebooks/silver_transform.py (PySpark, Databricks)
[ADLS Gen2 — Silver]
  /silver/ohlcv/          ← Delta table, partitioned by date
  /silver/spy/            ← Delta table
  /silver/_quality_log/   ← Delta table, one row per pipeline run
        │
        ▼ notebooks/gold_features.py (PySpark, Databricks)
[ADLS Gen2 — Gold]
  /gold/features/         ← Delta table, 18 features + label, partitioned by date
  /gold/signals/          ← Delta table, inference output only
        │
        ▼ scripts/run_inference.py (Python, EC2)
  LightGBM model loaded from ml/models/lgbm_model.pkl
  Scores written to /gold/signals/
        │
        ▼ scripts/publish_signals.py (Python, EC2)
  data/signal_summary.json committed and pushed to GitHub
  deanslist.dev rebuilds nightly via GitHub Actions
```

### Bronze Layer

`scripts/ingest_polygon.py` pulls daily OHLCV bars from Polygon.io for all tickers in
`config/sp500_tickers.csv` plus SPY. Writes raw parquet to ADLS Gen2 under Hive-style
date partitions. Also writes `_metadata.json` containing pull timestamp, API version,
tickers requested, and tickers returned.

No transformation occurs in Bronze. Nulls, malformed values, and data quality anomalies
land exactly as received from the API.

### Silver Layer

`notebooks/silver_transform.py` runs as a PySpark job on Databricks. Reads today's
Bronze partition plus the master ticker list from `config/sp500_tickers.csv`.

Key operations:
- **Backbone join**: generates a full `(ticker, date)` backbone from the master ticker
  list, left-joins Bronze data against it to expose missing tickers as null rows
- **Forward-fill**: fills missing close prices from the last known value in Silver history;
  sets volume to 0 for filled rows
- **Schema enforcement**: explicit casts to correct types; no schema inference
- **Deduplication**: on `(ticker, date)`, keeps row with highest volume
- **Adjusted price validation**: flags rows where adjusted close differs from unadjusted
  close by >20% on a non-split date
- **Completeness check**: if fewer than 450 tickers returned from Polygon, writes a
  warning to `_quality_log`; does not fail the pipeline

Silver tables are Delta format for ACID guarantees on daily appends.

### Gold Layer

`notebooks/gold_features.py` runs as a PySpark job on Databricks. Reads the full Silver
history to compute rolling window features across all tickers simultaneously.

**Window function convention:** All rolling aggregations use `rowsBetween(-N, -1)` to
exclude the current row. This prevents lookahead bias — today's price action must not
appear in features that are supposed to represent conditions *before* today.

**Label definition:** A row is labeled 1 if:
1. Today's close is below `prior_high_20d` (stock is below resistance), AND
2. The maximum close over the next 5 trading days exceeds `prior_high_20d`

This labels the ignition point (the setup day), not the breakout days themselves.
The last 5 rows per ticker have null labels — inference only, not training.

### Gold Table Schema (`/gold/features/`)

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Ticker symbol |
| `date` | date | Trading date |
| `close` | double | Adjusted closing price |
| `volume` | long | Daily volume |
| `return_5d` | double | 5-day price return |
| `return_20d` | double | 20-day price return |
| `vol_20d` | double | 20-day realized volatility (std of daily returns) |
| `vol_60d` | double | 60-day realized volatility |
| `vol_ratio` | double | vol_20d / vol_60d (compression signal, <1.0 = compressed) |
| `prior_high_20d` | double | Max close over prior 20 trading days (excludes today) |
| `dist_from_20d_high` | double | (close - prior_high_20d) / prior_high_20d |
| `prior_high_52w` | double | Max close over prior 252 trading days |
| `dist_from_52w_high` | double | (close - prior_high_52w) / prior_high_52w |
| `vol_ratio_5d_20d` | double | 5-day avg volume / 20-day avg volume |
| `volume_trend_10d` | double | Linear slope of volume over prior 10 days |
| `rel_strength_5d` | double | Stock 5-day return minus SPY 5-day return |
| `rel_strength_20d` | double | Stock 20-day return minus SPY 20-day return |
| `day_of_week` | int | 0=Monday, 4=Friday |
| `month` | int | 1–12 |
| `is_earnings_season` | int | 1 if weeks 2–5 of Jan/Apr/Jul/Oct, else 0 |
| `label` | int | 1 = breakout within 5 days, 0 = no breakout, null = inference only |

### ML Model

Trained in `ml/train.py`, serialized to `ml/models/lgbm_model.pkl`.

**Algorithm:** LightGBM binary classifier (`objective: binary`)

**Imbalance handling:** `is_unbalance: True` (breakouts are ~15–25% of rows)

**Validation strategy:** Multi-fold walk-forward with 20-day purge gap between
training end and validation start:

```
Fold 1: Train 2021      → Validate 2022
Fold 2: Train 2021-22   → Validate 2023
Fold 3: Train 2021-23   → Validate 2024
Test:   Train 2021-24   → Test 2025 (touched once, final evaluation only)
```

Hyperparameters tuned against mean validation AUC across all three folds.
Test AUC is the number reported in the README and signal_summary.json.

Retraining is manual — not part of the nightly DAG. Run `ml/train.py` when
sufficient new data has accumulated (roughly quarterly).

### Airflow DAG (`dags/market_signal_pipeline.py`)

Six tasks, linear with gatekeeper at Bronze:

```
ingest_polygon → validate_bronze → transform_silver →
transform_gold_features → run_inference → publish_signals
```

`validate_bronze`: reads `_metadata.json`, fails the DAG if fewer than 450 tickers
returned from Polygon. Prevents bad ingestion days from propagating to Silver.

`transform_silver` / `transform_gold_features`: use `DatabricksSubmitRunOperator`
to submit notebooks to Databricks Free Edition via REST API. Airflow runs on EC2;
Databricks is the compute target. These are separate systems.

`run_inference`: runs locally on EC2, loads serialized LightGBM model, scores the
latest 5 rows per ticker from the Gold features table.

`publish_signals`: writes `data/signal_summary.json`, runs `git commit && git push`.

### Deployment

Airflow runs in Docker on the same EC2 instance as KitchenSync. The nightly DAG
is scheduled to trigger at 6:00 PM ET (post-market close).

GitHub Actions triggers the deanslist.dev Astro site to rebuild after each push
of `signal_summary.json`.

## Ticker Universe

Fixed at S&P 500 composition as of project start (`config/sp500_tickers.csv`).
Composition changes (additions/removals) are not tracked. This introduces minor
survivorship bias that is acknowledged in the README. Handling dynamic composition
is out of scope for this portfolio project.

## Key Design Decisions

**Daily bars over minute bars:** Polygon free tier restricts minute-level historical
data. Daily OHLCV is sufficient for a 5-day forward label and all engineered features.
The computation pattern (rolling windows across 500 tickers) still justifies PySpark
regardless of bar granularity.

**ADLS Gen2 + Delta Lake over raw parquet:** ACID transactions protect against
half-written appends if a job fails mid-write. Time travel allows querying historical
signal states for backtesting. Schema enforcement prevents malformed Polygon responses
from silently corrupting the feature table.

**Airflow on EC2, not managed Airflow:** Cost. A Docker Compose Airflow instance on
an already-running EC2 machine adds no incremental infrastructure cost.

**Databricks Free Edition for PySpark:** Serverless-only, quota-limited, but sufficient
for a nightly batch job that runs one PySpark notebook per stage. Compute is free.

**No retraining in the nightly DAG:** Retraining on potentially corrupted or incomplete
data would degrade the model silently. Training is a deliberate manual step.

**Walk-forward validation, not random k-fold:** Financial time series have temporal
autocorrelation. Random splits allow training data to contain dates adjacent to
validation dates, inflating metrics. Walk-forward enforces a strict temporal boundary.

## Environment Variables

```
POLYGON_API_KEY=
ADLS_ACCOUNT_NAME=
ADLS_ACCOUNT_KEY=
DATABRICKS_HOST=
DATABRICKS_TOKEN=
AZURE_TENANT_ID=
AZURE_CLIENT_ID=
AZURE_CLIENT_SECRET=
```

All secrets stored in `.env` (not committed) and as Airflow Variables/Connections
in the Docker Compose Airflow instance.
