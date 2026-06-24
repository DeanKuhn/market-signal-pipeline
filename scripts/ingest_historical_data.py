#   ------------------------------------------
#   --- BRONZE HISTORICAL INGESTION SCRIPT ---
#   ------------------------------------------


# STRUCTURE
#   load_tickers()
#   load_checkpoints()
#   for ticker in remaining_tickers:
#       fetch_ticker_history()  # API call + 13 second sleep
#       update_checkpoint()     # (stay within 5/min limit)
#   consolidate_by_date()
#   write_parquet_partitions()
#   write_metadata_sidecars()


# IMPORTS
import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv


# ENV/VARIABLES
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
SLEEP_BETWEEN_CALLS = 13  # seconds — free tier allows 5/min
TICKERS_FILE = Path("config/sp500_tickers.csv")
CHECKPOINT_FILE = Path("config/.backfill_progress.json")
OUTPUT_DIR = Path("data/bronze/ohlcv")


# PYARROW SCHEMA
OHLCV_SCHEMA = pa.schema([
    pa.field("ticker", pa.string()),
    pa.field("date", pa.date32()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("volume", pa.int64()),
    pa.field("vwap", pa.float64()),
])


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# FUNCTIONS
def load_tickers() -> list[str]:
    df = pd.read_csv(TICKERS_FILE)
    tickers = df["ticker"].dropna().str.strip().tolist()
    log.info(f"Loaded {len(tickers)} tickers from {TICKERS_FILE}")
    return tickers

def load_checkpoint() -> set[str]:
    if not CHECKPOINT_FILE.exists():
        return set()
    with open (CHECKPOINT_FILE) as f:
        # If exists, data becomes a Python dict from json data
        data = json.load(f)
    # Then turns into a set
    completed = set(data.get("completed", []))  # [] is a fallback
    log.info(f"Checkpoint found: {len(completed)} tickers already completed")
    return completed

def save_checkpoint(completed: set[str]) -> None:
    # Returns none if no dat
    #
    # PLACEMARKER
    #
    with open (CHECKPOINT_FILE, "w") as f:
        # Dump Python sorted set (completed) to JSON file
        json.dump({"completed": sorted(completed)}, f)

def fetch_ticker_history(
    ticker: str, start: str, end: str) -> pd.DataFrame | None:
    # Builds DataFrame on success and None on failure
    # so the caller can decide whether to skip or abort

    # Build the URL
    url = f"{BASE_URL}/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }

    # GET request
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.warning(f"Request failed for {ticker}: {e}")
        return None

    # Make GET req


    # Handle errors (bad status, empty results, network failure)
    # Parse results
    # Convert timestamp to date
    # Add ticker column
    # Return DataFrame

