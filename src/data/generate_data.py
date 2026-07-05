"""
generate_data.py — Part 1: Synthetic Indian Retail Transaction Data Generator
==============================================================================
Generates 1M+ realistic Indian kirana/retail transactions covering 50 SKUs
over 2 years (2023-2024), with:
  - Festival seasonality (Diwali, Holi, Navratri, Eid, etc.)
  - Monthly price variance (5-15%) with inflation trend
  - Realistic stock tracking with reorder simulation
  - Cash/UPI/credit payment mix (45/45/10)
  - Partitioned CSVs (by year-month) ready for GCS upload

Usage:
    python src/data/generate_data.py

Output:
    data/raw/transactions/transactions_YYYY-MM.csv  (24 files)
    data/output/sku_catalog.json
"""

import os
import sys
import json
import logging
import random
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: project root and .env
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("smb.generate_data")

# ---------------------------------------------------------------------------
# Constants from env
# ---------------------------------------------------------------------------
DATA_START = date.fromisoformat(os.getenv("DATA_START_DATE", "2023-01-01"))
DATA_END   = date.fromisoformat(os.getenv("DATA_END_DATE",   "2024-12-31"))
STORE_ID   = os.getenv("STORE_ID", "STORE_MUMBAI_001")
RANDOM_SEED = 42

RAW_DIR    = PROJECT_ROOT / "data" / "raw" / "transactions"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SKU Catalog (50 SKUs across 8 categories)
# ---------------------------------------------------------------------------
SKU_CATALOG: List[Dict] = [
    # Grains & Staples (10)
    {"sku_id": "SKU001", "sku_name": "Aashirvaad Atta 5kg",       "category": "Grains & Staples", "base_price": 260.0,  "unit": "bag",    "reorder_point": 20, "max_stock": 200},
    {"sku_id": "SKU002", "sku_name": "Aashirvaad Atta 10kg",      "category": "Grains & Staples", "base_price": 490.0,  "unit": "bag",    "reorder_point": 15, "max_stock": 150},
    {"sku_id": "SKU003", "sku_name": "India Gate Basmati 5kg",    "category": "Grains & Staples", "base_price": 580.0,  "unit": "bag",    "reorder_point": 15, "max_stock": 120},
    {"sku_id": "SKU004", "sku_name": "Sona Masoori Rice 5kg",     "category": "Grains & Staples", "base_price": 280.0,  "unit": "bag",    "reorder_point": 20, "max_stock": 180},
    {"sku_id": "SKU005", "sku_name": "Toor Dal 1kg",               "category": "Grains & Staples", "base_price": 148.0,  "unit": "pack",   "reorder_point": 30, "max_stock": 300},
    {"sku_id": "SKU006", "sku_name": "Chana Dal 1kg",              "category": "Grains & Staples", "base_price": 110.0,  "unit": "pack",   "reorder_point": 30, "max_stock": 300},
    {"sku_id": "SKU007", "sku_name": "Moong Dal 1kg",              "category": "Grains & Staples", "base_price": 132.0,  "unit": "pack",   "reorder_point": 25, "max_stock": 250},
    {"sku_id": "SKU008", "sku_name": "Poha 500g",                  "category": "Grains & Staples", "base_price": 48.0,   "unit": "pack",   "reorder_point": 40, "max_stock": 400},
    {"sku_id": "SKU009", "sku_name": "Suji / Rava 1kg",           "category": "Grains & Staples", "base_price": 52.0,   "unit": "pack",   "reorder_point": 35, "max_stock": 350},
    {"sku_id": "SKU010", "sku_name": "Maida 1kg",                  "category": "Grains & Staples", "base_price": 38.0,   "unit": "pack",   "reorder_point": 40, "max_stock": 400},
    # Oils & Fats (6)
    {"sku_id": "SKU011", "sku_name": "Fortune Sunflower Oil 1L",   "category": "Oils & Fats",      "base_price": 148.0,  "unit": "bottle", "reorder_point": 30, "max_stock": 300},
    {"sku_id": "SKU012", "sku_name": "Fortune Sunflower Oil 5L",   "category": "Oils & Fats",      "base_price": 720.0,  "unit": "can",    "reorder_point": 15, "max_stock": 120},
    {"sku_id": "SKU013", "sku_name": "Patanjali Mustard Oil 1L",   "category": "Oils & Fats",      "base_price": 168.0,  "unit": "bottle", "reorder_point": 25, "max_stock": 200},
    {"sku_id": "SKU014", "sku_name": "Parachute Coconut Oil 500ml","category": "Oils & Fats",      "base_price": 118.0,  "unit": "bottle", "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU015", "sku_name": "Amul Desi Ghee 500g",        "category": "Oils & Fats",      "base_price": 310.0,  "unit": "tin",    "reorder_point": 20, "max_stock": 150},
    {"sku_id": "SKU016", "sku_name": "Dalda Vanaspati 1kg",         "category": "Oils & Fats",      "base_price": 120.0,  "unit": "pack",   "reorder_point": 25, "max_stock": 200},
    # Sugar & Salt (4)
    {"sku_id": "SKU017", "sku_name": "Sugar 1kg",                   "category": "Sugar & Salt",     "base_price": 48.0,   "unit": "pack",   "reorder_point": 50, "max_stock": 500},
    {"sku_id": "SKU018", "sku_name": "Sugar 5kg",                   "category": "Sugar & Salt",     "base_price": 220.0,  "unit": "bag",    "reorder_point": 25, "max_stock": 200},
    {"sku_id": "SKU019", "sku_name": "Iodized Salt 1kg",            "category": "Sugar & Salt",     "base_price": 22.0,   "unit": "pack",   "reorder_point": 60, "max_stock": 600},
    {"sku_id": "SKU020", "sku_name": "Jaggery Powder 500g",         "category": "Sugar & Salt",     "base_price": 65.0,   "unit": "pack",   "reorder_point": 40, "max_stock": 400},
    # Beverages (6)
    {"sku_id": "SKU021", "sku_name": "Tata Tea Premium 250g",       "category": "Beverages",        "base_price": 92.0,   "unit": "pack",   "reorder_point": 40, "max_stock": 350},
    {"sku_id": "SKU022", "sku_name": "Red Label Tea 500g",          "category": "Beverages",        "base_price": 168.0,  "unit": "pack",   "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU023", "sku_name": "Bru Coffee 100g",             "category": "Beverages",        "base_price": 115.0,  "unit": "jar",    "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU024", "sku_name": "Horlicks 500g",               "category": "Beverages",        "base_price": 310.0,  "unit": "jar",    "reorder_point": 20, "max_stock": 150},
    {"sku_id": "SKU025", "sku_name": "Boost 500g",                  "category": "Beverages",        "base_price": 280.0,  "unit": "jar",    "reorder_point": 20, "max_stock": 150},
    {"sku_id": "SKU026", "sku_name": "Complan 200g",                "category": "Beverages",        "base_price": 195.0,  "unit": "pack",   "reorder_point": 25, "max_stock": 200},
    # Snacks (6)
    {"sku_id": "SKU027", "sku_name": "Parle-G 100g",                "category": "Snacks",           "base_price": 10.0,   "unit": "pack",   "reorder_point": 100,"max_stock": 800},
    {"sku_id": "SKU028", "sku_name": "Marie Lite Biscuits 250g",    "category": "Snacks",           "base_price": 38.0,   "unit": "pack",   "reorder_point": 60, "max_stock": 500},
    {"sku_id": "SKU029", "sku_name": "Monaco Crackers 200g",        "category": "Snacks",           "base_price": 42.0,   "unit": "pack",   "reorder_point": 50, "max_stock": 400},
    {"sku_id": "SKU030", "sku_name": "Lays Classic 50g",            "category": "Snacks",           "base_price": 20.0,   "unit": "pack",   "reorder_point": 80, "max_stock": 600},
    {"sku_id": "SKU031", "sku_name": "Kurkure Masala 90g",          "category": "Snacks",           "base_price": 30.0,   "unit": "pack",   "reorder_point": 70, "max_stock": 550},
    {"sku_id": "SKU032", "sku_name": "Haldiram Namkeen 200g",       "category": "Snacks",           "base_price": 55.0,   "unit": "pack",   "reorder_point": 60, "max_stock": 500},
    # Personal Care (8)
    {"sku_id": "SKU033", "sku_name": "Lifebuoy Soap 125g",          "category": "Personal Care",    "base_price": 28.0,   "unit": "bar",    "reorder_point": 80, "max_stock": 600},
    {"sku_id": "SKU034", "sku_name": "Dettol Soap 75g",             "category": "Personal Care",    "base_price": 42.0,   "unit": "bar",    "reorder_point": 60, "max_stock": 500},
    {"sku_id": "SKU035", "sku_name": "Dove Soap 75g",               "category": "Personal Care",    "base_price": 52.0,   "unit": "bar",    "reorder_point": 50, "max_stock": 400},
    {"sku_id": "SKU036", "sku_name": "Head & Shoulders 200ml",      "category": "Personal Care",    "base_price": 230.0,  "unit": "bottle", "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU037", "sku_name": "Clinic Plus Shampoo 200ml",   "category": "Personal Care",    "base_price": 155.0,  "unit": "bottle", "reorder_point": 35, "max_stock": 300},
    {"sku_id": "SKU038", "sku_name": "Colgate MaxFresh 200g",       "category": "Personal Care",    "base_price": 98.0,   "unit": "tube",   "reorder_point": 40, "max_stock": 350},
    {"sku_id": "SKU039", "sku_name": "Pepsodent 150g",              "category": "Personal Care",    "base_price": 72.0,   "unit": "tube",   "reorder_point": 40, "max_stock": 350},
    {"sku_id": "SKU040", "sku_name": "Nivea Body Lotion 200ml",     "category": "Personal Care",    "base_price": 195.0,  "unit": "bottle", "reorder_point": 25, "max_stock": 200},
    # Household (5)
    {"sku_id": "SKU041", "sku_name": "Surf Excel 500g",             "category": "Household",        "base_price": 115.0,  "unit": "pack",   "reorder_point": 40, "max_stock": 300},
    {"sku_id": "SKU042", "sku_name": "Vim Bar 200g",                "category": "Household",        "base_price": 32.0,   "unit": "bar",    "reorder_point": 60, "max_stock": 500},
    {"sku_id": "SKU043", "sku_name": "Harpic 500ml",                "category": "Household",        "base_price": 115.0,  "unit": "bottle", "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU044", "sku_name": "Phenyl Floor Cleaner 1L",     "category": "Household",        "base_price": 95.0,   "unit": "bottle", "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU045", "sku_name": "Agarbatti Incense 100 sticks","category": "Household",        "base_price": 42.0,   "unit": "pack",   "reorder_point": 50, "max_stock": 400},
    # Dairy & Health (5)
    {"sku_id": "SKU046", "sku_name": "Amul Milk Powder 500g",       "category": "Dairy & Health",   "base_price": 265.0,  "unit": "pack",   "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU047", "sku_name": "Nestomalt 200g",              "category": "Dairy & Health",   "base_price": 148.0,  "unit": "pack",   "reorder_point": 35, "max_stock": 300},
    {"sku_id": "SKU048", "sku_name": "Milo 200g",                   "category": "Dairy & Health",   "base_price": 165.0,  "unit": "pack",   "reorder_point": 30, "max_stock": 250},
    {"sku_id": "SKU049", "sku_name": "Pediasure Vanilla 200g",      "category": "Dairy & Health",   "base_price": 345.0,  "unit": "pack",   "reorder_point": 20, "max_stock": 150},
    {"sku_id": "SKU050", "sku_name": "Ensure Nutrition Drink 200g","category": "Dairy & Health",   "base_price": 398.0,  "unit": "pack",   "reorder_point": 15, "max_stock": 120},
]

FESTIVALS = [
    # 2023
    ("2023-01-14", "Makar Sankranti",  2.0, 3),
    ("2023-01-26", "Republic Day",     1.5, 2),
    ("2023-03-07", "Holi",             3.0, 5),
    ("2023-03-30", "Ram Navami",       1.8, 3),
    ("2023-04-21", "Eid-ul-Fitr",      2.5, 5),
    ("2023-08-15", "Independence Day", 1.8, 2),
    ("2023-09-19", "Ganesh Chaturthi", 2.2, 5),
    ("2023-10-15", "Navratri Start",   2.0, 3),
    ("2023-10-24", "Dussehra",         2.8, 5),
    ("2023-11-12", "Diwali",           4.5, 10),
    ("2023-11-13", "Diwali Day 2",     4.0, 0),
    ("2023-12-25", "Christmas",        1.8, 3),
    ("2023-12-31", "New Year Eve",     2.5, 3),
    # 2024
    ("2024-01-15", "Makar Sankranti",  2.0, 3),
    ("2024-01-22", "Ram Mandir",       2.5, 5),
    ("2024-03-24", "Holi",             3.0, 5),
    ("2024-04-10", "Eid-ul-Fitr",      2.5, 5),
    ("2024-08-15", "Independence Day", 1.8, 2),
    ("2024-09-07", "Ganesh Chaturthi", 2.2, 5),
    ("2024-10-03", "Navratri Start",   2.0, 3),
    ("2024-10-12", "Dussehra",         2.8, 5),
    ("2024-11-01", "Diwali",           4.5, 10),
    ("2024-11-02", "Diwali Day 2",     4.0, 0),
    ("2024-12-25", "Christmas",        1.8, 3),
    ("2024-12-31", "New Year Eve",     2.5, 3),
]

CATEGORY_SEASONALITY = {
    "Grains & Staples":  [1.1, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 1.3, 1.3, 1.1],
    "Oils & Fats":       [1.2, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.2, 1.3, 1.2],
    "Sugar & Salt":      [1.0, 1.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.3, 1.4, 1.1],
    "Beverages":         [1.3, 1.2, 1.1, 1.1, 1.4, 1.5, 1.4, 1.3, 1.1, 1.0, 1.0, 1.3],
    "Snacks":            [1.0, 1.0, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.3, 1.4, 1.2],
    "Personal Care":     [1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.1, 1.1, 1.0],
    "Household":         [1.1, 1.0, 1.2, 1.1, 1.0, 1.0, 1.2, 1.1, 1.2, 1.2, 1.3, 1.1],
    "Dairy & Health":    [1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.2],
}

PAYMENT_MODES    = ["cash", "upi", "credit"]
PAYMENT_WEIGHTS  = [0.45, 0.45, 0.10]
CUSTOMER_TYPES   = ["regular", "new", "credit"]
CUSTOMER_WEIGHTS = [0.60, 0.30, 0.10]
DOW_MULTIPLIERS  = {0: 1.05, 1: 1.00, 2: 1.00, 3: 1.05, 4: 1.15, 5: 1.30, 6: 1.25}


def build_festival_multipliers(start: date, end: date) -> Dict[date, float]:
    multipliers: Dict[date, float] = {}
    for fest_date_str, _, peak_mult, days_before in FESTIVALS:
        fest_date = date.fromisoformat(fest_date_str)
        for d in range(-days_before, 1):
            target = fest_date + timedelta(days=d)
            if start <= target <= end:
                ramp = (days_before + d + 1) / (days_before + 1) if days_before > 0 else 1.0
                boost = 1.0 + (peak_mult - 1.0) * ramp
                multipliers[target] = max(multipliers.get(target, 1.0), boost)
    return multipliers


def monthly_price(base_price: float, year: int, month: int, rng: np.random.Generator) -> float:
    variance_pct = rng.uniform(-0.12, 0.18)
    months_elapsed = (year - 2023) * 12 + (month - 1)
    inflation = 1.0 + months_elapsed * 0.005
    return round(base_price * (1 + variance_pct) * inflation, 2)


def generate_transactions() -> int:
    """Generate all transactions and write partitioned CSVs. Returns total rows."""
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    logger.info("Building festival multiplier calendar...")
    fest_mults = build_festival_multipliers(DATA_START, DATA_END)

    stock: Dict[str, int] = {s["sku_id"]: s["max_stock"] // 2 for s in SKU_CATALOG}
    sku_map = {s["sku_id"]: s for s in SKU_CATALOG}

    month_prices: Dict[Tuple[str, int, int], float] = {}
    for sku in SKU_CATALOG:
        for year in [2023, 2024]:
            for month in range(1, 13):
                month_prices[(sku["sku_id"], year, month)] = monthly_price(
                    sku["base_price"], year, month, rng
                )

    all_dates = pd.date_range(DATA_START, DATA_END, freq="D")
    month_groups: Dict[str, List[Dict]] = {}

    logger.info(f"Generating data for {len(all_dates)} days "
                f"({DATA_START} to {DATA_END}) with {len(SKU_CATALOG)} SKUs...")

    for dt in all_dates:
        current_date = dt.date()
        year, month, dow = current_date.year, current_date.month, current_date.weekday()
        month_key = f"{year}-{month:02d}"
        if month_key not in month_groups:
            month_groups[month_key] = []

        dow_mult  = DOW_MULTIPLIERS[dow]
        fest_mult = fest_mults.get(current_date, 1.0)
        n_tx = int(1500 * dow_mult * fest_mult * rng.uniform(0.85, 1.15))
        n_tx = max(n_tx, 800)

        weights = np.array([
            CATEGORY_SEASONALITY[s["category"]][month - 1] for s in SKU_CATALOG
        ])
        weights = weights / weights.sum()

        chosen_skus = rng.choice(
            [s["sku_id"] for s in SKU_CATALOG], size=n_tx, p=weights, replace=True
        )

        for sku_id in chosen_skus:
            sku = sku_map[sku_id]
            unit_price = month_prices[(sku_id, year, month)]
            qty = min(max(1, int(rng.lognormal(0.5, 0.6))), 10)

            stock_before = stock[sku_id]
            if stock_before <= 0:
                continue
            qty = min(qty, stock_before)
            stock_after = stock_before - qty
            stock[sku_id] = stock_after
            if stock_after < sku["reorder_point"]:
                stock[sku_id] = sku["max_stock"]

            if fest_mult > 1.5:
                unit_price = round(unit_price * rng.uniform(1.0, 1.05), 2)

            ts = datetime.combine(
                current_date,
                time(int(rng.integers(7, 21)), int(rng.integers(0, 60)), int(rng.integers(0, 60)))
            )

            month_groups[month_key].append({
                "transaction_id": str(uuid.uuid4()),
                "date":           current_date.isoformat(),
                "timestamp":      ts.isoformat(),
                "sku_id":         sku_id,
                "sku_name":       sku["sku_name"],
                "category":       sku["category"],
                "quantity":       qty,
                "unit_price":     unit_price,
                "total_amount":   round(unit_price * qty, 2),
                "payment_mode":   rng.choice(PAYMENT_MODES,  p=PAYMENT_WEIGHTS),
                "customer_type":  rng.choice(CUSTOMER_TYPES, p=CUSTOMER_WEIGHTS),
                "stock_before":   stock_before,
                "stock_after":    stock_after,
                "store_id":       STORE_ID,
                "sales_month":    month_key,
            })

        if current_date.day == 1:
            logger.info(f"  {month_key}: generating...")

    logger.info("Writing partitioned CSV files...")
    total_rows = 0
    for month_key, rows in sorted(month_groups.items()):
        out_file = RAW_DIR / f"transactions_{month_key}.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        total_rows += len(rows)
        logger.info(f"  [{month_key}] {len(rows):>8,} rows -> {out_file.name}")

    return total_rows


def save_sku_catalog() -> None:
    out_path = OUTPUT_DIR / "sku_catalog.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(SKU_CATALOG, f, indent=2, ensure_ascii=False)
    logger.info(f"SKU catalog saved -> {out_path}")


def main() -> None:
    logger.info("=" * 60)
    logger.info("SMB Intelligence -- Part 1: Data Generation")
    logger.info(f"Date range : {DATA_START} to {DATA_END}")
    logger.info(f"Store ID   : {STORE_ID}")
    logger.info(f"SKU count  : {len(SKU_CATALOG)}")
    logger.info("=" * 60)
    try:
        total_rows = generate_transactions()
        save_sku_catalog()
        logger.info(f"Generation complete: {total_rows:,} total rows")
        if total_rows < 1_000_000:
            logger.warning(f"Row count {total_rows:,} < 1,000,000 target!")
            sys.exit(1)
        logger.info(f"Target met: {total_rows:,} >= 1,000,000")
    except Exception as exc:
        logger.exception(f"Data generation failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
