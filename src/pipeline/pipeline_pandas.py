"""
pipeline_pandas.py -- Part 2: Pandas Feature Engineering Pipeline
=================================================================
Transforms raw 1.45M+ row transaction data using pure pandas.
This is the BASELINE for the cuDF benchmark comparison.

Transformations:
  1. Date parsing + temporal features (year, month, dow, is_weekend, quarter)
  2. Cash flow features (effective_revenue, credit_given, cashflow_delta)
  3. Stockout risk flag (stock_after < 50)
  4. Category label encoding
  5. Festival proximity (days_to_next_festival, is_festival_week)
  6. Daily-SKU aggregation (GroupBy sku_id + date)
  7. Rolling windows per SKU (7d, 30d revenue; 7d price std)
  8. Cumulative metrics (cumulative revenue, qty per SKU)

Output grain: one row per (sku_id, date) -- 50 SKUs x 731 days = 36,550 rows
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("smb.pipeline_pandas")

FESTIVAL_DATES = [
    "2023-01-14", "2023-01-26", "2023-03-07", "2023-04-21",
    "2023-08-15", "2023-09-19", "2023-10-15", "2023-10-24",
    "2023-11-12", "2023-12-25", "2023-12-31",
    "2024-01-15", "2024-01-22", "2024-03-24", "2024-04-10",
    "2024-08-15", "2024-09-07", "2024-10-03", "2024-10-12",
    "2024-11-01", "2024-12-25", "2024-12-31",
]

CATEGORY_CODES = {
    "Grains & Staples": 0, "Oils & Fats": 1, "Sugar & Salt": 2,
    "Beverages": 3, "Snacks": 4, "Personal Care": 5,
    "Household": 6, "Dairy & Health": 7,
}


class PandasPipeline:
    """Pure-pandas feature engineering pipeline. Call .transform(df) to run."""

    def _raw_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"]       = pd.to_datetime(df["date"])
        df["year"]       = df["date"].dt.year.astype("int16")
        df["month"]      = df["date"].dt.month.astype("int8")
        df["day"]        = df["date"].dt.day.astype("int8")
        df["dow"]        = df["date"].dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("int8")
        df["quarter"]    = df["date"].dt.quarter.astype("int8")
        is_non_credit            = (df["payment_mode"] != "credit").astype("float32")
        df["effective_revenue"]  = (df["total_amount"] * is_non_credit).astype("float32")
        df["credit_given"]       = (df["total_amount"] * (df["payment_mode"] == "credit").astype("float32")).astype("float32")
        df["cashflow_delta"]     = (df["effective_revenue"] - df["credit_given"]).astype("float32")
        df["stockout_risk"]      = (df["stock_after"] < 50).astype("int8")
        df["category_encoded"]   = df["category"].map(CATEGORY_CODES).fillna(-1).astype("int8")
        return df

    def _aggregate_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby(["sku_id", "date"], observed=True).agg(
            sku_name=         ("sku_name",       "first"),
            category=         ("category",        "first"),
            category_encoded= ("category_encoded","first"),
            year=             ("year",            "first"),
            month=            ("month",           "first"),
            day=              ("day",             "first"),
            dow=              ("dow",             "first"),
            is_weekend=       ("is_weekend",      "first"),
            quarter=          ("quarter",         "first"),
            daily_revenue=    ("total_amount",    "sum"),
            daily_qty=        ("quantity",        "sum"),
            daily_txn_count=  ("transaction_id",  "count"),
            daily_avg_price=  ("unit_price",      "mean"),
            daily_cashflow=   ("cashflow_delta",   "sum"),
            daily_credit=     ("credit_given",     "sum"),
            daily_stockout=   ("stockout_risk",    "max"),
        ).reset_index()
        for col in ["daily_revenue", "daily_cashflow", "daily_credit", "daily_avg_price"]:
            agg[col] = agg[col].astype("float32")
        for col in ["daily_qty", "daily_txn_count"]:
            agg[col] = agg[col].astype("int32")
        return agg

    def _rolling_features(self, agg: pd.DataFrame) -> pd.DataFrame:
        agg = agg.sort_values(["sku_id", "date"]).set_index("date")
        grp = agg.groupby("sku_id", observed=True)
        agg["rolling_revenue_7d"]   = grp["daily_revenue"].transform(lambda x: x.rolling(7,  min_periods=1).sum())
        agg["rolling_revenue_30d"]  = grp["daily_revenue"].transform(lambda x: x.rolling(30, min_periods=1).sum())
        agg["rolling_qty_7d"]       = grp["daily_qty"].transform(lambda x: x.rolling(7, min_periods=1).sum())
        agg["rolling_price_std_7d"] = grp["daily_avg_price"].transform(lambda x: x.rolling(7, min_periods=2).std().fillna(0))
        agg["cum_revenue"] = grp["daily_revenue"].cumsum()
        agg["cum_qty"]     = grp["daily_qty"].cumsum()
        return agg.reset_index()

    def _festival_features(self, agg: pd.DataFrame) -> pd.DataFrame:
        fest = pd.to_datetime(FESTIVAL_DATES).sort_values()
        unique_dates = pd.to_datetime(agg["date"].unique())
        days_map = {}
        for d in unique_dates:
            future = fest[fest >= d]
            days_map[d] = int((future.min() - d).days) if not future.empty else 999
        agg["days_to_festival"] = agg["date"].map(days_map).astype("int16")
        agg["is_festival_week"] = (agg["days_to_festival"] <= 7).astype("int8")
        return agg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: raw 1.45M rows -> 36K row daily-SKU feature table."""
        logger.info(f"PandasPipeline.transform() -- input shape: {df.shape}")
        df = self._raw_transforms(df)
        agg = self._aggregate_daily(df)
        agg = self._rolling_features(agg)
        agg = self._festival_features(agg)
        logger.info(f"PandasPipeline.transform() -- output shape: {agg.shape}")
        return agg
