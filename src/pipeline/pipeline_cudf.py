"""
pipeline_cudf.py -- Part 2: cuDF GPU-Accelerated Feature Engineering Pipeline
=============================================================================
GPU-accelerated version of the pandas pipeline using NVIDIA RAPIDS cuDF.
Architecturally IDENTICAL to pipeline_pandas.py. Automatic pandas fallback
when no GPU or cuDF installation is detected.

GPU Environment Setup (Linux/CUDA required):
    conda create -n rapids-smb python=3.11
    conda install -c rapidsai -c conda-forge -c nvidia cudf=24.02 cuda-version=12.0

Expected speedup on NVIDIA T4/A100: 8-15x for GroupBy + rolling on 1.45M rows.
"""

import logging
import pandas as pd

logger = logging.getLogger("smb.pipeline_cudf")

_CUDF_AVAILABLE = False
_CUDF_REASON    = ""

try:
    import cudf  # type: ignore
    _test = cudf.DataFrame({"a": [1, 2, 3]})
    del _test
    _CUDF_AVAILABLE = True
    logger.info("cuDF detected -- GPU acceleration enabled")
except ImportError as exc:
    _CUDF_REASON = f"cuDF not installed ({exc}). Requires Python 3.10/3.11, CUDA GPU, and conda-installed rapids."
    logger.warning(f"cuDF unavailable: {_CUDF_REASON}")
except Exception as exc:
    _CUDF_REASON = f"cuDF init error ({type(exc).__name__}: {exc})"
    logger.warning(f"cuDF unavailable: {_CUDF_REASON}")


def cudf_available() -> bool:
    return _CUDF_AVAILABLE


def cudf_reason() -> str:
    return _CUDF_REASON


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


class CuDFPipeline:
    """
    cuDF GPU-accelerated pipeline with automatic pandas fallback.
    Always returns a pandas DataFrame.
    """

    def __init__(self) -> None:
        self.used_gpu = _CUDF_AVAILABLE
        if not _CUDF_AVAILABLE:
            logger.info(f"CuDFPipeline: falling back to pandas -- {_CUDF_REASON}")
            from src.pipeline.pipeline_pandas import PandasPipeline
            self._fallback = PandasPipeline()
        else:
            self._fallback = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not _CUDF_AVAILABLE:
            logger.info("CuDFPipeline.transform() -- using pandas FALLBACK")
            return self._fallback.transform(df)
        logger.info(f"CuDFPipeline.transform() -- GPU mode, input: {df.shape}")
        return self._transform_gpu(df)

    def _transform_gpu(self, pandas_df: pd.DataFrame) -> pd.DataFrame:
        """GPU pipeline: transfer -> cuDF transforms -> GroupBy -> rolling -> return pandas."""
        import cudf  # type: ignore  # noqa: F811
        logger.info("  Transferring data to GPU...")
        gdf = cudf.from_pandas(pandas_df)

        gdf["date"]       = cudf.to_datetime(gdf["date"])
        gdf["year"]       = gdf["date"].dt.year.astype("int16")
        gdf["month"]      = gdf["date"].dt.month.astype("int8")
        gdf["day"]        = gdf["date"].dt.day.astype("int8")
        gdf["dow"]        = gdf["date"].dt.dayofweek.astype("int8")
        gdf["is_weekend"] = (gdf["dow"] >= 5).astype("int8")
        gdf["quarter"]    = gdf["date"].dt.quarter.astype("int8")

        is_non_credit           = (gdf["payment_mode"] != "credit").astype("float32")
        gdf["effective_revenue"]= (gdf["total_amount"] * is_non_credit).astype("float32")
        gdf["credit_given"]     = (gdf["total_amount"] * (gdf["payment_mode"] == "credit").astype("float32")).astype("float32")
        gdf["cashflow_delta"]   = (gdf["effective_revenue"] - gdf["credit_given"]).astype("float32")
        gdf["stockout_risk"]    = (gdf["stock_after"] < 50).astype("int8")

        cat_map = cudf.DataFrame({
            "category": cudf.Series(list(CATEGORY_CODES.keys())),
            "category_encoded": cudf.Series(list(CATEGORY_CODES.values()), dtype="int8")
        })
        gdf = gdf.merge(cat_map, on="category", how="left")
        gdf["category_encoded"] = gdf["category_encoded"].fillna(-1).astype("int8")

        # GroupBy aggregation on GPU -- the key speedup operation
        agg_gdf = gdf.groupby(["sku_id", "date"]).agg(
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

        # Rolling + cumulative: transfer back to CPU for multi-group rolling
        agg_pd = agg_gdf.sort_values(["sku_id", "date"]).to_pandas().set_index("date")
        grp = agg_pd.groupby("sku_id", observed=True)
        agg_pd["rolling_revenue_7d"]   = grp["daily_revenue"].transform(lambda x: x.rolling(7, min_periods=1).sum())
        agg_pd["rolling_revenue_30d"]  = grp["daily_revenue"].transform(lambda x: x.rolling(30, min_periods=1).sum())
        agg_pd["rolling_qty_7d"]       = grp["daily_qty"].transform(lambda x: x.rolling(7, min_periods=1).sum())
        agg_pd["rolling_price_std_7d"] = grp["daily_avg_price"].transform(lambda x: x.rolling(7, min_periods=2).std().fillna(0))
        agg_pd["cum_revenue"] = grp["daily_revenue"].cumsum()
        agg_pd["cum_qty"]     = grp["daily_qty"].cumsum()
        agg_pd = agg_pd.reset_index()

        # Festival proximity
        import pandas as pd_local
        fest = pd_local.to_datetime(FESTIVAL_DATES).sort_values()
        unique_dates = pd_local.to_datetime(agg_pd["date"].unique())
        days_map = {}
        for d in unique_dates:
            future = fest[fest >= d]
            days_map[d] = int((future.min() - d).days) if not future.empty else 999
        agg_pd["days_to_festival"] = agg_pd["date"].map(days_map).astype("int16")
        agg_pd["is_festival_week"] = (agg_pd["days_to_festival"] <= 7).astype("int8")

        logger.info(f"CuDFPipeline.transform() -- GPU output shape: {agg_pd.shape}")
        return agg_pd
