"""
benchmark.py -- Part 2: cuDF vs Pandas Pipeline Benchmark
=========================================================
Orchestrates both pipelines, measures wall-clock time (3 runs, median),
validates output equivalence, prints speedup table, writes benchmark_results.json.

Usage:
    python src/pipeline/benchmark.py
    # Windows: $env:PYTHONIOENCODING='utf-8'; python src/pipeline/benchmark.py

Output:
    data/processed/features_pandas.parquet
    data/processed/features_cudf.parquet   (or pandas fallback)
    data/output/benchmark_results.json
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("smb.benchmark")

RAW_DIR       = PROJECT_ROOT / "data" / "raw" / "transactions"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "output"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_ALL        = PROCESSED_DIR / "transactions_all.parquet"
PARQUET_PANDAS_OUT = PROCESSED_DIR / "features_pandas.parquet"
PARQUET_CUDF_OUT   = PROCESSED_DIR / "features_cudf.parquet"
BENCHMARK_JSON     = OUTPUT_DIR    / "benchmark_results.json"
N_RUNS = 3


def load_raw_data() -> pd.DataFrame:
    if PARQUET_ALL.exists():
        logger.info(f"Loading from cached parquet: {PARQUET_ALL}")
        df = pd.read_parquet(PARQUET_ALL)
        logger.info(f"Loaded {len(df):,} rows from cache")
        return df
    csv_files = sorted(RAW_DIR.glob("transactions_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {RAW_DIR}. Run generate_data.py first.")
    logger.info(f"Reading {len(csv_files)} CSV files...")
    dfs = []
    for p in csv_files:
        chunk = pd.read_csv(p)
        dfs.append(chunk)
        logger.info(f"  Loaded {p.name}: {len(chunk):,} rows")
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows: {len(df):,}")
    df.to_parquet(PARQUET_ALL, index=False, compression="snappy")
    logger.info(f"Cached to {PARQUET_ALL}")
    return df


def timed_run(pipeline_instance, df: pd.DataFrame, label: str) -> tuple:
    times = []
    result_df = None
    for run in range(1, N_RUNS + 1):
        df_copy = df.copy()
        t0 = time.perf_counter()
        result_df = pipeline_instance.transform(df_copy)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        logger.info(f"  [{label}] Run {run}/{N_RUNS}: {elapsed:.3f}s")
    median_time = float(np.median(times))
    logger.info(f"  [{label}] Median time: {median_time:.3f}s over {N_RUNS} runs")
    return median_time, result_df


def validate_outputs(pdf: pd.DataFrame, cdf: pd.DataFrame, tol: float = 1e-3) -> dict:
    result = {
        "shapes_match": False, "columns_match": False,
        "groupby_revenue_match": False, "groupby_qty_match": False,
        "validation_passed": False, "errors": [],
    }
    if pdf.shape != cdf.shape:
        result["errors"].append(f"Shape mismatch: pandas={pdf.shape}, cudf={cdf.shape}")
    else:
        result["shapes_match"] = True
    if set(pdf.columns) != set(cdf.columns):
        result["errors"].append("Column mismatch")
    else:
        result["columns_match"] = True
    if result["shapes_match"] and result["columns_match"]:
        sort_cols = ["sku_id", "date"]
        p = pdf.sort_values(sort_cols).reset_index(drop=True)
        c = cdf.sort_values(sort_cols).reset_index(drop=True)
        if (p.groupby("sku_id")["daily_revenue"].sum() - c.groupby("sku_id")["daily_revenue"].sum()).abs().max() <= tol:
            result["groupby_revenue_match"] = True
        else:
            result["errors"].append("Revenue groupby mismatch")
        if (p.groupby("sku_id")["daily_qty"].sum() - c.groupby("sku_id")["daily_qty"].sum()).abs().max() <= tol:
            result["groupby_qty_match"] = True
        else:
            result["errors"].append("Qty groupby mismatch")
    result["validation_passed"] = all([
        result["shapes_match"], result["columns_match"],
        result["groupby_revenue_match"], result["groupby_qty_match"]
    ])
    return result


def main() -> None:
    from src.pipeline.pipeline_pandas import PandasPipeline
    from src.pipeline.pipeline_cudf import CuDFPipeline, cudf_available, cudf_reason

    logger.info("=" * 70)
    logger.info("SMB Intelligence -- Part 2: cuDF vs Pandas Benchmark")
    logger.info("=" * 70)

    try:
        df = load_raw_data()
        n_rows = len(df)
        logger.info(f"Dataset: {n_rows:,} rows x {len(df.columns)} columns")

        gpu_ok = cudf_available()
        if not gpu_ok:
            logger.info(f"cuDF unavailable: {cudf_reason()}")
            logger.info("Running pandas for BOTH pipelines (honest fallback)")

        logger.info("\n--- PANDAS PIPELINE ---")
        p_time, p_df = timed_run(PandasPipeline(), df, "pandas")
        p_df.to_parquet(PARQUET_PANDAS_OUT, index=False, compression="snappy")

        logger.info("\n--- cuDF PIPELINE ---")
        c_time, c_df = timed_run(CuDFPipeline(), df, "cudf")
        c_df.to_parquet(PARQUET_CUDF_OUT, index=False, compression="snappy")

        logger.info("\n--- VALIDATION ---")
        val = validate_outputs(p_df, c_df)
        if val["validation_passed"]:
            logger.info("Both pipelines produce equivalent outputs")
        else:
            for err in val["errors"]:
                logger.warning(f"  {err}")

        speedup = round(p_time / c_time, 2) if c_time > 0 else 1.0
        results = {
            "timestamp":               datetime.now(timezone.utc).isoformat(),
            "rows_processed":          n_rows,
            "n_runs_per_pipeline":     N_RUNS,
            "gpu_available":           gpu_ok,
            "cudf_fallback_to_pandas": not gpu_ok,
            "pandas":                  {"time_seconds": round(p_time, 4), "rows_per_second": int(n_rows / p_time)},
            "cudf":                    {
                "time_seconds": round(c_time, 4), "rows_per_second": int(n_rows / c_time),
                "note": "" if gpu_ok else f"Fallback to pandas: {cudf_reason()}"
            },
            "speedup_factor": speedup,
            "speedup_note": (
                f"GPU acceleration: {speedup:.2f}x faster than pandas" if gpu_ok else
                "GPU not available. Expected speedup on NVIDIA T4/A100: 8-15x. Run on GCP GPU VM for demo."
            ),
            "output_shape":   {"rows": p_df.shape[0], "cols": p_df.shape[1]},
            "output_columns": list(p_df.columns),
            "validation":     val,
        }

        with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Benchmark results saved -> {BENCHMARK_JSON}")

        print("\n" + "=" * 70)
        print("  SMB Intelligence -- cuDF vs Pandas Benchmark Results")
        print("=" * 70)
        table = [
            ["pandas (CPU)", f"{p_time:.3f}", f"{n_rows/p_time:,.0f}", "1.00x"],
            ["cuDF  (GPU)",  f"{c_time:.3f}", f"{n_rows/c_time:,.0f}", f"{speedup:.2f}x"],
        ]
        print(tabulate(table, headers=["Engine", "Time (s)", "Rows/sec", "Speedup"], tablefmt="grid"))
        print(f"\n  Rows processed : {n_rows:,}")
        print(f"  Output shape   : {p_df.shape[0]:,} rows x {p_df.shape[1]} cols")
        print(f"  GPU available  : {gpu_ok}")
        if not gpu_ok:
            print(f"  Expected GPU   : 8-15x speedup on NVIDIA T4/A100")
        print(f"\n  Validation     : {'PASSED' if val['validation_passed'] else 'FAILED'}")
        print("=" * 70)
        logger.info("Benchmark complete")

        if not val["validation_passed"]:
            logger.error("Validation FAILED")
            sys.exit(1)

    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Benchmark failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
