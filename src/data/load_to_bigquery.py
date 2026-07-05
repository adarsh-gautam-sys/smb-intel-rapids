"""
load_to_bigquery.py -- Part 1: Load GCS CSVs into BigQuery
===========================================================
Loads transaction CSVs from GCS into BigQuery table:
    {BIGQUERY_DATASET}.raw_transactions

Features:
  - Creates dataset + table if they don't exist
  - DATE partitioning on `date` field, clustering on (sku_id, category)
  - Loads from GCS wildcard: gs://bucket/raw/transactions/*/*.csv
  - WRITE_TRUNCATE (idempotent)
  - Verifies row count via COUNT(*)
  - Prints required completion message

Usage:
    python src/data/load_to_bigquery.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("smb.load_to_bigquery")

GCP_PROJECT_ID    = os.getenv("GCP_PROJECT_ID", "")
GCS_BUCKET_NAME   = os.getenv("GCS_BUCKET_NAME", "smb-intelligence-bucket")
BIGQUERY_DATASET  = os.getenv("BIGQUERY_DATASET", "smb_intelligence")
BIGQUERY_LOCATION = os.getenv("BIGQUERY_LOCATION", "asia-south1")
CREDS_PATH        = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

TABLE_ID         = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.raw_transactions"
GCS_WILDCARD_URI = f"gs://{GCS_BUCKET_NAME}/raw/transactions/*/*.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BQ_SCHEMA = [
    {"name": "transaction_id", "type": "STRING",   "mode": "REQUIRED"},
    {"name": "date",           "type": "DATE",      "mode": "REQUIRED"},
    {"name": "timestamp",      "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "sku_id",         "type": "STRING",    "mode": "REQUIRED"},
    {"name": "sku_name",       "type": "STRING",    "mode": "REQUIRED"},
    {"name": "category",       "type": "STRING",    "mode": "REQUIRED"},
    {"name": "quantity",       "type": "INTEGER",   "mode": "REQUIRED"},
    {"name": "unit_price",     "type": "FLOAT64",   "mode": "REQUIRED"},
    {"name": "total_amount",   "type": "FLOAT64",   "mode": "REQUIRED"},
    {"name": "payment_mode",   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "customer_type",  "type": "STRING",    "mode": "NULLABLE"},
    {"name": "stock_before",   "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "stock_after",    "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "store_id",       "type": "STRING",    "mode": "REQUIRED"},
    {"name": "sales_month",    "type": "STRING",    "mode": "REQUIRED"},
]


def get_bq_client():
    try:
        from google.cloud import bigquery  # type: ignore
        if CREDS_PATH and Path(CREDS_PATH).exists():
            from google.oauth2 import service_account  # type: ignore
            creds = service_account.Credentials.from_service_account_file(CREDS_PATH)
            return bigquery.Client(project=GCP_PROJECT_ID, credentials=creds)
        return bigquery.Client(project=GCP_PROJECT_ID)
    except Exception as exc:
        raise RuntimeError(f"BQ client init failed: {exc}") from exc


def ensure_dataset(client) -> None:
    from google.cloud import bigquery  # type: ignore
    from google.api_core.exceptions import NotFound  # type: ignore
    dataset_ref = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}"
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_ref!r} exists")
    except NotFound:
        logger.info(f"Creating dataset {dataset_ref!r} in {BIGQUERY_LOCATION}...")
        ds = bigquery.Dataset(dataset_ref)
        ds.location = BIGQUERY_LOCATION
        ds.description = "SMB Intelligence Platform"
        client.create_dataset(ds, timeout=30)
        logger.info(f"Dataset created")


def ensure_table(client) -> None:
    from google.cloud import bigquery  # type: ignore
    from google.api_core.exceptions import NotFound  # type: ignore
    try:
        client.get_table(TABLE_ID)
        logger.info(f"Table {TABLE_ID!r} exists")
    except NotFound:
        schema = [bigquery.SchemaField(c["name"], c["type"], mode=c["mode"]) for c in BQ_SCHEMA]
        table = bigquery.Table(TABLE_ID, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="date"
        )
        table.clustering_fields = ["sku_id", "category"]
        client.create_table(table, timeout=30)
        logger.info(f"Table created: DATE partition on date, clustered on (sku_id, category)")


def load_from_gcs(client) -> int:
    from google.cloud import bigquery  # type: ignore
    logger.info(f"Starting load job from {GCS_WILDCARD_URI}...")
    schema = [bigquery.SchemaField(c["name"], c["type"], mode=c["mode"]) for c in BQ_SCHEMA]
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        null_marker="",
        allow_quoted_newlines=True,
    )
    try:
        load_job = client.load_table_from_uri(
            GCS_WILDCARD_URI, TABLE_ID, job_config=job_config, location=BIGQUERY_LOCATION
        )
        logger.info(f"Job submitted: {load_job.job_id}. Waiting...")
        load_job.result(timeout=600)
        if load_job.errors:
            raise RuntimeError(f"Load errors: {load_job.errors}")
        logger.info(f"Load complete: {load_job.output_rows:,} rows, {load_job.output_bytes:,} bytes")
        return load_job.output_rows
    except Exception as exc:
        raise RuntimeError(f"Load job failed: {exc}") from exc


def verify_row_count(client) -> int:
    try:
        result = list(client.query(f"SELECT COUNT(*) AS n FROM `{TABLE_ID}`").result())[0]
        count = result.n
        logger.info(f"COUNT(*) verified: {count:,} rows")
        return count
    except Exception as exc:
        logger.warning(f"COUNT(*) failed: {exc}")
        return -1


def main() -> None:
    logger.info("=" * 60)
    logger.info("SMB Intelligence -- Part 1: BigQuery Load")
    logger.info(f"Target  : {TABLE_ID}")
    logger.info(f"Source  : {GCS_WILDCARD_URI}")
    logger.info("=" * 60)
    if not GCP_PROJECT_ID:
        logger.error("GCP_PROJECT_ID not set. Cannot load.")
        sys.exit(1)
    try:
        client = get_bq_client()
        ensure_dataset(client)
        ensure_table(client)
        output_rows = load_from_gcs(client)
        verified = verify_row_count(client)
        final_count = verified if verified > 0 else output_rows
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "table": TABLE_ID,
            "rows_loaded": final_count,
            "status": "success" if final_count >= 1_000_000 else "warning_low_count",
        }
        with open(OUTPUT_DIR / "bq_load_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        # Required AGENTS.md completion message
        print(f"Loaded {final_count:,} rows into {BIGQUERY_DATASET}.raw_transactions")
        logger.info(f"COMPLETION: Loaded {final_count:,} rows into {BIGQUERY_DATASET}.raw_transactions")
        if final_count < 1_000_000:
            logger.warning(f"Row count {final_count:,} < 1,000,000")
            sys.exit(1)
    except Exception as exc:
        logger.exception(f"BigQuery load failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
