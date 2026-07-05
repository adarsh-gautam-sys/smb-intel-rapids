"""
upload_to_gcs.py -- Part 1: Upload Partitioned CSVs to Google Cloud Storage
===========================================================================
Uploads all data/raw/transactions/transactions_YYYY-MM.csv files to GCS at
gs://{GCS_BUCKET_NAME}/raw/transactions/year=YYYY/month=MM/transactions.csv

Features:
  - Creates GCS bucket if it does not exist
  - Concurrent uploads (ThreadPoolExecutor, 4 workers)
  - 3x exponential backoff retry per file
  - Verifies upload by checking GCS object size
  - Saves upload manifest to data/output/upload_manifest.json

Usage:
    python src/data/upload_to_gcs.py
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("smb.upload_to_gcs")

GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID", "")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "smb-intelligence-bucket")
GCS_REGION      = os.getenv("GCS_REGION", "asia-south1")
CREDS_PATH      = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
MAX_RETRIES     = 3
MAX_WORKERS     = 4

RAW_DIR       = PROJECT_ROOT / "data" / "raw" / "transactions"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUTPUT_DIR / "upload_manifest.json"


def get_gcs_client():
    try:
        from google.cloud import storage  # type: ignore
        if CREDS_PATH and Path(CREDS_PATH).exists():
            from google.oauth2 import service_account  # type: ignore
            creds = service_account.Credentials.from_service_account_file(CREDS_PATH)
            return storage.Client(project=GCP_PROJECT_ID, credentials=creds)
        return storage.Client(project=GCP_PROJECT_ID)
    except Exception as exc:
        raise RuntimeError(f"GCS client init failed: {exc}") from exc


def ensure_bucket(client, bucket_name: str) -> None:
    try:
        bucket = client.lookup_bucket(bucket_name)
        if bucket is None:
            logger.info(f"Creating bucket {bucket_name!r} in {GCS_REGION}...")
            new_bucket = client.bucket(bucket_name)
            new_bucket.storage_class = "STANDARD"
            client.create_bucket(new_bucket, location=GCS_REGION)
            logger.info(f"Bucket gs://{bucket_name} created")
        else:
            logger.info(f"Bucket gs://{bucket_name} exists")
    except Exception as exc:
        raise RuntimeError(f"Bucket operation failed: {exc}") from exc


def gcs_path_for(filename: str) -> Optional[str]:
    stem = Path(filename).stem
    if not stem.startswith("transactions_"):
        return None
    parts = stem.replace("transactions_", "").split("-")
    if len(parts) != 2:
        return None
    year, month = parts
    return f"raw/transactions/year={year}/month={month}/transactions.csv"


def upload_file(client, local_path: Path, gcs_object_path: str) -> Dict:
    result = {
        "local_file": str(local_path),
        "gcs_uri": f"gs://{GCS_BUCKET_NAME}/{gcs_object_path}",
        "local_size_bytes": local_path.stat().st_size,
        "gcs_size_bytes": None,
        "status": "pending",
        "attempts": 0,
        "error": None,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        result["attempts"] = attempt
        try:
            blob = client.bucket(GCS_BUCKET_NAME).blob(gcs_object_path)
            blob.upload_from_filename(str(local_path))
            blob.reload()
            if blob.size == 0:
                raise ValueError("0-byte upload")
            result["gcs_size_bytes"] = blob.size
            result["status"] = "success"
            logger.info(f"  Uploaded {local_path.name} -> gs://{GCS_BUCKET_NAME}/{gcs_object_path} ({blob.size:,} bytes)")
            return result
        except Exception as exc:
            result["error"] = str(exc)
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                logger.warning(f"  Retry {attempt}/{MAX_RETRIES} for {local_path.name} in {wait}s: {exc}")
                time.sleep(wait)
            else:
                result["status"] = "failed"
                logger.error(f"  FAILED {local_path.name} after {MAX_RETRIES} attempts: {exc}")
    return result


def upload_all() -> List[Dict]:
    csv_files = sorted(RAW_DIR.glob("transactions_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {RAW_DIR}. Run generate_data.py first.")
    logger.info(f"Found {len(csv_files)} CSV files to upload")
    client = get_gcs_client()
    ensure_bucket(client, GCS_BUCKET_NAME)
    tasks = [(p, gcs_path_for(p.name)) for p in csv_files if gcs_path_for(p.name)]
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_file, client, lp, gp): (lp, gp) for lp, gp in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main() -> None:
    logger.info("=" * 60)
    logger.info("SMB Intelligence -- Part 1: GCS Upload")
    logger.info(f"Destination: gs://{GCS_BUCKET_NAME}/raw/transactions/")
    logger.info("=" * 60)
    if not GCP_PROJECT_ID:
        logger.error("GCP_PROJECT_ID not set. Cannot upload.")
        sys.exit(1)
    try:
        results = upload_all()
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bucket": GCS_BUCKET_NAME,
            "total_files": len(results),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "files": results,
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest saved -> {MANIFEST_PATH}")
        fail_count = manifest["failed"]
        if fail_count > 0:
            logger.error(f"{fail_count} uploads failed. Check manifest.")
            sys.exit(1)
        logger.info(f"All {manifest['successful']} files uploaded successfully")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Upload pipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
