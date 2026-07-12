from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "public" / "data"
DEFAULT_STATIC_DIR = PROJECT_ROOT / "public"


def _run_aws(args: list[str], dry_run: bool) -> None:
    command = ["aws", *args]
    if dry_run:
        print("DRY RUN:", json.dumps(command, ensure_ascii=True))
        return
    subprocess.run(command, check=True)


def _load_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "latest_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}. Run scripts/export_static_dashboard_data.py first.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def publish(data_dir: Path, static_dir: Path, bucket: str, prefix: str, dry_run: bool) -> None:
    manifest = _load_manifest(data_dir)
    run_id = manifest["run_id"]
    prefix = prefix.strip("/")
    prod_prefix = f"{prefix}/prod" if prefix else "prod"
    static_prefix = f"{prefix}/static" if prefix else "static"
    run_dir = data_dir / "runs" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir referenced by manifest: {run_dir}")

    run_s3 = f"s3://{bucket}/{prod_prefix}/runs/{run_id}/"
    latest_manifest_s3 = f"s3://{bucket}/{prod_prefix}/latest_manifest.json"
    static_s3 = f"s3://{bucket}/{static_prefix}/"

    _run_aws(
        [
            "s3",
            "sync",
            str(run_dir),
            run_s3,
            "--cache-control",
            "public, max-age=31536000, immutable",
            "--delete",
        ],
        dry_run,
    )
    _run_aws(
        [
            "s3",
            "cp",
            str(data_dir / "latest_manifest.json"),
            latest_manifest_s3,
            "--cache-control",
            "no-cache, max-age=0, must-revalidate",
            "--content-type",
            "application/json",
        ],
        dry_run,
    )
    _run_aws(
        [
            "s3",
            "sync",
            str(static_dir),
            static_s3,
            "--exclude",
            "data/*",
            "--cache-control",
            "no-cache, max-age=0, must-revalidate",
        ],
        dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomically publish static dashboard artifacts to S3 with AWS CLI.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    publish(args.data_dir, args.static_dir, args.bucket, args.prefix, args.dry_run)


if __name__ == "__main__":
    main()
