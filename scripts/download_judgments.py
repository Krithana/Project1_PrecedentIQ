"""
Download Supreme Court & High Court judgments from the AWS Open Data Registry.

These buckets are PUBLIC — no AWS account, no credentials, no billing setup needed.
We use botocore's UNSIGNED config, which is the Python equivalent of the CLI's
`--no-sign-request` flag.

Run:
    python scripts/download_judgments.py
"""
import os
import sys
import yaml
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(HERE, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def get_client():
    # UNSIGNED = "don't ask for AWS credentials, this bucket allows anonymous reads"
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_keys(client, bucket, prefix):
    """List every object under a prefix, handling pagination."""
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append((obj["Key"], obj["Size"]))
    return keys


def download_file(client, bucket, key, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"  [skip] already downloaded: {dest_path}")
        return
    client.download_file(bucket, key, dest_path)


def main():
    cfg = load_config()
    client = get_client()
    raw_dir = os.path.join(HERE, cfg["paths"]["raw_dir"])

    # --- Supreme Court judgments ---
    sc_bucket = cfg["aws"]["sc_bucket"]
    print(f"\n=== Supreme Court judgments ({sc_bucket}) ===")
    for year in cfg["aws"]["sc_years"]:
        prefix = f"data/tar/year={year}/english/"
        try:
            keys = list_keys(client, sc_bucket, prefix)
        except ClientError as e:
            print(f"  ERROR listing {prefix}: {e}")
            continue
        if not keys:
            print(f"  [warn] no objects found under {prefix} — check the year exists in the bucket")
            continue
        for key, size in tqdm(keys, desc=f"SC {year}"):
            dest = os.path.join(raw_dir, "supreme_court", str(year), os.path.basename(key))
            download_file(client, sc_bucket, key, dest)

    # --- High Court judgments ---
    hc_bucket = cfg["aws"]["hc_bucket"]
    print(f"\n=== High Court judgments ({hc_bucket}) ===")
    for year in cfg["aws"].get("hc_years", []):
        for court in cfg["aws"]["hc_courts"]:
            prefix = f"data/tar/year={year}/court={court}/"
            try:
                keys = list_keys(client, hc_bucket, prefix)
            except ClientError as e:
                print(f"  ERROR listing {prefix}: {e}")
                continue
            if not keys:
                print(f"  [warn] no objects found under {prefix} — check the year/court code")
                continue
            for key, size in tqdm(keys, desc=f"HC {year}/{court}"):
                relative_key = key[len(prefix):].replace("/", os.sep)
                dest = os.path.join(raw_dir, "high_court", str(year), court, relative_key)
                download_file(client, hc_bucket, key, dest)

    print("\nDone. Run `python scripts/verify_setup.py` next to confirm what landed on disk.")


if __name__ == "__main__":
    main()
