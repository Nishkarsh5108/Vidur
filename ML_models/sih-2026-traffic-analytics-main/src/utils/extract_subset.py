"""
Streaming extractor for large IDD archives with strict disk-quota safety.

Extracts:
1. 100% of all annotation / metadata files (.xml, .json, .txt, .csv) regardless of position.
2. Image files (.jpg, .jpeg, .png) up to a configurable size ceiling (default: 19.0 GB).
3. Live progress logging to avoid running out of disk space on Drive E:.
"""

import os
import sys
import time
import tarfile
import argparse
from pathlib import Path


def extract_idd_with_quota(
    archive_path: str = r"C:\Users\TUSHAR BASAK\Downloads\idd-detection.tar.gz",
    dest_dir: str = r"e:\SIH_2026\data\raw\idd",
    max_image_gb: float = 18.5
):
    archive_file = Path(archive_path)
    if not archive_file.is_file():
        raise FileNotFoundError(f"Archive not found at: {archive_path}")

    target_dir = Path(dest_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(max_image_gb * 1024 * 1024 * 1024)
    print("=" * 65)
    print("      IDD DATASET STREAMING EXTRACTOR (SAFETY QUOTA)")
    print("=" * 65)
    print(f"Source Archive:      {archive_file}")
    print(f"Destination:         {target_dir}")
    print(f"Max Image Quota:     {max_image_gb:.2f} GB ({max_bytes:,} bytes)")
    print("=" * 65)

    extracted_img_bytes = 0
    extracted_img_count = 0
    extracted_annot_count = 0
    skipped_img_count = 0

    t_start = time.time()
    last_log_time = t_start

    print("[*] Opening compressed archive stream...")
    with tarfile.open(archive_file, "r:gz") as tar:
        for member in tar:
            name_lower = member.name.lower()

            # Always extract directories
            if member.isdir():
                target_dir.joinpath(member.name).mkdir(parents=True, exist_ok=True)
                continue

            # Prioritize annotations and metadata (never skip)
            is_annot = any(name_lower.endswith(ext) for ext in [".xml", ".json", ".txt", ".csv"])
            is_image = any(name_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png"])

            if is_annot:
                tar.extract(member, path=target_dir)
                extracted_annot_count += 1
            elif is_image:
                if (extracted_img_bytes + member.size) <= max_bytes:
                    tar.extract(member, path=target_dir)
                    extracted_img_bytes += member.size
                    extracted_img_count += 1
                else:
                    skipped_img_count += 1
            else:
                # Other auxiliary files
                tar.extract(member, path=target_dir)

            # Progress logging every 5 seconds
            now = time.time()
            if now - last_log_time >= 5.0:
                cur_gb = extracted_img_bytes / (1024 ** 3)
                pct = (extracted_img_bytes / max_bytes) * 100.0
                rate_mb = (extracted_img_bytes / (1024 ** 2)) / max(1.0, now - t_start)
                print(
                    f"[*] Extracted: {cur_gb:.2f}/{max_image_gb:.1f} GB ({pct:.1f}%) | "
                    f"Images: {extracted_img_count:,} | Annotations: {extracted_annot_count:,} | "
                    f"Rate: {rate_mb:.1f} MB/s"
                )
                last_log_time = now

    total_time = time.time() - t_start
    total_gb = extracted_img_bytes / (1024 ** 3)
    print("\n" + "=" * 65)
    print("                 EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Total Extracted Images:      {extracted_img_count:,}")
    print(f"Total Extracted Annotations: {extracted_annot_count:,}")
    print(f"Total Image Volume:          {total_gb:.2f} GB")
    print(f"Images Skipped (Quota hit):  {skipped_img_count:,}")
    print(f"Total Elapsed Time:          {total_time / 60.0:.2f} minutes")
    print(f"Output Location:             {target_dir}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract IDD archive up to a safe disk quota")
    parser.add_argument("--archive", type=str, default=r"C:\Users\TUSHAR BASAK\Downloads\idd-detection.tar.gz")
    parser.add_argument("--dest", type=str, default=r"e:\SIH_2026\data\raw\idd")
    parser.add_argument("--max-gb", type=float, default=18.5, help="Maximum image volume in GB (default: 18.5)")
    args = parser.parse_args()

    extract_idd_with_quota(
        archive_path=args.archive,
        dest_dir=args.dest,
        max_image_gb=args.max_gb
    )


if __name__ == "__main__":
    main()
