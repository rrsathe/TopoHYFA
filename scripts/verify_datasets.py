#!/usr/bin/env python
"""
Dataset Integrity Verification Tool for TopoHYFA.
Computes and verifies MD5/SHA256 checksums and file sizes of the GTEx v8 datasets.
"""

import hashlib
import os
import sys

DATASETS = {
    "GTEX_data.csv.zip": {
        "url": "https://figshare.com/articles/dataset/Processed_GTEx_v8_data/22650763",
        "download_url": "https://figshare.com/ndownloader/files/40208074",
        "size": 431765777,
        "md5": "a50db13daf93498136fae21d1302c000",
        "sha256": None,
    },
    "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt": {
        "url": "https://gtexportal.org/home/datasets",
        "download_url": "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt",
        "size": 20271,
        "md5": "90297fc31512902f4459c757180fe575",
        "sha256": "821bdaff39e7a9a1d166919b3c786724c2b79c2861aeb936a2537a0f59b066f7",
    },
}


def get_hash(filepath, hash_type="md5"):
    hasher = hashlib.md5() if hash_type == "md5" else hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    print("=========================================")
    print("   TopoHYFA Dataset Integrity Auditor    ")
    print("=========================================\n")

    failed = False

    for filename, meta in DATASETS.items():
        filepath = os.path.join(data_dir, filename)
        print(f"Checking dataset: {filename}...")

        if not os.path.exists(filepath):
            print(f"  [MISSING] File not found at {filepath}")
            print(f"  Download from: {meta['download_url']}")
            print(f"  Expected Size: {meta['size']} bytes")
            if meta["md5"]:
                print(f"  Expected MD5:  {meta['md5']}")
            if meta["sha256"]:
                print(f"  Expected SHA256: {meta['sha256']}")
            print()
            failed = True
            continue

        actual_size = os.path.getsize(filepath)
        if actual_size != meta["size"]:
            print(f"  [FAIL] Size mismatch: expected {meta['size']} bytes, got {actual_size} bytes")
            failed = True
        else:
            print(f"  [OK] File size verified ({actual_size} bytes)")

        if meta["md5"]:
            actual_md5 = get_hash(filepath, "md5")
            if actual_md5 != meta["md5"]:
                print(f"  [FAIL] MD5 mismatch: expected {meta['md5']}, got {actual_md5}")
                failed = True
            else:
                print(f"  [OK] MD5 checksum verified ({actual_md5})")

        if meta["sha256"]:
            actual_sha = get_hash(filepath, "sha256")
            if actual_sha != meta["sha256"]:
                print(f"  [FAIL] SHA256 mismatch: expected {meta['sha256']}, got {actual_sha}")
                failed = True
            else:
                print(f"  [OK] SHA256 checksum verified ({actual_sha})")
        print()

    if failed:
        print(
            "[FAIL] Dataset integrity checks FAILED! Please download/fix the missing or corrupt files."
        )
        sys.exit(1)
    else:
        print("[OK] Dataset integrity successfully verified. All datasets are correct.")
        sys.exit(0)


if __name__ == "__main__":
    main()
