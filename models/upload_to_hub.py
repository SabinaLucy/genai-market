"""
upload_to_hub.py
================
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_file, list_repo_files

load_dotenv()

# Configuration 

HF_USERNAME   = "SabinaLucy"
REPO_NAME     = "volarix-models"
REPO_ID       = f"{HF_USERNAME}/{REPO_NAME}"
REPO_TYPE     = "model"


# Keys = local relative path from project root.
# Values = destination filename on HF Hub.
MODEL_FILES = {
    "models/lstm_model.pt"            : "lstm_model.pt",
    "models/lstm_config.json"         : "lstm_config.json",
    "models/regime_classifier.pkl"    : "regime_classifier.pkl",
    "models/conformal_intervals.pkl"  : "conformal_intervals.pkl",
    "models/explainability_cache.pkl" : "explainability_cache.pkl",
    "models/scaler.pkl"               : "scaler.pkl",
    "models/hmm_regime.pkl"           : "hmm_regime.pkl",
    "models/scaler_features.json"     : "scaler_features.json",
}

#  Helpers 

def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def verify_local_files(project_root: Path) -> bool:
    """Check every required file exists and print a size table."""
    print("\n Local file check ")
    all_ok = True
    for local_rel, _ in MODEL_FILES.items():
        full = project_root / local_rel
        if full.exists():
            size = sizeof_fmt(full.stat().st_size)
            print(f"  ✓  {local_rel:<45}  {size:>10}")
        else:
            print(f"  ✗  {local_rel:<45}  MISSING")
            all_ok = False
    print()
    return all_ok


def get_existing_hub_files(api: HfApi) -> set:
    """Return the set of filenames currently in the Hub repo."""
    try:
        return {f.rfilename for f in api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)}
    except Exception:
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload Volarix models to HF Hub")
    parser.add_argument(
        "--tag",
        default="",
        help="Version tag (e.g. v2). Files go to <tag>/<filename> on Hub. "
             "Default: root of repo (no subfolder).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload files even if they already exist on Hub.",
    )
    args = parser.parse_args()

    # ── Auth ─────────────────────────────────────────────────────────────────
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in .env — cannot authenticate.")
        sys.exit(1)

    api = HfApi(token=token)

    # ── Project root (two levels up from models/upload_to_hub.py) ────────────
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root : {project_root}")
    print(f"Hub repo     : {REPO_ID}")
    print(f"Tag          : {args.tag or '(none — root)'}")
    print(f"Dry run      : {args.dry_run}")

    # ── Verify local files ────────────────────────────────────────────────────
    if not verify_local_files(project_root):
        print("ERROR: one or more model files are missing. Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN — no files will be uploaded.\n")
        for local_rel, hub_name in MODEL_FILES.items():
            dest = f"{args.tag}/{hub_name}" if args.tag else hub_name
            print(f"  would upload  {local_rel}  →  {dest}")
        return

    #  Create repo if it doesn't exist 
    try:
        create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True, token=token)
        print(f"\nRepo ready: https://huggingface.co/{REPO_ID}")
    except Exception as exc:
        print(f"ERROR creating repo: {exc}")
        sys.exit(1)

    #  Existing files on Hub (skip if already present, unless --force) 
    existing = get_existing_hub_files(api) if not args.force else set()

    #  Upload 
    print("\n Uploading ")
    success, skipped, failed = [], [], []

    for local_rel, hub_name in MODEL_FILES.items():
        dest_path = f"{args.tag}/{hub_name}" if args.tag else hub_name
        local_path = project_root / local_rel

        if dest_path in existing:
            print(f"  SKIP  {dest_path}  (already on Hub — use --force to overwrite)")
            skipped.append(dest_path)
            continue

        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=dest_path,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
            )
            elapsed = time.time() - t0
            size = sizeof_fmt(local_path.stat().st_size)
            print(f"  ✓  {dest_path:<50}  {size:>10}  ({elapsed:.1f}s)")
            success.append(dest_path)
        except Exception as exc:
            print(f"  ✗  {dest_path}  FAILED: {exc}")
            failed.append(dest_path)

    #  Summary 
    print("\n Summary ")
    print(f"  Uploaded : {len(success)}")
    print(f"  Skipped  : {len(skipped)}")
    print(f"  Failed   : {len(failed)}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)

    tag_path = f"/{args.tag}" if args.tag else ""
    print(f"\nDone. View at: https://huggingface.co/{REPO_ID}{tag_path}")
    print("\nNext step: set VOLARIX_MODEL_TAG in your HF Space secrets")
    print(f"  if using a tag: VOLARIX_MODEL_TAG={args.tag}")
    print(f"  if no tag:      leave VOLARIX_MODEL_TAG unset (or empty)")


if __name__ == "__main__":
    main()
