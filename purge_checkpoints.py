#!/usr/bin/env python3
"""
Purge checkpoints and image dumps for modSTOVE pretraining.

Examples:
    # Wipe everything (with confirmation prompt)
    python purge_checkpoints.py

    # Wipe only the 'v1' run
    python purge_checkpoints.py --run-name v1

    # Skip the confirmation prompt (for scripting)
    python purge_checkpoints.py --yes
"""

import argparse
import glob
import os
import shutil
import sys
from typing import List


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(REPO_DIR, "checkpoints")
IMAGES_DIR = os.path.join(REPO_DIR, "images")


def _list_checkpoints(run_name: str = "") -> List[str]:
    """List checkpoint files, optionally filtered to a single run."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    if run_name:
        pattern = os.path.join(CHECKPOINT_DIR, f"modstove_{run_name}_*.ckpt")
    else:
        pattern = os.path.join(CHECKPOINT_DIR, "*.ckpt")
    return sorted(glob.glob(pattern))


def _list_image_dirs(run_name: str = "") -> List[str]:
    """List per-checkpoint image dump directories, optionally filtered."""
    if not os.path.isdir(IMAGES_DIR):
        return []
    if run_name:
        pattern = os.path.join(IMAGES_DIR, f"modstove_{run_name}_*")
    else:
        pattern = os.path.join(IMAGES_DIR, "*")
    return sorted(p for p in glob.glob(pattern) if os.path.isdir(p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Only purge files for this run-name (e.g. 'v1'). "
             "If omitted, purges everything under checkpoints/ and images/.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    args = parser.parse_args()

    ckpts = _list_checkpoints(args.run_name)
    img_dirs = _list_image_dirs(args.run_name)

    if not ckpts and not img_dirs:
        scope = f"run '{args.run_name}'" if args.run_name else "checkpoints/ or images/"
        print(f"Nothing to purge for {scope}.")
        return 0

    scope_label = f"run '{args.run_name}'" if args.run_name else "ALL runs"
    print(f"About to delete ({scope_label}):")
    print(f"  {len(ckpts)} checkpoint file(s) under {CHECKPOINT_DIR}")
    print(f"  {len(img_dirs)} image director(y/ies) under {IMAGES_DIR}")

    if not args.yes:
        try:
            resp = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("Aborted.")
            return 1

    for path in ckpts:
        try:
            os.remove(path)
        except OSError as e:
            print(f"  warning: could not remove {path}: {e}", file=sys.stderr)

    for path in img_dirs:
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"  warning: could not remove {path}: {e}", file=sys.stderr)

    print(f"Deleted {len(ckpts)} checkpoint(s) and {len(img_dirs)} image director(y/ies).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
