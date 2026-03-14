"""
reset_finalized.py — Clear finalized status and segments for a session.

Usage:
    python scripts/reset_finalized.py <session_dir>

Example:
    python scripts/reset_finalized.py "sessions/schema-fish-morphometrics"

What it does:
  1. Removes the `finalizedDetection` key from every label JSON in <session_dir>/labels/
  2. Clears <session_dir>/finalized_images.json
  3. Deletes all files in <session_dir>/segments/

After running, restart the app and re-finalize images to regenerate clean segments.
"""

import json
import os
import shutil
import sys


def reset_session(session_dir: str) -> None:
    session_dir = os.path.abspath(session_dir)
    if not os.path.isdir(session_dir):
        print(f"ERROR: Directory not found: {session_dir}")
        sys.exit(1)

    labels_dir = os.path.join(session_dir, "labels")
    finalized_list_path = os.path.join(session_dir, "finalized_images.json")
    segments_dir = os.path.join(session_dir, "segments")

    # --- 1. Clear finalizedDetection from label JSONs ---
    cleared_labels = 0
    if os.path.isdir(labels_dir):
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(labels_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "finalizedDetection" in data:
                    del data["finalizedDetection"]
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    cleared_labels += 1
                    print(f"  cleared: {fname}")
            except Exception as e:
                print(f"  WARNING: could not process {fname}: {e}")
    else:
        print(f"  No labels/ directory found at {labels_dir}")

    # --- 2. Clear finalized_images.json ---
    if os.path.exists(finalized_list_path):
        try:
            with open(finalized_list_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            print("\n  cleared: finalized_images.json")
        except Exception as e:
            print(f"\n  WARNING: could not clear finalized_images.json: {e}")
    else:
        print("\n  No finalized_images.json found")

    # --- 3. Delete all segment files ---
    cleared_segments = 0
    if os.path.isdir(segments_dir):
        cleared_segments = len(os.listdir(segments_dir))
        shutil.rmtree(segments_dir)
        os.makedirs(segments_dir)
        print(f"\n  Deleted {cleared_segments} files from segments/")
    else:
        print(f"\n  No segments/ directory found at {segments_dir}")

    print(f"\nDone. Cleared {cleared_labels} label(s), {cleared_segments} segment file(s).")
    print("Restart the app and re-finalize images to regenerate segments.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    reset_session(sys.argv[1])
