"""Frame Extraction v2 — candidate frames for manual selection.

Requirements: python -m pip install opencv-python numpy
Run: python frame_extraction_v2.py
"""
from pathlib import Path
import csv
import os
import tempfile

import cv2
import numpy as np

VIDEO_FOLDER = Path(r"C:\Users\User\Documents\Test fourth law")
CANDIDATES_FOLDER = VIDEO_FOLDER / "dataset" / "candidates"
CSV_PATH = CANDIDATES_FOLDER / "frames_manifest.csv"

# Video ID: (subfolder, number of candidate frames)
SAMPLING_CONFIG = {
    "8968356": ("train_A", 180),
    "5382494": ("train_B", 40),
    "8457857": ("train_C", 100),
    "3405804": ("train_D", 100),
}


def main():
    if not VIDEO_FOLDER.is_dir():
        raise FileNotFoundError(f"Video folder not found: {VIDEO_FOLDER}")

    CANDIDATES_FOLDER.mkdir(parents=True, exist_ok=True)
    records = []
    matched_ids = set()
    total_saved = 0

    for video_path in sorted(VIDEO_FOLDER.glob("*")):
        if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
            continue

        video_id = video_path.name.split("_")[0].split("-")[0]
        if video_id not in SAMPLING_CONFIG:
            continue
        if video_id in matched_ids:
            print(f"WARNING: Duplicate video ID {video_id}: {video_path.name}; skipping")
            continue
        matched_ids.add(video_id)
        video_name, requested = SAMPLING_CONFIG[video_id]
        output_dir = CANDIDATES_FOLDER / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"ERROR: Cannot open {video_path.name}")
            continue

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if total_frames <= 0:
                print(f"ERROR: No frames found in {video_path.name}")
                continue
            if fps <= 0:
                print(f"WARNING: Invalid FPS for {video_path.name}; timestamps unavailable")

            indices = np.unique(np.linspace(0, total_frames - 1,
                                            num=min(requested, total_frames),
                                            dtype=int))
            saved = existing = failed = 0
            for idx in indices:
                idx = int(idx)
                output_name = f"{video_name}_frame_{idx:06d}.jpg"
                output_path = output_dir / output_name
                status = "existing"
                if output_path.exists():
                    existing += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        status = "read_failed"
                        failed += 1
                    elif not cv2.imwrite(str(output_path), frame):
                        status = "write_failed"
                        failed += 1
                    else:
                        status = "saved"
                        saved += 1
                records.append({
                    "video_id": video_id,
                    "video_file": video_path.name,
                    "video_group": video_name,
                    "frame_index": idx,
                    "timestamp_seconds": f"{idx / fps:.3f}" if fps > 0 else "",
                    "image_path": str(output_path.relative_to(VIDEO_FOLDER)),
                    "status": status,
                    "selected": "",  # Fill in manually later; this script does not make selections.
                })
            total_saved += saved
            print(f"{video_name}: {saved} new, {existing} already present, "
                  f"{failed} failed / {len(indices)} requested")
        finally:
            cap.release()

    for video_id, (name, _) in SAMPLING_CONFIG.items():
        if video_id not in matched_ids:
            print(f"WARNING: No MP4 found for {name} (ID {video_id})")

    if records:
        # Never overwrite a manifest that may contain manual selections.
        manifest_path = CSV_PATH
        if manifest_path.exists():
            n = 2
            while True:
                candidate = CANDIDATES_FOLDER / f"frames_manifest_run{n}.csv"
                if not candidate.exists():
                    manifest_path = candidate
                    break
                n += 1
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"Manifest: {manifest_path}")
    print(f"Completed. New frames saved: {total_saved}; candidate records: {len(records)}")


if __name__ == "__main__":
    main()
