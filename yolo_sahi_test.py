"""Compare YOLO11m + SAHI sliced inference on the 10 curated test frames.

Run from the project folder: python yolo_sahi_test.py
Dependencies: pip install ultralytics sahi opencv-python
"""
from pathlib import Path
import csv
import time

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset" / "yolo_test"
OUTPUT = ROOT / "runs" / "detect" / "test_sahi_640"
MODEL_PATH = ROOT / "yolo11m.pt"

# COCO IDs: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_IDS = {2, 3, 5, 7}
CONFIDENCE = 0.10
SLICE_SIZE = 640
OVERLAP_RATIO = 0.20


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f"Input folder not found: {SOURCE}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    files = sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"No images found in: {SOURCE}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(MODEL_PATH),
        confidence_threshold=CONFIDENCE,
        device="cuda:0",
    )
    summary = []
    print(f"Processing {len(files)} images using SAHI on CUDA GPU...")

    for index, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARNING] Cannot read: {path.name}")
            continue

        start = time.perf_counter()
        prediction = get_sliced_prediction(
            str(path),
            model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            perform_standard_pred=False,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        elapsed = time.perf_counter() - start
        count = 0
        for obj in prediction.object_prediction_list:
            if int(obj.category.id) not in VEHICLE_IDS:
                continue
            x1, y1, x2, y2 = (round(v) for v in obj.bbox.to_xyxy())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            count += 1

        destination = OUTPUT / path.name
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"Failed to save {destination}")
        summary.append((path.name, count, round(elapsed, 3)))
        print(f"[{index}/{len(files)}] {path.name}: {count} vehicles, {elapsed:.2f} sec")

    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "vehicle_detections", "seconds"])
        writer.writerows(summary)
    print(f"\nDone. Results: {OUTPUT}")
    print("Note: detections are predictions, not verified ground-truth labels.")


if __name__ == "__main__":
    main()
