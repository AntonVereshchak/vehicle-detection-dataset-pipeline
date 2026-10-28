"""Create CVAT-compatible YOLO 1.1 draft annotations for 10 test frames.

Place next to yolo11m.pt in the project root and run:
    python yolo_sahi_cvat_10.py

CVAT: Task > Actions > Upload annotations > YOLO 1.1 > choose ZIP.
Predictions are NOT verified ground truth.
"""
from pathlib import Path
import csv
import shutil
import time
import zipfile

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset" / "yolo_test"
MODEL_PATH = ROOT / "yolo11m.pt"
OUTPUT = ROOT / "dataset" / "cvat_test_10"
ANNOTATION_DIR = OUTPUT / "obj_train_data"
PREVIEW_DIR = OUTPUT / "previews"
ZIP_PATH = OUTPUT / "cvat_yolo_1_1_annotations.zip"

VEHICLE_IDS = {2, 3, 5, 7}  # COCO car, motorcycle, bus, truck -> vehicle (0)
CONFIDENCE = 0.10
SLICE_SIZE = 640
OVERLAP_RATIO = 0.20


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f"Missing image folder: {SOURCE}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Missing model: {MODEL_PATH}")
    images = sorted(p for p in SOURCE.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != 10:
        raise SystemExit(f"Expected exactly 10 test images, found {len(images)} in {SOURCE}")
    if len({p.stem for p in images}) != len(images):
        raise SystemExit("Image basenames must be unique.")

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(MODEL_PATH),
        confidence_threshold=CONFIDENCE,
        device="cuda:0",
    )
    summary = []
    print("Generating YOLO + SAHI draft annotations for 10 images...")
    for i, path in enumerate(images, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        h, w = frame.shape[:2]
        start = time.perf_counter()
        result = get_sliced_prediction(
            str(path), model,
            slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            perform_standard_pred=False,
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        labels = []
        for obj in result.object_prediction_list:
            if int(obj.category.id) not in VEHICLE_IDS:
                continue
            x1, y1, x2, y2 = (float(v) for v in obj.bbox.to_xyxy())
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 1)
        (ANNOTATION_DIR / f"{path.stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        shutil.copy2(path, ANNOTATION_DIR / path.name)
        if not cv2.imwrite(str(PREVIEW_DIR / path.name), frame):
            raise RuntimeError(f"Cannot save preview for {path.name}")
        seconds = time.perf_counter() - start
        summary.append((path.name, len(labels), round(seconds, 2)))
        print(f"[{i}/10] {path.name}: {len(labels)} boxes, {seconds:.1f}s")

    # Standard Darknet YOLO 1.1 annotation archive layout supported by CVAT.
    (OUTPUT / "obj.names").write_text("Vehicle\n", encoding="utf-8")
    (OUTPUT / "obj.data").write_text("classes = 1\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n", encoding="utf-8")
    (OUTPUT / "train.txt").write_text(
        "\n".join(f"data/obj_train_data/{p.name}" for p in images) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "annotation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "vehicle_detections", "seconds"])
        writer.writerows(summary)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("obj.names", "obj.data", "train.txt"):
            zf.write(OUTPUT / name, name)
        for path in sorted(ANNOTATION_DIR.iterdir()):
            zf.write(path, f"obj_train_data/{path.name}")
    print(f"\nCVAT ZIP: {ZIP_PATH}")
    print("IMPORTANT: These are unverified model predictions. Review every frame in CVAT.")


if __name__ == "__main__":
    main()
