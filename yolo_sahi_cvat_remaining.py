"""Generate CVAT YOLO 1.1 draft annotations for selected frames, excluding 10 reviewed A frames.

Put this file in the project root beside yolo11m.pt and run:
    python yolo_sahi_cvat_remaining.py

Outputs separate CVAT-importable archives for A_remaining, B, C and D.
Predictions require manual review; do not treat as ground truth.
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
SELECTED = ROOT / "dataset" / "selected"
REVIEWED = ROOT / "dataset" / "yolo_test"
MODEL_PATH = ROOT / "yolo11m.pt"
OUTPUT = ROOT / "dataset" / "cvat_remaining"
VEHICLE_IDS = {2, 3, 5, 7}
CONFIDENCE = 0.10
SLICE_SIZE = 640
OVERLAP_RATIO = 0.20
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def process_group(name, images, model):
    group = OUTPUT / name
    annotation_dir = group / "obj_train_data"
    preview_dir = group / "previews"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for i, path in enumerate(images, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        height, width = frame.shape[:2]
        started = time.perf_counter()
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
            x1, y1, x2, y2 = map(float, obj.bbox.to_xyxy())
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            xc = (x1 + x2) / (2 * width)
            yc = (y1 + y2) / (2 * height)
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 1)

        (annotation_dir / f"{path.stem}.txt").write_text(
            "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
        )
        # CVAT YOLO 1.1 import expects images and matching labels in the archive.
        shutil.copy2(path, annotation_dir / path.name)
        if not cv2.imwrite(str(preview_dir / path.name), frame):
            raise RuntimeError(f"Cannot write preview: {path.name}")
        elapsed = round(time.perf_counter() - started, 2)
        summary.append((path.name, len(labels), elapsed))
        print(f"[{name} {i}/{len(images)}] {path.name}: {len(labels)} boxes, {elapsed}s", flush=True)

    (group / "obj.names").write_text("Vehicle\n", encoding="utf-8")
    (group / "obj.data").write_text(
        "classes = 1\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n", encoding="utf-8"
    )
    (group / "train.txt").write_text(
        "\n".join(f"data/obj_train_data/{p.name}" for p in images) + "\n", encoding="utf-8"
    )
    with (group / "annotation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "vehicle_detections", "seconds"])
        writer.writerows(summary)

    archive = group / f"cvat_{name}_yolo_1_1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename in ("obj.names", "obj.data", "train.txt"):
            zf.write(group / filename, filename)
        for path in sorted(annotation_dir.iterdir()):
            zf.write(path, f"obj_train_data/{path.name}")
    print(f"Created: {archive}\n", flush=True)


def main():
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model missing: {MODEL_PATH}")
    if not REVIEWED.is_dir():
        raise SystemExit(f"Reviewed 10-frame folder missing: {REVIEWED}")
    reviewed_names = {p.name for p in images_in(REVIEWED)}
    if len(reviewed_names) != 10:
        raise SystemExit(f"Expected 10 reviewed images, found {len(reviewed_names)}")

    groups = {}
    for letter in "ABCD":
        folder = SELECTED / f"train_{letter}"
        if not folder.is_dir():
            raise SystemExit(f"Selected frames folder missing: {folder}")
        images = images_in(folder)
        if letter == "A":
            missing = reviewed_names - {p.name for p in images}
            if missing:
                raise SystemExit(f"Reviewed A images not found in train_A: {sorted(missing)}")
            images = [p for p in images if p.name not in reviewed_names]
        groups["A_remaining" if letter == "A" else letter] = images

    counts = {name: len(images) for name, images in groups.items()}
    print(f"Frames by group: {counts}; total={sum(counts.values())}")
    if counts != {"A_remaining": 60, "B": 20, "C": 44, "D": 44}:
        raise SystemExit("Unexpected frame counts. Verify dataset/selected and dataset/yolo_test before proceeding.")
    all_names = [p.name for images in groups.values() for p in images]
    if len(all_names) != len(set(all_names)):
        raise SystemExit("Duplicate filenames across groups; rename or reorganize before proceeding.")

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(MODEL_PATH),
        confidence_threshold=CONFIDENCE, device="cuda:0"
    )
    for name, images in groups.items():
        process_group(name, images, model)
    print("DONE. Import each ZIP into a matching CVAT task containing exactly those images.")
    print("These are unverified predictions; manually review every image.")


if __name__ == "__main__":
    main()
