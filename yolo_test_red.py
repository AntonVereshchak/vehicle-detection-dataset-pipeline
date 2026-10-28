from ultralytics import YOLO
import cv2
from pathlib import Path

model = YOLO("yolo11m.pt")

output_dir = Path("runs/detect/test_red")
output_dir.mkdir(parents=True, exist_ok=True)

results = model.predict(
    source="dataset/yolo_test",
    device=0,
    imgsz=1920,
    conf=0.10,
    classes=[2, 3, 5, 7],
    max_det=200
)

for result in results:
    image = result.orig_img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Червоний колір у форматі BGR
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            1
        )

    output_path = output_dir / Path(result.path).name
    cv2.imwrite(str(output_path), image)

print("Done! Results saved to:", output_dir)