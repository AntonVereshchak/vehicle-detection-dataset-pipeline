from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    # Original pretrained model — NOT fine-tuned
    model = YOLO("yolo11m.pt")

    results = model.val(
        data="dataset/vehicle_split_104/data.yaml",
        split="val",
        imgsz=1280,
        batch=4,
        device=0,
        workers=0,
        conf=0.001,
        iou=0.7,
        plots=True,
        project="baseline_validation",
        name="yolo11m_original",
        exist_ok=True,
    )

    print("\n===== BASELINE RESULTS =====")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")
    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")


if __name__ == "__main__":
    freeze_support()
    main()