from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    model = YOLO("yolo11m.pt")

    model.train(
        data="dataset/vehicle_split_104/data.yaml",
        epochs=30,
        imgsz=1280,
        batch=4,
        device=0,
        workers=0,
        patience=10,
        project="runs/vehicle_training",
        name="yolo11m_verified104",
        exist_ok=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()