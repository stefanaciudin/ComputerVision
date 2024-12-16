from ultralytics import YOLO

def main():
    model = YOLO("yolov5su.pt")

    model.train(
        data="C:/Users/stefa/ComputerVision/lab6/config.yaml",
        epochs=20,
        imgsz=640
    )

if __name__ == "__main__":
    main()
