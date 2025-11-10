from ultralytics import YOLO


class InferenceEngine:
    def __init__(self, model_path="yolo11n.pt", device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, frame):
        return self.model.predict(frame, device=self.device,
                                  conf=0.25, imgsz=640, verbose=False)
