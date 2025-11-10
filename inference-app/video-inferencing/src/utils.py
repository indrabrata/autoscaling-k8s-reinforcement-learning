import cv2
import signal
import sys


def draw_predictions(frame, preds, model):
    """Draw YOLO predictions on a frame."""
    if preds is None:
        return frame

    try:
        result = preds[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return frame
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            cls = int(box.cls[0]) if hasattr(box, "cls") else 0
            label = model.names[cls] if cls in model.names else str(cls)

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, txt, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    except Exception:
        pass

    return frame


def register_signal_handlers(cleanup_fn):
    """Graceful shutdown for SIGINT/SIGTERM."""

    def handler(sig, frame):
        print("Shutting down...")
        cleanup_fn()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
