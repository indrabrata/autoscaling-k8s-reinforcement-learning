from dotenv import load_dotenv

load_dotenv()

import os
import time

import cv2
import psutil  # <--- new

from .inference import InferenceEngine
from .streamer import Streamer
from .utils import draw_predictions, register_signal_handlers


def draw_system_usage(frame):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent

    text = f"CPU: {cpu:.1f}%  MEM: {mem:.1f}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def main():
    # --- Config ---
    rtsp_host = os.environ.get("RTSP_HOST", "mediamtx")
    rtsp_port = int(os.environ.get("RTSP_PORT", "8554"))
    rtsp_path = os.environ.get("RTSP_PATH", "mystream")
    rtsp_url = os.environ.get("RTSP_URL",
                               f"rtsp://{rtsp_host}:{rtsp_port}/{rtsp_path}")

    input_source = os.environ.get("INPUT_SOURCE", "0")
    model_path = os.environ.get("YOLO_MODEL", "yolo11n.pt")
    device = os.environ.get("YOLO_DEVICE", "cpu")
    fps = int(os.environ.get("FPS", "15"))
    show_preview = os.environ.get("SHOW_PREVIEW", "0") in ("1", "true", "True")

    # --- Init inference ---
    engine = InferenceEngine(model_path, device)

    # --- Capture source ---
    try:
        src = int(input_source)
    except ValueError:
        src = input_source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source {input_source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # --- Init streamer ---
    streamer = Streamer(width, height, fps, rtsp_url)

    def cleanup():
        if cap and cap.isOpened():
            cap.release()
        streamer.close()
        if show_preview:
            cv2.destroyAllWindows()

    register_signal_handlers(cleanup)

    # --- Main loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preds = engine.predict(frame)
        annotated = draw_predictions(frame.copy(), preds, engine.model)

        # Overlay CPU + memory usage
        annotated = draw_system_usage(annotated)

        if show_preview:
            cv2.imshow("YOLO Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        streamer.write(annotated)

        time.sleep(1.0 / fps)

    cleanup()


if __name__ == "__main__":
    main()
