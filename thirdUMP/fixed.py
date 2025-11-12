import argparse
import logging
import multiprocessing
import queue
import sys
import threading
from functools import lru_cache

import cv2
import numpy as np
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("imx500_detections.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ---------------------- Detection Class ----------------------
class Detection:
    def __init__(self, coords, category, conf):
        self.category = category
        self.conf = conf
        self.box = coords


# ---------------------- Parse Detections ----------------------
def parse_detections(metadata):
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return []

    boxes = np_outputs[0][0]
    scores = np_outputs[1][0]
    classes = np_outputs[2][0]

    detections = []
    img_width, img_height = 640, 480  # preview size

    for i in range(len(scores)):
        conf = float(scores[i])
        if conf < args.threshold:
            continue

        x1, y1, x2, y2 = boxes[i]

        # Detect if normalized (0-1)
        if x2 <= 1.5 and y2 <= 1.5:
            x1 *= img_width
            x2 *= img_width
            y1 *= img_height
            y2 *= img_height

        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        if w <= 0 or h <= 0:
            continue

        detections.append(Detection((x, y, w, h), int(classes[i]), conf))
        logging.info(f"Detection: class={int(classes[i])}, conf={conf:.2f}, box=({x},{y},{w},{h})")

    return detections


# ---------------------- Labels ----------------------
@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


# ---------------------- Drawing ----------------------
def draw_detections(jobs):
    labels = get_labels()
    last_detections = []

    while (job := jobs.get()) is not None:
        request, async_result = job
        detections = async_result.get() or last_detections
        last_detections = detections

        with MappedArray(request, "main") as m:
            frame = m.array

            for detection in detections:
                x, y, w, h = detection.box
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
# --- Adjust bounding box scaling to match display frame ---
                frame_h, frame_w = frame.shape[:2]
                model_w, model_h = 640, 640  # YOLO model input size (adjust if different)

# Scale coordinates correctly
                x = int(x * frame_w / model_w)
                y = int(y * frame_h / model_h)
                w = int(w * frame_w / model_w)
                h = int(h * frame_h / model_h)

# Ensure bounding box stays inside frame
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))

# Draw corrected bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("IMX500 YOLO Live", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                logging.info("Exiting on user request...")
                cv2.destroyAllWindows()
                os._exit(0)

        request.release()


# ---------------------- CLI Arguments ----------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .rpk model")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.txt file")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection threshold")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for preview")
    return parser.parse_args()


# ---------------------- Main ----------------------
if __name__ == "__main__":
    args = get_args()

    # Initialize IMX500 + Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"

    with open(args.labels, "r") as f:
        intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    main_config = {"format": "RGB888", "size": (640, 480)}
    config = picam2.create_preview_configuration(
        main=main_config,
        controls={"FrameRate": args.fps},
        buffer_count=8
    )

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)
    logging.info(f"Camera started at {args.fps} FPS")

    pool = multiprocessing.Pool(processes=2)
    jobs = queue.Queue()

    threading.Thread(target=draw_detections, args=(jobs,), daemon=True).start()

    while True:
        request = picam2.capture_request()
        metadata = request.get_metadata()

        if metadata:
            async_result = pool.apply_async(parse_detections, (metadata,))
            jobs.put((request, async_result))
        else:
            request.release()
