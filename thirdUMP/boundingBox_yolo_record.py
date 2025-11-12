import argparse
import multiprocessing
import queue
import sys
import threading
from functools import lru_cache
import datetime
import logging
import os

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)
# Video recording imports
import cv2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('imx500_detections.log'),   
    ]
)

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = coords

def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects."""
    threshold = args.threshold
    
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return None
    
    # Your model has 4 outputs: boxes, scores, classes, valid_detections
    boxes = np_outputs[0][0]  # Shape: (300, 4)
    scores = np_outputs[1][0]  # Shape: (300,)
    classes = np_outputs[2][0]  # Shape: (300,)
    
    logging.info(f"Max score: {np.max(scores):.4f}")
    
    # Get the image dimensions from metadata
    image_width = metadata.get('FrameWidth', 1920)  # default to 1920 if not found
    image_height = metadata.get('FrameHeight', 1080)  # default to 1080 if not found
    
    detections = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            # Get normalized coordinates from model output (assumed to be normalized)
            box = boxes[i]
            x1, y1, x2, y2 = box
            
            # Scale coordinates to actual image dimensions
            x = int(x1 * image_width)
            y = int(y1 * image_height)
            w = int((x2 - x1) * image_width)
            h = int((y2 - y1) * image_height)
            
            detection = Detection((x, y, w, h), classes[i], scores[i], metadata)
            detections.append(detection)
            
            logging.info(f"Detection: class={int(classes[i])}, conf={scores[i]:.4f}, box=({x},{y},{w},{h})")
    
    return detections

@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def draw_detections(jobs):
    """Draw the detections for this request onto the ISP output and ensure they are recorded."""
    labels = get_labels()
    # Wait for result from child processes in the order submitted.
    last_detections = []
    while (job := jobs.get()) is not None:
        request, async_result = job
        detections = async_result.get()
        if detections is None:
            detections = last_detections
        last_detections = detections
        # Lock the buffer while drawing to ensure recording gets the annotations
        with MappedArray(request, 'main') as m:
            for detection in detections:
                x, y, w, h = detection.box
                label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = x + 5
                text_y = y + 15

                # Create a copy of the array to draw the background with opacity
                overlay = m.array.copy()

                # Draw the background rectangle on the overlay
                cv2.rectangle(overlay,
                              (text_x, text_y - text_height),
                              (text_x + text_width, text_y + baseline),
                              (255, 255, 255),  # Background color (white)
                              cv2.FILLED)

                alpha = 0.3
                cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

                # Draw text on top of the background
                cv2.putText(m.array, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw detection box
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            if intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
                color = (255, 0, 0)  # red
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))

            # Write the frame with annotations to video file
            out.write(m.array)
            
            # Display the frame
            cv2.imshow('IMX500 Object Detection', m.array)
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to quit
                global running
                running = False
        request.release()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="imx500_network_yolov8n_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second", default=30)
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    running = True

    # Create output directory for recordings if it doesn't exist
    output_dir = "recordings"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"detection_{timestamp}.mp4")

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    # Set up camera configuration
    main = {'format': 'RGB888', 'size': (1920, 1080)}  # Full HD recording
    config = picam2.create_preview_configuration(main, controls={"FrameRate": args.fps}, buffer_count=12)
    
    # Set up OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, float(args.fps), (1920, 1080))

    imx500.show_network_fw_progress_bar()
    picam2.start(config)
    logging.info(f"Started recording to {output_file}")
    
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    pool = multiprocessing.Pool(processes=4)
    jobs = queue.Queue()

    thread = threading.Thread(target=draw_detections, args=(jobs,))
    thread.start()

    try:
        while running:
            # The request gets released by handle_results
            request = picam2.capture_request()
            metadata = request.get_metadata()
            if metadata:
                async_result = pool.apply_async(parse_detections, (metadata,))
                jobs.put((request, async_result))
            else:
                request.release()
    except KeyboardInterrupt:
        running = False
    finally:
        # Clean up
        picam2.stop()
        out.release()
        logging.info(f"Recording saved to {output_file}")
        cv2.destroyAllWindows()
        pool.terminate()
        pool.join()
        jobs.put(None)  # Signal the display thread to exit
        thread.join()