import argparse
import logging
import multiprocessing
import queue
import sys
import threading
import os
import time
from functools import lru_cache

import cv2
import numpy as np

try:
    from picamera2 import Picamera2, MappedArray
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics
except Exception as e:
    print(f"ERROR importing Picamera2: {e}")
    sys.exit(1)

# Import gpiozero for servo control
SERVO_AVAILABLE = False
try:
    from gpiozero import AngularServo
    SERVO_AVAILABLE = True
    print("✓ gpiozero imported successfully")
except ImportError:
    print("✗ gpiozero not available. Install with: sudo apt-get install python3-gpiozero")

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


# ---------------------- Servo Controller (Pi 5 / gpiozero) ----------------------
class ServoController:
    def __init__(self, pan_pin=17, min_angle=-90, max_angle=90, enabled=True):
        """Initialize servo controller with gpiozero (Pi 5 compatible)"""
        self.enabled = enabled and SERVO_AVAILABLE
        self.current_angle = 0
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.pan_pin = pan_pin
        self.servo = None
        
        if not self.enabled:
            logging.warning("Servo control disabled")
            return
        
        try:
            # Initialize AngularServo with the EXACT same parameters that worked in your test
            self.servo = AngularServo(
                pan_pin, 
                min_pulse_width=0.0006,  # 0.6ms - same as your working test
                max_pulse_width=0.0023   # 2.3ms - same as your working test
            )
            
            # Initialize at center position
            self.center()
            logging.info(f"✓ Servo initialized on GPIO {pan_pin} using gpiozero")
            
        except Exception as e:
            logging.error(f"✗ Servo initialization failed: {e}")
            self.enabled = False
    
    def move_to_angle(self, angle):
        """Move servo to specific angle (-90 to 90)"""
        if not self.enabled or self.servo is None:
            return
        
        try:
            # Clamp angle to limits
            angle = max(self.min_angle, min(self.max_angle, angle))
            
            # gpiozero AngularServo uses -90 to +90 degree range natively
            self.servo.angle = angle
            self.current_angle = angle
            
        except Exception as e:
            logging.error(f"Error moving servo to {angle}°: {e}")
    
    def adjust_by_offset(self, offset, sensitivity=0.3):
        """Adjust servo by offset amount with sensitivity control"""
        if not self.enabled:
            return
        
        adjustment = offset * sensitivity
        new_angle = self.current_angle + adjustment
        self.move_to_angle(new_angle)
    
    def center(self):
        """Return servo to center position"""
        self.move_to_angle(0)
    
    def cleanup(self):
        """Clean up servo resources"""
        if self.servo is not None and self.enabled:
            try:
                self.center()
                time.sleep(0.5)
                self.servo.close()
                logging.info("Servo cleanup completed")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")


# ---------------------- Person Tracker ----------------------
class PersonTracker:
    def __init__(self, frame_width=640, frame_height=480, 
                 dead_zone=0.15, person_class_id=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_center_x = frame_width / 2
        self.dead_zone_pixels = frame_width * dead_zone
        self.person_class_id = person_class_id
        self.tracked_person = None
        logging.info(f"Person tracker initialized (dead zone: {dead_zone*100}%)")
    
    def find_person(self, detections, labels):
        """Find the most prominent person in detections"""
        persons = []
        
        for detection in detections:
            try:
                label = labels[int(detection.category)].lower()
                if 'person' in label or int(detection.category) == self.person_class_id:
                    x, y, w, h = detection.box
                    area = w * h
                    center_x = x + w / 2
                    persons.append({
                        'detection': detection,
                        'area': area,
                        'center_x': center_x,
                        'center_y': y + h / 2
                    })
            except Exception as e:
                continue
        
        if not persons:
            return None
        
        return max(persons, key=lambda p: p['area'])
    
    def calculate_offset(self, person_center_x):
        """Calculate how far the person is from center"""
        offset = person_center_x - self.frame_center_x
        
        if abs(offset) < self.dead_zone_pixels:
            return 0
        
        max_offset = self.frame_width / 2
        normalized_offset = offset / max_offset
        
        return normalized_offset * 30  # Scale to degrees


# Global variables
imx500 = None
args = None
intrinsics = None

# ---------------------- Parse Detections ----------------------
def parse_detections(metadata):
    try:
        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return []

        boxes = np_outputs[0][0]
        scores = np_outputs[1][0]
        classes = np_outputs[2][0]

        detections = []
        img_width, img_height = 640, 480

        for i in range(len(scores)):
            conf = float(scores[i])
            if conf < args.threshold:
                continue

            x1, y1, x2, y2 = boxes[i]

            if x2 <= 1.5 and y2 <= 1.5:
                x1 *= img_width
                x2 *= img_width
                y1 *= img_height
                y2 *= img_height

            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            if w <= 0 or h <= 0:
                continue

            detections.append(Detection((x, y, w, h), int(classes[i]), conf))

        return detections
    except Exception as e:
        logging.error(f"Error in parse_detections: {e}")
        return []


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


# ---------------------- Drawing & Tracking ----------------------
def draw_detections(jobs, servo_controller, person_tracker):
    """Draw detections and control servo - runs in separate thread"""
    logging.info("Draw thread started")
    labels = get_labels()
    last_detections = []
    frame_count = 0

    while (job := jobs.get()) is not None:
        try:
            request, async_result = job
            detections = async_result.get() or last_detections
            last_detections = detections

            with MappedArray(request, "main") as m:
                frame = m.array
                frame_h, frame_w = frame.shape[:2]
                model_w, model_h = 640, 640

                frame_count += 1

                # Find person to track
                tracked_person = person_tracker.find_person(detections, labels)
                
                for detection in detections:
                    x, y, w, h = detection.box
                    label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

                    x = int(x * frame_w / model_w)
                    y = int(y * frame_h / model_h)
                    w = int(w * frame_w / model_w)
                    h = int(h * frame_h / model_h)

                    x = max(0, min(x, frame_w - 1))
                    y = max(0, min(y, frame_h - 1))

                    is_tracked = (tracked_person and 
                                 detection == tracked_person['detection'])
                    
                    color = (0, 255, 0) if is_tracked else (255, 0, 0)
                    thickness = 3 if is_tracked else 2
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                    cv2.putText(frame, label, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if is_tracked:
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                        cv2.putText(frame, "TRACKING", (x, y - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Move servo to track person
                if tracked_person and servo_controller.enabled:
                    offset = person_tracker.calculate_offset(tracked_person['center_x'])
                    if offset != 0:
                        servo_controller.adjust_by_offset(offset, args.sensitivity)
                        if frame_count % 10 == 0:
                            logging.info(f"MOVING SERVO: offset={offset:.1f}°, angle={servo_controller.current_angle:.1f}°")
                
                # Draw center line and dead zone
                center_x = frame_w // 2
                dead_zone = int(person_tracker.dead_zone_pixels)
                cv2.line(frame, (center_x, 0), (center_x, frame_h), (0, 255, 255), 1)
                cv2.rectangle(frame, 
                             (center_x - dead_zone, 0), 
                             (center_x + dead_zone, frame_h), 
                             (0, 255, 255), 1)
                
                # Display info
                if servo_controller.enabled:
                    servo_status = f"Servo: {servo_controller.current_angle:.1f}deg [ACTIVE]"
                    status_color = (0, 255, 0)
                else:
                    servo_status = "Servo: DISABLED"
                    status_color = (0, 0, 255)
                    
                cv2.putText(frame, servo_status, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"Detections: {len(detections)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("IMX500 Person Tracking", frame)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    logging.info("Exiting on user request...")
                    servo_controller.center()
                    time.sleep(0.5)
                    cv2.destroyAllWindows()
                    os._exit(0)
                elif key == ord("c"):
                    servo_controller.center()
                    logging.info("Servo centered manually")

            request.release()
        except Exception as e:
            logging.error(f"Error in draw_detections: {e}")
            try:
                request.release()
            except:
                pass


# ---------------------- CLI Arguments ----------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .rpk model")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels.txt file")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection threshold")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for preview")
    parser.add_argument("--pan-pin", type=int, default=17, help="GPIO pin for pan servo")
    parser.add_argument("--dead-zone", type=float, default=0.15, help="Dead zone as fraction of frame width")
    parser.add_argument("--sensitivity", type=float, default=0.3, help="Servo sensitivity (0.1-1.0)")
    parser.add_argument("--no-servo", action="store_true", help="Disable servo control")
    return parser.parse_args()


# ---------------------- Main ----------------------
if __name__ == "__main__":
    args = get_args()
    
    print("\n=== Raspberry Pi 5 Person Tracking with gpiozero ===\n")

    # Check files
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.labels):
        print(f"ERROR: Labels file not found: {args.labels}")
        sys.exit(1)

    # Initialize servo and tracker
    print("Initializing servo controller...")
    servo_controller = ServoController(pan_pin=args.pan_pin, enabled=not args.no_servo)
    
    if not servo_controller.enabled:
        print("✗ Servo initialization failed!")
        response = input("Continue without servo? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("Initializing person tracker...")
    person_tracker = PersonTracker(frame_width=640, frame_height=480, 
                                   dead_zone=args.dead_zone)

    # Initialize IMX500 + Picamera2
    print("Loading IMX500 model...")
    try:
        imx500 = IMX500(args.model)
    except Exception as e:
        print(f"ERROR initializing IMX500: {e}")
        sys.exit(1)
    
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
    logging.info(f"Camera started - person tracking {'ENABLED' if servo_controller.enabled else 'DISABLED'}")

    pool = multiprocessing.Pool(processes=2)
    jobs = queue.Queue()

    threading.Thread(target=draw_detections, 
                    args=(jobs, servo_controller, person_tracker), 
                    daemon=True).start()

    print("\n=== System Ready ===")
    print("Press 'Q' to quit | Press 'C' to center servo")
    print("Watch for 'MOVING SERVO' messages in terminal\n")

    try:
        while True:
            request = picam2.capture_request()
            metadata = request.get_metadata()

            if metadata:
                async_result = pool.apply_async(parse_detections, (metadata,))
                jobs.put((request, async_result))
            else:
                request.release()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        servo_controller.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()