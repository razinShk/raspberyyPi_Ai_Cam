# ---------------------- Servo Controller (Pi 5 / gpiozero) ----------------------
# Replace the lgpio imports at the top of your file with:
SERVO_AVAILABLE = False
try:
    from gpiozero import AngularServo
    SERVO_AVAILABLE = True
    print("✓ gpiozero imported successfully")
except ImportError:
    print("✗ gpiozero not available. Install with: sudo apt-get install python3-gpiozero")

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