import cv2
import threading
import time
import os
import datetime
from vision import VisionEngine

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None


class CameraApp:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.current_src = int(src)
        
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        
        self.vision = VisionEngine()
        
        # State Management
        self.state = "DEFAULT" # DEFAULT, TRACKING, STATIC_ZOOM, MANUAL_PAN
        self.recording = False
        self.video_writer = None
        self.last_gestures = [] # Stores last detected gestures list
        
        # Zoom & Crop window properties
        self.zoom_factor = 2.0
        self.target_zoom_factor = 2.0
        self.crop_w = int(self.frame_width / self.zoom_factor)
        self.crop_h = int(self.frame_height / self.zoom_factor)
        self.crop_x = (self.frame_width - self.crop_w) // 2
        self.crop_y = (self.frame_height - self.crop_h) // 2
        self.target_crop_x = self.crop_x
        self.target_crop_y = self.crop_y
        self.gestures_enabled = True
        
        # Gesture Debouncing & Cooldown
        self.gesture_counts = {}
        self.gesture_cooldowns = {}
        self.debounce_frames = 6   # Consecutive frames a gesture must be seen
        self.cooldown_seconds = 1.5 # Cooldown after a gesture action triggers
        
        # Drag Pan state
        self.last_fist_pos = None
        self.gesture_panning_active = False
        
        # Pinch-Zoom state (double fist)
        self.last_fist_distance = None

        # Countdown Timer state
        self.countdown_type = None  # None, "record", or "capture"
        self.countdown_start_time = None
        self.countdown_duration = 3.0

        # Virtual Camera State
        self.virtual_camera_active = False
        self.virtual_cam = None

        self.lock = threading.Lock()

        self.current_frame = None
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _check_gesture(self, gesture, gestures):
        """Checks if a gesture passes debounce and cooldown filters."""
        # Custom sensitivity/debounce thresholds
        thresholds = {
            "PEACE": 20,          # 1.0 second of deliberate pose at 20 FPS
            "CLAPPERBOARD": 8,    # ~0.4 second of deliberate pose at 20 FPS
            "SINGLE_L": 6,
            "DOUBLE_L": 6
        }
        cooldowns = {
            "PEACE": 5.0,         # 5 second cooldown to avoid rapid fire captures
            "CLAPPERBOARD": 5.0,  # 5 second cooldown
            "SINGLE_L": 1.5,
            "DOUBLE_L": 1.5
        }
        
        limit = thresholds.get(gesture, self.debounce_frames)
        cooldown = cooldowns.get(gesture, self.cooldown_seconds)
        
        if gesture in gestures:
            self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
            if self.gesture_counts[gesture] >= limit:
                now = time.time()
                last_triggered = self.gesture_cooldowns.get(gesture, 0)
                if now - last_triggered > cooldown:
                    self.gesture_cooldowns[gesture] = now
                    self.gesture_counts[gesture] = 0
                    return True
        else:
            self.gesture_counts[gesture] = 0
        return False

    def update_zoom(self, zoom):
        """Updates the zoom factor and recalculates crop window dimensions thread-safely."""
        with self.lock:
            if self.state == "DEFAULT":
                self.state = "STATIC_ZOOM"
            
            new_zoom = max(1.0, min(float(zoom), 4.0))
            
            # Find current center of the crop window
            current_center_x = self.target_crop_x + self.crop_w // 2
            current_center_y = self.target_crop_y + self.crop_h // 2
            
            self.target_zoom_factor = new_zoom
            
            new_crop_w = int(self.frame_width / self.target_zoom_factor)
            new_crop_h = int(self.frame_height / self.target_zoom_factor)
            
            self.target_crop_x = int(current_center_x - new_crop_w // 2)
            self.target_crop_y = int(current_center_y - new_crop_h // 2)
            
            self.target_crop_x = max(0, min(self.target_crop_x, self.frame_width - new_crop_w))
            self.target_crop_y = max(0, min(self.target_crop_y, self.frame_height - new_crop_h))

    def start_recording(self):
        """Triggers a 3-second countdown to start saving the stream to disk."""
        with self.lock:
            if self.recording:
                return
            if self.countdown_type is None:
                self.countdown_type = "record"
                self.countdown_start_time = time.time()
                print("[SYSTEM] Starting 3-second countdown for recording...")

    def start_photo_countdown(self):
        """Triggers a 3-second countdown to capture a photo."""
        with self.lock:
            if self.countdown_type is None:
                self.countdown_type = "capture"
                self.countdown_start_time = time.time()
                print("[SYSTEM] Starting 3-second countdown for photo capture...")

    def get_countdown_remaining(self):
        """Returns the remaining countdown time in integer seconds, or 0. Thread-safe."""
        with self.lock:
            if self.countdown_type and self.countdown_start_time:
                elapsed = time.time() - self.countdown_start_time
                return max(0, int(self.countdown_duration - elapsed) + 1)
            return 0

    def _start_recording_immediate(self):
        """Initializes VideoWriter and starts saving the stream. Caller MUST hold self.lock."""
        if self.recording:
            return
        
        # Auto-create recordings folder in current directory
        os.makedirs("recordings", exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("recordings", f"recording_{timestamp}.mp4")
        
        # Universal AVI/MP4 compatible codec on Windows OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.frame_width, self.frame_height))
        self.recording = True
        print(f"[REC] Started recording to {filename}")

    def _capture_photo_immediate(self, frame):
        """Saves the current cropped/zoomed frame as a JPEG image. Caller MUST hold self.lock."""
        if frame is None:
            return
        
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("recordings", f"photo_{timestamp}.jpg")
        
        # Crop frame if zoomed
        h, w, _ = frame.shape
        if self.state != "DEFAULT":
            cropped = frame[self.crop_y:self.crop_y+self.crop_h, self.crop_x:self.crop_x+self.crop_w]
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                photo_frame = cv2.resize(cropped, (w, h))
            else:
                photo_frame = frame.copy()
        else:
            photo_frame = frame.copy()
            
        cv2.imwrite(filename, photo_frame)
        print(f"[PHOTO] Captured photo to {filename}")

    def _stop_recording_internal(self):
        """Internal helper to stop recording. Caller MUST already hold self.lock."""
        if not self.recording:
            return
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        print("[REC] Stopped recording")

    def stop_recording(self):
        """Stops recording and releases the VideoWriter object."""
        with self.lock:
            # Also cancel any active recording countdown
            if self.countdown_type == "record":
                self.countdown_type = None
                self.countdown_start_time = None
                print("[SYSTEM] Cancelled recording countdown.")
            self._stop_recording_internal()

    def change_camera(self, src_index):
        """Changes the camera source index thread-safely."""
        with self.lock:
            # Stop recording first to prevent corrupting video writers
            self._stop_recording_internal()
            
            # Release old capture
            if self.cap is not None:
                self.cap.release()
            
            # Open new capture source
            self.cap = cv2.VideoCapture(int(src_index))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.current_src = int(src_index)
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            
            # Recalculate zoom crop properties
            self.crop_w = int(self.frame_width / self.zoom_factor)
            self.crop_h = int(self.frame_height / self.zoom_factor)
            self.crop_x = (self.frame_width - self.crop_w) // 2
            self.crop_y = (self.frame_height - self.crop_h) // 2
            self.target_crop_x = self.crop_x
            self.target_crop_y = self.crop_y
            self.target_zoom_factor = self.zoom_factor
            
            print(f"[SYSTEM] Switched camera to source index: {src_index}")

    def toggle_virtual_camera(self, enable: bool):
        """Toggles output streaming to a virtual webcam device."""
        if pyvirtualcam is None:
            raise RuntimeError("pyvirtualcam library is not installed.")
            
        with self.lock:
            if enable:
                if self.virtual_camera_active:
                    return
                # Initialize pyvirtualcam
                try:
                    # Note: Using PixelFormat.BGR is directly compatible with OpenCV's frame output
                    self.virtual_cam = pyvirtualcam.Camera(
                        width=self.frame_width,
                        height=self.frame_height,
                        fps=20,
                        fmt=pyvirtualcam.PixelFormat.BGR
                    )
                    self.virtual_camera_active = True
                    print(f"[VIRTUAL CAM] Started streaming to device: {self.virtual_cam.device}")
                except Exception as e:
                    self.virtual_camera_active = False
                    self.virtual_cam = None
                    print(f"[VIRTUAL CAM] Failed to initialize virtual camera: {e}")
                    raise RuntimeError(f"Could not open virtual camera: {e}. Please ensure a driver (like OBS Virtual Camera or Unity Capture) is installed.")
            else:
                if not self.virtual_camera_active:
                    return
                self.virtual_camera_active = False
                if self.virtual_cam is not None:
                    self.virtual_cam.close()
                    self.virtual_cam = None
                print("[VIRTUAL CAM] Stopped streaming to virtual camera")

    def reset_to_defaults(self):
        """Cleans and resets camera state, targets, and zoom factors to defaults."""
        with self.lock:
            self.state = "DEFAULT"
            self.gesture_panning_active = False
            self.target_zoom_factor = 2.0
            self.zoom_factor = 2.0
            self.crop_w = int(self.frame_width / self.zoom_factor)
            self.crop_h = int(self.frame_height / self.zoom_factor)
            self.crop_x = (self.frame_width - self.crop_w) // 2
            self.crop_y = (self.frame_height - self.crop_h) // 2
            self.target_crop_x = self.crop_x
            self.target_crop_y = self.crop_y
            print("[SYSTEM] Reset configuration, targets, and zoom to defaults.")

    def pan_camera(self, dx, dy):
        """Manually pans the target camera viewport by the given coordinate deltas."""
        with self.lock:
            if self.state != "MANUAL_PAN":
                self.state = "MANUAL_PAN"
            self.gesture_panning_active = False
            
            self.target_crop_x = max(0, min(self.target_crop_x + dx, self.frame_width - self.crop_w))
            self.target_crop_y = max(0, min(self.target_crop_y + dy, self.frame_height - self.crop_h))

    def _capture_loop(self):
        while self.running:
            frame = None
            with self.lock:
                if self.cap is not None and self.cap.isOpened():
                    success, frame = self.cap.read()
                    if not success:
                        frame = None
            # Sleep outside the lock so change_camera can acquire it
            if frame is None:
                time.sleep(0.03)  # Reduce sleep to run closer to ~30 FPS
                continue
            
            # Flip horizontally for selfie view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process with VisionEngine
            face_box, gestures, fist_positions = self.vision.process_frame(frame)
            
            # Process countdown logic
            action_to_trigger = None
            with self.lock:
                if self.countdown_type is not None and self.countdown_start_time is not None:
                    elapsed = time.time() - self.countdown_start_time
                    if elapsed >= self.countdown_duration:
                        action_to_trigger = self.countdown_type
                        self.countdown_type = None
                        self.countdown_start_time = None

            if action_to_trigger == "record":
                with self.lock:
                    self._start_recording_immediate()
            elif action_to_trigger == "capture":
                with self.lock:
                    self._capture_photo_immediate(frame)

            # Debounced state transitions (gated by gestures_enabled)
            gestures_to_process = gestures if self.gestures_enabled else []

            if self._check_gesture("DOUBLE_L", gestures_to_process):
                self.state = "DEFAULT"
                
            elif self._check_gesture("SINGLE_L", gestures_to_process):
                if self.state == "DEFAULT":
                    self.state = "TRACKING"
                elif self.state == "TRACKING":
                    self.state = "STATIC_ZOOM"
                elif self.state == "STATIC_ZOOM":
                    self.state = "TRACKING"
                    
            elif self._check_gesture("CLAPPERBOARD", gestures_to_process):
                if self.recording:
                    self.stop_recording()
                else:
                    with self.lock:
                        if self.countdown_type is None:
                            self.countdown_type = "record"
                            self.countdown_start_time = time.time()
                            print("[SYSTEM] Starting 3-second countdown for recording...")
                    
            elif self._check_gesture("PEACE", gestures_to_process):
                with self.lock:
                    if self.countdown_type is None:
                        self.countdown_type = "capture"
                        self.countdown_start_time = time.time()
                        print("[SYSTEM] Starting 3-second countdown for photo capture...")
                
            # Double-Fist Pinch Zoom (continuous, updates target zoom/crop)
            elif "DOUBLE_FIST" in gestures_to_process and len(fist_positions) == 2 and self.state in ["STATIC_ZOOM", "MANUAL_PAN", "TRACKING"]:
                if self.state == "TRACKING":
                    self.state = "STATIC_ZOOM"
                current_dist = ((fist_positions[0][0] - fist_positions[1][0]) ** 2 +
                                (fist_positions[0][1] - fist_positions[1][1]) ** 2) ** 0.5
                
                mid_x = (fist_positions[0][0] + fist_positions[1][0]) // 2
                mid_y = (fist_positions[0][1] + fist_positions[1][1]) // 2
                
                if self.last_fist_distance is not None:
                    delta = current_dist - self.last_fist_distance
                    zoom_delta = delta / 200.0
                    self.target_zoom_factor = max(1.0, min(self.target_zoom_factor + zoom_delta, 4.0))
                    
                    target_w = int(self.frame_width / self.target_zoom_factor)
                    target_h = int(self.frame_height / self.target_zoom_factor)
                    self.target_crop_x = max(0, min(int(mid_x - target_w // 2), w - target_w))
                    self.target_crop_y = max(0, min(int(mid_y - target_h // 2), h - target_h))
                self.last_fist_distance = current_dist

            # Single Closed Fist for panning (continuous, updates target crop)
            elif "CLOSED_FIST" in gestures_to_process and self.state in ["STATIC_ZOOM", "MANUAL_PAN"]:
                self.state = "MANUAL_PAN"
                self.gesture_panning_active = True
                if len(fist_positions) > 0:
                    current_fist = fist_positions[0]
                    if self.last_fist_pos:
                        dx = self.last_fist_pos[0] - current_fist[0]
                        dy = self.last_fist_pos[1] - current_fist[1]
                        
                        self.target_crop_x = max(0, min(self.target_crop_x + dx, w - self.crop_w))
                        self.target_crop_y = max(0, min(self.target_crop_y + dy, h - self.crop_h))
                        
                    self.last_fist_pos = current_fist
            else:
                self.last_fist_pos = None
                self.last_fist_distance = None
                if self.state == "MANUAL_PAN" and self.gesture_panning_active:
                    self.state = "STATIC_ZOOM"
                    self.gesture_panning_active = False
            
            # Handle PTZ Logic and coordinate smoothing
            with self.lock:
                if self.state != "MANUAL_PAN":
                    self.gesture_panning_active = False
                
                if self.state == "DEFAULT":
                    self.target_zoom_factor = 1.0
                    self.target_crop_x = 0
                    self.target_crop_y = 0
                
                elif self.state == "TRACKING" and face_box:
                    fx, fy, fw, fh = face_box
                    cx = fx + fw // 2
                    cy = fy + fh // 2
                    
                    ideal_face_ratio = 0.25
                    face_ratio = fw / w if w > 0 else 0.25
                    if face_ratio > 0.02:
                        target_zoom = ideal_face_ratio / face_ratio * 1.5
                        self.target_zoom_factor = max(1.0, min(target_zoom, 4.0))
                    
                    target_w = int(self.frame_width / self.target_zoom_factor)
                    target_h = int(self.frame_height / self.target_zoom_factor)
                    self.target_crop_x = max(0, min(cx - target_w // 2, w - target_w))
                    self.target_crop_y = max(0, min(cy - int(target_h * 0.4), h - target_h))

                # Smoothly interpolate crop window and zoom factor thread-safely
                # Snappy (0.25) for manual states to feel responsive, and smooth (0.08) for tracking to eliminate jitter
                lerp_factor = 0.25 if self.state in ["MANUAL_PAN", "STATIC_ZOOM"] else 0.08
                
                self.zoom_factor += (self.target_zoom_factor - self.zoom_factor) * lerp_factor
                self.zoom_factor = max(1.0, min(self.zoom_factor, 4.0))
                
                self.crop_w = int(self.frame_width / self.zoom_factor)
                self.crop_h = int(self.frame_height / self.zoom_factor)
                
                self.crop_x += (self.target_crop_x - self.crop_x) * lerp_factor
                self.crop_y += (self.target_crop_y - self.crop_y) * lerp_factor
                
                self.crop_x = max(0, min(int(self.crop_x), w - self.crop_w))
                self.crop_y = max(0, min(int(self.crop_y), h - self.crop_h))

            # Apply crop and resize back to output resolution
            cropped = frame[self.crop_y:self.crop_y+self.crop_h, self.crop_x:self.crop_x+self.crop_w]
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                out_frame = cv2.resize(cropped, (w, h))
            else:
                out_frame = frame.copy()

            # Record frame (write clean frame before drawing HUD)
            with self.lock:
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(out_frame)

            # Send to virtual camera outside the main lock to prevent blocking other threads
            send_to_vcam = False
            vcam_device = None
            with self.lock:
                if self.virtual_camera_active and self.virtual_cam is not None:
                    send_to_vcam = True
                    vcam_device = self.virtual_cam

            if send_to_vcam and vcam_device is not None:
                try:
                    vcam_device.send(out_frame)
                    vcam_device.sleep_until_next_frame()
                except Exception as e:
                    print(f"[VIRTUAL CAM] Send error: {e}")
                    with self.lock:
                        self.virtual_camera_active = False
                        if self.virtual_cam is not None:
                            try:
                                self.virtual_cam.close()
                            except Exception:
                                pass
                            self.virtual_cam = None


            # Optional HUD
            cv2.putText(out_frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(out_frame, f"Gestures: {', '.join(gestures)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw Countdown HUD if active
            with self.lock:
                if self.countdown_type is not None:
                    elapsed = time.time() - self.countdown_start_time
                    remaining = max(0, int(self.countdown_duration - elapsed) + 1)
                    if remaining > 0:
                        text = f"{self.countdown_type.upper()} IN {remaining}..."
                        # Red for record, green for capture
                        color = (0, 0, 255) if self.countdown_type == "record" else (0, 255, 0)
                        
                        # Calculate center based on width and height
                        center_x = w // 2
                        center_y = h // 2
                        
                        # Draw countdown label in center of frame
                        font_scale = 1.6
                        thickness = 4
                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
                        text_x = center_x - text_w // 2
                        text_y = center_y + text_h // 2
                        
                        cv2.putText(out_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
                        
                        # Draw pulsing circle around center
                        pulse_size = int(80 + (time.time() % 1) * 30)
                        cv2.circle(out_frame, (center_x, center_y), pulse_size, color, 3)

            with self.lock:
                self.current_frame = out_frame
                self.last_gestures = gestures

    def get_frame(self):
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()
            
    def stop(self):
        self.running = False
        self.thread.join()
        self.stop_recording()
        # Close virtual camera safely
        with self.lock:
            if self.virtual_cam is not None:
                try:
                    self.virtual_cam.close()
                except Exception:
                    pass
                self.virtual_cam = None
            self.virtual_camera_active = False
        self.cap.release()

