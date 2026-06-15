import cv2
import threading
import time
import os
import datetime
from vision import VisionEngine

class CameraApp:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
        self.crop_w = int(self.frame_width / self.zoom_factor)
        self.crop_h = int(self.frame_height / self.zoom_factor)
        self.crop_x = (self.frame_width - self.crop_w) // 2
        self.crop_y = (self.frame_height - self.crop_h) // 2
        
        # Gesture Debouncing & Cooldown
        self.gesture_counts = {}
        self.gesture_cooldowns = {}
        self.debounce_frames = 6   # Consecutive frames a gesture must be seen
        self.cooldown_seconds = 1.5 # Cooldown after a gesture action triggers
        
        # Drag Pan state
        self.last_fist_pos = None
        
        # Pinch-Zoom state (double fist)
        self.last_fist_distance = None
        
        self.lock = threading.Lock()
        self.current_frame = None
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

    def _check_gesture(self, gesture, gestures):
        """Checks if a gesture passes debounce and cooldown filters."""
        if gesture in gestures:
            self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
            if self.gesture_counts[gesture] >= self.debounce_frames:
                now = time.time()
                last_triggered = self.gesture_cooldowns.get(gesture, 0)
                if now - last_triggered > self.cooldown_seconds:
                    self.gesture_cooldowns[gesture] = now
                    self.gesture_counts[gesture] = 0
                    return True
        else:
            self.gesture_counts[gesture] = 0
        return False

    def update_zoom(self, zoom):
        """Updates the zoom factor and recalculates crop window dimensions thread-safely."""
        with self.lock:
            self.zoom_factor = max(1.0, min(float(zoom), 4.0))
            self.crop_w = int(self.frame_width / self.zoom_factor)
            self.crop_h = int(self.frame_height / self.zoom_factor)
            # Clip crop offsets to fit inside frame
            self.crop_x = max(0, min(self.crop_x, self.frame_width - self.crop_w))
            self.crop_y = max(0, min(self.crop_y, self.frame_height - self.crop_h))

    def start_recording(self):
        """Initializes VideoWriter and starts saving the stream to disk."""
        with self.lock:
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
            
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            
            # Recalculate zoom crop properties
            self.crop_w = int(self.frame_width / self.zoom_factor)
            self.crop_h = int(self.frame_height / self.zoom_factor)
            self.crop_x = (self.frame_width - self.crop_w) // 2
            self.crop_y = (self.frame_height - self.crop_h) // 2
            
            print(f"[SYSTEM] Switched camera to source index: {src_index}")

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
                time.sleep(0.1)
                continue
            
            # Flip horizontally for selfie view
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Process with VisionEngine
            face_box, gestures, fist_positions = self.vision.process_frame(frame)
            
            # Debounced state transitions
            if self._check_gesture("DOUBLE_L", gestures):
                self.state = "DEFAULT"
                
            elif self._check_gesture("SINGLE_L", gestures):
                if self.state == "DEFAULT":
                    self.state = "TRACKING"
                elif self.state == "TRACKING":
                    self.state = "STATIC_ZOOM"
                elif self.state == "STATIC_ZOOM":
                    self.state = "TRACKING"
                    
            elif self._check_gesture("CLAPPERBOARD", gestures):
                # Clapperboard toggles recording
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
                    
            elif self._check_gesture("PEACE", gestures):
                # Peace sign resets state and zoom
                self.state = "DEFAULT"
                self.update_zoom(2.0)
                
            # Double-Fist Pinch Zoom (continuous, no debounce cooldown)
            elif "DOUBLE_FIST" in gestures and len(fist_positions) == 2 and self.state in ["STATIC_ZOOM", "MANUAL_PAN", "TRACKING"]:
                current_dist = ((fist_positions[0][0] - fist_positions[1][0]) ** 2 +
                                (fist_positions[0][1] - fist_positions[1][1]) ** 2) ** 0.5
                if self.last_fist_distance is not None:
                    # Hands moving apart → zoom in, hands moving together → zoom out
                    delta = current_dist - self.last_fist_distance
                    # Scale: ~200px of fist movement = 1.0 zoom change
                    zoom_delta = delta / 200.0
                    new_zoom = max(1.0, min(self.zoom_factor + zoom_delta, 4.0))
                    self.zoom_factor = new_zoom
                    self.crop_w = int(self.frame_width / self.zoom_factor)
                    self.crop_h = int(self.frame_height / self.zoom_factor)
                    self.crop_x = max(0, min(self.crop_x, w - self.crop_w))
                    self.crop_y = max(0, min(self.crop_y, h - self.crop_h))
                self.last_fist_distance = current_dist

            # Single Closed Fist for panning (continuous, no debounce cooldown)
            elif "CLOSED_FIST" in gestures and self.state in ["STATIC_ZOOM", "MANUAL_PAN"]:
                self.state = "MANUAL_PAN"
                if len(fist_positions) > 0:
                    current_fist = fist_positions[0]
                    if self.last_fist_pos:
                        # Pan happens in opposite direction to fist displacement (like drag)
                        dx = self.last_fist_pos[0] - current_fist[0]
                        dy = self.last_fist_pos[1] - current_fist[1]
                        
                        self.crop_x = max(0, min(self.crop_x + dx, w - self.crop_w))
                        self.crop_y = max(0, min(self.crop_y + dy, h - self.crop_h))
                        
                    self.last_fist_pos = current_fist
            else:
                self.last_fist_pos = None
                self.last_fist_distance = None
                if self.state == "MANUAL_PAN":
                    self.state = "STATIC_ZOOM"
            
            # Handle PTZ Logic
            if self.state == "DEFAULT":
                out_frame = frame.copy()
            else:
                # Tracking or Static Zoom
                if self.state == "TRACKING" and face_box:
                    fx, fy, fw, fh = face_box
                    cx = fx + fw // 2
                    cy = fy + fh // 2
                    
                    # Dynamic zoom: face width ~25% of frame = ideal framing
                    # Larger face (closer) → less zoom, smaller face (farther) → more zoom
                    ideal_face_ratio = 0.25  # Target: face occupies 25% of crop width
                    face_ratio = fw / w if w > 0 else 0.25
                    if face_ratio > 0.02:  # Only if face is reasonably detected
                        target_zoom = ideal_face_ratio / face_ratio * 2.0
                        target_zoom = max(1.0, min(target_zoom, 4.0))
                        # Smooth lerp towards target zoom
                        self.zoom_factor += (target_zoom - self.zoom_factor) * 0.05
                        self.zoom_factor = max(1.0, min(self.zoom_factor, 4.0))
                        self.crop_w = int(self.frame_width / self.zoom_factor)
                        self.crop_h = int(self.frame_height / self.zoom_factor)
                    
                    target_x = max(0, min(cx - self.crop_w // 2, w - self.crop_w))
                    target_y = max(0, min(cy - int(self.crop_h * 0.4), h - self.crop_h)) # Face in upper portion
                    
                    # Smooth lerp interpolation
                    self.crop_x += int((target_x - self.crop_x) * 0.2)
                    self.crop_y += int((target_y - self.crop_y) * 0.2)
                
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

            # Optional HUD
            cv2.putText(out_frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(out_frame, f"Gestures: {', '.join(gestures)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
        self.cap.release()
