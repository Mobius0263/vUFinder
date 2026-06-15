# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp
# pyrefly: ignore [missing-import]
import numpy as np
import time
# pyrefly: ignore [missing-import]
from mediapipe.tasks import python
# pyrefly: ignore [missing-import]
from mediapipe.tasks.python import vision

class VisionEngine:
    def __init__(self):
        # Initialize modern MediaPipe Tasks FaceDetector
        face_base_options = python.BaseOptions(model_asset_path='models/face_detector.tflite')
        face_options = vision.FaceDetectorOptions(base_options=face_base_options)
        self.face_detector = vision.FaceDetector.create_from_options(face_options)
        
        # Initialize modern MediaPipe Tasks HandLandmarker
        hand_base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        # Debouncing and state management
        self.last_fist_time = 0
        self.fist_count = 0
        self.last_clapper_time = 0
        self.clapper_count = 0

    def process_frame(self, frame_bgr):
        """
        Process the BGR frame to extract face bounding box and gestures.
        Returns:
            face_box: (x, y, w, h) of the detected face, or None
            gestures: List of string detected gestures
            fist_positions: List of (x, y) for detected closed fists (for tracking drag)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, [], []
            
        h_img, w_img, _ = frame_bgr.shape
        
        # Convert BGR frame to mp.Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Face Detection
        face_box = None
        try:
            face_results = self.face_detector.detect(mp_image)
            if face_results.detections:
                # Find detection with highest score
                best_face = max(face_results.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
                bbox = best_face.bounding_box
                # Bounding box in Tasks is in integer pixels
                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height
                face_box = (max(0, x), max(0, y), min(w, w_img - x), min(h, h_img - y))
        except Exception as e:
            print("Face detection error:", e)

        # Hand Detection & Gesture analysis
        gestures = set()
        fist_positions = []
        l_shapes = 0
        
        try:
            hand_results = self.hand_landmarker.detect(mp_image)
            if hand_results.hand_landmarks:
                for hand_landmarks in hand_results.hand_landmarks:
                    lmList = []
                    for id, lm in enumerate(hand_landmarks):
                        cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                        lmList.append([id, cx, cy])
                    
                    if len(lmList) > 0:
                        tips = [4, 8, 12, 16, 20]
                        fingers_up = []
                        
                        # Thumb (comparing X coordinate to thumb base IP joint)
                        if lmList[tips[0]][1] > lmList[tips[0] - 1][1]:
                            fingers_up.append(1)
                        else:
                            fingers_up.append(0)
                            
                        # 4 fingers (comparing Y coordinate of tip and PIP joint)
                        for id in range(1, 5):
                            if lmList[tips[id]][2] < lmList[tips[id] - 2][2]:
                                fingers_up.append(1)
                            else:
                                fingers_up.append(0)
                                
                        # Detect L-shape (Index up, thumb extended, middle/ring/pinky closed)
                        if fingers_up[1] == 1 and sum(fingers_up[2:]) == 0:
                            thumb_extended = abs(lmList[4][1] - lmList[5][1]) > 20
                            if thumb_extended:
                                l_shapes += 1
                                
                        # Detect Peace sign (Index and Middle up, others closed)
                        if fingers_up[1] == 1 and fingers_up[2] == 1 and sum(fingers_up[3:]) == 0:
                            gestures.add("PEACE")
                            
                        # Detect Closed Fist (Grab)
                        if sum(fingers_up[1:]) == 0:
                            gestures.add("CLOSED_FIST")
                            # Center of fist roughly around MCP of middle finger (landmark 9)
                            fist_positions.append((lmList[9][1], lmList[9][2]))
                
                # Combine gestures
                if l_shapes == 1:
                    gestures.add("SINGLE_L")
                elif l_shapes == 2:
                    gestures.add("DOUBLE_L")
                    
                if len(fist_positions) == 2:
                    gestures.add("DOUBLE_FIST")
                    
                # Clapperboard detection: two hands close together
                if len(hand_results.hand_landmarks) == 2:
                    h1_x = int(hand_results.hand_landmarks[0][9].x * w_img)
                    h1_y = int(hand_results.hand_landmarks[0][9].y * h_img)
                    h2_x = int(hand_results.hand_landmarks[1][9].x * w_img)
                    h2_y = int(hand_results.hand_landmarks[1][9].y * h_img)
                    dist = np.hypot(h1_x - h2_x, h1_y - h2_y)
                    if dist < 100:
                        gestures.add("CLAPPERBOARD")
        except Exception as e:
            print("Hand landmarker error:", e)
            
        return face_box, list(gestures), fist_positions
