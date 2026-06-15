# vUFinder - Smart Camera Portal

vUFinder is a premium, glassmorphic webcam dashboard built on Python 3.13, Flask, OpenCV, and MediaPipe. It upgrades your standard webcam feed with responsive face-tracking, manual mouse/scroll controls, gesture-based operations, local HD video/photo captures, and OBS-style virtual camera streaming.

---

## ✨ Features

- **Dynamic Face Tracking (PTZ)**: Automatically zooms and pans to frame your face (keeping it in focus as you move around).
- **Manual Control Overrides**:
  - **Click-and-Drag**: Click and drag your mouse directly on the video stream to pan.
  - **Scroll Wheel**: Scroll up/down on the stream to zoom in/out.
  - **Control Dashboard**: Adjust the zoom slider or override the system mode.
- **Intelligent Gesture Control System**:
  - ✌️ **Peace Sign**: Triggers a 3-second countdown to capture a high-quality photo.
  - 🎬 **Clapperboard**: Triggers a 3-second countdown to start/stop video recording.
  - ☝️ **Single L Shape**: Toggles face-tracking on/off.
  - 👐 **Double L Shape**: Resets the camera crop and zoom to defaults.
  - ✊ **Closed Fist**: Continuous drag-panning (move fist in space to pan).
  - ✊✊ **Double Fist Pinch**: Continuous zoom (spread/close fists to zoom).
- **Gesture Control Mute**: Toggle switch to temporarily disable gesture action triggers to avoid accidental captures.
- **Virtual Camera Output**: Streams your processed, cropped, and zoomed feed directly to an OBS-style virtual webcam (`pyvirtualcam`) for use in Zoom, Teams, or Discord.
- **Recordings Gallery**: Play, download, and delete your saved recordings and photos directly inside a built-in media viewer modal.
- **Friendly Hardware Selection**: Lists your connected webcam devices by their actual directshow hardware names (e.g. "Integrated Camera").

---

## 🛠️ Installation & Setup

Ensure you are running **Python 3.13** on a **Windows** system.

1. **Clone or navigate to the repository directory**:
   ```powershell
   cd c:\Users\user\Documents\camerapp
   ```

2. **Create and activate a Python virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
   *Note: On Windows, `pygrabber` is utilized to retrieve camera names, and `pyvirtualcam` enables virtual output streaming.*

4. **Verify MediaPipe Models**:
   The app uses the modern MediaPipe Tasks framework. Make sure the TFLite models are placed in the `models/` directory:
   - `models/hand_landmarker.task`
   - `models/face_detector.tflite`
   *(These are already bundled with the project).*

---

## 🚀 Running the Application

Start the Flask application server:

```powershell
.venv\Scripts\python app.py
```

Once the server initializes (loading the TensorFlow Lite models), open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🎮 How to Use

### 1. Manual Navigation
- **Pan**: Click and drag your mouse on the preview feed.
- **Zoom**: Hover over the feed and scroll your mouse wheel, or use the **Manual Zoom Factor** slider on the control panel.
- **Reset**: Click the **Reset Device** button to center and zoom out.

### 2. Gesture Actions
- Hold a gesture in the frame to trigger it. 
- Actions like **Capture Photo** and **Record** display a 3-second glassmorphic countdown HUD overlay over the stream before execution.
- If you don't want gestures triggering actions while speaking or moving, click the **Gestures: ENABLED** button under the *Gesture Control Mute* section to mute them.

### 3. Virtual Streaming
- Install a virtual camera driver (such as **OBS Virtual Camera** or **Unity Capture**).
- Click **Start Virtual Cam** on the control panel.
- Open your video calling software (Zoom, Teams) and select the virtual camera source. It will stream your active cropped, smoothed feed!
