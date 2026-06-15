# pyrefly: ignore [missing-import]
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
# pyrefly: ignore [missing-import]
import cv2
import os
import datetime
from camera import CameraApp

app = Flask(__name__)
cam_app = CameraApp()
detected_cameras = []

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        frame = cam_app.get_frame()
        if frame is None:
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with cam_app.lock:
        state = cam_app.state
        recording = cam_app.recording
        zoom = cam_app.zoom_factor
        gestures = list(cam_app.last_gestures) if cam_app.last_gestures else []
        countdown_type = cam_app.countdown_type
        virtual_camera_active = cam_app.virtual_camera_active
        gestures_enabled = cam_app.gestures_enabled
    countdown_remaining = cam_app.get_countdown_remaining()
    return jsonify({
        "state": state,
        "recording": recording,
        "zoom": zoom,
        "gestures": gestures,
        "countdown_type": countdown_type,
        "countdown_remaining": countdown_remaining,
        "virtual_camera_active": virtual_camera_active,
        "gestures_enabled": gestures_enabled
    })

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    if cam_app.recording:
        cam_app.stop_recording()
    else:
        cam_app.start_recording()
    return jsonify({"recording": cam_app.recording, "countdown": cam_app.countdown_type is not None})

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    cam_app.start_photo_countdown()
    return jsonify({"success": True})

@app.route('/toggle_virtual_camera', methods=['POST'])
def toggle_virtual_camera():
    data = request.json or {}
    enable = data.get("enable", False)
    try:
        cam_app.toggle_virtual_camera(enable)
        return jsonify({
            "success": True,
            "virtual_camera_active": cam_app.virtual_camera_active
        })
    except Exception as e:
        print(f"[ERROR] Failed to toggle virtual camera: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/set_state', methods=['POST'])
def set_state():
    data = request.json or {}
    state_val = data.get("state", "DEFAULT")
    valid_states = ["DEFAULT", "TRACKING", "STATIC_ZOOM", "MANUAL_PAN"]
    if state_val in valid_states:
        with cam_app.lock:
            cam_app.state = state_val
        return jsonify({"state": cam_app.state})
    return jsonify({"error": "Invalid state"}), 400

@app.route('/toggle_gestures', methods=['POST'])
def toggle_gestures():
    data = request.json or {}
    enabled = data.get("enabled", True)
    with cam_app.lock:
        cam_app.gestures_enabled = enabled
    return jsonify({"success": True, "gestures_enabled": cam_app.gestures_enabled})

@app.route('/pan', methods=['POST'])
def pan():
    data = request.json or {}
    dx = data.get("dx", 0)
    dy = data.get("dy", 0)
    cam_app.pan_camera(dx, dy)
    return jsonify({"success": True, "crop_x": cam_app.crop_x, "crop_y": cam_app.crop_y})

@app.route('/reset', methods=['POST'])
def reset():
    cam_app.reset_to_defaults()
    return jsonify({"state": cam_app.state, "zoom": cam_app.zoom_factor})

@app.route('/zoom', methods=['POST'])
def zoom():
    data = request.json or {}
    zoom_val = data.get("zoom", 2.0)
    cam_app.update_zoom(zoom_val)
    return jsonify({"zoom": cam_app.zoom_factor})

@app.route('/recordings', methods=['GET'])
def list_recordings():
    folder = "recordings"
    if not os.path.exists(folder):
        return jsonify([])
    
    files = []
    for filename in sorted(os.listdir(folder), reverse=True):
        if filename.endswith(".mp4") or filename.endswith(".jpg"):
            path = os.path.join(folder, filename)
            try:
                stat = os.stat(path)
                size_mb = round(stat.st_size / (1024 * 1024), 2)
                size_display = f"{size_mb} MB" if size_mb > 0.01 else f"{round(stat.st_size / 1024, 2)} KB"
                created_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                files.append({
                    "name": filename,
                    "size": size_display,
                    "created": created_time
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return jsonify(files)

@app.route('/recordings/<path:filename>', methods=['GET'])
def download_recording(filename):
    return send_from_directory("recordings", filename, as_attachment=True)

@app.route('/recordings/<path:filename>', methods=['DELETE'])
def delete_recording(filename):
    path = os.path.join("recordings", filename)
    try:
        if os.path.exists(path) and (filename.endswith(".mp4") or filename.endswith(".jpg")):
            os.remove(path)
            return jsonify({"success": True})
        return jsonify({"success": False, "error": "File not found"}), 404
    except Exception as e:
        print(f"[ERROR] Failed to delete file {filename}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/cameras', methods=['GET'])
def get_cameras():
    global detected_cameras
    scan = request.args.get('scan', 'false').lower() == 'true'
    if scan or not detected_cameras:
        print("[SYSTEM] Enumerating camera devices...")
        device_names = []
        try:
            from pygrabber.dshow_graph import FilterGraph
            device_names = FilterGraph().get_input_devices()
        except Exception as e:
            print(f"[SYSTEM] pygrabber failed to enumerate: {e}")

        detected_cameras = []
        if device_names:
            for i, name in enumerate(device_names):
                # Skip opening/reading test since the camera might be actively occupied by our capture thread.
                # pygrabber already verified it exists on Windows.
                detected_cameras.append({"index": i, "name": name})
        else:
            active_idx = getattr(cam_app, 'current_src', 0)
            for i in range(4):
                if i == active_idx:
                    detected_cameras.append({"index": i, "name": f"Camera {i}"})
                    continue
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    detected_cameras.append({"index": i, "name": f"Camera {i}"})
                    cap.release()
        if not detected_cameras:
            detected_cameras = [{"index": 0, "name": "Camera 0"}]
    return jsonify(detected_cameras)


@app.route('/change_camera', methods=['POST'])
def change_camera():
    data = request.json or {}
    camera_index = data.get("index", 0)
    try:
        cam_app.change_camera(camera_index)
        return jsonify({"success": True, "index": camera_index})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Run Flask on all interfaces to allow local/tunnel access
    app.run(host='0.0.0.0', port=5000, threaded=True)
