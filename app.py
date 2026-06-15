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
    return jsonify({
        "state": state,
        "recording": recording,
        "zoom": zoom,
        "gestures": gestures
    })

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    if cam_app.recording:
        cam_app.stop_recording()
    else:
        cam_app.start_recording()
    return jsonify({"recording": cam_app.recording})

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

@app.route('/reset', methods=['POST'])
def reset():
    with cam_app.lock:
        cam_app.state = "DEFAULT"
    cam_app.update_zoom(2.0)
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
        if filename.endswith(".mp4"):
            path = os.path.join(folder, filename)
            try:
                stat = os.stat(path)
                size_mb = round(stat.st_size / (1024 * 1024), 2)
                created_time = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                files.append({
                    "name": filename,
                    "size": f"{size_mb} MB",
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
    if os.path.exists(path) and filename.endswith(".mp4"):
        os.remove(path)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "File not found"}), 404

@app.route('/cameras', methods=['GET'])
def get_cameras():
    global detected_cameras
    scan = request.args.get('scan', 'false').lower() == 'true'
    if scan or not detected_cameras:
        print("[SYSTEM] Scanning for camera devices...")
        detected_cameras = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                success, _ = cap.read()
                if success:
                    detected_cameras.append(i)
                cap.release()
        if not detected_cameras:
            detected_cameras = [0]
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
