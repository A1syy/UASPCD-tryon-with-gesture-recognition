#!/usr/bin/env python3
"""
Web-based Face Filter Application
Modern GUI menggunakan Flask + WebSocket untuk real-time video streaming.
Desain mengikuti 10 Nielsen's Usability Heuristics.
"""

import os
import sys
import cv2
import time
import glob
import base64
import json
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from typing import Optional, List, Dict, Any
import threading

# Optional: MediaPipe Hands for gesture control
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

# Import FilterEngine
try:
    from filter_ref import FilterEngine
except ImportError:
    print("❌ Error: filter_ref.py tidak ditemukan!")
    print("   Pastikan filter_ref.py ada di folder yang sama.")
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'webcam-filter-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


class WebcamFilterWebApp:
    def __init__(self, masks_folder: Optional[str] = None):
        self.masks_folder = masks_folder
        self.engine: Optional[FilterEngine] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # App state
        self.running = False
        self.paused = False
        self.show_info = True
        self.current_mask_idx = 0
        self.streaming = False
        
        # FPS tracking
        self.fps = 0.0
        self.frame_times = []
        self.max_frame_times = 30
        
        # Available masks
        self.available_masks: List[str] = []
        self.mask_display_names: List[str] = []
        
        # Parameters
        self.param_scale = 200
        self.param_offset_x = 0
        self.param_offset_y = -25
        self.param_yaw = 150
        self.param_pitch = 150
        self.param_roll = 0
        
        # Hand gesture control
        self.enable_hand_control = False
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.last_gesture_time = 0.0
        self.gesture_cooldown = 0.8
        self.last_gesture = ""
        
        # Thread lock
        self.lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize camera and filter engine."""
        print("🎥 Inisialisasi kamera...")
        
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except:
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            print("❌ Gagal membuka kamera!")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari kamera!")
            return False
            
        print(f"✅ Kamera ready: {frame.shape[1]}x{frame.shape[0]}")
        
        print("🔧 Inisialisasi FilterEngine...")
        try:
            self.engine = FilterEngine(masks_folder=self.masks_folder, det_scale=0.75)
            print("✅ FilterEngine ready")
        except Exception as e:
            print(f"❌ Gagal membuat FilterEngine: {e}")
            return False
        
        self.scan_masks()
        
        if HAS_MEDIAPIPE:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
                print("🖐️  Hand gesture control ready (MediaPipe)")
            except Exception as e:
                print(f"⚠️  Failed to init MediaPipe Hands: {e}")
                self.hands = None
        
        self.running = True
        return True
    
    def scan_masks(self):
        """Scan masks folder for available mask files."""
        self.available_masks = ["[No Mask]"]
        self.mask_display_names = ["No Mask"]
        
        if not self.masks_folder or not os.path.isdir(self.masks_folder):
            print("⚠️ No masks folder found")
            return
        
        mask_files = glob.glob(os.path.join(self.masks_folder, "*.png"))
        mask_files.extend(glob.glob(os.path.join(self.masks_folder, "*.jpg")))
        
        for mask_path in sorted(mask_files):
            filename = os.path.basename(mask_path)
            self.available_masks.append(filename)
            display_name = os.path.splitext(filename)[0]
            self.mask_display_names.append(display_name)
        
        print(f"📁 Found {len(self.available_masks)-1} mask(s)")
    
    def get_masks_info(self) -> List[Dict[str, Any]]:
        """Get list of masks with thumbnail info."""
        masks_info = []
        for i, (filename, display_name) in enumerate(zip(self.available_masks, self.mask_display_names)):
            info = {
                "id": i,
                "filename": filename,
                "display_name": display_name,
                "thumbnail": None
            }
            
            if i > 0 and self.masks_folder:
                path = os.path.join(self.masks_folder, filename)
                if os.path.exists(path):
                    try:
                        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            h, w = img.shape[:2]
                            scale = 80 / max(h, w)
                            thumb = cv2.resize(img, (int(w*scale), int(h*scale)))
                            if thumb.shape[2] == 4:
                                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGRA2RGBA)
                            else:
                                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                            _, buffer = cv2.imencode('.png', thumb)
                            info["thumbnail"] = base64.b64encode(buffer).decode('utf-8')
                    except:
                        pass
            
            masks_info.append(info)
        return masks_info
    
    def set_mask(self, idx: int):
        """Set current mask by index."""
        with self.lock:
            if 0 <= idx < len(self.available_masks):
                self.current_mask_idx = idx
                if idx == 0:
                    self.engine.clear_mask()
                else:
                    self.engine.set_mask(self.available_masks[idx])
    
    def update_params(self):
        """Update engine parameters."""
        if self.engine:
            self.engine.set_manual_scale_percent(self.param_scale)
            self.engine.set_offset_x(self.param_offset_x)
            self.engine.set_offset_y(self.param_offset_y)
            self.engine.set_yaw_percent(self.param_yaw)
            self.engine.set_pitch_percent(self.param_pitch)
            self.engine.set_roll_offset(self.param_roll)
    
    def reset_params(self):
        """Reset all parameters to default."""
        self.param_scale = 200
        self.param_offset_x = 0
        self.param_offset_y = -25
        self.param_yaw = 150
        self.param_pitch = 150
        self.param_roll = 0
        if self.engine:
            self.engine.reset_to_defaults()
    
    def detect_gesture(self, frame) -> Optional[str]:
        """Detect hand gesture."""
        if not self.enable_hand_control or self.hands is None:
            return None
        
        current_time = time.time()
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return None
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand skeleton
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Get finger states
        landmarks = hand_landmarks.landmark
        
        # Finger tip and pip indices
        finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
        finger_pips = [6, 10, 14, 18]
        
        fingers_up = []
        for tip, pip in zip(finger_tips, finger_pips):
            fingers_up.append(landmarks[tip].y < landmarks[pip].y)
        
        # Thumb
        thumb_up = landmarks[4].x < landmarks[3].x
        
        gesture = None
        
        # Palm open (all fingers up)
        if all(fingers_up) and thumb_up:
            gesture = "CLEAR"
            self.last_gesture_time = current_time
        
        # V sign (index + middle up, ring + pinky down)
        elif fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            gesture = "NEXT"
            self.last_gesture_time = current_time
        
        # Index only
        elif fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            gesture = "PREV"
            self.last_gesture_time = current_time
        
        if gesture:
            self.last_gesture = gesture
        
        return gesture
    
    def process_gesture(self, gesture: str):
        """Process detected gesture."""
        if gesture == "CLEAR":
            self.set_mask(0)
        elif gesture == "NEXT":
            next_idx = (self.current_mask_idx + 1) % len(self.available_masks)
            self.set_mask(next_idx)
        elif gesture == "PREV":
            prev_idx = (self.current_mask_idx - 1) % len(self.available_masks)
            self.set_mask(prev_idx)
    
    def get_frame(self):
        """Get processed frame."""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Calculate FPS
        current_time = time.time()
        self.frame_times.append(current_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        
        if not self.paused:
            # Detect gesture
            gesture = self.detect_gesture(frame)
            if gesture:
                self.process_gesture(gesture)
            
            # Process frame with filter
            self.update_params()
            frame = self.engine.process_frame(frame)
        
        # Add info overlay if enabled
        if self.show_info:
            status = "PAUSED" if self.paused else "LIVE"
            cv2.putText(frame, f"FPS: {self.fps:.1f} | {status}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.enable_hand_control and self.last_gesture:
                cv2.putText(frame, f"Gesture: {self.last_gesture}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def generate_frames(self):
        """Generator for video streaming."""
        while self.running and self.streaming:
            frame = self.get_frame()
            if frame is None:
                continue
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.streaming = False
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()


# Global app instance
filter_app = None


def init_app(masks_folder: str):
    """Initialize the filter application."""
    global filter_app
    filter_app = WebcamFilterWebApp(masks_folder=masks_folder)
    if not filter_app.initialize():
        print("❌ Failed to initialize application")
        return False
    return True


# Flask Routes
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global filter_app
    if filter_app:
        filter_app.streaming = True
        return Response(filter_app.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not initialized", 500


@app.route('/api/masks', methods=['GET'])
def get_masks():
    """Get list of available masks."""
    global filter_app
    if filter_app:
        return jsonify({
            "success": True,
            "masks": filter_app.get_masks_info(),
            "current": filter_app.current_mask_idx
        })
    return jsonify({"success": False, "error": "App not initialized"})


@app.route('/api/mask', methods=['POST'])
def set_mask():
    """Set current mask."""
    global filter_app
    if filter_app:
        data = request.json
        idx = data.get('index', 0)
        filter_app.set_mask(idx)
        return jsonify({
            "success": True,
            "current": filter_app.current_mask_idx,
            "mask_name": filter_app.mask_display_names[filter_app.current_mask_idx]
        })
    return jsonify({"success": False, "error": "App not initialized"})


@app.route('/api/params', methods=['GET'])
def get_params():
    """Get current parameters."""
    global filter_app
    if filter_app:
        return jsonify({
            "success": True,
            "params": {
                "scale": filter_app.param_scale,
                "offset_x": filter_app.param_offset_x,
                "offset_y": filter_app.param_offset_y,
                "yaw": filter_app.param_yaw,
                "pitch": filter_app.param_pitch,
                "roll": filter_app.param_roll
            }
        })
    return jsonify({"success": False, "error": "App not initialized"})


@app.route('/api/params', methods=['POST'])
def set_params():
    """Update parameters."""
    global filter_app
    if filter_app:
        data = request.json
        if 'scale' in data:
            filter_app.param_scale = int(data['scale'])
        if 'offset_x' in data:
            filter_app.param_offset_x = int(data['offset_x'])
        if 'offset_y' in data:
            filter_app.param_offset_y = int(data['offset_y'])
        if 'yaw' in data:
            filter_app.param_yaw = int(data['yaw'])
        if 'pitch' in data:
            filter_app.param_pitch = int(data['pitch'])
        if 'roll' in data:
            filter_app.param_roll = int(data['roll'])
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "App not initialized"})


@app.route('/api/reset', methods=['POST'])
def reset_params():
    """Reset all parameters to default."""
    global filter_app
    if filter_app:
        filter_app.reset_params()
        return jsonify({
            "success": True,
            "params": {
                "scale": filter_app.param_scale,
                "offset_x": filter_app.param_offset_x,
                "offset_y": filter_app.param_offset_y,
                "yaw": filter_app.param_yaw,
                "pitch": filter_app.param_pitch,
                "roll": filter_app.param_roll
            }
        })
    return jsonify({"success": False, "error": "App not initialized"})


@app.route('/api/toggle', methods=['POST'])
def toggle_setting():
    """Toggle app settings (pause, info, hand_control)."""
    global filter_app
    if filter_app:
        data = request.json
        setting = data.get('setting')
        
        if setting == 'pause':
            filter_app.paused = not filter_app.paused
            return jsonify({"success": True, "value": filter_app.paused})
        elif setting == 'info':
            filter_app.show_info = not filter_app.show_info
            return jsonify({"success": True, "value": filter_app.show_info})
        elif setting == 'hand_control':
            filter_app.enable_hand_control = not filter_app.enable_hand_control
            return jsonify({"success": True, "value": filter_app.enable_hand_control})
        
        return jsonify({"success": False, "error": "Unknown setting"})
    return jsonify({"success": False, "error": "App not initialized"})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current app status."""
    global filter_app
    if filter_app:
        return jsonify({
            "success": True,
            "status": {
                "running": filter_app.running,
                "paused": filter_app.paused,
                "show_info": filter_app.show_info,
                "hand_control": filter_app.enable_hand_control,
                "fps": round(filter_app.fps, 1),
                "current_mask": filter_app.mask_display_names[filter_app.current_mask_idx],
                "current_mask_idx": filter_app.current_mask_idx,
                "last_gesture": filter_app.last_gesture
            }
        })
    return jsonify({"success": False, "error": "App not initialized"})


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Web-based Face Filter Application')
    parser.add_argument('--masks-folder', type=str, default='mask',
                       help='Path to folder containing mask images')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to run the web server on')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎬 WEBCAM FILTER - WEB MODE")
    print("="*60)
    
    if init_app(args.masks_folder):
        print(f"\n🌐 Starting web server at http://{args.host}:{args.port}")
        print("   Open this URL in your browser")
        print("   Press Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        socketio.run(app, host=args.host, port=args.port, debug=False)
    else:
        print("❌ Failed to start application")
        sys.exit(1)
