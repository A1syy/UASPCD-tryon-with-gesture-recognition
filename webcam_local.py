#!/usr/bin/env python3
"""
Local Webcam Filter GUI
Aplikasi desktop untuk menerapkan face filter secara real-time menggunakan webcam.

Controls (GUI):
    - Gunakan trackbar/slider untuk adjust parameter
    - Pilih mask dari dropdown list
    - Tombol untuk reset, pause, dll
    - ESC untuk keluar
"""

import os
import sys
import cv2
import time
import glob
import numpy as np
from typing import Optional, List, Tuple, Dict
import mediapipe as mp

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


class WebcamFilterApp:
    def __init__(self, masks_folder: Optional[str] = None):
        """
        Initialize webcam filter application.
        
        Args:
            masks_folder: Path ke folder yang berisi mask files
        """
        self.masks_folder = masks_folder
        self.engine: Optional[FilterEngine] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # App state
        self.running = False
        self.paused = 0  # 0 = running, 1 = paused
        self.show_info = 1  # 0 = hide, 1 = show
        self.current_mask_name = "None"
        self.current_mask_idx = 0
        
        # FPS tracking
        self.fps = 0.0
        self.frame_times = []
        self.max_frame_times = 30
        
        # Available masks
        self.available_masks: List[str] = []
        self.mask_display_names: List[str] = []
        
        # Window settings
        self.window_name = "Webcam Filter Preview"
        self.control_window = "Filter Controls"
        self.display_scale = 1.0
        
        # Layout settings
        self.preview_width = 960   # Larger, cleaner preview
        self.preview_height = 720
        self.control_width = 500   # Control panel width
        self.control_height = 720  # Match preview height for single window
        
        # Trackbar values (will be set by callbacks)
        self.param_scale = 200
        self.param_offset_x = 0
        self.param_offset_y = -25
        self.param_yaw = 150
        self.param_pitch = 150
        self.param_roll = 0

        # UI state for dropdown in single-window panel
        self.ui_dropdown_open = False
        self.dropdown_start_idx = 0
        self.max_dropdown_items = 10
        self.dropdown_item_rects: List[Tuple[int, int, int, int, int]] = []  # (x1,y1,x2,y2,mask_index)
        self.dropdown_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.panel_x_offset = self.preview_width
        self.panel_y_offset = 0

        # Custom sliders in panel (side-by-side UI)
        self.controls_scroll = 0
        self.controls_max_scroll = 0
        self.slider_rects: List[Tuple[int, int, int, int, str]] = []  # (x1,y1,x2,y2,key)
        self.ui_dragging_slider: Optional[str] = None
        self.scroll_up_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.scroll_down_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        # Whole-panel scrolling
        self.panel_scroll = 0
        self.panel_content_height = self.control_height
        self.panel_scroll_up_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.panel_scroll_down_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.scrollbar_track_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.scrollbar_thumb_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.ui_dragging_scrollbar: bool = False
        self.drag_scroll_offset: int = 0
        # Slider container viewport (for wheel-into-container scrolling)
        self.slider_container_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        
        # Hand gesture control
        self.enable_hand_control = 0
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.last_gesture_time = 0.0
        self.gesture_cooldown = 0.8  # seconds
        self.hand_overlay_text = ""
        
    def initialize(self) -> bool:
        """Initialize camera and filter engine."""
        print("🎥 Inisialisasi kamera...")
        
        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except:
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            print("❌ Gagal membuka kamera!")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari kamera!")
            return False
            
        print(f"✅ Kamera ready: {frame.shape[1]}x{frame.shape[0]}")
        
        # Initialize filter engine
        print("🔧 Inisialisasi FilterEngine...")
        try:
            self.engine = FilterEngine(masks_folder=self.masks_folder, det_scale=0.75)
            print("✅ FilterEngine ready")
        except Exception as e:
            print(f"❌ Gagal membuat FilterEngine: {e}")
            return False
        
        # Scan for available masks
        self.scan_masks()
        
        # Initialize hand detection if available
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
            
        return True
    
    def scan_masks(self):
        """Scan masks folder for available mask files."""
        self.available_masks = ["[No Mask]"]
        self.mask_display_names = ["No Mask"]
        
        if not self.masks_folder or not os.path.isdir(self.masks_folder):
            print("⚠️ No masks folder found")
            return
        
        # Look for PNG files
        mask_files = glob.glob(os.path.join(self.masks_folder, "*.png"))
        mask_files.extend(glob.glob(os.path.join(self.masks_folder, "*.jpg")))
        
        for mask_path in sorted(mask_files):
            filename = os.path.basename(mask_path)
            self.available_masks.append(filename)
            # Clean display name
            display_name = os.path.splitext(filename)[0]
            self.mask_display_names.append(display_name)
        
        print(f"📁 Found {len(self.available_masks)-1} mask(s)")
        
    def create_gui(self):
        """Create GUI in a single window with controls alongside preview."""
        # Single window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.preview_width + self.control_width, self.preview_height)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        # Note: Using custom sliders in the side panel (no OpenCV trackbars)

    def create_control_panel(self) -> np.ndarray:
        """Create detailed control panel with all information (scrollable)."""
        # Render to a tall offscreen content, then crop to viewport
        content_h = max(self.control_height * 2, 1500)
        content = np.ones((content_h, self.control_width, 3), dtype=np.uint8) * 245

        y = 25
        x_margin = 20

        # Header
        cv2.rectangle(content, (0, 0), (self.control_width, 60), (50, 50, 50), -1)
        cv2.putText(content, "WEBCAM FILTER CONTROLS", (x_margin, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        y = 80

        # Current mask
        cv2.putText(content, "CURRENT MASK", (x_margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        y += 35

        mask_text = self.mask_display_names[self.current_mask_idx] if self.current_mask_idx < len(self.mask_display_names) else "None"
        if len(mask_text) > 25:
            mask_text = mask_text[:22] + "..."

        cv2.rectangle(content, (x_margin, y - 28), (self.control_width - x_margin, y + 8), (70, 130, 220), -1)
        cv2.rectangle(content, (x_margin, y - 28), (self.control_width - x_margin, y + 8), (40, 40, 40), 3)
        cv2.putText(content, mask_text, (x_margin + 15, y - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        y += 45

        # Mask dropdown UI
        cv2.putText(content, "MASK SELECT (Dropdown)", (x_margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2)
        y += 28

        # Collapsed dropdown box
        box_h = 32
        box_w = self.control_width - 2 * x_margin
        box_x1 = x_margin
        box_y1 = y
        box_x2 = box_x1 + box_w
        box_y2 = box_y1 + box_h
        self.dropdown_rect = (box_x1, box_y1, box_x2, box_y2)

        cv2.rectangle(content, (box_x1, box_y1), (box_x2, box_y2), (70, 130, 220), -1)
        cv2.rectangle(content, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), 2)

        current_text = self.mask_display_names[self.current_mask_idx] if self.current_mask_idx < len(self.mask_display_names) else "None"
        if len(current_text) > 28:
            current_text = current_text[:25] + "..."
        cv2.putText(content, current_text, (box_x1 + 10, box_y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Dropdown arrow indicator
        arrow_x = box_x2 - 20
        arrow_y = box_y1 + box_h // 2
        if self.ui_dropdown_open:
            pts = np.array([[arrow_x - 8, arrow_y - 3], [arrow_x + 8, arrow_y - 3], [arrow_x, arrow_y + 7]], np.int32)
        else:
            pts = np.array([[arrow_x - 8, arrow_y + 3], [arrow_x + 8, arrow_y + 3], [arrow_x, arrow_y - 7]], np.int32)
        cv2.fillPoly(content, [pts], (255, 255, 255))

        y = box_y2 + 6

        # Expanded dropdown list
        self.dropdown_item_rects = []
        if self.ui_dropdown_open:
            max_items = self.max_dropdown_items
            start = self.dropdown_start_idx
            end = min(start + max_items, len(self.mask_display_names))
            item_h = 26
            list_h = (end - start) * item_h
            cv2.rectangle(content, (box_x1, y), (box_x2, y + list_h), (240, 240, 240), -1)
            cv2.rectangle(content, (box_x1, y), (box_x2, y + list_h), (120, 120, 120), 1)

            yy = y
            for i in range(start, end):
                item_text = self.mask_display_names[i]
                if len(item_text) > 28:
                    item_text = item_text[:25] + "..."
                # Highlight selection
                if i == self.current_mask_idx:
                    cv2.rectangle(content, (box_x1 + 2, yy + 2), (box_x2 - 2, yy + item_h - 2), (220, 240, 255), -1)
                    color = (0, 80, 180)
                    weight = 2
                else:
                    color = (60, 60, 60)
                    weight = 1

                cv2.putText(content, item_text, (box_x1 + 10, yy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, weight)
                # store rect for click handling
                self.dropdown_item_rects.append((box_x1, yy, box_x2, yy + item_h, i))
                yy += item_h

            y = yy + 8

        # ===== SLIDER CONTROLS (custom UI) =====
        cv2.putText(content, "SLIDER CONTROLS", (x_margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2)
        y += 24

        # Container for sliders (with scroll arrows)
        container_x1 = x_margin
        container_y1 = y
        container_w = self.control_width - 2 * x_margin
        container_h = 260
        container_x2 = container_x1 + container_w
        container_y2 = container_y1 + container_h

        # Expose container rect for mouse/wheel handling (content coordinates)
        self.slider_container_rect = (container_x1, container_y1, container_x2, container_y2)

        cv2.rectangle(content, (container_x1, container_y1), (container_x2, container_y2), (230, 230, 230), -1)
        cv2.rectangle(content, (container_x1, container_y1), (container_x2, container_y2), (160, 160, 160), 1)

        # Scroll arrows
        self.scroll_up_rect = (container_x2 - 26, container_y1 + 6, container_x2 - 6, container_y1 + 26)
        self.scroll_down_rect = (container_x2 - 26, container_y2 - 26, container_x2 - 6, container_y2 - 6)
        ux1, uy1, ux2, uy2 = self.scroll_up_rect
        dx1, dy1, dx2, dy2 = self.scroll_down_rect
        cv2.rectangle(content, (ux1, uy1), (ux2, uy2), (200, 200, 200), -1)
        cv2.rectangle(content, (ux1, uy1), (ux2, uy2), (120, 120, 120), 1)
        cv2.rectangle(content, (dx1, dy1), (dx2, dy2), (200, 200, 200), -1)
        cv2.rectangle(content, (dx1, dy1), (dx2, dy2), (120, 120, 120), 1)
        # Up arrow
        pts_up = np.array([[ux1 + 6, uy2 - 6], [ux2 - 6, uy2 - 6], [(ux1 + ux2)//2, uy1 + 6]], np.int32)
        cv2.fillPoly(content, [pts_up], (80, 80, 80))
        # Down arrow
        pts_dn = np.array([[(dx1 + dx2)//2, dy2 - 6], [dx1 + 6, dy1 + 6], [dx2 - 6, dy1 + 6]], np.int32)
        cv2.fillPoly(content, [pts_dn], (80, 80, 80))

        # Define sliders list
        slider_defs = [
            {"key": "scale", "label": "Scale %", "min": 50, "max": 400, "val": self.param_scale},
            {"key": "offset_x", "label": "Offset X", "min": -200, "max": 200, "val": self.param_offset_x},
            {"key": "offset_y", "label": "Offset Y", "min": -200, "max": 200, "val": self.param_offset_y},
            {"key": "yaw", "label": "Yaw %", "min": 0, "max": 300, "val": self.param_yaw},
            {"key": "pitch", "label": "Pitch %", "min": 0, "max": 300, "val": self.param_pitch},
            {"key": "roll", "label": "Roll Deg", "min": -180, "max": 180, "val": self.param_roll},
        ]

        # Optional toggles (pause, info) as switches
        switches = [
            {"key": "pause", "label": "Pause", "val": 1 if self.paused else 0},
            {"key": "show_info", "label": "Show Info", "val": 1 if self.show_info else 0},
            {"key": "hand_control", "label": "Hand Control", "val": 1 if self.enable_hand_control else 0},
        ]

        # Compute scroll limits
        per_slider_h = 40
        total_content_h = len(slider_defs) * per_slider_h + len(switches) * 34
        visible_h = container_h - 14
        self.controls_max_scroll = max(0, total_content_h - visible_h)
        scroll = int(self.controls_scroll)
        scroll = max(0, min(scroll, self.controls_max_scroll))
        self.controls_scroll = scroll

        # Draw sliders within container (respect bounds)
        self.slider_rects = []
        cur_y = container_y1 + 10 - scroll
        bar_margin_x = container_x1 + 10
        # leave room for right-side arrows and value text
        bar_w = container_w - 80
        visible_top = container_y1
        visible_bottom = container_y2
        for sd in slider_defs:
            # Skip if outside visible area
            if cur_y + per_slider_h < visible_top or cur_y > visible_bottom - 10:
                cur_y += per_slider_h
                continue
            # Label
            cv2.putText(content, sd["label"], (bar_margin_x, cur_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
            # Bar rect
            bar_y = cur_y + 22
            bar_x1 = bar_margin_x
            bar_x2 = bar_x1 + bar_w
            # Slightly thicker bar for a cleaner look
            cv2.rectangle(content, (bar_x1, bar_y), (bar_x2, bar_y + 10), (210, 210, 210), -1)
            cv2.rectangle(content, (bar_x1, bar_y), (bar_x2, bar_y + 10), (150, 150, 150), 1)
            # Knob position
            rng = sd["max"] - sd["min"]
            norm = 0.0 if rng == 0 else (sd["val"] - sd["min"]) / rng
            knob_x = int(bar_x1 + norm * (bar_w - 14))
            cv2.rectangle(content, (knob_x, bar_y - 2), (knob_x + 14, bar_y + 12), (70, 130, 220), -1)
            cv2.rectangle(content, (knob_x, bar_y - 2), (knob_x + 14, bar_y + 12), (40, 40, 40), 1)
            # Value text to the right of the bar for neatness
            val = sd["val"]
            if sd["key"] in ("offset_x", "offset_y"):
                val_str = f"{val:+d}"
            elif sd["key"] == "roll":
                val_str = f"{val:+d}°"
            else:
                val_str = f"{val}%" if sd["key"] in ("scale", "yaw", "pitch") else str(val)
            cv2.putText(content, val_str, (bar_x2 + 10, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
            # Save rect for interaction
            self.slider_rects.append((bar_x1, bar_y, bar_x2, bar_y + 10, sd["key"]))
            cur_y += per_slider_h

        # Draw switches (toggle buttons)
        sw_y = cur_y + 6
        for sw in switches:
            if sw_y + 28 < visible_top or sw_y > visible_bottom - 6:
                sw_y += 34
                continue
            cv2.putText(content, sw["label"], (bar_margin_x, sw_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
            box_x1 = bar_margin_x
            box_y1 = sw_y + 18
            box_x2 = box_x1 + 60
            box_y2 = box_y1 + 20
            cv2.rectangle(content, (box_x1, box_y1), (box_x2, box_y2), (200, 200, 200), -1)
            cv2.rectangle(content, (box_x1, box_y1), (box_x2, box_y2), (140, 140, 140), 1)
            if sw["val"]:
                cv2.rectangle(content, (box_x1 + 2, box_y1 + 2), (box_x2 - 2, box_y2 - 2), (70, 130, 220), -1)
            # store interaction rect as a slider key as well
            self.slider_rects.append((box_x1, box_y1, box_x2, box_y2, sw["key"]))
            sw_y += 34

        # Advance y past the container to avoid overlaps
        y = container_y2 + 20

        # Current parameters
        y += 15
        cv2.line(content, (x_margin, y), (self.control_width - x_margin, y), (200, 200, 200), 2)
        y += 25

        cv2.putText(content, "CURRENT SETTINGS", (x_margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2)
        y += 28

        if self.engine:
            params = [
                f"Scale: {self.engine.manual_scale_percent}%",
                f"Offset X: {self.engine.offset_x:+d}",
                f"Offset Y: {self.engine.offset_y:+d}",
                f"Yaw: {self.engine.yaw_percent}%",
                f"Pitch: {self.engine.pitch_percent}%",
                f"Roll: {self.engine.roll_offset:+.0f}°",
            ]

            for param in params:
                cv2.rectangle(content, (x_margin, y - 18), (self.control_width - x_margin, y + 4), (250, 250, 250), -1)
                cv2.rectangle(content, (x_margin, y - 18), (self.control_width - x_margin, y + 4), (200, 200, 200), 1)
                cv2.putText(content, param, (x_margin + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
                y += 26

        # Instructions
        y += 15
        cv2.line(content, (x_margin, y), (self.control_width - x_margin, y), (200, 200, 200), 2)
        y += 25

        cv2.putText(content, "INSTRUCTIONS", (x_margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2)
        y += 28

        instructions = [
            ("Slider Controls:", True),
            ("Mask - Select filter mask", False),
            ("Scale % - Adjust mask size", False),
            ("Offset X/Y - Position (200=center)", False),
            ("Yaw/Pitch - Rotation intensity", False),
            ("Roll Deg - Tilt angle (180=center)", False),
            ("", False),
            ("Hand Gestures (enable 'Hand Control'):", True),
            ("1) Aktifkan toggle 'Hand Control'", False),
            ("2) Arahkan telapak tangan ke kamera (skeleton muncul)", False),
            ("3) Gestur: Palm → Clear, V → Next, Index → Prev", False),
            ("4) Ada jeda ~0.8s antar aksi (cooldown)", False),
            ("5) Pastikan tangan terlihat jelas (cahaya cukup)", False),
            ("", False),
            ("Keyboard Shortcuts:", True),
            ("ESC - Quit application", False),
            ("R - Reset all parameters", False),
            ("", False),
            ("Status:", True),
            (f"FPS: {self.fps:.1f}", False),
            ("Pause: " + ("YES" if self.paused else "NO"), False),
        ]

        for text, is_header in instructions:
            if text == "":
                y += 8
                continue

            if is_header:
                cv2.putText(content, text, (x_margin + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2)
                y += 24
            else:
                cv2.putText(content, text, (x_margin + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 80), 1)
                y += 22

            # Finalize content height and crop to viewport
            self.panel_content_height = min(content_h, y + 20)
            max_scroll = max(0, self.panel_content_height - self.control_height)
            self.panel_scroll = max(0, min(self.panel_scroll, max_scroll))
            start = self.panel_scroll
            end = start + self.control_height
            panel = content[start:end, :, :].copy()

            # Draw overall panel scroll buttons on viewport
            up_rect = (self.control_width - 24, 6, self.control_width - 6, 24)
            dn_rect = (self.control_width - 24, self.control_height - 24, self.control_width - 6, self.control_height - 6)
            self.panel_scroll_up_rect = up_rect
            self.panel_scroll_down_rect = dn_rect
            ux1, uy1, ux2, uy2 = up_rect
            dx1, dy1, dx2, dy2 = dn_rect
            cv2.rectangle(panel, (ux1, uy1), (ux2, uy2), (210, 210, 210), -1)
            cv2.rectangle(panel, (ux1, uy1), (ux2, uy2), (120, 120, 120), 1)
            cv2.rectangle(panel, (dx1, dy1), (dx2, dy2), (210, 210, 210), -1)
            cv2.rectangle(panel, (dx1, dy1), (dx2, dy2), (120, 120, 120), 1)
            # Up triangle
            pts_up = np.array([[ux1 + 6, uy2 - 6], [ux2 - 6, uy2 - 6], [(ux1 + ux2)//2, uy1 + 6]], np.int32)
            cv2.fillPoly(panel, [pts_up], (80, 80, 80))
            # Down triangle
            pts_dn = np.array([[(dx1 + dx2)//2, dy2 - 6], [dx1 + 6, dy1 + 6], [dx2 - 6, dy1 + 6]], np.int32)
            cv2.fillPoly(panel, [pts_dn], (80, 80, 80))

            # Draw vertical scrollbar track + thumb
            track_x1 = self.control_width - 10
            track_x2 = self.control_width - 6
            track_y1 = 26
            track_y2 = self.control_height - 26
            self.scrollbar_track_rect = (track_x1, track_y1, track_x2, track_y2)
            cv2.rectangle(panel, (track_x1, track_y1), (track_x2, track_y2), (190, 190, 190), -1)
            cv2.rectangle(panel, (track_x1, track_y1), (track_x2, track_y2), (120, 120, 120), 1)

            max_scroll = max(0, self.panel_content_height - self.control_height)
            track_h = track_y2 - track_y1
            # Thumb height proportional to visible area
            thumb_h = max(40, int(self.control_height * track_h / max(self.panel_content_height, 1)))
            # Thumb position based on current scroll
            thumb_y = track_y1 if max_scroll == 0 else track_y1 + int(self.panel_scroll * (track_h - thumb_h) / max_scroll)
            thumb_x1 = track_x1 + 1
            thumb_x2 = track_x2 - 1
            thumb_y1 = thumb_y
            thumb_y2 = thumb_y + thumb_h
            self.scrollbar_thumb_rect = (thumb_x1, thumb_y1, thumb_x2, thumb_y2)
            cv2.rectangle(panel, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (140, 140, 140), -1)
            cv2.rectangle(panel, (thumb_x1, thumb_y1), (thumb_x2, thumb_y2), (100, 100, 100), 1)

            return panel
    def on_mask_change(self, val):
        """Called when mask selection changes."""
        self.current_mask_idx = val
        if val == 0:
            self.clear_mask()
        else:
            if val < len(self.available_masks):
                mask_name = self.available_masks[val]
                self.load_mask(mask_name)
    
    def on_scale_change(self, val):
        """Called when scale slider changes."""
        self.param_scale = max(50, val)
        if self.engine:
            self.engine.set_manual_scale_percent(self.param_scale)
    
    def on_offset_x_change(self, val):
        """Called when offset X slider changes (200 = center)."""
        self.param_offset_x = val - 200
        if self.engine:
            self.engine.set_offset_x(self.param_offset_x)
    
    def on_offset_y_change(self, val):
        """Called when offset Y slider changes (200 = center)."""
        self.param_offset_y = val - 200
        if self.engine:
            self.engine.set_offset_y(self.param_offset_y)
    
    def on_yaw_change(self, val):
        """Called when yaw slider changes."""
        self.param_yaw = val
        if self.engine:
            self.engine.set_yaw_percent(val)
    
    def on_pitch_change(self, val):
        """Called when pitch slider changes."""
        self.param_pitch = val
        if self.engine:
            self.engine.set_pitch_percent(val)

    def on_roll_change(self, val):
        """Called when roll slider changes."""
        self.param_roll = val - 180
        if self.engine:
            self.engine.set_roll_offset(self.param_roll)

    def on_pause_change(self, val):
        """Called when pause toggle changes."""
        self.paused = val

    def on_info_change(self, val):
        """Called when show info toggle changes."""
        self.show_info = val

    def _set_param_by_key(self, key: str, val: int):
        """Helper to set parameter by key and propagate to engine."""
        if key == "scale":
            self.param_scale = max(50, min(400, val))
            if self.engine:
                self.engine.set_manual_scale_percent(self.param_scale)
        elif key == "offset_x":
            self.param_offset_x = max(-200, min(200, val))
            if self.engine:
                self.engine.set_offset_x(self.param_offset_x)
        elif key == "offset_y":
            self.param_offset_y = max(-200, min(200, val))
            if self.engine:
                self.engine.set_offset_y(self.param_offset_y)
        elif key == "yaw":
            self.param_yaw = max(0, min(300, val))
            if self.engine:
                self.engine.set_yaw_percent(self.param_yaw)
        elif key == "pitch":
            self.param_pitch = max(0, min(300, val))
            if self.engine:
                self.engine.set_pitch_percent(self.param_pitch)
        elif key == "roll":
            self.param_roll = max(-180, min(180, val))
            if self.engine:
                self.engine.set_roll_offset(self.param_roll)

    def on_mouse(self, event, x, y, flags, param):
        """Mouse handler for single-window UI (dropdown + sliders)."""
        # Map mouse to panel coordinates
        panel_x = x - self.panel_x_offset
        panel_y = y - self.panel_y_offset
        # Map to content coordinates (for stored rects)
        content_y = panel_y + self.panel_scroll

        # Robust mouse wheel handling (process before generic returns)
        wheel_event = getattr(cv2, 'EVENT_MOUSEWHEEL', None)
        if wheel_event is not None and event == wheel_event:
            # Decode signed wheel delta from flags high word (Windows HighGUI)
            delta = (flags >> 16) & 0xFFFF
            if delta & 0x8000:
                delta -= 0x10000
            # Scroll amount per wheel tick
            step = 80 if abs(delta) >= 120 else 40

            # If wheel inside slider container viewport, scroll inner controls
            cx1, cy1, cx2, cy2 = self.slider_container_rect
            # Compare against content coordinates for Y, panel for X
            if cx1 <= panel_x <= cx2 and cy1 <= content_y <= cy2:
                self.controls_scroll = max(0, min(self.controls_max_scroll, self.controls_scroll - step if delta > 0 else self.controls_scroll + step))
            else:
                max_scroll = max(0, self.panel_content_height - self.control_height)
                self.panel_scroll = max(0, min(max_scroll, self.panel_scroll - step if delta > 0 else self.panel_scroll + step))
            return

        # Handle dragging for sliders
        if event == cv2.EVENT_MOUSEMOVE and self.ui_dragging_slider:
            # Find the rect for this slider (bar) to compute value
            key = self.ui_dragging_slider
            for x1, y1, x2, y2, k in self.slider_rects:
                if k == key:
                    if x1 <= panel_x <= x2 and y1 <= content_y <= y2:
                        # Map to value range
                        if key in ("scale", "offset_x", "offset_y", "yaw", "pitch", "roll"):
                            if key == "scale":
                                vmin, vmax = 50, 400
                            elif key == "offset_x" or key == "offset_y":
                                vmin, vmax = -200, 200
                            elif key == "yaw" or key == "pitch":
                                vmin, vmax = 0, 300
                            else:
                                vmin, vmax = -180, 180
                            norm = (panel_x - x1) / float(max(1, (x2 - x1)))
                            val = int(vmin + norm * (vmax - vmin))
                            self._set_param_by_key(key, val)
                    break
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.ui_dragging_slider = None
            self.ui_dragging_scrollbar = False
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            # Non-click events already handled above (move/wheel). Ignore others.
            return

        if panel_x < 0 or panel_x >= self.control_width:
            # clicked on preview area
            # if dropdown is open, clicking outside closes it
            if self.ui_dropdown_open:
                self.ui_dropdown_open = False
            return

        # Whole-panel scroll buttons
        ux1, uy1, ux2, uy2 = self.panel_scroll_up_rect
        dx1, dy1, dx2, dy2 = self.panel_scroll_down_rect
        if ux1 <= panel_x <= ux2 and uy1 <= panel_y <= uy2:
            self.panel_scroll = max(0, self.panel_scroll - 80)
            return
        if dx1 <= panel_x <= dx2 and dy1 <= panel_y <= dy2:
            max_scroll = max(0, self.panel_content_height - self.control_height)
            self.panel_scroll = min(max_scroll, self.panel_scroll + 80)
            return

        # Scrollbar thumb drag
        tx1, ty1, tx2, ty2 = self.scrollbar_thumb_rect
        sx1, sy1, sx2, sy2 = self.scrollbar_track_rect
        max_scroll = max(0, self.panel_content_height - self.control_height)
        track_h = sy2 - sy1
        thumb_h = ty2 - ty1
        if event == cv2.EVENT_LBUTTONDOWN and tx1 <= panel_x <= tx2 and ty1 <= panel_y <= ty2:
            self.ui_dragging_scrollbar = True
            self.drag_scroll_offset = panel_y - ty1
            return
        if event == cv2.EVENT_MOUSEMOVE and self.ui_dragging_scrollbar:
            new_thumb_top = panel_y - self.drag_scroll_offset
            new_thumb_top = max(sy1, min(sy2 - thumb_h, new_thumb_top))
            # Map back to scroll
            self.panel_scroll = 0 if max_scroll == 0 else int((new_thumb_top - sy1) * max_scroll / max(1, (track_h - thumb_h)))
            return
        # Click in scrollbar track moves scroll position
        if event == cv2.EVENT_LBUTTONDOWN and sx1 <= panel_x <= sx2 and sy1 <= panel_y <= sy2:
            rel = (panel_y - sy1) / float(max(1, track_h))
            self.panel_scroll = int(rel * max_scroll)
            return

        # Mouse wheel scrolling
        if event == getattr(cv2, 'EVENT_MOUSEWHEEL', None):
            delta_up = (flags > 0)
            if delta_up:
                self.panel_scroll = max(0, self.panel_scroll - 80)
            else:
                self.panel_scroll = min(max_scroll, self.panel_scroll + 80)
            return

        # Check dropdown box click
        x1, y1, x2, y2 = self.dropdown_rect
        if x1 <= panel_x <= x2 and y1 <= content_y <= y2:
            # toggle open/close
            self.ui_dropdown_open = not self.ui_dropdown_open
            return

        # If open, check item clicks
        if self.ui_dropdown_open:
            for ix1, iy1, ix2, iy2, mask_idx in self.dropdown_item_rects:
                if ix1 <= panel_x <= ix2 and iy1 <= content_y <= iy2:
                    self.current_mask_idx = mask_idx
                    if mask_idx == 0:
                        self.clear_mask()
                    else:
                        if mask_idx < len(self.available_masks):
                            self.load_mask(self.available_masks[mask_idx])
                    # close dropdown after selection
                    self.ui_dropdown_open = False
                    break

        # Check sliders/switches clicks
        for sx1, sy1, sx2, sy2, key in self.slider_rects:
            if sx1 <= panel_x <= sx2 and sy1 <= content_y <= sy2:
                if key in ("pause", "show_info"):
                    # toggle switches
                    if key == "pause":
                        self.paused = 0 if self.paused else 1
                    else:
                        self.show_info = 0 if self.show_info else 1
                elif key == "hand_control":
                    self.enable_hand_control = 0 if self.enable_hand_control else 1
                else:
                    # start dragging for continuous sliders
                    self.ui_dragging_slider = key
                    # Also set initial value on click
                    norm = (panel_x - sx1) / float(max(1, (sx2 - sx1)))
                    if key == "scale":
                        self._set_param_by_key(key, int(50 + norm * (400 - 50)))
                    elif key == "offset_x":
                        self._set_param_by_key(key, int(-200 + norm * (400)))
                    elif key == "offset_y":
                        self._set_param_by_key(key, int(-200 + norm * (400)))
                    elif key == "yaw":
                        self._set_param_by_key(key, int(0 + norm * (300)))
                    elif key == "pitch":
                        self._set_param_by_key(key, int(0 + norm * (300)))
                    elif key == "roll":
                        self._set_param_by_key(key, int(-180 + norm * (360)))
                break
        
    def load_mask(self, mask_name: str) -> bool:
        """Load mask by filename."""
        if not self.engine or mask_name == "[No Mask]":
            return False
            
        try:
            ok = self.engine.set_mask(mask_name)
            if ok:
                self.current_mask_name = mask_name
                print(f"✅ Mask loaded: {mask_name}")
                return True
            else:
                print(f"⚠️ Gagal load mask: {mask_name}")
                return False
        except Exception as e:
            print(f"❌ Error loading mask {mask_name}: {e}")
            return False
            
    def clear_mask(self):
        """Clear current mask."""
        if self.engine:
            self.engine.clear_mask()
            self.current_mask_name = "None"
            print("🔄 Mask cleared")

    def select_next_mask(self):
        if len(self.available_masks) == 0:
            return
        self.current_mask_idx = (self.current_mask_idx + 1) % len(self.available_masks)
        if self.current_mask_idx == 0:
            self.clear_mask()
        else:
            self.load_mask(self.available_masks[self.current_mask_idx])

    def select_prev_mask(self):
        if len(self.available_masks) == 0:
            return
        self.current_mask_idx = (self.current_mask_idx - 1) % len(self.available_masks)
        if self.current_mask_idx == 0:
            self.clear_mask()
        else:
            self.load_mask(self.available_masks[self.current_mask_idx])

    def select_mask_by_number(self, num: int):
        # map 1..N to mask index, 0 reserved for [No Mask]
        if num <= 0:
            self.current_mask_idx = 0
            self.clear_mask()
            return
        idx = min(num, len(self.available_masks) - 1)
        self.current_mask_idx = idx
        if idx == 0:
            self.clear_mask()
        else:
            self.load_mask(self.available_masks[idx])
    
    def reset_parameters(self):
        """Reset all parameters to defaults."""
        if not self.engine:
            return
        
        self.engine.reset_to_defaults()
        
        # Reset internal state for custom sliders
        self.param_scale = 200
        self.param_offset_x = 0
        self.param_offset_y = -25
        self.param_yaw = 150
        self.param_pitch = 150
        self.param_roll = 0
        
        print("🔄 Parameters reset to defaults")
    
    def update_fps(self, frame_time: float):
        """Update FPS calculation."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        if len(self.frame_times) > 0:
            self.fps = len(self.frame_times) / sum(self.frame_times)
            
    def draw_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw minimal info overlay on frame."""
        if not self.show_info:
            return frame
            
        h, w = frame.shape[:2]
        
        # Top-left corner info box (expands if hand control enabled)
        box_w = 240
        box_h = 80 if not self.enable_hand_control else 150
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        status = "PAUSED" if self.paused else "LIVE"
        status_color = (0, 165, 255) if self.paused else (0, 255, 0)
        cv2.putText(frame, status, (20, 65), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 2)

        # Hand control legend/help
        if self.enable_hand_control:
            base_y = 95
            # Status or last gesture
            txt = self.hand_overlay_text if self.hand_overlay_text else "HandCtrl: ON"
            cv2.putText(frame, txt, (20, base_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            # Compact legend
            cv2.putText(frame, "Open is Clear", (20, base_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, "2/V is Next", (20, base_y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, "one/Index is Prev", (20, base_y + 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def resize_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit display window while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        
        # Calculate scaling to fit preview window
        scale_w = self.preview_width / w
        scale_h = self.preview_height / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create canvas and center the frame
        canvas = np.zeros((self.preview_height, self.preview_width, 3), dtype=np.uint8)
        y_offset = (self.preview_height - new_h) // 2
        x_offset = (self.preview_width - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
        
    def run(self):
        """Main application loop."""
        if not self.initialize():
            print("❌ Initialization failed!")
            return
        
        # Create GUI
        self.create_gui()
            
        print("\n" + "="*60)
        print("🎬 WEBCAM FILTER - GUI MODE")
        print("="*60)
        print("📺 Single Window - Preview + Controls bersebelahan")
        print("Press 'R' to reset, ESC to quit")
        print("="*60 + "\n")
        
        self.running = True
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ Failed to read frame")
                    break
                
                # Process frame (if not paused)
                if not self.paused and self.engine:
                    try:
                        frame = self.engine.process_frame(frame)
                    except Exception as e:
                        print(f"⚠️ Processing error: {e}")

                # Hand gesture control (runs on original frame to avoid artifacts)
                if self.enable_hand_control and self.hands is not None:
                    try:
                        self._process_hand_gestures(frame)
                    except Exception as e:
                        # Avoid spamming errors
                        pass
                
                # Draw info overlay on frame
                frame = self.draw_info(frame)
                
                # Resize frame for display (left side)
                display_frame = self.resize_frame_for_display(frame)

                # Create control panel (right side)
                control_panel = self.create_control_panel()

                # Compose single canvas side-by-side
                combined_h = max(display_frame.shape[0], control_panel.shape[0])
                combined_w = display_frame.shape[1] + control_panel.shape[1]
                combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

                # place preview
                combined[0:display_frame.shape[0], 0:display_frame.shape[1]] = display_frame
                # place panel
                combined[0:control_panel.shape[0], display_frame.shape[1]:display_frame.shape[1]+control_panel.shape[1]] = control_panel

                # Update offsets for mouse handling
                self.panel_x_offset = display_frame.shape[1]
                self.panel_y_offset = 0

                cv2.imshow(self.window_name, combined)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.running = False
                elif key in (ord('r'), ord('R')):  # Reset
                    self.reset_parameters()
                
                # Update FPS
                frame_time = time.time() - frame_start
                self.update_fps(frame_time)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        finally:
            self.cleanup()

    # ===== Hand gesture detection =====
    def _finger_extended(self, lm, tip_idx: int, pip_idx: int) -> bool:
        try:
            return lm[tip_idx].y < lm[pip_idx].y
        except Exception:
            return False

    def _process_hand_gestures(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        h, w = frame.shape[:2]
        # Draw skeleton if present
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                try:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                except Exception:
                    pass

        # Gesture detection with cooldown
        now = time.time()
        if (now - self.last_gesture_time) < self.gesture_cooldown:
            return
        if not res.multi_hand_landmarks:
            self.hand_overlay_text = "HandCtrl: ON"
            return

        lm = res.multi_hand_landmarks[0].landmark
        index = self._finger_extended(lm, 8, 6)
        middle = self._finger_extended(lm, 12, 10)
        ring = self._finger_extended(lm, 16, 14)
        little = self._finger_extended(lm, 20, 18)

        # Palm open => CLEAR; V sign => NEXT; Pointing (index only) => PREV
        if index and middle and ring and little:
            self.clear_mask()
            self.hand_overlay_text = "Gesture: CLEAR (Palm)"
            self.last_gesture_time = now
        elif index and middle and (not ring) and (not little):
            self.select_next_mask()
            self.hand_overlay_text = "Gesture: NEXT (V)"
            self.last_gesture_time = now
        elif index and (not middle) and (not ring) and (not little):
            self.select_prev_mask()
            self.hand_overlay_text = "Gesture: PREV (Index)"
            self.last_gesture_time = now
            
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        if self.cap:
            self.cap.release()
            
        if self.engine:
            self.engine.close()
            
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local Webcam Filter with GUI")
    parser.add_argument(
        "--masks-folder",
        default=None,
        help="Path ke folder berisi mask files (default: auto-detect 'mask' atau 'masks' folder)"
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Mask file untuk di-load saat startup (e.g., face1.png)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect masks folder if not provided
    masks_folder = args.masks_folder
    if not masks_folder:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for name in ("mask", "masks"):
            candidate = os.path.join(script_dir, name)
            if os.path.isdir(candidate):
                masks_folder = candidate
                print(f"ℹ️  Auto-detected masks folder: {masks_folder}")
                break
    
    # Create and run app
    app = WebcamFilterApp(masks_folder=masks_folder)
    
    # Pre-load mask if specified
    if args.mask:
        # We need to initialize first to load the mask
        if app.initialize():
            app.load_mask(args.mask)
            app.run()
        else:
            print("❌ Failed to initialize app")
            sys.exit(1)
    else:
        app.run()


if __name__ == "__main__":
    main()