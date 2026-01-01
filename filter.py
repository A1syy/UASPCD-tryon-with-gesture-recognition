# import sys, math, cv2, numpy as np, mediapipe as mp
# from PyQt5.QtWidgets import (
#     QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QWidget,
#     QVBoxLayout, QHBoxLayout, QMessageBox, QSlider
# )
# from PyQt5.QtCore import QTimer, Qt
# from PyQt5.QtGui import QImage, QPixmap
# print(">>> Filter Auto Sliders starting up...")
# # ---------------- helper ----------------

# def cvimg_to_qtimg(cv_img):
#     if cv_img is None: return QImage()
#     h, w = cv_img.shape[:2]
#     rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#     return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)


# def load_mask_rgba(path):
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     if img is None: raise FileNotFoundError(path)
#     if img.shape[2] == 3:  # kalau mask tidak ada alpha → tambahkan
#         b,g,r = cv2.split(img)
#         a = np.ones_like(b)*255
#         img = cv2.merge([b,g,r,a])
#     return img


# def clamp(v, a, b):
#     return max(a, min(b, v))


# def rotate_image(img, angle_x=0, angle_y=0, angle_z=0):
#     """ Rotasi 3D sederhana: pitch (X), yaw (Y), roll (Z)
#     Fast-path: jika perubahan sudut sangat kecil, kembalikan gambar tanpa warp untuk performa.
#     """
#     h, w = img.shape[:2]
#     if h == 0 or w == 0:
#         return img

#     # if rotation is negligible, skip expensive warp
#     if abs(angle_x) < 1.0 and abs(angle_y) < 1.0 and abs(angle_z) < 0.5:
#         return img

#     f = 500  # focal length asumsi
#     cx, cy = w//2, h//2

#     # koordinat corner mask
#     pts = np.array([
#         [-w/2, -h/2, 0],
#         [ w/2, -h/2, 0],
#         [ w/2,  h/2, 0],
#         [-w/2,  h/2, 0]
#     ], dtype=np.float32)

#     # rotasi matriks
#     ax, ay, az = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)

#     Rx = np.array([
#         [1, 0, 0],
#         [0, np.cos(ax), -np.sin(ax)],
#         [0, np.sin(ax),  np.cos(ax)]
#     ])
#     Ry = np.array([
#         [ np.cos(ay), 0, np.sin(ay)],
#         [ 0, 1, 0],
#         [-np.sin(ay), 0, np.cos(ay)]
#     ])
#     Rz = np.array([
#         [np.cos(az), -np.sin(az), 0],
#         [np.sin(az),  np.cos(az), 0],
#         [0, 0, 1]
#     ])

#     R = Rz @ Ry @ Rx  # urutan: roll → yaw → pitch

#     pts3d = pts @ R.T
#     # perspektif proyeksi sederhana
#     pts2d = pts3d[:, :2] * (f / (f + pts3d[:,2].reshape(-1,1))) + [cx, cy]

#     dst = np.array(pts2d, dtype=np.float32)
#     src = np.array([[0,0],[w,0],[w,h],[0,h]],dtype=np.float32)

#     M = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
#     return warped


# class FilterApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle('Auto + Manual Face Mask Filter (Improved Responsiveness)')

#         self.base_face_h = None
#         self.base_mask_h = None
#         self.base_scale = 1.0
#         self.scale_factor = 1.0

#         # smoothing / previous values untuk mengurangi jitter
#         # NOTE: made defaults more responsive
#         self.smooth = 0.72        # high smoothing when idle
#         self.min_smooth = 0.12    # low smoothing when big/head movement
#         self.prev_cx = None
#         self.prev_cy = None
#         self.prev_yaw = None
#         self.prev_pitch = None
#         self.prev_roll = None

#         # tambahan: state per-face untuk multi-face (maks 4)
#         self.max_faces = 4
#         # dict keyed by detection index -> previous smoothed values
#         self.prev_states = {}          # { idx: {'cx':..., 'cy':..., 'yaw':..., 'pitch':..., 'roll':...} }
#         self.base_face_hs = {}         # { idx: initial_face_h }
#         self.base_scale_map = {}       # { idx: base_scale }
#         self.scale_factor_map = {}     # { idx: current_scale_factor }

#         # batasan mapping otomatis (deg)
#         self.MAX_YAW = 55
#         self.MAX_PITCH = 35

#         # detection scale (process a smaller frame for faster detection)
#         # increased to 0.75 for better accuracy while staying fast
#         self.det_scale = 0.75

#         # --- UI
#         self.video_label = QLabel('Camera feed will appear here')
#         self.video_label.setAlignment(Qt.AlignCenter)
#         self.video_label.setStyleSheet('background-color:#222; color:#eee;')
#         self.video_label.setMinimumSize(800,600)

#         self.load_btn = QPushButton('Load Mask')
#         self.reset_btn = QPushButton('Reset All Sliders')
#         self.start_btn = QPushButton('Start Camera')
#         self.stop_btn = QPushButton('Stop Camera')

#         # slider ukuran (multiplier terhadap automatic scale)
#         self.scale_slider = QSlider(Qt.Horizontal)
#         self.scale_slider.setRange(50, 350)
#         self.scale_slider.setValue(200)

#         # slider vertical offset (px)
#         self.offset_slider = QSlider(Qt.Horizontal)
#         self.offset_slider.setRange(-250, 200)
#         self.offset_slider.setValue(-50)

#         # slider horizontal offset (px)
#         self.offset_x_slider = QSlider(Qt.Horizontal)
#         self.offset_x_slider.setRange(-200, 200)
#         self.offset_x_slider.setValue(0)

#         # slider yaw sensitivity (%) -- now acts as multiplier for automatic yaw
#         self.yaw_slider = QSlider(Qt.Horizontal)
#         self.yaw_slider.setRange(50, 400)
#         self.yaw_slider.setValue(150)   # default more sensitive now

#         # slider pitch sensitivity (%) -- acts as multiplier for automatic pitch
#         self.pitch_slider = QSlider(Qt.Horizontal)
#         self.pitch_slider.setRange(50, 400)
#         self.pitch_slider.setValue(150)  # default more sensitive now

#         # slider roll (degrees) -> still additive because roll computed from eyes is usually stable
#         self.roll_slider = QSlider(Qt.Horizontal)
#         self.roll_slider.setRange(-45, 45)
#         self.roll_slider.setValue(0)

#         btns = QHBoxLayout()
#         btns.addWidget(self.load_btn)
#         btns.addWidget(self.reset_btn)
#         btns.addWidget(self.start_btn)
#         btns.addWidget(self.stop_btn)

#         layout = QVBoxLayout()
#         layout.addWidget(self.video_label)
#         layout.addLayout(btns)
#         layout.addWidget(QLabel("Mask Size (%) -- multiplicative manual adjust (auto driven)"))
#         layout.addWidget(self.scale_slider)
#         layout.addWidget(QLabel("Mask Vertical Offset (px) -- auto anchored to nose, slider adds pixels"))
#         layout.addWidget(self.offset_slider)
#         layout.addWidget(QLabel("Mask Horizontal Offset (px) -- auto anchored to nose, slider adds pixels"))
#         layout.addWidget(self.offset_x_slider)
#         # layout.addWidget(QLabel("Yaw Sensitivity (%) -- multiplier for auto yaw (100% = default)"))
#         # layout.addWidget(self.yaw_slider)
#         # layout.addWidget(QLabel("Pitch Sensitivity (%) -- multiplier for auto pitch (100% = default)"))
#         # layout.addWidget(self.pitch_slider)
#         # layout.addWidget(QLabel("Roll (Tilt degrees) -- additive offset to auto roll"))
#         # layout.addWidget(self.roll_slider)

#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)

#         # state
#         self.mask_img = None
#         self.cap = None
#         self.timer = QTimer(); self.timer.timeout.connect(self.update_frame)
#         self.running = False

#         # face detection (still mediapipe FaceDetection — same technology)
#         self.face_detection = mp.solutions.face_detection.FaceDetection(
#             model_selection=1, min_detection_confidence=0.5
#         )

#         # connect
#         self.load_btn.clicked.connect(self.load_mask)
#         self.reset_btn.clicked.connect(self.reset_sliders)
#         self.start_btn.clicked.connect(self.start_camera)
#         self.stop_btn.clicked.connect(self.stop_camera)

#     def load_mask(self):
#         path, _ = QFileDialog.getOpenFileName(self,'Open Mask','','Images (*.png)')
#         if not path: return
#         try:
#             self.mask_img = load_mask_rgba(path)
#             # reset per-face base maps when a new mask is loaded
#             self.base_face_hs.clear()
#             self.base_scale_map.clear()
#             self.scale_factor_map.clear()
#             self.prev_states.clear()
#             self.base_mask_h = None
#             QMessageBox.information(self,'Loaded',f'Mask loaded: {path}')
#         except Exception as e:
#             QMessageBox.critical(self,'Error',str(e))

#     def reset_sliders(self):
#         """Kembalikan semua slider ke posisi default"""
#         self.scale_slider.setValue(200)
#         self.offset_slider.setValue(-50)
#         self.offset_x_slider.setValue(0)
#         self.yaw_slider.setValue(150)
#         self.pitch_slider.setValue(150)
#         self.roll_slider.setValue(0)

#     def start_camera(self):
#         if self.running: return
#         if self.mask_img is None:
#             QMessageBox.warning(self,'Missing','Load a mask PNG first')
#             return
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             QMessageBox.critical(self,'Camera','Failed to open webcam'); return
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
#         self.running = True
#         # faster timer tick; smoothing/adaptive will handle jitter
#         self.timer.start(15)

#     def stop_camera(self):
#         if not self.running: return
#         self.timer.stop()
#         if self.cap:
#             self.cap.release()
#         self.cap = None
#         self.running = False
#         self.video_label.clear()
#         self.video_label.setText("Camera stopped")
#         self.video_label.setStyleSheet('background-color:#222; color:#eee;')

#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if not ret: return
#         h, w = frame.shape[:2]

#         # process smaller frame for faster detection (works because we use relative coords)
#         small = cv2.resize(frame, (0,0), fx=self.det_scale, fy=self.det_scale)
#         res = self.face_detection.process(cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
#         overlay = np.zeros((h, w, 4), dtype=np.uint8)

#         if res.detections:
#             # process up to self.max_faces detections
#             n_proc = min(self.max_faces, len(res.detections))

#             for idx in range(n_proc):
#                 det = res.detections[idx]
#                 bboxC = det.location_data.relative_bounding_box
#                 x, y, bw, bh = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)

#                 # automatic scale initialization per-face
#                 if idx not in self.base_face_hs:
#                     self.base_face_hs[idx] = bh
#                     mh, mw = self.mask_img.shape[:2]
#                     # set global base_mask_h once if not already set
#                     if self.base_mask_h is None:
#                         self.base_mask_h = mh
#                     # base scale for this face
#                     self.base_scale_map[idx] = bh / mh if mh != 0 else 1.0
#                     self.scale_factor_map[idx] = self.base_scale_map[idx]

#                 # update target scale for this face (smooth per-face)
#                 target_scale = (bh / self.base_face_hs[idx]) * self.base_scale_map[idx]
#                 cur_sf = self.scale_factor_map.get(idx, self.base_scale_map[idx])
#                 cur_sf = 0.88*cur_sf + 0.12*target_scale
#                 self.scale_factor_map[idx] = cur_sf

#                 # --- automatic head pose estimates using FaceDetection keypoints (approximate)
#                 kps = det.location_data.relative_keypoints
#                 try:
#                     r_eye = kps[0]; l_eye = kps[1]; nose = kps[2]
#                     rx, ry = int(r_eye.x * w), int(r_eye.y * h)
#                     lx, ly = int(l_eye.x * w), int(l_eye.y * h)
#                     nx, ny = int(nose.x * w), int(nose.y * h)
#                 except Exception:
#                     nx, ny = x + bw//2, y + bh//2
#                     rx, ry, lx, ly = nx - 10, ny, nx + 10, ny

#                 # roll: angle between eyes
#                 dx = lx - rx
#                 dy = ly - ry
#                 roll_auto = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0.0

#                 # yaw: nose x offset relative to bbox center -> map to degrees (invert sign)
#                 norm_x = (nx - (x + bw/2)) / (bw/2 + 1e-6)
#                 norm_x = clamp(norm_x, -1.0, 1.0)
#                 yaw_auto = -norm_x * self.MAX_YAW   # inverted

#                 # pitch: nose y offset relative to bbox center -> map to degrees (fix direction)
#                 norm_y = (ny - (y + bh/2)) / (bh/2 + 1e-6)
#                 norm_y = clamp(norm_y, -1.0, 1.0)
#                 pitch_auto = norm_y * self.MAX_PITCH   # removed the minus

#                 # per-face adaptive smoothing based on motion magnitude
#                 prev = self.prev_states.get(idx)
#                 if prev is None:
#                     prev = {
#                         'cx': nx,
#                         'cy': ny,
#                         'yaw': yaw_auto,
#                         'pitch': pitch_auto,
#                         'roll': roll_auto
#                     }
#                     self.prev_states[idx] = prev

#                 # compute motion indicators (per-face)
#                 motion_pos = math.hypot(nx - prev['cx'], ny - prev['cy']) / max(1.0, bw)
#                 motion_pose = (abs(yaw_auto - prev['yaw']) / (self.MAX_YAW + 1e-6) + abs(pitch_auto - prev['pitch']) / (self.MAX_PITCH + 1e-6)) * 0.5
#                 motion = clamp(motion_pos + motion_pose, 0.0, 1.0)
#                 # alpha: larger motion -> smaller alpha (less smoothing)
#                 alpha = self.smooth * (1.0 - motion) + self.min_smooth * motion

#                 # exponential smoothing with adaptive alpha (update prev in-place)
#                 prev['cx'] = alpha * prev['cx'] + (1.0 - alpha) * nx
#                 prev['cy'] = alpha * prev['cy'] + (1.0 - alpha) * ny
#                 prev['yaw'] = alpha * prev['yaw'] + (1.0 - alpha) * yaw_auto
#                 prev['pitch'] = alpha * prev['pitch'] + (1.0 - alpha) * pitch_auto
#                 prev['roll'] = alpha * prev['roll'] + (1.0 - alpha) * roll_auto

#                 # final values: automatic (smoothed) multiplied by slider sensitivity (manual)
#                 yaw_sens = self.yaw_slider.value() / 100.0
#                 pitch_sens = self.pitch_slider.value() / 100.0

#                 yaw = prev['yaw'] * yaw_sens
#                 pitch = prev['pitch'] * pitch_sens
#                 roll = prev['roll'] + self.roll_slider.value()

#                 # final position with pixel offsets (shared manual offsets)
#                 manual_scale = self.scale_slider.value() / 100.0
#                 final_scale = self.scale_factor_map[idx] * manual_scale

#                 cx = int(prev['cx']) + self.offset_x_slider.value()
#                 cy = int(prev['cy']) + self.offset_slider.value()

#                 # resize mask
#                 mh, mw = self.mask_img.shape[:2]
#                 new_w = max(1, int(mw * final_scale))
#                 new_h = max(1, int(mh * final_scale))
#                 mask_resized = cv2.resize(self.mask_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#                 # rotate/warp mask according to combined pose
#                 mask_rotated = rotate_image(mask_resized, angle_x=pitch, angle_y=yaw, angle_z=roll)

#                 # compute ROI on frame
#                 x1 = max(0, cx - mask_rotated.shape[1]//2)
#                 y1 = max(0, cy - mask_rotated.shape[0]//2)
#                 x2 = min(w, x1 + mask_rotated.shape[1])
#                 y2 = min(h, y1 + mask_rotated.shape[0])

#                 # crop rotated mask to fit into frame
#                 mask_cropped = mask_rotated[:y2-y1, :x2-x1]

#                 if mask_cropped.size != 0 and mask_cropped.shape[0] > 0 and mask_cropped.shape[1] > 0:
#                     alpha_m = mask_cropped[...,3:] / 255.0
#                     roi = overlay[y1:y2, x1:x2]
#                     if roi.shape[0] == alpha_m.shape[0] and roi.shape[1] == alpha_m.shape[1]:
#                         roi[:] = (roi*(1-alpha_m) + mask_cropped*alpha_m).astype(np.uint8)

#             # cleanup prev_states keys if detections dropped (keep only active indices)
#             active_keys = set(range(n_proc))
#             for k in list(self.prev_states.keys()):
#                 if k not in active_keys:
#                     self.prev_states.pop(k, None)
#                     self.base_face_hs.pop(k, None)
#                     self.base_scale_map.pop(k, None)
#                     self.scale_factor_map.pop(k, None)
        
#         alpha = overlay[...,3:] / 255.0
#         frame = (frame*(1-alpha) + overlay[...,:3]*alpha).astype(np.uint8)

#         qtimg = cvimg_to_qtimg(frame)
#         pix = QPixmap.fromImage(qtimg).scaled(
#             self.video_label.width(), self.video_label.height(),
#             Qt.KeepAspectRatio)
#         self.video_label.setPixmap(pix)


# def main():
#     app=QApplication(sys.argv)
#     win=FilterApp(); win.resize(1000,800); win.show()
#     sys.exit(app.exec_())

# if __name__=="__main__":
#     print(">>> entering main()")
#     main()
