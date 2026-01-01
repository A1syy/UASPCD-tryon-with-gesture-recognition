# filter_ref.py
"""
Headless face-mask filter engine (refactor dari filter.py)

Usage:
    engine = FilterEngine(masks_folder="masks")
    engine.set_mask("face1.png")          # or engine.set_mask_path("/full/path/face1.png")
    out = engine.process_frame(frame_bgr) # frame_bgr = np.ndarray (H,W,3) BGR

Notes:
- No GUI. Returns processed BGR frame.
- Mediapipe instance created once.
- Thread-safe for mask changes via set_mask_path() (uses an RLock).
- Uses greedy matching to maintain identity of faces across frames (prevents mask "switching").
"""
import os
import math
import time
import threading
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import mediapipe as mp


def load_mask_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        # grayscale - expand to RGBA
        b = img
        g = img
        r = img
        a = np.ones_like(b) * 255
        img = np.stack([b, g, r, a], axis=-1)
    elif img.shape[2] == 3:  # kalau mask tidak ada alpha → tambahkan
        b, g, r = cv2.split(img)
        a = np.ones_like(b) * 255
        img = cv2.merge([b, g, r, a])
    return img.astype(np.uint8)


def clamp(v, a, b):
    return max(a, min(b, v))


def rotate_image(img, angle_x=0, angle_y=0, angle_z=0):
    """ Rotasi 3D sederhana: pitch (X), yaw (Y), roll (Z)
    Fast-path: jika perubahan sudut sangat kecil, kembalikan gambar tanpa warp untuk performa.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img

    # if rotation is negligible, skip expensive warp
    if abs(angle_x) < 1.0 and abs(angle_y) < 1.0 and abs(angle_z) < 0.5:
        return img

    f = 500  # focal length asumsi
    cx, cy = w // 2, h // 2

    pts = np.array([
        [-w / 2, -h / 2, 0],
        [ w / 2, -h / 2, 0],
        [ w / 2,  h / 2, 0],
        [-w / 2,  h / 2, 0]
    ], dtype=np.float32)

    ax, ay, az = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(ax), -math.sin(ax)],
        [0, math.sin(ax),  math.cos(ax)]
    ])
    Ry = np.array([
        [ math.cos(ay), 0, math.sin(ay)],
        [ 0, 1, 0],
        [-math.sin(ay), 0, math.cos(ay)]
    ])
    Rz = np.array([
        [math.cos(az), -math.sin(az), 0],
        [math.sin(az),  math.cos(az), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx  # urutan: roll → yaw → pitch

    pts3d = pts @ R.T
    pts2d = pts3d[:, :2] * (f / (f + pts3d[:, 2].reshape(-1, 1))) + [cx, cy]

    dst = np.array(pts2d, dtype=np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped


class FilterEngine:
    def __init__(self,
                 masks_folder: Optional[str] = None,
                 det_scale: float = 0.75,
                 max_faces: int = 4):
        """
        masks_folder: optional folder where masks live; used by set_mask(name)
        det_scale: downscale factor for detection for performance (0.5..1.0)
        """
        self.masks_folder = masks_folder
        self.det_scale = det_scale
        self.max_faces = max_faces

        # Default parameters (mirror GUI defaults)
        self.manual_scale_percent = 200
        self.offset_y = -25
        self.offset_x = 0
        self.yaw_percent = 150
        self.pitch_percent = 150
        self.roll_offset = 0

        # smoothing / per-face previous states (keyed by track_id)
        self.smooth = 0.60
        self.min_smooth = 0.20
        self.prev_states: Dict[int, Dict[str, Any]] = {}
        self.base_face_hs: Dict[int, int] = {}
        self.base_scale_map: Dict[int, float] = {}
        self.scale_factor_map: Dict[int, float] = {}

        # mapping limits
        self.MAX_YAW = 55
        self.MAX_PITCH = 35

        # mask image (RGBA uint8) - start with None
        self.mask_img: Optional[np.ndarray] = None
        self.base_mask_h: Optional[int] = None

        # tracking state
        self.next_track_id = 0
        self.track_last_seen: Dict[int, float] = {}
        self.track_timeout = 1.0  # seconds to keep a track without seeing it

        # thread-safety lock to protect mask loading & state clear from other thread
        self.lock = threading.RLock()

        # Mediapipe face detection instance (single)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    # ---------------- mask control ----------------
    def set_masks_folder(self, folder: str):
        with self.lock:
            self.masks_folder = folder

    def set_mask(self, filename: str) -> bool:
        """Set mask by filename relative to masks_folder (if provided)."""
        if self.masks_folder:
            path = os.path.join(self.masks_folder, filename)
        else:
            path = filename
        return self.set_mask_path(path)

    def set_mask_path(self, path: str) -> bool:
        """Load mask image from absolute/relative path. Thread-safe."""
        try:
            m = load_mask_rgba(path)
        except Exception as e:
            # do not raise here; return False so caller (server) can handle
            print(f"[FilterEngine] set_mask_path: failed to load '{path}': {e}")
            return False

        with self.lock:
            self.mask_img = m
            self.base_mask_h = int(m.shape[0])
            # reset per-face maps so new mask reinitializes
            self.prev_states.clear()
            self.base_face_hs.clear()
            self.base_scale_map.clear()
            self.scale_factor_map.clear()
            self.next_track_id = 0
            self.track_last_seen.clear()
        return True

    def clear_mask(self):
        with self.lock:
            self.mask_img = None
            self.base_mask_h = None
            self.prev_states.clear()
            self.base_face_hs.clear()
            self.base_scale_map.clear()
            self.scale_factor_map.clear()
            self.next_track_id = 0
            self.track_last_seen.clear()

    # ---------------- New: custom mask composition (base + mata + mulut) ----------------
    # These helpers add support to compose 3-layer custom masks and set them as current mask.
    def _find_layer_file(self, kind: str, idx: int) -> Optional[str]:
        """
        Try to find a candidate file for a given layer kind and index.
        kind: 'base' | 'mata' | 'mulut'
        idx: integer id (1..n). If idx <= 0 -> return None
        Returns absolute or relative path string if found, else None.
        """
        if idx is None:
            return None
        try:
            idx = int(idx)
        except Exception:
            return None
        if idx <= 0:
            return None

        # candidate prefixes for each semantic kind (support some synonyms)
        prefixes_map = {
            'base': ['base', 'face'],
            'mata': ['mata', 'eyes', 'eye'],
            'mulut': ['mulut', 'mouth']
        }

        candidates = []
        prefs = prefixes_map.get(kind, [kind])
        for p in prefs:
            candidates.extend([
                f"{p}{idx}.png",
                f"{p}_{idx}.png",
                f"{p}-{idx}.png",
                os.path.join(p, f"{p}{idx}.png"),
                os.path.join(p, f"{p}_{idx}.png")
            ])

        # If masks_folder set, search there first
        search_paths = []
        if self.masks_folder:
            for c in candidates:
                search_paths.append(os.path.join(self.masks_folder, c))
        # Also try candidates relative to current working directory
        for c in candidates:
            search_paths.append(c)

        # Try each candidate
        for p in search_paths:
            try:
                if os.path.isabs(p):
                    if os.path.isfile(p):
                        return p
                else:
                    # allow either relative or absolute
                    if os.path.isfile(p):
                        return p
            except Exception:
                continue

        # Not found
        return None

    def _blend_region_rgba(self, base: np.ndarray, overlay: np.ndarray, x_off: int, y_off: int) -> np.ndarray:
        """
        Blend `overlay` (RGBA uint8) onto `base` (RGBA uint8) at offsets (x_off, y_off).
        Returns modified base (new array).
        Handles clipping if overlay exceeds base bounds.
        """
        bh, bw = base.shape[:2]
        oh, ow = overlay.shape[:2]

        # compute target region on base
        x1 = max(0, x_off)
        y1 = max(0, y_off)
        x2 = min(bw, x_off + ow)
        y2 = min(bh, y_off + oh)

        if x2 <= x1 or y2 <= y1:
            return base  # nothing to blend

        # overlay region coords
        ox1 = x1 - x_off
        oy1 = y1 - y_off
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        base_region = base[y1:y2, x1:x2].astype(np.float32) / 255.0
        over_region = overlay[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0

        alpha_o = over_region[..., 3:4]
        alpha_b = base_region[..., 3:4]

        out_a = alpha_o + alpha_b * (1.0 - alpha_o)
        # avoid division by zero for RGB calculation
        denom = np.clip(out_a, 1e-6, 1.0)

        out_rgb = (over_region[..., :3] * alpha_o + base_region[..., :3] * alpha_b * (1.0 - alpha_o)) / denom

        # write back
        base[y1:y2, x1:x2, :3] = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
        base[y1:y2, x1:x2, 3:4] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)

        return base

    def _compose_layers(self, layers: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Given a list of RGBA uint8 images (some may be None), compose them into a single RGBA image.
        Strategy:
         - Choose canvas size = max width, max height among provided layers (fallback 512x512)
         - Center each layer on canvas and alpha-blend in order provided.
        Returns final RGBA uint8 image or None if no layers.
        """
        valid_layers = [l for l in layers if (l is not None and l.size != 0)]
        if not valid_layers:
            return None

        max_h = max([l.shape[0] for l in valid_layers])
        max_w = max([l.shape[1] for l in valid_layers])

        # fallback minimum reasonable size
        if max_h <= 0:
            max_h = 512
        if max_w <= 0:
            max_w = 512

        canvas = np.zeros((max_h, max_w, 4), dtype=np.uint8)  # fully transparent

        for layer in valid_layers:
            lh, lw = layer.shape[:2]
            x_off = (max_w - lw) // 2
            y_off = (max_h - lh) // 2
            canvas = self._blend_region_rgba(canvas, layer, x_off, y_off)

        return canvas

    def _compose_layers_normalized(self, layers: List[Optional[np.ndarray]], anchor_idx: int = 0):
        """
        Compose layers after normalizing each provided layer to a common canvas defined by
        the anchor layer's size. This keeps relative visual scale consistent across
        combinations (e.g., eyes won't appear smaller when combined with a big base).

        Returns (composed_rgba, anchor_h, anchor_w). If composition fails, returns (None, None, None).
        """
        # choose anchor layer (prefer anchor_idx; fallback to first valid)
        anchor = None
        if 0 <= anchor_idx < len(layers):
            anchor = layers[anchor_idx]
        if anchor is None or anchor.size == 0:
            for l in layers:
                if l is not None and l.size != 0:
                    anchor = l
                    break
        if anchor is None or anchor.size == 0:
            return None, None, None

        ah, aw = int(anchor.shape[0]), int(anchor.shape[1])
        if ah <= 0 or aw <= 0:
            return None, None, None

        canvas = np.zeros((ah, aw, 4), dtype=np.uint8)

        for layer in layers:
            if layer is None or layer.size == 0:
                continue
            lh, lw = layer.shape[:2]
            if lh <= 0 or lw <= 0:
                continue
            # scale to fit within anchor canvas while preserving aspect ratio
            s = min(ah / float(lh), aw / float(lw))
            new_w = max(1, int(round(lw * s)))
            new_h = max(1, int(round(lh * s)))
            if new_w != lw or new_h != lh:
                resized = cv2.resize(layer, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized = layer
            x_off = (aw - new_w) // 2
            y_off = (ah - new_h) // 2
            canvas = self._blend_region_rgba(canvas, resized, x_off, y_off)

        return canvas, ah, aw

    def set_custom_mask_by_paths(self, base_path: Optional[str], mata_path: Optional[str], mulut_path: Optional[str]) -> bool:
        """
        Compose custom mask from full file paths (any may be None).
        Thread-safe and resets tracking state similar to set_mask_path.
        Returns True on success (mask set or cleared if all None), False on error.
        """
        try:
            base_img = load_mask_rgba(base_path) if base_path else None
            mata_img = load_mask_rgba(mata_path) if mata_path else None
            mulut_img = load_mask_rgba(mulut_path) if mulut_path else None
        except Exception as e:
            print(f"[FilterEngine] set_custom_mask_by_paths: failed to load layer: {e}")
            return False

        layers = [base_img, mata_img, mulut_img]

        # Prefer base as anchor if provided, else fall back to next available layer
        anchor_idx = 0 if base_img is not None else (1 if mata_img is not None else 2)
        composed, anchor_h, anchor_w = self._compose_layers_normalized(layers, anchor_idx=anchor_idx)

        if composed is None:
            # no layers => clear mask
            with self.lock:
                self.mask_img = None
                self.base_mask_h = None
                self.prev_states.clear()
                self.base_face_hs.clear()
                self.base_scale_map.clear()
                self.scale_factor_map.clear()
                self.next_track_id = 0
                self.track_last_seen.clear()
            print("[FilterEngine] set_custom_mask_by_paths: no layers provided -> cleared mask")
            return True

        with self.lock:
            self.mask_img = composed
            # Use anchor height for baseline scaling, not the composite canvas if different
            self.base_mask_h = int(anchor_h) if anchor_h is not None else int(composed.shape[0])
            # reset per-face maps so new mask reinitializes
            self.prev_states.clear()
            self.base_face_hs.clear()
            self.base_scale_map.clear()
            self.scale_factor_map.clear()
            self.next_track_id = 0
            self.track_last_seen.clear()
        print("[FilterEngine] set_custom_mask_by_paths: composed mask set (h=%d,w=%d) anchor_h=%s" % (composed.shape[0], composed.shape[1], str(anchor_h)))
        return True

    def set_custom_mask(self, base_id: Optional[int], mata_id: Optional[int], mulut_id: Optional[int]) -> bool:
        """
        Compose and set custom mask from component IDs.
        IDs: int (1..N), 0/None => not used.
        Naming conventions tried: base{n}.png, base_{n}.png, mata{n}.png, eyes{n}.png, mulut{n}.png, mouth{n}.png, etc.
        Returns True if mask set (or cleared when all are none), False on failure.
        """
        # Resolve paths
        base_path = self._find_layer_file('base', base_id) if base_id is not None else None
        mata_path = self._find_layer_file('mata', mata_id) if mata_id is not None else None
        mulut_path = self._find_layer_file('mulut', mulut_id) if mulut_id is not None else None

        if base_id and not base_path:
            print(f"[FilterEngine] set_custom_mask: base id {base_id} not found (tried in masks_folder)")
        if mata_id and not mata_path:
            print(f"[FilterEngine] set_custom_mask: mata id {mata_id} not found")
        if mulut_id and not mulut_path:
            print(f"[FilterEngine] set_custom_mask: mulut id {mulut_id} not found")

        # If no paths found and all ids are empty => clear mask
        if not base_path and not mata_path and not mulut_path:
            with self.lock:
                self.mask_img = None
                self.base_mask_h = None
                self.prev_states.clear()
                self.base_face_hs.clear()
                self.base_scale_map.clear()
                self.scale_factor_map.clear()
                self.next_track_id = 0
                self.track_last_seen.clear()
            print("[FilterEngine] set_custom_mask: no layers found -> cleared mask")
            return True

        # Delegate to path-based setter (handles loading + composition)
        return self.set_custom_mask_by_paths(base_path, mata_path, mulut_path)

    # ---------------- parameter setters ----------------
    def set_manual_scale_percent(self, v: int):
        self.manual_scale_percent = int(v)

    def set_offset_y(self, v: int):
        self.offset_y = int(v)

    def set_offset_x(self, v: int):
        self.offset_x = int(v)

    def set_yaw_percent(self, v: int):
        self.yaw_percent = int(v)

    def set_pitch_percent(self, v: int):
        self.pitch_percent = int(v)

    def set_roll_offset(self, degrees: float):
        self.roll_offset = float(degrees)

    def reset_to_defaults(self):
        self.manual_scale_percent = 200
        self.offset_y = -25
        self.offset_x = 0
        self.yaw_percent = 150
        self.pitch_percent = 150
        self.roll_offset = 0
        with self.lock:
            self.prev_states.clear()
            self.base_face_hs.clear()
            self.base_scale_map.clear()
            self.scale_factor_map.clear()
            self.next_track_id = 0
            self.track_last_seen.clear()

    # ---------------- helper: matching ----------------
    def _match_detections_to_tracks(self, detections: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Greedy nearest-neighbor match detections -> existing tracks.
        Returns assignments: detection_index -> track_id
        Creates new track_id for unmatched detections.
        """
        assignments: Dict[int, int] = {}
        used_tracks = set()
        track_ids = list(self.prev_states.keys())

        for di, d in enumerate(detections):
            best_tid = None
            best_dist = float('inf')
            for tid in track_ids:
                if tid in used_tracks:
                    continue
                prev = self.prev_states.get(tid)
                if prev is None:
                    continue
                dist = math.hypot(d['nx'] - prev['cx'], d['ny'] - prev['cy'])
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            # threshold: either proportional to face width or fixed minimum
            threshold = max(40.0, d['bw'] * 0.5)
            if best_tid is not None and best_dist <= threshold:
                assignments[di] = best_tid
                used_tracks.add(best_tid)
                self.track_last_seen[best_tid] = time.time()
            else:
                # create a new track id
                new_tid = self.next_track_id
                self.next_track_id += 1
                assignments[di] = new_tid
                # init prev state placeholder; will be updated in processing loop
                self.prev_states[new_tid] = {
                    'cx': d['nx'],
                    'cy': d['ny'],
                    'yaw': 0.0,
                    'pitch': 0.0,
                    'roll': 0.0
                }
                # initialize base face height scale maps for this track
                mh = self.base_mask_h if self.base_mask_h is not None else 1
                self.base_face_hs[new_tid] = d['bh']
                self.base_scale_map[new_tid] = (d['bh'] / mh) if mh != 0 else 1.0
                self.scale_factor_map[new_tid] = self.base_scale_map[new_tid]
                self.track_last_seen[new_tid] = time.time()
        return assignments

    # ---------------- main processing ----------------
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Process a single BGR frame (numpy array). Returns processed BGR frame.
        If no mask is set, returns the original frame (copy).
        Thread-safe with respect to set_mask_path/clear_mask.
        """
        if frame_bgr is None:
            return frame_bgr
        with self.lock:
            if self.mask_img is None:
                return frame_bgr.copy()
            # use references while holding lock; we will modify prev_states etc. under same lock
            mask_img = self.mask_img.copy()
            base_mask_h_local = self.base_mask_h

            # copy some params to local for speed/readability
            det_scale = float(self.det_scale)
            max_faces = int(self.max_faces)
            yaw_percent = float(self.yaw_percent)
            pitch_percent = float(self.pitch_percent)
            manual_scale_percent = float(self.manual_scale_percent)
            offset_x = int(self.offset_x)
            offset_y = int(self.offset_y)
            roll_offset = float(self.roll_offset)
            smooth = float(self.smooth)
            min_smooth = float(self.min_smooth)
            MAX_YAW = float(self.MAX_YAW)
            MAX_PITCH = float(self.MAX_PITCH)

        # Use detection on downscaled image
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (0, 0), fx=det_scale, fy=det_scale)
        res = self.face_detection.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        if res and res.detections:
            # Build detection list with coordinates mapped to original image size
            dets: List[Dict[str, Any]] = []
            n_considered = min(max_faces, len(res.detections))
            for i in range(n_considered):
                det = res.detections[i]
                bboxC = det.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                # keypoints
                kps = det.location_data.relative_keypoints
                try:
                    nose = kps[2]
                    nx = int(nose.x * w)
                    ny = int(nose.y * h)
                except Exception:
                    nx = x + bw // 2
                    ny = y + bh // 2
                dets.append({'det': det, 'x': x, 'y': y, 'bw': bw, 'bh': bh, 'nx': nx, 'ny': ny})

            # Lock while matching and updating internal states (ensure set_mask won't clear mid-update)
            with self.lock:
                assignments = self._match_detections_to_tracks(dets)

                # process each detection with assigned track id
                for di, d in enumerate(dets):
                    tid = assignments.get(di)
                    if tid is None:
                        continue
                    det = d['det']
                    x, y, bw, bh = d['x'], d['y'], d['bw'], d['bh']
                    nx, ny = d['nx'], d['ny']

                    # ensure per-track base init (some created in matching already)
                    if tid not in self.base_face_hs:
                        self.base_face_hs[tid] = bh
                    if tid not in self.base_scale_map:
                        mh = base_mask_h_local if base_mask_h_local is not None else 1
                        self.base_scale_map[tid] = (bh / mh) if mh != 0 else 1.0
                        self.scale_factor_map[tid] = self.base_scale_map[tid]

                    # pose estimates
                    kps = det.location_data.relative_keypoints
                    try:
                        r_eye = kps[0]; l_eye = kps[1]; nose = kps[2]
                        rx, ry = int(r_eye.x * w), int(r_eye.y * h)
                        lx, ly = int(l_eye.x * w), int(l_eye.y * h)
                        nx, ny = int(nose.x * w), int(nose.y * h)
                    except Exception:
                        # fallback: use previous values already set
                        nx, ny = nx, ny
                        rx, ry, lx, ly = nx - 10, ny, nx + 10, ny

                    dx = lx - rx
                    dy = ly - ry
                    roll_auto = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0.0

                    norm_x = (nx - (x + bw / 2)) / (bw / 2 + 1e-6)
                    norm_x = clamp(norm_x, -1.0, 1.0)
                    yaw_auto = -norm_x * MAX_YAW

                    norm_y = (ny - (y + bh / 2)) / (bh / 2 + 1e-6)
                    norm_y = clamp(norm_y, -1.0, 1.0)
                    pitch_auto = norm_y * MAX_PITCH

                    # per-track smoothing (prev states exist because matching created it)
                    prev = self.prev_states.get(tid)
                    if prev is None:
                        prev = {'cx': nx, 'cy': ny, 'yaw': yaw_auto, 'pitch': pitch_auto, 'roll': roll_auto}
                        self.prev_states[tid] = prev

                    motion_pos = math.hypot(nx - prev['cx'], ny - prev['cy']) / max(1.0, bw)
                    motion_pose = (abs(yaw_auto - prev['yaw']) / (MAX_YAW + 1e-6) + abs(pitch_auto - prev['pitch']) / (MAX_PITCH + 1e-6)) * 0.5
                    motion = clamp(motion_pos + motion_pose, 0.0, 1.0)
                    alpha = smooth * (1.0 - motion) + min_smooth * motion

                    prev['cx'] = alpha * prev['cx'] + (1.0 - alpha) * nx
                    prev['cy'] = alpha * prev['cy'] + (1.0 - alpha) * ny
                    prev['yaw'] = alpha * prev['yaw'] + (1.0 - alpha) * yaw_auto
                    prev['pitch'] = alpha * prev['pitch'] + (1.0 - alpha) * pitch_auto
                    prev['roll'] = alpha * prev['roll'] + (1.0 - alpha) * roll_auto

                    # final values
                    yaw_sens = yaw_percent / 100.0
                    pitch_sens = pitch_percent / 100.0

                    yaw = prev['yaw'] * yaw_sens
                    pitch = prev['pitch'] * pitch_sens
                    roll = prev['roll'] + roll_offset

                    # ----------------------------
                    # Dynamic scale update (SMOOTHED) - replaces static usage of scale_factor_map
                    # ----------------------------
                    manual_scale = manual_scale_percent / 100.0

                    # compute target scale based on current face height relative to the initial baseline
                    mh = base_mask_h_local if base_mask_h_local is not None else 1
                    try:
                        base_face_h = float(self.base_face_hs.get(tid, bh))
                    except Exception:
                        base_face_h = float(bh)
                    try:
                        base_scale = float(self.base_scale_map.get(tid, 1.0))
                    except Exception:
                        base_scale = 1.0

                    if base_face_h != 0:
                        # target scale: how much the mask should be scaled so its height matches face height
                        target_scale = (bh / base_face_h) * base_scale
                    else:
                        target_scale = (bh / float(mh)) if mh != 0 else 1.0

                    # smoothing constant for scale updates (0.88 keeps it stable; 0.12 follows changes).
                    # This keeps mask resizing smooth while still responsive to large movements.
                    scale_smooth_alpha = 0.88
                    cur_sf = float(self.scale_factor_map.get(tid, target_scale))
                    cur_sf = scale_smooth_alpha * cur_sf + (1.0 - scale_smooth_alpha) * target_scale
                    self.scale_factor_map[tid] = cur_sf

                    final_scale = cur_sf * manual_scale
                    # ----------------------------

                    cx = int(prev['cx']) + int(offset_x)
                    cy = int(prev['cy']) + int(offset_y)

                    # resize and rotate mask
                    mh, mw = mask_img.shape[:2]
                    new_w = max(1, int(mw * final_scale))
                    new_h = max(1, int(mh * final_scale))
                    mask_resized = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    mask_rotated = rotate_image(mask_resized, angle_x=pitch, angle_y=yaw, angle_z=roll)

                    # compute ROI on frame
                    x1 = max(0, cx - mask_rotated.shape[1] // 2)
                    y1 = max(0, cy - mask_rotated.shape[0] // 2)
                    x2 = min(w, x1 + mask_rotated.shape[1])
                    y2 = min(h, y1 + mask_rotated.shape[0])

                    if y2 <= y1 or x2 <= x1:
                        continue

                    mask_cropped = mask_rotated[:y2 - y1, :x2 - x1]

                    if mask_cropped.size == 0:
                        continue

                    # Composite mask_cropped over overlay using proper alpha stacking
                    # convert to float normalized
                    mask_rgb_f = (mask_cropped[..., :3].astype(np.float32) / 255.0)
                    mask_a_f = (mask_cropped[..., 3:4].astype(np.float32) / 255.0)

                    roi = overlay[y1:y2, x1:x2]
                    roi_rgb_f = (roi[..., :3].astype(np.float32) / 255.0)
                    roi_a_f = (roi[..., 3:4].astype(np.float32) / 255.0)

                    out_a = mask_a_f + roi_a_f * (1.0 - mask_a_f)
                    # prevent division by zero
                    denom = np.clip(out_a, 1e-6, 1.0)
                    out_rgb = (mask_rgb_f * mask_a_f + roi_rgb_f * roi_a_f * (1.0 - mask_a_f)) / denom

                    # write back (convert to uint8)
                    roi[..., :3] = (np.clip(out_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                    roi[..., 3:4] = (np.clip(out_a, 0.0, 1.0) * 255.0).astype(np.uint8)

                    overlay[y1:y2, x1:x2] = roi
                    # update last seen
                    self.track_last_seen[tid] = time.time()

                # cleanup stale tracks not seen recently
                now = time.time()
                stale = [tid for tid, t in self.track_last_seen.items() if (now - t) > self.track_timeout]
                for tid in stale:
                    self.prev_states.pop(tid, None)
                    self.base_face_hs.pop(tid, None)
                    self.base_scale_map.pop(tid, None)
                    self.scale_factor_map.pop(tid, None)
                    self.track_last_seen.pop(tid, None)

        # Composite overlay RGBA onto frame BGR
        # convert to floats normalized
        frame_f = frame_bgr.astype(np.float32) / 255.0
        over_rgb_f = overlay[..., :3].astype(np.float32) / 255.0
        over_a_f = overlay[..., 3:4].astype(np.float32) / 255.0

        out_rgb = over_rgb_f * over_a_f + frame_f * (1.0 - over_a_f)
        out = (np.clip(out_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return out

    def close(self):
        try:
            if self.face_detection:
                self.face_detection.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# quick demo (optional) when run directly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", help="path to mask png", required=False)
    parser.add_argument("--masks_folder", help="masks folder", default=None)
    args = parser.parse_args()

    engine = FilterEngine(masks_folder=args.masks_folder)
    if args.mask:
        ok = engine.set_mask(args.mask)
        print("set_mask ok:", ok)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        raise SystemExit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = engine.process_frame(frame)
            cv2.imshow("FilterRef - out", out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
