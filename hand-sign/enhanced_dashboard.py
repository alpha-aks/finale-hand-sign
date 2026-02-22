import streamlit as st
import cv2 as cv
import numpy as np
import os
import time
import shutil
import stat
from pathlib import Path
import pandas as pd
from datetime import datetime
from PIL import Image
import json
import sys
import threading
import random
import string

try:
    from sklearn.neighbors import KNeighborsClassifier
except Exception:
    KNeighborsClassifier = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

MP_AVAILABLE = False
MP_TASKS_AVAILABLE = False
mp = None
python = None
vision = None
USE_NEW_API = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        python = mp_python
        vision = mp_vision
        MP_TASKS_AVAILABLE = True
        USE_NEW_API = True
    except Exception:
        MP_TASKS_AVAILABLE = False
        USE_NEW_API = False
except Exception as exc_mp:
    st.error(
        "MediaPipe not installed correctly.\n\n"
        f"Python: {sys.executable}\n"
        f"Import error: {exc_mp}"
    )

st.set_page_config(
    page_title="Hand Gesture Recognition - Enhanced",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

_THIS_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = str(_THIS_DIR / "training_data")
MODELS_DIR = str(_THIS_DIR / "trained_models")
CONFIG_FILE = str(_THIS_DIR / "gesture_config.json")


def safe_rmtree(path: str):
    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            raise

    shutil.rmtree(path, onerror=_onerror)

class SimpleGestureRecognizer:
    
    
    def __init__(self):
        self.gesture_templates = {}
        self.load_templates()
    
    def load_templates(self):
        
        if not os.path.exists(BASE_DATA_DIR):
            return
        
        for gesture_name in os.listdir(BASE_DATA_DIR):
            gesture_dir = os.path.join(BASE_DATA_DIR, gesture_name)
            if os.path.isdir(gesture_dir):
                images = []
                for img_file in os.listdir(gesture_dir)[:20]:
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(gesture_dir, img_file)
                        img = cv.imread(img_path)
                        if img is not None:
                            img_resized = cv.resize(img, (100, 100))
                            images.append(img_resized)
                
                if images:
                    self.gesture_templates[gesture_name] = images
    
    def compare_histograms(self, img1, img2):
        
        hsv1 = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
        hsv2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
        
        hist1 = cv.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        cv.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        cv.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        
        return cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    
    def recognize(self, image):
        
        if not self.gesture_templates:
            return None, 0.0
        
        img_resized = cv.resize(image, (100, 100))
        
        best_gesture = None
        best_score = 0.0
        
        for gesture_name, templates in self.gesture_templates.items():
            scores = []
            for template in templates:
                score = self.compare_histograms(img_resized, template)
                scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0.0
            
            if avg_score > best_score:
                best_score = avg_score
                best_gesture = gesture_name
        
        if best_score > 0.15:
            return best_gesture, best_score
        
        return None, 0.0


class LandmarkKNNGestureRecognizer:


    def __init__(
        self,
        max_images_per_gesture: int = 80,
        n_neighbors: int = 5,
        min_confidence: float = 0.55,
    ):
        if not (MP_AVAILABLE and MP_TASKS_AVAILABLE and mp is not None and python is not None and vision is not None):
            raise RuntimeError("MediaPipe Tasks not available for landmark recognition")

        model_path = Path(CONFIG_FILE).resolve().parent / "model" / "hand_landmarker.task"
        if not model_path.exists():
            raise RuntimeError(f"HandLandmarker model not found at: {model_path}")

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._max_images_per_gesture = int(max_images_per_gesture)
        self._n_neighbors = int(max(1, n_neighbors))
        self._min_confidence = float(min_confidence)

        self._knn = None
        self._classes = []
        self._train_stats = {
            "sklearn_ok": KNeighborsClassifier is not None,
            "samples": 0,
            "num_classes": 0,
            "per_gesture_added": {},
            "per_gesture_seen": {},
        }
        self._train_from_training_images()

    def get_debug_info(self) -> dict:
        info = dict(self._train_stats) if isinstance(self._train_stats, dict) else {}
        info["knn_trained"] = self._knn is not None
        info["classes"] = list(self._classes)
        return info

    @staticmethod
    def _normalize_landmarks(landmarks_xyz) -> np.ndarray:
        pts = np.array([[float(p[0]), float(p[1])] for p in landmarks_xyz], dtype=np.float32)
        if pts.shape[0] != 21:
            raise ValueError("Expected 21 landmarks")

        base = pts[0].copy()
        pts = pts - base
        scale = np.max(np.linalg.norm(pts, axis=1))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = np.max(np.abs(pts))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0

        pts = pts / float(scale)
        return pts.flatten()

    def _extract_landmarks(self, image_bgr):
        image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        return [(p.x, p.y, getattr(p, 'z', 0.0)) for p in lm]

    def _train_from_training_images(self):
        if not os.path.exists(BASE_DATA_DIR):
            return

        X = []
        y = []
        classes = []

        gesture_names = [
            d for d in os.listdir(BASE_DATA_DIR)
            if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
        ]

        per_gesture_added = {}
        per_gesture_seen = {}

        for gesture_name in sorted(gesture_names):
            gesture_dir = os.path.join(BASE_DATA_DIR, gesture_name)
            img_paths = [
                os.path.join(gesture_dir, f)
                for f in os.listdir(gesture_dir)
                if f.lower().endswith('.jpg')
            ]
            if not img_paths:
                continue

            per_gesture_seen[gesture_name] = len(img_paths)

            if len(img_paths) > self._max_images_per_gesture:
                img_paths = random.sample(img_paths, self._max_images_per_gesture)

            added = 0
            for img_path in img_paths:
                img = cv.imread(img_path)
                if img is None:
                    continue
                landmarks = self._extract_landmarks(img)
                if not landmarks:
                    continue
                try:
                    feat = self._normalize_landmarks(landmarks)
                except Exception:
                    continue
                X.append(feat)
                y.append(gesture_name)
                added += 1

            if added > 0:
                classes.append(gesture_name)

            per_gesture_added[gesture_name] = added

        self._classes = classes
        self._train_stats = {
            "sklearn_ok": KNeighborsClassifier is not None,
            "samples": int(len(X)),
            "num_classes": int(len(set(y))),
            "per_gesture_added": dict(per_gesture_added),
            "per_gesture_seen": dict(per_gesture_seen),
        }
        if len(set(y)) < 2 or len(X) < 10:
            self._knn = None
            return

        if KNeighborsClassifier is None:
            self._knn = None
            return

        n_neighbors = min(self._n_neighbors, max(1, len(X)))
        self._knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
        self._knn.fit(np.asarray(X, dtype=np.float32), np.asarray(y))

    def recognize_from_landmarks(self, landmarks_xyz):
        if self._knn is None:
            return None, 0.0, []

        feat = self._normalize_landmarks(landmarks_xyz).reshape(1, -1)
        proba = self._knn.predict_proba(feat)[0]
        classes = list(self._knn.classes_)
        order = np.argsort(-proba)
        top = [(classes[i], float(proba[i])) for i in order[:3]]
        best_label, best_p = top[0]

        if best_p < self._min_confidence:
            return None, float(best_p), top
        return best_label, float(best_p), top

    def recognize(self, image):
        landmarks = self._extract_landmarks(image)
        if not landmarks:
            return None, 0.0
        label, conf, _top = self.recognize_from_landmarks(landmarks)
        return label, conf


def create_gesture_recognizer():
    mp_model_ok = False
    if MP_AVAILABLE and MP_TASKS_AVAILABLE:
        mp_model_ok = (Path(CONFIG_FILE).resolve().parent / "model" / "hand_landmarker.task").exists()

    if mp_model_ok:
        try:
            max_imgs = int(st.session_state.get('rec_max_images_per_gesture', 80))
            min_conf = float(st.session_state.get('rec_min_confidence', 0.55))
            st.session_state.pop('recognizer_init_error', None)
            return LandmarkKNNGestureRecognizer(
                max_images_per_gesture=max_imgs,
                min_confidence=min_conf,
            )
        except Exception as e:
            st.session_state['recognizer_init_error'] = str(e)
            return SimpleGestureRecognizer()
    return SimpleGestureRecognizer()


def _sanitize_gesture_name(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def _is_path_within_base_dir(path_str: str) -> bool:
    try:
        base = Path(BASE_DATA_DIR).resolve()
        target = Path(path_str).resolve()
        return base == target or base in target.parents
    except Exception:
        return False


def _unique_destination_path(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for i in range(1, 10_000):
        alt = dest_dir / f"{stem}__{i:04d}{suffix}"
        if not alt.exists():
            return alt
    raise RuntimeError("Unable to find a unique filename for destination")


def _refresh_gesture_recognizer(reason: str = ""):
    st.session_state['gesture_recognizer'] = create_gesture_recognizer()
    st.session_state['last_retrain_at'] = time.time()
    st.session_state['last_retrain_reason'] = str(reason or "")


def _move_training_image(src_path: str, new_gesture: str) -> str:
    if not src_path:
        raise ValueError("Missing source path")
    if not _is_path_within_base_dir(src_path):
        raise ValueError("Source path is not inside training_data")

    new_gesture = _sanitize_gesture_name(new_gesture)
    if not new_gesture:
        raise ValueError("Missing new gesture label")

    src = Path(src_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    dest_dir = Path(BASE_DATA_DIR).resolve() / new_gesture
    dest = _unique_destination_path(dest_dir, src.name)
    shutil.move(str(src), str(dest))
    return str(dest)


def _delete_training_image(src_path: str):
    if not src_path:
        raise ValueError("Missing source path")
    if not _is_path_within_base_dir(src_path):
        raise ValueError("Source path is not inside training_data")

    src = Path(src_path).resolve()
    if not src.exists():
        return
    src.unlink()


class SimpleHandDetector:
    
    
    def __init__(self):
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.kernel = np.ones((3, 3), np.uint8)
        
    def detect_skin_color(self, image):
        
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv.inRange(hsv, lower_skin, upper_skin)
        
        mask = cv.erode(mask, self.kernel, iterations=2)
        mask = cv.dilate(mask, self.kernel, iterations=2)
        mask = cv.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def detect_motion(self, image):
        
        fg_mask = self.bg_subtractor.apply(image)
        fg_mask = cv.threshold(fg_mask, 200, 255, cv.THRESH_BINARY)[1]
        fg_mask = cv.dilate(fg_mask, self.kernel, iterations=2)
        return fg_mask
    
    def detect_hands(self, image):
        
        try:
            skin_mask = self.detect_skin_color(image)
            
            contours, _ = cv.findContours(skin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv.contourArea)
                area = cv.contourArea(largest_contour)
                
                return {
                    'contour': largest_contour,
                    'area': area,
                    'mask': skin_mask
                }
            
            return None
        except Exception as e:
            return None
    
    def has_hand(self, results, min_area=5000):
        
        if results is None:
            return False
        
        return results.get('area', 0) > min_area


class MediaPipeHandsDetector:


    def __init__(
        self,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_presence_confidence=0.8,
        min_tracking_confidence=0.7,
        min_bbox_area_ratio=0.02,
        max_bbox_area_ratio=0.75,
    ):
        if not (MP_AVAILABLE and MP_TASKS_AVAILABLE and mp is not None and python is not None and vision is not None):
            raise RuntimeError("MediaPipe Tasks HandLandmarker is not available")

        model_path = Path(CONFIG_FILE).resolve().parent / "model" / "hand_landmarker.task"
        if not model_path.exists():
            raise RuntimeError(f"HandLandmarker model not found at: {model_path}")

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._min_bbox_area_ratio = float(min_bbox_area_ratio)
        self._max_bbox_area_ratio = float(max_bbox_area_ratio)

    def detect_hands(self, image):
        try:
            h, w = image.shape[:2]
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self._landmarker.detect(mp_image)

            if not result.hand_landmarks:
                return None

            landmarks = result.hand_landmarks[0]
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]

            x_min = max(int(min(xs) * w) - 10, 0)
            y_min = max(int(min(ys) * h) - 10, 0)
            x_max = min(int(max(xs) * w) + 10, w - 1)
            y_max = min(int(max(ys) * h) + 10, h - 1)

            bw = max(x_max - x_min, 0)
            bh = max(y_max - y_min, 0)

            bbox_area = float(bw * bh)
            frame_area = float(max(w * h, 1))
            area_ratio = bbox_area / frame_area

            if area_ratio < self._min_bbox_area_ratio or area_ratio > self._max_bbox_area_ratio:
                return None

            aspect = (bw / bh) if bh > 0 else 999.0
            if aspect < 0.35 or aspect > 2.8:
                return None

            score = None
            try:
                if result.handedness and result.handedness[0] and result.handedness[0][0]:
                    score = float(result.handedness[0][0].score)
            except Exception:
                score = None

            return {
                'bbox': (x_min, y_min, bw, bh),
                'area': bbox_area,
                'area_ratio': area_ratio,
                'score': score,
                'landmarks': [(lm.x, lm.y, getattr(lm, 'z', 0.0)) for lm in landmarks],
            }
        except Exception:
            return None

    def has_hand(self, results, min_area=5000):
        if results is None:
            return False
        return results.get('area', 0) > min_area


def ensure_directories():
    
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_config():
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_config(config):
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def update_gesture_config(gesture_name):
    
    if not gesture_name:
        return

    config = load_config()
    if gesture_name not in config:
        config[gesture_name] = {
            "created": datetime.now().isoformat(),
            "description": f"Gesture: {gesture_name}"
        }
    config[gesture_name]["last_updated"] = datetime.now().isoformat()
    config[gesture_name]["image_count"] = get_gesture_image_count(gesture_name)
    save_config(config)


def get_gesture_list():
    
    if not os.path.exists(BASE_DATA_DIR):
        return []

    gestures = [
        d for d in os.listdir(BASE_DATA_DIR)
        if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
    ]
    return sorted(gestures)


def get_gesture_info(gesture_name):
    
    config = load_config()
    info = config.get(gesture_name, {})
    if "image_count" not in info:
        info["image_count"] = get_gesture_image_count(gesture_name)
    return info


def save_gesture_image(gesture_name, image):
    
    gesture_dir = os.path.join(BASE_DATA_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    
    existing_images = len([f for f in os.listdir(gesture_dir) if f.endswith('.jpg')])
    
    filename = f"{gesture_name}_{existing_images + 1:04d}.jpg"
    filepath = os.path.join(gesture_dir, filename)
    cv.imwrite(filepath, image)
    
    return filepath, existing_images + 1


def get_gesture_image_count(gesture_name):
    
    gesture_dir = os.path.join(BASE_DATA_DIR, gesture_name)
    if not os.path.exists(gesture_dir):
        return 0
    return len([f for f in os.listdir(gesture_dir) if f.endswith('.jpg')])


def list_gesture_image_paths(gesture_name):
    gesture_dir = Path(BASE_DATA_DIR) / gesture_name
    if not gesture_dir.exists() or not gesture_dir.is_dir():
        return []
    return sorted(str(p) for p in gesture_dir.glob("*.jpg"))


def build_training_data_index():
    rows = []
    for gesture in get_gesture_list():
        for img_path in Path(BASE_DATA_DIR, gesture).glob("*.jpg"):
            try:
                stat = img_path.stat()
                rows.append(
                    {
                        "gesture": gesture,
                        "filename": img_path.name,
                        "path": str(img_path),
                        "size_kb": round(stat.st_size / 1024, 1),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    }
                )
            except OSError:
                continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["gesture", "filename"], kind="stable").reset_index(drop=True)
    return df


HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


def _landmarks_to_pixels(landmarks_xyz, w: int, h: int):
    if not landmarks_xyz:
        return []
    xs = [float(p[0]) for p in landmarks_xyz]
    ys = [float(p[1]) for p in landmarks_xyz]
    normalized = (max(xs) <= 2.0 and max(ys) <= 2.0)

    pts = []
    for x, y in zip(xs, ys):
        if normalized:
            px = int(np.clip(x, 0.0, 1.0) * (w - 1))
            py = int(np.clip(y, 0.0, 1.0) * (h - 1))
        else:
            px = int(np.clip(x, 0, w - 1))
            py = int(np.clip(y, 0, h - 1))
        pts.append((px, py))
    return pts


def _draw_landmark_edges(
    image,
    landmarks_xyz,
    color=(0, 255, 0),
    thickness=2,
    draw_skeleton=True,
    draw_points=True,
    draw_hull=True,
):
    h, w = image.shape[:2]
    pts = _landmarks_to_pixels(landmarks_xyz, w=w, h=h)
    if len(pts) < 2:
        return image

    if draw_skeleton:
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv.line(image, pts[a], pts[b], color, thickness)

    if draw_points:
        for p in pts:
            cv.circle(image, p, 3, color, -1)

    if draw_hull and len(pts) >= 3:
        hull = cv.convexHull(np.array(pts, dtype=np.int32).reshape(-1, 1, 2))
        cv.polylines(image, [hull], isClosed=True, color=color, thickness=max(1, thickness))

    return image


def _safe_crop(frame, bbox):
    if frame is None or bbox is None:
        return frame
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return frame
    h_img, w_img = frame.shape[:2]
    x0 = int(np.clip(x, 0, max(w_img - 1, 0)))
    y0 = int(np.clip(y, 0, max(h_img - 1, 0)))
    x1 = int(np.clip(x + w, 0, w_img))
    y1 = int(np.clip(y + h, 0, h_img))
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1]


def _compute_edge_metrics(roi_bgr, canny_low: int = 80, canny_high: int = 160):
    if roi_bgr is None:
        return {"edge_pixel_ratio": None, "edge_pixels": None, "roi_pixels": None}
    if roi_bgr.size == 0:
        return {"edge_pixel_ratio": None, "edge_pixels": None, "roi_pixels": None}
    try:
        gray = cv.cvtColor(roi_bgr, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(gray, int(canny_low), int(canny_high))
        edge_pixels = int(np.count_nonzero(edges))
        roi_pixels = int(edges.size)
        ratio = float(edge_pixels) / float(max(roi_pixels, 1))
        return {"edge_pixel_ratio": ratio, "edge_pixels": edge_pixels, "roi_pixels": roi_pixels}
    except Exception:
        return {"edge_pixel_ratio": None, "edge_pixels": None, "roi_pixels": None}


def _compute_landmark_hull_metrics(frame_bgr, landmarks_xyz):
    if frame_bgr is None or not landmarks_xyz:
        return {"hull_area": None, "hull_perimeter": None}

    h, w = frame_bgr.shape[:2]
    pts = _landmarks_to_pixels(landmarks_xyz, w=w, h=h)
    if len(pts) < 3:
        return {"hull_area": None, "hull_perimeter": None}
    try:
        hull = cv.convexHull(np.array(pts, dtype=np.int32).reshape(-1, 1, 2))
        area = float(cv.contourArea(hull))
        peri = float(cv.arcLength(hull, True))
        return {"hull_area": area, "hull_perimeter": peri}
    except Exception:
        return {"hull_area": None, "hull_perimeter": None}


def _extract_model_data_row(frame_bgr, results, detected_gesture, confidence, include_landmarks: bool, canny_low: int, canny_high: int):
    row = {
        "ts": float(time.time()),
        "gesture": detected_gesture or "",
        "confidence": float(confidence or 0.0),
        "detector_type": str(st.session_state.get('detector_type', '')),
    }

    if results is None:
        return row

    bbox = results.get('bbox')
    if bbox is not None:
        try:
            x, y, bw, bh = bbox
            row.update({"bbox_x": int(x), "bbox_y": int(y), "bbox_w": int(bw), "bbox_h": int(bh)})
            row["bbox_cx"] = float(x + (bw / 2.0))
            row["bbox_cy"] = float(y + (bh / 2.0))
        except Exception:
            pass

    for k in ("area", "area_ratio", "score"):
        if results.get(k) is not None:
            try:
                row[k] = float(results.get(k))
            except Exception:
                row[k] = None

    roi = frame_bgr
    if bbox is not None:
        roi = _safe_crop(frame_bgr, bbox)
    edge = _compute_edge_metrics(roi, canny_low=canny_low, canny_high=canny_high)
    row.update(edge)

    landmarks = results.get('landmarks')
    if landmarks:
        row.update(_compute_landmark_hull_metrics(frame_bgr, landmarks))
        row["num_landmarks"] = int(len(landmarks))
        try:
            pts = _landmarks_to_pixels(landmarks, w=frame_bgr.shape[1], h=frame_bgr.shape[0])
            if pts and len(pts) >= 1:
                row["wrist_x"] = int(pts[0][0])
                row["wrist_y"] = int(pts[0][1])
        except Exception:
            pass
        if include_landmarks:
            try:
                pts = _landmarks_to_pixels(landmarks, w=frame_bgr.shape[1], h=frame_bgr.shape[0])
                for i, (px, py) in enumerate(pts[:21]):
                    row[f"lm_{i}_x"] = int(px)
                    row[f"lm_{i}_y"] = int(py)
            except Exception:
                pass

    return row


def draw_hand_detection(
    image,
    results,
    min_area=5000,
    show_landmark_edges: bool = True,
    show_bbox: bool = True,
    show_contour: bool = True,
    show_hull: bool = True,
    show_skeleton: bool = True,
    show_points: bool = True,
    show_position_text: bool = False,
):
    
    h, w = image.shape[:2]
    
    has_hand = results is not None and results.get('area', 0) > min_area
    
    color = (0, 255, 0) if has_hand else (0, 0, 255)
    status = "✓ HAND DETECTED" if has_hand else "✗ NO HAND - Show your hand"
    
    cv.putText(image, status, (10, 40), 
               cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    
    thickness = 5 if has_hand else 2
    cv.rectangle(image, (0, 0), (w-1, h-1), color, thickness)
    
    if has_hand and results is not None and show_contour and 'contour' in results:
        contour = results['contour']
        
        cv.drawContours(image, [contour], -1, (0, 255, 0), 2)
        
        x, y, bw, bh = cv.boundingRect(contour)
        cv.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        
        area = results.get('area', 0)
        cv.putText(image, f"Area: {int(area)}", (10, 80),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if has_hand and results is not None and show_bbox and 'bbox' in results:
        x, y, bw, bh = results['bbox']
        if bw > 0 and bh > 0:
            cv.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    if (
        has_hand
        and results is not None
        and show_landmark_edges
        and 'landmarks' in results
        and results.get('landmarks')
    ):
        if show_position_text:
            try:
                pts = _landmarks_to_pixels(results['landmarks'], w=w, h=h)
                if pts and len(pts) >= 1:
                    wx, wy = pts[0]
                    cv.putText(image, f"Wrist: ({wx},{wy})", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception:
                pass
        _draw_landmark_edges(
            image,
            results['landmarks'],
            color=(0, 255, 0),
            thickness=2,
            draw_skeleton=show_skeleton,
            draw_points=show_points,
            draw_hull=show_hull,
        )
    
    return image


def create_hand_detector(detector_type: str):
    if detector_type == "MediaPipe" and MP_AVAILABLE and MP_TASKS_AVAILABLE:
        min_bbox_area_ratio = float(st.session_state.get('mp_min_bbox_area_ratio', 0.02))
        return MediaPipeHandsDetector(
            min_detection_confidence=float(st.session_state.get('mp_min_detection_conf', 0.8)),
            min_presence_confidence=float(st.session_state.get('mp_min_presence_conf', 0.8)),
            min_tracking_confidence=float(st.session_state.get('mp_min_tracking_conf', 0.7)),
            min_bbox_area_ratio=min_bbox_area_ratio,
        )
    return SimpleHandDetector()


def _update_detection_streak(raw_has_hand: bool, streak_key: str, required: int = 3) -> bool:
    current = int(st.session_state.get(streak_key, 0))
    current = (current + 1) if raw_has_hand else 0
    st.session_state[streak_key] = current
    return current >= required


def main():
    st.markdown('<h1 class="main-header">👋 Enhanced Hand Gesture Recognition</h1>', 
                unsafe_allow_html=True)
    
    ensure_directories()

    mp_model_ok = False
    if MP_AVAILABLE and MP_TASKS_AVAILABLE:
        mp_model_ok = (Path(CONFIG_FILE).resolve().parent / "model" / "hand_landmarker.task").exists()

    detector_options = ["Legacy"] + (["MediaPipe"] if mp_model_ok else [])
    default_detector = "MediaPipe" if mp_model_ok else "Legacy"
    current_detector = st.session_state.get('detector_type', default_detector)
    selected_detector = st.sidebar.selectbox(
        "Hand Detector",
        options=detector_options,
        index=detector_options.index(current_detector) if current_detector in detector_options else 0,
        help="MediaPipe is recommended: it detects real hands and reduces false positives.",
    )

    st.session_state['detector_type'] = selected_detector

    if st.session_state.get('detector_type') == "MediaPipe":
        st.sidebar.caption("MediaPipe strictness")
        st.session_state.mp_min_detection_conf = st.sidebar.slider(
            "Min detection confidence",
            min_value=0.5,
            max_value=0.95,
            value=float(st.session_state.get('mp_min_detection_conf', 0.8)),
            step=0.05,
        )
        st.session_state.mp_min_tracking_conf = st.sidebar.slider(
            "Min tracking confidence",
            min_value=0.3,
            max_value=0.9,
            value=float(st.session_state.get('mp_min_tracking_conf', 0.7)),
            step=0.05,
        )
        st.session_state.mp_min_presence_conf = st.sidebar.slider(
            "Min presence confidence",
            min_value=0.3,
            max_value=0.95,
            value=float(st.session_state.get('mp_min_presence_conf', 0.8)),
            step=0.05,
            help="Increase to reduce background false positives.",
        )
        st.session_state.mp_min_bbox_area_ratio = st.sidebar.slider(
            "Min hand size (frame %) ",
            min_value=0.005,
            max_value=0.15,
            value=float(st.session_state.get('mp_min_bbox_area_ratio', 0.02)),
            step=0.005,
            help="Increase if background is still detected. If your hand is far, decrease it.",
        )

    detector_type = st.session_state.get('detector_type', 'Legacy')
    if (
        'hand_detector' not in st.session_state
        or st.session_state.get('_hand_detector_type') != detector_type
        or detector_type == "MediaPipe"
    ):
        st.session_state['hand_detector'] = create_hand_detector(detector_type)
        st.session_state['_hand_detector_type'] = detector_type
    
    if 'gesture_recognizer' not in st.session_state:
        st.session_state['gesture_recognizer'] = create_gesture_recognizer()
    
    if 'is_capturing' not in st.session_state:
        st.session_state.is_capturing = False
    
    if 'captured_count' not in st.session_state:
        st.session_state.captured_count = 0
    
    if 'min_hand_area' not in st.session_state:
        st.session_state.min_hand_area = 5000
    
    st.sidebar.title("⚙️ Controls")
    
    with st.sidebar.expander("🔧 Detection Settings", expanded=False):
        st.write("**Adjust if hand not detected:**")

        st.session_state.det_required_streak = st.slider(
            "Stable frames required",
            min_value=1,
            max_value=6,
            value=int(st.session_state.get('det_required_streak', 3)),
            step=1,
            help="Set to 1 for immediate detection (like before). Higher reduces flicker.",
        )
        
        min_area = st.slider(
            "Min Hand Area",
            min_value=200,
            max_value=20000,
            value=5000,
            step=500,
            help="Minimum area to consider as hand"
        )
        
        if 'hand_detector' in st.session_state:
            st.session_state.min_hand_area = min_area

        st.session_state.show_hand_edge = st.checkbox(
            "Show hand edge (landmarks)",
            value=bool(st.session_state.get('show_hand_edge', True)),
            help="MediaPipe mode: draws a hand outline/skeleton so you can see the hand edge.",
        )
        st.session_state.show_hand_skeleton = st.checkbox(
            "Show skeleton",
            value=bool(st.session_state.get('show_hand_skeleton', True)),
        )
        st.session_state.show_hand_points = st.checkbox(
            "Show landmark points",
            value=bool(st.session_state.get('show_hand_points', True)),
        )
        st.session_state.show_hand_hull = st.checkbox(
            "Show hull/outline",
            value=bool(st.session_state.get('show_hand_hull', True)),
        )
        st.session_state.show_hand_bbox = st.checkbox(
            "Show bounding box",
            value=bool(st.session_state.get('show_hand_bbox', True)),
        )
        st.session_state.show_hand_position_text = st.checkbox(
            "Show hand position text",
            value=bool(st.session_state.get('show_hand_position_text', False)),
            help="Shows wrist pixel coordinates while detecting.",
        )
        
        st.info("👋 If hand not detected: set Stable frames=1, lower Min Hand Area, and (if using MediaPipe) lower presence/detection confidence.")
    
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio(
        "Select Mode",
        ["📚 Train New Gesture", "🔍 Test & Recognize", "🔤 Alphabet (Train + Type)", "⌨️ Type Mode", "📊 View Training Data"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    if mode == "📚 Train New Gesture":
        train_mode()
    elif mode == "🔍 Test & Recognize":
        test_mode()
    elif mode == "🔤 Alphabet (Train + Type)":
        alphabet_mode()
    elif mode == "⌨️ Type Mode":
        typing_mode()
    elif mode == "📊 View Training Data":
        view_data_mode()


def _alphabet_labels():
    return [c for c in string.ascii_lowercase]


def _sanitize_text_to_letters(text: str) -> str:
    text = (text or "").strip().lower()
    return "".join([ch for ch in text if ch in string.ascii_lowercase])


def alphabet_mode():
    st.header("🔤 Alphabet (Train + Type)")
    st.caption("Train A–Z as gestures (labels 'a'..'z') and use guided typing: it accepts the current target letter, then automatically moves to the next.")

    tab_train, tab_type = st.tabs(["📚 Train Alphabet", "⌨️ Alphabet Typer"])

    with tab_train:
        st.subheader("📚 Train Alphabet")
        letters = _alphabet_labels()

        col_left, col_right = st.columns([2, 1])
        with col_right:
            st.markdown("**Training Settings**")
            letters_to_train = st.multiselect(
                "Letters to include",
                options=[c.upper() for c in letters],
                default=[c.upper() for c in letters],
            )
            letters_to_train = [c.lower() for c in letters_to_train]
            if not letters_to_train:
                st.warning("Select at least one letter.")
                return

            target_images = st.number_input(
                "Target images per letter",
                min_value=10,
                max_value=300,
                value=60,
                step=10,
            )
            capture_delay = st.slider(
                "Capture delay (seconds)",
                min_value=0.1,
                max_value=2.0,
                value=0.3,
                step=0.1,
            )

            with st.expander("🧠 Auto-training", expanded=False):
                st.session_state.auto_retrain_on_new_data = st.checkbox(
                    "Auto-retrain recognizer",
                    value=bool(st.session_state.get('auto_retrain_on_new_data', True)),
                )
                st.session_state.auto_retrain_every_n = st.number_input(
                    "Retrain every N new images (0 = only at end)",
                    min_value=0,
                    max_value=200,
                    value=int(st.session_state.get('auto_retrain_every_n', 0)),
                    step=5,
                )
                st.session_state.auto_retrain_min_interval = st.number_input(
                    "Min seconds between retrains",
                    min_value=0.0,
                    max_value=30.0,
                    value=float(st.session_state.get('auto_retrain_min_interval', 2.0)),
                    step=0.5,
                )

            st.markdown("---")
            start = st.button("▶️ Start Alphabet Training", type="primary", use_container_width=True)
            stop = st.button("⏹️ Stop", use_container_width=True)

        with col_left:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

        if 'alphabet_is_capturing' not in st.session_state:
            st.session_state['alphabet_is_capturing'] = False
        if 'alphabet_queue' not in st.session_state:
            st.session_state['alphabet_queue'] = []
        if 'alphabet_index' not in st.session_state:
            st.session_state['alphabet_index'] = 0
        if 'alphabet_new_images_since_retrain' not in st.session_state:
            st.session_state['alphabet_new_images_since_retrain'] = 0

        if start:
            st.session_state['alphabet_is_capturing'] = True
            st.session_state['alphabet_queue'] = list(letters_to_train)
            st.session_state['alphabet_index'] = 0
            st.session_state['alphabet_new_images_since_retrain'] = 0
            st.rerun()

        if stop:
            st.session_state['alphabet_is_capturing'] = False
            if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                _refresh_gesture_recognizer(reason="alphabet_stop")
            st.rerun()

        if not st.session_state.get('alphabet_is_capturing'):
            existing = {c: get_gesture_image_count(c) for c in letters_to_train}
            progress_placeholder.dataframe(
                pd.DataFrame(
                    [{"letter": k.upper(), "images": v} for k, v in existing.items()]
                ).sort_values(["letter"], kind="stable"),
                use_container_width=True,
                hide_index=True,
            )
            status_placeholder.info("Click 'Start Alphabet Training' to begin.")
            return

        queue = st.session_state.get('alphabet_queue') or []
        idx = int(st.session_state.get('alphabet_index', 0))
        if idx >= len(queue):
            st.session_state['alphabet_is_capturing'] = False
            if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                _refresh_gesture_recognizer(reason="alphabet_done")
            status_placeholder.success("✅ Alphabet training complete")
            return

        current_letter = queue[idx]
        current_count = get_gesture_image_count(current_letter)
        status_placeholder.success(f"Training letter: **{current_letter.upper()}** ({current_count}/{int(target_images)})")

        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.3)

        last_capture_time = 0.0
        max_seconds = 12.0
        t0 = time.time()

        while st.session_state.get('alphabet_is_capturing') and (time.time() - t0) < max_seconds:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Failed to access camera")
                break

            frame = cv.flip(frame, 1)
            display_frame = frame.copy()

            min_area = st.session_state.get('min_hand_area', 5000)
            results = st.session_state.hand_detector.detect_hands(frame)
            raw_has_hand = results is not None and results.get('area', 0) > min_area
            required = int(st.session_state.get('det_required_streak', 3))
            has_hand = _update_detection_streak(raw_has_hand, streak_key="alphabet_hand_streak", required=required)

            display_frame = draw_hand_detection(
                display_frame,
                results,
                min_area,
                show_landmark_edges=bool(st.session_state.get('show_hand_edge', True)),
                show_bbox=bool(st.session_state.get('show_hand_bbox', True)),
                show_contour=True,
                show_hull=bool(st.session_state.get('show_hand_hull', True)),
                show_skeleton=bool(st.session_state.get('show_hand_skeleton', True)),
                show_points=bool(st.session_state.get('show_hand_points', True)),
                show_position_text=bool(st.session_state.get('show_hand_position_text', False)),
            )

            cv.putText(display_frame, f"Train: {current_letter.upper()}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

            now = time.time()
            if has_hand and (now - last_capture_time) >= float(capture_delay):
                roi = frame
                if results is not None and 'bbox' in results:
                    roi = _safe_crop(frame, results['bbox'])
                elif results is not None and 'contour' in results:
                    x, y, w, h = cv.boundingRect(results['contour'])
                    if w > 0 and h > 0:
                        roi = frame[y:y + h, x:x + w]

                _path, _count = save_gesture_image(current_letter, roi)
                update_gesture_config(current_letter)
                last_capture_time = now
                st.session_state['alphabet_new_images_since_retrain'] = int(st.session_state.get('alphabet_new_images_since_retrain', 0)) + 1

                if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                    every_n = int(st.session_state.get('auto_retrain_every_n', 0))
                    min_interval = float(st.session_state.get('auto_retrain_min_interval', 2.0))
                    last_at = float(st.session_state.get('last_retrain_at', 0.0))
                    if every_n > 0 and st.session_state['alphabet_new_images_since_retrain'] >= every_n and (time.time() - last_at) >= min_interval:
                        _refresh_gesture_recognizer(reason=f"alphabet_auto:{current_letter}")
                        st.session_state['alphabet_new_images_since_retrain'] = 0

                current_count = get_gesture_image_count(current_letter)
                if current_count >= int(target_images):
                    st.session_state['alphabet_index'] = idx + 1
                    if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                        _refresh_gesture_recognizer(reason=f"alphabet_letter_done:{current_letter}")
                    break

            frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.02)

        cap.release()
        st.rerun()

    with tab_type:
        st.subheader("⌨️ Alphabet Typer")
        st.caption("Shows the next letter to type. When you sign it and hold it stable, it appends the letter and advances.")

        recognizer = st.session_state.get('gesture_recognizer')
        if recognizer is None or not hasattr(recognizer, 'get_debug_info'):
            st.warning("Recognizer not ready. Go to Test mode and click 'Reload Training Data'.")
            return

        info = recognizer.get_debug_info() if hasattr(recognizer, 'get_debug_info') else {}
        if not info.get('knn_trained'):
            st.warning("Alphabet typer needs a trained landmark recognizer. Train at least 2 letters (e.g., 'a' and 'b') then click 'Reload Training Data'.")
            with st.expander("Recognizer debug"):
                st.json(info)
            return

        col_a, col_b = st.columns([2, 1])
        with col_b:
            st.markdown("**Typing Settings**")
            seq_mode = st.radio("Sequence", ["A→Z", "Custom text"], horizontal=True)
            if seq_mode == "A→Z":
                seq = "".join(_alphabet_labels())
            else:
                seq = _sanitize_text_to_letters(st.text_input("Text (letters only)", value="hello"))
            if not seq:
                st.warning("Provide at least one letter.")
                return

            min_conf = st.slider(
                "Min confidence",
                min_value=0.3,
                max_value=0.95,
                value=float(st.session_state.get('rec_min_confidence', 0.55)),
                step=0.05,
            )
            hold_frames = st.slider("Hold frames", min_value=1, max_value=10, value=4, step=1)
            st.markdown("---")
            if st.button("🔄 Reset typed text", use_container_width=True):
                st.session_state['alpha_typed'] = ""
                st.session_state['alpha_pos'] = 0
                st.session_state['alpha_hold'] = 0
                st.session_state['alpha_last'] = None
                st.rerun()

        with col_a:
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            typed_placeholder = st.empty()
            topk_placeholder = st.empty()

        if 'alpha_typed' not in st.session_state:
            st.session_state['alpha_typed'] = ""
        if 'alpha_pos' not in st.session_state:
            st.session_state['alpha_pos'] = 0
        if 'alpha_hold' not in st.session_state:
            st.session_state['alpha_hold'] = 0
        if 'alpha_last' not in st.session_state:
            st.session_state['alpha_last'] = None

        pos = int(st.session_state.get('alpha_pos', 0))
        if pos >= len(seq):
            status_placeholder.success("✅ Done")
            typed_placeholder.text_area("Typed", value=st.session_state['alpha_typed'], height=120)
            return

        expected = seq[pos]
        typed_placeholder.text_area("Typed", value=st.session_state['alpha_typed'], height=120)
        status_placeholder.info(f"Next letter: **{expected.upper()}** ({pos+1}/{len(seq)})")

        run = st.checkbox("Start typing", value=False)
        if not run:
            return

        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.3)

        max_seconds = 12.0
        t0 = time.time()

        while (time.time() - t0) < max_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            display_frame = frame.copy()

            min_area = st.session_state.get('min_hand_area', 5000)
            results = st.session_state.hand_detector.detect_hands(frame)
            raw_has_hand = results is not None and results.get('area', 0) > min_area
            required = int(st.session_state.get('det_required_streak', 3))
            has_hand = _update_detection_streak(raw_has_hand, streak_key="alpha_type_hand_streak", required=required)

            display_frame = draw_hand_detection(
                display_frame,
                results,
                min_area,
                show_landmark_edges=bool(st.session_state.get('show_hand_edge', True)),
                show_bbox=bool(st.session_state.get('show_hand_bbox', True)),
                show_contour=True,
                show_hull=bool(st.session_state.get('show_hand_hull', True)),
                show_skeleton=bool(st.session_state.get('show_hand_skeleton', True)),
                show_points=bool(st.session_state.get('show_hand_points', True)),
                show_position_text=bool(st.session_state.get('show_hand_position_text', False)),
            )

            cv.putText(display_frame, f"Next: {expected.upper()}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

            detected = None
            conf = 0.0
            topk = []
            if has_hand and results is not None and 'landmarks' in results:
                detected, conf, topk = recognizer.recognize_from_landmarks(results['landmarks'])

            if detected is not None and conf >= float(min_conf) and detected == expected:
                if st.session_state.get('alpha_last') == detected:
                    st.session_state['alpha_hold'] = int(st.session_state.get('alpha_hold', 0)) + 1
                else:
                    st.session_state['alpha_last'] = detected
                    st.session_state['alpha_hold'] = 1

                if int(st.session_state.get('alpha_hold', 0)) >= int(hold_frames):
                    st.session_state['alpha_typed'] = st.session_state.get('alpha_typed', "") + expected
                    st.session_state['alpha_pos'] = int(st.session_state.get('alpha_pos', 0)) + 1
                    st.session_state['alpha_hold'] = 0
                    st.session_state['alpha_last'] = None
                    break
            else:
                st.session_state['alpha_hold'] = 0
                st.session_state['alpha_last'] = detected

            if topk:
                df_top = pd.DataFrame(topk, columns=["label", "prob"]).assign(prob=lambda d: d["prob"].map(lambda x: f"{x:.1%}"))
                topk_placeholder.dataframe(df_top, use_container_width=True, hide_index=True)
            else:
                topk_placeholder.empty()

            frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.02)

        cap.release()
        st.rerun()


def train_mode():
    
    st.header("📚 Train New Gesture")
    
    st.markdown(
        """
        <div class="info-box">
        <b>How it works:</b><br>
        1. Type a name for your gesture (e.g., "hello", "peace", "thumbs_up")<br>
        2. Click "Start Capturing"<br>
        3. Show your hand gesture to the camera<br>
        4. Images are automatically captured when hand is detected<br>
        5. Collect 50-200 images for best results
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Training Settings")

        with st.expander("🧠 Auto-training", expanded=False):
            st.session_state.auto_retrain_on_new_data = st.checkbox(
                "Auto-retrain recognizer",
                value=bool(st.session_state.get('auto_retrain_on_new_data', True)),
                help="Rebuilds the landmark-KNN so new labels apply immediately.",
            )
            st.session_state.auto_retrain_every_n = st.number_input(
                "Retrain every N new images (0 = only at end)",
                min_value=0,
                max_value=200,
                value=int(st.session_state.get('auto_retrain_every_n', 0)),
                step=5,
            )
            st.session_state.auto_retrain_min_interval = st.number_input(
                "Min seconds between retrains",
                min_value=0.0,
                max_value=30.0,
                value=float(st.session_state.get('auto_retrain_min_interval', 2.0)),
                step=0.5,
            )
        
        gesture_name = st.text_input(
            "Gesture Name",
            value="",
            placeholder="e.g., hello, peace, thumbs_up",
            help="Enter a unique name for this gesture"
        )

        gesture_name = _sanitize_gesture_name(gesture_name)
        
        target_images = st.number_input(
            "Target Images",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of images to capture"
        )
        
        capture_delay = st.slider(
            "Capture Delay (seconds)",
            min_value=0.1,
            max_value=2.0,
            value=0.3,
            step=0.1,
            help="Delay between captures"
        )
        
        st.markdown("---")
        
        if gesture_name:
            current_count = get_gesture_image_count(gesture_name)
            st.metric("Images Collected", current_count)
            
            if current_count > 0:
                progress = min(current_count / target_images, 1.0)
                st.progress(progress)
        
        st.markdown("---")
        
        if gesture_name:
            col_start, col_stop = st.columns(2)
            
            with col_start:
                if st.button("▶️ Start Capturing", type="primary", disabled=st.session_state.is_capturing):
                    st.session_state.is_capturing = True
                    st.session_state.captured_count = 0
                    st.rerun()
            
            with col_stop:
                if st.button("⏹️ Stop", disabled=not st.session_state.is_capturing):
                    st.session_state.is_capturing = False

                    update_gesture_config(gesture_name)

                    if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                        _refresh_gesture_recognizer(reason=f"train_stop:{gesture_name}")

                    st.rerun()
        else:
            st.warning("⚠️ Please enter a gesture name first")
        
        if st.session_state.is_capturing:
            st.success(f"🔴 CAPTURING... ({st.session_state.captured_count} captured)")
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        if not gesture_name:
            st.info("👆 Enter a gesture name in the right panel to start")
            return
        
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Initialize session state for training camera
        if 'training_cap' not in st.session_state:
            st.session_state.training_cap = None
        if 'training_last_capture_time' not in st.session_state:
            st.session_state.training_last_capture_time = 0
        
        if st.session_state.is_capturing:
            # Initialize camera on first capture
            if st.session_state.training_cap is None:
                st.session_state.training_cap = cv.VideoCapture(0, cv.CAP_DSHOW)
                if not st.session_state.training_cap.isOpened():
                    st.session_state.training_cap = cv.VideoCapture(0)
                st.session_state.training_cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.training_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Process multiple frames before rerunning to avoid flickering
            max_frames = 30  # Process 30 frames per rerun (~1 second at 30fps)
            if 'new_images_since_retrain' not in st.session_state:
                st.session_state['new_images_since_retrain'] = 0
            for frame_num in range(max_frames):
                ret, frame = st.session_state.training_cap.read()
                if not ret:
                    break
                
                frame = cv.flip(frame, 1)
                display_frame = frame.copy()
                
                min_area = st.session_state.get('min_hand_area', 5000)
                
                results = st.session_state.hand_detector.detect_hands(frame)
                raw_has_hand = results is not None and results.get('area', 0) > min_area
                required = int(st.session_state.get('det_required_streak', 3))
                has_hand = _update_detection_streak(raw_has_hand, streak_key="train_hand_streak", required=required)
                
                display_frame = draw_hand_detection(
                    display_frame,
                    results,
                    min_area,
                    show_landmark_edges=bool(st.session_state.get('show_hand_edge', True)),
                    show_bbox=bool(st.session_state.get('show_hand_bbox', True)),
                    show_contour=True,
                    show_hull=bool(st.session_state.get('show_hand_hull', True)),
                    show_skeleton=bool(st.session_state.get('show_hand_skeleton', True)),
                    show_points=bool(st.session_state.get('show_hand_points', True)),
                    show_position_text=bool(st.session_state.get('show_hand_position_text', False)),
                )
                
                current_time = time.time()
                if has_hand:
                    if current_time - st.session_state.training_last_capture_time >= capture_delay:
                        roi = frame
                        if results is not None and 'contour' in results:
                            x, y, w, h = cv.boundingRect(results['contour'])
                            if w > 0 and h > 0:
                                roi = frame[y:y + h, x:x + w]
                        elif results is not None and 'bbox' in results:
                            x, y, w, h = results['bbox']
                            if w > 0 and h > 0:
                                roi = frame[y:y + h, x:x + w]

                        filepath, count = save_gesture_image(gesture_name, roi)
                        st.session_state.captured_count = count
                        st.session_state.training_last_capture_time = current_time

                        st.session_state['new_images_since_retrain'] = int(st.session_state.get('new_images_since_retrain', 0)) + 1

                        if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                            every_n = int(st.session_state.get('auto_retrain_every_n', 0))
                            min_interval = float(st.session_state.get('auto_retrain_min_interval', 2.0))
                            last_at = float(st.session_state.get('last_retrain_at', 0.0))
                            should_retrain = (
                                every_n > 0
                                and int(st.session_state.get('new_images_since_retrain', 0)) >= every_n
                                and (time.time() - last_at) >= min_interval
                            )
                            if should_retrain:
                                _refresh_gesture_recognizer(reason=f"train_auto:{gesture_name}")
                                st.session_state['new_images_since_retrain'] = 0
                        
                        if count >= target_images:
                            st.session_state.is_capturing = False
                            if st.session_state.training_cap is not None:
                                st.session_state.training_cap.release()
                                st.session_state.training_cap = None
                            update_gesture_config(gesture_name)
                            if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                                _refresh_gesture_recognizer(reason=f"train_target:{gesture_name}")
                            status_placeholder.success(f"✅ Target reached! Collected {count} images")
                            time.sleep(0.5)
                            st.rerun()
                
                cv.putText(display_frame, f"Captured: {st.session_state.captured_count}/{target_images}", 
                          (10, display_frame.shape[0] - 20),
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
                
                if has_hand:
                    status_placeholder.success("✅ Hand detected - capturing images...")
                else:
                    status_placeholder.warning("⚠️ Show your hand to the camera. Adjust 'Min Hand Area' in sidebar if needed.")
            
            # Only rerun after processing multiple frames
            time.sleep(0.1)
            st.rerun()
        else:
            # Clean up camera when not capturing
            if st.session_state.training_cap is not None:
                st.session_state.training_cap.release()
                st.session_state.training_cap = None
            
            if gesture_name and get_gesture_image_count(gesture_name) > 0:
                update_gesture_config(gesture_name)
                if bool(st.session_state.get('auto_retrain_on_new_data', True)):
                    _refresh_gesture_recognizer(reason=f"train_end:{gesture_name}")
                status_placeholder.success(f"✅ Training session complete! Collected {get_gesture_image_count(gesture_name)} images")
            else:
                status_placeholder.info("Click '▶️ Start Capturing' to begin training")


def test_mode():
    
    st.header("🔍 Test & Recognize Gestures")
    
    gestures = get_gesture_list()
    
    if not gestures:
        st.warning("No gestures trained yet. Train at least one gesture first.")
        return
    
    st.success(
        f"Ready to recognize {len(gestures)} gestures: {', '.join(gestures)}"
    )

    with st.sidebar.expander("🧠 Recognition Settings", expanded=False):
        st.session_state.rec_min_confidence = st.slider(
            "Min recognition confidence",
            min_value=0.3,
            max_value=0.95,
            value=float(st.session_state.get('rec_min_confidence', 0.55)),
            step=0.05,
            help="Increase to reduce wrong labels (may show more 'Unknown').",
        )
        st.session_state.rec_max_images_per_gesture = st.slider(
            "Max training images per gesture",
            min_value=10,
            max_value=200,
            value=int(st.session_state.get('rec_max_images_per_gesture', 80)),
            step=10,
            help="Higher = potentially better accuracy, slower reload.",
        )
    
    if st.button("🔄 Reload Training Data"):
        st.session_state['gesture_recognizer'] = create_gesture_recognizer()
        st.success("Training data reloaded!")

    if st.session_state.get('recognizer_init_error'):
        st.sidebar.warning(f"Recognizer fallback in use: {st.session_state.get('recognizer_init_error')}")
    
    # Initialize session state for testing
    if 'test_active' not in st.session_state:
        st.session_state.test_active = False
    if 'test_cap' not in st.session_state:
        st.session_state.test_cap = None
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📊 Detection Results")
        gesture_placeholder = st.empty()
        confidence_placeholder = st.empty()
        topk_placeholder = st.empty()

        debug_detection = st.checkbox("Show detection debug", value=False)
        debug_placeholder = st.empty()

        st.markdown("---")
        st.subheader("🧾 Model Data (edge values)")
        log_model_data = st.checkbox("Log model data", value=bool(st.session_state.get('log_model_data', False)))
        st.session_state['log_model_data'] = bool(log_model_data)
        log_max = st.slider("Log length (frames)", min_value=0, max_value=500, value=int(st.session_state.get('model_data_log_max', 120)), step=20)
        st.session_state['model_data_log_max'] = int(log_max)

        include_landmarks = st.checkbox("Include all 21 landmarks", value=bool(st.session_state.get('model_data_include_landmarks', False)))
        st.session_state['model_data_include_landmarks'] = bool(include_landmarks)

        canny_low = st.slider("Canny low", min_value=0, max_value=255, value=int(st.session_state.get('model_data_canny_low', 80)), step=5)
        canny_high = st.slider("Canny high", min_value=0, max_value=255, value=int(st.session_state.get('model_data_canny_high', 160)), step=5)
        st.session_state['model_data_canny_low'] = int(canny_low)
        st.session_state['model_data_canny_high'] = int(canny_high)

        if 'model_data_log' not in st.session_state:
            st.session_state['model_data_log'] = []

        model_data_placeholder = st.empty()
        download_placeholder = st.empty()

        recognizer = st.session_state.get('gesture_recognizer')
        if recognizer is None:
            st.caption("Recognizer: None")
        else:
            caption = f"Recognizer: {type(recognizer).__name__}"
            if hasattr(recognizer, 'get_debug_info'):
                info = recognizer.get_debug_info()
                caption += f" | trained={info.get('knn_trained')} samples={info.get('samples')} classes={info.get('num_classes')} sklearn_ok={info.get('sklearn_ok')}"
            st.caption(caption)

            if hasattr(recognizer, 'get_debug_info'):
                info = recognizer.get_debug_info()
                if not info.get('knn_trained'):
                    num_classes = int(info.get('num_classes') or 0)
                    samples = int(info.get('samples') or 0)
                    sklearn_ok = bool(info.get('sklearn_ok'))
                    if not sklearn_ok:
                        st.warning("Landmark recognizer is disabled because scikit-learn is not available. Install scikit-learn, then click 'Reload Training Data'.")
                    elif num_classes < 2:
                        st.warning("Landmark recognizer is not trained yet (will output 'Unknown') because you need at least **2 different gestures**. Train one more gesture (e.g., 'peace'), then click 'Reload Training Data'.")
                    elif samples < 10:
                        st.warning("Landmark recognizer is not trained yet (will output 'Unknown') because there are too few usable samples. Collect more images per gesture, then click 'Reload Training Data'.")
                    else:
                        st.warning("Landmark recognizer is not trained yet (will output 'Unknown'). Click 'Reload Training Data' after collecting images.")
                    with st.expander("Recognizer debug"):
                        st.json(info)
        
        st.markdown("---")
        st.subheader("📝 Trained Gestures")
        for gesture in gestures:
            count = get_gesture_image_count(gesture)
            st.write(f"**{gesture}**: {count} images")
    
    with col1:
        st.subheader("📹 Live Camera")
        
        col_start, col_stop = st.columns([1, 1])
        with col_start:
            if st.button("▶️ Start Testing", use_container_width=True, disabled=st.session_state.test_active):
                st.session_state.test_active = True
                st.session_state.test_cap = cv.VideoCapture(0, cv.CAP_DSHOW)
                if not st.session_state.test_cap.isOpened():
                    st.session_state.test_cap = cv.VideoCapture(0)
                st.session_state.test_cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.test_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Testing", use_container_width=True, disabled=not st.session_state.test_active):
                st.session_state.test_active = False
                if st.session_state.test_cap is not None:
                    st.session_state.test_cap.release()
                    st.session_state.test_cap = None
                st.rerun()
        
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if st.session_state.test_active and st.session_state.test_cap is not None:
            max_frames = 30  # Process 30 frames per rerun
            for frame_num in range(max_frames):
                ret, frame = st.session_state.test_cap.read()
                if not ret:
                    break
                
                frame = cv.flip(frame, 1)
                display_frame = frame.copy()
                
                min_area = st.session_state.get('min_hand_area', 5000)
                
                results = st.session_state.hand_detector.detect_hands(frame)
                raw_has_hand = results is not None and results.get('area', 0) > min_area
                required = int(st.session_state.get('det_required_streak', 3))
                has_hand = _update_detection_streak(raw_has_hand, streak_key="test_hand_streak", required=required)
                
                display_frame = draw_hand_detection(
                    display_frame,
                    results,
                    min_area,
                    show_landmark_edges=bool(st.session_state.get('show_hand_edge', True)),
                    show_bbox=bool(st.session_state.get('show_hand_bbox', True)),
                    show_contour=True,
                    show_hull=bool(st.session_state.get('show_hand_hull', True)),
                    show_skeleton=bool(st.session_state.get('show_hand_skeleton', True)),
                    show_points=bool(st.session_state.get('show_hand_points', True)),
                    show_position_text=bool(st.session_state.get('show_hand_position_text', False)),
                )
                
                detected_gesture = None
                confidence = 0.0
                topk = []
                
                if has_hand:
                    roi = frame
                    if results is not None and 'contour' in results:
                        x, y, w, h = cv.boundingRect(results['contour'])
                        if w > 0 and h > 0:
                            roi = frame[y:y + h, x:x + w]
                    elif results is not None and 'bbox' in results:
                        x, y, w, h = results['bbox']
                        if w > 0 and h > 0:
                            roi = frame[y:y + h, x:x + w]

                    recognizer = st.session_state.get('gesture_recognizer')
                    if recognizer is not None and hasattr(recognizer, 'recognize_from_landmarks') and results is not None and 'landmarks' in results:
                        detected_gesture, confidence, topk = recognizer.recognize_from_landmarks(results['landmarks'])
                    else:
                        detected_gesture, confidence = recognizer.recognize(roi) if recognizer is not None else (None, 0.0)
                    
                    if detected_gesture:
                        cv.putText(display_frame, f"{detected_gesture.upper()}", 
                                  (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv.putText(display_frame, f"Confidence: {confidence:.1%}", 
                                  (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        with gesture_placeholder.container():
                            st.metric("🎯 Detected Gesture", detected_gesture.upper())
                        
                        with confidence_placeholder.container():
                            st.metric("📊 Confidence", f"{confidence:.1%}")

                        if topk:
                            with topk_placeholder.container():
                                df_top = pd.DataFrame(topk, columns=["gesture", "probability"]) \
                                    .assign(probability=lambda d: d["probability"].map(lambda x: f"{x:.1%}"))
                                st.dataframe(df_top, use_container_width=True, hide_index=True)
                        else:
                            with topk_placeholder.container():
                                st.empty()
                        
                        status_placeholder.success(f"✅ Recognized: {detected_gesture}")
                    else:
                        with gesture_placeholder.container():
                            st.metric("🎯 Detected Gesture", "Unknown")
                        with topk_placeholder.container():
                            st.empty()
                        status_placeholder.info("Hand detected but gesture not recognized. Show a trained gesture.")
                else:
                    with gesture_placeholder.container():
                        st.metric("🎯 Detected Gesture", "No hand")
                    with topk_placeholder.container():
                        st.empty()
                    status_placeholder.warning("Show your hand to the camera")

                if st.session_state.get('log_model_data'):
                    row = _extract_model_data_row(
                        frame,
                        results,
                        detected_gesture,
                        confidence,
                        include_landmarks=bool(st.session_state.get('model_data_include_landmarks', False)),
                        canny_low=int(st.session_state.get('model_data_canny_low', 80)),
                        canny_high=int(st.session_state.get('model_data_canny_high', 160)),
                    )
                    log = list(st.session_state.get('model_data_log', []))
                    log.append(row)
                    max_len = int(st.session_state.get('model_data_log_max', 120))
                    if max_len > 0 and len(log) > max_len:
                        log = log[-max_len:]
                    elif max_len == 0:
                        log = []
                    st.session_state['model_data_log'] = log

                with model_data_placeholder.container():
                    latest = None
                    log = st.session_state.get('model_data_log', [])
                    if log:
                        latest = log[-1]
                    if latest:
                        st.write({
                            "gesture": latest.get("gesture"),
                            "confidence": latest.get("confidence"),
                            "edge_pixel_ratio": latest.get("edge_pixel_ratio"),
                            "hull_area": latest.get("hull_area"),
                            "hull_perimeter": latest.get("hull_perimeter"),
                            "wrist": (latest.get("wrist_x"), latest.get("wrist_y")),
                            "bbox_center": (latest.get("bbox_cx"), latest.get("bbox_cy")),
                            "bbox": (latest.get("bbox_x"), latest.get("bbox_y"), latest.get("bbox_w"), latest.get("bbox_h")),
                        })
                        with st.expander("See full latest row"):
                            st.json(latest)
                    else:
                        st.caption("Enable 'Log model data' to view edge/feature values.")

                with download_placeholder.container():
                    log = st.session_state.get('model_data_log', [])
                    if log:
                        df_log = pd.DataFrame(log)
                        st.download_button(
                            "⬇️ Download model data CSV",
                            data=df_log.to_csv(index=False).encode("utf-8"),
                            file_name="model_data_log.csv",
                            mime="text/csv",
                        )

                if debug_detection:
                    with debug_placeholder.container():
                        st.write({
                            "detector_type": st.session_state.get('detector_type'),
                            "raw_has_hand": raw_has_hand,
                            "has_hand(after_streak)": has_hand,
                            "required_streak": int(st.session_state.get('det_required_streak', 3)),
                            "streak": int(st.session_state.get('test_hand_streak', 0)),
                            "min_hand_area": int(st.session_state.get('min_hand_area', 5000)),
                            "area": float(results.get('area')) if results else None,
                            "score": float(results.get('score')) if (results and results.get('score') is not None) else None,
                            "area_ratio": float(results.get('area_ratio')) if (results and results.get('area_ratio') is not None) else None,
                        })
                else:
                    with debug_placeholder.container():
                        st.empty()
                
                frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
            
            # Only rerun after processing multiple frames
            time.sleep(0.1)
            st.rerun()


def speak_text(text):
    """Speak the given text using pyttsx3 in a separate thread"""
    if pyttsx3 is None or not text:
        return
    
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()


def typing_mode():
    
    st.header("⌨️ Type Using Hand Gestures")
    
    gestures = get_gesture_list()
    
    if not gestures:
        st.warning("No gestures trained yet. Train at least one gesture first in **Train New Gesture** mode.")
        return
    
    st.info(f"Available gestures: {', '.join(gestures)}")
    
    # Initialize session state for typing
    if 'typed_text' not in st.session_state:
        st.session_state.typed_text = ""
    if 'gesture_to_char' not in st.session_state:
        st.session_state.gesture_to_char = {gesture: gesture[0].upper() for gesture in gestures}
    if 'typing_active' not in st.session_state:
        st.session_state.typing_active = False
    if 'typing_cap' not in st.session_state:
        st.session_state.typing_cap = None
    if 'last_gesture' not in st.session_state:
        st.session_state.last_gesture = None
    if 'gesture_hold_count' not in st.session_state:
        st.session_state.gesture_hold_count = 0
    if 'pending_char' not in st.session_state:
        st.session_state.pending_char = None
    if 'pending_gesture' not in st.session_state:
        st.session_state.pending_gesture = None
    
    # Setup gesture-to-character mapping
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Character Mapping")
        st.write("Map each gesture to a character:")
        
        for gesture in gestures:
            char = st.text_input(
                f"{gesture} → ",
                value=st.session_state.gesture_to_char.get(gesture, gesture[0].upper()),
                max_chars=3,
                key=f"char_{gesture}"
            )
            st.session_state.gesture_to_char[gesture] = char
    
    with col2:
        st.subheader("📝 Live Typing")
        
        col_text, col_actions = st.columns([3, 1])
        
        with col_text:
            st.text_area(
                "Your typed text:",
                value=st.session_state.typed_text,
                height=150,
                disabled=True,
                key="display_text"
            )
        
        with col_actions:
            st.write("")
            st.write("")
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.typed_text = ""
                st.rerun()
            if st.button("⌫ Backspace", use_container_width=True):
                st.session_state.typed_text = st.session_state.typed_text[:-1]
                st.rerun()
            if st.button("💾 Copy", use_container_width=True):
                st.success("Text copied to clipboard!")
            if st.button("🔊 Speak", use_container_width=True):
                if st.session_state.typed_text:
                    speak_text(st.session_state.typed_text)
                    st.success("Speaking...")
                else:
                    st.info("Nothing to speak yet!")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Typing Recognition")
        
        auto_speak = st.checkbox("🔊 Auto-speak recognized text", value=False, key="auto_speak_check")
        
        col_start, col_stop = st.columns([1, 1])
        with col_start:
            if st.button("▶️ Start Typing", use_container_width=True, disabled=st.session_state.typing_active):
                st.session_state.typing_active = True
                st.session_state.typing_cap = cv.VideoCapture(0, cv.CAP_DSHOW)
                if not st.session_state.typing_cap.isOpened():
                    st.session_state.typing_cap = cv.VideoCapture(0)
                st.session_state.typing_cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.typing_cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Typing", use_container_width=True, disabled=not st.session_state.typing_active):
                st.session_state.typing_active = False
                if st.session_state.typing_cap is not None:
                    st.session_state.typing_cap.release()
                    st.session_state.typing_cap = None
                st.rerun()
        
        frame_placeholder = st.empty()
        gesture_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if st.session_state.typing_active and st.session_state.typing_cap is not None:
            max_frames = 30  # Process 30 frames per rerun
            for frame_num in range(max_frames):
                ret, frame = st.session_state.typing_cap.read()
                if not ret:
                    break
                
                frame = cv.flip(frame, 1)
                display_frame = frame.copy()
                
                min_area = st.session_state.get('min_hand_area', 5000)
                gesture_hold_threshold = 10
                
                results = st.session_state.hand_detector.detect_hands(frame)
                raw_has_hand = results is not None and results.get('area', 0) > min_area
                required = int(st.session_state.get('det_required_streak', 3))
                has_hand = _update_detection_streak(raw_has_hand, streak_key="typing_hand_streak", required=required)
                
                display_frame = draw_hand_detection(
                    display_frame,
                    results,
                    min_area,
                    show_landmark_edges=bool(st.session_state.get('show_hand_edge', True)),
                    show_bbox=bool(st.session_state.get('show_hand_bbox', True)),
                    show_contour=True,
                    show_hull=bool(st.session_state.get('show_hand_hull', True)),
                    show_skeleton=bool(st.session_state.get('show_hand_skeleton', True)),
                    show_points=bool(st.session_state.get('show_hand_points', True)),
                    show_position_text=bool(st.session_state.get('show_hand_position_text', False)),
                )
                
                detected_gesture = None
                confidence = 0.0
                
                if has_hand:
                    roi = frame
                    if results is not None and 'contour' in results:
                        x, y, w, h = cv.boundingRect(results['contour'])
                        if w > 0 and h > 0:
                            roi = frame[y:y + h, x:x + w]

                    recognizer = st.session_state.get('gesture_recognizer')
                    if recognizer is not None and hasattr(recognizer, 'recognize_from_landmarks') and results is not None and 'landmarks' in results:
                        detected_gesture, confidence, _top = recognizer.recognize_from_landmarks(results['landmarks'])
                    else:
                        detected_gesture, confidence = recognizer.recognize(roi) if recognizer is not None else (None, 0.0)
                    
                    if detected_gesture and gestures and detected_gesture in gestures:
                        cv.putText(display_frame, f"{detected_gesture.upper()}", 
                                  (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv.putText(display_frame, f"Confidence: {confidence:.1%}", 
                                  (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Handle gesture recognition and character addition
                        if detected_gesture == st.session_state.last_gesture:
                            st.session_state.gesture_hold_count += 1
                            if st.session_state.gesture_hold_count >= gesture_hold_threshold:
                                char = st.session_state.gesture_to_char.get(detected_gesture, "?")
                                st.session_state.pending_char = char
                                st.session_state.pending_gesture = detected_gesture
                        else:
                            st.session_state.last_gesture = detected_gesture
                            st.session_state.gesture_hold_count = 0
                        
                        with gesture_placeholder.container():
                            st.metric("🎯 Detected Gesture", detected_gesture.upper())
                        
                        with status_placeholder.container():
                            st.success(f"✅ Recognized: {detected_gesture}")
                    else:
                        st.session_state.gesture_hold_count = 0
                        st.session_state.last_gesture = None
                        with gesture_placeholder.container():
                            st.metric("🎯 Detected Gesture", "Unknown")
                        status_placeholder.info("Hand detected. Show a trained gesture to type.")
                else:
                    st.session_state.gesture_hold_count = 0
                    with gesture_placeholder.container():
                        st.metric("🎯 Detected Gesture", "No hand")
                    status_placeholder.warning("Show your hand to the camera to type")
                
                frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
            
            # Only rerun after processing multiple frames
            time.sleep(0.1)
            st.rerun()
    
    with col2:
        st.subheader("📋 Character Map")
        st.write("Current mapping:")
        for gesture, char in st.session_state.gesture_to_char.items():
            st.write(f"**{gesture}** → `{char}`")
        
        st.markdown("---")
        st.subheader("✅ Confirm Character")
        
        if st.session_state.pending_char:
            st.info(f"Hold gesture: **{st.session_state.pending_gesture}**")
            st.metric("Pending Character", st.session_state.pending_char, help="Press Confirm to add this character")
            
            col_confirm, col_skip = st.columns(2)
            with col_confirm:
                if st.button("✅ Confirm", use_container_width=True, type="primary"):
                    st.session_state.typed_text += st.session_state.pending_char
                    speak_text(st.session_state.pending_char)
                    st.session_state.pending_char = None
                    st.session_state.pending_gesture = None
                    st.session_state.gesture_hold_count = 0
                    st.session_state.last_gesture = None
                    st.success(f"Added: {st.session_state.typed_text}")
                    time.sleep(0.3)
                    st.rerun()
            
            with col_skip:
                if st.button("❌ Reject", use_container_width=True):
                    st.session_state.pending_char = None
                    st.session_state.pending_gesture = None
                    st.session_state.gesture_hold_count = 0
                    st.session_state.last_gesture = None
                    st.rerun()
        else:
            st.info("Make a gesture and hold it to see pending character here")


def view_data_mode():
    
    st.header("📊 Training Data Overview")
    
    config = load_config()
    gestures = get_gesture_list()
    
    if not gestures:
        st.warning("No training data collected yet. Start training in **Train New Gesture** mode!")
        return
    
    st.subheader("📈 Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Gestures", len(gestures))
    
    with col2:
        total_images = sum(get_gesture_image_count(g) for g in gestures)
        st.metric("Total Images", total_images)
    
    with col3:
        avg_images = total_images / len(gestures) if gestures else 0
        st.metric("Avg Images/Gesture", f"{avg_images:.0f}")
    
    st.markdown("---")

    st.subheader("🗃️ Training Data Database")
    st.caption("This table is built from all .jpg files found under the training_data folders.")

    df = build_training_data_index()

    summary_rows = []
    for gesture in gestures:
        info = get_gesture_info(gesture)
        summary_rows.append(
            {
                "gesture": gesture,
                "images": get_gesture_image_count(gesture),
                "created": (info.get("created") or "")[:10],
                "last_updated": (info.get("last_updated") or "")[:10],
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["images", "gesture"], ascending=[False, True], kind="stable")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.write("**Per-gesture summary**")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with col_right:
        st.write("**Per-image database (all files)**")
        if df.empty:
            st.warning("No .jpg files found in the training data directory.")
        else:
            filter_gestures = st.multiselect("Filter gestures", options=sorted(df["gesture"].unique().tolist()), default=sorted(df["gesture"].unique().tolist()))
            filtered_df = df[df["gesture"].isin(filter_gestures)] if filter_gestures else df.iloc[0:0]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download database as CSV",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="training_data_database.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.subheader("🏷️ Relabel / Manage Images")
    st.caption("Move an image to another gesture folder (relabel) and optionally retrain immediately.")

    if df.empty:
        st.info("No images to relabel yet.")
    else:
        df_work = df.copy().reset_index(drop=True)
        options = list(range(len(df_work)))
        selected_idx = st.selectbox(
            "Select an image",
            options=options,
            format_func=lambda i: f"{df_work.loc[i, 'gesture']}/{df_work.loc[i, 'filename']}",
        )

        row = df_work.loc[int(selected_idx)]
        src_path = str(row.get('path') or "")
        src_gesture = str(row.get('gesture') or "")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("**Preview**")
            try:
                st.image(Image.open(src_path), use_container_width=True)
            except Exception:
                st.code(src_path)

        with col_b:
            st.write("**Relabel**")
            gesture_choices = sorted(set(get_gesture_list()))
            dest_choice = st.selectbox("Move to gesture", options=gesture_choices + ["(new gesture)"])
            new_gesture = dest_choice
            if dest_choice == "(new gesture)":
                new_gesture = st.text_input("New gesture name", value="")
            new_gesture = _sanitize_gesture_name(new_gesture)

            retrain_after = st.checkbox(
                "Retrain after change",
                value=bool(st.session_state.get('auto_retrain_on_new_data', True)),
            )

            col_move, col_del = st.columns(2)
            with col_move:
                if st.button("📦 Move (Relabel)", type="primary", use_container_width=True):
                    try:
                        if not new_gesture:
                            st.error("Please provide a destination gesture name.")
                        else:
                            dest_path = _move_training_image(src_path, new_gesture)
                            update_gesture_config(new_gesture)
                            if src_gesture and src_gesture != new_gesture:
                                if get_gesture_image_count(src_gesture) == 0:
                                    cfg = load_config()
                                    if src_gesture in cfg:
                                        del cfg[src_gesture]
                                        save_config(cfg)
                            if retrain_after:
                                _refresh_gesture_recognizer(reason=f"relabel:{src_gesture}->{new_gesture}")
                            st.success(f"Moved to: {dest_path}")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Move failed: {exc}")
            with col_del:
                if st.button("🗑️ Delete Image", use_container_width=True):
                    try:
                        _delete_training_image(src_path)
                        if src_gesture and get_gesture_image_count(src_gesture) == 0:
                            cfg = load_config()
                            if src_gesture in cfg:
                                del cfg[src_gesture]
                                save_config(cfg)
                        if retrain_after:
                            _refresh_gesture_recognizer(reason=f"delete_image:{src_gesture}")
                        st.success("Deleted image")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

    st.markdown("---")

    st.subheader("📋 Gesture Details")
    preview_count = st.slider("Preview images per gesture", min_value=0, max_value=50, value=5, step=5)

    for gesture in gestures:
        with st.expander(f"**{gesture.upper()}**", expanded=False):
            count = get_gesture_image_count(gesture)
            info = get_gesture_info(gesture)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Images Collected:** {count}")
                if "created" in info:
                    st.write(f"**Created:** {info['created'][:10]}")
                if "last_updated" in info:
                    st.write(f"**Last Updated:** {info['last_updated'][:10]}")

                image_paths = list_gesture_image_paths(gesture)
                if image_paths and preview_count > 0:
                    st.write("**Preview Images:**")
                    cols = st.columns(5)
                    for idx, img_path in enumerate(image_paths[:preview_count]):
                        with cols[idx % 5]:
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            except Exception:
                                st.write(Path(img_path).name)

            with col2:
                if st.button(f"🗑️ Delete {gesture}", key=f"del_{gesture}"):
                    gesture_dir = os.path.join(BASE_DATA_DIR, gesture)
                    if os.path.exists(gesture_dir):
                        try:
                            safe_rmtree(gesture_dir)
                        except PermissionError as exc:
                            st.error(f"Unable to delete '{gesture}'. Close any open files and try again.\n{exc}")
                            return

                    config = load_config()
                    if gesture in config:
                        del config[gesture]
                    save_config(config)

                    st.success(f"Deleted {gesture}")
                    st.rerun()
    
    st.markdown("---")
    
    st.subheader("💾 Data Location")
    st.code(f"Training Data: {os.path.abspath(BASE_DATA_DIR)}")
    st.info("Use the training_data folders as your dataset for training a model.")


if __name__ == "__main__":
    main()
