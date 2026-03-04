import atexit
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory

keyboard_controller = None
keyboard_backend = None
keyboard_error = None

try:
    import pyautogui as _pyautogui

    _pyautogui.FAILSAFE = False
    _pyautogui.PAUSE = 0.0
    keyboard_controller = _pyautogui
    keyboard_backend = "pyautogui"
except Exception as py_exc:
    try:
        import pydirectinput as _pydirectinput

        _pydirectinput.PAUSE = 0.0
        keyboard_controller = _pydirectinput
        keyboard_backend = "pydirectinput"
    except Exception as pd_exc:
        keyboard_error = f"pyautogui: {py_exc}; pydirectinput: {pd_exc}"


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
APP_VERSION = "pose-runner-backend-2026-03-02.1"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


class PoseEngine:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.lock = threading.Lock()
        self.pause_key = "esc"
        self.last_press = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0, self.pause_key: 0.0}
        self.cooldown = 0.55
        self.pause_cooldown = 0.9
        self.clap_is_closed = False
        self.neutral_hip_x = None
        self.neutral_hip_y = None
        self.neutral_torso = None
        self.neutral_shoulder_width = None
        self.controls_enabled = keyboard_controller is not None
        self.controls_backend = keyboard_backend
        self.controls_error = keyboard_error
        self.init_error = None

        self.landmarker = None
        self.pose_connections = []
        self.pose_enum = None

        self._init_landmarker()

    def _ensure_pose_model(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if POSE_MODEL_PATH.exists():
            return
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

    def _init_landmarker(self):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._ensure_pose_model()
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                output_segmentation_masks=False,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.pose_connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
            self.pose_enum = vision.PoseLandmark
        except Exception as exc:
            self.init_error = f"Inisialisasi MediaPipe gagal: {exc}"

    def _open_camera(self):
        if self.cap is not None and self.cap.isOpened():
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            self.cap = cap
        else:
            self.cap = None

    def _press_key(self, key: str, now: float, cooldown: float | None = None):
        if not self.controls_enabled:
            return
        used_cooldown = self.cooldown if cooldown is None else cooldown
        if now - self.last_press.get(key, 0.0) < used_cooldown:
            return
        keyboard_controller.press(key)
        self.last_press[key] = now

    def _make_error_frame(self, message: str) -> bytes:
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Pose stream error",
            (24, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 170, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            message,
            (24, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return b""
        return encoded.tobytes()

    def _draw_landmarks(self, frame, landmarks):
        height, width = frame.shape[:2]

        for connection in self.pose_connections:
            start_idx, end_idx = connection.start, connection.end
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            p1 = (int(landmarks[start_idx].x * width), int(landmarks[start_idx].y * height))
            p2 = (int(landmarks[end_idx].x * width), int(landmarks[end_idx].y * height))
            cv2.line(frame, p1, p2, (95, 205, 255), 2)

        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            if x < 0 or y < 0 or x >= width or y >= height:
                continue

            color = (255, 220, 80)
            if idx in (11, 12, 23, 24):
                color = (60, 255, 120)
            cv2.circle(frame, (x, y), 4, color, -1)

    def _get_body_metrics(self, landmarks):
        left_shoulder = landmarks[int(self.pose_enum.LEFT_SHOULDER)]
        right_shoulder = landmarks[int(self.pose_enum.RIGHT_SHOULDER)]
        left_hip = landmarks[int(self.pose_enum.LEFT_HIP)]
        right_hip = landmarks[int(self.pose_enum.RIGHT_HIP)]

        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2.0
        hip_center_x = (left_hip.x + right_hip.x) / 2.0
        hip_center_y = (left_hip.y + right_hip.y) / 2.0
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        torso_height = max(hip_center_y - shoulder_center_y, 0.12)

        return {
            "shoulder_center_x": shoulder_center_x,
            "shoulder_center_y": shoulder_center_y,
            "hip_center_x": hip_center_x,
            "hip_center_y": hip_center_y,
            "shoulder_width": shoulder_width,
            "torso_height": torso_height,
        }

    def _update_neutral_pose(self, metrics, alpha: float):
        if self.neutral_hip_x is None:
            self.neutral_hip_x = metrics["hip_center_x"]
            self.neutral_hip_y = metrics["hip_center_y"]
            self.neutral_torso = metrics["torso_height"]
            self.neutral_shoulder_width = metrics["shoulder_width"]
            return

        self.neutral_hip_x = (1.0 - alpha) * self.neutral_hip_x + alpha * metrics["hip_center_x"]
        self.neutral_hip_y = (1.0 - alpha) * self.neutral_hip_y + alpha * metrics["hip_center_y"]
        self.neutral_torso = (1.0 - alpha) * self.neutral_torso + alpha * metrics["torso_height"]
        self.neutral_shoulder_width = (
            (1.0 - alpha) * self.neutral_shoulder_width + alpha * metrics["shoulder_width"]
        )

    def _extract_action(self, landmarks):
        metrics = self._get_body_metrics(landmarks)
        self._update_neutral_pose(metrics, alpha=0.14 if self.neutral_hip_x is None else 0.02)

        dx = metrics["hip_center_x"] - self.neutral_hip_x
        dy = metrics["hip_center_y"] - self.neutral_hip_y
        torso_ref = max(self.neutral_torso or metrics["torso_height"], 0.12)
        shoulder_ref = max(self.neutral_shoulder_width or metrics["shoulder_width"], 0.14)

        left_right_threshold = max(0.16 * shoulder_ref, 0.022)
        jump_threshold = max(0.20 * torso_ref, 0.032)
        crouch_threshold = max(0.18 * torso_ref, 0.032)

        action = "none"
        if dy < -jump_threshold:
            action = "up"
        elif dy > crouch_threshold:
            action = "down"
        elif dx > left_right_threshold:
            action = "right"
        elif dx < -left_right_threshold:
            action = "left"

        if action == "none":
            # Keep baseline adaptive only when user is in neutral state.
            self._update_neutral_pose(metrics, alpha=0.06)

        return action, {"dx": dx, "dy": dy}

    def _is_visible(self, landmark, min_visibility: float = 0.5) -> bool:
        visibility = getattr(landmark, "visibility", 1.0)
        return visibility >= min_visibility

    def _detect_clap_event(self, landmarks) -> bool:
        left_wrist = landmarks[int(self.pose_enum.LEFT_WRIST)]
        right_wrist = landmarks[int(self.pose_enum.RIGHT_WRIST)]
        left_shoulder = landmarks[int(self.pose_enum.LEFT_SHOULDER)]
        right_shoulder = landmarks[int(self.pose_enum.RIGHT_SHOULDER)]

        if not all(
            (
                self._is_visible(left_wrist),
                self._is_visible(right_wrist),
                self._is_visible(left_shoulder),
                self._is_visible(right_shoulder),
            )
        ):
            self.clap_is_closed = False
            return False

        wrist_dist_x = abs(left_wrist.x - right_wrist.x)
        wrist_dist_y = abs(left_wrist.y - right_wrist.y)
        shoulder_width = max(abs(left_shoulder.x - right_shoulder.x), 0.15)
        wrist_mid_y = (left_wrist.y + right_wrist.y) / 2.0
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0

        # Clap detected when both wrists are close together in front of upper body.
        clap_close = (
            wrist_dist_x < (shoulder_width * 0.33)
            and wrist_dist_y < 0.08
            and wrist_mid_y < (shoulder_mid_y + 0.18)
        )
        clap_event = clap_close and not self.clap_is_closed
        self.clap_is_closed = clap_close
        return clap_event

    def _detect_landmarks(self, frame_rgb):
        if self.landmarker is None:
            return None

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(image)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def get_frame(self) -> bytes:
        with self.lock:
            self._open_camera()
            if self.cap is None:
                return self._make_error_frame("Camera tidak ditemukan. Cek izin/perangkat kamera.")

            ok, frame = self.cap.read()
            if not ok:
                return self._make_error_frame("Gagal membaca frame kamera.")

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            landmarks = self._detect_landmarks(rgb)
            action = "none"
            clap_event = False
            debug = {"dx": 0.0, "dy": 0.0}

            if landmarks is not None:
                self._draw_landmarks(frame, landmarks)
                action, debug = self._extract_action(landmarks)
                if action in self.last_press:
                    self._press_key(action, time.time())
                clap_event = self._detect_clap_event(landmarks)
                if clap_event:
                    self._press_key(self.pause_key, time.time(), cooldown=self.pause_cooldown)
            else:
                self.clap_is_closed = False

            cv2.putText(
                frame,
                f"Action: {action.upper()}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (50, 255, 130),
                2,
                cv2.LINE_AA,
            )

            if self.controls_enabled:
                control_state = f"ON ({self.controls_backend})"
            else:
                control_state = "OFF (keyboard backend unavailable)"
            cv2.putText(
                frame,
                f"Keyboard Control: {control_state}",
                (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (190, 230, 255),
                2,
                cv2.LINE_AA,
            )
            if self.controls_error:
                cv2.putText(
                    frame,
                    "Keyboard error: cek /health",
                    (18, 164),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 170, 255),
                    2,
                    cv2.LINE_AA,
                )
            clap_state = "DETECTED" if clap_event else ("HOLD" if self.clap_is_closed else "READY")
            cv2.putText(
                frame,
                f"Clap => {self.pause_key.upper()}: {clap_state}",
                (18, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 220, 120),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Body Shift dx:{debug['dx']:+.3f} dy:{debug['dy']:+.3f}",
                (18, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (160, 200, 255),
                2,
                cv2.LINE_AA,
            )

            if self.init_error:
                cv2.putText(
                    frame,
                    self.init_error,
                    (18, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 170, 255),
                    2,
                    cv2.LINE_AA,
                )

            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                return self._make_error_frame("Gagal encode frame video.")
            return encoded.tobytes()

    def close(self):
        with self.lock:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            if self.landmarker is not None:
                try:
                    self.landmarker.close()
                except RuntimeError:
                    # Can happen on interpreter shutdown when thread pool is already closed.
                    pass


engine = PoseEngine(camera_index=0)
atexit.register(engine.close)


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/game")
def game():
    return send_from_directory(BASE_DIR, "game.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "app_version": APP_VERSION,
            "controls_enabled": engine.controls_enabled,
            "controls_backend": engine.controls_backend,
            "controls_error": engine.controls_error,
            "landmarker_ready": engine.landmarker is not None,
            "camera_index": engine.camera_index,
            "pause_key": engine.pause_key,
            "error": engine.init_error,
        }
    )


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = engine.get_frame()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.02)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
