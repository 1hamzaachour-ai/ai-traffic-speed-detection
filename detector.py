"""Vehicle detection, tracking, speed estimation and line counting.

Per-frame pipeline
------------------
1. YOLO11 (COCO weights) restricted to car, motorcycle, bus and truck.
2. ByteTrack gives every vehicle a persistent ID.
3. A vehicle is measured while its ground point (bottom-centre of its box) is
   inside the calibrated road ROI.
4. That point is mapped from image pixels to a top-down plane measured in
   metres (``cv2.getPerspectiveTransform``), so distances do not depend on the
   camera angle.
5. Speed is the least-squares velocity over a rolling buffer of recent
   positions, so box jitter does not turn into speed jitter. It is reported
   only where the road is close enough for that to hold
   (``MAX_METRES_PER_PIXEL``), and a new track's first frames are skipped.
6. A virtual horizontal line counts each vehicle once, per class.

This module never imports Streamlit, so it can be used and tested on its own.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import supervision as sv

MODEL_WEIGHTS = "yolo11n.pt"

# COCO class ids kept by the detector, with the names shown to the user.
VEHICLE_CLASSES: dict[int, str] = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
HEAVY_CLASS_IDS = frozenset({5, 7})  # bus, truck

MS_TO_KMH = 3.6
# Box jitter is about a pixel. Where one pixel covers more road than this, jitter alone moves a
# vehicle by metres between frames, so speed is not reported there. On a 1280 px wide motorway
# clip, readings scattered around a vehicle's own typical speed by a median of 1-3 km/h below
# 0.75 m/px and by 9-19 km/h above 1 m/px.
MAX_METRES_PER_PIXEL = 0.75
EDGE_MARGIN = 0.015  # boxes this close to the frame edge (as a fraction of it) count as cut off
# A new track's first boxes are unreliable (a car emerging from behind a truck, confidence still
# rising), which read as bursts of 150-250 km/h. They are skipped. Worst reading on the sample:
# 246 km/h without the skip, 150 km/h with it.
SETTLE_FRAMES = 6
DEFAULT_ROI = ((0.38, 0.42), (0.62, 0.42), (0.97, 0.97), (0.03, 0.97))

# Box colour = status. The index is used as a palette lookup.
STATUS_NORMAL, STATUS_SPEEDING, STATUS_HEAVY = 0, 1, 2
STATUS_PALETTE = sv.ColorPalette(
    [
        sv.Color.from_hex("#16a34a"),  # green: under the limit
        sv.Color.from_hex("#dc2626"),  # red: speeding
        sv.Color.from_hex("#2563eb"),  # blue: bus / truck moving normally
    ]
)

ROI_COLOR = (0, 215, 255)  # BGR amber
LINE_COLOR = (255, 255, 0)  # BGR cyan
GRID_COLOR = (255, 255, 255)

LOG_COLUMNS = [
    "vehicle_id",
    "vehicle_type",
    "detected_at",
    "first_seen_s",
    "last_seen_s",
    "crossed_line_s",
    "max_speed_kmh",
    "avg_speed_kmh",
    "speeding",
    "speed_limit_kmh",
]


# --------------------------------------------------------------------------- #
# Calibration and geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Calibration:
    """Road calibration that does not depend on the video resolution.

    ``roi`` holds four corners (x, y) as fractions of the frame width and
    height, ordered top-left, top-right, bottom-right, bottom-left. Values may
    lie outside 0-1, because a road ROI often runs past the bottom edge.
    ``width_m`` x ``length_m`` is the real size of that quadrilateral on the
    road surface. ``line_pos`` places the counting line inside the ROI:
    0 is its top edge and 1 its bottom edge.
    """

    roi: tuple[tuple[float, float], ...] = DEFAULT_ROI
    width_m: float = 25.0
    length_m: float = 100.0
    line_pos: float = 0.6

    def roi_pixels(self, width: int, height: int) -> np.ndarray:
        return np.asarray(self.roi, dtype=np.float32) * np.array([width, height], dtype=np.float32)

    def problem(self) -> str | None:
        """Return why this calibration cannot be used, or None when it is valid."""
        if self.width_m <= 0 or self.length_m <= 0:
            return "Road width and length must be greater than zero."
        corners = np.asarray(self.roi, dtype=np.float32).reshape(-1, 1, 2)
        if corners.shape[0] != 4:
            return "The ROI needs exactly four corners."
        if not cv2.isContourConvex(corners):
            return (
                "The ROI corners cross over. Keep the order top-left, top-right, "
                "bottom-right, bottom-left."
            )
        if cv2.contourArea(corners) < 0.002:
            return "The ROI is too small, or its corners lie on one line."
        return None


class ViewTransformer:
    """Perspective map between image pixels and a top-down plane in metres.

    The four ROI corners in the image are mapped to the corners of a
    ``width_m`` x ``length_m`` rectangle, so one unit on the output plane is
    one metre on the road surface.
    """

    def __init__(self, roi_px: np.ndarray, width_m: float, length_m: float):
        source = np.asarray(roi_px, dtype=np.float32).reshape(4, 2)
        target = np.array(
            [[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]], dtype=np.float32
        )
        self.matrix = cv2.getPerspectiveTransform(source, target)
        self.inverse = np.linalg.inv(self.matrix)

    @staticmethod
    def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] == 0:
            return points
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    def to_metres(self, points_px: np.ndarray) -> np.ndarray:
        return self._apply(self.matrix, points_px)

    def to_pixels(self, points_m: np.ndarray) -> np.ndarray:
        return self._apply(self.inverse, points_m)

    def metres_per_pixel(self, points_px: np.ndarray) -> np.ndarray:
        """Road distance covered by one pixel step downwards at each point."""
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        step = self.to_metres(points + np.array([0.0, 1.0])) - self.to_metres(points)
        return np.hypot(step[:, 0], step[:, 1])


def _pt(point) -> tuple[int, int]:
    """Plain-int pixel coordinates, which is what OpenCV drawing calls expect."""
    return int(round(float(point[0]))), int(round(float(point[1])))


def _nice_step(extent: float, max_lines: int = 10) -> float:
    """Smallest 'round' grid spacing that keeps the line count reasonable."""
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000):
        if extent / step <= max_lines:
            return float(step)
    return float(extent)


class RoadGeometry:
    """A calibration resolved against one frame size: ROI, transform, line."""

    def __init__(self, calibration: Calibration, width: int, height: int):
        self.calibration = calibration
        self.size = (width, height)
        self.roi = calibration.roi_pixels(width, height)
        self.transformer = ViewTransformer(self.roi, calibration.width_m, calibration.length_m)
        self._contour = self.roi.reshape(-1, 1, 2)
        self.top = max(float(self.roi[:, 1].min()), 0.0)
        self.bottom = min(float(self.roi[:, 1].max()), height - 1.0)
        self.line_y = self.top + calibration.line_pos * (self.bottom - self.top)
        self.zone_row = self._find_zone_row()

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: which (x, y) points lie inside the ROI."""
        return np.array(
            [cv2.pointPolygonTest(self._contour, (float(x), float(y)), False) >= 0 for x, y in points],
            dtype=bool,
        )

    def span_at(self, y: float) -> tuple[float, float] | None:
        """Left and right x where the horizontal line at row ``y`` meets the ROI edges."""
        crossings = []
        for i in range(4):
            (x1, y1), (x2, y2) = self.roi[i], self.roi[(i + 1) % 4]
            if y1 != y2 and (y1 - y) * (y2 - y) <= 0:
                crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        return (min(crossings), max(crossings)) if crossings else None

    def _find_zone_row(self) -> float | None:
        """First image row, going down the ROI, where speeds become reliable.

        None means no part of the ROI is close enough for that.
        """
        rows = np.arange(self.top, self.bottom, 1.0)
        centres = [(row, self.span_at(row)) for row in rows]
        points = np.array([(sum(span) / 2, row) for row, span in centres if span], dtype=np.float64)
        if points.size == 0:
            return None
        reliable = self.transformer.metres_per_pixel(points) <= MAX_METRES_PER_PIXEL
        if reliable.all():
            return self.top
        return float(points[np.argmax(reliable), 1]) if reliable.any() else None

    def draw(self, scene: np.ndarray, show_grid: bool = False) -> None:
        """Draw the ROI, the counting line and, optionally, a metric grid."""
        polygon = np.round(self.roi).astype(np.int32)
        tint = scene.copy()
        cv2.fillPoly(tint, [polygon], ROI_COLOR)
        cv2.addWeighted(tint, 0.14, scene, 0.86, 0, dst=scene)
        if show_grid:
            self._draw_grid(scene)
        cv2.polylines(scene, [polygon], True, ROI_COLOR, 2, cv2.LINE_AA)
        for corner in polygon:
            cv2.circle(scene, _pt(corner), 5, ROI_COLOR, -1, cv2.LINE_AA)

        scale = self.size[0] / 1280
        span = self.span_at(self.line_y)
        if span is not None:
            y = round(self.line_y)
            (x0, _), (x1, _) = _pt((span[0], 0)), _pt((span[1], 0))
            cv2.line(scene, (x0, y), (x1, y), LINE_COLOR, 2, cv2.LINE_AA)
            cv2.putText(
                scene, "COUNT LINE", (max(x0, 0) + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * scale, LINE_COLOR, max(1, round(scale)), cv2.LINE_AA,
            )

        # Dashed line where speeds start being measured (nearer the camera than this line).
        if self.zone_row is not None and self.zone_row > self.top + 3:
            span = self.span_at(self.zone_row)
            if span is not None:
                y = round(self.zone_row)
                x0, x1 = _pt((span[0], 0))[0], _pt((span[1], 0))[0]
                dash = max(6, round(14 * scale))
                for x in range(x0, x1, 2 * dash):
                    cv2.line(scene, (x, y), (min(x + dash, x1), y), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    scene, "SPEED ZONE", (max(x0, 0) + 8, y + round(16 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * scale, (255, 255, 255), 1, cv2.LINE_AA,
                )

    def _draw_grid(self, scene: np.ndarray) -> None:
        """Metric grid projected onto the road, to check the calibration by eye."""
        cal = self.calibration
        layer = scene.copy()
        scale = self.size[0] / 1280
        length_step, width_step = _nice_step(cal.length_m), _nice_step(cal.width_m, 6)
        labels = []
        for y_m in np.arange(length_step, cal.length_m, length_step):
            a, b = self.transformer.to_pixels(np.array([[0, y_m], [cal.width_m, y_m]]))
            cv2.line(layer, _pt(a), _pt(b), GRID_COLOR, 1, cv2.LINE_AA)
            labels.append((f"{y_m:g} m", (_pt(a)[0] + 6, _pt(a)[1] - 4)))
        for x_m in np.arange(width_step, cal.width_m, width_step):
            a, b = self.transformer.to_pixels(np.array([[x_m, 0], [x_m, cal.length_m]]))
            cv2.line(layer, _pt(a), _pt(b), GRID_COLOR, 1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.55, scene, 0.45, 0, dst=scene)
        for text, origin in labels:  # distance from the far edge of the ROI, outlined to stay readable
            for color, thickness in (((0, 0, 0), 3), (GRID_COLOR, 1)):
                cv2.putText(scene, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, thickness, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Speed and counting
# --------------------------------------------------------------------------- #
class SpeedEstimator:
    """Smoothed per-vehicle speed from positions on the top-down plane.

    Each vehicle keeps a rolling buffer of ``(time, x_m, y_m)``. Its velocity
    is the least-squares slope of position against time over that buffer, so
    zero-mean pixel jitter in the boxes averages out instead of being
    multiplied by the frame rate. The reported speed is the mean of the last
    few such estimates, which stops the label from flickering.
    """

    SPEED_AVERAGE = 4  # speed estimates averaged for display
    MAX_GAP_S = 1.0  # a longer pause in tracking restarts the buffer
    MAX_KMH = 250.0  # faster than this means a lost or swapped ID, not a car

    def __init__(self, buffer_frames: int = 10):
        self.buffer_frames = max(3, int(buffer_frames))
        self.min_samples = max(3, self.buffer_frames // 2)
        self._positions: dict[int, deque] = {}
        self._speeds: dict[int, deque] = {}

    def update(self, track_id: int, timestamp: float, x_m: float, y_m: float) -> float | None:
        """Add one position; return the smoothed speed in km/h once available."""
        positions = self._positions.setdefault(track_id, deque(maxlen=self.buffer_frames))
        if positions and timestamp - positions[-1][0] > self.MAX_GAP_S:
            positions.clear()
            self._speeds.pop(track_id, None)
        positions.append((timestamp, x_m, y_m))
        speeds = self._speeds.setdefault(track_id, deque(maxlen=self.SPEED_AVERAGE))
        if len(positions) >= self.min_samples:
            kmh = self._velocity_kmh(positions)
            if kmh is not None and kmh <= self.MAX_KMH:
                speeds.append(kmh)
        return float(np.mean(speeds)) if speeds else None

    @staticmethod
    def _velocity_kmh(positions: deque) -> float | None:
        data = np.asarray(positions, dtype=np.float64)
        t = data[:, 0] - data[:, 0].mean()
        denominator = float(t @ t)
        if denominator < 1e-12:
            return None
        vx = float(t @ (data[:, 1] - data[:, 1].mean())) / denominator
        vy = float(t @ (data[:, 2] - data[:, 2].mean())) / denominator
        return math.hypot(vx, vy) * MS_TO_KMH

    def forget(self, track_id: int) -> None:
        self._positions.pop(track_id, None)
        self._speeds.pop(track_id, None)

    def reset(self) -> None:
        self._positions.clear()
        self._speeds.clear()


class LineCrossing:
    """Detects the moment a tracked point passes a horizontal line."""

    def __init__(self) -> None:
        self._below: dict[int, bool] = {}
        self._line_y: float | None = None

    def update(self, track_id: int, y: float, line_y: float) -> bool:
        """True on the frame where the point changes side of the line."""
        if line_y != self._line_y:  # the line moved: old sides are meaningless
            self._below.clear()
            self._line_y = line_y
        below = y >= line_y
        previous = self._below.get(track_id)
        self._below[track_id] = below
        return previous is not None and previous != below

    def forget(self, track_id: int) -> None:
        self._below.pop(track_id, None)


# --------------------------------------------------------------------------- #
# Per-vehicle log
# --------------------------------------------------------------------------- #
@dataclass
class VehicleRecord:
    """Everything remembered about one tracked vehicle."""

    SUSTAIN = 3  # consecutive estimates a speed must hold to count as a peak

    track_id: int
    first_seen: float
    last_seen: float
    class_votes: Counter = field(default_factory=Counter)
    crossed_at: float | None = None
    speed_sum: float = 0.0
    speed_samples: int = 0
    peak_speed: float = 0.0
    _recent: deque = field(
        default_factory=lambda: deque(maxlen=VehicleRecord.SUSTAIN), init=False, repr=False
    )

    @property
    def class_id(self) -> int:
        """Most frequent class so far, so a car/truck flip does not change the label."""
        return self.class_votes.most_common(1)[0][0]

    @property
    def mean_speed(self) -> float | None:
        return self.speed_sum / self.speed_samples if self.speed_samples else None

    def observe_speed(self, kmh: float) -> None:
        self.speed_sum += kmh
        self.speed_samples += 1
        self._recent.append(kmh)
        if len(self._recent) == self.SUSTAIN:
            # The slowest of the last few estimates: a one-frame spike cannot raise it.
            self.peak_speed = max(self.peak_speed, min(self._recent))


@dataclass(frozen=True)
class Summary:
    """Numbers behind the dashboard cards."""

    total_counted: int
    active: int
    avg_speed_kmh: float | None
    violations: int
    counts: dict[str, int]
    frames: int


# --------------------------------------------------------------------------- #
# Detector and video input
# --------------------------------------------------------------------------- #
class VehicleDetector:
    """YOLO11 restricted to vehicles. One instance can be shared between sessions."""

    def __init__(self, weights: str = MODEL_WEIGHTS, device: str | None = None):
        from ultralytics import YOLO  # heavy import, so it is deferred until needed
        import torch

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_name = (
            torch.cuda.get_device_name(0) if self.device.startswith("cuda") else "CPU"
        )
        self._lock = threading.Lock()  # the model keeps predictor state between calls
        self.model = YOLO(weights)
        self.detect(np.zeros((360, 640, 3), dtype=np.uint8), confidence=0.25)  # warm-up

    def detect(self, frame: np.ndarray, confidence: float, imgsz: int = 640) -> sv.Detections:
        with self._lock:
            result = self.model.predict(
                frame,
                conf=confidence,
                iou=0.6,
                imgsz=imgsz,
                classes=list(VEHICLE_CLASSES),
                agnostic_nms=True,  # one box per vehicle even if car/truck scores overlap
                device=self.device,
                verbose=False,
            )[0]
        return sv.Detections.from_ultralytics(result)


class _TrackerInput:
    """The small slice of an Ultralytics ``Boxes`` object that BYTETracker reads."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "_TrackerInput":
        return _TrackerInput(self.xywh[index], self.conf[index], self.cls[index])


class VehicleTracker:
    """ByteTrack: persistent vehicle IDs from per-frame detections.

    Detections at or above the confidence threshold can start or continue a
    track. Weaker ones (down to ``LOW_THRESHOLD``) may only continue a track
    that already exists, which keeps an ID alive while a vehicle is briefly
    hard to see. The Kalman-smoothed box is returned, so the ground point that
    feeds the speed estimate jitters less than the raw detection.
    """

    LOW_THRESHOLD = 0.10  # detector confidence floor; the tracker needs weak boxes too

    def __init__(self, confidence: float, fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker  # imported lazily with torch

        self.args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=confidence,
            track_low_thresh=self.LOW_THRESHOLD,
            new_track_thresh=confidence,
            track_buffer=max(10, round(fps)),  # keep a lost track for about one second
            match_thresh=0.8,
            fuse_score=True,
        )
        self._tracker = BYTETracker(self.args)

    def set_confidence(self, confidence: float) -> None:
        self.args.track_high_thresh = self.args.new_track_thresh = confidence

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Feed one frame of detections; return the tracked vehicles with ``tracker_id`` set."""
        if len(detections):
            x1, y1, x2, y2 = detections.xyxy.T.astype(np.float64)
            xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
            view = _TrackerInput(xywh, detections.confidence.astype(np.float64), detections.class_id)
        else:
            view = _TrackerInput(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
        rows = self._tracker.update(view)  # [x1, y1, x2, y2, track_id, score, cls, idx] per track
        if len(rows) == 0:
            tracked = sv.Detections.empty()
            tracked.tracker_id = np.empty(0, dtype=int)
            return tracked
        return sv.Detections(
            xyxy=rows[:, :4].astype(np.float32),
            confidence=rows[:, 5].astype(np.float32),
            class_id=rows[:, 6].astype(int),
            tracker_id=rows[:, 4].astype(int),
        )


class LatestFrame:
    """Reads a camera on a background thread and always serves the newest frame.

    Cameras buffer frames. When processing is slower than the camera, a plain
    ``read()`` returns old frames, and stamping them with the time of the read
    would make every vehicle look slower than it is. Here each frame is stamped
    the moment it arrives, and frames nobody asked for are dropped.
    """

    FAILURES_BEFORE_GIVING_UP = 30

    def __init__(self, capture) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._latest: tuple[np.ndarray, float, int] | None = None  # frame, seconds, sequence
        self._sequence = 0
        self._served = 0
        self._failed = False
        self._stopped = False
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        failures = 0
        while not self._stopped:
            ok, frame = self._capture.read()
            arrived = time.perf_counter() - self._start
            good = bool(ok) and frame is not None
            with self._condition:
                if good:
                    failures = 0
                    self._sequence += 1
                    self._latest = (frame, arrived, self._sequence)
                else:
                    failures += 1
                    self._failed = failures >= self.FAILURES_BEFORE_GIVING_UP
                self._condition.notify_all()
            if not good:
                if self._failed:
                    return
                time.sleep(0.02)  # a camera that returns nothing must not spin the CPU

    def read(self, timeout: float = 5.0) -> tuple[np.ndarray, float] | None:
        """The newest frame not served yet and its arrival time, or None on failure."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None or self._latest[2] == self._served:
                if self._failed or time.monotonic() >= deadline:
                    return None
                self._condition.wait(timeout=0.05)
            frame, arrived, sequence = self._latest
            self._served = sequence
            return frame, arrived

    def stop(self) -> None:
        self._stopped = True
        self._thread.join(timeout=2.0)


class FrameSource:
    """Frame reader for video files and webcams with trustworthy timestamps.

    Files are timed from the frame number and the container FPS, so speeds do
    not depend on how fast this machine processes frames. Webcams report an
    unreliable FPS, so each frame is timed with the monotonic clock when it
    arrives (see ``LatestFrame``).
    """

    def __init__(self, source: str | int):
        self.is_live = isinstance(source, int)
        if self.is_live and sys.platform == "win32":
            self._capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            self._capture.release()
            kind = "camera" if self.is_live else "video"
            raise RuntimeError(f"Could not open {kind} '{source}'.")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 30.0
        frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_count = 0 if self.is_live else max(frames, 0)
        self.position = 0  # frames consumed so far, skipped ones included
        self._live = LatestFrame(self._capture) if self.is_live else None

    def read(self, stride: int = 1) -> tuple[np.ndarray, float] | None:
        """Next frame and its timestamp in seconds, skipping ``stride - 1`` frames of a file."""
        if self._live is not None:
            return self._live.read()
        for _ in range(max(stride, 1) - 1):
            if not self._capture.grab():
                return None
            self.position += 1
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        timestamp = self.position / self.fps
        self.position += 1
        return frame, timestamp

    def release(self) -> None:
        if self._live is not None:
            self._live.stop()  # the reader thread must be done before the camera is released
        self._capture.release()


def read_first_frame(source: str | int) -> np.ndarray | None:
    """One frame for the calibration preview, or None when the source is unreadable."""
    try:
        reader = FrameSource(source)
    except RuntimeError:
        return None
    try:
        for _ in range(6 if reader.is_live else 1):  # cameras need a few frames to settle
            item = reader.read()
        return item[0] if item else None
    finally:
        reader.release()


def fit_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale wide frames (4K and so on) so drawing and streaming stay cheap."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    return cv2.resize(frame, (max_width, round(height * max_width / width)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# The analyser
# --------------------------------------------------------------------------- #
@dataclass
class AnalyzerSettings:
    """Values that may change while a video is running."""

    confidence: float = 0.40
    speed_limit_kmh: float = 80.0
    smoothing_frames: int = 10
    imgsz: int = 640
    show_grid: bool = False


class TrafficAnalyzer:
    """Runs the full pipeline on a stream of frames and keeps the statistics."""

    TRAIL_FRAMES = 40
    PRUNE_AFTER = 120  # processed frames without a sighting before state is dropped

    def __init__(
        self,
        detector: VehicleDetector,
        calibration: Calibration,
        fps: float,
        settings: AnalyzerSettings | None = None,
        max_width: int = 1280,
    ):
        problem = calibration.problem()
        if problem:
            raise ValueError(problem)
        self.detector = detector
        self.calibration = calibration
        self.settings = settings or AnalyzerSettings()
        self.fps = fps
        self.max_width = max_width
        self.started_at = datetime.now()
        self._geometry: RoadGeometry | None = None
        self._box = sv.BoxAnnotator(color=STATUS_PALETTE, thickness=2, color_lookup=sv.ColorLookup.INDEX)
        self._label = sv.LabelAnnotator(
            color=STATUS_PALETTE, text_color=sv.Color.WHITE, text_scale=0.5, text_thickness=1,
            text_padding=5, text_position=sv.Position.TOP_LEFT, color_lookup=sv.ColorLookup.INDEX,
        )
        self.reset()

    # -- state ------------------------------------------------------------- #
    def reset(self) -> None:
        """Forget every vehicle and start counting again from zero."""
        self.tracker = VehicleTracker(self.settings.confidence, self.fps)
        self.speed = SpeedEstimator(self.settings.smoothing_frames)
        self.line = LineCrossing()
        self.records: dict[int, VehicleRecord] = {}
        self._trails: dict[int, deque] = {}
        self._born: dict[int, int] = {}  # frame on which the tracker first reported each ID
        self._last_frame: dict[int, int] = {}  # frame on which it last did
        self.frame_no = 0
        self.active = 0
        self.started_at = datetime.now()

    def configure(self, settings: AnalyzerSettings | None = None, calibration: Calibration | None = None) -> None:
        """Apply new settings or a new calibration without losing the counts."""
        if settings is not None:
            if settings.smoothing_frames != self.settings.smoothing_frames:
                self.speed = SpeedEstimator(settings.smoothing_frames)
            self.tracker.set_confidence(settings.confidence)
            self.settings = settings
        if calibration is not None and calibration != self.calibration:
            problem = calibration.problem()
            if problem:
                raise ValueError(problem)
            self.calibration = calibration
            self._geometry = None
            self.speed.reset()  # positions in metres are only valid for the old mapping

    def _geometry_for(self, width: int, height: int) -> RoadGeometry:
        geometry = self._geometry
        if geometry is None or geometry.size != (width, height):
            geometry = self._geometry = RoadGeometry(self.calibration, width, height)
        return geometry

    # -- per frame --------------------------------------------------------- #
    def process(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Analyse one frame taken at ``timestamp`` seconds; return it annotated."""
        scene = fit_width(frame, self.max_width)
        height, width = scene.shape[:2]
        geometry = self._geometry_for(width, height)
        limit = self.settings.speed_limit_kmh
        self.frame_no += 1

        detections = self.detector.detect(scene, VehicleTracker.LOW_THRESHOLD, self.settings.imgsz)
        detections = self.tracker.update(detections)
        for track_id in detections.tracker_id.tolist():
            self._born.setdefault(track_id, self.frame_no)
            self._last_frame[track_id] = self.frame_no
        anchors = (
            detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            if len(detections)
            else np.empty((0, 2), dtype=np.float32)
        )
        # A box cut off by the frame edge has a wrong ground point, so it is not measured.
        x1, _, x2, y2 = detections.xyxy.T
        margin_x, margin_y = EDGE_MARGIN * width, EDGE_MARGIN * height
        whole = (x1 > margin_x) & (x2 < width - margin_x) & (y2 < height - margin_y)
        inside = geometry.contains(anchors) & whole
        detections, anchors = detections[inside], anchors[inside]
        metres = geometry.transformer.to_metres(anchors)
        precise = geometry.transformer.metres_per_pixel(anchors) <= MAX_METRES_PER_PIXEL

        status = np.zeros(len(detections), dtype=int)
        labels: list[str] = []
        for i in range(len(detections)):
            track_id = int(detections.tracker_id[i])
            record = self.records.get(track_id)
            if record is None:
                record = self.records[track_id] = VehicleRecord(track_id, timestamp, timestamp)
            record.last_seen = timestamp
            record.class_votes[int(detections.class_id[i])] += 1

            x, y = float(anchors[i, 0]), float(anchors[i, 1])
            kmh = None
            settled = self.frame_no - self._born[track_id] >= SETTLE_FRAMES
            if precise[i] and settled:  # farther away, a pixel of jitter would be metres of error
                kmh = self.speed.update(track_id, timestamp, float(metres[i, 0]), float(metres[i, 1]))
            if kmh is not None:
                record.observe_speed(kmh)
            if self.line.update(track_id, y, geometry.line_y) and record.crossed_at is None:
                record.crossed_at = timestamp
            self._extend_trail(track_id, x, y)

            speeding = record.peak_speed > limit or (kmh is not None and kmh > limit)
            if speeding:
                status[i] = STATUS_SPEEDING
            elif record.class_id in HEAVY_CLASS_IDS:
                status[i] = STATUS_HEAVY
            speed_text = f"{kmh:.0f}" if kmh is not None else "--"
            labels.append(f"#{track_id} | {VEHICLE_CLASSES[record.class_id]} | {speed_text} km/h")

        self.active = len(detections)
        if self.frame_no % 60 == 0:
            self._prune()

        geometry.draw(scene, self.settings.show_grid)
        self._draw_trails(scene, detections, status)
        scene = self._box.annotate(scene, detections, custom_color_lookup=status)
        scene = self._label.annotate(scene, detections, labels=labels, custom_color_lookup=status)
        self._draw_hud(scene)
        return scene

    def _extend_trail(self, track_id: int, x: float, y: float) -> None:
        trail = self._trails.setdefault(track_id, deque(maxlen=self.TRAIL_FRAMES))
        if trail:  # light smoothing so box jitter does not make the trail wavy
            x, y = 0.6 * x + 0.4 * trail[-1][0], 0.6 * y + 0.4 * trail[-1][1]
        trail.append((x, y))

    def _draw_trails(self, scene: np.ndarray, detections: sv.Detections, status: np.ndarray) -> None:
        scale = scene.shape[1] / 1280
        for i in range(len(detections)):
            trail = self._trails.get(int(detections.tracker_id[i]))
            if not trail or len(trail) < 2:
                continue
            color = STATUS_PALETTE.by_idx(int(status[i])).as_bgr()
            for j in range(1, len(trail)):
                thickness = max(1, round((1 + 2 * j / len(trail)) * scale))  # thicker near the car
                cv2.line(scene, _pt(trail[j - 1]), _pt(trail[j]), color, thickness, cv2.LINE_AA)

    def _draw_hud(self, scene: np.ndarray) -> None:
        """Class counters (top-left) and the speed limit (top-right)."""
        scale = scene.shape[1] / 1280
        counts = self.counts_by_class()
        text = "COUNTED   " + "    ".join(f"{name} {n}" for name, n in counts.items())
        text += f"    |    TOTAL {sum(counts.values())}"
        font, size, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, round(scale))
        (text_w, text_h), _ = cv2.getTextSize(text, font, size, thickness)
        pad = int(10 * scale)
        self._darken(scene, 0, 0, text_w + 2 * pad, text_h + 2 * pad)
        cv2.putText(scene, text, (pad, pad + text_h), font, size, (255, 255, 255), thickness, cv2.LINE_AA)

        limit = f"LIMIT {self.settings.speed_limit_kmh:.0f} km/h"
        (limit_w, limit_h), _ = cv2.getTextSize(limit, font, size, thickness)
        x0 = scene.shape[1] - limit_w - 2 * pad
        self._darken(scene, x0, 0, limit_w + 2 * pad, limit_h + 2 * pad)
        cv2.putText(scene, limit, (x0 + pad, pad + limit_h), font, size, (255, 255, 255), thickness, cv2.LINE_AA)

    @staticmethod
    def _darken(scene: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        region = scene[y : y + h, x : x + w]
        region[:] = (region * 0.35).astype(np.uint8)

    def _prune(self) -> None:
        cutoff = self.frame_no - self.PRUNE_AFTER
        for track_id in [t for t, seen in self._last_frame.items() if seen < cutoff]:
            del self._last_frame[track_id]
            self._born.pop(track_id, None)
            self._trails.pop(track_id, None)
            self.speed.forget(track_id)
            self.line.forget(track_id)

    # -- results ----------------------------------------------------------- #
    def counts_by_class(self) -> dict[str, int]:
        """Vehicles that crossed the counting line, per class."""
        counts = dict.fromkeys(VEHICLE_CLASSES.values(), 0)
        for record in self.records.values():
            if record.crossed_at is not None:
                counts[VEHICLE_CLASSES[record.class_id]] += 1
        return counts

    def summary(self) -> Summary:
        counts = self.counts_by_class()
        means = [r.mean_speed for r in self.records.values() if r.mean_speed is not None]
        limit = self.settings.speed_limit_kmh
        return Summary(
            total_counted=sum(counts.values()),
            active=self.active,
            avg_speed_kmh=float(np.mean(means)) if means else None,
            violations=sum(1 for r in self.records.values() if r.peak_speed > limit),
            counts=counts,
            frames=self.frame_no,
        )

    def log_dataframe(self) -> pd.DataFrame:
        """One row per vehicle that was measured or counted."""
        limit = self.settings.speed_limit_kmh
        rows = []
        for record in sorted(self.records.values(), key=lambda r: r.first_seen):
            if record.speed_samples == 0 and record.crossed_at is None:
                continue  # seen too briefly to say anything about it
            detected_at = self.started_at + timedelta(seconds=record.first_seen)
            rows.append(
                {
                    "vehicle_id": record.track_id,
                    "vehicle_type": VEHICLE_CLASSES[record.class_id],
                    "detected_at": detected_at.isoformat(sep=" ", timespec="seconds"),
                    "first_seen_s": round(record.first_seen, 2),
                    "last_seen_s": round(record.last_seen, 2),
                    "crossed_line_s": None if record.crossed_at is None else round(record.crossed_at, 2),
                    "max_speed_kmh": round(record.peak_speed, 1) if record.peak_speed else None,
                    "avg_speed_kmh": None if record.mean_speed is None else round(record.mean_speed, 1),
                    "speeding": record.peak_speed > limit,
                    "speed_limit_kmh": limit,
                }
            )
        return pd.DataFrame(rows, columns=LOG_COLUMNS)
