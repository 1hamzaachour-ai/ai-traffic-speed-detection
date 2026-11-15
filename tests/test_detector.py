"""Tests for the geometry, speed, counting and pipeline logic in detector.py.

They use a synthetic pinhole camera looking down a road, so every expected
number (metres, km/h) comes from the scene itself and not from the code under
test. No YOLO weights or video files are needed.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest
import supervision as sv

from detector import (
    STATUS_PALETTE,
    AnalyzerSettings,
    Calibration,
    LatestFrame,
    LineCrossing,
    RoadGeometry,
    SpeedEstimator,
    TrafficAnalyzer,
    VehicleRecord,
    ViewTransformer,
)

WIDTH, HEIGHT = 1280, 720
FOCAL, CAM_HEIGHT, PITCH = 1000.0, 10.0, math.radians(15)
ROAD_W, ROAD_L = 25.0, 100.0  # road patch used for calibration (metres)
NEAR_Y, FAR_Y = 20.0, 120.0  # distance ahead of the camera where the patch starts and ends


def project(x: float, y: float) -> tuple[float, float, float]:
    """Ground point (x right, y ahead, in metres) -> (pixel u, pixel v, depth)."""
    yc = -y * math.sin(PITCH) + CAM_HEIGHT * math.cos(PITCH)
    zc = y * math.cos(PITCH) + CAM_HEIGHT * math.sin(PITCH)
    return WIDTH / 2 + FOCAL * x / zc, HEIGHT / 2 + FOCAL * yc / zc, zc


def road_calibration() -> Calibration:
    corners = [(-ROAD_W / 2, FAR_Y), (ROAD_W / 2, FAR_Y), (ROAD_W / 2, NEAR_Y), (-ROAD_W / 2, NEAR_Y)]
    roi = tuple((project(x, y)[0] / WIDTH, project(x, y)[1] / HEIGHT) for x, y in corners)
    return Calibration(roi=roi, width_m=ROAD_W, length_m=ROAD_L, line_pos=0.6)


def transformer() -> ViewTransformer:
    return ViewTransformer(road_calibration().roi_pixels(WIDTH, HEIGHT), ROAD_W, ROAD_L)


# --------------------------------------------------------------------------- #
# Perspective transform
# --------------------------------------------------------------------------- #
def test_interior_points_map_to_true_metres():
    """Four corners fix the whole ground plane: interior points must come out right."""
    view = transformer()
    rng = np.random.default_rng(1)
    for _ in range(50):
        x, y = rng.uniform(-12, 12), rng.uniform(NEAR_Y + 1, FAR_Y - 1)
        u, v, _ = project(x, y)
        mx, my = view.to_metres(np.array([[u, v]]))[0]
        assert mx == pytest.approx(x + ROAD_W / 2, abs=0.02)
        assert my == pytest.approx(FAR_Y - y, abs=0.02)  # far edge is metre 0


def test_pixels_round_trip():
    view = transformer()
    points = np.array([[300.0, 400.0], [700.0, 250.0]])
    assert view.to_pixels(view.to_metres(points)) == pytest.approx(points, abs=1e-3)


def test_empty_input_is_fine():
    assert transformer().to_metres(np.empty((0, 2))).shape == (0, 2)


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #
def drive(speed_kmh: float, start_y: float, seconds: float, fps: float, noise_px: float = 0.0,
          lane_x: float = 0.0, seed: int = 0) -> list[float]:
    """Feed a car approaching the camera into a SpeedEstimator; return its readings."""
    view, rng = transformer(), np.random.default_rng(seed)
    estimator, readings = SpeedEstimator(buffer_frames=10), []
    for i in range(int(seconds * fps)):
        t = i / fps
        u, v, _ = project(lane_x, start_y - speed_kmh / 3.6 * t)
        u, v = u + rng.normal(0, noise_px), v + rng.normal(0, noise_px)
        mx, my = view.to_metres(np.array([[u, v]]))[0]
        reading = estimator.update(1, t, float(mx), float(my))
        if reading is not None:
            readings.append(reading)
    return readings


def test_constant_speed_is_recovered():
    readings = drive(90.0, start_y=100.0, seconds=3.0, fps=30)
    assert readings[-1] == pytest.approx(90.0, rel=0.01)


def test_speed_follows_the_clock_not_the_frame_count():
    """Same footage tagged with half the FPS means the car moved at half the pace."""
    fast = drive(72.0, 100.0, 2.0, fps=30)[-1]
    slow = drive(36.0, 100.0, 2.0, fps=15)[-1]  # same frames per metre, half the FPS
    assert fast == pytest.approx(72.0, rel=0.01)
    assert slow == pytest.approx(36.0, rel=0.01)


def test_sideways_movement_counts_too():
    view, estimator = transformer(), SpeedEstimator(buffer_frames=10)
    reading = None
    for i in range(30):
        t = i / 30
        u, v, _ = project(-8.0 + 10.0 * t, 60.0 - 20.0 * t)  # 10 m/s across + 20 m/s ahead
        mx, my = view.to_metres(np.array([[u, v]]))[0]
        reading = estimator.update(1, t, float(mx), float(my))
    assert reading == pytest.approx(math.hypot(10, 20) * 3.6, rel=0.01)


def test_pixel_noise_is_smoothed_in_the_near_field():
    readings = drive(80.0, start_y=70.0, seconds=1.5, fps=30, noise_px=0.5, seed=3)
    assert np.mean(readings[-10:]) == pytest.approx(80.0, rel=0.06)


def test_no_speed_before_enough_samples():
    estimator = SpeedEstimator(buffer_frames=10)
    assert estimator.update(1, 0.0, 5.0, 5.0) is None
    assert estimator.update(1, 0.1, 5.0, 6.0) is None


def test_long_gap_restarts_the_buffer():
    estimator = SpeedEstimator(buffer_frames=6)
    for i in range(8):
        estimator.update(1, i / 30, 5.0, 10.0 + i * 0.5)
    assert estimator.update(1, 5.0, 5.0, 90.0) is None  # 5 s later: start over


def test_impossible_speeds_are_ignored():
    estimator = SpeedEstimator(buffer_frames=6)
    out = [estimator.update(1, i / 30, 5.0, i * 10.0) for i in range(8)]  # 300 m/s
    assert all(value is None for value in out)


# --------------------------------------------------------------------------- #
# Camera input
# --------------------------------------------------------------------------- #
class FakeCamera:
    """Delivers a numbered frame every ``period`` seconds, like a webcam."""

    def __init__(self, period: float = 0.01, working: bool = True):
        self.period, self.working, self.count = period, working, 0

    def read(self):
        time.sleep(self.period)
        if not self.working:
            return False, None
        self.count += 1
        return True, np.full((2, 2, 3), self.count % 250, dtype=np.uint8)


def test_camera_serves_the_newest_frame_stamped_on_arrival():
    """A slow consumer must get recent frames, not the backlog, and honest timestamps."""
    reader = LatestFrame(FakeCamera(period=0.01))
    try:
        first = reader.read()
        time.sleep(0.2)  # the consumer is busy while about 20 frames arrive
        second = reader.read()
        assert first is not None and second is not None
        skipped = int(second[0][0, 0, 0]) - int(first[0][0, 0, 0])
        assert skipped >= 8  # the backlog was dropped, not queued
        assert 0.15 <= second[1] - first[1] <= 0.6  # time between arrivals, not between reads
    finally:
        reader.stop()


def test_camera_frame_is_never_served_twice():
    reader = LatestFrame(FakeCamera(period=0.05))
    try:
        values = [int(reader.read()[0][0, 0, 0]) for _ in range(4)]
        assert values == sorted(set(values))  # strictly increasing
    finally:
        reader.stop()


def test_dead_camera_gives_up_instead_of_hanging():
    reader = LatestFrame(FakeCamera(period=0.0, working=False))
    try:
        started = time.monotonic()
        assert reader.read(timeout=5.0) is None
        assert time.monotonic() - started < 4.0  # it noticed the failure before the timeout
    finally:
        reader.stop()
        assert not reader._thread.is_alive()


# --------------------------------------------------------------------------- #
# Line crossing and records
# --------------------------------------------------------------------------- #
def test_line_crossing_fires_once_per_pass():
    line, hits = LineCrossing(), []
    for y in (90, 95, 99, 101, 105, 110):
        hits.append(line.update(7, y, 100.0))
    assert hits == [False, False, False, True, False, False]


def test_moving_the_line_does_not_count_vehicles():
    line = LineCrossing()
    line.update(7, 90, 100.0)
    assert line.update(7, 90, 80.0) is False  # the point is now "below" a new line


def test_class_vote_beats_a_single_misread():
    record = VehicleRecord(1, 0.0, 0.0)
    for class_id in (2, 2, 7, 2, 2):
        record.class_votes[class_id] += 1
    assert record.class_id == 2


def test_one_frame_spike_is_not_a_peak():
    record = VehicleRecord(1, 0.0, 0.0)
    for kmh in (70, 71, 150, 70, 71, 72):
        record.observe_speed(kmh)
    assert record.peak_speed == pytest.approx(70.0)  # the 150 never survives a 3-wide minimum
    record = VehicleRecord(2, 0.0, 0.0)
    for kmh in (70, 85, 86, 87):
        record.observe_speed(kmh)
    assert record.peak_speed == pytest.approx(85.0)


# --------------------------------------------------------------------------- #
# Calibration and geometry
# --------------------------------------------------------------------------- #
def test_calibration_validation():
    assert Calibration().problem() is None
    assert road_calibration().problem() is None
    bow_tie = Calibration(roi=((0.2, 0.4), (0.8, 0.9), (0.8, 0.4), (0.2, 0.9)))
    assert "cross" in bow_tie.problem()
    assert Calibration(width_m=0).problem() is not None
    flat = Calibration(roi=((0.1, 0.5), (0.4, 0.5), (0.7, 0.5), (0.9, 0.5)))
    assert flat.problem() is not None


def test_line_sits_inside_the_roi_and_spans_its_width():
    geometry = RoadGeometry(road_calibration(), WIDTH, HEIGHT)
    top, bottom = geometry.roi[:, 1].min(), geometry.roi[:, 1].max()
    assert top < geometry.line_y < bottom
    left, right = geometry.span_at(geometry.line_y)
    assert left < right
    assert geometry.contains(np.array([[(left + right) / 2, geometry.line_y]]))[0]
    assert not geometry.contains(np.array([[left - 5, geometry.line_y]]))[0]


def test_speed_zone_starts_where_a_pixel_covers_075_metres():
    """In the pinhole scene one pixel covers (depth ** 2 / 10000) m, so 0.75 m is 87 m ahead."""
    geometry = RoadGeometry(road_calibration(), WIDTH, HEIGHT)
    expected_row = project(0.0, 87.0)[1]
    assert geometry.zone_row == pytest.approx(expected_row, abs=2.0)


def test_roi_beyond_the_frame_is_allowed():
    overhang = Calibration(roi=((0.33, 0.36), (0.6, 0.37), (1.3, 1.0), (-0.14, 1.0)))
    assert overhang.problem() is None
    geometry = RoadGeometry(overhang, WIDTH, HEIGHT)
    assert geometry.line_y <= HEIGHT - 1


# --------------------------------------------------------------------------- #
# Whole pipeline with a scripted detector
# --------------------------------------------------------------------------- #
class ScriptedDetector:
    """Stands in for YOLO: reports one vehicle driving along the road."""

    def __init__(self, class_id: int, speed_kmh: float, start_y: float = 110.0, fps: float = 30.0):
        self.class_id, self.speed, self.start_y, self.fps = class_id, speed_kmh / 3.6, start_y, fps
        self.frame = 0

    def detect(self, frame, confidence, imgsz=640):
        y = self.start_y - self.speed * self.frame / self.fps
        self.frame += 1
        u, v, depth = project(-3.5, y)
        half_w, height = FOCAL * 1.0 / depth, FOCAL * 1.5 / depth
        box = np.array([[u - half_w, v - height, u + half_w, v]], dtype=np.float32)
        return sv.Detections(
            xyxy=box, confidence=np.array([0.9], dtype=np.float32), class_id=np.array([self.class_id])
        )


def run_pipeline(class_id: int, speed_kmh: float, limit: float):
    """Drive from 110 m to 25 m ahead: across the counting line (about 31 m) but inside the ROI."""
    fps = 30.0
    analyzer = TrafficAnalyzer(
        ScriptedDetector(class_id, speed_kmh), road_calibration(), fps,
        AnalyzerSettings(speed_limit_kmh=limit),
    )
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame_count = int((110.0 - 25.0) / (speed_kmh / 3.6) * fps)
    frames = [analyzer.process(blank.copy(), i / fps) for i in range(frame_count)]
    return analyzer, frames


def count_pixels(frame: np.ndarray, rgb: tuple[int, int, int], tolerance: int = 12) -> int:
    bgr = np.array(rgb[::-1])
    return int((np.abs(frame.astype(int) - bgr).max(axis=2) <= tolerance).sum())


def status_rgb(index: int) -> tuple[int, int, int]:
    return STATUS_PALETTE.by_idx(index).as_rgb()


def test_pipeline_measures_counts_and_logs_one_car():
    analyzer, frames = run_pipeline(class_id=2, speed_kmh=72.0, limit=80.0)
    summary = analyzer.summary()
    assert summary.total_counted == 1
    assert summary.counts == {"Car": 1, "Motorcycle": 0, "Bus": 0, "Truck": 0}
    assert summary.avg_speed_kmh == pytest.approx(72.0, rel=0.03)
    assert summary.violations == 0

    log = analyzer.log_dataframe()
    assert len(log) == 1
    row = log.iloc[0]
    assert row.vehicle_type == "Car" and not row.speeding
    assert row.max_speed_kmh == pytest.approx(72.0, rel=0.03)
    assert not np.isnan(row.crossed_line_s)

    # Under the limit and not heavy: the box is green, never red or blue.
    assert count_pixels(frames[-1], status_rgb(0)) > 100
    assert count_pixels(frames[-1], status_rgb(1)) == 0
    assert count_pixels(frames[-1], status_rgb(2)) == 0


def test_pipeline_flags_speeding_in_red():
    analyzer, frames = run_pipeline(class_id=2, speed_kmh=90.0, limit=80.0)
    assert analyzer.summary().violations == 1
    assert bool(analyzer.log_dataframe().iloc[0].speeding)
    assert count_pixels(frames[-1], status_rgb(1)) > 100


def test_changing_the_limit_re_evaluates_violations():
    analyzer, _ = run_pipeline(class_id=2, speed_kmh=72.0, limit=80.0)
    assert analyzer.summary().violations == 0
    analyzer.configure(AnalyzerSettings(speed_limit_kmh=60.0))
    assert analyzer.summary().violations == 1


def test_heavy_vehicles_are_blue_unless_speeding():
    analyzer, frames = run_pipeline(class_id=7, speed_kmh=60.0, limit=80.0)
    assert analyzer.summary().counts["Truck"] == 1
    assert count_pixels(frames[-1], status_rgb(2)) > 100
    analyzer, frames = run_pipeline(class_id=5, speed_kmh=95.0, limit=80.0)
    assert count_pixels(frames[-1], status_rgb(1)) > 100
    assert count_pixels(frames[-1], status_rgb(2)) == 0


def test_far_vehicles_are_tracked_but_not_timed():
    """Beyond the speed zone the vehicle is boxed and counted as active, with no speed."""
    analyzer = TrafficAnalyzer(
        ScriptedDetector(2, 72.0, start_y=118.0), road_calibration(), 30.0, AnalyzerSettings()
    )
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i in range(30):  # 118 m down to 98 m ahead: all farther than the 87 m zone edge
        analyzer.process(blank.copy(), i / 30)
    summary = analyzer.summary()
    assert summary.active == 1 and summary.avg_speed_kmh is None


class FixedBox:
    """Reports the same box every frame."""

    def __init__(self, box):
        self.box = np.array([box], dtype=np.float32)

    def detect(self, frame, confidence, imgsz=640):
        return sv.Detections(
            xyxy=self.box.copy(), confidence=np.array([0.9], dtype=np.float32), class_id=np.array([2])
        )


def test_boxes_cut_off_by_the_frame_edge_are_not_measured():
    to_bottom = Calibration(roi=((0.4, 0.3), (0.6, 0.3), (0.95, 1.0), (0.05, 1.0)))
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for bottom, expected in ((600.0, 1), (HEIGHT - 3.0, 0)):  # the second one touches the edge
        analyzer = TrafficAnalyzer(FixedBox([600, bottom - 80, 680, bottom]), to_bottom, 30.0)
        for i in range(3):
            analyzer.process(blank.copy(), i / 30)
        assert analyzer.summary().active == expected


def test_vehicle_outside_the_roi_is_ignored():
    analyzer = TrafficAnalyzer(
        ScriptedDetector(2, 72.0, start_y=250.0), road_calibration(), 30.0, AnalyzerSettings()
    )
    blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i in range(20):  # still beyond the far edge of the calibrated patch
        analyzer.process(blank.copy(), i / 30)
    assert analyzer.summary().active == 0
    assert analyzer.log_dataframe().empty


def test_reset_clears_everything():
    analyzer, _ = run_pipeline(class_id=2, speed_kmh=72.0, limit=80.0)
    analyzer.reset()
    assert analyzer.summary().total_counted == 0
    assert analyzer.log_dataframe().empty


def test_invalid_calibration_is_rejected():
    with pytest.raises(ValueError):
        TrafficAnalyzer(ScriptedDetector(2, 50.0), Calibration(width_m=0), 30.0)


def test_empty_frames_do_not_crash():
    class Nothing:
        def detect(self, frame, confidence, imgsz=640):
            return sv.Detections.empty()

    analyzer = TrafficAnalyzer(Nothing(), road_calibration(), 30.0)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i in range(5):
        out = analyzer.process(frame.copy(), i / 30)
    assert out.shape == frame.shape and analyzer.summary().active == 0


def test_wide_frames_are_downscaled_to_the_working_width():
    analyzer = TrafficAnalyzer(
        ScriptedDetector(2, 72.0), road_calibration(), 30.0, AnalyzerSettings(), max_width=640
    )
    out = analyzer.process(np.zeros((1440, 2560, 3), dtype=np.uint8), 0.0)
    assert out.shape == (360, 640, 3)
    assert cv2.countNonZero(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)) > 0  # ROI and HUD were drawn
