"""Streamlit demo of the AI Traffic & Vehicle Speed Detection System.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

from detector import (
    AnalyzerSettings,
    Calibration,
    FrameSource,
    RoadGeometry,
    Summary,
    TrafficAnalyzer,
    VehicleDetector,
    fit_width,
    read_first_frame,
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "samples"
PREVIEW_WIDTH = 1280
JPEG_QUALITY = 85
# Streamlit only frees streamed frames when a script run ends, so a long run would grow without
# limit. Every this many frames the run ends and the next one resumes where it stopped.
CHUNK_FRAMES = 300

st.set_page_config(page_title="AI Traffic & Speed Detection", page_icon="🚦", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 2.1rem; font-variant-numeric: tabular-nums;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Input sources
# --------------------------------------------------------------------------- #
DEFAULT_SPEED_LIMIT = 80


@dataclass(frozen=True)
class Sample:
    filename: str
    url: str
    calibration: Calibration
    speed_limit: int  # the posted limit on that road


# ROI corners are fractions of the frame, ordered top-left, top-right,
# bottom-right, bottom-left. Width and length are the real size of that patch.
SAMPLES = {
    "Motorway, overhead view (4K)": Sample(
        filename="vehicles.mp4",
        url="https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
        calibration=Calibration(
            roi=((0.326, 0.364), (0.598, 0.372), (1.312, 1.0), (-0.143, 1.0)),
            width_m=25.0,
            length_m=250.0,
            line_pos=0.5,
        ),
        speed_limit=130,
    ),
}


@dataclass(frozen=True)
class Source:
    key: str  # changes whenever the input changes
    target: str | int  # file path or camera index
    live: bool
    preset: Calibration | None = None
    speed_limit: int = DEFAULT_SPEED_LIMIT


def download_sample(sample: Sample) -> Path:
    """Fetch a sample clip once; later runs use the cached file."""
    SAMPLE_DIR.mkdir(exist_ok=True)
    target = SAMPLE_DIR / sample.filename
    if not target.exists():
        partial = target.with_suffix(".part")
        with st.spinner(f"Downloading {sample.filename} (first use only) ..."):
            urllib.request.urlretrieve(sample.url, partial)
        partial.replace(target)
    return target


def store_upload(uploaded) -> str:
    """OpenCV needs a real file, so keep the uploaded bytes in a temp file."""
    key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("upload_key") != key:
        previous = st.session_state.get("upload_path")
        if previous:
            try:
                os.remove(previous)
            except OSError:
                pass  # still open in the previous run; the OS cleans the temp folder
        descriptor, path = tempfile.mkstemp(suffix=Path(uploaded.name).suffix or ".mp4")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(uploaded.getbuffer())
        st.session_state.upload_key, st.session_state.upload_path = key, path
    return st.session_state.upload_path


# --------------------------------------------------------------------------- #
# Calibration widgets <-> Calibration object
# --------------------------------------------------------------------------- #
CORNERS = (("tl", "Top-left"), ("tr", "Top-right"), ("br", "Bottom-right"), ("bl", "Bottom-left"))


def apply_preset(calibration: Calibration) -> None:
    """Load a calibration into the widget state (call before the widgets exist)."""
    for (name, _), (x, y) in zip(CORNERS, calibration.roi):
        st.session_state[f"roi_{name}_x"] = round(x * 100, 1)
        st.session_state[f"roi_{name}_y"] = round(y * 100, 1)
    st.session_state.width_m = float(calibration.width_m)
    st.session_state.length_m = float(calibration.length_m)
    st.session_state.line_pos = round(calibration.line_pos * 100)


def read_calibration() -> Calibration:
    state = st.session_state
    roi = tuple((state[f"roi_{name}_x"] / 100, state[f"roi_{name}_y"] / 100) for name, _ in CORNERS)
    return Calibration(
        roi=roi,
        width_m=float(state.width_m),
        length_m=float(state.length_m),
        line_pos=state.line_pos / 100,
    )


# --------------------------------------------------------------------------- #
# Model and run state
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YOLO11n (the first start downloads about 6 MB) ...")
def load_detector() -> VehicleDetector:
    return VehicleDetector()


class Run:
    """One analysis run: the open video, the analyser and the last frame shown."""

    def __init__(self, source: Source, calibration: Calibration, settings: AnalyzerSettings, stride: int):
        self.frames = FrameSource(source.target)
        self.stride = 1 if source.live else stride
        try:
            self.analyzer = TrafficAnalyzer(
                load_detector(), calibration, fps=self.frames.fps / self.stride, settings=settings
            )
        except BaseException:
            self.frames.release()  # do not keep a camera or file open after a failed start
            raise
        self.last_frame = None
        self.finished = False
        self.rate = 0.0  # processed frames per second, smoothed
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # names the CSV file

    def close(self) -> None:
        self.frames.release()


def dispose_run() -> None:
    run = st.session_state.get("run")
    if run is not None:
        run.close()
    st.session_state.run = None
    st.session_state.running = False


def toggle_running() -> None:
    state = st.session_state
    if state.running:
        state.running = False
        return
    if state.run is not None and state.run.finished:
        dispose_run()  # restart from the beginning
    state.running = True


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
state = st.session_state
state.setdefault("run", None)
state.setdefault("running", False)

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Input source", ["Upload video", "Sample video", "Webcam"], index=1, key="mode")

    source: Source | None = None
    if mode == "Upload video":
        uploaded = st.file_uploader("Traffic video", type=["mp4", "avi"])
        if uploaded is not None:
            source = Source(key=f"upload:{uploaded.name}:{uploaded.size}", target=store_upload(uploaded), live=False)
    elif mode == "Sample video":
        sample_name = st.selectbox("Sample", list(SAMPLES))
        sample = SAMPLES[sample_name]
        try:
            source = Source(
                key=f"sample:{sample_name}",
                target=str(download_sample(sample)),
                live=False,
                preset=sample.calibration,
                speed_limit=sample.speed_limit,
            )
            st.caption(f"Preset: posted limit {sample.speed_limit} km/h, road patch {sample.calibration.width_m:g} x {sample.calibration.length_m:g} m.")
        except Exception as error:  # network trouble must not break the whole page
            st.error(f"Could not download the sample ({error}). Upload a video instead.")
    else:
        camera = st.number_input("Camera index", min_value=0, max_value=9, value=0, step=1)
        st.caption("Uses a camera attached to the machine that runs Streamlit.")
        source = Source(key=f"webcam:{camera}", target=int(camera), live=True)

    # A new source starts from its own calibration preset and a clean slate.
    source_key = source.key if source else "none"
    if state.get("source_key") != source_key:
        dispose_run()
        state.source_key = source_key
        state.preview = None
        apply_preset(source.preset if source and source.preset else Calibration())
        state.speed_limit = source.speed_limit if source else DEFAULT_SPEED_LIMIT

    st.subheader("Detection")
    confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
    speed_limit = st.slider("Speed limit (km/h)", 30, 140, step=5, key="speed_limit")

    with st.expander("Calibration", expanded=True):
        st.caption(
            "Drag the corners onto the road so the amber area covers a stretch of known size, "
            "then enter that size in metres. Speeds are measured below the dashed SPEED ZONE line, "
            "where the road is close enough to measure precisely."
        )
        for name, label in CORNERS:
            st.slider(f"{label} X (% of width)", -50.0, 150.0, key=f"roi_{name}_x", step=0.1, format="%.1f")
            st.slider(f"{label} Y (% of height)", -50.0, 150.0, key=f"roi_{name}_y", step=0.1, format="%.1f")
        st.number_input("Road width (m)", min_value=1.0, max_value=200.0, step=1.0, key="width_m")
        st.number_input("Road length (m)", min_value=5.0, max_value=2000.0, step=5.0, key="length_m")
        st.slider("Counting line position (% down the ROI)", 0, 100, key="line_pos")
        show_grid = st.checkbox(
            "Show metric grid", value=False, help="Lines every few metres on the road, to check the calibration by eye."
        )

    with st.expander("Performance and smoothing"):
        smoothing = st.slider("Speed smoothing buffer (frames)", 5, 30, 10)
        imgsz = st.select_slider("Detection resolution", options=[480, 640, 960, 1280], value=640)
        stride = st.select_slider(
            "Process every Nth frame", options=[1, 2, 3, 4], value=1, disabled=bool(source and source.live)
        )
        realtime = st.checkbox("Play at video speed", value=True, help="Off processes as fast as the machine allows.")

calibration = read_calibration()
settings = AnalyzerSettings(
    confidence=confidence,
    speed_limit_kmh=float(speed_limit),
    smoothing_frames=smoothing,
    imgsz=int(imgsz),
    show_grid=show_grid,
)
problem = calibration.problem()

# --------------------------------------------------------------------------- #
# Create or update the run before drawing anything that depends on it
# --------------------------------------------------------------------------- #
start_error: str | None = None
if source is not None and state.running and state.run is None and problem is None:
    try:
        state.run = Run(source, calibration, settings, stride)
    except RuntimeError as error:  # the camera or file could not be opened
        state.running = False
        start_error = str(error)
    except Exception as error:  # most likely the model could not be loaded or downloaded
        state.running = False
        start_error = (
            f"Could not start the analysis ({type(error).__name__}: {error}). "
            "The first start needs internet access to download yolo11n.pt."
        )

run: Run | None = state.run
if run is not None:
    run.analyzer.configure(settings, calibration if problem is None else None)

# --------------------------------------------------------------------------- #
# Page body
# --------------------------------------------------------------------------- #
st.title("🚦 AI Traffic & Vehicle Speed Detection")
st.caption("YOLO11 detection · ByteTrack IDs · perspective-corrected speed in km/h · virtual line counting")

kpi_columns = st.columns(4)
kpi_slots = [column.container(border=True).empty() for column in kpi_columns]

if run is not None and run.finished:
    button_label = "▶ Restart"
elif state.running:
    button_label = "⏸ Pause"
elif run is not None:
    button_label = "▶ Resume"
else:
    button_label = "▶ Start"

button_a, button_b, status_column = st.columns([1, 1, 4], vertical_alignment="center")
button_a.button(  # a fixed key keeps it one widget while its label changes, so no stale copy lingers
    button_label, type="primary", on_click=toggle_running, key="toggle_run",
    disabled=source is None or problem is not None, width="stretch",
)
button_b.button("↺ Reset", on_click=dispose_run, disabled=run is None, width="stretch")
status_slot = status_column.empty()
progress_slot = st.empty()
video_slot = st.empty()
counts_slot = st.empty()
download_slot = st.empty()
table_slot = st.empty()

if problem:
    st.error(problem)
if start_error:
    st.error(start_error)


def render_kpis(summary: Summary | None) -> None:
    average = "--" if summary is None or summary.avg_speed_kmh is None else f"{summary.avg_speed_kmh:.1f}"
    cards = [
        ("Total Vehicles Counted", "0" if summary is None else str(summary.total_counted),
         "Vehicles that crossed the counting line since the run started."),
        ("Active Vehicles in Frame", "0" if summary is None else str(summary.active),
         "Tracked vehicles currently inside the ROI."),
        ("Average Traffic Speed (km/h)", average,
         "Mean of each measured vehicle's own average speed."),
        ("Speeding Violations", "0" if summary is None else str(summary.violations),
         "Vehicles that stayed above the limit for at least three consecutive measurements."),
    ]
    for slot, (title, value, hint) in zip(kpi_slots, cards):
        slot.metric(title, value, help=hint)


def render_counts(summary: Summary | None) -> None:
    icons = {"Car": "🚗", "Motorcycle": "🏍️", "Bus": "🚌", "Truck": "🚚"}
    counts = summary.counts if summary else dict.fromkeys(icons, 0)
    counts_slot.markdown("  ·  ".join(f"{icons[name]} **{name}** {number}" for name, number in counts.items()))


def render_log(run: Run, button: bool = True) -> None:
    """Refresh the vehicle table and, optionally, the CSV download button."""
    log = run.analyzer.log_dataframe()
    table_slot.dataframe(log, width="stretch", hide_index=True, height=260)
    if button:
        state.csv_renders = state.get("csv_renders", 0) + 1  # a widget key may be used once per run
        download_slot.download_button(
            "⬇ Download CSV log",
            log.to_csv(index=False).encode("utf-8"),
            file_name=f"traffic_log_{run.stamp}.csv",
            mime="text/csv",
            key=f"csv_download_{state.csv_renders}",
            on_click="ignore",  # a click must not rerun the script and stop the live video
            disabled=log.empty,
        )


def show_frame(frame) -> None:
    """Send a BGR frame to the page as a compact JPEG (Streamlit's own encoder is 5x larger)."""
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    video_slot.image(jpeg.tobytes(), width="stretch")


# --------------------------------------------------------------------------- #
# Idle, paused or finished: draw whatever we have, then stop
# --------------------------------------------------------------------------- #
if not state.running:
    if run is not None:
        summary = run.analyzer.summary()
        render_kpis(summary)
        render_counts(summary)
        render_log(run)
        if run.last_frame is not None:
            show_frame(run.last_frame)
        status_slot.caption("Finished." if run.finished else "Paused.")
    else:
        render_kpis(None)
        render_counts(None)
        if source is None:
            status_slot.caption("Choose an input source in the sidebar.")
        else:
            if state.get("preview") is None:
                first = read_first_frame(source.target)
                state.preview = False if first is None else fit_width(first, PREVIEW_WIDTH)
            if state.preview is False:
                st.warning("Could not read a frame from this source.")
            else:
                preview = state.preview.copy()
                if problem is None:
                    geometry = RoadGeometry(calibration, preview.shape[1], preview.shape[0])
                    geometry.draw(preview, show_grid)
                    if geometry.zone_row is None:
                        st.warning(
                            "No part of this ROI is close enough to measure speed reliably. "
                            "Move its far edge nearer the camera or use a higher-resolution video."
                        )
                show_frame(preview)
                status_slot.caption("Align the amber ROI with the road, then press Start.")
    st.stop()

# --------------------------------------------------------------------------- #
# Running: process frames until the video ends or the user interrupts
# --------------------------------------------------------------------------- #
frames = run.frames
device_name = load_detector().device_name
frame_budget = run.stride / frames.fps if realtime and not frames.is_live else 0.0
last_refresh = last_table = last_button = 0.0

# A fresh run starts with empty slots, so refill them at once to avoid a visible blink.
summary = run.analyzer.summary()
render_kpis(summary)
render_counts(summary)
render_log(run)
if run.last_frame is not None:
    show_frame(run.last_frame)

for _ in range(CHUNK_FRAMES):
    started = time.perf_counter()
    item = frames.read(run.stride)
    if item is None:
        run.finished = True
        state.running = False
        break
    frame, timestamp = item
    run.last_frame = run.analyzer.process(frame, timestamp)
    show_frame(run.last_frame)

    now = time.perf_counter()
    instant = 1.0 / max(now - started, 1e-6)
    run.rate = instant if run.rate == 0.0 else 0.9 * run.rate + 0.1 * instant
    if now - last_refresh >= 0.25:
        summary = run.analyzer.summary()
        render_kpis(summary)
        render_counts(summary)
        if frames.frame_count:
            progress_slot.progress(min(frames.position / frames.frame_count, 1.0))
        where = "live" if frames.is_live else f"frame {frames.position} / {frames.frame_count}"
        status_slot.caption(f"Running on {device_name} · {where} · {run.rate:.1f} fps")
        last_refresh = now
    if now - last_table >= 1.5:
        # Replacing the button under the cursor can swallow a click, so it refreshes less often.
        render_log(run, button=now - last_button >= 5.0)
        if now - last_button >= 5.0:
            last_button = now
        last_table = now
    if now - started < frame_budget:
        time.sleep(frame_budget - (now - started))

# Either the video ended (the buttons and CSV then show the final state) or this chunk is done
# (the next run picks up from session state). Both need a fresh script run.
st.rerun()
