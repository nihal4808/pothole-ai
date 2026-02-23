"""
ui/streamlit_app.py
Streamlit Dashboard for Pothole Detection and Severity Estimation.

Features:
- Image/video upload mode
- Live webcam detection
- Toggleable overlays (bbox, mask, depth)
- Severity scores, confidence display, FPS counter
- Dark-themed modern UI

Usage:
    streamlit run ui/streamlit_app.py
"""

import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.detection import PotholeDetector
from models.segmentation import PotholeSegmentor
from models.depth import DepthEstimator
from pipeline.preprocess import ImagePreprocessor
from pipeline.multitask_inference import MultiTaskPipeline
from pipeline.severity_estimator import SeverityEstimator
from utils.visualization import compose_visualization


# ─── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pothole AI — Detection & Severity",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 20px; margin: 8px 0;
        border: 1px solid #2a2a4a; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 4px; }
    .severity-low { color: #00e676; font-weight: 700; }
    .severity-medium { color: #ffab00; font-weight: 700; }
    .severity-high { color: #ff1744; font-weight: 700; }
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.2rem; font-weight: 800;
    }
    div[data-testid="stSidebar"] { background: #0a0f1a; }
</style>
""", unsafe_allow_html=True)


# ─── Pipeline Initialization ────────────────────────────────────────

@st.cache_resource
def load_pipeline(det_model, seg_model, depth_model, device, conf):
    """Load and cache the multi-task pipeline."""
    pipeline = MultiTaskPipeline(
        detector=PotholeDetector(model_path=det_model, conf_threshold=conf, device=device),
        segmentor=PotholeSegmentor(model_path=seg_model, conf_threshold=conf, device=device),
        depth_estimator=DepthEstimator(model_type=depth_model, device=device),
        preprocessor=ImagePreprocessor.default(),
        severity_estimator=SeverityEstimator(),
    )
    pipeline.load_all_models()
    return pipeline


# ─── Sidebar ────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar controls and return config dict."""
    with st.sidebar:
        st.markdown('<p class="header-gradient">🕳️ Pothole AI</p>', unsafe_allow_html=True)
        st.caption("Multi-Task Deep Learning Framework")
        st.divider()

        # Mode selection
        mode = st.radio("📸 Input Mode", ["Upload Image", "Upload Video", "Live Camera"],
                        index=0)
        st.divider()

        # Model config
        st.subheader("⚙️ Model Settings")
        device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
        det_model = st.text_input("Detection Model", "yolov8n.pt")
        seg_model = st.text_input("Segmentation Model", "yolov8n-seg.pt")
        depth_model = st.selectbox("Depth Model",
                                   ["MiDaS_small", "DPT_Hybrid", "DPT_Large"], index=0)
        st.divider()

        # Overlay toggles
        st.subheader("🎨 Overlay Controls")
        show_bbox = st.checkbox("Bounding Boxes", value=True)
        show_mask = st.checkbox("Segmentation Mask", value=True)
        show_depth = st.checkbox("Depth Heatmap", value=False)
        show_fps = st.checkbox("FPS Counter", value=True)
        st.divider()

        st.caption("Built with YOLOv8 + MiDaS + Streamlit")

    return {
        "mode": mode, "device": device, "conf": conf,
        "det_model": det_model, "seg_model": seg_model,
        "depth_model": depth_model, "show_bbox": show_bbox,
        "show_mask": show_mask, "show_depth": show_depth,
        "show_fps": show_fps,
    }


# ─── Metric Cards ───────────────────────────────────────────────────

def render_metrics(result):
    """Render metric cards for pipeline results."""
    cols = st.columns(4)

    with cols[0]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{result.count}</div>
            <div class="metric-label">Potholes Detected</div>
        </div>""", unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{result.fps:.1f}</div>
            <div class="metric-label">FPS</div>
        </div>""", unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{result.total_time_ms:.0f}ms</div>
            <div class="metric-label">Total Latency</div>
        </div>""", unsafe_allow_html=True)

    with cols[3]:
        max_sev = "NONE"
        if result.potholes:
            sev_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": 0}
            max_sev = max(result.potholes, key=lambda p: sev_order.get(p.severity, 0)).severity
        css_class = f"severity-{max_sev.lower()}" if max_sev != "NONE" else ""
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value {css_class}">{max_sev}</div>
            <div class="metric-label">Max Severity</div>
        </div>""", unsafe_allow_html=True)


def render_pothole_table(result):
    """Render detailed pothole info table."""
    if not result.potholes:
        st.info("No potholes detected in this frame.")
        return

    st.subheader("📊 Detection Details")
    for i, p in enumerate(result.potholes):
        sev_class = f"severity-{p.severity.lower()}"
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Pothole #{i+1}", f"{p.confidence:.0%}")
        with col2:
            st.metric("Area", f"{p.mask_area_pixels:,} px")
        with col3:
            st.metric("Depth", f"{p.mean_depth:.3f}")
        with col4:
            st.markdown(f'<span class="{sev_class}" style="font-size:1.3rem">'
                        f'{p.severity} ({p.severity_score:.0%})</span>',
                        unsafe_allow_html=True)


# ─── Image Processing ───────────────────────────────────────────────

def process_and_display(pipeline, image, cfg):
    """Run pipeline and display results."""
    result = pipeline.run(image)
    depth_map = result.depth_result.depth_map if result.depth_result else None

    vis = compose_visualization(
        image, result.potholes, depth_map,
        fps=result.fps,
        show_bbox=cfg["show_bbox"],
        show_mask=cfg["show_mask"],
        show_depth=cfg["show_depth"],
        show_fps=cfg["show_fps"],
    )

    # Convert BGR to RGB for Streamlit
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    render_metrics(result)
    st.image(vis_rgb, caption="Detection Result", use_container_width=True)
    render_pothole_table(result)

    # Timing breakdown
    with st.expander("⏱️ Timing Breakdown"):
        for key, val in result.timing_breakdown.items():
            st.text(f"  {key}: {val:.1f} ms")


# ─── Main App ───────────────────────────────────────────────────────

def main():
    cfg = render_sidebar()

    # Header
    st.markdown('<h1 class="header-gradient">Pothole Detection & Severity Estimation</h1>',
                unsafe_allow_html=True)
    st.caption("Hybrid Multi-Task Deep Learning Framework — Real-Time Analysis")
    st.divider()

    # Load pipeline
    try:
        with st.spinner("🔄 Loading AI models... This may take a moment."):
            pipeline = load_pipeline(
                cfg["det_model"], cfg["seg_model"], cfg["depth_model"],
                cfg["device"], cfg["conf"],
            )
        st.success("✅ Models loaded successfully!", icon="🚀")
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.stop()

    # ── Upload Image Mode ────────────────────────────────────────
    if cfg["mode"] == "Upload Image":
        uploaded = st.file_uploader("Upload a road image",
                                    type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                process_and_display(pipeline, image, cfg)
            else:
                st.error("Could not decode the uploaded image.")

    # ── Upload Video Mode ────────────────────────────────────────
    elif cfg["mode"] == "Upload Video":
        uploaded = st.file_uploader("Upload a road video",
                                    type=["mp4", "avi", "mkv", "mov"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            metrics_placeholder = st.empty()
            stop_btn = st.button("⏹️ Stop Processing")

            frame_count = 0
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 3 != 0:  # Process every 3rd frame for speed
                    continue

                result = pipeline.run(frame)
                depth_map = result.depth_result.depth_map if result.depth_result else None
                vis = compose_visualization(
                    frame, result.potholes, depth_map,
                    fps=result.fps, show_bbox=cfg["show_bbox"],
                    show_mask=cfg["show_mask"], show_depth=cfg["show_depth"],
                    show_fps=cfg["show_fps"],
                )
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                stframe.image(vis_rgb, use_container_width=True)

            cap.release()
            st.success(f"✅ Processed {frame_count} frames.")

    # ── Live Camera Mode ─────────────────────────────────────────
    elif cfg["mode"] == "Live Camera":
        st.warning("📷 Live camera requires a connected webcam.")
        run = st.checkbox("Start Camera", value=False)

        if run:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while run and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera read failed.")
                    break

                result = pipeline.run(frame)
                depth_map = result.depth_result.depth_map if result.depth_result else None
                vis = compose_visualization(
                    frame, result.potholes, depth_map,
                    fps=result.fps, show_bbox=cfg["show_bbox"],
                    show_mask=cfg["show_mask"], show_depth=cfg["show_depth"],
                    show_fps=cfg["show_fps"],
                )
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                stframe.image(vis_rgb, use_container_width=True)

            cap.release()


if __name__ == "__main__":
    main()
