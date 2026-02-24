import asyncio
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pothole AI - Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for the frontend
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Initializing multi-task AI pipeline...")
    # Load default models on startup (auto selects device)
    pipeline = MultiTaskPipeline(
        detector=PotholeDetector(model_path="yolov8n.pt", conf_threshold=0.25, device="auto"),
        segmentor=PotholeSegmentor(model_path="yolov8n-seg.pt", conf_threshold=0.25, device="auto"),
        depth_estimator=DepthEstimator(model_type="MiDaS_small", device="auto"),
        preprocessor=ImagePreprocessor.default(),
        severity_estimator=SeverityEstimator(),
    )
    pipeline.load_all_models()
    logger.info("Pipeline initialized successfully.")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return "Frontend not built yet. Create static/index.html."
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_base64(image_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze a single uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        return {"error": "Failed to decode image."}

    # Run inference
    result = pipeline.run(image_bgr)
    
    depth_map = result.depth_result.depth_map if result.depth_result else None
    
    vis = compose_visualization(
        image_bgr, result.potholes, depth_map,
        fps=result.fps, show_bbox=True, show_mask=True, show_depth=False, show_fps=False
    )
    
    vis_b64 = encode_image_base64(vis)
    
    max_severity = "NONE"
    if result.potholes:
        sev_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": 0}
        max_severity = max(result.potholes, key=lambda p: sev_order.get(p.severity, 0)).severity
    
    return {
        "image_b64": f"data:image/jpeg;base64,{vis_b64}",
        "metrics": {
            "count": result.count,
            "latency_ms": round(result.total_time_ms, 1),
            "max_severity": max_severity,
            "fps": round(result.fps, 1)
        }
    }

@app.websocket("/ws/camera")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming from client camera.
    Receives base64 encoded JPEG frames, processes them, returns annotated base64 frames + metrics.
    """
    await websocket.accept()
    logger.info("Client connected to webcam stream.")
    
    try:
        while True:
            # Receive text data (base64 data URI format)
            data = await websocket.receive_text()
            
            # Options (can be sent via a config message, but for simplicity assuming defaults)
            show_bbox = True
            show_mask = True
            show_depth = False
            
            try:
                # Strip data:image/jpeg;base64,
                if "," in data:
                    b64_str = data.split(",")[1]
                else:
                    b64_str = data
                    
                image_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                # Process frame
                result = pipeline.run(frame)
                
                depth_map = result.depth_result.depth_map if result.depth_result else None
                
                vis = compose_visualization(
                    frame, result.potholes, depth_map,
                    fps=result.fps, show_bbox=show_bbox, show_mask=show_mask, show_depth=show_depth, show_fps=True
                )
                
                vis_b64 = encode_image_base64(vis)
                
                max_severity = "NONE"
                if result.potholes:
                    sev_order = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "UNKNOWN": 0}
                    max_severity = max(result.potholes, key=lambda p: sev_order.get(p.severity, 0)).severity
                
                # Send back response JSON
                response = {
                    "image_b64": f"data:image/jpeg;base64,{vis_b64}",
                    "metrics": {
                        "count": result.count,
                        "latency_ms": round(result.total_time_ms, 1),
                        "max_severity": max_severity,
                        "fps": round(result.fps, 1)
                    }
                }
                
                await websocket.send_text(json.dumps(response))
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                
    except WebSocketDisconnect:
        logger.info("Client disconnected from webcam stream.")
