"""
inference.py
CLI Inference Script for Pothole Detection.

Runs the full multi-task pipeline (detection + segmentation + depth + severity)
on images, videos, or live webcam feed.

Usage:
    # Single image
    python inference.py --source road_image.jpg

    # Video file
    python inference.py --source road_video.mp4

    # Live webcam
    python inference.py --source 0

    # With options
    python inference.py --source road.jpg --det-model weights/best_det.pt \\
                        --depth-model DPT_Hybrid --save-output --no-depth
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.detection import PotholeDetector
from models.segmentation import PotholeSegmentor
from models.depth import DepthEstimator
from pipeline.preprocess import ImagePreprocessor
from pipeline.multitask_inference import MultiTaskPipeline
from pipeline.severity_estimator import SeverityEstimator
from utils.visualization import compose_visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pothole AI — Multi-task inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--source", type=str, required=True,
        help="Image path, video path, or camera index (e.g., '0' for webcam).",
    )

    # Model weights
    parser.add_argument(
        "--det-model", type=str, default="yolov8n.pt",
        help="YOLOv8 detection weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--seg-model", type=str, default="yolov8n-seg.pt",
        help="YOLOv8-Seg segmentation weights (default: yolov8n-seg.pt).",
    )
    parser.add_argument(
        "--depth-model", type=str, default="MiDaS_small",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS depth model variant (default: MiDaS_small).",
    )

    # Thresholds
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'.",
    )

    # Toggles
    parser.add_argument("--no-detect", action="store_true", help="Disable detection.")
    parser.add_argument("--no-segment", action="store_true", help="Disable segmentation.")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth estimation.")

    # Visualization
    parser.add_argument("--show-bbox", action="store_true", default=True, help="Show bounding boxes.")
    parser.add_argument("--show-mask", action="store_true", default=True, help="Show segmentation masks.")
    parser.add_argument("--show-depth", action="store_true", default=True, help="Show depth heatmap.")

    # Output
    parser.add_argument(
        "--save-output", action="store_true",
        help="Save annotated output to disk.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for saved outputs (default: output/).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Do not display output window (useful for headless servers).",
    )

    return parser.parse_args()


def is_camera_source(source: str) -> bool:
    """Check if source is a camera index."""
    try:
        int(source)
        return True
    except ValueError:
        return False


def is_video_source(source: str) -> bool:
    """Check if source is a video file."""
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
    return Path(source).suffix.lower() in video_exts


def process_image(
    pipeline: MultiTaskPipeline,
    image_path: str,
    args: argparse.Namespace,
) -> None:
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return

    logger.info(f"Processing image: {image_path} ({image.shape[1]}x{image.shape[0]})")

    # Run pipeline
    result = pipeline.run(image)

    # Log results
    logger.info(f"  Potholes detected: {result.count}")
    logger.info(f"  Total time: {result.total_time_ms:.1f}ms ({result.fps:.1f} FPS)")

    for i, pothole in enumerate(result.potholes):
        logger.info(
            f"  #{i+1}: conf={pothole.confidence:.2f}, "
            f"area={pothole.mask_area_pixels}px, "
            f"depth={pothole.mean_depth:.3f}, "
            f"severity={pothole.severity} ({pothole.severity_score:.2f})"
        )

    # Visualize
    depth_map = result.depth_result.depth_map if result.depth_result else None
    vis = compose_visualization(
        image, result.potholes, depth_map,
        fps=result.fps,
        show_bbox=args.show_bbox,
        show_mask=args.show_mask,
        show_depth=args.show_depth and not args.no_depth,
    )

    # Save
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"result_{Path(image_path).stem}.jpg"
        )
        cv2.imwrite(out_path, vis)
        logger.info(f"  Saved to: {out_path}")

    # Display
    if not args.no_display:
        cv2.imshow("Pothole AI", vis)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    pipeline: MultiTaskPipeline,
    source: str,
    args: argparse.Namespace,
) -> None:
    """Process video file or camera stream."""
    # Open video source
    if is_camera_source(source):
        cap = cv2.VideoCapture(int(source))
        logger.info(f"Opening camera: {source}")
    else:
        cap = cv2.VideoCapture(source)
        logger.info(f"Opening video: {source}")

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info(f"  Resolution: {width}x{height}, Source FPS: {fps_source:.0f}")

    # Video writer for saving
    writer = None
    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)
        out_name = (
            f"result_cam{source}.mp4"
            if is_camera_source(source)
            else f"result_{Path(source).stem}.mp4"
        )
        out_path = os.path.join(args.output_dir, out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_source, (width, height))
        logger.info(f"  Saving to: {out_path}")

    # FPS smoothing
    fps_history = []
    frame_count = 0

    logger.info("Starting inference... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run pipeline
            result = pipeline.run(frame)

            # Smooth FPS
            fps_history.append(result.fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            smooth_fps = np.mean(fps_history)

            # Visualize
            depth_map = result.depth_result.depth_map if result.depth_result else None
            vis = compose_visualization(
                frame, result.potholes, depth_map,
                fps=smooth_fps,
                show_bbox=args.show_bbox,
                show_mask=args.show_mask,
                show_depth=args.show_depth and not args.no_depth,
            )

            # Save frame
            if writer:
                writer.write(vis)

            # Display
            if not args.no_display:
                cv2.imshow("Pothole AI", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit requested.")
                    break

            # Periodic logging
            if frame_count % 100 == 0:
                logger.info(
                    f"  Frame {frame_count}: {result.count} potholes, "
                    f"{smooth_fps:.1f} FPS"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info(f"Processed {frame_count} frames total.")


def main():
    args = parse_args()

    # Build pipeline
    logger.info("=" * 60)
    logger.info("  Pothole AI — Multi-Task Inference")
    logger.info("=" * 60)

    pipeline = MultiTaskPipeline(
        detector=PotholeDetector(
            model_path=args.det_model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        ) if not args.no_detect else None,
        segmentor=PotholeSegmentor(
            model_path=args.seg_model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
        ) if not args.no_segment else None,
        depth_estimator=DepthEstimator(
            model_type=args.depth_model,
            device=args.device,
        ) if not args.no_depth else None,
        preprocessor=ImagePreprocessor.default(),
        enable_detection=not args.no_detect,
        enable_segmentation=not args.no_segment,
        enable_depth=not args.no_depth,
    )

    # Load models
    logger.info("Loading models...")
    pipeline.load_all_models()
    logger.info("All models loaded. Starting inference...")

    # Route to appropriate handler
    source = args.source

    if is_camera_source(source) or is_video_source(source):
        process_video(pipeline, source, args)
    else:
        # Assume image
        if os.path.isfile(source):
            process_image(pipeline, source, args)
        elif os.path.isdir(source):
            # Process all images in directory
            img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            images = sorted([
                str(f) for f in Path(source).iterdir()
                if f.suffix.lower() in img_exts
            ])
            logger.info(f"Found {len(images)} images in {source}")
            for img_path in images:
                process_image(pipeline, img_path, args)
        else:
            logger.error(f"Source not found: {source}")
            sys.exit(1)


if __name__ == "__main__":
    main()
