"""
export_onnx.py — ONNX Model Export Pipeline.

Usage:
    python export_onnx.py --all
    python export_onnx.py --det --det-model weights/best.pt
    python export_onnx.py --depth --depth-model DPT_Hybrid
"""

import argparse, logging, os, sys
from pathlib import Path
import numpy as np, torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_yolo_onnx(model_path, output_dir, imgsz=640, simplify=False,
                     fp16=False, dynamic=True, opset=17, model_type="detect"):
    """Export YOLOv8 detection/segmentation model to ONNX."""
    from ultralytics import YOLO
    logger.info(f"Exporting YOLOv8 {model_type}: {model_path}")
    model = YOLO(model_path)
    result = model.export(format="onnx", imgsz=imgsz, simplify=simplify,
                          half=fp16, dynamic=dynamic, opset=opset)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        dest = os.path.join(output_dir, Path(result).name)
        if str(result) != dest:
            shutil.move(str(result), dest)
            return dest
    return str(result)


def export_midas_onnx(model_type="MiDaS_small", output_dir="weights",
                      imgsz=256, opset=17, dynamic=True):
    """Export MiDaS depth model to ONNX."""
    logger.info(f"Exporting MiDaS: {model_type}")
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.eval()
    input_size = imgsz if model_type == "MiDaS_small" else 384
    dummy = torch.randn(1, 3, input_size, input_size)
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, f"midas_{model_type.lower()}.onnx")
    dyn_axes = {"input": {0: "batch", 2: "h", 3: "w"},
                "output": {0: "batch", 2: "h", 3: "w"}} if dynamic else None
    torch.onnx.export(model, dummy, onnx_path, opset_version=opset,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes=dyn_axes, do_constant_folding=True)
    try:
        import onnx
        onnx.checker.check_model(onnx.load(onnx_path))
        logger.info("  ONNX verification: PASSED")
    except Exception as e:
        logger.warning(f"  ONNX verification warning: {e}")
    logger.info(f"  Size: {os.path.getsize(onnx_path)/1024/1024:.1f} MB")
    return onnx_path


def verify_onnx(onnx_path, input_shape=(1, 3, 640, 640)):
    """Verify ONNX model with ORT."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        inp = sess.get_inputs()[0]
        dummy = np.random.randn(*input_shape).astype(np.float32)
        sess.run(None, {inp.name: dummy})
        logger.info(f"  Verify {Path(onnx_path).name}: PASSED")
        return True
    except Exception as e:
        logger.error(f"  Verify FAILED: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="Export models to ONNX.")
    p.add_argument("--all", action="store_true")
    p.add_argument("--det", action="store_true")
    p.add_argument("--seg", action="store_true")
    p.add_argument("--depth", action="store_true")
    p.add_argument("--det-model", default="yolov8n.pt")
    p.add_argument("--seg-model", default="yolov8n-seg.pt")
    p.add_argument("--depth-model", default="MiDaS_small",
                   choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"])
    p.add_argument("--output-dir", default="weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--verify", action="store_true", default=True)
    a = p.parse_args()

    if not any([a.all, a.det, a.seg, a.depth]):
        p.error("Specify --all, --det, --seg, or --depth")

    exported = []
    if a.all or a.det:
        exported.append(("Det", export_yolo_onnx(a.det_model, a.output_dir,
                         a.imgsz, a.simplify, a.fp16, True, a.opset, "detect")))
    if a.all or a.seg:
        exported.append(("Seg", export_yolo_onnx(a.seg_model, a.output_dir,
                         a.imgsz, a.simplify, a.fp16, True, a.opset, "segment")))
    if a.all or a.depth:
        exported.append(("Depth", export_midas_onnx(a.depth_model, a.output_dir,
                         opset=a.opset)))
    if a.verify:
        for name, path in exported:
            verify_onnx(path, (1, 3, a.imgsz, a.imgsz))
    for name, path in exported:
        logger.info(f"  {name}: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
