"""Segmentation service: wraps segment.py logic."""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.storage import get_storage, generate_storage_key
from api.models.orm import PredictionJob


COLORS = {
    "container": (255, 140, 0),
    "sampah_berserakan": (0, 165, 255),
    "sampah_overload": (0, 0, 255),
}

# Lazy model cache: loads each YOLO model once per process
_seg_model_cache: dict = {}


def norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def clean_mask(mask: np.ndarray, open_px: int = 0, close_px: int = 5, min_area_px: int = 30) -> np.ndarray:
    """Clean mask with morphological operations."""
    m = mask.astype(np.uint8)
    if open_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if min_area_px:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                keep[lab == i] = 1
        m = keep
    return m.astype(bool)


def predict_instances(
    image_bgr: np.ndarray,
    model_path: str,
    device: str = "cpu",
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.7,
) -> Tuple[List[dict], dict]:
    """Predict instances using YOLO segmentation model."""
    from ultralytics import YOLO
    
    # Lazy-load and cache the model per process
    if model_path not in _seg_model_cache:
        _seg_model_cache[model_path] = YOLO(model_path)
    model = _seg_model_cache[model_path]
    
    r = model.predict(source=image_bgr, device=device, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    
    out = []
    if r.masks is not None:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        for poly, c, p in zip(r.masks.xy, cls, cf):
            if len(poly) >= 3:
                out.append(dict(
                    cls=model.names[int(c)],
                    conf=float(p),
                    poly=np.asarray(poly, np.float32)
                ))
    return out, model.names


def rasterize(polys: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize polygons to binary mask."""
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p)).astype(np.int32)], 1)
    return m.astype(bool)


def class_masks(instances: List[dict], shape: Tuple[int, int]) -> dict:
    """Group masks by class."""
    by = {}
    for ins in instances:
        by.setdefault(ins["cls"], []).append(ins["poly"])
    return {k: rasterize(v, shape) for k, v in by.items()}


def waste_mask_from(instances: List[dict], shape: Tuple[int, int], waste_classes: List[str]) -> np.ndarray:
    """Create combined waste mask from specified classes."""
    keep = {norm(c) for c in waste_classes}
    return rasterize([i["poly"] for i in instances if norm(i["cls"]) in keep], shape)


def overlay_instances(image_bgr: np.ndarray, instances: List[dict], alpha: float = 0.45) -> np.ndarray:
    """Draw instance overlays on image."""
    out = image_bgr.copy()
    for ins in instances:
        col = COLORS.get(norm(ins["cls"]), (0, 200, 0))
        m = rasterize([ins["poly"]], out.shape[:2])
        out[m] = (alpha * np.array(col) + (1 - alpha) * out[m]).astype(np.uint8)
        cv2.polylines(out, [np.round(ins["poly"]).astype(np.int32)], True, col, 2)
        x, y = np.round(ins["poly"].min(axis=0)).astype(int)
        cv2.putText(out, f"{ins['cls']} {ins['conf']:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


async def run_segmentation(
    db: AsyncSession,
    job: PredictionJob,
    image: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Run segmentation and save outputs."""
    storage = get_storage()
    
    # Load segmentation model
    seg_model_path = Path(settings.MODELS_DIR) / (job.seg_model or settings.DEFAULT_SEG_MODEL)
    if not seg_model_path.exists():
        raise FileNotFoundError(f"Segmentation model not found: {job.seg_model or settings.DEFAULT_SEG_MODEL} (looked in {settings.MODELS_DIR})")
    
    waste_classes = job.seg_classes or ["Sampah_Berserakan", "Sampah_Overload"]
    
    # Predict instances
    instances, names = predict_instances(
        image, str(seg_model_path),
        device="cpu", imgsz=job.seg_conf and settings.DEFAULT_SEG_IMGSZ or settings.DEFAULT_SEG_IMGSZ,
        conf=job.seg_conf
    )
    
    # Check if waste classes exist in model
    have = {norm(v) for v in names.values()} if isinstance(names, dict) else {norm(v) for v in names}
    missing = [c for c in waste_classes if norm(c) not in have]
    if missing:
        raise ValueError(f"Classes {missing} not found in model. Available: {sorted(have)}")
    
    # Create masks per class
    h, w = image.shape[:2]
    class_masks_dict = class_masks(instances, (h, w))
    
    # Combined waste mask
    waste_mask = clean_mask(
        waste_mask_from(instances, (h, w), waste_classes),
        open_px=0, close_px=5, min_area_px=30
    )
    
    # Save individual class masks
    for cls_name, cls_mask in class_masks_dict.items():
        key = generate_storage_key(str(job.camera_id), str(job.id), f"mask_{norm(cls_name)}", "png")
        import io
        buf = io.BytesIO()
        cv2.imwrite(buf, (cls_mask * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        buf.seek(0)
        storage.save(key, buf, "image/png")
    
    # Save combined waste mask
    mask_key = generate_storage_key(str(job.camera_id), str(job.id), "mask", "png")
    buf = io.BytesIO()
    cv2.imwrite(buf, (waste_mask * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    buf.seek(0)
    storage.save(mask_key, buf, "image/png")
    
    # Save instances JSON
    inst_json = [
        dict(cls=i["cls"], conf=i["conf"], polygon=np.round(i["poly"], 1).tolist())
        for i in instances
    ]
    inst_key = generate_storage_key(str(job.camera_id), str(job.id), "instances_json", "json")
    buf = io.BytesIO()
    buf.write(json.dumps(inst_json, indent=2).encode())
    buf.seek(0)
    storage.save(inst_key, buf, "application/json")
    
    # Save overlay
    overlay_img = overlay_instances(image, instances)
    overlay_key = generate_storage_key(str(job.camera_id), str(job.id), "overlay", "jpg")
    buf = io.BytesIO()
    cv2.imwrite(buf, overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    buf.seek(0)
    storage.save(overlay_key, buf, "image/jpeg")
    
    return {
        "instances": inst_json,
        "waste_mask": waste_mask,
        "class_masks": {k: v for k, v in class_masks_dict.items()},
        "mask_key": mask_key,
        "overlay_key": overlay_key,
        "instances_json_key": inst_key,
        "total_waste_pixels": int(waste_mask.sum()),
        "waste_fraction": float(waste_mask.mean()),
    }