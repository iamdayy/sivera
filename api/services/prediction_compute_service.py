"""Storage-free prediction computation, invoked by the bounded vision subprocess."""
import base64
import io
import json
from types import SimpleNamespace

import cv2
import numpy as np

from api.core.config import settings
from api.schemas.prediction import PREDICTION_OUTPUT_TYPES
from api.services.depth_model_service import validate_scale_model
from api.services.depth_scale_service import validate_saved_scale
from api.services.depth_service import apply_scale, depth_colormap, height_heatmap, lift_to_ground, predict_depth
from api.services.ground_model_service import compute_bev
from api.services.image_service import decode_prediction_image, encode_image
from api.services.segment_service import compute_segmentation, overlay_instances
from api.services.volume_service import (
    add_scale_bar, build_job_summary, estimate_per_instance, estimate_volume_convexhull,
    estimate_volume_grid, render_volume_map,
)


COMPUTE_ERRORS = {
    "prediction_invalid_image", "prediction_image_too_large", "prediction_resolution_mismatch",
    "calibration_intrinsics_missing", "calibration_invalid_intrinsics", "prediction_invalid_depth",
    "prediction_invalid_points", "prediction_invalid_grid_parameters", "prediction_volume_grid_too_large",
    "prediction_invalid_segmentation", "prediction_invalid_model", "prediction_encoding_failed",
    "segmentation_model_unavailable", "segmentation_classes_unavailable", "depth_model_unavailable",
    "depth_scale_identity_missing", "depth_scale_model_mismatch", "depth_scale_invalid_saved_fit",
}


def encoded(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def segmentation_payload(segmentation: dict) -> dict:
    mask = segmentation["waste_mask"]
    has_mask = bool(mask.any())
    overlay = np.zeros((*mask.shape, 4), np.uint8) if has_mask else None
    if has_mask:
        overlay[mask] = [60, 80, 240, 110]
    return {
        "has_mask": has_mask,
        "waste_fraction_pct": float(100 * mask.mean()) if has_mask else None,
        "mask_png_base64": encoded(encode_image(mask.astype(np.uint8) * 255, ".png")) if has_mask else None,
        "overlay_png_base64": encoded(encode_image(overlay, ".png")) if has_mask else None,
        "instances": segmentation["instances"],
        "mask_diagnostics": segmentation["mask_diagnostics"],
    }


def remap_segmentation(image: np.ndarray, segmentation: dict, calibration, camera) -> tuple:
    from api.services.prediction_orchestrator import prepare_prediction_image

    prepared = prepare_prediction_image(image, calibration, camera, "rtsp_frame")
    original = np.asarray(calibration.k_original, np.float64)
    target = np.asarray(calibration.k_undist or calibration.k_matrix, np.float64)
    distortion = np.asarray(calibration.dist_coeffs, np.float64)
    maps = cv2.initUndistortRectifyMap(original, distortion, None, target, image.shape[1::-1], cv2.CV_32FC1)

    def transform(mask: np.ndarray) -> np.ndarray:
        return cv2.remap(mask.astype(np.uint8), *maps, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    mask = cv2.imdecode(np.frombuffer(base64.b64decode(segmentation["mask_png_base64"], validate=True), np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != image.shape[:2]:
        raise ValueError("prediction_invalid_segmentation")
    instances = []
    for item in segmentation["instances"]:
        points = np.asarray(item["polygon"], np.float64).reshape(-1, 1, 2)
        polygon = cv2.undistortPoints(points, original, distortion, P=target).reshape(-1, 2)
        if not np.isfinite(polygon).all():
            raise ValueError("prediction_invalid_segmentation")
        instances.append({**item, "polygon": polygon.tolist()})
    return prepared, {**segmentation, "waste_mask": transform(mask), "instances": instances}, transform


def compute_measurement(image: np.ndarray, segmentation: dict, context: dict, options: dict,
                        raw_instances: list | None = None, mask_transform=None) -> dict:
    calibration = SimpleNamespace(**context["calibration"])
    ground = SimpleNamespace(**context["ground_model"])
    scale = SimpleNamespace(**context["depth_scale"])
    mask, instances = segmentation["waste_mask"], segmentation["instances"]
    artifacts, estimates = {}, []
    cell_m = options.get("volume_cell_m", settings.DEFAULT_VOLUME_CELL_M)
    min_points = options.get("volume_min_pts_per_cell", settings.DEFAULT_VOLUME_MIN_PTS_PER_CELL)
    requested = options.get("return_outputs")
    requested = set(PREDICTION_OUTPUT_TYPES if requested is None else requested)

    def add_bytes(kind: str, content: bytes, mime: str, extension: str, width=None, height=None, meta=None) -> None:
        if kind in requested:
            artifacts[kind] = {"data": encoded(content), "mime_type": mime, "extension": extension,
                               "width": width, "height": height, "meta": meta}

    def add_image(kind: str, value: np.ndarray, meta=None) -> None:
        if kind in requested:
            extension = ".png" if kind == "mask" else ".jpg"
            add_bytes(kind, encode_image(value, extension), "image/png" if extension == ".png" else "image/jpeg",
                      extension[1:], value.shape[1], value.shape[0], meta)

    add_image("mask", mask.astype(np.uint8) * 255, {"waste_pixels": int(mask.sum()),
              "mask_diagnostics": segmentation.get("mask_diagnostics"), "coordinate_space": "undistorted"})
    if "overlay" in requested:
        drawable = [{"cls": item["cls"], "conf": item["conf"], "poly": np.asarray(item["polygon"])} for item in instances]
        add_image("overlay", overlay_instances(image, drawable))
    if "instances_json" in requested:
        add_bytes("instances_json", json.dumps(instances, allow_nan=False).encode(), "application/json", "json")
    volume_results = {"volume_grid": None, "volume_convex_hull": None, "instances": []}
    depth_results = {"n_points": 0, "height_stats": None, "footprint": None}
    if mask.any():
        model = validate_scale_model(scale)
        validate_saved_scale(scale)
        depth_raw = predict_depth(image, str(model), settings.DEFAULT_DEPTH_IMGSZ, "cpu")
        validate_scale_model(scale)
        if depth_raw.shape != mask.shape:
            raise ValueError("prediction_invalid_depth")
        depth_m = apply_scale(depth_raw, scale.scale_s, scale.offset_b, scale.clip_min_m, scale.clip_max_m)
        X, Y, H, uv = lift_to_ground(np.asarray(calibration.k_matrix), np.asarray(ground.R_matrix),
                                    np.asarray(ground.t_vector).reshape(3, 1), depth_m, mask)
        grid_volume, grid_area, grid, grid_info = estimate_volume_grid(X, Y, H, cell_m, min_points)
        hull_volume, hull_area = estimate_volume_convexhull(X, Y, H)
        for method, volume, area in (("grid", grid_volume, grid_area), ("convex_hull", hull_volume, hull_area)):
            if volume is not None and np.isfinite(volume) and volume >= 0:
                estimates.append({
                    "method": method, "volume_m3": volume, "volume_liter": volume * 1000,
                    "footprint_m2": area, "cell_m": cell_m if method == "grid" else None,
                    "height_mean_m": float(H.mean()), "height_median_m": float(np.median(H)),
                    "n_cells": int(np.isfinite(grid).sum()) if method == "grid" else None, "n_points": len(H),
                })
                volume_results["volume_grid" if method == "grid" else "volume_convex_hull"] = {
                    "volume_m3": volume, "volume_liter": volume * 1000, "footprint_m2": area,
                    **({"cell_m": cell_m} if method == "grid" else {}),
                }
        volume_results["instances"] = estimate_per_instance(
            instances if raw_instances is None else raw_instances,
            {"X": X, "Y": Y, "height": H, "uv": uv}, image.shape[:2], cell_m, min_points, mask_transform,
        )
        depth_results = {
            "n_points": len(H),
            "height_stats": {"min": float(H.min()), "max": float(H.max()), "mean": float(H.mean()),
                             "median": float(np.median(H)), "std": float(H.std())} if len(H) else None,
            "footprint": {"x_min": float(X.min()), "x_max": float(X.max()),
                          "y_min": float(Y.min()), "y_max": float(Y.max())} if len(H) else None,
        }
        if "depth_raw_vis" in requested:
            add_image("depth_raw_vis", depth_colormap(depth_raw))
        if "depth_scaled_vis" in requested:
            add_image("depth_scaled_vis", depth_colormap(depth_m))
        if "depth_masked" in requested:
            add_image("depth_masked", depth_colormap(depth_m, mask))
        if "height_map" in requested:
            add_image("height_map", cv2.addWeighted(image, 0.6, height_heatmap(X, Y, H, uv, image.shape), 0.7, 0))
        if "lifted_points_npz" in requested:
            buffer = io.BytesIO()
            np.savez(buffer, X=X, Y=Y, height=H, uv=uv)
            add_bytes("lifted_points_npz", buffer.getvalue(), "application/octet-stream", "npz")
        if "volume_heatmap" in requested:
            add_image("volume_heatmap", add_scale_bar(render_volume_map(grid, grid_info), cell_m))
        if "volume_result_npz" in requested:
            buffer = io.BytesIO()
            np.savez(buffer, vol_grid=grid_volume if grid_volume is not None else np.nan,
                     foot_grid=grid_area if grid_area is not None else np.nan,
                     vol_hull=hull_volume if hull_volume is not None else np.nan,
                     foot_hull=hull_area if hull_area is not None else np.nan,
                     grid_height=grid, cell_m=cell_m, x_min=grid_info.get("x_min", 0),
                     y_min=grid_info.get("y_min", 0), nx=grid_info.get("nx", 0), ny=grid_info.get("ny", 0))
            add_bytes("volume_result_npz", buffer.getvalue(), "application/octet-stream", "npz")
        if requested & {"bev_image", "bev_mask_proyeksi", "bev_overlay"}:
            mapping = {"H_i2w": np.asarray(ground.H_i2w), "H_w2i": np.asarray(ground.H_w2i),
                       "R": np.asarray(ground.R_matrix), "t": np.asarray(ground.t_vector).reshape(3, 1),
                       "K": np.asarray(calibration.k_matrix), "cam_height": ground.cam_height,
                       "pitch_down_deg": ground.pitch_down_deg}
            bev, bev_mask, overlay, info = compute_bev(image, mask.astype(np.uint8), mapping,
                                                     ppm=ground.bev_ppm, margin_m=ground.bev_margin_m)
            add_image("bev_image", bev, {"bev_info": info})
            add_image("bev_mask_proyeksi", bev_mask)
            add_image("bev_overlay", overlay)
    summary = build_job_summary(mask, instances, volume_results, depth_results)
    rows = []
    for index, item in enumerate(instances):
        volume = volume_results["instances"][index] if index < len(volume_results["instances"]) else {}
        rows.append({"cls_name": item["cls"], "confidence": item["conf"], "polygon_px": item["polygon"],
                     "polygon_world": [], "volume_m3": volume.get("volume_m3"),
                     "footprint_m2": volume.get("footprint_m2"), "n_pixels": volume.get("n_pts", 0)})
    return {"summary": summary, "volumes": estimates, "instances": rows, "artifacts": artifacts}


def compute_request(mode: str, payload: dict) -> dict:
    cv2.setNumThreads(settings.VISION_CPU_THREADS)
    options = payload["options"]
    # Internal lossless images can be larger than the already-validated user upload.
    config = settings.model_copy(update={"MAX_UPLOAD_SIZE_MB": settings.VISION_MAX_MESSAGE_BYTES // 1048576})
    image = decode_prediction_image(base64.b64decode(payload["image"], validate=True), config)
    job = SimpleNamespace(seg_model=options.get("seg_model"), seg_conf=options.get("seg_conf", settings.DEFAULT_SEG_CONF),
                          seg_classes=options.get("seg_classes"))
    if mode == "segment":
        return segmentation_payload(compute_segmentation(job, image))
    context = payload["context"]
    calibration, camera = SimpleNamespace(**context["calibration"]), SimpleNamespace(**context["camera"])
    if mode == "measure":
        prepared, segmentation, transform = remap_segmentation(image, payload["segmentation"], calibration, camera)
        return compute_measurement(prepared, segmentation, context, options, payload["segmentation"]["instances"], transform)
    from api.services.prediction_orchestrator import prepare_prediction_image
    prepared = prepare_prediction_image(image, calibration, camera, options["source_type"])
    segmentation = compute_segmentation(job, prepared)
    return compute_measurement(prepared, segmentation, context, options)
