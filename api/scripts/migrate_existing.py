"""Migration script to import existing calibration data into the new API database."""
import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.core.database import init_db, async_session_factory
from api.models.orm import Camera, Calibration, GroundReference, GroundModel, DepthScale


async def migrate_existing_data(db: AsyncSession):
    """Migrate existing calib.npz, ground_ref.json, ground_model.npz, depth_scale_params.npz to database."""
    
    base_path = Path(__file__).parent.parent.parent / "src"
    
    print("=" * 60)
    print("MIGRATING EXISTING DATA TO DATABASE")
    print("=" * 60)
    
    # 1. Create camera record
    print("\n1. Creating camera record...")
    camera = Camera(
        name="CCTV-01",
        rtsp_url=None,
        resolution_w=2752,  # Actual processing resolution
        resolution_h=1536,
        fps=25,
        metadata_={
            "original_calib_resolution": [1280, 720],
            "migrated_from": "legacy_pipeline",
        },
    )
    db.add(camera)
    await db.flush()
    print(f"   Camera created: {camera.id}")
    
    # 2. Load and migrate calib.npz
    print("\n2. Loading calib.npz...")
    calib_path = base_path / "calib.npz"
    if not calib_path.exists():
        raise FileNotFoundError(f"calib.npz not found at {calib_path}")
    
    calib_data = np.load(calib_path)
    print(f"   K_undist shape: {calib_data['K_undist'].shape}")
    print(f"   dist: {calib_data['dist'].ravel()}")
    print(f"   size: {calib_data['size']}")
    print(f"   rms: {calib_data['rms']}")
    
    calibration = Calibration(
        camera_id=camera.id,
        version=1,
        k_matrix=calib_data["K_undist"].tolist(),  # Store K_undist as primary K
        dist_coeffs=calib_data["dist"].ravel().tolist(),
        rms_reproj=float(calib_data["rms"]),
        coverage_w=100.0,  # Estimated
        coverage_h=100.0,
        chessboard_cols=9,
        chessboard_rows=6,
        square_m=0.025,
        calib_images=[str(p) for p in sorted((base_path / "calib_imgs").glob("*.jpg"))],
        notes="Migrated from legacy calib.npz. Original calibration at 1280x720, processing at 2752x1536.",
        is_active=True,
    )
    db.add(calibration)
    await db.flush()
    print(f"   Calibration created: {calibration.id} (v{calibration.version})")
    
    # 3. Load and migrate ground_ref.json
    print("\n3. Loading ground_ref.json...")
    ground_ref_path = base_path / "ground_ref.json"
    if not ground_ref_path.exists():
        raise FileNotFoundError(f"ground_ref.json not found at {ground_ref_path}")
    
    with open(ground_ref_path) as f:
        ground_ref_data = json.load(f)
    
    print(f"   image_points: {ground_ref_data['image_pts']}")
    print(f"   world_points: {ground_ref_data['world_xy']}")
    
    ground_ref = GroundReference(
        camera_id=camera.id,
        calibration_id=calibration.id,
        version=1,
        image_points=ground_ref_data["image_pts"],
        world_points=ground_ref_data["world_xy"],
        world_units="meter",
        description="Migrated from legacy ground_ref.json. 4 corner points of 2x3m rectangle.",
        is_active=True,
    )
    db.add(ground_ref)
    await db.flush()
    print(f"   GroundReference created: {ground_ref.id} (v{ground_ref.version})")
    
    # 4. Load and migrate ground_model.npz
    print("\n4. Loading ground_model.npz...")
    gm_path = base_path / "output" / "ground_model.npz"
    if not gm_path.exists():
        raise FileNotFoundError(f"ground_model.npz not found at {gm_path}")
    
    gm_data = np.load(gm_path)
    print(f"   R: {gm_data['R'].shape}")
    print(f"   t: {gm_data['t'].shape}")
    print(f"   cam_height: {gm_data['cam_height']}")
    print(f"   pitch_down_deg: {gm_data['pitch_down_deg']}")
    print(f"   H_i2w: {gm_data['H_i2w'].shape}")
    
    # Calculate reprojection error from earlier analysis
    reproj_err = 30.3  # From analysis - mean of [30.3, 20.9, 11.9, 19.1]
    
    ground_model = GroundModel(
        camera_id=camera.id,
        calibration_id=calibration.id,
        ground_ref_id=ground_ref.id,
        R_matrix=gm_data["R"].tolist(),
        t_vector=gm_data["t"].ravel().tolist(),
        H_i2w=gm_data["H_i2w"].tolist(),
        H_w2i=gm_data["H_w2i"].tolist(),
        cam_height=float(gm_data["cam_height"]),
        pitch_down_deg=float(gm_data["pitch_down_deg"]),
        reproj_err_px=reproj_err,
        bev_ppm=200,
        bev_margin_m=1.0,
    )
    db.add(ground_model)
    await db.flush()
    print(f"   GroundModel created: {ground_model.id}")
    
    # 5. Load and migrate depth_scale_params.npz
    print("\n5. Loading depth_scale_params.npz...")
    ds_path = base_path / "output" / "depth_scale_params.npz"
    if not ds_path.exists():
        raise FileNotFoundError(f"depth_scale_params.npz not found at {ds_path}")
    
    ds_data = np.load(ds_path)
    print(f"   s: {ds_data['s']}")
    print(f"   b: {ds_data['b']}")
    print(f"   residuals: {ds_data['residuals']}")
    
    residuals = ds_data["residuals"].tolist()
    mean_res = float(np.abs(ds_data["residuals"]).mean())
    max_res = float(np.abs(ds_data["residuals"]).max())
    print(f"   mean_residual: {mean_res:.4f} m")
    print(f"   max_residual: {max_res:.4f} m")
    
    depth_scale = DepthScale(
        camera_id=camera.id,
        ground_model_id=ground_model.id,
        depth_model="yolo26n-depth.pt",
        scale_s=float(ds_data["s"]),
        offset_b=float(ds_data["b"]),
        residuals=residuals,
        mean_residual_m=mean_res,
        max_residual_m=max_res,
        clip_min_m=0.1,
        clip_max_m=50.0,
    )
    db.add(depth_scale)
    await db.flush()
    print(f"   DepthScale created: {depth_scale.id}")
    
    await db.commit()
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Camera ID: {camera.id}")
    print(f"Active Calibration: {calibration.id} (v{calibration.version})")
    print(f"Active Ground Reference: {ground_ref.id} (v{ground_ref.version})")
    print(f"Ground Model: {ground_model.id}")
    print(f"Depth Scale: {depth_scale.id}")
    
    return {
        "camera_id": camera.id,
        "calibration_id": calibration.id,
        "ground_ref_id": ground_ref.id,
        "ground_model_id": ground_model.id,
        "depth_scale_id": depth_scale.id,
    }


async def main():
    await init_db()
    async with async_session_factory() as db:
        try:
            await migrate_existing_data(db)
        except Exception as e:
            await db.rollback()
            print(f"\nMIGRATION FAILED: {e}")
            raise
        finally:
            await db.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())