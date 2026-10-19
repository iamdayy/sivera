"""Tahap 4 - Estimasi depth map + proyeksi ke koordinat tanah (meter).

Alur setelah ipm.py:
  1) Prediksi depth map dari citra undistort menggunakan model YOLO depth:
       python depth.py predict --image output/undist.jpg --model yolo26n-depth.pt \
           --out output

  2) Kalibrasi skala depth (relatif → meter absolut) menggunakan titik acuan
     dari ground_model.npz. Model depth monocular menghasilkan nilai RELATIF;
     tahap ini menentukan faktor skala (s) dan offset (b) via least squares dari
     titik acuan yang sudah diketahui jaraknya ke kamera:
       python depth.py scale --depth output/depth_raw.npy \
           --ground output/ground_model.npz --ref ground_ref.json \
           --out output

  3) Lift piksel sampah ke koordinat 3D bidang tanah (X, Y, tinggi di atas tanah):
       python depth.py lift --image output/undist.jpg \
           --depth output/depth_scaled.npy --mask output/mask.png \
           --ground output/ground_model.npz --out output

  4) (Opsional) Tampilkan peta tinggi sampah sebagai heatmap:
       python depth.py vis --depth output/depth_scaled.npy \
           --mask output/mask.png --out output

CATATAN:
  * Depth model YOLO (monocular) menghasilkan nilai RELATIF, bukan meter.
    Kalibrasi skala wajib dilakukan sebelum lift jika butuh nilai metrik.
  * Akurasi skala bergantung pada sebaran titik acuan: makin banyak titik
    dengan variasi jarak ke kamera, makin stabil estimasi skala.
  * Depth di tepi citra bisa tidak akurat pada model monocular — hindari
    mengandalkan piksel < 5% dari tepi untuk perhitungan volume.
  * lift_to_ground menggunakan konvensi dunia yang sama dengan ipm.py:
    bidang tanah Z_world = 0, Z_world positif ke atas. Output X, Y
    langsung aligned ke world frame — konsisten dengan ground_ref.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ================================================================ prediksi
def predict_depth(image_bgr, model_path, imgsz=768, device="cpu"):
    """Prediksi depth map dari citra BGR.

    Mengembalikan depth map (H, W) float32 dengan nilai RELATIF (belum meter).
    Nilai lebih besar = lebih jauh (sesuai konvensi model YOLO depth).
    """
    from ultralytics import YOLO  # lazy import agar modul bisa ditest tanpa ultralytics

    model = YOLO(model_path)
    results = model.predict(source=image_bgr, imgsz=imgsz, device=device, verbose=False)
    res = results[0]

    if getattr(res, "depth", None) is None:
        raise RuntimeError(
            "Model tidak menghasilkan atribut 'depth'. "
            "Pastikan model adalah YOLO depth estimation dan versi Ultralytics >= 8.3."
        )

    depth = res.depth
    # Ultralytics bisa mengembalikan tensor atau ndarray tergantung versi
    if hasattr(depth, "data"):
        depth = depth.data
    if hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))

    h, w = image_bgr.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth


# ================================================================ kalibrasi skala
def _pixel_to_ray(K, u, v):
    """Arah sinar kamera (belum dinormalisasi) untuk piksel (u, v)."""
    return np.array([(u - K[0, 2]) / K[0, 0],
                     (v - K[1, 2]) / K[1, 1],
                     1.0], dtype=np.float64)


def _depth_from_pose(K, R, t, world_pts):
    """Hitung depth sepanjang sumbu optik (Z kamera) untuk titik dunia 3D."""
    obj = np.asarray(world_pts, np.float64)       # (N, 3)
    cam = (R @ obj.T).T + t.ravel()               # titik di koordinat kamera
    return cam[:, 2]                               # Z kamera = depth optik


def calibrate_scale(depth_raw, K, R, t, img_pts, world_xy, min_pts=3):
    """Hitung faktor skala (s) dan offset (b): depth_meter = s * depth_raw + b.

    Menggunakan titik acuan bidang tanah (Z=0) yang sudah diketahui posisi
    dunianya. Regresi linear least squares: d_gt = s * d_raw + b.

    Parameter
    ---------
    depth_raw : ndarray (H, W)  — depth relatif dari model
    K         : ndarray (3, 3)  — matriks intrinsik (undistorted)
    R, t      : pose kamera dari ground_model.npz
    img_pts   : (N, 2) piksel titik acuan di citra
    world_xy  : (N, 2) koordinat dunia titik acuan (Z=0)
    min_pts   : jumlah titik minimum untuk regresi

    Mengembalikan
    -------------
    s, b   : float   — depth_meter ≈ s * depth_raw + b
    res_px : ndarray — residual per titik (meter)
    """
    img_pts = np.asarray(img_pts, np.float64)
    world_xy = np.asarray(world_xy, np.float64)
    world_pts = np.hstack([world_xy, np.zeros((len(world_xy), 1))])   # Z=0 di tanah

    d_gt = _depth_from_pose(K, R, t, world_pts)   # depth GT (meter, sumbu optik)
    h, w = depth_raw.shape[:2]

    d_raw = []
    valid_gt = []
    for (u, v), dg in zip(img_pts, d_gt):
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < h and 0 <= ui < w and dg > 0:
            dr = float(depth_raw[vi, ui])
            if dr > 0 and np.isfinite(dr):
                d_raw.append(dr)
                valid_gt.append(dg)

    if len(valid_gt) < min_pts:
        raise RuntimeError(
            f"Hanya {len(valid_gt)} titik valid untuk kalibrasi skala; "
            f"minimal {min_pts}. Tambah titik acuan atau periksa depth map."
        )

    d_raw = np.array(d_raw, np.float64)
    d_gt = np.array(valid_gt, np.float64)

    # Least squares: [d_raw, 1] @ [s, b]^T = d_gt
    A = np.stack([d_raw, np.ones_like(d_raw)], axis=1)
    (s, b), *_ = np.linalg.lstsq(A, d_gt, rcond=None)

    residuals = (s * d_raw + b) - d_gt
    return float(s), float(b), residuals


def apply_scale(depth_raw, s, b, clip_min=0.1, clip_max=50.0):
    """Terapkan skala linear, clip ke rentang masuk akal (meter).

    clip_min : jarak minimum yang dianggap valid (hindari nilai noise < 0)
    clip_max : batas maksimum — sesuaikan dengan jangkauan kamera di lapangan
    """
    depth_m = s * depth_raw.astype(np.float32) + b
    return np.clip(depth_m, clip_min, clip_max)


# ================================================================ lift ke tanah
def lift_to_ground(K, R, t, depth_m, waste_mask):
    """Proyeksikan piksel sampah ke koordinat bidang tanah menggunakan depth.

    Konvensi koordinat dunia (sama dengan ipm.py / solvePnP):
      - Bidang tanah ada di Z_world = 0
      - Z_world positif = ke atas
      - X_world, Y_world sesuai titik acuan yang dipakai pada solvePnP

    Alur:
      1. Rekonstruksi titik 3D di koordinat kamera dari depth + piksel.
      2. Transform balik ke koordinat dunia: P_world = R^T @ P_cam + cam_pos
      3. Height = Z_world (sudah dalam meter, langsung tanpa perlu h_cam).
      4. X, Y = komponen X_world dan Y_world (aligned ke world frame).

    Mengembalikan
    -------------
    X, Y   : ndarray (N,) — posisi di bidang tanah, aligned ke world frame (meter)
    height : ndarray (N,) — tinggi tiap piksel di atas tanah (Z_world, meter, >= 0)
    uv     : ndarray (N, 2) — koordinat piksel yang diproses [u, v]
    """
    v_idx, u_idx = np.nonzero(waste_mask)
    z = depth_m[v_idx, u_idx]                   # depth sepanjang sumbu optik kamera

    # Rekonstruksi titik 3D di koordinat kamera
    P_cam = np.stack([
        (u_idx - K[0, 2]) / K[0, 0] * z,
        (v_idx - K[1, 2]) / K[1, 1] * z,
        z
    ], axis=1)                                   # (N, 3)

    # Transform ke koordinat dunia: P_world = R^T @ P_cam + cam_pos
    # R^T = R^{-1} karena R adalah matriks rotasi ortonormal
    cam_pos = -R.T @ t.ravel()                   # posisi kamera di koordinat dunia
    P_world = (R.T @ P_cam.T).T + cam_pos        # (N, 3)

    # Bidang tanah Z_world = 0 → tinggi = Z_world titik tersebut
    height = np.clip(P_world[:, 2], 0.0, None)   # (N,)
    X = P_world[:, 0]                            # koordinat X dunia
    Y = P_world[:, 1]                            # koordinat Y dunia

    return X, Y, height, np.stack([u_idx, v_idx], axis=1)


# ================================================================ visualisasi
def depth_colormap(depth, mask=None, colormap=cv2.COLORMAP_TURBO):
    """Render depth map ke citra BGR berwarna.

    Hanya piksel dalam mask yang dinormalisasi (jika mask diberikan).
    Area di luar mask ditampilkan gelap.
    """
    d = depth.copy().astype(np.float32)
    if mask is not None:
        roi = d[mask > 0]
        if roi.size == 0:
            return np.zeros((*depth.shape, 3), np.uint8)
        lo, hi = np.percentile(roi, [2, 98])
    else:
        lo, hi = np.percentile(d, [2, 98])

    if hi <= lo:
        hi = lo + 1e-6
    d_norm = np.clip((d - lo) / (hi - lo), 0, 1)
    d_u8 = (d_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_u8, colormap)

    if mask is not None:
        m = (mask > 0).astype(np.uint8)
        colored = cv2.bitwise_and(colored, colored, mask=m)

    return colored


def height_heatmap(X, Y, height, uv, img_shape, colormap=cv2.COLORMAP_JET):
    """Render peta tinggi sampah ke citra BGR (tinggi di atas tanah per piksel)."""
    out = np.zeros((*img_shape[:2], 3), np.uint8)
    if len(height) == 0:
        return out

    lo, hi = 0.0, np.percentile(height, 98)
    if hi <= lo:
        hi = lo + 0.01
    norm_h = np.clip((height - lo) / (hi - lo), 0, 1)
    colors = cv2.applyColorMap((norm_h * 255).astype(np.uint8).reshape(-1, 1, 1),
                               colormap).reshape(-1, 3)
    for (u, v), col in zip(uv, colors):
        out[v, u] = col
    return out


# ================================================================ perintah CLI
def _load_ground(ground_npz):
    d = np.load(ground_npz)
    return d["R"], d["t"], d["K"], float(d["cam_height"])


def _load_ref(path):
    ref = json.load(open(path))
    return np.float64(ref["image_pts"]), np.float64(ref["world_xy"])


def cmd_predict(a):
    img = cv2.imread(a.image)
    if img is None:
        raise FileNotFoundError(a.image)

    print("Memprediksi depth map...")
    depth = predict_depth(img, a.model, imgsz=a.imgsz, device=a.device)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "depth_raw.npy", depth)

    vis = depth_colormap(depth)
    cv2.imwrite(str(out / "depth_raw_vis.jpg"), vis)

    print(f"Depth raw: min={depth.min():.3f}  max={depth.max():.3f}  "
          f"mean={depth.mean():.3f}  (nilai RELATIF, belum meter)")
    print(f"Tersimpan: {out / 'depth_raw.npy'}  +  depth_raw_vis.jpg")


def cmd_scale(a):
    depth_raw = np.load(a.depth)
    R, t, K, cam_h_pose = _load_ground(a.ground)
    img_pts, world_xy = _load_ref(a.ref)

    s, b, residuals = calibrate_scale(depth_raw, K, R, t, img_pts, world_xy)
    print(f"Skala kalibrasi: s={s:.5f}  b={b:.5f}  "
          f"(depth_meter = {s:.4f} * depth_raw + {b:.4f})")
    print(f"Residual kalibrasi: rata-rata {np.abs(residuals).mean():.4f} m  "
          f"maks {np.abs(residuals).max():.4f} m")

    if abs(s) < 1e-4:
        print("PERINGATAN: faktor skala mendekati nol — depth model mungkin tidak cocok "
              "dengan titik acuan. Periksa kembali ground_ref.json dan ground_model.npz.")
    if np.abs(residuals).mean() > 0.5:
        print("PERINGATAN: residual kalibrasi > 0.5 m — tambah titik acuan atau periksa "
              "apakah depth map dan ground_model berasal dari citra yang sama.")

    depth_m = apply_scale(depth_raw, s, b, clip_min=a.clip_min, clip_max=a.clip_max)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "depth_scaled.npy", depth_m)
    np.savez(out / "depth_scale_params.npz", s=s, b=b,
             residuals=residuals, clip_min=a.clip_min, clip_max=a.clip_max)

    vis = depth_colormap(depth_m)
    cv2.imwrite(str(out / "depth_scaled_vis.jpg"), vis)
    print(f"Depth meter: min={depth_m.min():.3f} m  max={depth_m.max():.3f} m")
    print(f"Tersimpan: {out / 'depth_scaled.npy'}  +  depth_scaled_vis.jpg  "
          f"+  depth_scale_params.npz")

    # Informasi cross-check dengan tinggi kamera dari pose
    print(f"\nCross-check: tinggi kamera dari pose solvePnP = {cam_h_pose:.3f} m")
    print("  -> nilai depth pada piksel titik acuan seharusnya mendekati angka ini "
          "(tergantung sudut kamera).")


def cmd_lift(a):
    img = cv2.imread(a.image)
    if img is None:
        raise FileNotFoundError(a.image)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(a.mask)
    depth_m = np.load(a.depth)
    R, t, K, _ = _load_ground(a.ground)

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("Ukuran citra dan mask berbeda (keduanya harus sesudah undistort).")
    if depth_m.shape != img.shape[:2]:
        raise ValueError("Ukuran depth_scaled.npy berbeda dari citra — jalankan ulang cmd_scale.")

    waste_mask = mask > 127
    if not waste_mask.any():
        print("PERINGATAN: mask kosong — tidak ada piksel sampah untuk di-lift.")
        return

    X, Y, height, uv = lift_to_ground(K, R, t, depth_m, waste_mask)

    n_px = len(height)
    print(f"Piksel sampah di-lift: {n_px}")
    print(f"Tinggi di atas tanah: min={height.min():.3f} m  "
          f"max={height.max():.3f} m  median={np.median(height):.3f} m  "
          f"mean={height.mean():.3f} m")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "lifted_points.npz", X=X, Y=Y, height=height, uv=uv)

    hmap = height_heatmap(X, Y, height, uv, img.shape)
    overlay = cv2.addWeighted(img, 0.6, hmap, 0.7, 0)
    cv2.imwrite(str(out / "height_map.jpg"), overlay)

    dvis = depth_colormap(depth_m, mask)
    cv2.imwrite(str(out / "depth_masked.jpg"), dvis)

    print(f"Tersimpan: lifted_points.npz  +  height_map.jpg  +  depth_masked.jpg")
    print("\nLangkah selanjutnya: gunakan X, Y, height dari lifted_points.npz "
          "untuk estimasi volume (integrasi tinggi × luas per piksel).")


def cmd_vis(a):
    depth = np.load(a.depth)
    mask = None
    if a.mask:
        m = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(a.mask)
        mask = m

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    vis_full = depth_colormap(depth, colormap=cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out / "depth_vis_full.jpg"), vis_full)

    if mask is not None:
        vis_masked = depth_colormap(depth, mask, colormap=cv2.COLORMAP_TURBO)
        cv2.imwrite(str(out / "depth_vis_masked.jpg"), vis_masked)
        print(f"Tersimpan: depth_vis_full.jpg  +  depth_vis_masked.jpg")
    else:
        print(f"Tersimpan: depth_vis_full.jpg")

    print(f"Rentang depth: {depth.min():.4f} .. {depth.max():.4f}  "
          f"(satuan tergantung apakah sudah di-scale)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tahap 4 - Depth estimation & lift ke tanah")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -- predict
    p = sub.add_parser("predict", help="Prediksi depth map dari citra")
    p.add_argument("--image",  required=True, help="citra undistort (BGR)")
    p.add_argument("--model",  required=True, help="path model YOLO depth (.pt)")
    p.add_argument("--imgsz",  type=int, default=768)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out",    default="output")
    p.set_defaults(fn=cmd_predict)

    # -- scale
    p = sub.add_parser("scale", help="Kalibrasi skala relatif → meter")
    p.add_argument("--depth",    required=True, help="depth_raw.npy")
    p.add_argument("--ground",   required=True, help="ground_model.npz dari ipm.py")
    p.add_argument("--ref",      required=True, help="ground_ref.json dari ipm.py pick")
    p.add_argument("--out",      default="output")
    p.add_argument("--clip-min", type=float, default=0.1,  dest="clip_min",
                   help="batas depth minimum setelah scaling (m)")
    p.add_argument("--clip-max", type=float, default=50.0, dest="clip_max",
                   help="batas depth maksimum setelah scaling (m)")
    p.set_defaults(fn=cmd_scale)

    # -- lift
    p = sub.add_parser("lift", help="Lift piksel sampah ke koordinat tanah 3D")
    p.add_argument("--image",  required=True, help="citra undistort (BGR)")
    p.add_argument("--depth",  required=True, help="depth_scaled.npy")
    p.add_argument("--mask",   required=True, help="mask.png dari segment.py")
    p.add_argument("--ground", required=True, help="ground_model.npz dari ipm.py")
    p.add_argument("--out",    default="output")
    p.set_defaults(fn=cmd_lift)

    # -- vis
    p = sub.add_parser("vis", help="Visualisasi depth map sebagai colormap")
    p.add_argument("--depth", required=True, help="depth_raw.npy atau depth_scaled.npy")
    p.add_argument("--mask",  default=None,  help="(opsional) mask untuk crop visualisasi")
    p.add_argument("--out",   default="output")
    p.set_defaults(fn=cmd_vis)

    args = ap.parse_args()
    args.fn(args)
