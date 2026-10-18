"""Tahap 5 - Estimasi volume sampah dari lifted_points.npz.

Alur setelah depth.py lift:
  1) Estimasi volume dari titik yang sudah di-lift ke koordinat tanah:
       python volume.py estimate --lifted output/lifted_points.npz \
           --ground output/ground_model.npz --out output

  2) (Opsional) Estimasi per-instance menggunakan instances.json dari segment.py:
       python volume.py estimate --lifted output/lifted_points.npz \
           --ground output/ground_model.npz \
           --instances output/instances.json \
           --calib calib.npz --out output

  3) Tampilkan ringkasan hasil:
       python volume.py report --result output/volume_result.npz

METODE ESTIMASI:
  * Grid (default): Bagi bidang tanah menjadi sel grid. Tiap sel ambil median
    tinggi dari piksel yang jatuh di dalamnya. Volume = Σ (median_height × luas_sel).
    Robust terhadap outlier; sel kosong tidak berkontribusi.

  * Convex hull (alternatif): Proyeksikan footprint ke BEV, hitung luas convex hull,
    kalikan dengan median tinggi keseluruhan. Cepat tapi overestimate jika bentuk
    tumpukan tidak cembung.

  * Per-instance: Ulangi metode grid untuk tiap instance segmentasi secara terpisah.

CATATAN:
  * Akurasi volume sangat bergantung pada akurasi kalibrasi skala depth.
    Selalu cross-check dengan objek referensi yang diketahui volumenya.
  * Piksel di tepi tumpukan (kontak tanah) cenderung memiliki height ≈ 0 dan
    tidak bias — namun piksel di puncak bisa underestimate jika depth model
    kehilangan detail di area padat.
  * Tumpukan yang sebagian tertutup (occluded) menghasilkan estimasi volume
    MINIMUM (volume terlihat), bukan total volume sebenarnya.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ================================================================ estimasi grid
def estimate_volume_grid(X, Y, height, cell_m=0.05, min_pts_per_cell=2):
    """Estimasi volume dengan metode grid median.

    Bagi bidang X-Y menjadi sel berukuran cell_m × cell_m meter.
    Volume tiap sel = median(height) × cell_m².
    Sel dengan jumlah piksel < min_pts_per_cell diabaikan.

    Parameter
    ---------
    X, Y    : ndarray (N,) — koordinat horizontal (meter)
    height  : ndarray (N,) — tinggi tiap titik di atas tanah (meter)
    cell_m  : float        — ukuran sel grid (meter); lebih kecil = lebih detail
    min_pts : int          — minimum titik per sel agar sel dianggap valid

    Mengembalikan
    -------------
    volume_m3    : float   — total volume estimasi (m³)
    footprint_m2 : float   — luas bidang yang terisi (m²)
    grid_height  : ndarray — peta tinggi median per sel (untuk visualisasi)
    grid_info    : dict    — metadata grid (origin, cell_m, shape)
    """
    if len(height) == 0:
        return 0.0, 0.0, np.zeros((1, 1)), {}

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    nx = max(1, int(np.ceil((x_max - x_min) / cell_m)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell_m)))

    # Indeks sel tiap titik
    ix = np.clip(((X - x_min) / cell_m).astype(int), 0, nx - 1)
    iy = np.clip(((Y - y_min) / cell_m).astype(int), 0, ny - 1)

    grid_height = np.full((ny, nx), np.nan, dtype=np.float32)
    cell_area = cell_m ** 2

    # Median tinggi per sel menggunakan bincount trick
    for j in range(ny):
        for i in range(nx):
            mask = (ix == i) & (iy == j)
            pts = height[mask]
            if len(pts) >= min_pts_per_cell:
                grid_height[j, i] = float(np.median(pts))

    valid = np.isfinite(grid_height) & (grid_height > 0)
    volume_m3 = float(np.nansum(grid_height[valid]) * cell_area)
    footprint_m2 = float(valid.sum() * cell_area)

    grid_info = dict(x_min=x_min, y_min=y_min, cell_m=cell_m,
                     nx=nx, ny=ny, shape=(ny, nx))
    return volume_m3, footprint_m2, grid_height, grid_info


def estimate_volume_convexhull(X, Y, height):
    """Estimasi volume dengan metode convex hull × median tinggi.

    Cepat, tapi overestimate untuk tumpukan berbentuk tidak cembung.

    Mengembalikan volume_m3, footprint_m2 (luas convex hull).
    """
    if len(height) < 3:
        return 0.0, 0.0

    try:
        from scipy.spatial import ConvexHull
        pts2d = np.stack([X, Y], axis=1)
        hull = ConvexHull(pts2d)
        footprint_m2 = float(hull.volume)   # scipy: .volume = luas di 2D
    except Exception:
        # Fallback: bounding box jika scipy tidak tersedia atau hull gagal
        footprint_m2 = float((X.max() - X.min()) * (Y.max() - Y.min()))

    med_h = float(np.median(height))
    volume_m3 = footprint_m2 * med_h
    return volume_m3, footprint_m2


# ================================================================ per-instance
def _poly_to_mask(poly, shape):
    """Konversi poligon piksel ke mask biner (H, W)."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(poly, np.float32)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def estimate_per_instance(instances_json, lifted_npz, img_shape,
                          cell_m=0.05, min_pts_per_cell=2):
    """Estimasi volume per-instance segmentasi.

    Parameter
    ---------
    instances_json : path ke instances.json dari segment.py
    lifted_npz     : path ke lifted_points.npz dari depth.py lift
    img_shape      : (H, W) citra asli (untuk rasterisasi poligon)

    Mengembalikan list dict per instance:
        { cls, conf, volume_m3, footprint_m2, n_pts }
    """
    instances = json.load(open(instances_json))
    d = np.load(lifted_npz)
    X, Y, height, uv = d["X"], d["Y"], d["height"], d["uv"]   # uv: (N, 2) = [u, v]

    H, W = img_shape[:2]
    results = []
    for ins in instances:
        poly = ins["polygon"]
        m = _poly_to_mask(poly, (H, W))

        # Cari titik lifted yang berada dalam mask instance ini
        u_arr, v_arr = uv[:, 0], uv[:, 1]
        in_inst = m[v_arr, u_arr]                 # bool (N,)

        xi, yi, hi = X[in_inst], Y[in_inst], height[in_inst]
        n_pts = int(in_inst.sum())

        if n_pts < 3:
            vol, foot = 0.0, 0.0
        else:
            vol, foot, _, _ = estimate_volume_grid(xi, yi, hi, cell_m, min_pts_per_cell)

        results.append(dict(
            cls=ins["cls"],
            conf=ins["conf"],
            volume_m3=round(vol, 6),
            footprint_m2=round(foot, 6),
            n_pts=n_pts,
        ))
    return results


# ================================================================ visualisasi
def render_volume_map(grid_height, grid_info, colormap=cv2.COLORMAP_JET):
    """Render grid tinggi menjadi heatmap BGR."""
    gh = grid_height.copy()
    finite = np.isfinite(gh)
    if not finite.any():
        return np.zeros((*gh.shape, 3), np.uint8)

    lo, hi = 0.0, np.percentile(gh[finite], 98)
    if hi <= lo:
        hi = lo + 0.01

    norm = np.clip((gh - lo) / (hi - lo), 0.0, 1.0)
    norm[~finite] = 0.0
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, colormap)
    colored[~finite] = 0
    return colored


def add_scale_bar(img, cell_m, ppm_display=None):
    """Tambahkan skala bar dan label meter ke pojok kanan bawah citra."""
    h, w = img.shape[:2]
    bar_m = 1.0                                    # panjang bar = 1 m
    if ppm_display is None:
        ppm_display = min(w, h) / 5.0             # fallback kasar
    bar_px = int(round(bar_m / cell_m))
    bar_px = min(bar_px, w // 3)

    x1, y1 = w - bar_px - 20, h - 30
    x2, y2 = x1 + bar_px, y1 + 10
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.putText(img, f"{bar_m:g} m", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ================================================================ perintah CLI
def cmd_estimate(a):
    d = np.load(a.lifted)
    X, Y, height, uv = d["X"], d["Y"], d["height"], d["uv"]

    gm = np.load(a.ground)
    # Resolusi piksel pada bidang tanah dari ground_model (opsional, dipakai sbg info)
    cam_height = float(gm["cam_height"])

    print(f"Titik sampah tersedia : {len(height)}")
    print(f"Tinggi kamera (pose)  : {cam_height:.3f} m")
    print(f"Rentang height        : {height.min():.3f} .. {height.max():.3f} m")

    # --- metode grid
    vol_grid, foot_grid, grid_h, ginfo = estimate_volume_grid(
        X, Y, height, cell_m=a.cell, min_pts_per_cell=a.min_pts
    )

    # --- metode convex hull
    vol_hull, foot_hull = estimate_volume_convexhull(X, Y, height)

    print(f"\n{'='*50}")
    print(f"  ESTIMASI VOLUME SAMPAH")
    print(f"{'='*50}")
    print(f"  Metode grid  ({a.cell*100:.0f} cm × {a.cell*100:.0f} cm):")
    print(f"    Volume      : {vol_grid:.4f} m³  ({vol_grid*1000:.1f} liter)")
    print(f"    Footprint   : {foot_grid:.4f} m²")
    print(f"    Tinggi rata : {vol_grid/foot_grid:.3f} m" if foot_grid > 0 else "")
    print(f"  Metode convex hull:")
    print(f"    Volume      : {vol_hull:.4f} m³  ({vol_hull*1000:.1f} liter)")
    print(f"    Footprint   : {foot_hull:.4f} m²")
    print(f"{'='*50}")
    print("  CATATAN: angka ini adalah ESTIMASI MINIMUM jika tumpukan sebagian")
    print("  tidak terlihat kamera (occluded).")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Simpan hasil numerik
    np.savez(out / "volume_result.npz",
             vol_grid=vol_grid, foot_grid=foot_grid,
             vol_hull=vol_hull, foot_hull=foot_hull,
             grid_height=grid_h,
             cell_m=a.cell,
             **{k: ginfo[k] for k in ("x_min", "y_min", "nx", "ny")})

    result_json = dict(
        method_grid=dict(volume_m3=round(vol_grid, 6),
                         volume_liter=round(vol_grid * 1000, 3),
                         footprint_m2=round(foot_grid, 6),
                         cell_m=a.cell),
        method_convexhull=dict(volume_m3=round(vol_hull, 6),
                               volume_liter=round(vol_hull * 1000, 3),
                               footprint_m2=round(foot_hull, 6)),
    )

    # Per-instance jika diminta
    if a.instances:
        if a.img_shape is None:
            raise SystemExit("--instances butuh --img-shape H W untuk rasterisasi poligon.")
        H_img, W_img = a.img_shape
        inst_results = estimate_per_instance(
            a.instances, a.lifted, (H_img, W_img), a.cell, a.min_pts
        )
        result_json["instances"] = inst_results
        print(f"\nPer-instance ({len(inst_results)} objek):")
        for r in inst_results:
            print(f"  [{r['cls']}] conf={r['conf']:.2f}  "
                  f"vol={r['volume_m3']:.4f} m³  "
                  f"foot={r['footprint_m2']:.4f} m²  "
                  f"pts={r['n_pts']}")

    json.dump(result_json, open(out / "volume_result.json", "w"), indent=2)

    # Visualisasi heatmap grid
    hmap = render_volume_map(grid_h, ginfo)
    hmap = add_scale_bar(hmap, a.cell)
    cv2.imwrite(str(out / "volume_heatmap.jpg"), hmap)

    print(f"\nTersimpan: volume_result.npz  +  volume_result.json  +  volume_heatmap.jpg")


def cmd_report(a):
    res = np.load(a.result, allow_pickle=True)
    print(f"\n{'='*50}")
    print(f"  LAPORAN VOLUME SAMPAH")
    print(f"{'='*50}")
    print(f"  Grid  : {float(res['vol_grid']):.4f} m³  "
          f"= {float(res['vol_grid'])*1000:.1f} liter  "
          f"(footprint {float(res['foot_grid']):.4f} m²)")
    print(f"  Hull  : {float(res['vol_hull']):.4f} m³  "
          f"= {float(res['vol_hull'])*1000:.1f} liter  "
          f"(footprint {float(res['foot_hull']):.4f} m²)")
    print(f"  Sel   : {float(res['cell_m'])*100:.0f} cm × {float(res['cell_m'])*100:.0f} cm")
    print(f"{'='*50}")

    if a.json:
        rj = json.load(open(a.json))
        if "instances" in rj:
            print(f"\nPer-instance:")
            total = 0.0
            for r in rj["instances"]:
                print(f"  [{r['cls']}]  {r['volume_m3']:.4f} m³  "
                      f"({r['volume_m3']*1000:.1f} L)  "
                      f"footprint {r['footprint_m2']:.4f} m²")
                total += r["volume_m3"]
            print(f"  Total instance : {total:.4f} m³  ({total*1000:.1f} L)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tahap 5 - Estimasi volume sampah")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -- estimate
    p = sub.add_parser("estimate", help="Hitung volume dari lifted_points.npz")
    p.add_argument("--lifted",     required=True, help="lifted_points.npz dari depth.py lift")
    p.add_argument("--ground",     required=True, help="ground_model.npz dari ipm.py")
    p.add_argument("--out",        default="output")
    p.add_argument("--cell",       type=float, default=0.05,
                   help="ukuran sel grid (meter); default 5 cm")
    p.add_argument("--min-pts",    type=int, default=2, dest="min_pts",
                   help="minimum titik per sel grid agar valid")
    p.add_argument("--instances",  default=None,
                   help="(opsional) instances.json dari segment.py untuk estimasi per-instance")
    p.add_argument("--img-shape",  type=int, nargs=2, metavar=("H", "W"), dest="img_shape",
                   default=None,
                   help="ukuran citra (H W) — wajib jika --instances dipakai")
    p.set_defaults(fn=cmd_estimate)

    # -- report
    p = sub.add_parser("report", help="Tampilkan ringkasan dari volume_result.npz")
    p.add_argument("--result", required=True, help="volume_result.npz")
    p.add_argument("--json",   default=None,  help="(opsional) volume_result.json")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)
