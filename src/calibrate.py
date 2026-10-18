"""Tahap 2 - Kalibrasi kamera (papan catur) -> calib.npz, plus pembantu undistort.

Pakai:
  python calibrate_camera.py --images calib_imgs --cols 9 --rows 6 --square 0.025 \
      --out calib.npz --undistort-test test/image_6.jpg

--cols/--rows : jumlah SUDUT DALAM papan (bukan jumlah kotak).
--square      : sisi satu kotak (meter).

Aturan pengambilan foto kalibrasi:
  * Ambil dari kamera/stream YANG SAMA dengan resolusi/profil yang sama dengan
    pemakaian nanti (main stream dan sub stream CCTV bisa beda FOV/crop).
  * 15-30 foto; papan rata di bidang kaku; variasikan jarak dan kemiringan,
    dan buat papan menyentuh tepi + sudut frame (distorsi paling besar di sana).
  * Kunci fokus/zoom. Jangan ubah setelan kamera sesudahnya.
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def calibrate(image_paths, cols, rows, square_m):
    pattern = (cols, rows)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    obj_pts, img_pts, used = [], [], []
    size = None
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]                     # (w, h)
        elif gray.shape[::-1] != size:
            print(f"[lewati] {p}: ukuran {gray.shape[::-1]} != {size}")
            continue
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if not found:
            print(f"[lewati] {p}: papan tidak terdeteksi")
            continue
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(str(p))

    if len(used) < 10:
        raise RuntimeError(f"Hanya {len(used)} foto valid; minimal 10 (disarankan 15-30).")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    per_view = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        d = proj.reshape(-1, 2) - i.reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))))

    allc = np.concatenate([c.reshape(-1, 2) for c in img_pts])
    coverage = ((allc[:, 0].max() - allc[:, 0].min()) / size[0],
                (allc[:, 1].max() - allc[:, 1].min()) / size[1])
    return dict(K=K, dist=dist, size=np.array(size), rms=rms,
                per_view=per_view, used=used, coverage=coverage)


def make_undistorter(K, dist, size, alpha=0.0):
    """Peta remap + K baru. alpha=0: tanpa tepi hitam (citra sedikit di-zoom)."""
    size = (int(size[0]), int(size[1]))
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, size, cv2.CV_16SC2)
    return K_new, map1, map2


def undistort(img, map1, map2, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.remap(img, map1, map2, interp)


def load_calib(path):
    d = np.load(path)
    return d["K"], d["dist"], tuple(int(x) for x in d["size"]), d["K_undist"]


def report(res):
    K, dist, size = res["K"], res["dist"], tuple(res["size"])
    print(f"\nFoto dipakai : {len(res['used'])}")
    print(f"Ukuran citra : {size[0]} x {size[1]}")
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print("dist (k1,k2,p1,p2,k3):", np.round(dist.ravel(), 4))
    print(f"RMS reproyeksi: {res['rms']:.3f} px")
    worst = int(np.argmax(res["per_view"]))
    print(f"Galat per foto: rata-rata {np.mean(res['per_view']):.3f}, "
          f"terburuk {res['per_view'][worst]:.3f} px ({Path(res['used'][worst]).name})")
    print(f"Cakupan sudut papan: {100*res['coverage'][0]:.0f}% lebar, {100*res['coverage'][1]:.0f}% tinggi")

    notes = []
    if res["rms"] > 1.0:
        notes.append("RMS > 1 px: kalibrasi buruk - buang foto buram/terburuk, tambah variasi pose.")
    elif res["rms"] > 0.5:
        notes.append("RMS 0.5-1 px: dapat diterima, tapi coba perbaiki bila bisa.")
    if min(res["coverage"]) < 0.8:
        notes.append("Papan belum menjangkau tepi frame: distorsi di tepi tak terkoreksi baik.")
    if abs(K[0, 2] - size[0] / 2) > 0.05 * size[0] or abs(K[1, 2] - size[1] / 2) > 0.05 * size[1]:
        notes.append("Titik pusat (cx, cy) jauh dari tengah citra: curigai data kurang bervariasi.")
    if abs(K[0, 0] / K[1, 1] - 1) > 0.03:
        notes.append("fx dan fy beda >3%: tidak wajar untuk piksel persegi.")
    for n in notes:
        print("PERINGATAN:", n)
    if not notes:
        print("Gerbang lulus: RMS baik, cakupan baik, parameter wajar.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder foto papan catur")
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--square", type=float, required=True, help="sisi kotak (m)")
    ap.add_argument("--out", default="calib.npz")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--undistort-test", default=None, help="citra untuk pratinjau undistort")
    a = ap.parse_args()

    paths = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                   for p in glob.glob(str(Path(a.images) / ext)))
    res = calibrate(paths, a.cols, a.rows, a.square)
    report(res)

    K_new, map1, map2 = make_undistorter(res["K"], res["dist"], res["size"], a.alpha)
    np.savez(a.out, K=res["K"], dist=res["dist"], size=res["size"],
             K_undist=K_new, rms=res["rms"])
    print(f"\nTersimpan: {a.out}  (K_undist = K untuk citra SESUDAH undistort)")

    if a.undistort_test:
        img = cv2.imread(a.undistort_test)
        und = undistort(img, map1, map2)
        pair = np.hstack([img, und])
        cv2.imwrite("undistort_preview.jpg", pair)
        print("Pratinjau (kiri asli, kanan undistort): undistort_preview.jpg")
        print("Cek: garis lurus di dunia nyata (tepi bangunan, sambungan paving) harus lurus di kanan.")