"""Tahap 3 - IPM (Inverse Perspective Mapping) berskala meter.
 
Masukan: citra SUDAH di-undistort + K_undist dari calib.npz + titik acuan di tanah.
 
  1) Tandai titik acuan (klik; urutan kiri-atas, kanan-atas, kanan-bawah, kiri-bawah):
       python ipm.py pick  --image output/undist.jpg --rect 2.0 3.0 --out ground_ref.json
     (atau --world "0,0;2,0;2,3;0,3" untuk titik/ukuran sembarang, satuan meter)
  2) Jalankan IPM:
       python ipm.py run   --image output/undist.jpg --mask output/mask.png \
                           --calib calib.npz --ref ground_ref.json --out output
  3) Validasi skala dengan panjang yang diketahui di tanah (mis. 1 m):
       python ipm.py check --image output/undist.jpg --calib calib.npz \
                           --ref ground_ref.json --length 1.0
 
CATATAN PENTING: homografi hanya benar untuk titik yang BERADA DI BIDANG TANAH.
Permukaan tumpukan ada di atas tanah, jadi mask yang di-warp = "proyeksi", BUKAN
footprint. Yang valid di tahap ini: titik acuan, dan garis kontak dasar tumpukan
(piksel mask terbawah tiap kolom).
"""
import argparse
import json
from pathlib import Path
 
import cv2
import numpy as np
 
 
# ---------------------------------------------------------------- util klik
def click_points(image, n=None, title="klik titik | u=undo | Enter=selesai | Esc=batal"):
    pts = []
 
    def cb(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
 
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, cb)
    while True:
        view = image.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(view, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
        cv2.imshow(title, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (10, 13) or (n and len(pts) >= n):
            break
        if k == ord("u") and pts:
            pts.pop()
        if k == 27:
            pts = []
            break
    cv2.destroyAllWindows()
    return np.float32(pts)
 
 
# ---------------------------------------------------------------- geometri
def ground_model(img_pts, world_xy, K):
    """Homografi tanah<->citra + pose kamera dari titik acuan (bidang tanah Z=0)."""
    img_pts = np.asarray(img_pts, np.float64)
    world_xy = np.asarray(world_xy, np.float64)
    if len(img_pts) < 4 or len(img_pts) != len(world_xy):
        raise ValueError("Butuh >= 4 titik acuan dan jumlah titik citra == titik dunia.")
 
    H_w2i, _ = cv2.findHomography(world_xy, img_pts, 0)           # kuadrat terkecil
    H_i2w = np.linalg.inv(H_w2i)
 
    obj = np.hstack([world_xy, np.zeros((len(world_xy), 1))])
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, np.asarray(K, np.float64), None,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP gagal.")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img_pts, np.asarray(K, np.float64), None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
 
    proj, _ = cv2.projectPoints(obj, rvec, tvec, np.asarray(K, np.float64), None)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)
 
    cam = (-R.T @ tvec).ravel()                                   # posisi kamera (dunia)
    axis_w = R.T @ np.array([0.0, 0.0, 1.0])                      # sumbu optik (dunia)
    pitch_down = np.degrees(np.arcsin(np.clip(-axis_w[2] * np.sign(cam[2]), -1, 1)))
    return dict(H_w2i=H_w2i, H_i2w=H_i2w, R=R, t=tvec.ravel(), K=np.asarray(K, np.float64),
                cam_height=float(abs(cam[2])), pitch_down_deg=float(pitch_down),
                reproj_err=err)
 
 
def to_world(H_i2w, px, ref_sign=None):
    """Piksel -> (X, Y) meter di tanah. valid=False utk titik di atas cakrawala/tak hingga."""
    px = np.asarray(px, np.float64).reshape(-1, 2)
    p = np.hstack([px, np.ones((len(px), 1))]) @ H_i2w.T
    w = p[:, 2]
    xy = p[:, :2] / w[:, None]
    valid = np.isfinite(xy).all(axis=1)
    if ref_sign is not None:
        valid &= np.sign(w) == ref_sign
    return xy, valid
 
 
def ref_w_sign(H_i2w, img_pts):
    p = np.hstack([np.asarray(img_pts, np.float64), np.ones((len(img_pts), 1))]) @ H_i2w.T
    return np.sign(np.median(p[:, 2]))
 
 
def base_contour(mask):
    """Piksel mask TERBAWAH di tiap kolom = perkiraan garis kontak dasar tumpukan."""
    h = mask.shape[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, bool)
    v = h - 1 - np.argmax(mask[::-1, cols], axis=0)
    pts = np.stack([cols, v], axis=1).astype(np.float32)
    truncated = v >= h - 1                                         # tersentuh tepi bawah frame
    return pts, truncated
 
 
def bev_matrix(H_i2w, xmin, ymin, ppm):
    S = np.array([[ppm, 0, -xmin * ppm], [0, ppm, -ymin * ppm], [0, 0, 1.0]])
    return S @ H_i2w
 
 
# ---------------------------------------------------------------- perintah
def load_ref(path):
    ref = json.load(open(path))
    return np.float64(ref["image_pts"]), np.float64(ref["world_xy"])
 
 
def parse_world(s):
    return [[float(v) for v in pt.split(",")] for pt in s.split(";")]
 
 
def cmd_pick(a):
    img = cv2.imread(a.image)
    if img is None:
        raise FileNotFoundError(a.image)
    if a.rect:
        W, L = a.rect
        world = [[0, 0], [W, 0], [W, L], [0, L]]
    elif a.world:
        world = parse_world(a.world)
    else:
        raise SystemExit("Beri --rect W L atau --world.")
    print(f"Klik {len(world)} titik sesuai urutan titik dunia: {world}")
    pts = click_points(img, n=len(world))
    if len(pts) != len(world):
        raise SystemExit("Jumlah klik tidak cocok; dibatalkan.")
    json.dump({"image_pts": pts.tolist(), "world_xy": world}, open(a.out, "w"), indent=2)
    print("Tersimpan:", a.out)
 
 
def cmd_run(a):
    img = cv2.imread(a.image)
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise FileNotFoundError("Citra atau mask tidak ditemukan.")
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("Ukuran citra dan mask harus sama (keduanya sesudah undistort).")
    mask = mask > 127
 
    K = np.load(a.calib)["K_undist"]
    img_pts, world = load_ref(a.ref)
    gm = ground_model(img_pts, world, K)
    sgn = ref_w_sign(gm["H_i2w"], img_pts)
 
    print(f"Reproyeksi titik acuan: rata-rata {gm['reproj_err'].mean():.2f} px, "
          f"maks {gm['reproj_err'].max():.2f} px")
    print(f"Tinggi kamera (dari pose): {gm['cam_height']:.2f} m   "
          f"sudut tunduk: {gm['pitch_down_deg']:.1f} derajat")
    print("  -> bandingkan dengan tinggi pemasangan yang diukur (meteran) dan sudut sebenarnya.")
 
    base_px, trunc = base_contour(mask)
    base_w, ok = to_world(gm["H_i2w"], base_px, sgn)
    ok &= (np.abs(base_w) < 200).all(axis=1)
    base_w, base_px, trunc = base_w[ok], base_px[ok], trunc[ok]
 
    xs = np.r_[world[:, 0], base_w[:, 0]] if len(base_w) else world[:, 0]
    ys = np.r_[world[:, 1], base_w[:, 1]] if len(base_w) else world[:, 1]
    xmin, xmax = xs.min() - a.margin, xs.max() + a.margin
    ymin, ymax = ys.min() - a.margin, ys.max() + a.margin
    wpx, hpx = int(round((xmax - xmin) * a.ppm)), int(round((ymax - ymin) * a.ppm))
    if max(wpx, hpx) > 6000:
        raise SystemExit(f"BEV {wpx}x{hpx} px terlalu besar: turunkan --ppm atau kecilkan --margin.")
 
    Hb = bev_matrix(gm["H_i2w"], xmin, ymin, a.ppm)
    bev_img = cv2.warpPerspective(img, Hb, (wpx, hpx))
    bev_proj = cv2.warpPerspective(mask.astype(np.uint8) * 255, Hb, (wpx, hpx),
                                   flags=cv2.INTER_NEAREST)
 
    # overlay validasi: grid 1 m, titik acuan, garis kontak dasar
    ov = bev_img.copy()
    to_bev = lambda xy: np.round((np.asarray(xy) - [xmin, ymin]) * a.ppm).astype(int)
    for gx in np.arange(np.ceil(xmin), xmax, 1.0):
        x = int(round((gx - xmin) * a.ppm))
        cv2.line(ov, (x, 0), (x, hpx - 1), (255, 255, 255), 1)
        cv2.putText(ov, f"{gx:g}m", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for gy in np.arange(np.ceil(ymin), ymax, 1.0):
        y = int(round((gy - ymin) * a.ppm))
        cv2.line(ov, (0, y), (wpx - 1, y), (255, 255, 255), 1)
        cv2.putText(ov, f"{gy:g}m", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for p in to_bev(world):
        cv2.circle(ov, tuple(int(v) for v in p), 6, (0, 0, 255), -1)
    if len(base_w):
        poly = to_bev(base_w).reshape(-1, 1, 2)
        cv2.polylines(ov, [poly], False, (0, 255, 255), 2)
 
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "bev_image.jpg"), bev_img)
    cv2.imwrite(str(out / "bev_mask_proyeksi.png"), bev_proj)
    cv2.imwrite(str(out / "bev_overlay.jpg"), ov)
    np.savez(out / "ground_model.npz", H_i2w=gm["H_i2w"], H_w2i=gm["H_w2i"], R=gm["R"],
             t=gm["t"], K=gm["K"], cam_height=gm["cam_height"], pitch_down_deg=gm["pitch_down_deg"])
 
    print(f"\nBEV: {wpx} x {hpx} px, skala {a.ppm:g} px/m "
          f"(1 piksel = {1 / a.ppm * 100:.2f} cm), cakupan X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}] m")
    if len(base_w):
        x5, x95 = np.percentile(base_w[:, 0], [5, 95])
        print(f"Garis kontak dasar (persentil 5-95 agar kebal ujung): X {x5:.2f}..{x95:.2f} m "
              f"(lebar ~{x95 - x5:.2f} m), Y median {np.median(base_w[:, 1]):.2f} m")
        print("  Ujung kiri/kanan kontur bisa berupa titik tumpukan yang TERANGKAT "
              "(bukan kontak tanah) sehingga terproyeksi keliru; hanya bagian tengah yang andal.")
        if trunc.any():
            print(f"PERINGATAN: {int(trunc.sum())} kolom mask menyentuh tepi bawah frame "
                  "(dasar tumpukan terpotong) - garis kontak di sana tidak valid.")
    print("PERINGATAN: bev_mask_proyeksi.png = proyeksi permukaan, BUKAN footprint "
          "(titik di atas tanah bergeser menjauhi kamera).")
    print("Tersimpan di:", out.resolve())
 
 
def cmd_check(a):
    img = cv2.imread(a.image)
    K = np.load(a.calib)["K_undist"]
    img_pts, world = load_ref(a.ref)
    gm = ground_model(img_pts, world, K)
    sgn = ref_w_sign(gm["H_i2w"], img_pts)
    if a.pts:
        pts = np.float32(a.pts).reshape(2, 2)
    else:
        pts = click_points(img, n=2, title="klik 2 ujung panjang yang diketahui (di tanah)")
    if len(pts) != 2:
        raise SystemExit("Perlu 2 titik.")
    w, ok = to_world(gm["H_i2w"], pts, sgn)
    if not ok.all():
        raise SystemExit("Titik berada di atas cakrawala bidang tanah.")
    d = float(np.linalg.norm(w[0] - w[1]))
    print(f"jarak terukur {d:.3f} m  vs  sebenarnya {a.length:.3f} m  ->  "
          f"galat {100 * (d - a.length) / a.length:+.1f}%")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
 
    p = sub.add_parser("pick")
    p.add_argument("--image", required=True)
    p.add_argument("--rect", type=float, nargs=2, metavar=("W", "L"))
    p.add_argument("--world", type=str)
    p.add_argument("--out", default="ground_ref.json")
    p.set_defaults(fn=cmd_pick)
 
    p = sub.add_parser("run")
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", default="output")
    p.add_argument("--ppm", type=float, default=200, help="piksel per meter pada BEV")
    p.add_argument("--margin", type=float, default=1.0, help="margin BEV di luar titik (m)")
    p.set_defaults(fn=cmd_run)
 
    p = sub.add_parser("check")
    p.add_argument("--image", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--length", type=float, required=True)
    p.add_argument("--pts", type=float, nargs=4, metavar=("U1", "V1", "U2", "V2"))
    p.set_defaults(fn=cmd_check)
 
    args = ap.parse_args()
    args.fn(args)