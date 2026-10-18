"""Tahap 1 - Segmentasi sampah -> mask biner (mode instance atau semantic).

Masukan harus citra yang SUDAH di-undistort (lihat calibrate_camera.py), supaya
mask, depth (nanti), dan K berada di satu kerangka koordinat yang sama.

Mode instance (YOLO-seg, kelas: Container, Sampah_Berserakan, Sampah_Overload):
  python segment.py --image output/undist.jpg --model best.pt --mode instance \
      --waste-classes Sampah_Berserakan Sampah_Overload --imgsz 1280 --conf 0.25 --out output

Mode semantic (model lama):
  python segment.py --image output/undist.jpg --model best.pt --mode semantic --waste-id 0

Keluaran: mask.png (gabungan kelas sampah), mask_<kelas>.png per kelas (mode instance),
instances.json (kelas, conf, poligon per instance), overlay.png.
Container TIDAK ikut mask sampah (kecuali kamu memasukkannya di --waste-classes).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLORS = {"container": (255, 140, 0), "sampah_berserakan": (0, 165, 255), "sampah_overload": (0, 0, 255)}


def norm(s):
    return s.strip().lower().replace(" ", "_")


# ---------------------------------------------------------------- pembersihan
def clean_mask(mask, open_px=0, close_px=5, min_area_px=30):
    """Opening (buang bintik), closing (tutup lubang kecil), buang komponen < min_area_px.

    PERINGATAN: objek sampah pada citra CCTV bisa sangat kecil (puluhan sampai ratusan piksel).
    min_area besar bisa menghapus objek kecil yang berdiri sendiri, dan closing besar menggabungkan objek berdekatan; mulai dari nilai kecil.
    """
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


# ---------------------------------------------------------------- mode instance
def predict_instances(image_bgr, model_path, device="cpu", imgsz=1280, conf=0.25, iou=0.7):
    """-> (list instance {cls, conf, poly(n,2) piksel citra}, names)."""
    from ultralytics import YOLO  # impor di sini agar fungsi lain bisa dites tanpa ultralytics

    model = YOLO(model_path)
    r = model.predict(source=image_bgr, device=device, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    out = []
    if r.masks is not None:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        for poly, c, p in zip(r.masks.xy, cls, cf):      # masks.xy: poligon dalam piksel citra asli
            if len(poly) >= 3:
                out.append(dict(cls=model.names[int(c)], conf=float(p), poly=np.asarray(poly, np.float32)))
    return out, model.names


def rasterize(polys, shape):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p)).astype(np.int32)], 1)
    return m.astype(bool)


def class_masks(instances, shape):
    by = {}
    for ins in instances:
        by.setdefault(ins["cls"], []).append(ins["poly"])
    return {k: rasterize(v, shape) for k, v in by.items()}


def waste_mask_from(instances, shape, waste_classes):
    keep = {norm(c) for c in waste_classes}
    return rasterize([i["poly"] for i in instances if norm(i["cls"]) in keep], shape)


# ---------------------------------------------------------------- mode semantic (lama)
def segment_semantic(image_bgr, model_path, waste_id, device="cpu", imgsz=None):
    from ultralytics import YOLO

    h, w = image_bgr.shape[:2]
    model = YOLO(model_path, task="semantic")
    kw = dict(source=image_bgr, device=device, retina_masks=True, verbose=False)
    if imgsz:
        kw["imgsz"] = imgsz
    res = model.predict(**kw)[0]
    sem = res.semantic_mask.data.cpu().numpy()          # (H, W) ID kelas
    ids, counts = np.unique(sem, return_counts=True)
    print("names:", model.names)
    print("kelas pada prediksi:", {int(i): int(c) for i, c in zip(ids, counts)})
    mask = sem == waste_id
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


# ---------------------------------------------------------------- util
def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else 1.0


def overlay(image_bgr, mask=None, instances=None, color=(0, 200, 0), alpha=0.45):
    out = image_bgr.copy()
    if instances:
        for ins in instances:
            col = COLORS.get(norm(ins["cls"]), (0, 200, 0))
            m = rasterize([ins["poly"]], out.shape[:2])
            out[m] = (alpha * np.array(col) + (1 - alpha) * out[m]).astype(np.uint8)
            cv2.polylines(out, [np.round(ins["poly"]).astype(np.int32)], True, col, 2)
            x, y = np.round(ins["poly"].min(axis=0)).astype(int)
            cv2.putText(out, f"{ins['cls']} {ins['conf']:.2f}", (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    elif mask is not None:
        out[mask] = (alpha * np.array(color) + (1 - alpha) * out[mask]).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["instance", "semantic"], default="instance")
    ap.add_argument("--waste-classes", nargs="+", default=["Sampah_Berserakan", "Sampah_Overload"],
                    help="(instance) nama kelas yang dihitung sampah")
    ap.add_argument("--waste-id", type=int, help="(semantic) ID kelas sampah")
    ap.add_argument("--out", default="output")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--open-px", type=int, default=0)
    ap.add_argument("--close-px", type=int, default=5)
    ap.add_argument("--min-area-px", type=int, default=30)
    ap.add_argument("--gt", default=None, help="mask acuan (putih=sampah) untuk menghitung IoU")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        raise FileNotFoundError(a.image)
    h, w = img.shape[:2]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.mode == "instance":
        instances, names = predict_instances(img, a.model, a.device, a.imgsz, a.conf)
        print("names:", names)
        have = {norm(v) for v in names.values()} if isinstance(names, dict) else {norm(v) for v in names}
        missing = [c for c in a.waste_classes if norm(c) not in have]
        if missing:
            raise SystemExit(f"Kelas {missing} tidak ada pada model. Kelas model: {sorted(have)}")
        print("instance terdeteksi:", {c: sum(1 for i in instances if i['cls'] == c) for c in {i['cls'] for i in instances}})
        for c, m in class_masks(instances, (h, w)).items():
            cv2.imwrite(str(out / f"mask_{norm(c)}.png"), m.astype(np.uint8) * 255)
        mask = clean_mask(waste_mask_from(instances, (h, w), a.waste_classes),
                          a.open_px, a.close_px, a.min_area_px)
        json.dump([dict(cls=i["cls"], conf=i["conf"], polygon=np.round(i["poly"], 1).tolist()) for i in instances],
                  open(out / "instances.json", "w"))
        cv2.imwrite(str(out / "overlay.png"), overlay(img, instances=instances))
    else:
        if a.waste_id is None:
            raise SystemExit("Mode semantic butuh --waste-id.")
        mask = clean_mask(segment_semantic(img, a.model, a.waste_id, a.device, a.imgsz or None),
                          a.open_px or 5, a.close_px or 15, a.min_area_px or int(0.0005 * h * w))
        cv2.imwrite(str(out / "overlay.png"), overlay(img, mask=mask))

    cv2.imwrite(str(out / "mask.png"), mask.astype(np.uint8) * 255)
    frac = mask.mean()
    print(f"piksel sampah: {int(mask.sum())} ({100 * frac:.2f}% frame)")
    if mask.sum() == 0:
        print("PERINGATAN: tidak ada sampah terdeteksi - turunkan --conf atau cek model/kelas.")
    elif frac > 0.6:
        print("PERINGATAN: proporsi mask tidak wajar (> 60% frame) - cek kelas/keluaran model.")
    if a.gt:
        gt = cv2.imread(a.gt, 0) > 127
        print(f"IoU terhadap acuan: {iou(mask, gt):.3f}")