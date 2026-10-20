# SIVERA : Sistem Identifikasi Volume dan Estimasi Ruang berbasis AI

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Ultralytics%20YOLO-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black" alt="YOLO" />
</p>

> **SIVERA** (Sistem Identifikasi Volume dan Estimasi Ruang berbasis AI) adalah platform komputer visi canggih untuk estimasi volume dan analisis ruang 3D menggunakan kamera tunggal (monocular) dengan pendekatan *depth estimation* dan *inverse perspective mapping*.

---

## 🎯 Tentang Proyek

SIVERA dirancang untuk mengatasi tantangan estimasi volume objek dan pemodelan ruang 3D dalam lingkungan real-world dengan menggunakan:
- **Monocular Depth Estimation** (YOLO + Depth Anything v2)
- **Inverse Perspective Mapping (IPM)** untuk transformasi *Bird's Eye View*
- **Kalibrasi Kamera** otomatis menggunakan papan catur (chessboard)
- **Ground Plane Modeling** dengan referensi titik tanah
- **Segmentasi Instance** untuk deteksi objek individual
- **Volume Estimation** berbasis voxel grid dan convex hull

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIVERA Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📷 Input Image → 🔍 Segmentation → 📏 Depth Estimation         │
│       ↓              ↓                    ↓                     │
│  🎯 Calibration → 🗺️ Ground Model → 📐 IPM Transform           │
│       ↓              ↓                    ↓                     │
│  📦 Volume Estimation (Voxel Grid + Convex Hull)                │
│       ↓                                                         │
│  📊 Output: 3D Volume, BEV Map, Annotated Images, JSON Report  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Fitur Utama

### 1. **Manajemen Kamera** (`/cameras`)
- Registrasi kamera dengan metadata lokasi (PostGIS support)
- Konfigurasi intrinsik/ekstrinsik kamera
- CRUD lengkap untuk registry kamera

### 2. **Kalibrasi Otomatis** (`/cameras/{id}/calibrations`)
- Upload gambar papan catur → kalibrasi intrinsik (K, distorsi, RMS)
- Versioning kalibrasi dengan aktivasi/deaktivasi
- Preview undistort untuk validasi visual

### 3. **Referensi Tanah & Ground Model** (`/cameras/{id}/ground-refs`)
- Submit titik referensi tanah (image ↔ world coordinates)
- Komputasi homography matrix + pose kamera (R, t, H)
- BEV (Bird's Eye View) preview untuk verifikasi

### 4. **Prediksi Asinkron** (`/cameras/{id}/predict`)
- Submit job prediksi, antrekan melalui Celery worker (mengembalikan HTTP 202 Accepted)
- Polling status job melalui `GET /cameras/{id}/predict/{job_id}`
- Output: gambar teranotasi, NPZ (data numerik), JSON report
- **Catatan:** Sumber `rtsp_frame` dan `latest_rtsp` dikenali dalam schema tetapi handler saat ini melempar `NotImplementedError` (belum diimplementasikan). Hanya `upload` yang diproses.

### 5. **Pipeline AI Terintegrasi**
| Service | Deskripsi | Model/Algoritma |
|---------|-----------|-----------------|
| `segment_service` | Instance segmentation | YOLOv8/v11-seg |
| `depth_service` | Monocular depth estimation | Depth Anything v2 |
| `calibration_service` | Kamera intrinsik calibration | OpenCV chessboard |
| `ground_model_service` | IPM & ground plane | Homography + PnP |
| `volume_service` | Volume estimation | Voxel grid + Convex hull |
| `prediction_orchestrator` | Pipeline end-to-end | Chained services |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|------------|
| **API Framework** | FastAPI (async) |
| **Database** | PostgreSQL + PostGIS + SQLAlchemy 2.0 (async) |
| **Migrations** | Alembic |
| **Task Queue** | Celery + Redis |
| **Storage** | Local filesystem / S3-compatible (abstraction layer) |
| **Auth** | API Key based |
| **Computer Vision** | OpenCV, Ultralytics YOLO, Depth Anything v2 |
| **Scientific Python** | NumPy, SciPy, scikit-image |
| **Containerization** | Docker + Docker Compose |

---

## 🗄️ Skema Database (10 Tabel)

| Tabel | Deskripsi |
|-------|-----------|
| `cameras` | Registry kamera dengan lokasi geografis |
| `calibrations` | Data kalibrasi versioned (K, dist, RMS error) |
| `ground_references` | Titik referensi tanah versioned |
| `ground_models` | Homography matrix + pose kamera (R, t, H) |
| `depth_scales` | Kalibrasi skala depth (scale, bias) |
| `prediction_jobs` | Job prediksi asinkron dengan status |
| `prediction_outputs` | File output (images, NPZ, JSON) |
| `detected_instances` | Hasil deteksi per-instance |
| `volume_estimates` | Estimasi volume (voxel grid + convex hull) |
| `api_keys` | Manajemen API key untuk autentikasi |

---

## ⚡ Quick Start

### Prasyarat
- Docker & Docker Compose
- GPU NVIDIA (direkomendasikan untuk inference AI)
- Minimal 8GB RAM, 20GB disk space

### 1. Clone & Konfigurasi
```bash
git clone <repository-url>
cd sivera

# Copy environment template
cp .env.example .env
# Edit .env dengan konfigurasi Anda
```

### 2. Jalankan dengan Docker Compose
```bash
# Build dan jalankan semua services
docker-compose up --build -d

# Cek status
docker-compose ps

# Lihat logs
docker-compose logs -f api
```

### 3. Akses API
- **API Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 🔧 Konfigurasi Environment

| Variabel | Deskripsi | Default |
|----------|-----------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://user:pass@db:5432/sivera` |
| `REDIS_URL` | Redis connection untuk Celery | `redis://redis:6379/0` |
| `STORAGE_BACKEND` | `local` atau `s3` | `local` |
| `STORAGE_LOCAL_PATH` | Path penyimpanan lokal | `./data/storage` |
| `S3_ENDPOINT_URL` | S3 endpoint (MinIO, AWS, dll) | - |
| `S3_BUCKET` | Nama bucket S3 | - |
| `API_KEY` | API key untuk autentikasi | `changeme` |
| `YOLO_MODEL_PATH` | Path model YOLO segmentation | `models/yolo26n-seg.pt` |
| `DEPTH_MODEL_PATH` | Path model Depth Anything | `models/depth_anything_v2_vits.pth` |

---

## 📚 Dokumentasi API

### Autentikasi
Semua endpoint memerlukan header:
```http
X-API-Key: <API_KEY>
```

### Contoh Penggunaan

#### Registrasi Kamera
```bash
curl -X POST "http://localhost:8000/cameras" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Cam-01",
    "location": "Gudang A - Lantai 1",
    "latitude": -6.2088,
    "longitude": 106.8456,
    "height_m": 3.5,
    "angle_deg": -15.0
  }'
```

#### Kalibrasi Kamera
```bash
curl -X POST "http://localhost:8000/cameras/1/calibrations" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "images=@chessboard1.jpg" \
  -F "images=@chessboard2.jpg" \
  -F "images=@chessboard3.jpg" \
  -F "square_size_m=0.025" \
  -F "pattern_cols=9" \
  -F "pattern_rows=6"
```

#### Submit Ground Reference
```bash
curl -X POST "http://localhost:8000/cameras/1/ground-references" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {"u": 320, "v": 480, "x": 0.0, "y": 0.0, "z": 0.0},
      {"u": 640, "v": 480, "x": 2.0, "y": 0.0, "z": 0.0},
      {"u": 320, "v": 240, "x": 0.0, "y": 2.0, "z": 0.0},
      {"u": 640, "v": 240, "x": 2.0, "y": 2.0, "z": 0.0}
    ]
  }'
```

#### Jalankan Prediksi
```bash
curl -X POST "http://localhost:8000/cameras/1/predict" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "image=@scene.jpg" \
  -F "source_type=upload" \
  -F "seg_conf=0.5"
```
Respons `202 Accepted` berisi `job_id` untuk polling status.

#### Cek Status Job
```bash
curl -X GET "http://localhost:8000/cameras/1/predict/{job_id}" \
  -H "X-API-Key: YOUR_API_KEY"
```

---

## 🧪 Testing

```bash
# Unit tests (tanpa database)
docker-compose exec api pytest tests/ -v

# Jalankan semua tes
pytest tests/ -v
```

---

## 📁 Struktur Proyek

```
sivera/
├── api/                          # FastAPI Application
│   ├── main.py                   # App factory & lifespan
│   ├── core/                     # Core configurations
│   │   ├── config.py             # Pydantic Settings
│   │   ├── database.py           # SQLAlchemy async engine/session
│   │   ├── storage.py            # Storage abstraction (Local/S3)
│   │   └── security.py           # API Key authentication
│   ├── models/                   # SQLAlchemy ORM models
│   │   └── orm.py                # 10 tabel database
│   ├── schemas/                  # Pydantic schemas (request/response)
│   ├── routes/                   # API endpoints (REST)
│   ├── services/                 # Business logic & CV pipeline
│   │   ├── calibration_service.py
│   │   ├── depth_service.py
│   │   ├── ground_model_service.py
│   │   ├── segment_service.py
│   │   ├── volume_service.py
│   │   └── prediction_orchestrator.py
│   ├── workers/                  # Celery async tasks
│   │   ├── celery_app.py
│   │   └── tasks/predict_task.py
│   ├── scripts/                  # Utility scripts
│   └── alembic/                  # Database migrations
├── docker-compose.yml            # Multi-container orchestration
├── Dockerfile                    # API container definition
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── cctv.pt                       # YOLO model weights
├── yolo26n-depth.pt              # Depth model weights
└── README.md                     # This file
```

---

## 🤝 Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/nama-fitur`)
3. Commit changes (`git commit -m 'feat: tambah fitur X'`)
4. Push ke branch (`git push origin feature/nama-fitur`)
5. Buat Pull Request

### Standar Kode
- **Formatter**: `black` + `isort`
- **Linter**: `ruff` / `flake8`
- **Type Checking**: `mypy`
- **Pre-commit hooks** direkomendasikan

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **MIT License** - lihat file [LICENSE](LICENSE) untuk detail.

---

## 👥 Tim Pengembang

- **Lead Developer**: [Nama Developer]
- **Computer Vision Engineer**: [Nama Engineer]
- **Backend Engineer**: [Nama Engineer]

---

## 📞 Dukungan & Kontak

- **Issues**: [GitHub Issues](https://github.com/your-org/sivera/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/sivera/discussions)
- **Email**: support@sivera.ai

---

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection & segmentation
- [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) - Monocular depth estimation
- [OpenCV](https://opencv.org/) - Computer vision library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - Database toolkit
- [Celery](https://docs.celeryq.dev/) - Distributed task queue

---

> **SIVERA** - *Mengubah Perspektif 2D menjadi Intelijen Ruang 3D*