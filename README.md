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

### 4. MediaMTX

Service `mediamtx` memakai image `bluenviron/mediamtx:1.20.0-ffmpeg`, konfigurasi `deploy/mediamtx.yml`, dan override environment di `docker-compose.yml`. File `mediamtx.yml` lokal di root tidak digunakan dan tidak diubah.

- RTSP `8554`, RTMP `1935`, HLS `8888`, dan WebRTC signaling `8889` dipublikasikan hanya ke localhost host. Backend mengakses nama service `mediamtx` pada jaringan Docker.
- Media WebRTC: `8189/udp` harus bisa dijangkau browser. Signaling melalui FastAPI, bukan akses publik port `8889`.
- Control API: `9997`, dipublikasikan hanya ke `127.0.0.1` host.
- Nama environment mengikuti [field konfigurasi MediaMTX v1.20.0](https://github.com/bluenviron/mediamtx/blob/v1.20.0/mediamtx.yml): `MTX_API=true`, `MTX_APIADDRESS=:9997`. Nama seperti `MTX_API_ENABLED` dan `MTX_API_ADDRESS` tidak dikenali.
- Healthcheck melakukan GET `/v3/paths/list` dari dalam container; tidak memerlukan stream aktif.

Jalankan ulang hanya MediaMTX dan periksa API:

```bash
docker compose up -d --no-deps --force-recreate --wait mediamtx
docker compose exec mediamtx wget -q -O - http://127.0.0.1:9997/v3/config/global/get
```

Akses admin tanpa login hanya diizinkan dari localhost **container**, untuk healthcheck. Host/backend memerlukan `MEDIAMTX_API_USER` + `MEDIAMTX_API_PASSWORD` (permission `api`). Reader memakai user/password berbeda dengan permission `read` hanya pada namespace SIVERA. Tidak ada publish/read anonim. Jangan mengirim kredensial ini ke dashboard atau membuka admin anonim.

#### Mengaktifkan integrasi fase 1

1. Isi `.env` dengan dua password acak yang berbeda untuk `MEDIAMTX_API_PASSWORD` dan `MEDIAMTX_READ_PASSWORD`. Compose menolak password kosong. Gunakan secret manager untuk produksi.
2. Set `MEDIAMTX_ENABLED=true`, `MEDIAMTX_ALLOWED_RTSP_HOSTS` berisi hostname/IP kamera secara persis (dipisahkan koma), dan `MEDIAMTX_WEBRTC_ADDITIONAL_HOSTS` berisi IP LAN/domain publik MediaMTX yang dapat dijangkau browser. Default `127.0.0.1` hanya untuk browser pada host yang sama.
3. Set `CORS_ORIGINS` ke origin dashboard. Produksi harus memakai HTTPS pada FastAPI/reverse proxy. UDP `8189` tetap menuju MediaMTX; signaling proxy tidak merelay media. Bila UDP tidak bisa dijangkau, konfigurasi TCP/TURN sesuai jaringan diperlukan.
4. Jalankan migrasi **sebelum** API baru, lalu restart MediaMTX dan API:

```bash
python -m alembic -c api/alembic.ini upgrade head
docker compose up -d --no-deps --force-recreate --wait mediamtx
docker compose up -d api
```

Perintah migrasi di atas memakai `DATABASE_URL` dari environment/`.env`. Jalankan dari root repo dalam environment Python yang memiliki dependencies. Untuk database lama yang dibuat dengan `create_all` tetapi belum memiliki revision Alembic, audit schema dan baseline revision terlebih dahulu; jangan menjalankan initial migration atau `stamp head` tanpa pemeriksaan. Startup `create_all` tidak menambahkan kolom pada tabel yang sudah ada.

#### Kontrak dashboard

Semua request berikut menggunakan `X-API-Key`; key yang terikat kamera tidak dapat mengakses kamera lain.

| Endpoint | Scope | Respons |
|---|---|---|
| `GET /api/v1/cameras/{camera_id}/stream` | `camera:read` | Desired state, status, `synced`, `whep_url`, codec aktual |
| `PATCH /api/v1/cameras/{camera_id}/stream` | `camera:write` | Body `{"enabled": true}` atau `false`; desired state tersimpan |
| `OPTIONS .../stream/whep` | `camera:read` | 204, `Accept-Post`, ICE server `Link` |
| `POST .../stream/whep` | `camera:read` | `application/sdp` offer, 201 SDP answer, `Location`, `ETag`, `Link` |
| `PATCH <Location>` | `camera:read`, key pemilik | `application/trickle-ice-sdpfrag`, `If-Match`; 204 atau 200 ICE restart answer |
| `DELETE <Location>` | `camera:read`, key pemilik | 204; menghapus sesi upstream dan mapping |

`Location` adalah URL **backend**, bukan URL internal MediaMTX. Gunakan header autentikasi juga pada OPTIONS/PATCH/DELETE, termasuk saat player ditutup. Script `reader.js` bawaan MediaMTX v1.20.0 tidak menyertakan auth pada PATCH/DELETE; dashboard memerlukan adapter sendiri. `ETag`, `Location`, `Link`, `Accept-Post`, dan `Accept-Patch` diekspos lewat CORS. Preflight CORS tidak mengakses stream; OPTIONS WHEP biasa tetap terautentikasi.

Sesi dibatasi `MEDIAMTX_SESSION_TTL_SEC` (default 24 jam, batas absolut). Player harus reconnect setelah kedaluwarsa. Sesi upstream hilang mendapat 404; layanan unavailable mendapat error aman 503. SDP dibatasi 64 KiB secara default. Offer POST tidak di-retry otomatis agar tidak membuat sesi ganda; timeout offer adalah `STREAM_TIMEOUT + 15` detik untuk startup source on-demand.

Stream baru default disabled. Source URL disimpan backend; path UUID dikelola on-demand lewat RTSP TCP. Rekonsiliasi berjalan setelah update dan setiap 30 detik, menggunakan advisory lock PostgreSQL untuk koordinasi lintas worker. CRUD tetap menyimpan desired state saat MediaMTX mati. PATCH stream dapat mengembalikan `synced=false`; dashboard melakukan GET/poll sampai sinkron. `disabled` = tidak diaktifkan, `idle` = source belum menyediakan frame/tanpa penonton, `online` = MediaMTX melaporkan online **dan available**, `offline` = source tidak online/available dengan reader atau available path, `unknown` = belum sinkron atau status tidak tersedia. Field upstream `online` saja bukan bukti ada frame pada v1.20.0. Idle bukan bukti kamera online maupun offline.

**Perubahan kontrak kamera:** `rtsp_url` pada respons selalu `null` (deprecated), gunakan `rtsp_configured`. Field write `rtsp_url` tetap menerima URL lengkap; dashboard harus mempertahankan nilai yang tidak diedit, bukan mengirim `null` saat menyimpan field lain. `stream_enabled` adalah field respons; enable dilakukan lewat `/stream`. Pembatasan key kamera juga diterapkan pada CRUD kamera. Koordinasikan adaptasi ini di `sivera-dashboard`.

Allowlist host kosong menolak semua source. Batasi egress MediaMTX ke jaringan/port kamera yang diizinkan; allowlist hostname bukan perlindungan lengkap terhadap DNS rebinding. Jangan menyimpan source credentials di metadata atau log. SQL parameter logging disembunyikan. Backend memproksikan hanya field signaling yang diperlukan, tidak meneruskan error mentah upstream.

Playback fase 1 tidak mencakup snapshot/prediksi RTSP atau transcoding. Capture berkala tersedia di fase 2 berikut. Kamera harus menyediakan codec browser-compatible (misalnya H264 yang sesuai). Uji video, ICE, reconnect, dan cleanup di browser nyata pada jaringan deployment sebelum rilis dashboard; respons SDP saja tidak membuktikan video dapat diputar.

Tes unit/route: `python -m pytest tests -q`. Tes integrasi opt-in ada di `tests/test_stream_integration.py`: `SIVERA_TEST_DATABASE_URL` harus menunjuk **database kosong disposable**, bukan database aplikasi. MediaMTX juga harus disposable, dengan `SIVERA_TEST_MEDIAMTX_API_PORT`, `SIVERA_TEST_MEDIAMTX_WHEP_PORT`, `SIVERA_TEST_MEDIAMTX_API_PASSWORD`, dan `SIVERA_TEST_MEDIAMTX_READ_PASSWORD`. Opsional `SIVERA_TEST_RTSP_SOURCE` untuk offer/PATCH/DELETE stream sintetis dan `SIVERA_TEST_MEDIAMTX_CONTAINER` untuk tes restart. Tes ini mengubah schema/path dan bisa restart container yang ditentukan; jangan arahkan ke produksi.

---

### 5. Capture kalibrasi RTSP (fase 2)

Capture membaca relay asli `sivera_<camera_uuid>` melalui RTSP TCP, bukan WebRTC atau path transcoding. FFmpeg menghasilkan JPEG tanpa resize, crop, undistort, atau autorotation. Dimensi frame pertama menjadi resolusi sesi; perubahan resolusi atau URL sumber menghentikan sesi dengan error. `captured_at` adalah waktu UTC frame diterima worker, **bukan** timestamp sensor kamera.

API, `worker`, dan `capture-worker` memakai volume bernama `artifacts` pada `/app/storage`. Prediction worker membaca queue `prediction` serta `celery` untuk menghabiskan job lama; capture worker hanya queue `capture` dengan concurrency 1. Task baru default menuju `prediction`, capture selalu menuju `capture`. Image backend sekarang memerlukan FFmpeg. Migrasi `004_calibration_captures` wajib dijalankan sebelum API/worker baru. Storage yang tidak writable menggagalkan startup, tidak beralih ke root lain.

**Sebelum deployment:** artifact lama di writable layer container `/app/storage` tidak otomatis masuk volume baru. Backup/salin data lama ke volume bersama dan verifikasi pemilik UID 1000 sebelum recreate container. Jangan menghapus container lama atau menjalankan `docker compose down -v` sebelum backup. Untuk host API/worker terpisah gunakan storage S3/private atau filesystem yang benar-benar dibagi; volume lokal Docker tidak dibagi antar-host.

```bash
python -m alembic -c api/alembic.ini upgrade head
docker compose up -d --build api worker capture-worker
```

Perintah ini hanya contoh deployment, bukan dijalankan otomatis. Konfigurasi MediaMTX fase 1 harus sudah tersedia. Enable stream terlebih dahulu; idle diperbolehkan karena sumber on-demand. Request start tidak menunggu kamera menyediakan frame.

| Endpoint di `/api/v1/cameras/{camera_id}/calibration-captures` | Scope | Hasil |
|---|---|---|
| `POST` | `calibration:write` | 202, `session_id` dan progress awal |
| `GET /{session_id}` | `calibration:read` | Status dan progress persisten |
| `POST /{session_id}/stop` | `calibration:write` | 200, stop idempoten, gambar tersimpan dipertahankan |
| `GET /{session_id}/images?offset=0&limit=20` | `calibration:read` | Daftar gambar dan `download_url` backend |
| `GET /{session_id}/images/{image_id}` | `calibration:read` | JPEG, private/no-store |

```bash
curl -X POST "http://localhost:8000/api/v1/cameras/$CAMERA_ID/calibration-captures" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"chessboard_cols":9,"chessboard_rows":6,"square_m":0.025,"interval_sec":2,"target_images":20,"max_duration_sec":300,"auto_calibrate":false}'
```

`target_images` fase 2 menghitung **frame tersimpan**, belum gambar lolos deteksi chessboard/ketajaman/diversitas. `auto_calibrate` default `false`; `true` mendapat 422. Endpoint `/compute`, quality selection, dan hasil kalibrasi nonaktif baru tersedia pada fase 3. Capture tidak mengubah kalibrasi aktif.

Progress: `attempted`, `captured`, `failed_attempts`, resolusi aktual, `last_capture_at`, `heartbeat_at`, `deadline_at`, `error_code`, dan `finish_reason`. Percobaan yang dibatalkan/crash dapat membuat `attempted > captured + failed_attempts`. Status `queued -> running -> completed`; stop berjalan melalui `stopping -> stopped`. Batas waktu menghasilkan `expired`, retry habis atau worker hilang menghasilkan `failed`. `completed` berarti target capture tercapai, bukan kalibrasi selesai. Semua API key tetap dibatasi scope dan kamera, termasuk akses `/files` ke artifact capture; gunakan fetch berheader API key untuk preview, bukan URL dengan key di query.

- Satu sesi aktif per kamera dipaksakan partial unique index PostgreSQL; start kedua mendapat 409 `capture_already_active` beserta session_id bila diketahui.
- `max_duration_sec` mulai saat sesi dibuat, termasuk antrean, maksimum 900 detik. Interval 1..60 detik dihitung **setelah percobaan sebelumnya selesai**, bukan jadwal presisi; target 1..100.
- Session row menjadi outbox dispatch. Broker unavailable tetap mendapat 202 queued; reconciler API retry maksimal satu dispatch per 15 detik/sesi. Queue timeout default 120 detik menghasilkan `capture_queue_timeout`.
- Heartbeat diperbarui sekitar satu detik selama capture/wait. Default stale 45 detik menghasilkan `capture_worker_lost`; redelivery tidak melanjutkan sesi running lama. Worker token mencegah proses lama menulis setelah sesi ditutup. Operator membuat sesi baru setelah gagal.
- Default timeout FFmpeg 15 detik per frame, maksimum 3 kegagalan berturut-turut, frame 10 MiB/16 megapixel (dibatasi sebelum decode dan saat validasi JPEG). Task hard limit 960 detik. Timer `timeout` dari coreutils membatasi FFmpeg bahkan setelah child Celery mati. Stop/deadline membatalkan process group; DB hanya dipegang selama transaksi pendek, tidak selama capture/storage.
- Local writes atomik. Gambar sesi stopped/failed/expired tetap tersedia. Tidak ada retention otomatis: pantau kapasitas disk. Crash persis setelah storage write sebelum DB commit dapat meninggalkan file orphan; camera deletion menghapus metadata lewat cascade, bukan file fisik. Bucket S3 wajib private; lifecycle/cleanup artifact harus dikelola operator.

Tes tambahan: `tests/test_calibration_capture_integration.py` memakai database/MediaMTX disposable, `SIVERA_TEST_MEDIAMTX_RTSP_PORT`, source sintetis, serta FFmpeg di PATH. Tes queue memerlukan `SIVERA_TEST_CELERY_BROKER_URL`, `SIVERA_TEST_STORAGE_ROOT`, dan capture-worker yang menunjuk layanan/storage yang sama. `SIVERA_TEST_CAPTURE_CRASH=true` mengizinkan SIGKILL pada child capture-worker disposable di host pengujian; jangan aktifkan untuk worker pengguna/produksi.

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
