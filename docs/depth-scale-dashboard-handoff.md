# Pesan perubahan untuk dashboard

Backend menyediakan alur depth-scale berikut. Perubahan dashboard dikerjakan di repository dashboard; backend tidak mengubah source, snapshot, atau rencana historis dashboard.

## Alur dan request

Semua operasi memakai `X-API-Key`. Scope read adalah `ground_ref:read`, write `ground_ref:write`; key terikat kamera hanya boleh mengakses kamera itu.

1. Baca kalibrasi aktif dan `GET /api/v1/cameras/{camera_id}/ground-models/active`. Ground-model aktif sekarang dipilih agar cocok dengan kalibrasi/reference aktif, sama dengan prediction. Simpan ID yang dikembalikan, jangan memakai ID lama dari cache.
2. Tetapkan base path `B = /api/v1/cameras/{camera_id}/ground-models/{ground_model_id}/depth-scales`. `GET B/active` mengembalikan scale atau 404 jika belum ada. 404 bukan error jaringan dan bukan alasan mengisi scale arbitrer.
3. Unggah gambar melalui `POST B/test-images`, multipart dengan satu field `file`. Biarkan HTTP client membuat multipart boundary. Respons 201:

   ```json
   {
     "image_id": "<UUID>",
     "test_image": "cameras/<camera UUID>/depth-scale-images/<ground UUID>/<image UUID>/source.image",
     "image_url": "<backend URL>/test-images/<image UUID>/image",
     "width": 1920,
     "height": 1080,
     "coordinate_system": "undistorted_pixels",
     "distance_type": "camera_axis_z_m"
   }
   ```

   Dimensi di atas hanya contoh. Gambar raw harus tepat berukuran kalibrasi. Kalibrasi legacy tanpa K asli memerlukan gambar yang sudah undistorted pada dimensi kamera; jangan mengundistort ulang. `/files` bukan API upload.
4. Fetch `image_url` dengan header key dari origin backend yang dipercaya, lalu tampilkan blob URL. Responsnya PNG undistorted; revoke blob URL saat diganti/unmount. Pilih titik pada preview ini. Konversikan posisi tampilan ke ukuran `width`/`height` sebenarnya, termasuk offset letterboxing; bulatkan menjadi pixel integer dan pastikan `0 <= x < width`, `0 <= y < height`.
5. Kirim `POST B` dengan JSON berikut. Nilai titik/jarak hanya contoh struktur; gunakan pengukuran sebenarnya:

   ```json
   {
     "test_image": "<test_image dari respons upload>",
     "depth_model": "yolo26n-depth.pt",
     "known_distances": [
       {"x": 100, "y": 200, "distance_m": 3.0},
       {"x": 300, "y": 400, "distance_m": 5.0},
       {"x": 500, "y": 600, "distance_m": 7.0}
     ],
     "clip_min_m": 0.1,
     "clip_max_m": 50.0
   }
   ```

   Minimal tiga, maksimal 100 titik berbeda, dengan kedalaman yang bervariasi. `distance_m` adalah Z pada sumbu optik kamera dalam meter, bukan panjang ray/rangefinder. Jika alat memberikan jarak Euclidean `r`, operator harus memastikan geometri dan titik pengukuran sesuai sebelum mengonversi `Z = r / sqrt(1 + ((x-cx)/fx)^2 + ((y-cy)/fy)^2)` memakai K undistorted. Jangan otomatis menganggap semua angka rangefinder sebagai Z. Nama model harus berupa file `.pt` yang sudah terpasang di backend, bukan path/URL. Alternatif `test_image` adalah key yang terdaftar tepat dalam `calib_images` kalibrasi ini; alur upload baru lebih mudah untuk memastikan koordinat preview.
6. Creation sinkron mengembalikan 201 `DepthScaleResponse` dan langsung mengaktifkan hasil yang lolos, menggantikan scale aktif untuk ground model itu secara atomik. Jangan polling seolah ada `job_id`. Sediakan status sedang menghitung dan timeout client di atas batas inferensi deployment (default backend 120 detik, ditambah waktu upload/processing). Matikan submit ganda dan jangan retry otomatis setelah timeout jaringan karena request tidak idempoten; refetch list/active dahulu.
7. `GET B` mengembalikan `{items,total}`; `GET B/{depth_scale_id}` membaca detail. `POST B/{depth_scale_id}/activate` mengaktifkan hasil tersimpan dengan respons 200. Creation/activation dapat ditolak jika ground model tidak lagi cocok dengan kalibrasi/reference aktif. Refetch active scale setelah perubahan dan sebelum menawarkan prediction. Submit prediction tetap multipart 202, lalu polling job seperti kontrak sebelumnya.

## Error dan tampilan hasil

- 400: gambar/resolusi/intrinsics salah, titik di luar gambar, depth tidak valid, fit degenerat atau residual terlalu besar. Tampilkan kode dari `detail.code`, beri kesempatan memperbaiki gambar/pengukuran. Semua titik dipakai atau request ditolak; backend tidak membuang titik secara diam-diam.
- 422: validasi JSON/form, termasuk kurang titik, pixel noninteger/duplikat, angka nonfinite, clip terbalik, atau nama model/key tidak valid. Normalisasi error FastAPI berupa list, object `{code}`, dan string legacy.
- 404: resource atau gambar tidak ada/tidak terdaftar; `/active` 404 berarti scale belum tersedia.
- 409 `depth_scale_prerequisites_changed`: muat ulang kalibrasi/ground model, unggah ulang jika ID ground model berubah. `depth_scale_invalid_saved_fit`: hasil lama tidak layak diaktifkan; buat hasil baru dengan pengukuran benar.
- 503: model/storage/inferensi/persistence tidak tersedia. Kode `depth_scale_busy` dan `depth_scale_timeout` dapat ditampilkan sebagai kondisi sementara, tetapi jangan retry POST otomatis.
- 401/403: key invalid atau scope/kamera ditolak. Jangan mengubahnya menjadi empty state.

Tampilkan `scale_s`, `offset_b`, `mean_residual_m`, `max_residual_m`, jumlah residual, dan status aktif. Residual per titik adalah `fitted - measured` dalam meter; agregat memakai nilai absolut. Default batas fit adalah mean 0.25 m, max 0.5 m, dan relative per titik 10%, tetapi operator dapat mengubah setting. Label hasil sebagai fit yang lolos pemeriksaan, bukan akurasi metrik terjamin. Pengukuran held-out dan validasi volume kamera asli tetap diperlukan. Jangan menampilkan estimasi yang hilang sebagai nol.

BEV memakai `test_image_path`/`mask_path` sebagai storage key relatif, bukan path OS. Gambar mengikuti raw/legacy contract yang sama; mask harus sudah undistorted dengan dimensi sesuai dan scope baca domain artifact tersebut. Preview tidak membuktikan rekonstruksi 3D.

Tim dashboard perlu memperbarui service/types depth-scale, form upload dan titik, preview berheader, invalidasi cache per camera/calibration/ground ID, serta snapshot OpenAPI dari backend terbaru. Jangan menimpa pekerjaan dashboard yang sudah ada. Pengaturan `MEDIAMTX_ALLOWED_RTSP_HOSTS` tidak lagi dipakai untuk validasi host RTSP; validasi skema/port/fragment tetap berlaku. Operator tetap wajib membatasi egress jaringan kamera di lapisan deployment. Tidak ada perubahan image/version MediaMTX.
