# Revisi Total Baseline Repo untuk Migrasi ke SEANO Alfin7

## 1. Tujuan revisi
Migrasi baseline dari SEANO BIMA30 ke SEANO Alfin7 mengubah:
- geometri kapal,
- geometri kamera,
- resolusi citra,
- status validitas parameter visual risk.

Akibatnya, angka lama yang terkait BIMA30 tidak boleh lagi menjadi default aktif atau design-lock.

## 2. Nilai yang boleh langsung dikunci sekarang
Gunakan nilai berikut sebagai baseline resmi baru yang sudah pasti:

- `vehicle_name = SEANO Alfin7`
- `L = 0.70 m`
- `B = 0.50 m`
- `h_c = 0.58 m`
- `image_width = 640 px`
- `image_height = 480 px`
- `c_x = 320 px`
- `c_y = 240 px`
- `W_p_min = 0.50 m`

## 3. Nilai yang jangan dikunci numerik dulu
Jangan pakai angka final untuk parameter ini sampai ada FOV valid atau hasil kalibrasi kamera:

- `camera_hfov_deg`
- `camera_vfov_deg`
- `f_x`
- `f_y`
- `center_band_ratio`
- `bottom_danger_ratio`
- `near_area_ratio`

## 4. File yang wajib direvisi

### A. Dokumentasi
1. `README.md`
2. `docs/BASELINE_PARAMETER_LOCK.md`
3. `docs/ARCHITECTURE.md`
4. `docs/LAUNCH_STATUS_MAP.md`
5. `docs/HARDWARE_BENCH_RESULTS.md`
6. `docs/PHASE6_RESULTS_SUMMARY.md` bila ada narasi yang mengunci geometri lama
7. `docs/RUNBOOK.md` bila ada contoh parameter visual yang masih berbasis BIMA30

### B. Runtime / launch
1. `seano_ca_ws/src/seano_vision/seano_vision/risk_evaluator_node.py`
2. `seano_ca_ws/src/seano_vision/launch/demo_full_ca.launch.py`
3. `seano_ca_ws/src/seano_vision/launch/phase5_mission_avoid_integration.launch.py`
4. `seano_ca_ws/src/seano_vision/launch/phase7_cuav_usb_hardware.launch.py`

## 5. Perubahan inti yang harus dilakukan di kode

### 5.1 risk_evaluator_node.py
Masalah saat ini:
- node masih punya default lama untuk:
  - `camera_hfov_deg = 112.96`
  - `center_band_ratio = 0.212`
  - `bottom_danger_ratio = 0.647`
  - `near_area_ratio = 0.0183`

Revisi yang disarankan:
- jangan biarkan node mengandalkan angka BIMA30 sebagai default aktif;
- ganti menjadi parameter yang dibaca dari launch atau YAML profil;
- untuk sementara, gunakan salah satu dua strategi:

#### Strategi aman A
Pertahankan parameter wajib, tetapi beri nilai placeholder konservatif dan log warning keras bahwa angka belum final.

#### Strategi aman B
Lebih baik: node fail-fast bila profil Alfin7 aktif tetapi `camera_hfov_deg`, `center_band_ratio`, `bottom_danger_ratio`, atau `near_area_ratio` belum diberikan eksplisit.

Rekomendasi akhir:
- gunakan Strategi B untuk hardware final;
- gunakan Strategi A hanya untuk eksperimen transisional.

### 5.2 demo_full_ca.launch.py
Masalah saat ini:
- launch ini membuat `risk_evaluator_node`, tetapi belum meneruskan parameter geometri visual.

Revisi:
- tambahkan launch arguments baru:
  - `ca_camera_hfov_deg`
  - `ca_center_band_ratio`
  - `ca_bottom_danger_ratio`
  - `ca_near_area_ratio`
- teruskan semua argumen itu ke `risk_evaluator_node`.

### 5.3 phase5_mission_avoid_integration.launch.py
Masalah saat ini:
- `phase5` meng-include `demo_full_ca.launch.py`, tetapi belum mem-forward parameter geometri visual.

Revisi:
- tambahkan forwarding untuk:
  - `ca_camera_hfov_deg`
  - `ca_center_band_ratio`
  - `ca_bottom_danger_ratio`
  - `ca_near_area_ratio`
- gunakan profil simulasi terpisah dari hardware jika memang diperlukan.

### 5.4 phase7_cuav_usb_hardware.launch.py
Keadaan saat ini:
- resolusi kamera sudah sesuai baseline baru: `640x480`.

Revisi:
- pertahankan `ca_camera_device_width = 640`
- pertahankan `ca_camera_device_height = 480`
- tambahkan argumen geometri visual yang sama seperti pada `demo_full_ca.launch.py`
- teruskan ke include CA pipeline
- jangan lagi mengandalkan default BIMA30 dari node

## 6. Desain konfigurasi yang paling benar
Buat file YAML baseline baru, misalnya:

`seano_ca_ws/src/seano_vision/config/alfin7_baseline.yaml`

Contoh isi awal:

```yaml
vehicle:
  name: seano_alfin7
  length_m: 0.70
  beam_m: 0.50
  camera_height_m: 0.58

image:
  width_px: 640
  height_px: 480
  cx_px: 320
  cy_px: 240

risk_geometry:
  camera_hfov_deg: null
  camera_vfov_deg: null
  center_band_ratio: null
  bottom_danger_ratio: null
  near_area_ratio: null
  status: pending_camera_calibration
```

Catatan:
- jangan isi angka palsu agar repo terlihat lengkap;
- `null` jauh lebih jujur daripada warisan angka BIMA30 yang salah.

## 7. Struktur revisi untuk docs/BASELINE_PARAMETER_LOCK.md
Ganti bagian design-lock agar menjadi seperti ini:

```md
## 2. Vehicle and Camera Baseline (Design-Lock)

| Parameter | Nilai | Status | Catatan |
|---|---:|---|---|
| nama platform | SEANO Alfin7 | design-lock | platform final penelitian |
| panjang kapal `L` | 0.70 m | design-lock | basis geometri platform final |
| lebar kapal `B` | 0.50 m | design-lock | basis protected corridor minimum |
| tinggi kamera `h_c` | 0.58 m | design-lock | baseline kamera final |
| resolusi citra `W` | 640 px | design-lock | runtime image size final |
| resolusi citra `H` | 480 px | design-lock | runtime image size final |
| pusat citra `c_x` | 320 px | design-lock | turunan dari resolusi final |
| pusat citra `c_y` | 240 px | design-lock | turunan dari resolusi final |
| protected corridor minimum `W_p,min` | 0.50 m | design-lock | minimum sama dengan beam kapal |
| pitch kamera `alpha` | 0 deg | baseline-default | boleh berubah bila mounting berubah |
```

Lalu pindahkan parameter lama berikut keluar dari design-lock:
- `camera_hfov_deg`
- `center_band_ratio`
- `bottom_danger_ratio`
- `near_area_ratio`
- `b_t`
- `d_t`
- `W_o`
- `H_o`

Status yang disarankan:
- `camera_hfov_deg`, `camera_vfov_deg`, `f_x`, `f_y` -> `not-final-yet`
- `center_band_ratio`, `bottom_danger_ratio`, `near_area_ratio` -> `not-final-yet`
- `d_t` -> `baseline-default` atau `not-final-yet`
- `b_t` -> `not-final-yet` sampai spacing thruster Alfin7 dipastikan

## 8. Rumus yang berlaku sekarang

### center_band_ratio
```text
center_band_ratio = 2 * atan(W_p / (2*d_t)) / HFOV_rad
```
Untuk minimum corridor:
```text
center_band_ratio_min = 2 * atan(0.25 / d_t) / HFOV_rad
```

### bottom_danger_ratio
```text
bottom_danger_ratio = (240 + f_y * tan(atan(0.58 / d_t) - alpha)) / 480
```
Jika `alpha = 0`:
```text
bottom_danger_ratio = (240 + f_y * (0.58 / d_t)) / 480
```

### near_area_ratio
```text
near_area_ratio = (f_x * f_y * W_o * H_o) / (640 * 480 * d_t^2)
```

## 9. Revisi README yang disarankan
Ubah narasi pembuka agar tidak lagi mengunci BIMA30. Gunakan bentuk ini:

```md
## Vision-Based Collision Avoidance for SEANO USV

This repository contains a ROS 2 Humble based collision avoidance stack for the SEANO USV platform. The current thesis baseline is migrated to SEANO Alfin7 for hardware validation, while simulation and hardware integration are maintained through the active Phase 5 and Phase 7 launch baselines.
```

Lalu tambahkan satu subbagian pendek:

```md
### Current Final Hardware Geometry
- platform: SEANO Alfin7
- length: 0.70 m
- beam: 0.50 m
- camera height: 0.58 m
- processing resolution: 640 x 480
```

## 10. Keputusan engineering yang paling penting
1. Resolusi `640x480` sudah sinkron dengan hardware launch dan tidak perlu diubah lagi.
2. Default visual-risk lama dari BIMA30 harus dicabut dari status baseline aktif.
3. Repo harus membedakan tegas:
   - geometri final yang sudah pasti,
   - parameter visual yang masih menunggu FOV/kalibrasi,
   - tuning dinamik kapal yang masih menunggu uji bench atau uji air.
4. Jangan mengisi angka final untuk `center_band_ratio`, `bottom_danger_ratio`, dan `near_area_ratio` tanpa FOV atau hasil kalibrasi kamera.

## 11. Minimum patch plan
Urutan edit paling aman:
1. `docs/BASELINE_PARAMETER_LOCK.md`
2. `README.md`
3. `risk_evaluator_node.py`
4. `demo_full_ca.launch.py`
5. `phase5_mission_avoid_integration.launch.py`
6. `phase7_cuav_usb_hardware.launch.py`
7. dokumen pendukung lain

## 12. Catatan akhir
Untuk menyelesaikan revisi numerik final pada `camera_hfov_deg`, `center_band_ratio`, `bottom_danger_ratio`, dan `near_area_ratio`, masih dibutuhkan salah satu dari dua input berikut:
- FOV kamera A95 yang valid untuk mode 4:3, atau
- hasil kalibrasi kamera (`f_x`, `f_y`, distorsi)
