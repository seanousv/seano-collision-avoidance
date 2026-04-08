# LAUNCH STATUS MAP

Dokumen ini adalah peta kerja cepat untuk membedakan launch yang:
- aktif dipakai untuk simulasi
- aktif dipakai untuk hardware
- dipakai untuk bench/debug
- ditunda auditnya sampai setelah pengujian kapal

> Prinsip:
> - Jangan rename file launch aktif sebelum pengujian kapal selesai.
> - Jangan pindahkan file launch aktif ke folder lain dulu.
> - Gunakan dokumen ini sebagai referensi operasional harian.

---

## 0. CARA MEMBACA REPO SAAT INI

Repo ini saat ini harus dibaca sebagai **baseline aktif TA**, bukan hanya workspace eksperimen umum.

Interpretasi operasional yang dipakai sekarang:
- baseline simulasi utama dipusatkan di `phase5_mission_avoid_integration.launch.py`
- baseline hardware utama dipusatkan di `phase7_cuav_usb_hardware.launch.py`
- nama file launch aktif sengaja dipertahankan dulu agar konsisten dengan evidence, slide progres, dan baseline pengujian

Dokumen ini membantu menjaga agar operator/pengembang tidak salah memilih launch aktif di tengah banyaknya file legacy atau bench launch.

## 1. LAUNCH AKTIF UTAMA

### [ACTIVE_SIM] phase5_mission_avoid_integration.launch.py
**Fungsi utama**
- simulasi SITL mission -> avoid -> rejoin -> mission
- baseline utama simulasi

**Dipakai saat**
- uji SITL end-to-end
- validasi state machine `MISSION -> AVOID -> REJOIN -> MISSION`
- pengambilan rosbag untuk Phase 6 metrics

**Catatan**
- ini adalah launch utama simulasi
- jalankan bersama SITL + MAVROS + Mission Planner

---

### [ACTIVE_HW] phase7_cuav_usb_hardware.launch.py
**Fungsi utama**
- integrasi hardware penuh Jetson + CUAV X7+ + USB camera + detector + risk + watchdog + control

**Dipakai saat**
- bench hardware penuh
- uji air / uji kapal nyata
- pengujian mission with collision avoidance pada hardware

**Catatan**
- ini adalah launch utama hardware
- satu run paling lengkap untuk target collision avoidance nyata
- parameter aktif hardware sebaiknya dibaca bersama `docs/BASELINE_PARAMETER_LOCK.md`

---

## 2. LAUNCH AKTIF BENCH / DEBUG

### [BENCH_CAM] phase2_camera_usb_test.launch.py
**Fungsi**
- validasi kamera USB by-id
- baseline source kamera hardware

**Dipakai saat**
- cek kamera
- debug pipeline image
- validasi sebelum menjalankan launch hardware besar

---

### [BENCH_DET] demo_detect.launch.py
**Fungsi**
- jalur ringan `camera -> detector`

**Dipakai saat**
- cek raw camera
- cek annotated image
- cek detections
- precheck lapangan cepat

---

### [BENCH_RISK] demo_risk.launch.py
**Fungsi**
- jalur `camera -> detector -> risk`

**Dipakai saat**
- cek keluaran risk
- cek command tanpa integrasi hardware penuh
- precheck lapangan sebelum Phase 7

---

### [BENCH_FULL_CA] demo_full_ca.launch.py
**Fungsi**
- pipeline CA lengkap untuk bench/debug
- memuat detector, risk, watchdog, dan modul CA lain sesuai toggle

**Dipakai saat**
- bench perception/CA pipeline
- debugging HUD `/ca/debug_image`
- pengujian modul CA non-hardware

---

## 3. NODE AKTIF INTI SISTEM

### Perception
- `camera_node.py`
- `detector_node.py`
- `risk_evaluator_node.py`
- `watchdog_failsafe_node.py`

### Control
- `command_mux_node.py`
- `actuator_safety_limiter_node.py`
- `mavros_rc_override_bridge_node.py`
- `auto_controller_stub_node.py`

### Mission / Mode
- `mission_mode_manager_node.py`

---

## 4. PETA PEMAKAIAN CEPAT

### Jika ingin simulasi SITL menyeluruh
Pakai:
- `phase5_mission_avoid_integration.launch.py`

### Jika ingin uji kapal nyata / full hardware
Pakai:
- `phase7_cuav_usb_hardware.launch.py`

### Jika ingin cek kamera + detector saja
Pakai:
- `phase2_camera_usb_test.launch.py`
- `demo_detect.launch.py`

### Jika ingin cek risk tanpa integrasi penuh
Pakai:
- `demo_risk.launch.py`

### Jika ingin bench full CA pipeline
Pakai:
- `demo_full_ca.launch.py`

---

## 5. LAUNCH KANDIDAT AUDIT SETELAH PENGUJIAN KAPAL

> Status:
> - jangan dipindah dulu
> - jangan dihapus dulu
> - audit setelah pengujian kapal selesai

### [AUDIT_LATER]
- `demo_mavros_actuation_test.launch.py`
- `lake_auto_demo.launch.py`
- `lake_auto_demo_hw.launch.py`
- `phase1_maneuver_record.launch.py`
- `phase1_maneuver_test.launch.py`
- `phase2_camera_detector_test.launch.py`
- `phase2_camera_detector_watchdog_test.launch.py`
- `phase2_camera_source_test.launch.py`
- `phase3_watchdog_camera_only.launch.py`
- `phase4_takeover_actuation_test.launch.py`
- `run_auto_stack.launch.py`

---

## 6. ATURAN OPERASIONAL

### Untuk simulasi
- fokus ke `phase5`
- gunakan SITL + Mission Planner + MAVROS + Phase 5

### Untuk hardware nyata
- fokus ke `phase7`
- gunakan browser monitoring:
  - raw
  - annotated
  - HUD

### Untuk precheck cepat
- gunakan `demo_detect` atau `demo_risk`

---

## 7. RULE OF THUMB

### Gunakan ini:
- **SITL utama** -> `phase5_mission_avoid_integration.launch.py`
- **Hardware utama** -> `phase7_cuav_usb_hardware.launch.py`
- **Deteksi saja** -> `demo_detect.launch.py`
- **Risk saja** -> `demo_risk.launch.py`
- **Bench CA lengkap** -> `demo_full_ca.launch.py`

### Jangan lakukan dulu:
- rename file launch aktif
- pindahkan launch aktif ke folder lain
- refactor struktur launch sebelum pengujian kapal selesai

---

## 8. TUJUAN DOKUMEN INI

Dokumen ini dibuat agar operator/pengembang bisa cepat memilih launch yang tepat tanpa perlu menebak-nebak fungsi setiap file di folder `launch/`.

Jika pengujian kapal sudah selesai dan baseline final sudah dikunci, barulah daftar `[AUDIT_LATER]` bisa ditinjau untuk:
- dipertahankan
- dipindah ke folder arsip
- atau dihapus
