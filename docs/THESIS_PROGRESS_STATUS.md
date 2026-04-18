# THESIS PROGRESS STATUS
## SEANO Mission-Aware Collision Avoidance — Status Baseline TA Saat Ini

Dokumen ini dipakai untuk menyatakan **posisi proyek saat ini secara jujur, profesional, dan sinkron** dengan:
- baseline simulasi yang sudah menghasilkan evidence kuantitatif,
- baseline hardware bench yang sudah menghasilkan evidence integrasi,
- batas klaim yang **sudah boleh** dan **belum boleh** dinyatakan final,
- serta prioritas kerja berikutnya menuju pembuktian lapangan.

Dokumen ini **bukan** roadmap awal. Dokumen ini adalah **status operational thesis baseline**.

---

## 1. Cara Membaca Proyek Saat Ini

Repo ini saat ini harus dibaca sebagai:

1. **simulation baseline** sebagai sumber utama evidence kuantitatif perilaku sistem,
2. **hardware bench baseline** sebagai sumber utama evidence integrasi pada platform target,
3. **field validation** sebagai tahap berikutnya yang masih harus dijalankan secara bertahap dan aman.

Artinya, proyek ini **sudah melampaui proof-of-concept simulasi sederhana**, tetapi **belum boleh langsung diklaim** sebagai sistem avoidance lapangan yang final tanpa uji air terkontrol.

---

## 2. Baseline Aktif yang Saat Ini Menjadi Acuan Resmi

### 2.1 Simulation baseline
Launch utama:
- `phase5_mission_avoid_integration.launch.py`

Peran:
- validasi state machine `MISSION -> AVOID -> REJOIN -> MISSION`
- validasi takeover, release, dan rejoin
- validasi recovery saat perception hilang dan kembali aktif
- rosbag recording
- extraction metrics kuantitatif

### 2.2 Hardware bench baseline
Launch utama:
- `phase7_cuav_usb_hardware.launch.py`

Peran:
- Jetson runtime
- USB camera
- detector
- risk evaluator
- watchdog/failsafe
- command chain
- MAVROS
- FCU / autopilot nyata
- browser monitoring

---

## 3. Status Simulasi Saat Ini

Status umum:
- **kuat untuk evidence perilaku sistem**
- **layak dipakai di laporan, seminar, dan jurnal**

Yang sudah berhasil ditunjukkan oleh baseline simulasi:
- mission-following normal,
- hazard-triggered takeover,
- release kembali setelah kondisi aman,
- rejoin ke mission,
- repeated hazard handling,
- failsafe activation,
- failsafe recovery.

### 3.1 Makna akademik simulasi
Simulation baseline saat ini bukan hanya demo visual. Baseline ini sudah berfungsi sebagai:
- bukti integrasi end-to-end,
- bukti state-machine behavior,
- sumber metrik kuantitatif,
- dasar interpretasi engineering pada paper dan laporan TA.

### 3.2 Klaim yang sudah boleh dibuat dari simulasi
Sudah boleh dinyatakan bahwa:
- collision avoidance mission-aware **berhasil tervalidasi pada baseline SITL**,
- sistem dapat melakukan **avoid -> release -> rejoin**,
- sistem tetap dapat menunjukkan recovery pada repeated hazard,
- sistem sudah memiliki jalur failsafe dan recovery yang tervalidasi.

### 3.3 Klaim yang belum boleh dilebihkan dari simulasi
Belum boleh dinyatakan bahwa:
- performa simulasi otomatis sama dengan performa lapangan,
- obstacle avoidance nyata di air sudah final,
- tuning hardware final sudah selesai hanya karena simulasi sudah kuat.

---

## 4. Status Hardware Bench Saat Ini

Status umum:
- **matang sebagai integration-level evidence**
- **belum identik dengan field success**

Yang sudah berhasil dibuktikan pada hardware bench:
- perception-to-command berjalan di Jetson,
- Jetson terhubung ke FCU melalui MAVROS,
- topic state/battery/IMU dapat terbaca,
- Mission Planner berhasil divalidasi dalam jalur bench,
- browser monitoring raw / annotated / HUD dapat dipakai,
- hardware baseline sudah mendekati arsitektur operasional nyata.

### 4.1 Posisi hardware bench dalam TA
Hardware bench saat ini harus dibaca sebagai:
- bukti bahwa stack sudah hidup pada platform target,
- bukti bahwa integrasi dengan autopilot bukan lagi asumsi,
- jembatan menuju dockside test dan obstacle test terkontrol.

### 4.2 Klaim yang sudah boleh dibuat dari hardware bench
Sudah boleh dinyatakan bahwa:
- runtime target Jetson aktif,
- FCU nyata telah terhubung,
- MAVROS bridge tervalidasi,
- Mission Planner integration tervalidasi di bench,
- perception chain aktif pada hardware,
- command chain aktif hingga tahap integrasi bench.

### 4.3 Klaim yang belum boleh dibuat dari hardware bench
Belum boleh dinyatakan bahwa:
- mission complete dengan avoid + rejoin di air sudah final,
- obstacle run di lapangan sudah tervalidasi penuh,
- rejoin lapangan sudah konsisten dalam berbagai kondisi air dan pencahayaan.

---

## 5. Parameter Baseline yang Saat Ini Harus Dianggap Resmi

Dokumen parameter resmi ada di:
- `docs/BASELINE_PARAMETER_LOCK.md`

Prinsip pembacaan:
- geometri platform final dan resolusi citra final boleh dianggap **baseline design-lock values**,
- parameter visual yang masih bergantung pada FOV atau kalibrasi kamera (`camera_hfov_deg`, `camera_vfov_deg`, `f_x`, `f_y`, `center_band_ratio`, `bottom_danger_ratio`, `near_area_ratio`) harus tetap dianggap **not-final-yet** sampai data kamera valid tersedia,
- parameter dinamik kapal dan aktuasi tetap dianggap **runtime-tunable** sampai data uji nyata cukup.

---

## 6. Batas Validasi Saat Ini

Secara ringkas:

### Sudah tervalidasi
- architecture-level integration
- mode logic dan supervisory behavior
- simulation quantitative evidence
- hardware bench integration
- monitoring and observability

### Belum tervalidasi final
- on-water repeated obstacle run
- on-water release + rejoin consistency
- full mission completion after real avoidance in controlled water trial
- final dynamic tuning for thrust and timing thresholds

Dokumen pendukung batas validasi:
- `docs/VALIDATION_BOUNDARY.md`

---

## 7. Risiko Interpretasi yang Harus Dihindari

Agar laporan tetap kuat dan kredibel, hindari interpretasi berikut:

1. Menyamakan **hardware bench success** dengan **field success**.
2. Menyamakan **synthetic/video-based SITL success** dengan **real perception robustness**.
3. Mengklaim threshold dinamik sudah final bila data uji kapal nyata belum cukup.
4. Menggambarkan proyek sebagai “masih sekadar eksperimen awal”, padahal baseline aktifnya sudah jauh lebih matang dari itu.

Framing yang paling tepat adalah:

> Proyek sudah memiliki simulation baseline yang kuat untuk evidence kuantitatif dan hardware bench baseline yang matang untuk evidence integrasi, namun finalisasi performa lapangan tetap memerlukan validasi air yang bertahap dan aman.

---

## 8. Prioritas Kerja Berikutnya

Urutan kerja yang paling rasional dari posisi proyek saat ini adalah:

### Prioritas 1 — kunci baseline dokumen dan parameter
- sinkronkan README, architecture, launch status, dan progress docs
- kunci parameter visual yang sudah final sementara
- pisahkan jelas parameter design-lock dan runtime-tuning

### Prioritas 2 — jaga simulation baseline tetap menjadi sumber evidence utama
- rapikan rosbag naming
- rapikan tabel/grafik hasil
- jaga agar phase5 tetap repeatable dan mudah diaudit

### Prioritas 3 — lanjutkan hardware secara konservatif
- dockside validation
- AUTO tanpa obstacle
- obstacle sederhana terkontrol
- release + rejoin terkontrol

### Prioritas 4 — baru setelah itu masuk ke klaim lapangan yang lebih kuat
- obstacle run nyata
- rejoin nyata
- mission continuation setelah avoidance

---

## 9. Kesimpulan Singkat

Posisi proyek saat ini paling tepat diringkas sebagai berikut:

- **simulation baseline** sudah cukup kuat untuk menjadi evidence kuantitatif utama TA,
- **hardware bench baseline** sudah cukup matang untuk menjadi evidence integrasi pada platform target,
- proyek sudah bergerak dari proof-of-concept menuju pembuktian sistem nyata,
- tetapi validasi lapangan penuh tetap harus dilakukan bertahap dan tidak boleh dilebihkan sebelum evidence on-water benar-benar cukup.
