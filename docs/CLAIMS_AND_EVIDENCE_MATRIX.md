# CLAIMS AND EVIDENCE MATRIX

Dokumen ini memetakan **klaim akademik** yang boleh diambil dari repository dan hasil pengujian yang ada saat ini.

Tujuannya:
- menjaga konsistensi antara kode, laporan, slide, dan jurnal,
- mencegah overclaim,
- memudahkan penyusunan kalimat hasil dan kesimpulan.

---

## 1. Cara membaca dokumen ini

Setiap klaim dibagi menjadi:
- **boleh diklaim sekarang**,
- **evidence minimum**,
- **batas klaim**,
- **status saat ini**.

Status yang dipakai:
- `READY_TO_CLAIM`
- `CLAIM_WITH_LIMITATION`
- `NOT_READY_TO_CLAIM`

---

## 2. Matrix klaim utama

### Claim A — Arsitektur mission-aware collision avoidance sudah terimplementasi

**Status**: `READY_TO_CLAIM`

**Evidence minimum**:
- `README.md`
- `docs/ARCHITECTURE.md`
- launch aktif `phase5` dan `phase7`
- node inti mission, control, risk, watchdog, dan bridge

**Batas klaim**:
- ini adalah klaim tentang **arsitektur dan implementasi sistem**,
  bukan bukti keberhasilan lapangan penuh.

---

### Claim B — Baseline simulasi sudah menunjukkan alur MISSION -> AVOID -> REJOIN -> MISSION

**Status**: `READY_TO_CLAIM`

**Evidence minimum**:
- `docs/PHASE6_RESULTS_SUMMARY.md`
- rosbag metrics Phase 6
- state/event dari mode manager
- slide dan ringkasan hasil simulasi

**Batas klaim**:
- klaim ini berlaku untuk **simulation baseline / SITL evidence**,
  bukan langsung untuk hardware nyata di air.

---

### Claim C — Perception, risk, watchdog, dan RC override chain sudah berjalan pada hardware bench

**Status**: `READY_TO_CLAIM`

**Evidence minimum**:
- `docs/HARDWARE_BENCH_RESULTS.md`
- `phase7_cuav_usb_hardware.launch.py`
- topic monitoring hardware
- bukti Jetson + CUAV + MAVROS + USB camera + browser monitoring

**Batas klaim**:
- ini adalah klaim **integrasi hardware bench**,
  belum berarti obstacle avoidance lapangan sudah final.

---

### Claim D — Sistem sudah mampu melakukan collision avoidance nyata di air secara penuh dan konsisten

**Status**: `NOT_READY_TO_CLAIM`

**Evidence minimum yang dibutuhkan**:
- AUTO mission run di air
- obstacle terdeteksi jelas
- command avoidance benar-benar mempengaruhi gerak kapal
- release terjadi
- `REJOIN` terjadi
- mission lanjut dan/atau selesai
- evidence video + rosbag + catatan operator

**Batas klaim**:
- tanpa evidence lapangan terkontrol yang lengkap, klaim ini belum boleh diambil.

---

### Claim E — Parameter visual baseline sudah cukup kuat sebagai baseline design-level

**Status**: `CLAIM_WITH_LIMITATION`

**Evidence minimum**:
- hasil perhitungan matematis internal
- nilai yang sudah masuk ke `risk_evaluator_node.py`
- `docs/BASELINE_PARAMETER_LOCK.md`

**Batas klaim**:
- parameter visual dapat disebut sebagai **design baseline**,
  tetapi tetap perlu dibedakan dari parameter runtime final lapangan.

---

### Claim F — Sistem memiliki lapisan keselamatan berjenjang

**Status**: `READY_TO_CLAIM`

**Evidence minimum**:
- `risk_evaluator_node.py`
- `watchdog_failsafe_node.py`
- `actuator_safety_limiter_node.py`
- `mission_mode_manager_node.py`
- `docs/ARCHITECTURE.md`

**Batas klaim**:
- ini adalah klaim tentang **mekanisme safety architecture**,
  bukan jaminan semua skenario lapangan sudah tertangani.

---

## 3. Klaim yang aman dipakai di laporan / sidang saat ini

Kalimat yang aman:
- sistem collision avoidance mission-aware telah diimplementasikan pada ROS 2 untuk USV differential-thrust,
- baseline simulasi telah menunjukkan transisi avoid dan rejoin secara repeatable,
- baseline hardware telah memvalidasi integrasi perception, decision, watchdog, RC override, dan monitoring,
- evidence saat ini sudah kuat untuk menyatakan kesiapan menuju uji lapangan terkontrol.

Kalimat yang harus hati-hati:
- sistem telah terbukti berhasil penuh di lapangan,
- mission complete dengan avoidance sudah konsisten di air,
- rejoin lapangan sudah tervalidasi final.

---

## 4. Klaim yang sebaiknya selalu diberi batasan

### Tentang simulasi
Gunakan batasan seperti:
- “pada baseline simulasi”,
- “dalam SITL”,
- “sebagai evidence kuantitatif awal”.

### Tentang hardware bench
Gunakan batasan seperti:
- “pada bench hardware”,
- “sebagai validasi integrasi”,
- “belum merepresentasikan pembuktian lapangan penuh”.

### Tentang parameter
Gunakan batasan seperti:
- “baseline design-level”,
- “sementara untuk runtime aktif”,
- “masih dapat dituning pada validasi lapangan terkontrol”.

---

## 5. Aturan praktis saat menulis hasil

Jika evidence berasal dari simulasi:
- jangan tulis seolah-olah itu evidence lapangan.

Jika evidence berasal dari hardware bench:
- jangan tulis seolah-olah itu already field success.

Jika evidence masih berupa desain matematis:
- jangan tulis seolah-olah itu sudah tervalidasi final oleh pengujian kapal.

---

## 6. Ringkasan akhir

Matrix ini menegaskan bahwa repo SEANO saat ini sudah cukup kuat untuk mengklaim:
- implementasi arsitektur,
- keberhasilan baseline simulasi,
- keberhasilan integrasi hardware bench,
- dan kesiapan menuju pengujian lapangan terkontrol.

Namun repo ini belum boleh dipakai untuk mengklaim:
- keberhasilan penuh end-to-end lapangan secara final dan berulang,
- atau validasi final semua parameter operasi nyata.
