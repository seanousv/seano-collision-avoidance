# HARDWARE BENCH RESULTS
## SEANO Collision Avoidance — Jetson + CUAV X7+ + USB Camera

Dokumen ini merangkum status **hardware bench baseline** untuk proyek collision avoidance SEANO.

Fokus dokumen ini bukan hasil simulasi SITL, melainkan hasil integrasi nyata pada:

- Jetson sebagai runtime utama
- CUAV X7+ sebagai flight controller / autopilot
- USB camera sebagai source persepsi
- MAVROS sebagai bridge ROS 2 <-> MAVLink
- browser monitoring untuk raw / annotated / HUD

Dokumen ini dipakai untuk:
- mencatat baseline hardware yang sudah tervalidasi,
- membedakan hasil hardware dari Phase 6 simulation evidence,
- menjadi referensi sebelum dockside test dan uji air,
- membantu penyusunan laporan TA, seminar, dan jurnal.

---

## 1. Scope Dokumen

Dokumen ini hanya membahas **hardware bench baseline**.

Yang termasuk:
- koneksi Jetson <-> CUAV X7+
- kamera USB
- detector
- risk evaluator
- watchdog
- browser monitoring
- debug HUD
- launch hardware utama `phase7_cuav_usb_hardware.launch.py`

Yang tidak termasuk:
- hasil Phase 6 simulation metrics
- hasil field-test final di air
- pembuktian mission complete end-to-end di lapangan secara final

---

## 2. Hardware Baseline Aktif

### Runtime utama
- Jetson sebagai komputer onboard
- ROS 2 Humble
- package utama: `seano_vision`

### Autopilot
- CUAV X7+
- koneksi serial tervalidasi melalui:
  - `/dev/ttyACM0`
- baud yang dipakai:
  - `115200`

### Camera source
- USB camera (UVC)
- source image utama untuk hardware

### Monitoring
- `web_video_server`
- browser-based monitoring dari laptop/operator

### Launch utama hardware
- `phase7_cuav_usb_hardware.launch.py`

---

## 3. Tujuan Hardware Bench

Target dari hardware bench baseline adalah memastikan rantai berikut bisa berjalan:

```text
USB camera
-> detector
-> risk evaluator
-> watchdog / failsafe
-> command chain
-> MAVROS
-> CUAV X7+
-> monitoring browser / HUD
````

Tahap hardware bench ini dipakai sebagai jembatan menuju:

* dockside validation,
* uji AUTO tanpa obstacle,
* uji obstacle terkontrol,
* pembuktian avoidance + rejoin di air.

---

## 4. Status Validasi Hardware yang Sudah Tercapai

## 4.1 Koneksi autopilot nyata

Status:

* tervalidasi

Bukti minimum:

* MAVROS dapat terhubung ke CUAV X7+
* `/mavros/state` terbaca
* status FCU dapat dipantau dari ROS 2

Makna:

* layer autopilot hardware sudah aktif,
* integrasi Jetson <-> FCU bukan lagi asumsi simulasi.

---

## 4.2 Input RC dan motor dasar

Status:

* tervalidasi parsial / bench-level

Bukti minimum:

* input RC terbaca
* motor dapat merespons pada pengujian dasar bench

Makna:

* sisi dasar aktuasi hardware sudah hidup,
* command path ke autopilot tidak lagi murni teoretis.

Catatan:

* ini belum otomatis berarti collision avoidance end-to-end di air sudah final.

---

## 4.3 USB camera

Status:

* tervalidasi

Bukti minimum:

* kamera terdeteksi sebagai device UVC
* source kamera dapat dibuka
* raw image dapat dipublish
* stream dapat dimonitor

Informasi praktis yang pernah tervalidasi:

* image source tersedia pada device video utama
* topic raw utama dipakai sebagai basis monitoring dan perception

Makna:

* hardware baseline sudah punya source persepsi nyata,
* tidak lagi hanya synthetic camera seperti pada simulasi.

---

## 4.4 Detector

Status:

* tervalidasi berjalan di Jetson

Bukti minimum:

* node detector start
* model berhasil di-load
* annotated image dapat dipublish
* detections keluar

Makna:

* perception chain pada hardware sudah hidup,
* object detection tidak lagi hanya terbukti di simulasi.

Catatan:

* kualitas hasil sangat bergantung pada pencahayaan, jarak obstacle, dan performa runtime.

---

## 4.5 Risk evaluator

Status:

* tervalidasi

Bukti minimum:

* `/ca/risk` terbit
* command collision avoidance dapat muncul
* debug HUD dapat menampilkan state keputusan

Makna:

* sistem bukan hanya “melihat object”,
* tetapi sudah melakukan penilaian risiko dan menerjemahkannya menjadi keputusan operasional.

---

## 4.6 Watchdog / failsafe

Status:

* tervalidasi aktif

Bukti minimum:

* watchdog node berjalan
* status watchdog muncul
* failsafe topic aktif saat dibutuhkan

Makna:

* baseline hardware sudah memiliki lapisan keselamatan jika perception dianggap tidak sehat.

Catatan:

* pada bench tertentu, false trigger masih bisa muncul bila stream image tidak stabil.

---

## 4.7 Browser monitoring

Status:

* tervalidasi

Bukti minimum:

* `web_video_server` berjalan
* raw camera dapat dibuka di browser
* annotated image dapat dibuka di browser
* `/ca/debug_image` dapat dibuka di browser

Makna:

* operator tidak perlu bergantung pada GUI lokal Jetson,
* monitoring bisa dilakukan lebih aman dari laptop melalui browser.

---

## 5. Topic Hardware yang Menjadi Patokan

Topic penting untuk hardware baseline:

### Image / monitoring

* `/seano/camera/image_raw_reliable`
* `/camera/image_annotated`
* `/ca/debug_image`

### Perception / decision

* `/camera/detections`
* `/ca/risk`
* `/ca/command`

### Safety

* `/ca/watchdog_status`
* `/ca/failsafe_active`

### Autopilot / MAVROS

* `/mavros/state`
* `/mavros/rc/override`

Topic-topic ini menjadi patokan utama untuk menyatakan bench hardware berjalan sehat.

---

## 6. Launch Hardware Resmi

Launch utama hardware baseline adalah:

```text
phase7_cuav_usb_hardware.launch.py
```

Fungsi launch ini:

* mengorkestrasi MAVROS,
* source kamera,
* detector,
* risk evaluator,
* watchdog failsafe,
* command mux,
* safety limiter,
* RC override bridge,
* mission/mode manager,
* dan jalur pengendalian hardware lainnya.

Kesimpulan operasional:

* untuk bench hardware penuh dan uji air, `phase7` adalah launch utama.

---

## 7. Jalur Monitoring Resmi

Untuk hardware baseline, monitoring resmi dilakukan melalui browser.

Objek monitoring utama:

1. raw camera stream
2. annotated detection stream
3. collision avoidance HUD

Manfaat monitoring browser:

* memudahkan operator memeriksa apa yang dilihat kamera,
* memudahkan diagnosis ketika deteksi tidak muncul,
* memudahkan observasi perubahan risk dan command,
* lebih praktis daripada bergantung pada viewer GUI langsung di Jetson.

---

## 8. Interpretasi Status Hardware Saat Ini

Status hardware saat ini **belum boleh dibaca** sebagai:

* “field-test final sudah berhasil penuh”
* atau “mission complete dengan avoid + rejoin sudah terbukti final di air”

Status yang lebih tepat adalah:

### Yang sudah bisa diklaim

1. Jetson runtime aktif
2. CUAV X7+ terhubung melalui MAVROS
3. USB camera tervalidasi
4. detector berjalan
5. risk evaluator berjalan
6. watchdog aktif
7. HUD/debug monitoring aktif
8. browser monitoring aktif
9. launch hardware terpadu tersedia melalui `phase7`

### Yang belum boleh diklaim final

1. obstacle avoidance end-to-end di air sudah final
2. release + rejoin di air sudah konsisten
3. mission complete dengan avoidance sudah tervalidasi berulang di lapangan
4. field-test penuh sudah matang di semua kondisi

---

## 9. Keterbatasan Hardware Bench Saat Ini

Keterbatasan yang masih relevan:

### 9.1 Perception sensitivity

Kinerja detector dan risk di hardware sangat dipengaruhi oleh:

* cahaya
* kontras obstacle
* jarak obstacle
* sudut kamera
* kestabilan stream image

### 9.2 Watchdog false trigger risk

Jika image stream:

* stale,
* putus,
* atau tidak konsisten,

maka watchdog dapat memicu status yang terlihat seperti `LOST_PERCEPTION` atau kondisi aman berhenti.

### 9.3 Belum setara dengan field success

Bench hardware yang sehat tidak otomatis berarti:

* kapal sudah terbukti avoid di air,
* kapal sudah terbukti rejoin,
* mission sudah selesai penuh.

Hardware bench adalah tahap persiapan penting, bukan akhir validasi.

---

## 10. Target Uji Berikutnya

Urutan target setelah hardware bench:

### Tahap 1 — dockside validation

Target:

* semua node utama aktif,
* browser monitoring aktif,
* raw / annotated / HUD stabil,
* MAVROS connected,
* obstacle sederhana bisa terlihat kamera.

### Tahap 2 — AUTO tanpa obstacle

Target:

* kapal mengikuti waypoint normal,
* baseline autopilot tidak terganggu,
* system health tetap stabil.

### Tahap 3 — obstacle terkontrol

Target:

* obstacle terlihat,
* detector bereaksi,
* risk naik,
* command berubah,
* kapal mulai keluar jalur.

### Tahap 4 — release + rejoin

Target:

* obstacle clear,
* takeover dilepas,
* masuk `REJOIN`,
* kapal kembali ke mission.

### Tahap 5 — mission complete

Target:

```text
AUTO mission -> obstacle detected -> avoid -> safe -> rejoin -> continue mission -> finish
```

Ini adalah target akhir pembuktian collision avoidance nyata.

---

## 11. Definisi Keberhasilan Hardware

### 11.1 Berhasil bench

Bench hardware dianggap berhasil jika:

* FCU connect,
* camera source stabil,
* detector publish,
* risk publish,
* watchdog hidup,
* HUD tampil,
* browser monitoring bisa dipakai.

### 11.2 Berhasil parsial lapangan

Lapangan dianggap berhasil parsial jika:

* kapal mengikuti mission dasar,
* obstacle terdeteksi,
* sistem menunjukkan respons aman,
* walaupun avoid atau rejoin belum sempurna.

### 11.3 Berhasil penuh lapangan

Lapangan dianggap berhasil penuh jika:

* kapal mengikuti waypoint,
* obstacle terdeteksi,
* kapal melakukan avoid,
* obstacle clear,
* kapal rejoin,
* mission dilanjutkan sampai waypoint berikutnya / selesai.

---

## 12. Evidence yang Harus Dikumpulkan

Untuk setiap run hardware, evidence yang disarankan:

### Visual evidence

* video raw camera
* video annotated image
* screenshot `/ca/debug_image`
* dokumentasi posisi obstacle

### Runtime evidence

* log terminal
* rosbag (jika direkam)
* topic state / risk / command

### Catatan operator

* mode awal
* obstacle dipasang atau tidak
* respons kapal
* apakah terjadi avoid
* apakah terjadi release
* apakah terjadi rejoin
* apakah mission lanjut

---

## 13. Rekomendasi Pemakaian Dokumen Ini

Dokumen ini dipakai bersama:

* `docs/LAUNCH_STATUS_MAP.md`
* `docs/RUNBOOK.md`
* `docs/ARCHITECTURE.md`

Fungsi masing-masing:

* `LAUNCH_STATUS_MAP.md` -> memilih launch yang tepat
* `RUNBOOK.md` -> langkah operasional menjalankan sistem
* `ARCHITECTURE.md` -> memahami struktur runtime
* `HARDWARE_BENCH_RESULTS.md` -> memahami status hasil hardware saat ini

---

## 14. Ringkasan Akhir

Status hardware proyek SEANO saat ini paling tepat diringkas sebagai:

* hardware bench baseline sudah aktif,
* perception chain pada hardware sudah hidup,
* FCU nyata sudah terhubung,
* monitoring browser sudah sehat,
* launch utama hardware sudah tersedia,
* tetapi pembuktian collision avoidance end-to-end di air tetap harus dilakukan bertahap dan aman.

Dengan kata lain:

**proyek ini sudah melewati proof-of-concept simulasi, masuk ke bench hardware fungsional, dan sedang menuju pembuktian field collision avoidance yang terkontrol.**

---

## 14.1 Dokumen pendamping yang harus dibaca bersama

Untuk membaca status hardware bench dengan benar, gunakan dokumen ini bersama:

- `docs/THESIS_PROGRESS_STATUS.md`
- `docs/BASELINE_PARAMETER_LOCK.md`
- `docs/VALIDATION_BOUNDARY.md`

Tujuannya agar hasil hardware bench tidak:
- diremehkan sebagai sekadar demo lokal,
- maupun dilebihkan seolah-olah semua validasi lapangan sudah final.
