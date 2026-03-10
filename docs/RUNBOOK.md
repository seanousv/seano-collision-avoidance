# RUNBOOK — SEANO Collision Avoidance
## Operasional Harian untuk Simulation Baseline dan Hardware Baseline

Dokumen ini adalah runbook operasional utama untuk repository **SEANO Collision Avoidance**.

Tujuan dokumen ini:
- memberi urutan startup yang jelas,
- memisahkan jalur **simulasi** dan **hardware nyata**,
- mengurangi trial-error saat pengujian,
- menjaga operator fokus ke launch yang memang aktif dipakai.

Runbook ini mengikuti baseline aktif repository saat ini:
- **Simulation baseline** → `phase5_mission_avoid_integration.launch.py`
- **Hardware baseline** → `phase7_cuav_usb_hardware.launch.py`

---

# 1. Prinsip Umum

## 1.1 Dua baseline aktif
Repository ini saat ini punya dua baseline aktif yang harus dibedakan dengan tegas:

### A. Simulation baseline
Dipakai untuk:
- SITL mission execution,
- validasi state machine `MISSION -> AVOID -> REJOIN -> MISSION`,
- rosbag recording,
- Phase 6 metrics extraction.

Launch utama:
- `phase5_mission_avoid_integration.launch.py`

### B. Hardware baseline
Dipakai untuk:
- Jetson + CUAV X7+ + USB camera,
- detector -> risk -> watchdog runtime,
- RC override bridge,
- command mux,
- actuator safety limiter,
- monitoring browser,
- dockside test dan uji air.

Launch utama:
- `phase7_cuav_usb_hardware.launch.py`

---

## 1.2 Rule of thumb
Gunakan aturan cepat berikut:

- Jika targetnya **validasi logika mission/avoid/rejoin secara repeatable**, pakai **simulation baseline**.
- Jika targetnya **uji bench hardware atau uji kapal nyata**, pakai **hardware baseline**.
- Jika targetnya **cek modul tertentu saja**, pakai launch bench/debug yang lebih ringan dulu.

---

## 1.3 Launch bench/debug yang tetap aktif
Launch berikut tetap berguna untuk precheck cepat:

- `phase2_camera_usb_test.launch.py`
  - cek kamera USB / source kamera by-id
- `demo_detect.launch.py`
  - cek jalur `camera -> detector`
- `demo_risk.launch.py`
  - cek jalur `camera -> detector -> risk`
- `demo_full_ca.launch.py`
  - cek pipeline CA lengkap untuk bench/debug non-hardware penuh

---

# 2. Workspace dan Build

## 2.1 Lokasi workspace
Contoh asumsi lokasi repo:
```bash
~/seano-collision-avoidance
````

Workspace ROS 2:

```bash
~/seano-collision-avoidance/seano_ca_ws
```

## 2.2 Build package utama

Jalankan setiap kali ada perubahan pada package `seano_vision`:

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select seano_vision --symlink-install
source install/setup.bash
```

## 2.3 Expected

Expected minimum:

* build selesai tanpa error,
* package `seano_vision` berhasil terpasang,
* tidak ada traceback Python dari launch/node aktif.

---

# 3. Simulation Baseline

## SITL + Mission Planner + MAVROS + Phase 5

Simulation baseline dipakai untuk pengujian end-to-end yang paling repeatable sebelum turun ke hardware nyata.

---

## 3.1 Tujuan utama simulasi

Gunakan simulasi untuk:

* validasi mode `MISSION`,
* validasi trigger `AVOID`,
* validasi pelepasan takeover,
* validasi `REJOIN`,
* validasi kembali ke jalur mission,
* rosbag evidence untuk analisis Phase 6.

---

## 3.2 Port map simulasi

Baseline port map:

* Mission Planner (Windows): `14550/UDP`
* MAVROS <-> SITL (WSL/Linux): `14551/UDP`
* MAVProxy / ArduPilot master: biasanya `5760/TCP`

Jika menggunakan WSL2, IP host Windows bisa berubah tiap sesi.
Ambil IP host Windows dari default gateway:

```bash
ip route | awk '/default/ {print $3; exit}'
```

---

## 3.3 Startup order simulasi

Urutan startup **wajib**:

1. ArduPilot SITL
2. Mission Planner
3. MAVROS
4. Launch `phase5_mission_avoid_integration.launch.py`

Kalau SITL restart:

* restart MAVROS,
* lalu restart stack SEANO.

---

## 3.4 Terminal 1 — Start SITL

Contoh:

```bash
cd ~/tools/ardupilot
WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
echo "WIN_HOST_IP=$WIN_HOST_IP"

sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
```

### Expected

* MAVProxy console muncul,
* map SITL muncul,
* data MAVLink keluar ke:

  * Mission Planner (`14550`)
  * MAVROS (`14551`)

---

## 3.5 Mission Planner — Connect

Di Mission Planner:

* pilih **UDP**
* port `14550`
* klik **Connect**

### Expected

* vehicle muncul,
* mode vehicle terbaca,
* waypoint mission bisa di-load.

---

## 3.6 Terminal 2 — Start MAVROS

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

### Verifikasi cepat

```bash
ros2 topic echo /mavros/state -n 1
```

### Expected

Minimal:

* `connected: true`
* mode terbaca
* armed status terbaca

---

## 3.7 Terminal 3 — Start simulation baseline utama

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py
```

### Peran launch ini

Launch ini dipakai untuk:

* mission-following,
* temporary avoidance takeover,
* release takeover,
* `REJOIN`,
* kembali ke mission.

---

## 3.8 Monitoring minimum simulasi

Buka terminal monitor tambahan sesuai kebutuhan.

### State machine

```bash
ros2 topic echo /ca/mode_manager_state
```

### Event state machine

```bash
ros2 topic echo /ca/mode_manager_event
```

### Status flight controller

```bash
ros2 topic echo /mavros/state
```

### Command collision avoidance

```bash
ros2 topic echo /ca/command
```

### Risk

```bash
ros2 topic echo /ca/risk
```

---

## 3.9 Kriteria lulus simulasi

Simulation baseline dianggap sehat jika urutan berikut bisa terjadi:

1. vehicle menjalankan mission waypoint,
2. trigger obstacle / hazard memaksa sistem masuk `AVOID`,
3. vehicle keluar dari jalur nominal,
4. saat hazard hilang sistem melepas takeover,
5. state masuk `REJOIN`,
6. vehicle kembali ke mission-following,
7. mission lanjut sampai waypoint berikutnya.

Target state:

```text
MISSION -> AVOID -> REJOIN -> MISSION
```

---

## 3.10 Jika simulasi tidak sehat

Gunakan aturan cepat berikut:

### Jika MAVROS tidak connect

Cek:

```bash
ros2 topic echo /mavros/state -n 1
```

Jika belum connect:

* cek SITL masih hidup,
* cek port `14551`,
* restart MAVROS.

### Jika state machine tidak kembali ke mission

Cek:

* event `REJOIN_*`,
* mode autopilot,
* apakah publisher takeover masih aktif,
* apakah ada failsafe yang masih menahan output.

### Jika perception pipeline mengganggu pengujian logika mission

Fokuskan dulu pengujian ke mode/state manager,
lalu naik bertahap ke full integration.

---

# 4. Hardware Baseline

## Jetson + CUAV X7+ + USB camera + Phase 7

Hardware baseline adalah jalur utama untuk bench hardware, dockside test, dan uji air.

---

## 4.1 Tujuan utama hardware baseline

Gunakan baseline ini untuk target nyata berikut:

1. autopilot berada di `AUTO` dan mengikuti waypoint,
2. kamera membaca kondisi depan kapal,
3. detector menghasilkan detections,
4. risk evaluator menghasilkan keputusan collision avoidance,
5. watchdog mengawasi kualitas persepsi,
6. command mux mengatur sumber command,
7. RC override bridge menerjemahkan command ke kontrol FC,
8. saat obstacle muncul kapal keluar jalur,
9. saat aman kapal kembali ke jalur mission,
10. mission berlanjut sampai selesai.

---

## 4.2 Startup order hardware

Urutan startup **wajib** untuk hardware nyata:

1. pastikan wiring dan power stabil,
2. nyalakan FC / CUAV X7+,
3. pastikan kamera USB terdeteksi,
4. start `phase7_cuav_usb_hardware.launch.py`,
5. start browser monitoring,
6. lakukan precheck topic dan status,
7. baru lanjut ke dockside test,
8. baru lanjut ke uji air.

---

## 4.3 Precheck sebelum launch hardware

Sebelum menjalankan `phase7`, cek perangkat inti.

### Cek device serial FC

```bash
ls /dev/ttyACM*
```

### Cek kamera USB

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id
```

### Cek package sudah ter-build

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 pkg prefix seano_vision
```

---

## 4.4 Jalankan hardware baseline utama

Contoh:

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200
```

Jika perlu, sesuaikan `fcu_url` dengan device aktual FC Anda.

---

## 4.5 Peran launch phase7

Launch ini adalah orkestrasi hardware utama yang dipakai untuk:

* camera source hardware,
* detector,
* risk evaluator,
* watchdog failsafe,
* command mux,
* actuator safety limiter,
* MAVROS,
* RC override bridge,
* mission mode manager,
* auto controller stub / control helper sesuai baseline saat ini.

Intinya:
**phase7 adalah satu run paling lengkap untuk target collision avoidance pada hardware nyata.**

---

## 4.6 Topic minimum yang harus hidup

Setelah `phase7` jalan, cek topic minimum ini.

### Image / monitoring

```bash
ros2 topic list | grep -E "image_raw_reliable|image_annotated|debug_image"
```

Expected minimal:

* `/seano/camera/image_raw_reliable`
* `/camera/image_annotated` atau topic annotated aktif yang setara
* `/ca/debug_image`

### Detections

```bash
ros2 topic echo /camera/detections
```

### Risk

```bash
ros2 topic echo /ca/risk
```

### Command

```bash
ros2 topic echo /ca/command
```

### MAVROS state

```bash
ros2 topic echo /mavros/state -n 1
```

### Watchdog status

```bash
ros2 topic echo /ca/watchdog_status
```

Jika topic status tidak muncul terus-menerus, cek nama topic aktual di runtime.

---

## 4.7 Browser monitoring

Monitoring browser sangat disarankan saat bench dan uji lapangan.

Jalankan `web_video_server`, lalu buka stream untuk:

* raw camera,
* annotated image,
* HUD / debug image.

Contoh pola URL:

```text
http://localhost:8081/stream?topic=/seano/camera/image_raw_reliable
http://localhost:8081/stream?topic=/camera/image_annotated
http://localhost:8081/stream?topic=/ca/debug_image
http://localhost:8081/snapshot?topic=/ca/debug_image
```

Catatan:

* port forwarding editor/SSH bisa membuat `8080` di remote terlihat sebagai `8081` di lokal,
* selalu lihat panel **Ports** untuk alamat final yang benar.

---

## 4.8 Bench test bertahap sebelum ke air

Jangan langsung loncat ke uji air penuh.
Gunakan urutan ini.

### Tahap 1 — kamera saja

```bash
ros2 launch seano_vision phase2_camera_usb_test.launch.py
```

Lulus jika:

* raw image stabil,
* FPS masuk akal,
* tidak ada reconnect loop abnormal.

### Tahap 2 — detector

```bash
ros2 launch seano_vision demo_detect.launch.py
```

Lulus jika:

* raw image masuk,
* annotated image keluar,
* detections keluar.

### Tahap 3 — risk

```bash
ros2 launch seano_vision demo_risk.launch.py
```

Lulus jika:

* detections terbaca,
* `/ca/risk` dan `/ca/command` keluar,
* HUD/debug image tampil.

### Tahap 4 — full phase7 bench

```bash
ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200
```

Lulus jika:

* MAVROS connect,
* image/detection/risk/command aktif,
* watchdog tidak salah-trigger saat kondisi sehat,
* browser monitoring bisa dipakai.

---

## 4.9 Dockside test checklist

Sebelum kapal masuk air, minimal lolos checklist berikut:

### FC / autopilot

* FC connect stabil
* mode terbaca
* GPS / heading / basic navigation sehat
* waypoint mission sudah di-load

### Camera / vision

* raw image tampil
* annotated image tampil
* HUD/debug image tampil
* detection ada saat obstacle muncul

### CA control chain

* risk aktif
* command aktif
* mux aktif
* safety limiter aktif
* RC override bridge aktif

### Operator readiness

* operator tahu cara pindah MANUAL / AUTO
* operator tahu cara abort
* operator tahu topic/monitor yang dilihat
* operator tahu kondisi kapan test dihentikan

---

## 4.10 Uji air — urutan paling aman

Dengan waktu lapangan yang terbatas, pakai urutan ini.

### Run A — mission tanpa obstacle

Tujuan:

* validasi kapal bisa mengikuti waypoint normal di `AUTO`
* memastikan baseline autopilot dan integrasi phase7 tidak merusak mission dasar

Lulus jika:

* kapal mengikuti mission,
* tidak ada trigger avoid palsu,
* tidak ada STOP/failsafe abnormal.

### Run B — obstacle statis sederhana

Tujuan:

* validasi obstacle terdeteksi,
* validasi `AVOID` aktif,
* validasi kapal keluar dari jalur nominal

Lulus jika:

* obstacle muncul di kamera,
* HUD menunjukkan risk/command masuk akal,
* kapal benar-benar mengubah jalur.

### Run C — obstacle clear / release

Tujuan:

* validasi takeover dilepas,
* validasi masuk `REJOIN`,
* validasi kapal kembali ke mission

Lulus jika:

* hazard hilang,
* command avoid berhenti,
* kapal kembali ke jalur mission dan lanjut waypoint berikutnya.

### Run D — mission lengkap

Tujuan:

* validasi end-to-end:

```text
AUTO mission -> obstacle detected -> avoid -> safe -> rejoin -> continue mission -> finish
```

Ini adalah run yang paling penting untuk membuktikan sistem collision avoidance berhasil.

---

## 4.11 Kriteria lulus hardware/uji air

Target minimal keberhasilan sistem adalah:

1. kapal mulai di `AUTO` dan mengikuti waypoint,
2. obstacle terlihat di kamera,
3. detector dan risk merespons,
4. command avoidance keluar,
5. kapal keluar dari jalur untuk menghindar,
6. setelah obstacle aman/tidak terlihat, takeover dilepas,
7. sistem masuk `REJOIN`,
8. kapal kembali ke mission path,
9. kapal lanjut ke waypoint berikutnya sampai misi selesai.

Kalau salah satu rantai di atas putus,
maka uji belum boleh disebut **collision avoidance end-to-end berhasil**.

---

# 5. Quick Commands

## Command ringkas yang paling sering dipakai

## 5.1 Build

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select seano_vision --symlink-install
source install/setup.bash
```

## 5.2 Simulation baseline

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py
```

## 5.3 Hardware baseline

```bash
ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200
```

## 5.4 Bench detect

```bash
ros2 launch seano_vision demo_detect.launch.py
```

## 5.5 Bench risk

```bash
ros2 launch seano_vision demo_risk.launch.py
```

## 5.6 Camera precheck

```bash
ros2 launch seano_vision phase2_camera_usb_test.launch.py
```

## 5.7 Browser monitoring helper

```bash
ros2 run web_video_server web_video_server
```

---

# 6. Hal yang Jangan Dilakukan

* Jangan pakai launch audit-later sebagai baseline utama untuk uji kapal.
* Jangan rename launch aktif menjelang pengujian lapangan.
* Jangan langsung uji air jika bench camera/detector/risk belum lulus.
* Jangan menyimpulkan collision avoidance berhasil hanya dari bench image; bukti utamanya adalah perilaku kapal:

  * keluar jalur saat obstacle,
  * kembali ke mission saat aman.

---

# 7. Rekomendasi Operasional untuk Hari Uji yang Singkat

Kalau waktu pengujian hanya satu hari, pakai strategi ini:

1. pagi:

   * bench camera
   * bench detector
   * bench risk
   * phase7 full bench
2. siang awal:

   * dockside AUTO tanpa obstacle
3. siang:

   * AUTO dengan obstacle statis sederhana
4. sore:

   * validasi release + rejoin + lanjut mission
5. akhir:

   * simpan evidence:

     * video monitoring
     * screenshot HUD
     * rosbag jika tersedia
     * catatan hasil tiap run

Fokus utama:
**jangan mengejar terlalu banyak skenario.**
Kejar dulu satu bukti kuat:

```text
mission -> avoid -> rejoin -> mission complete
```

---

# 8. Dokumen Pendamping

Baca dokumen ini bersama:

* `docs/LAUNCH_STATUS_MAP.md`
* `docs/ARCHITECTURE.md`
* `docs/PHASE6_TEST_MATRIX.md`
* `docs/PHASE6_RESULTS_SUMMARY.md`

---

# 9. Tujuan Dokumen Ini

Dokumen ini dibuat agar operator/pengembang bisa:

* cepat memilih jalur run yang benar,
* tidak mencampur simulasi dan hardware,
* tidak salah memilih launch utama,
* fokus ke target akhir proyek:

**USV mengikuti waypoint mission, menghindari obstacle saat berbahaya, lalu kembali ke jalur mission setelah aman sampai misi selesai.**
