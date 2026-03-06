# Runbook (WSL2 Simulation) — Phase 5 / Phase 6
## SITL + Mission Planner + MAVROS + SEANO Collision Avoidance

Runbook ini adalah prosedur startup dan pengujian yang repeatable untuk stack terbaru:

- ArduPilot SITL (WSL)
- Mission Planner (Windows)
- MAVROS2 (WSL)
- SEANO ROS 2 stack (WSL)
- Phase 5 mission / avoid / rejoin integration
- Phase 6 rosbag metrics extraction

Dokumen ini menggantikan jalur lama `run_auto_stack.launch.py` sebagai referensi utama untuk pengujian integrasi terbaru.

---

## 1. Baseline Port Map

- Mission Planner (Windows): `14550/UDP`
- MAVROS <-> SITL (WSL): `14551/UDP`
- MAVProxy / ArduPilot master (internal): `5760/TCP` (umumnya)

Catatan WSL2:
- IP Windows host bisa berubah setiap sesi.
- Ambil IP Windows host dari default gateway WSL:
  ```bash
  ip route | awk '/default/ {print $3; exit}'
````

---

## 2. Startup Order (Wajib)

Komponen dijalankan berurutan:

1. ArduPilot SITL (WSL)
2. Mission Planner (Windows)
3. MAVROS2 (WSL)
4. SEANO ROS 2 stack (WSL)

Kalau SITL di-restart:

* restart MAVROS,
* lalu restart stack SEANO.

---

## 3. Build Workspace

Jalankan ini setiap kali ada perubahan file:

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select seano_vision --symlink-install
source install/setup.bash
```

Expected:

* build selesai tanpa error,
* package `seano_vision` terpasang,
* tidak ada traceback Python dari launch/node terbaru.

---

## 4. Terminal 1 (WSL) — Start ArduPilot SITL

```bash
cd ~/tools/ardupilot
WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
echo "WIN_HOST_IP=$WIN_HOST_IP"

sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
```

Expected:

* MAVProxy console terbuka,
* map SITL terbuka,
* SITL mengirim MAVLink ke:

  * Mission Planner (Windows) `WIN_HOST_IP:14550`
  * MAVROS (WSL) `127.0.0.1:14551`

---

## 5. Mission Planner (Windows) — Connect

* Connection type: `UDP`
* Port: `14550`
* Klik **Connect**

Expected:

* vehicle terlihat,
* map menampilkan rover,
* mode vehicle terbaca.

---

## 6. Terminal 2 (WSL) — Start MAVROS2

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

Expected:

* `/mavros/state` menjadi `connected: true`

Verifikasi cepat:

```bash
ros2 topic echo /mavros/state -n 1
```

Expected minimal:

* `connected: true`
* `mode` terbaca
* `armed` sesuai kondisi

---

## 7. Phase 5 Runtime Modes

Launch utama:

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py ...
```

Mode pengujian dibagi menjadi 3 kasus utama:

* **Kasus A** — mode manager only
* **Kasus B** — takeover logic only
* **Kasus C** — full integration (synthetic camera)

---

## 8. Kasus A — Mode Manager Only

Tujuan:

* validasi state machine:

  * `MISSION -> AVOID -> REJOIN -> MISSION`
* tanpa gangguan perception pipeline
* tanpa bentrok publisher takeover manager

### Jalankan

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=false \
  use_takeover_manager:=false
```

### Monitor

Terminal monitor 1:

```bash
ros2 topic echo /ca/mode_manager_state
```

Terminal monitor 2:

```bash
ros2 topic echo /ca/mode_manager_event
```

Terminal monitor 3:

```bash
ros2 topic echo /mavros/state
```

### Trigger manual takeover

```bash
ros2 topic pub --once /seano/rc_override_enable std_msgs/msg/Bool "{data: true}"
sleep 3
ros2 topic pub --once /seano/rc_override_enable std_msgs/msg/Bool "{data: false}"
```

### Expected

State:

* awal: `MISSION`
* publish `true`: `AVOID`
* publish `false`: `REJOIN`
* beberapa saat kemudian: `MISSION`

Event minimum yang diharapkan:

* `TAKEOVER_ON`
* `MODE_REQ_SENT`
* `MODE_REQ_DONE`
* `TAKEOVER_OFF`
* `REJOIN_START`
* `REJOIN_MODE_MATCH`
* `REJOIN_DONE`

Yang tidak boleh terjadi:

* `FAILSAFE_ON/OFF` bolak-balik sendiri

---

## 9. Kasus B — Takeover Logic Only

Tujuan:

* validasi `auto_controller_stub_node`
* validasi jalur command hazard -> takeover -> release
* tanpa noise dari perception / watchdog

### Jalankan

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=false \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

### Monitor

```bash
ros2 topic echo /ca/mode_manager_state
ros2 topic echo /ca/mode_manager_event
ros2 topic echo /mavros/state
```

### Trigger command hazard

```bash
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_RIGHT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"
```

Opsional tambahan:

```bash
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_LEFT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"
```

### Expected

* `TURN_RIGHT` / `TURN_LEFT` memicu takeover
* `/seano/rc_override_enable` menjadi `true`
* state menjadi `AVOID`
* saat `HOLD_COURSE`, takeover release
* state menjadi `REJOIN`, lalu `MISSION`

---

## 10. Kasus C — Full Integration (Synthetic Camera)

Tujuan:

* validasi stack integrasi penuh
* tetap ringan di WSL
* tidak bergantung pada kamera hardware

### Default yang dipakai sekarang

Mode default:

* `ca_runtime_profile:=synthetic_light`

Artinya:

* source kamera: synthetic / dummy camera
* detector aktif
* risk aktif
* watchdog / freeze / fusion / waterline / vision quality tidak dibebani penuh seperti mode full

### Jalankan (default ringan)

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

### Verifikasi dummy camera

```bash
ros2 topic info /seano/camera/image_raw_reliable
ros2 topic hz /seano/camera/image_raw_reliable
```

Expected:

* topic ada walaupun kamera hardware tidak dipasang
* frekuensi sekitar 10 Hz
* stack tetap berjalan ringan

### Mode synthetic + watchdog

Kalau ingin mulai uji failsafe sintetis:

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true \
  ca_runtime_profile:=synthetic_watchdog
```

### Mode full

Kalau ingin jalur penuh:

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true \
  ca_runtime_profile:=full
```

Catatan:

* `full` adalah mode paling berat
* gunakan setelah `synthetic_light` dan `synthetic_watchdog` stabil

---

## 11. Validation Commands

### 11.1 MAVROS connected

```bash
ros2 topic echo /mavros/state -n 1
```

### 11.2 Mode manager state

```bash
ros2 topic echo /ca/mode_manager_state
```

### 11.3 Mode manager event

```bash
ros2 topic echo /ca/mode_manager_event
```

### 11.4 RC override

```bash
ros2 topic echo /mavros/rc/override -n 5
```

### 11.5 Duplicate publisher check

```bash
ros2 topic info -v /seano/rc_override_enable
ros2 topic info -v /ca/command_safe
ros2 topic info -v /seano/auto_enable
```

---

## 12. Phase 6 — Standard Rosbag Test Scenarios

Tujuan:

* menghasilkan bag yang konsisten
* bisa dihitung dengan `phase6_metrics_from_bag.py`
* menghasilkan metrik:

  * takeover count
  * takeover duration
  * reaction time
  * release time
  * rejoin time
  * failsafe count
  * mode mismatch ratio

### Skenario Uji 1 — Hazard / Rejoin

Gunakan mode:

* Kasus B, atau
* Kasus C synthetic_light

#### Start record

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_rejoin_run_01 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

#### Trigger sequence

Terminal lain:

```bash
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_RIGHT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"

sleep 2

ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_LEFT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"
```

#### Stop record

* hentikan launch dengan `Ctrl+C`

#### Expected

* takeover segments >= 1
* reaction time terisi
* release time terisi
* rejoin time terisi
* `REJOIN_DONE` muncul
* mismatch ratio rendah / nol

---

### Skenario Uji 2 — Failsafe

Gunakan mode:

* `synthetic_watchdog`

#### Start record

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_failsafe_run_01 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true \
  ca_runtime_profile:=synthetic_watchdog
```

#### Trigger failsafe

Cara paling sederhana untuk uji di WSL:

* stop node camera source dari terminal launch,
* atau hentikan launch lalu catat bahwa bag ini khusus failsafe,
* atau gunakan metode internal lain yang memang memicu `failsafe_active`.

Jika testing dilakukan dengan mematikan image source, expected:

* `/ca/failsafe_active` menjadi `true`
* mode manager masuk `FAILSAFE`
* vehicle masuk mode aman
* event `FAILSAFE_ON` terekam

#### Expected

* `failsafe.rises >= 1`
* event `FAILSAFE_ON` muncul
* kalau clear, `FAILSAFE_OFF` muncul

---

### Skenario Uji 3 — Repeated Hazard / Rejoin

Tujuan:

* mengukur konsistensi

Gunakan mode:

* Kasus C default

Record 2–3 bag terpisah:

* `phase6_rejoin_run_02`
* `phase6_rejoin_run_03`

Lalu bandingkan:

* `reaction_time_s.mean`
* `release_time_s.mean`
* `rejoin_time_s.mean`

---

## 13. Extract Metrics from Rosbag

Jalankan extractor:

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_metrics_from_bag.py \
  --bag ~/bags/phase6_rejoin_run_01
```

Expected:

* summary tercetak di terminal
* file tersimpan:

  ```bash
  ~/bags/phase6_rejoin_run_01/phase6_metrics.json
  ```

### Buka hasil JSON

```bash
python3 -m json.tool ~/bags/phase6_rejoin_run_01/phase6_metrics.json
```

### Fokus lihat bagian ini

* `override`
* `reaction_time_s`
* `release_time_s`
* `rejoin`
* `rejoin_time_s`
* `failsafe`
* `mode_mismatch`

---

## 14. Interpretasi Cepat Metrik

### `override.takeover_segments`

Jumlah segmen takeover yang terjadi.

### `reaction_time_s.mean`

Rata-rata waktu dari hazard command ke override ON.

### `release_time_s.mean`

Rata-rata waktu dari clear command ke override OFF.

### `rejoin_time_s.mean`

Rata-rata waktu dari `REJOIN_START` ke `REJOIN_DONE`.

### `failsafe.rises`

Jumlah event failsafe aktif.

### `mode_mismatch.mismatch_ratio`

Rasio saat state `MISSION` tetapi mode MAVROS bukan `AUTO`.

Target yang baik:

* takeover segment terisi
* reaction / release / rejoin terisi
* failsafe muncul pada skenario failsafe
* mismatch ratio serendah mungkin

---

## 15. Common Failure Modes

### Mission Planner tidak connect

* cek `WIN_HOST_IP`
* cek `--out udp:${WIN_HOST_IP}:14550`

### MAVROS `connected: false`

* cek SITL kirim ke `127.0.0.1:14551`
* cek MAVROS memakai:

  ```bash
  udp://0.0.0.0:14551@127.0.0.1:14551
  ```

### State `MISSION -> FAILSAFE -> REJOIN -> MISSION` berulang sendiri

* kemungkinan watchdog / perception chain aktif
* gunakan:

  * Kasus A untuk test mode manager murni
  * Kasus B untuk test takeover logic murni
  * Kasus C default (`synthetic_light`) untuk integrasi ringan

### Publish manual takeover tidak bekerja

* cek apakah `auto_controller_stub_node` aktif
* kalau aktif, topic `/seano/rc_override_enable` bisa punya publisher lain
* untuk uji manual takeover, gunakan **Kasus A**

### Metrik `rejoin_time_s` kosong

* bag tidak memiliki `REJOIN_START -> REJOIN_DONE`
* atau skenario tidak benar-benar memicu rejoin penuh

### Metrik `failsafe` kosong

* bag tidak merekam `/ca/failsafe_active`
* atau run yang direkam memang tidak memicu failsafe

---

## 16. Output Minimum yang Harus Dianggap Valid

Satu run Phase 5 / Phase 6 dianggap valid bila:

* `/mavros/state connected: true`
* state manager menunjukkan:

  * `MISSION -> AVOID -> REJOIN -> MISSION`
* event manager menunjukkan:

  * `TAKEOVER_ON`
  * `TAKEOVER_OFF`
  * `REJOIN_START`
  * `REJOIN_DONE`
* extractor menghasilkan:

  * `takeover_segments >= 1`
  * `reaction_time_s.n >= 1`
  * `release_time_s.n >= 1`
  * `rejoin_time_s.n >= 1`

---

## 17. Rekomendasi Operasional

Urutan kerja yang direkomendasikan:

1. Kasus A — validasi state machine
2. Kasus B — validasi takeover logic
3. Kasus C synthetic_light — validasi integrasi ringan
4. Phase 6 rejoin bag
5. Phase 6 failsafe bag
6. Ulang 2–3 run untuk konsistensi
7. Baru naik ke kamera hardware / USB

Dokumen ini menjadi baseline utama untuk pengujian simulasi terbaru.

```
