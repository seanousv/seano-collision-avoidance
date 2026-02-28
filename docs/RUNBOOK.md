# Runbook (WSL2 Simulation) — SITL + Mission Planner + MAVROS + SEANO

Runbook ini berisi prosedur startup simulasi yang repeatable untuk stack: ArduPilot SITL (WSL) + Mission Planner (Windows) + MAVROS2 (WSL) + SEANO ROS 2.

## Baseline Port Map

- Mission Planner (Windows): `14550/UDP`
- MAVROS ↔ SITL (WSL): `14551/UDP`
- MAVProxy/ArduPilot master (internal): `5760/TCP` (umumnya)

Catatan WSL2:
- IP Windows host yang terlihat dari WSL bisa berubah setiap sesi.
- IP Windows host selalu diambil dari default gateway WSL: `ip route`.

---

## Startup Order (Wajib)

Komponen harus dijalankan berurutan:
1) ArduPilot SITL (WSL)
2) Mission Planner (Windows)
3) MAVROS2 (WSL)
4) SEANO ROS 2 stack (WSL)

Jika SITL di-restart, MAVROS harus di-restart.

---

## Terminal 1 (WSL) — Start ArduPilot SITL (ArduRover rover-skid)

```bash
cd ~/tools/ardupilot

WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
echo "WIN_HOST_IP=$WIN_HOST_IP"

sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
````

Expected:

* MAVProxy console dan map terbuka.
* SITL mengirim MAVLink ke:

  * Mission Planner (Windows) `WIN_HOST_IP:14550`
  * MAVROS (WSL) `127.0.0.1:14551`

---

## Mission Planner (Windows) — Connect

* Connection type: UDP
* Port: 14550
* Klik Connect

Expected:

* Vehicle status terlihat (mode/params).
* Map menampilkan vehicle.

---

## Terminal 2 (WSL) — Start MAVROS2

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

Expected:

* `/mavros/state` menjadi `connected: true`.

---

## Terminal 3 (WSL) — Monitor MAVROS state (opsional)

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mavros/state
```

Expected:

* `connected: true`
* `mode` terbaca
* `armed` sesuai kondisi

---

## Terminal 4 (WSL) — Start SEANO stack

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision run_auto_stack.launch.py
```

Expected:

* Node berjalan tanpa fatal error.
* Heartbeat failsafe ter-publish: `/ca/failsafe_active`.

---

## Validation Commands

### 1) MAVROS connected

```bash
ros2 topic echo /mavros/state -n 1
```

### 2) Failsafe heartbeat frequency

```bash
ros2 topic hz /ca/failsafe_active
```

### 3) RC override publishing (jika sedang uji aktuasi)

```bash
ros2 topic echo /mavros/rc/override -n 5
```

### 4) Duplicate publisher check (jika perilaku aneh / “nyangkut”)

```bash
ros2 topic info -v /seano/auto/left_cmd
ros2 topic info -v /seano/auto/right_cmd
ros2 topic info -v /seano/manual/left_cmd
ros2 topic info -v /seano/manual/right_cmd
```

---

## Common Failure Modes

### Mission Planner tidak bisa connect

* Pastikan `WIN_HOST_IP` diambil dari `ip route`.
* Pastikan SITL memuat: `--out udp:${WIN_HOST_IP}:14550`.

### MAVROS `connected: false`

* Pastikan SITL mengirim ke `udp:127.0.0.1:14551`.
* Pastikan MAVROS memakai:
  `udp://0.0.0.0:14551@127.0.0.1:14551`.
* Restart MAVROS setelah restart SITL.

### Vehicle tidak bergerak meskipun PWM berubah

* Pastikan vehicle ARMED.
* Pastikan mode menerima RC override (umumnya MANUAL untuk pengujian).
* Pastikan limiter tidak memaksa STOP (log FAILSAFE / stale).

---

# Phase 1 — Standard Maneuver Test (Control Validation)

Tujuan: memvalidasi jalur aktuasi end-to-end dari ROS 2 sampai ArduPilot SITL:

* `/mavros/rc/override` berubah (PWM steer/throttle)
* vehicle bergerak di Mission Planner
* tersedia bukti rosbag untuk TA

## Prasyarat

* SITL + Mission Planner + MAVROS sudah berjalan dan `/mavros/state connected: true`.
* Workspace sudah ter-build dan tersource.

### Build (jika ada perubahan file)

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## A) Run — Phase 1 Test (tanpa rekam)

Uji standar 5 siklus + auto shutdown:

```bash
source /opt/ros/humble/setup.bash
source ~/seano-collision-avoidance/seano_ca_ws/install/setup.bash
ros2 launch seano_vision phase1_maneuver_test.launch.py max_cycles:=5 lr_to_steer_gain:=0.6
```

Expected (indikator benar):

* Stage test berputar: `WARMUP_STOP -> FORWARD -> TURN_LEFT -> TURN_RIGHT -> STOP`.
* PWM berubah pada `/mavros/rc/override` (tidak statis 1500).
* Setelah selesai, auto shutdown dan semua proses exit cleanly (tidak spam FAILSAFE).

Verifikasi cepat:

```bash
ros2 topic echo /mavros/rc/override -n 5
ros2 topic echo /seano/left_cmd -n 5
ros2 topic echo /seano/right_cmd -n 5
```

---

## B) Run — Phase 1 Test + Record (disarankan untuk bukti TA)

Satu perintah untuk menjalankan test + merekam rosbag:

```bash
source /opt/ros/humble/setup.bash
source ~/seano-collision-avoidance/seano_ca_ws/install/setup.bash
ros2 launch seano_vision phase1_maneuver_record.launch.py
```

Default uji terkontrol pada record:

* `max_cycles:=5`
* `lr_to_steer_gain:=0.6`
* `base_throttle:=0.45`
* `turn_delta:=0.06`

Override contoh:

```bash
ros2 launch seano_vision phase1_maneuver_record.launch.py lr_to_steer_gain:=0.5
```

---

## Verifikasi rosbag (setelah record selesai)

Ambil bag terbaru dan tampilkan ringkasan:

```bash
BAG="$(ls -1dt ~/bags/*phase1_maneuver* | head -n 1)"
echo "BAG=$BAG"
ros2 bag info "$BAG"
```

Expected:

* Ada `Storage id: sqlite3`, `Duration`, `Message Count`.
* Topic minimal ada dan count > 0:

  * `/mavros/rc/override`
  * `/mavros/state`
  * `/seano/left_cmd`, `/seano/right_cmd`
  * `/seano/manual/*`, `/seano/selected/*`

Opsional (replay bukti):

```bash
ros2 bag play "$BAG"
# terminal lain:
ros2 topic echo /mavros/rc/override -n 5
```

Jika metadata bag tidak terbaca:

```bash
ros2 bag reindex "$BAG"
```

---
