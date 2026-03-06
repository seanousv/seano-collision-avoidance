# Architecture Overview — Phase 5 / Phase 6

Dokumen ini menjelaskan arsitektur runtime terbaru dari modul collision avoidance SEANO pada simulasi WSL2 / ROS 2 Humble.

Arsitektur ini merepresentasikan baseline aktif yang sekarang dipakai untuk pengujian integrasi:
- differential-thrust control (`left_cmd/right_cmd`)
- mission / avoid / rejoin mode handling
- MAVROS RC override bridge
- synthetic camera path untuk pengujian ringan
- rosbag metrics extraction untuk evaluasi Phase 6

---

## 1. Tujuan Arsitektur

Tujuan utama arsitektur ini:

1. Menjaga **autopilot mission** tetap menjadi penggerak utama waypoint.
2. Memberi **lapisan collision avoidance** yang dapat mengambil alih sementara saat ada risiko.
3. Mengembalikan sistem ke mission secara formal melalui state:
   - `MISSION`
   - `AVOID`
   - `REJOIN`
   - `FAILSAFE`
4. Memisahkan:
   - perception,
   - decision,
   - control selection,
   - actuation safety,
   - autopilot mode handling,
   agar mudah diuji per lapis.
5. Menyediakan jalur simulasi yang repeatable untuk:
   - Case A — mode manager only
   - Case B — takeover logic only
   - Case C — full integration (synthetic camera)

---

## 2. Konteks Kendaraan

USV SEANO menggunakan **differential thruster** tanpa rudder.

Karena itu, antarmuka kontrol internal utama bukan rudder, tetapi:

- `left_cmd`
- `right_cmd`

Keputusan ini dipakai agar perilaku kontrol sesuai dengan fisik kendaraan:
- maju lurus: `left == right`
- belok kanan: `left > right`
- belok kiri: `right > left`

---

## 3. Arsitektur Tingkat Tinggi

```mermaid
flowchart LR

subgraph MissionLayer["Mission / Autopilot Layer"]
    MP["Mission Planner / Mission Waypoints"]
    FCU["ArduPilot FCU (SITL / HW)"]
    MP --> FCU
end

subgraph PerceptionLayer["Perception / Decision Layer"]
    CAM["Camera Source\n(USB / RTSP / Synthetic)"]
    DET["Detector"]
    RISK["Risk Evaluator"]
    WD["Watchdog / Failsafe"]
    CAM --> DET
    DET --> RISK
    CAM --> WD
    RISK --> WD
end

subgraph ControlLayer["Command / Control Layer"]
    TELEOP["teleop_diff_thruster_node"]
    STUB["auto_controller_stub_node"]
    MUX["command_mux_node"]
    LIM["actuator_safety_limiter_node"]
    BR["mavros_rc_override_bridge_node"]

    TELEOP -->|/seano/manual/left_cmd,right_cmd| MUX
    STUB -->|/seano/auto/left_cmd,right_cmd| MUX
    MUX -->|/seano/selected/left_cmd,right_cmd| LIM
    LIM -->|/seano/left_cmd,right_cmd| BR
    BR -->|/mavros/rc/override| FCU
end

subgraph ModeLayer["Mode / State Layer"]
    MM["mission_mode_manager_node"]
    STATE["/ca/mode_manager_state"]
    EVT["/ca/mode_manager_event"]
    MM --> STATE
    MM --> EVT
end

RISK -->|/ca/command_safe| STUB
WD -->|/ca/failsafe_active| STUB
WD -->|/ca/failsafe_active| LIM
FCU -->|/mavros/state| MM
STUB -->|/seano/rc_override_enable| MM
WD -->|/ca/failsafe_active| MM
MM -->|SetMode| FCU
````

---

## 4. Prinsip Integrasi Mission

Prinsip integrasi yang dipakai adalah:

* autopilot tetap menjalankan **mission waypoint** dalam mode normal,
* sistem collision avoidance hanya melakukan **takeover sementara**,
* takeover dilakukan melalui:

  * `RC override`
  * perpindahan mode ke `MANUAL` saat avoidance / failsafe,
* setelah aman, override dilepas,
* mode dipulihkan ke mission target,
* sistem masuk `REJOIN`,
* setelah stabil, state kembali `MISSION`.

Dengan pola ini, arsitektur yang dipakai adalah:

**MISSION -> AVOID -> REJOIN -> MISSION**

dan saat kondisi tidak aman dari sisi sistem:

**MISSION / AVOID -> FAILSAFE**

---

## 5. State Machine Resmi

```mermaid
stateDiagram-v2
    [*] --> MISSION

    MISSION --> AVOID: rc_override_enable = true
    AVOID --> REJOIN: rc_override_enable = false
    REJOIN --> MISSION: mission mode restored and stable

    MISSION --> FAILSAFE: failsafe_active = true
    AVOID --> FAILSAFE: failsafe_active = true
    REJOIN --> FAILSAFE: failsafe_active = true

    FAILSAFE --> REJOIN: failsafe cleared and takeover off
    FAILSAFE --> AVOID: failsafe cleared but takeover still on
```

Makna tiap state:

### `MISSION`

* autopilot menjalankan mission normal
* mode target umumnya `AUTO`
* RC override tidak aktif

### `AVOID`

* collision avoidance takeover aktif
* RC override aktif
* mode target umumnya `MANUAL`

### `REJOIN`

* takeover sudah dilepas
* sistem sedang mengembalikan mode mission dan menunggu stabil
* mode target umumnya `AUTO`
* state ini dipakai agar “kembali ke mission” bisa diukur, bukan lompat langsung ke `MISSION`

### `FAILSAFE`

* perception / sistem dianggap tidak aman
* mode target aman, umumnya `MANUAL`
* output aktuasi dibatasi oleh limiter / kebijakan aman

---

## 6. Jalur Aktuasi Inti

Arsitektur aktuasi aktif adalah:

```text
manual/auto command
-> command_mux_node
-> actuator_safety_limiter_node
-> mavros_rc_override_bridge_node
-> /mavros/rc/override
-> ArduPilot
```

Rinciannya:

1. **manual command**

   * berasal dari `teleop_diff_thruster_node`
   * publish:

     * `/seano/manual/left_cmd`
     * `/seano/manual/right_cmd`

2. **auto command**

   * berasal dari `auto_controller_stub_node`
   * publish:

     * `/seano/auto/left_cmd`
     * `/seano/auto/right_cmd`

3. **selection**

   * dilakukan oleh `command_mux_node`
   * memilih jalur manual atau auto berdasarkan `/seano/auto_enable`

4. **safety**

   * dilakukan oleh `actuator_safety_limiter_node`
   * menangani:

     * stale command
     * failsafe input
     * clamp output
     * safe-stop policy

5. **bridge**

   * `mavros_rc_override_bridge_node`
   * mengubah `left/right` menjadi RC override PWM

---

## 7. Node dan Tanggung Jawab

## 7.1 `teleop_diff_thruster_node`

Tujuan:

* manual control untuk validasi low-level

Input:

* keyboard / teleop

Output:

* `/seano/manual/left_cmd`
* `/seano/manual/right_cmd`

---

## 7.2 `auto_controller_stub_node`

Tujuan:

* takeover logic sementara untuk simulasi / integrasi
* membaca command decision lalu menerjemahkannya ke `left/right`

Input:

* `/ca/command_safe`
* `/ca/failsafe_active`

Output:

* `/seano/auto/left_cmd`
* `/seano/auto/right_cmd`
* `/seano/auto_enable`
* `/seano/rc_override_enable`

Peran arsitektural:

* node ini adalah jembatan antara hazard command dan takeover control

---

## 7.3 `command_mux_node`

Tujuan:

* memilih sumber command:

  * manual
  * auto

Input:

* `/seano/manual/left_cmd`
* `/seano/manual/right_cmd`
* `/seano/auto/left_cmd`
* `/seano/auto/right_cmd`
* `/seano/auto_enable`

Output:

* `/seano/selected/left_cmd`
* `/seano/selected/right_cmd`

---

## 7.4 `actuator_safety_limiter_node`

Tujuan:

* menegakkan safety policy sebelum command masuk ke autopilot

Input:

* `/seano/selected/left_cmd`
* `/seano/selected/right_cmd`
* `/ca/failsafe_active`

Output:

* `/seano/left_cmd`
* `/seano/right_cmd`

Tanggung jawab utama:

* timeout command
* safe-stop policy
* clamp output
* stale handling

---

## 7.5 `mavros_rc_override_bridge_node`

Tujuan:

* mengubah command `left/right` menjadi PWM RC override

Input:

* `/seano/left_cmd`
* `/seano/right_cmd`
* `/seano/rc_override_enable`

Output:

* `/mavros/rc/override`

Peran:

* menjadi jembatan aktuasi ROS 2 -> MAVROS -> ArduPilot

---

## 7.6 `mission_mode_manager_node`

Tujuan:

* mengelola mode autopilot dan state machine tingkat tinggi

Input:

* `/mavros/state`
* `/seano/rc_override_enable`
* `/ca/failsafe_active`

Output:

* `/ca/mode_manager_state`
* `/ca/mode_manager_event`

Action:

* `/mavros/set_mode`

Peran utama:

* `MISSION` -> target mission mode
* `AVOID` -> target avoid mode
* `FAILSAFE` -> target failsafe mode
* `REJOIN` -> restore mode mission dan tunggu stabil

Tambahan penting:

* node ini juga memakai enforcement periodik agar kasus
  **MISSION tetapi mode autopilot masih MANUAL**
  bisa dipulihkan otomatis

---

## 7.7 Perception Nodes

Lapisan perception dapat berisi:

* `camera_node`
* `detector_node`
* `waterline_horizon_node`
* `false_positive_guard_node`
* `multi_target_fusion_node`
* `vision_quality_node`
* `frame_freeze_detector_node`
* `risk_evaluator_node`
* `watchdog_failsafe_node`

Tidak semua node perception harus aktif pada setiap mode uji.

---

## 8. Runtime Profiles

Arsitektur runtime sekarang dibedakan menjadi beberapa mode uji.

## 8.1 Case A — Mode Manager Only

Tujuan:

* validasi state machine tanpa perception
* validasi `MISSION -> AVOID -> REJOIN -> MISSION`

Aktif:

* mode manager
* mux
* limiter
* bridge

Nonaktif:

* CA pipeline
* takeover manager

---

## 8.2 Case B — Takeover Logic Only

Tujuan:

* validasi hazard -> takeover -> release

Aktif:

* takeover manager
* mode manager
* mux
* limiter
* bridge

Nonaktif:

* full CA perception chain

---

## 8.3 Case C — Full Integration (Synthetic Camera)

Tujuan:

* validasi integrasi penuh tanpa bergantung pada kamera hardware

Default yang dipakai:

* **synthetic_light**

Karakteristik:

* synthetic / dummy camera aktif
* detector aktif
* risk aktif
* lebih ringan untuk WSL

Varian:

* `synthetic_light`
* `synthetic_watchdog`
* `full`

---

## 9. Synthetic Camera Path

Untuk simulasi ringan, sumber kamera dapat berupa **synthetic camera**.

Tujuannya:

* menghindari ketergantungan pada kamera USB/hardware saat logika inti belum final
* mengurangi beban uji di WSL
* menjaga pipeline perception tetap punya input gambar

Konsekuensi:

* Case C dapat tetap berjalan walaupun kamera hardware tidak aktif
* tahap migrasi ke kamera USB dilakukan setelah logic control / state / metrics stabil

---

## 10. Topic Reference Inti

| Topic                       | Type                     | Peran                             |
| --------------------------- | ------------------------ | --------------------------------- |
| `/seano/manual/left_cmd`    | std_msgs/Float32         | manual left command               |
| `/seano/manual/right_cmd`   | std_msgs/Float32         | manual right command              |
| `/seano/auto/left_cmd`      | std_msgs/Float32         | auto left command                 |
| `/seano/auto/right_cmd`     | std_msgs/Float32         | auto right command                |
| `/seano/auto_enable`        | std_msgs/Bool            | memilih jalur auto di mux         |
| `/seano/selected/left_cmd`  | std_msgs/Float32         | output mux                        |
| `/seano/selected/right_cmd` | std_msgs/Float32         | output mux                        |
| `/ca/failsafe_active`       | std_msgs/Bool            | status failsafe                   |
| `/seano/left_cmd`           | std_msgs/Float32         | final left command                |
| `/seano/right_cmd`          | std_msgs/Float32         | final right command               |
| `/seano/rc_override_enable` | std_msgs/Bool            | mengaktifkan RC override          |
| `/mavros/rc/override`       | mavros_msgs/OverrideRCIn | PWM override ke FCU               |
| `/mavros/state`             | mavros_msgs/State        | mode / armed / connection state   |
| `/ca/command_safe`          | std_msgs/String          | command hazard / decision output  |
| `/ca/mode_manager_state`    | std_msgs/String          | `MISSION/AVOID/REJOIN/FAILSAFE`   |
| `/ca/mode_manager_event`    | std_msgs/String          | event JSON untuk audit dan metrik |

---

## 11. Phase 6 Metrics Architecture

Arsitektur ini sekarang mendukung evaluasi berbasis rosbag.

Metrik utama yang diambil:

1. **takeover segments**
2. **takeover duration**
3. **reaction time**
4. **release time**
5. **rejoin time**
6. **failsafe rises**
7. **mission-mode mismatch ratio**

Sumber utama:

* `/ca/command_safe`
* `/seano/rc_override_enable`
* `/ca/failsafe_active`
* `/mavros/state`
* `/ca/mode_manager_state`
* `/ca/mode_manager_event`

Artinya, state machine bukan hanya untuk kontrol, tetapi juga untuk **evidence generation**.

---

## 12. Batasan Saat Ini

Arsitektur aktif saat ini sudah mendukung:

* takeover avoidance
* release
* rejoin
* metrics extraction

Namun masih ada batasan:

* return-to-path masih memakai pendekatan **mission resume / mode restore**
* belum ada path planner rejoin eksplisit yang menghitung lintasan baru secara mandiri
* perception hardware final belum menjadi baseline utama; synthetic camera masih dipakai untuk pengujian ringan

---

## 13. Arah Pengembangan Berikutnya

Tahap berikut yang sesuai dengan arsitektur ini:

1. standardisasi skenario uji Phase 6
2. pengumpulan beberapa rosbag uji hazard / rejoin / failsafe
3. pembandingan metrik antar-run
4. migrasi dari synthetic camera ke USB camera
5. porting ke Jetson / FCU hardware target
6. uji lapangan terkontrol

---

## 14. Validasi Arsitektur

Arsitektur dianggap tervalidasi bila:

1. `/mavros/state connected: true`
2. mode manager menunjukkan:

   * `MISSION -> AVOID -> REJOIN -> MISSION`
3. event manager menunjukkan:

   * `TAKEOVER_ON`
   * `TAKEOVER_OFF`
   * `REJOIN_START`
   * `REJOIN_DONE`
4. extractor metrics menghasilkan:

   * `reaction_time_s`
   * `release_time_s`
   * `rejoin_time_s`
   * `mode_mismatch.mismatch_ratio` rendah / nol

---

## 15. Ringkasan

Arsitektur aktif SEANO sekarang dapat dibaca sebagai:

* autopilot tetap menjalankan mission,
* collision avoidance mengambil alih sementara,
* command internal memakai `left/right`,
* safety limiter menjaga aktuasi tetap aman,
* mode manager mengelola `MISSION/AVOID/REJOIN/FAILSAFE`,
* synthetic camera mendukung pengujian ringan,
* rosbag metrics menyediakan bukti kuantitatif untuk Phase 6.
