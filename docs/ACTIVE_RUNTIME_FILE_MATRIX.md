# ACTIVE RUNTIME FILE MATRIX
## SEANO Collision Avoidance — Matrix file aktif baseline TA

Dokumen ini dibuat untuk menjawab pertanyaan yang paling sering muncul saat repo mulai besar:

- file mana yang benar-benar aktif untuk TA,
- file mana yang dipakai untuk simulasi,
- file mana yang dipakai untuk hardware,
- file mana yang penting untuk evidence,
- dan file mana yang masih bersifat bench/debug.

Tujuan dokumen ini adalah membuat repo lebih mudah dibaca oleh:

- pengembang,
- pembimbing,
- penguji sidang,
- dan diri sendiri saat kembali membuka repo setelah jeda kerja.

---

## 1. Prinsip pembacaan

Repo ini saat ini harus dibaca sebagai **baseline TA aktif**, bukan sekadar workspace eksperimen umum.

Artinya:

1. Tidak semua file di folder `launch/` punya bobot yang sama.
2. Tidak semua node di folder `seano_vision/` dipakai sebagai evidence utama.
3. Baseline resmi harus dipusatkan pada file yang benar-benar dipakai pada hasil simulasi dan hardware yang ditulis di laporan.

---

## 2. Matrix file aktif utama

| File | Peran utama | Dipakai di simulasi | Dipakai di hardware | Status | Catatan |
| --- | --- | --- | --- | --- | --- |
| `README.md` | pintu masuk repo dan baseline resmi | Ya | Ya | **Aktif** | harus selalu sinkron dengan posisi TA |
| `docs/ARCHITECTURE.md` | arsitektur runtime aktif | Ya | Ya | **Aktif** | referensi utama untuk sidang dan penulisan |
| `docs/LAUNCH_STATUS_MAP.md` | peta pemilihan launch | Ya | Ya | **Aktif** | mencegah salah pilih launch |
| `seano_ca_ws/src/seano_vision/launch/phase5_mission_avoid_integration.launch.py` | baseline utama simulasi SITL | Ya | Tidak | **Aktif utama** | sumber utama evidence simulasi |
| `seano_ca_ws/src/seano_vision/launch/phase7_cuav_usb_hardware.launch.py` | baseline utama hardware Jetson + CUAV + USB camera | Tidak | Ya | **Aktif utama** | baseline utama bench/dockside/field prep |
| `seano_ca_ws/src/seano_vision/launch/demo_full_ca.launch.py` | bench pipeline CA penuh | Ya | Ya | **Aktif bench** | dipakai sebagai include perception/CA |
| `seano_ca_ws/src/seano_vision/launch/demo_detect.launch.py` | bench camera + detector | Tidak | Ya | **Aktif bench** | precheck cepat detector |
| `seano_ca_ws/src/seano_vision/launch/demo_risk.launch.py` | bench camera + detector + risk | Tidak | Ya | **Aktif bench** | precheck cepat risk / command |
| `seano_ca_ws/src/seano_vision/launch/phase2_camera_usb_test.launch.py` | bring-up kamera USB | Tidak | Ya | **Aktif bench** | baseline validasi source kamera hardware |
| `seano_ca_ws/src/seano_vision/seano_vision/camera_node.py` | source kamera | Ya | Ya | **Aktif inti** | mendukung synthetic / USB / pipeline |
| `seano_ca_ws/src/seano_vision/seano_vision/detector_node.py` | deteksi objek | Ya | Ya | **Aktif inti** | inference dan annotated image |
| `seano_ca_ws/src/seano_vision/seano_vision/risk_evaluator_node.py` | risk, command, HUD | Ya | Ya | **Aktif inti** | pusat decision layer |
| `seano_ca_ws/src/seano_vision/seano_vision/watchdog_failsafe_node.py` | health check + failsafe | Ya | Ya | **Aktif inti** | validasi perception health |
| `seano_ca_ws/src/seano_vision/seano_vision/command_mux_node.py` | seleksi command manual/auto | Ya | Ya | **Aktif inti** | sebelum limiter |
| `seano_ca_ws/src/seano_vision/seano_vision/actuator_safety_limiter_node.py` | clamp + timeout + slew limit | Ya | Ya | **Aktif inti** | pagar terakhir sebelum bridge |
| `seano_ca_ws/src/seano_vision/seano_vision/mavros_rc_override_bridge_node.py` | bridge ke `/mavros/rc/override` | Ya | Ya | **Aktif inti** | penerjemah left/right ke RC override |
| `seano_ca_ws/src/seano_vision/seano_vision/auto_controller_stub_node.py` | takeover manager runtime | Ya | Ya | **Aktif inti** | nama file legacy, fungsi runtime aktif |
| `seano_ca_ws/src/seano_vision/seano_vision/mission_mode_manager_node.py` | mode/state mission-autopilot | Ya | Ya | **Aktif inti** | `MISSION / AVOID / REJOIN / FAILSAFE` |
| `seano_ca_ws/src/seano_vision/scripts/phase6_metrics_from_bag.py` | ekstraksi metrik satu bag | Ya | Potensial | **Aktif evaluasi** | utama untuk evidence simulasi |
| `seano_ca_ws/src/seano_vision/scripts/phase6_collect_results.py` | agregasi banyak bag | Ya | Potensial | **Aktif evaluasi** | rekap hasil multi-run |

---

## 3. Matrix file yang harus dibaca sebagai bench/debug

File berikut masih berguna, tetapi **bukan baseline utama hasil TA**.

| File | Fungsi | Status pembacaan |
| --- | --- | --- |
| `phase2_camera_source_test.launch.py` | source test generik | bench/debug |
| `phase2_camera_detector_test.launch.py` | bench detector | bench/debug |
| `phase2_camera_detector_watchdog_test.launch.py` | bench detector + watchdog | bench/debug |
| `phase3_watchdog_camera_only.launch.py` | watchdog tanpa pipeline penuh | bench/debug |
| `phase4_takeover_actuation_test.launch.py` | uji takeover/actuation | bench/debug |
| `phase1_maneuver_test.launch.py` | uji manuver dasar | legacy bench |
| `phase1_maneuver_record.launch.py` | rekam manuver dasar | legacy bench |
| `lake_auto_demo.launch.py` | demo lapangan lama | audit setelah baseline final |
| `lake_auto_demo_hw.launch.py` | demo hardware lapangan lama | audit setelah baseline final |
| `demo_mavros_actuation_test.launch.py` | bench MAVROS actuation | bench/debug |
| `run_auto_stack.launch.py` | helper launch lama | audit nanti |

---

## 4. Cara membaca node aktif inti

### 4.1 Perception / decision

| File | Input utama | Output utama | Catatan |
| --- | --- | --- | --- |
| `camera_node.py` | source kamera aktif | `/seano/camera/image_raw_reliable` | image source utama |
| `detector_node.py` | image raw | detections + annotated image | inference layer |
| `risk_evaluator_node.py` | detections + image + VQ/freeze opsional | `/ca/risk`, `/ca/command`, `/ca/debug_image` | decision layer utama |
| `watchdog_failsafe_node.py` | image / risk / mode / command / VQ / freeze | `/ca/failsafe_active`, `/ca/command_safe`, `/ca/watchdog_status` | health and safety layer |

### 4.2 Control

| File | Input utama | Output utama | Catatan |
| --- | --- | --- | --- |
| `auto_controller_stub_node.py` | `/ca/command` atau `/ca/command_safe`, failsafe | auto left/right + enable flags | runtime takeover manager |
| `command_mux_node.py` | manual left/right, auto left/right, auto_enable | selected left/right | selector, bukan final output |
| `actuator_safety_limiter_node.py` | selected left/right + failsafe | final left/right | safety gate terakhir |
| `mavros_rc_override_bridge_node.py` | final left/right | `/mavros/rc/override` | bridge ke autopilot |

### 4.3 Mission / autopilot mode

| File | Input utama | Output utama | Catatan |
| --- | --- | --- | --- |
| `mission_mode_manager_node.py` | `/mavros/state`, `/seano/rc_override_enable`, `/ca/failsafe_active` | `/ca/mode_manager_state`, `/ca/mode_manager_event` | state formal mission-aware |

---

## 5. Aturan praktis saat presentasi / sidang

Jika dosen bertanya “file mana yang benar-benar dipakai?”, jawabannya dipusatkan ke tiga tingkat berikut:

### Tingkat 1 — baseline utama
- `phase5_mission_avoid_integration.launch.py`
- `phase7_cuav_usb_hardware.launch.py`

### Tingkat 2 — node inti runtime
- `camera_node.py`
- `detector_node.py`
- `risk_evaluator_node.py`
- `watchdog_failsafe_node.py`
- `command_mux_node.py`
- `actuator_safety_limiter_node.py`
- `mavros_rc_override_bridge_node.py`
- `auto_controller_stub_node.py`
- `mission_mode_manager_node.py`

### Tingkat 3 — evaluasi evidence
- `phase6_metrics_from_bag.py`
- `phase6_collect_results.py`

Ini menjaga penjelasan tetap fokus dan tidak tenggelam di file bench/debug yang terlalu banyak.

---

## 6. Yang tidak boleh dicampur saat klaim hasil

Agar repo tetap profesional dan konsisten dengan laporan:

1. **hasil simulasi** jangan dicampur dengan klaim field success hardware,
2. **bench/debug launch** jangan dipresentasikan seolah baseline utama,
3. **legacy file** jangan dijadikan sumber klaim hasil final,
4. **nama file aktif** jangan diubah dulu sebelum baseline field final terkunci.

---

## 7. Kesimpulan

Repo ini sekarang sudah cukup besar, sehingga pembaca harus dibantu dengan peta yang jelas.

Dokumen ini menetapkan bahwa:

- `phase5` adalah baseline aktif utama simulasi,
- `phase7` adalah baseline aktif utama hardware,
- node inti sistem ada pada jalur perception → watchdog → takeover → mux → limiter → bridge → mode manager,
- file bench/debug tetap berguna, tetapi tidak boleh mengaburkan baseline resmi TA.
