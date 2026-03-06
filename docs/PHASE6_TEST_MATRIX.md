# Phase 6 Test Matrix and Results Template

Dokumen ini dipakai untuk membakukan pengujian Phase 6 agar:
- skenario uji selalu konsisten,
- nama bag konsisten,
- metrik yang diambil konsisten,
- hasil mudah dipindah ke laporan TA.

Dokumen ini dipakai setelah:
- `docs/RUNBOOK.md`
- `docs/ARCHITECTURE.md`

---

## 1. Tujuan

Tujuan Phase 6 adalah menghasilkan bukti kuantitatif bahwa sistem collision avoidance:
- dapat takeover saat hazard,
- dapat release setelah clear,
- dapat rejoin ke mission,
- dapat masuk failsafe bila diperlukan,
- tetap menjaga konsistensi mode mission.

---

## 2. Metrik Utama

Metrik yang dicatat untuk setiap run:

1. **takeover_segments**
2. **takeover_duration_mean_s**
3. **reaction_time_mean_s**
4. **release_time_mean_s**
5. **rejoin_time_mean_s**
6. **failsafe_rises**
7. **mode_mismatch_ratio**

Tambahan penting:
- `rejoin.done`
- `rejoin.cancelled`
- `rejoin.timeouts`

---

## 3. Aturan Penamaan Bag

Gunakan format nama bag berikut:

```text
phase6_<scenario>_run_<nn>
````

Contoh:

* `phase6_rejoin_run_01`
* `phase6_rejoin_run_02`
* `phase6_rejoin_run_03`
* `phase6_failsafe_run_01`
* `phase6_failsafe_run_02`
* `phase6_watchdog_run_01`

Aturan:

* `<scenario>` harus konsisten,
* `<nn>` dua digit,
* satu bag hanya untuk satu skenario utama.

---

## 4. Profil Runtime yang Dipakai

Gunakan profil berikut secara konsisten:

### A. `synthetic_light`

Dipakai untuk:

* uji hazard / takeover / release / rejoin
* beban ringan
* tanpa watchdog penuh

### B. `synthetic_watchdog`

Dipakai untuk:

* uji failsafe sintetis
* uji transisi ke FAILSAFE

### C. `full`

Dipakai hanya bila:

* ingin uji perception chain penuh,
* sistem synthetic_light dan synthetic_watchdog sudah stabil.

---

## 5. Matriks Skenario Uji

## 5.1 Skenario S1 — Hazard Right -> Rejoin

### Tujuan

Validasi:

* hazard memicu takeover,
* release terjadi saat clear,
* rejoin berhasil,
* mission pulih normal.

### Launch

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_rejoin_run_01 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

### Trigger

```bash
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_RIGHT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"
```

### Expected

* state: `MISSION -> AVOID -> REJOIN -> MISSION`
* event:

  * `TAKEOVER_ON`
  * `TAKEOVER_OFF`
  * `REJOIN_START`
  * `REJOIN_DONE`
* `takeover_segments >= 1`
* `reaction_time_s.n >= 1`
* `release_time_s.n >= 1`
* `rejoin_time_s.n >= 1`

---

## 5.2 Skenario S2 — Hazard Left -> Rejoin

### Tujuan

Sama seperti S1, tetapi arah hazard berlawanan.

### Launch

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_rejoin_run_02 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

### Trigger

```bash
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_LEFT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"
```

### Expected

* state: `MISSION -> AVOID -> REJOIN -> MISSION`
* metrik reaction/release/rejoin terisi

---

## 5.3 Skenario S3 — Repeated Hazard Cycle

### Tujuan

Mengukur konsistensi ketika ada beberapa hazard dalam satu run.

### Launch

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_rejoin_run_03 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

### Trigger

```bash
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_RIGHT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"

sleep 2

ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_LEFT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"
```

### Expected

* `takeover_segments >= 2`
* `rejoin.done >= 2`
* `rejoin.cancelled = 0`
* `rejoin.timeouts = 0`

---

## 5.4 Skenario S4 — Synthetic Watchdog / Failsafe

### Tujuan

Validasi transisi ke FAILSAFE.

### Launch

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_failsafe_run_01 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true \
  ca_runtime_profile:=synthetic_watchdog
```

### Trigger

Gunakan metode yang paling aman dan repeatable di setup Anda untuk memicu `failsafe_active`, misalnya:

* hentikan image source,
* hentikan node perception utama,
* atau metode internal yang memang memicu watchdog LOST.

### Expected

* state masuk `FAILSAFE`
* event `FAILSAFE_ON` muncul
* `failsafe.rises >= 1`

---

## 5.5 Skenario S5 — Failsafe Clear -> Rejoin / Mission

### Tujuan

Validasi pemulihan setelah failsafe.

### Launch

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_failsafe_run_02 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true \
  ca_runtime_profile:=synthetic_watchdog
```

### Trigger

1. Picu failsafe.
2. Kembalikan kondisi normal.
3. Amati apakah state kembali melalui:

   * `FAILSAFE -> REJOIN -> MISSION`
     atau
   * `FAILSAFE -> AVOID` bila takeover masih aktif.

### Expected

* `FAILSAFE_ON` dan `FAILSAFE_OFF` terekam
* bila clear dan takeover off:

  * `REJOIN_START`
  * `REJOIN_DONE`

---

## 6. Template Pencatatan Run

Gunakan tabel ini untuk setiap run.

| Run ID | Scenario             |    Runtime Profile | Bag Name               | Takeover Segments | Reaction Mean (s) | Release Mean (s) | Rejoin Mean (s) | Rejoin Done | Rejoin Cancelled | Rejoin Timeouts | Failsafe Rises | Mismatch Ratio | Pass/Fail | Catatan |
| ------ | -------------------- | -----------------: | ---------------------- | ----------------: | ----------------: | ---------------: | --------------: | ----------: | ---------------: | --------------: | -------------: | -------------: | --------- | ------- |
| R1     | S1 Hazard Right      |    synthetic_light | phase6_rejoin_run_01   |                   |                   |                  |                 |             |                  |                 |                |                |           |         |
| R2     | S2 Hazard Left       |    synthetic_light | phase6_rejoin_run_02   |                   |                   |                  |                 |             |                  |                 |                |                |           |         |
| R3     | S3 Repeated Hazard   |    synthetic_light | phase6_rejoin_run_03   |                   |                   |                  |                 |             |                  |                 |                |                |           |         |
| R4     | S4 Failsafe          | synthetic_watchdog | phase6_failsafe_run_01 |                   |                   |                  |                 |             |                  |                 |                |                |           |         |
| R5     | S5 Failsafe Recovery | synthetic_watchdog | phase6_failsafe_run_02 |                   |                   |                  |                 |             |                  |                 |                |                |           |         |

---

## 7. Template Ringkasan JSON -> Tabel

Setelah menjalankan:

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_metrics_from_bag.py \
  --bag ~/bags/<bag_name>
```

ambil nilai berikut dari `phase6_metrics.json`:

### Dari `override`

* `takeover_segments`
* `duration_s.mean`

### Dari `reaction_time_s`

* `mean`
* `n`

### Dari `release_time_s`

* `mean`
* `n`

### Dari `rejoin`

* `done`
* `cancelled`
* `timeouts`

### Dari `rejoin_time_s`

* `mean`
* `n`

### Dari `failsafe`

* `rises`

### Dari `mode_mismatch`

* `mismatch_ratio`

---

## 8. Kriteria Pass / Fail per Run

Sebuah run dianggap **PASS** bila sesuai skenario targetnya.

### Untuk skenario hazard / rejoin

PASS jika:

* `takeover_segments >= 1`
* `reaction_time_s.n >= 1`
* `release_time_s.n >= 1`
* `rejoin_time_s.n >= 1`
* `rejoin.done >= 1`
* `rejoin.cancelled = 0`
* `rejoin.timeouts = 0`
* `mode_mismatch_ratio` rendah / nol

FAIL jika:

* takeover tidak terjadi,
* rejoin tidak selesai,
* mode mismatch besar,
* bag tidak memuat event utama.

### Untuk skenario failsafe

PASS jika:

* `failsafe.rises >= 1`
* event `FAILSAFE_ON` muncul
* transisi mode aman terlihat

FAIL jika:

* failsafe tidak pernah aktif,
* bag tidak memuat topic failsafe,
* perilaku mode tidak sesuai.

---

## 9. Template Catatan Kualitatif

Selain angka, selalu catat observasi ini:

### A. Mission Planner

* mode vehicle berubah sesuai harapan / tidak
* rover terlihat keluar jalur / tidak
* rover kembali ke mission / tidak

### B. State ROS 2

* urutan state manager yang terlihat
* apakah ada state aneh / loncatan tidak wajar

### C. Event JSON

* apakah event lengkap
* apakah ada `MODE_REQ_SKIPPED`
* apakah ada `REJOIN_TIMEOUT`

### D. Kestabilan Runtime

* lag / tidak
* crash / tidak
* dummy camera stabil / tidak

---

## 10. Template Ringkasan untuk Laporan TA

Setelah beberapa run selesai, format ringkasan bisa dibuat seperti ini:

### Contoh narasi hasil

> Pada skenario hazard-rejoin, sistem berhasil melakukan takeover sebanyak X kali dengan durasi rata-rata Y s. Waktu reaksi rata-rata terhadap command hazard adalah Z s, waktu release rata-rata setelah kondisi clear adalah A s, dan waktu rejoin ke mission adalah B s. Pada run yang diuji, tidak ditemukan timeout rejoin dan rasio mismatch mode mission terhadap autopilot adalah C.

### Contoh tabel ringkas

| Scenario        | Run Count | Takeover Mean | Reaction Mean | Release Mean | Rejoin Mean | Failsafe Mean | Mismatch Ratio |
| --------------- | --------: | ------------: | ------------: | -----------: | ----------: | ------------: | -------------: |
| Hazard / Rejoin |           |               |               |              |             |             - |                |
| Failsafe        |           |             - |             - |            - |             |               |                |

---

## 11. Urutan Eksekusi yang Direkomendasikan

Urutan pengambilan data yang disarankan:

1. `S1` Hazard Right
2. `S2` Hazard Left
3. `S3` Repeated Hazard
4. `S4` Failsafe
5. `S5` Failsafe Recovery

Minimal target:

* 3 run hazard / rejoin
* 2 run failsafe

---

## 12. Checklist Sebelum Record

Sebelum mulai record, cek:

* [ ] SITL aktif
* [ ] Mission Planner connect
* [ ] MAVROS `connected: true`
* [ ] launch Phase 5 berjalan
* [ ] mode runtime benar
* [ ] bag name benar
* [ ] topic `/ca/mode_manager_event` aktif
* [ ] topic `/mavros/state` aktif
* [ ] topic `/seano/rc_override_enable` aktif

---

## 13. Checklist Setelah Record

Setelah record selesai, cek:

* [ ] folder bag ada
* [ ] extractor metrics berhasil jalan
* [ ] `phase6_metrics.json` tersimpan
* [ ] nilai reaction/release/rejoin terisi sesuai skenario
* [ ] hasil dicatat ke tabel run

---

## 14. Status Minimum untuk Lanjut ke Hardware

Phase 6 simulasi dianggap cukup matang bila:

* minimal 3 run hazard / rejoin sudah valid,
* minimal 2 run failsafe sudah valid,
* `rejoin.done` konsisten,
* `rejoin.timeout = 0` pada run normal,
* `mode_mismatch_ratio` sangat rendah / nol,
* sistem stabil di `synthetic_light` dan `synthetic_watchdog`.

Setelah itu, baru masuk ke:

* migrasi ke USB camera,
* porting ke hardware target,
* uji lapangan terkontrol.
