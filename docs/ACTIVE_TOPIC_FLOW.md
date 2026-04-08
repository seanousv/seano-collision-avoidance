# ACTIVE TOPIC FLOW
## SEANO Collision Avoidance — Alur topic aktif baseline TA

Dokumen ini menjelaskan alur topic yang benar-benar penting untuk membaca runtime SEANO.

Tujuan utamanya:

- memudahkan tracing saat debugging,
- memudahkan penjelasan saat sidang,
- memudahkan sinkronisasi antara kode, HUD, rosbag, dan laporan.

---

## 1. Prinsip umum

Rantai aktif sistem dibaca sebagai:

```text
camera
-> detector
-> risk evaluator
-> watchdog / command_safe
-> takeover manager
-> command mux
-> safety limiter
-> RC override bridge
-> MAVROS / ArduPilot
-> mode manager / event evidence
```

Namun detail topic yang dipakai tetap harus dibaca per layer.

---

## 2. Layer perception

### 2.1 Image source

Topic utama:
- `/seano/camera/image_raw_reliable`

Peran:
- image utama yang dipakai detector,
- image utama untuk monitoring raw,
- baseline image reference untuk bench hardware.

### 2.2 Detector output

Topic utama:
- `/camera/detections`
- `/camera/image_annotated`

Peran:
- hasil deteksi objek,
- visual evidence untuk operator,
- input dasar ke layer risk.

Catatan:
- pada beberapa konfigurasi, risk bisa membaca topic detections hasil filter/fusion, tetapi baseline pembacaan repo tetap harus menekankan bahwa detections adalah titik awal layer keputusan.

---

## 3. Layer decision

### 3.1 Risk evaluator

Topic utama keluar:
- `/ca/risk`
- `/ca/command`
- `/ca/mode`
- `/ca/metrics`
- `/ca/debug_image`

Makna:
- `/ca/risk` = skor risiko utama,
- `/ca/command` = keputusan avoidance sebelum diamankan watchdog,
- `/ca/mode` = state perception/risk internal,
- `/ca/debug_image` = HUD untuk diagnosis.

### 3.2 Watchdog / failsafe

Topic utama keluar:
- `/ca/command_safe`
- `/ca/failsafe_active`
- `/ca/failsafe_reason`
- `/ca/watchdog_status`

Makna:
- watchdog menentukan apakah jalur persepsi masih cukup sehat untuk dipercaya,
- command dari risk dapat diturunkan atau diganti menjadi versi aman,
- failsafe dapat diangkat walaupun detector/risk masih hidup bila jalur image atau VQ dianggap tidak sehat.

---

## 4. Layer control / actuation

### 4.1 Takeover manager

Input utama:
- `/ca/command` atau `/ca/command_safe`
- `/ca/failsafe_active`

Output utama:
- `/seano/auto/left_cmd`
- `/seano/auto/right_cmd`
- `/seano/auto_enable`
- `/seano/rc_override_enable`

Makna:
- di sinilah keputusan avoidance diterjemahkan menjadi perintah left/right normalized,
- node ini juga menentukan kapan takeover aktif dan kapan RC override dilepas.

### 4.2 Command mux

Input utama:
- `/seano/manual/left_cmd`
- `/seano/manual/right_cmd`
- `/seano/auto/left_cmd`
- `/seano/auto/right_cmd`
- `/seano/auto_enable`

Output:
- `/seano/selected/left_cmd`
- `/seano/selected/right_cmd`

Makna:
- selector command,
- bukan final actuator output.

### 4.3 Safety limiter

Input utama:
- `/seano/selected/left_cmd`
- `/seano/selected/right_cmd`
- `/ca/failsafe_active`

Output:
- `/seano/left_cmd`
- `/seano/right_cmd`
- `/seano/limiter_reason`

Makna:
- clamp,
- timeout stop,
- failsafe stop,
- slew limiting.

### 4.4 RC override bridge

Input utama:
- `/seano/left_cmd`
- `/seano/right_cmd`
- `/seano/rc_override_enable`

Output utama:
- `/mavros/rc/override`

Makna:
- jembatan ROS2 left/right command ke RC override MAVROS / ArduPilot.

---

## 5. Layer autopilot / mission mode

### 5.1 MAVROS state

Topic utama:
- `/mavros/state`

Makna:
- sumber pembacaan koneksi autopilot,
- mode FCU aktual,
- dasar validasi connected / not connected.

### 5.2 Mission mode manager

Input utama:
- `/mavros/state`
- `/seano/rc_override_enable`
- `/ca/failsafe_active`

Output utama:
- `/ca/mode_manager_state`
- `/ca/mode_manager_event`

Makna:
- ini layer state formal mission-aware,
- bukan layer perception.

State formal yang dipublish:
- `MISSION`
- `AVOID`
- `REJOIN`
- `FAILSAFE`

---

## 6. Topic yang paling penting untuk rosbag evidence

Untuk evidence simulasi dan hardware, topic yang paling kuat biasanya adalah:

### 6.1 Decision chain
- `/ca/risk`
- `/ca/command`
- `/ca/command_safe`
- `/ca/failsafe_active`

### 6.2 Takeover chain
- `/seano/auto_enable`
- `/seano/rc_override_enable`
- `/seano/left_cmd`
- `/seano/right_cmd`
- `/mavros/rc/override`

### 6.3 Mission chain
- `/mavros/state`
- `/ca/mode_manager_state`
- `/ca/mode_manager_event`

### 6.4 Visual evidence
- `/seano/camera/image_raw_reliable`
- `/camera/image_annotated`
- `/ca/debug_image`

---

## 7. Alur cepat yang harus diingat saat debugging

### Jika annotated image tidak muncul
cek berurutan:
1. `/seano/camera/image_raw_reliable`
2. `camera_node`
3. `detector_node`
4. model path / runtime device

### Jika risk tidak berubah
cek:
1. `/camera/detections`
2. `risk_evaluator_node`
3. topic detections yang dibaca risk
4. parameter threshold visual

### Jika command sudah ada tapi kapal tidak merespons
cek:
1. `/ca/command` atau `/ca/command_safe`
2. `/seano/auto_enable`
3. `/seano/rc_override_enable`
4. `/seano/selected/left_cmd`
5. `/seano/left_cmd`
6. `/mavros/rc/override`
7. `/mavros/state`

### Jika state rejoin tidak pernah selesai
cek:
1. `/mavros/state`
2. mode target mission default
3. `/ca/mode_manager_event`
4. timeout dan stable time pada mission mode manager

---

## 8. Ringkasan singkat untuk sidang

Jika harus menjelaskan cepat di depan penguji, versi paling ringkas adalah:

```text
kamera publish image,
detector menghasilkan detections,
risk evaluator mengubah detections menjadi risk dan command,
watchdog memastikan persepsi masih sehat,
takeover manager mengubah command menjadi left/right dan mengaktifkan override,
command melewati mux dan limiter,
bridge mengirim ke MAVROS,
mode manager memastikan autopilot dibaca secara formal sebagai MISSION / AVOID / REJOIN / FAILSAFE.
```

Itu adalah alur topic aktif yang paling penting untuk dibela sebagai arsitektur utama TA.
