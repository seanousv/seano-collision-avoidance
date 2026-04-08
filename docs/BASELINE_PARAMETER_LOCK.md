# BASELINE PARAMETER LOCK

## SEANO Collision Avoidance - Active Parameter Summary

Dokumen ini merangkum parameter aktif yang paling penting untuk baseline TA saat ini.

Tujuan dokumen ini:
- memudahkan audit parameter aktif,
- membantu sinkronisasi antara repo, laporan, dan sidang,
- membedakan parameter yang sudah cukup stabil dari parameter yang masih bisa berubah setelah data lapangan bertambah.

---

## 1. Parameter Visual / Geometry Baseline

Parameter visual aktif yang saat ini dipakai pada evaluator risiko:

- `camera_hfov_deg = 112.96`
- `center_band_ratio = 0.212`
- `bottom_danger_ratio = 0.647`
- `near_area_ratio = 0.0183`

Interpretasi:
- `camera_hfov_deg` dipakai sebagai pendekatan horizontal FOV untuk proxy bearing,
- `center_band_ratio` dipakai untuk mendefinisikan corridor tengah,
- `bottom_danger_ratio` dipakai untuk menilai danger zone bawah frame,
- `near_area_ratio` dipakai untuk proxy kedekatan berbasis area bounding box.

Catatan:
- angka-angka ini sudah cukup layak dipakai sebagai baseline aktif repo,
- tetapi tetap harus dibaca sebagai parameter desain yang dapat disempurnakan lagi bila ada hasil kalibrasi kamera final atau uji lapangan yang lebih matang.

---

## 2. Bobot Risk Evaluator Aktif

Bobot evaluasi risiko yang aktif di `risk_evaluator_node.py`:

- `w_proximity = 0.40`
- `w_center = 0.18`
- `w_approach = 0.16`
- `w_bearing_const = 0.10`
- `w_ttc = 0.16`

Parameter tambahan:
- `bearing_rate_bad_dps = 12.0`
- `risk_ema_alpha = 0.35`
- `vq_risk_floor = 0.80`

Interpretasi:
- evaluator risiko menekankan kedekatan dan posisi bahaya,
- pendekatan target dan konsistensi bearing ikut memengaruhi risiko,
- proxy vTTC ikut dipakai untuk menguatkan sense of urgency,
- EMA dipakai agar risk tidak terlalu noisy.

---

## 3. Threshold Command Aktif

Threshold keputusan aktif:

- `enter_avoid_risk = 0.55`
- `exit_avoid_risk = 0.35`
- `risk_slow_threshold = 0.45`
- `risk_turn_slow_threshold = 0.55`
- `risk_turn_threshold = 0.75`
- `risk_stop_threshold = 0.92`

Threshold vTTC aktif:

- `vttc_turn_threshold_s = 4.0`
- `vttc_stop_threshold_s = 1.2`

Interpretasi:
- sistem mulai masuk regime avoid pada risk menengah ke atas,
- release butuh risk turun lebih rendah untuk menjaga hysteresis,
- keputusan `STOP` dicadangkan untuk kondisi sangat tinggi / emergency.

---

## 4. Takeover Manager Active Tuning

Parameter takeover manager pada baseline aktif berada di dua jalur utama.

### Simulation baseline (`phase5`)
- `cruise_speed = 0.30`
- `turn_cmd = 0.55`
- `diff_mix_gain = 0.70`

### Hardware baseline (`phase7`)
- `cruise_speed = 0.30`
- `slow_factor = 0.55`
- `turn_speed_factor = 0.75`
- `turn_cmd = 0.50`
- `diff_mix_gain = 0.65`
- `speed_max = 0.55`

Interpretasi:
- baseline simulasi dan hardware sudah dibedakan secara wajar,
- hardware diberi tuning yang sedikit lebih konservatif dibanding simulasi,
- parameter ini masih boleh disesuaikan lagi jika ada data lapangan baru yang lebih representatif.

---

## 5. Bridge / PWM Baseline

### Simulation bridge baseline
- `input_mode = left_right`
- `output_mode = rc_thr_steer`
- `pwm_neutral = 1500`
- `pwm_fwd_max = 1900`
- `pwm_steer_left = 1100`
- `pwm_steer_right = 1900`

### Hardware bridge baseline
- `input_mode = left_right`
- `output_mode = rc_left_right`
- `pwm_neutral = 1500`
- `pwm_fwd_max = 1900`
- `pwm_rev_min = 1100`
- `pwm_output_min = 1000`
- `pwm_output_max = 2000`

Interpretasi:
- simulasi tetap memakai jalur `rc_thr_steer` agar cocok dengan rover-skid SITL,
- hardware memakai `rc_left_right` agar lebih dekat ke filosofi differential thrust internal,
- dokumen ini membantu menjelaskan mengapa output mode simulasi dan hardware tidak identik.

---

## 6. Timeout / Safety Baseline

Parameter timeout dan safety penting yang aktif:

### Mux / limiter
- `mux command_timeout_s = 0.6`
- `limiter input_timeout_s = 0.6`
- `limiter failsafe_timeout_s = 2.0`
- `limiter loop_hz = 20.0`

### Bridge
- `bridge command_timeout_s = 0.5`
- `bridge pub_hz = 20.0`

### Watchdog startup
- `synthetic profile startup_grace_s` dibuat lebih longgar dibanding jalur hardware ringan,
- `usb_watchdog` dan `full` profile tetap memakai startup grace yang lebih realistis untuk kamera nyata.

Interpretasi:
- timeout command sengaja dibuat cukup pendek untuk mencegah command stale bertahan terlalu lama,
- failsafe timeout dibuat lebih panjang agar jalur watchdog tidak terlalu sensitif terhadap noise singkat,
- publish/control loop 20 Hz dipilih sebagai kompromi praktis antara respons dan kestabilan.

---

## 7. Parameter yang Cukup Stabil vs Masih Dapat Berubah

### Cukup stabil untuk baseline repo saat ini
- struktur topic utama,
- filosofi left/right command,
- threshold avoid enter/exit,
- pola command severity,
- mode baseline `phase5` dan `phase7`,
- mapping high-level simulation vs hardware output mode.

### Masih dapat berubah bila ada evidence baru
- tuning dinamik `cruise_speed`, `turn_cmd`, `diff_mix_gain`,
- beberapa threshold risk akhir bila uji lapangan menunjukkan perlu koreksi,
- parameter watchdog atau VQ jika hardware stream ternyata lebih fluktuatif dari bench,
- mapping channel/pwm final jika FCU setup berubah.

---

## 8. Rule of Use

Dokumen ini tidak dimaksudkan sebagai pengunci absolut untuk semua masa depan proyek.

Dokumen ini berfungsi sebagai:
- **baseline audit aktif**,
- referensi parameter untuk membaca repo dan laporan saat ini,
- alat bantu agar perubahan parameter tidak terjadi tanpa jejak yang jelas.

Jika ada perubahan parameter penting setelah dockside atau field test, dokumen ini harus ikut diperbarui agar repo tetap sinkron dengan evidence terbaru.
