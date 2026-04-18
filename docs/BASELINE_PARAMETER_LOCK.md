# BASELINE PARAMETER LOCK

## Parameter Acuan Resmi untuk Thesis Baseline Saat Ini

Dokumen ini dipakai untuk membedakan tiga kategori parameter:

1. **design-lock**
   Sudah boleh dipakai sebagai nilai acuan resmi repo/laporan saat ini.

2. **baseline-default**
   Nilai default runtime aktif yang boleh dipakai untuk baseline sekarang, tetapi masih terbuka untuk tuning.

3. **not-final-yet**
   Belum boleh dianggap final karena masih menunggu data FOV kamera, kalibrasi kamera, data uji dinamik kapal, ESC/thruster, atau uji air.

---

## 1. Tujuan Dokumen

Dokumen ini dibuat agar:
- angka yang dipakai di repo konsisten dengan angka yang dipakai di laporan TA,
- pembaca bisa membedakan mana angka yang sudah dikunci dan mana yang masih tentatif,
- tuning lapangan tidak mengaburkan baseline engineering yang sudah ada,
- migrasi baseline dari platform lama ke **SEANO Alfin7** terdokumentasi jelas.

---

## 2. Vehicle and Camera Baseline

### 2.1 Design-lock

Parameter berikut sudah boleh dianggap sebagai **baseline resmi saat ini**.

| Parameter | Nilai | Status | Catatan |
|---|---:|---|---|
| panjang kapal `L` | 0.70 m | design-lock | basis geometri platform final |
| lebar kapal `B` | 0.50 m | design-lock | basis protected corridor minimum |
| tinggi kamera `h_c` | 0.58 m | design-lock | tinggi pemasangan kamera final |
| resolusi citra `W` | 640 px | design-lock | runtime image width final |
| resolusi citra `H` | 480 px | design-lock | runtime image height final |
| pusat citra `c_x` | 320 px | design-lock | `W/2` |
| pusat citra `c_y` | 240 px | design-lock | `H/2` |
| protected corridor minimum `W_p,min` | 0.50 m | design-lock | minimum sama dengan lebar fisik kapal |

### 2.2 Not-final-yet

Parameter berikut **belum boleh dikunci sebagai angka final** karena masih menunggu data kamera yang valid.

| Parameter | Nilai saat ini | Status | Catatan |
|---|---:|---|---|
| `camera_hfov_deg` | TBD | not-final-yet | menunggu datasheet valid atau hasil kalibrasi |
| `camera_vfov_deg` | TBD | not-final-yet | menunggu datasheet valid atau hasil kalibrasi |
| `f_x` | TBD | not-final-yet | menunggu kalibrasi kamera |
| `f_y` | TBD | not-final-yet | menunggu kalibrasi kamera |
| pitch kamera `alpha` | 0 deg (nominal) | not-final-yet | boleh diasumsikan nominal, belum final kalibrasi mount |

---

## 3. Visual Risk Geometry Parameters

Parameter berikut **belum boleh dianggap final** sampai FOV kamera atau hasil kalibrasi final tersedia.

| Parameter | Nilai saat ini | Status | Makna |
|---|---:|---|---|
| `center_band_ratio` | TBD | not-final-yet | koridor konflik minimum berbasis lebar kapal dan HFOV |
| `bottom_danger_ratio` | TBD | not-final-yet | zona bawah yang merepresentasikan objek dekat/berisiko |
| `near_area_ratio` | TBD | not-final-yet | ambang bbox near berbasis geometri visual final |

Catatan penting:
- nilai lama untuk platform 3.0 m tidak lagi menjadi baseline resmi,
- tiga parameter di atas baru boleh diisi setelah FOV kamera A95 valid atau hasil kalibrasi final tersedia,
- sampai saat itu, repo harus memperlakukan parameter ini sebagai **belum final**.

---

## 4. Risk Threshold Parameters

Parameter berikut **belum boleh diperlakukan sebagai final thesis constants**. Nilai default aktifnya tetap boleh dipakai untuk baseline uji, tetapi statusnya adalah **runtime-tunable**.

| Parameter | Default aktif repo | Status | Catatan |
|---|---:|---|---|
| `enter_avoid_risk` | 0.55 | baseline-default | masih perlu dijaga terhadap distribusi risk nyata |
| `exit_avoid_risk` | 0.35 | baseline-default | hysteresis sudah benar, tetapi belum final lapangan |
| `risk_slow_threshold` | 0.45 | baseline-default | masih bisa berubah setelah data hardware/air |
| `risk_turn_slow_threshold` | 0.55 | baseline-default | sama |
| `risk_turn_threshold` | 0.75 | baseline-default | sama |
| `risk_stop_threshold` | 0.92 | baseline-default | sama |
| `risk_ema_alpha` | 0.35 | baseline-default | sah sebagai default aktif, belum final absolut |
| `vq_risk_floor` | 0.80 | baseline-default | sah sebagai default aktif |

Prinsip:
- parameter ini ditentukan dari perilaku distribusi risk,
- bukan langsung dari dimensi kapal,
- sehingga finalisasi harus berbasis data run yang cukup.

---

## 5. vTTC / Timing Safety Thresholds

Parameter berikut harus dianggap **not-final-yet** sampai data dinamika kapal nyata cukup.

| Parameter | Default aktif repo | Status | Catatan |
|---|---:|---|---|
| `vttc_turn_threshold_s` | 4.0 s | not-final-yet | butuh validasi terhadap turning capability nyata |
| `vttc_stop_threshold_s` | 1.2 s | not-final-yet | butuh validasi terhadap stopping / avoidance authority |
| `min_cmd_hold_s` | 0.6 s | baseline-default | masih runtime-tunable |

---

## 6. Takeover / Differential-Thrust Control Parameters

### 6.1 Baseline simulasi (`phase5`)

| Parameter | Default aktif repo | Status | Catatan |
|---|---:|---|---|
| `cruise_speed` | 0.30 | baseline-default | sah untuk simulation baseline |
| `turn_cmd` | 0.55 | baseline-default | sah untuk simulation baseline |
| `diff_mix_gain` | 0.70 | baseline-default | sah untuk simulation baseline |

### 6.2 Baseline hardware (`phase7`)

| Parameter | Default aktif repo | Status | Catatan |
|---|---:|---|---|
| `cruise_speed` | 0.30 | baseline-default | masih bisa dituning lapangan |
| `slow_factor` | 0.55 | baseline-default | masih bisa dituning |
| `turn_speed_factor` | 0.75 | baseline-default | masih bisa dituning |
| `turn_cmd` | 0.50 | baseline-default | masih bisa dituning |
| `diff_mix_gain` | 0.65 | baseline-default | masih bisa dituning |
| `speed_max` | 0.55 | baseline-default | masih bisa dituning |

Status keseluruhan kelompok ini:
- boleh dipakai sebagai runtime baseline saat ini,
- belum final sebagai nilai dinamik kapal sampai data uji yaw response dan uji air cukup.

---

## 7. Bridge / PWM / Output Mapping

Parameter berikut adalah **baseline integration defaults**, bukan hasil final kalibrasi lapangan.

| Parameter | Default aktif repo | Status | Catatan |
|---|---:|---|---|
| `pwm_neutral` | 1500 | baseline-default | standar netral RC/PWM |
| `pwm_fwd_max` | 1900 | baseline-default | belum final bila ESC/thruster berubah |
| `pwm_rev_min` | 1100 | baseline-default | relevan bila reverse diaktifkan |
| `pwm_output_min` | 1000 | baseline-default | safety clamp |
| `pwm_output_max` | 2000 | baseline-default | safety clamp |
| `bridge_pub_hz` | 20.0 | baseline-default | cukup stabil untuk bench |
| `bridge_command_timeout_s` | 0.5 | baseline-default | sah untuk baseline saat ini |

Catatan:
- untuk sidang/laporan, parameter ini sebaiknya dipresentasikan sebagai integration-safe defaults,
- bukan sebagai hasil tuning final propulsion system.

---

## 8. Camera Runtime Defaults

| Parameter | Default aktif repo | Status | Catatan |
|---|---:|---|---|
| `ca_camera_device_width` | 640 | baseline-default | runtime practical default |
| `ca_camera_device_height` | 480 | baseline-default | runtime practical default |
| `ca_camera_device_fps` | 30 | baseline-default | runtime default |
| `ca_camera_max_fps` | 15.0 | baseline-default | publish/runtime throttle |
| `ca_det_imgsz` | 416 | baseline-default | detector runtime default |
| `ca_det_conf` | 0.20 | baseline-default | detector runtime default |
| `ca_det_iou` | 0.45 | baseline-default | detector runtime default |
| `ca_det_max_fps` | 8.0 | baseline-default | detector runtime practical default |

Catatan:
- angka ini sah sebagai operational defaults,
- tidak harus disebut sebagai angka desain final dalam laporan kecuali memang dipakai pada result utama yang dilaporkan.

---

## 9. Locking Rule untuk Repo dan Laporan

Gunakan aturan berikut:

### Boleh ditulis sebagai baseline resmi saat ini
- dimensi kapal Alfin7
- tinggi kamera final
- resolusi citra final
- pusat citra nominal
- protected corridor minimum
- active launch baseline (`phase5`, `phase7`)

### Boleh ditulis sebagai default runtime aktif, tetapi jangan disebut final absolut
- threshold risk
- threshold vTTC
- speed/turn/mix parameters
- PWM tuning
- timeout tuning
- detector runtime defaults

### Jangan diklaim final tanpa data tambahan
- `camera_hfov_deg`
- `camera_vfov_deg`
- `f_x`
- `f_y`
- `center_band_ratio`
- `bottom_danger_ratio`
- `near_area_ratio`
- final dynamic thresholds
- final hardware control gains
- final on-water safe envelope

---

## 10. Kesimpulan Singkat

Parameter thesis baseline saat ini harus dibaca seperti ini:
- geometri platform Alfin7 dan resolusi citra sudah cukup kuat untuk dikunci sebagai baseline design values,
- parameter visual berbasis FOV dan kalibrasi kamera masih berstatus not-final-yet,
- runtime thresholds dan dynamic control parameters masih berstatus baseline-default / tunable,
- repo dan laporan harus membedakan dengan tegas antara design-lock dan runtime-tuning.
