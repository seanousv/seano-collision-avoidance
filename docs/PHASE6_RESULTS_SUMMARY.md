# Phase 6 Results Summary

Dokumen ini merangkum hasil uji valid Phase 6 untuk baseline simulasi terbaru SEANO collision avoidance.

## 1. Tujuan

Tujuan dokumen ini:
- mengunci run valid yang dipakai sebagai hasil utama,
- memisahkan run valid vs run gagal / tidak dipakai,
- merangkum metrik utama agar siap dipakai pada pembahasan TA.

---

## 2. Run yang Dipakai sebagai Hasil Utama

Run berikut dipakai sebagai hasil utama karena valid dan konsisten:

1. `phase6_rejoin_run_01`
2. `phase6_rejoin_run_02`
3. `phase6_failsafe_run_03`

Run berikut **tidak** dipakai sebagai hasil utama:
- `phase6_failsafe_run_02` → run gagal / bag tidak valid
- `phase6_run_01` → bukan run standar Phase 6 final

---

## 3. Ringkasan Hasil Valid

| Bag | Scenario | Profile | Takeover | Reaction Mean (s) | Release Mean (s) | Rejoin Mean (s) | Rejoin Done | Cancel | Timeout | Failsafe Rises | Mismatch Ratio | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| phase6_rejoin_run_01 | hazard_rejoin | synthetic_light | 4 | 0.055 | 0.873 | 2.343 | 4 | 0 | 0 | 0 | 0.000 | VALID |
| phase6_rejoin_run_02 | hazard_rejoin | synthetic_light | 2 | 0.065 | 0.871 | 2.121 | 2 | 0 | 0 | 0 | 0.000 | VALID |
| phase6_failsafe_run_03 | failsafe | synthetic_watchdog | 0 | 0.000 | 0.000 | 2.178 | 3 | 0 | 0 | 3 | 0.000 | VALID |

Catatan:
- Untuk run failsafe, `reaction` dan `release` bernilai 0 karena skenario ini memang tidak berbasis hazard command `/ca/command_safe`.
- Untuk run hazard/rejoin, `failsafe_rises` bernilai 0 karena run tersebut memang fokus pada takeover dan rejoin normal.

---

## 4. Interpretasi Hasil

## 4.1 Hazard / Rejoin

Dari dua run valid hazard/rejoin:

- reaction time berada di kisaran **0.055–0.065 s**
- release time berada di kisaran **0.871–0.873 s**
- rejoin time berada di kisaran **2.121–2.343 s**
- semua run menunjukkan:
  - `rejoin.done > 0`
  - `rejoin.cancelled = 0`
  - `rejoin.timeouts = 0`
  - `mode_mismatch_ratio = 0.000`

Maknanya:
- sistem dapat merespons hazard dengan cepat,
- dapat melepas takeover secara konsisten,
- dapat kembali ke mission melalui state `REJOIN`,
- tidak ditemukan mismatch mode saat state mission aktif pada run valid.

## 4.2 Failsafe / Recovery

Dari satu run valid failsafe:

- `failsafe_rises = 3`
- `failsafe_falls = 3`
- `rejoin.done = 3`
- `rejoin.timeouts = 0`
- `rejoin_time_mean = 2.178 s`
- `mode_mismatch_ratio = 0.000`

Maknanya:
- transisi ke FAILSAFE berhasil terdeteksi,
- pemulihan dari failsafe berhasil,
- setelah failsafe clear, sistem dapat kembali masuk `REJOIN` lalu pulih ke mission.

---

## 5. Nilai Rata-Rata yang Bisa Dipakai untuk Pembahasan

## 5.1 Hazard / Rejoin (berdasarkan run valid)

### Reaction time
\[
\frac{0.055 + 0.065}{2} = 0.060 \text{ s}
\]

Rata-rata reaction time valid:
- **0.060 s**

### Release time
\[
\frac{0.873 + 0.871}{2} = 0.872 \text{ s}
\]

Rata-rata release time valid:
- **0.872 s**

### Rejoin time (hazard/rejoin)
\[
\frac{2.343 + 2.121}{2} = 2.232 \text{ s}
\]

Rata-rata rejoin time valid pada skenario hazard/rejoin:
- **2.232 s**

## 5.2 Failsafe / Recovery

Rejoin time valid pada skenario failsafe:
- **2.178 s**

Failsafe rises valid:
- **3**

---

## 6. Kesimpulan Sementara Phase 6

Berdasarkan run valid yang tersedia saat ini:

1. Sistem collision avoidance berhasil melakukan takeover pada skenario hazard.
2. Sistem berhasil melepas takeover setelah kondisi aman.
3. Sistem berhasil masuk ke state `REJOIN` dan kembali ke mission tanpa timeout pada run valid.
4. Sistem berhasil masuk ke FAILSAFE dan pulih kembali pada run failsafe valid.
5. Pada run valid, mismatch mode mission terhadap MAVROS tidak ditemukan (`0.000`).

Dengan demikian, baseline simulasi saat ini sudah mendukung bukti kuantitatif awal untuk:
- hazard response,
- release,
- rejoin,
- failsafe handling,
- mission-mode consistency.

---

## 7. Run yang Tidak Dipakai

## 7.1 `phase6_failsafe_run_02`
Alasan tidak dipakai:
- bag tidak memuat topic inti yang dibutuhkan untuk metrik failsafe secara valid
- hasil metrik tidak representatif

## 7.2 `phase6_run_01`
Alasan tidak dipakai:
- bukan run standar final pada baseline Phase 6 saat ini
- hanya disimpan sebagai log pengembangan

---

## 8. Rekomendasi Berikutnya

Agar hasil Phase 6 lebih kuat, target minimum berikutnya:

1. tambah **1 run failsafe valid** lagi
2. opsional tambah **1 run hazard/rejoin valid** lagi
3. setelah itu, gunakan hanya run valid untuk tabel akhir TA

Prioritas:
- `phase6_failsafe_run_04`
- opsional `phase6_rejoin_run_03`

---

## 9. Kalimat Ringkas untuk Laporan TA

Contoh narasi yang bisa dipakai:

> Pada pengujian simulasi, sistem collision avoidance berhasil melakukan avoidance takeover dan kembali ke mission melalui state REJOIN. Pada dua run hazard/rejoin yang valid, waktu reaksi rata-rata sistem berada di sekitar 0.060 s, waktu release rata-rata sekitar 0.872 s, dan waktu rejoin rata-rata sekitar 2.232 s. Pada skenario failsafe yang valid, sistem berhasil mendeteksi dan menangani 3 kejadian failsafe serta kembali ke mission dengan waktu rejoin rata-rata 2.178 s. Pada seluruh run valid, tidak ditemukan mismatch antara state mission manager dan mode MAVROS.

---

## 10. Status Dokumen

Dokumen ini adalah ringkasan hasil valid sementara dan harus diperbarui jika:
- ada run valid baru,
- ada perbaikan extractor/agregator,
- atau ada perubahan baseline runtime.
