# FIELD TEST EVIDENCE CHECKLIST
## SEANO Collision Avoidance — Checklist evidence uji dockside dan uji air

Dokumen ini dibuat agar setiap uji lapangan menghasilkan evidence yang cukup kuat untuk:

- analisis teknis,
- penulisan laporan TA,
- seminar/sidang,
- dan penyusunan jurnal.

Tujuan utamanya bukan membuat uji lebih rumit, tetapi mencegah situasi seperti:

- run sudah dilakukan tetapi tidak ada rosbag,
- obstacle muncul tetapi tidak ada screenshot HUD,
- kapal merespons tetapi tidak ada catatan mode/event,
- hasil uji tidak bisa dipakai sebagai evidence formal.

---

## 1. Sebelum run

### 1.1 Informasi run

Sebelum menjalankan uji, catat:
- tanggal
- lokasi
- tipe run
- nama operator
- nama observer/pencatat
- nama bag / nama log
- branch / commit repo yang dipakai

Contoh minimum:

```text
Tanggal: 2026-__-__
Lokasi: ________
Run type: dockside / AUTO_no_obstacle / obstacle_run / rejoin_run
Operator: ________
Observer: ________
Bag name: phase7_fieldtest_01
Git branch: ________
Git commit: ________
```

### 1.2 Kondisi sistem

Pastikan tercatat:
- launch yang dipakai
- profile runtime yang dipakai
- device kamera yang dipakai
- FCU URL yang dipakai
- apakah rosbag aktif
- apakah web video server aktif

### 1.3 Verifikasi awal wajib

Sebelum kapal dilepas:
- `/mavros/state` terhubung
- raw image aktif
- annotated image aktif
- HUD aktif
- `/ca/risk` keluar
- `/ca/command` keluar
- `/ca/watchdog_status` sehat

Jika salah satu belum sehat, run jangan dianggap evidence utama.

---

## 2. Saat run berlangsung

### 2.1 Visual evidence wajib

Minimal ambil:
- video raw camera atau screenshot raw,
- video/screenshot annotated image,
- video/screenshot `/ca/debug_image`,
- dokumentasi obstacle dari sudut luar kapal,
- dokumentasi posisi obstacle relatif terhadap jalur mission.

### 2.2 Runtime evidence wajib

Minimal salah satu dari ini harus ada:
- rosbag,
- terminal log,
- capture topik penting,
- rekaman Mission Planner / GCS bila relevan.

### 2.3 Topic yang paling penting dipantau

- `/ca/risk`
- `/ca/command`
- `/ca/command_safe`
- `/ca/failsafe_active`
- `/seano/rc_override_enable`
- `/mavros/state`
- `/ca/mode_manager_state`
- `/ca/mode_manager_event`

---

## 3. Setelah run selesai

### 3.1 Catatan hasil run

Untuk setiap run, isi minimal:
- obstacle ada / tidak ada
- detector melihat / tidak melihat
- risk naik / tidak
- command berubah / tidak
- override aktif / tidak
- kapal avoid / tidak
- obstacle clear / tidak
- release terjadi / tidak
- rejoin terjadi / tidak
- mission lanjut / tidak
- run dihentikan / tidak
- catatan anomali

Contoh format singkat:

```text
Obstacle visible: yes/no
Detection valid: yes/no
Risk rise observed: yes/no
Command change observed: yes/no
Override active: yes/no
Avoid motion observed: yes/no
Release observed: yes/no
Rejoin observed: yes/no
Mission resumed: yes/no
Anomaly: ________
```

### 3.2 Klasifikasi hasil

Setiap run harus diberi label salah satu:

- **Bench OK**
- **Partial field response**
- **Avoid observed**
- **Avoid + release observed**
- **Avoid + release + rejoin observed**
- **Mission-resume observed**
- **Invalid evidence**

Ini penting agar nanti tidak semua run dianggap setara.

---

## 4. Checklist evidence minimal per jenis run

## 4.1 Dockside validation

Harus ada:
- raw image
- annotated image
- HUD
- `/mavros/state`
- status watchdog
- catatan apakah semua node aktif

## 4.2 AUTO tanpa obstacle

Harus ada:
- bukti mission normal berjalan
- bukti obstacle memang tidak sengaja memicu avoid
- bukti state tetap stabil
- bukti command CA tidak salah aktif terus-menerus

## 4.3 Obstacle run

Harus ada:
- obstacle terlihat kamera
- annotated image menunjukkan obstacle atau alasan mengapa tidak terdeteksi
- risk naik
- command berubah
- respons kapal diamati

## 4.4 Release / rejoin run

Harus ada:
- obstacle clear atau obstacle keluar dari kondisi bahaya
- override turun
- `REJOIN_START`
- `REJOIN_DONE` atau alasan timeout/cancelled
- bukti mission dilanjutkan atau gagal dilanjutkan

---

## 5. Evidence yang paling kuat untuk laporan TA

Evidence paling kuat biasanya kombinasi berikut:

1. screenshot raw image,
2. screenshot annotated image,
3. screenshot HUD,
4. potongan rosbag / log topic,
5. event `TAKEOVER_ON`, `TAKEOVER_OFF`, `REJOIN_START`, `REJOIN_DONE`,
6. dokumentasi visual posisi obstacle di dunia nyata,
7. tabel ringkas hasil beberapa run.

Jika tujuh elemen ini tersedia, satu run biasanya sudah cukup kuat untuk dibahas di laporan.

---

## 6. Hal yang harus dihindari

Jangan anggap run sebagai evidence utama jika:
- kamera sempat mati/stale tapi tidak dicatat,
- HUD tidak terekam,
- obstacle tidak terdokumentasi,
- rosbag tidak ada sama sekali dan tidak ada capture topic lain,
- tidak jelas branch/commit mana yang dipakai,
- operator tidak mencatat apakah kapal benar-benar avoid atau hanya drift biasa.

---

## 7. Ringkasan

Setiap run lapangan sebaiknya menghasilkan empat jenis evidence:

1. **visual evidence**
2. **runtime evidence**
3. **operator note**
4. **result classification**

Kalau empat hal ini selalu dikumpulkan, maka hasil uji lapangan akan jauh lebih mudah dipakai untuk laporan, sidang, dan jurnal.
