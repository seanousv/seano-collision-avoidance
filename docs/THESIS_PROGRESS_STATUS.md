# THESIS PROGRESS STATUS

## SEANO Collision Avoidance - Current Thesis Baseline

Dokumen ini merangkum posisi proyek TA SEANO collision avoidance berdasarkan baseline repo aktif, hasil simulasi SITL, dan hasil hardware bench terbaru.

Dokumen ini dipakai untuk:
- menyelaraskan repo dengan progres TA saat ini,
- membantu pembacaan repo saat sidang atau penulisan laporan,
- membedakan apa yang sudah terbukti di simulasi,
- membedakan apa yang sudah tervalidasi di hardware bench,
- menegaskan batas klaim yang boleh dan belum boleh dibuat.

---

## 1. Posisi Proyek Saat Ini

Repo ini saat ini harus dibaca sebagai **baseline aktif TA**, bukan hanya workspace eksperimen umum.

Dua baseline aktif yang sedang dipakai adalah:
- **Simulation baseline** berpusat di `phase5_mission_avoid_integration.launch.py`
- **Hardware baseline** berpusat di `phase7_cuav_usb_hardware.launch.py`

Interpretasi praktisnya:
- simulasi menjadi sumber evidence kuantitatif utama,
- hardware bench menjadi sumber evidence integrasi nyata,
- field validation tetap dilakukan bertahap dan belum boleh dibaca sebagai selesai penuh sebelum evidence lapangan benar-benar cukup.

---

## 2. Evidence Simulasi yang Sudah Kuat

Simulation baseline saat ini sudah menunjukkan evidence end-to-end untuk:
- mission normal berjalan,
- obstacle memicu risk dan takeover,
- avoidance aktif sementara,
- release kembali ke mission,
- state `REJOIN` dapat diamati,
- recovery dari repeated hazard dan beberapa skenario failsafe sudah dievaluasi.

Struktur hasil yang paling penting:
- clean avoid-rejoin valid pada beberapa run utama,
- repeated hazard sudah diuji dan tetap recover,
- failsafe activation dan failsafe recovery sudah tervalidasi,
- rosbag metrics extraction sudah aktif untuk reaction, release, rejoin, dan mode mismatch.

Makna akademik:
- baseline simulasi sudah cukup kuat untuk menjadi evidence kuantitatif utama TA,
- tetapi tetap harus dibatasi jujur sebagai domain SITL, bukan pembuktian air final.

---

## 3. Evidence Hardware Bench yang Sudah Kuat

Hardware baseline saat ini sudah menunjukkan evidence integrasi nyata pada level bench untuk:
- Jetson runtime,
- CUAV X7+ melalui MAVROS,
- USB camera sebagai source persepsi,
- detector, risk evaluator, dan watchdog,
- RC override chain dan limiter,
- monitoring melalui browser / HUD.

Evidence bench paling penting yang sudah bisa dibaca kuat:
- pipeline Jetson dari image sampai command sudah hidup,
- full-chain Jetson + FCU + MAVROS sudah tervalidasi,
- integrasi Mission Planner / monitoring juga sudah tervalidasi berulang.

Makna akademik:
- proyek sudah melewati proof-of-concept simulasi murni,
- repo saat ini sudah berada pada fase hardware bench fungsional,
- tetapi belum boleh mengklaim field success end-to-end penuh sebelum uji air yang lebih lengkap.

---

## 4. Klaim yang Sudah Boleh Dilakukan

Klaim yang sudah cukup aman:
- arsitektur mission-aware collision avoidance sudah berjalan di simulasi,
- state `MISSION -> AVOID -> REJOIN -> MISSION` sudah tervalidasi di baseline SITL,
- evaluator risiko, watchdog, mode manager, limiter, dan RC bridge sudah terintegrasi,
- baseline hardware Jetson + CUAV + USB camera sudah hidup dan tervalidasi pada level bench,
- monitoring raw / annotated / HUD sudah aktif dan bisa dipakai untuk diagnosis.

---

## 5. Klaim yang Belum Boleh Dibuat Final

Klaim berikut belum boleh dianggap final:
- avoidance di air sudah konsisten dan matang di semua kondisi,
- rejoin di air sudah selalu berhasil,
- mission complete dengan avoidance sudah tervalidasi penuh di lapangan,
- performa deteksi hardware sudah final terhadap semua lighting / obstacle condition.

---

## 6. Batasan Saat Ini

Batasan yang masih harus dibaca aktif:
- `REJOIN` masih berorientasi mission resume / mode restore, belum full path replanning,
- perception hardware masih sensitif terhadap cahaya, kontras obstacle, dan kestabilan stream,
- beberapa parameter dinamik kapal belum boleh dianggap benar-benar final tanpa data uji lapangan yang lebih kuat,
- struktur repo masih menyimpan sejumlah launch legacy / audit-later yang belum dibersihkan karena baseline aktif masih diprioritaskan.

---

## 7. Prioritas Berikutnya

Prioritas teknis berikut yang paling konsisten dengan repo saat ini:
1. menjaga baseline simulasi tetap stabil sebagai sumber evidence kuantitatif utama,
2. menjaga baseline hardware `phase7` tetap stabil untuk bench dan dockside work,
3. memperkuat evidence lapangan bertahap dengan skenario yang sederhana dan aman,
4. menyinkronkan laporan, slide, dan repo supaya pembaca luar tidak salah membaca status proyek,
5. menunda refactor besar sampai baseline akhir benar-benar terkunci.

---

## 8. Ringkasan

Ringkasan status repo saat ini:
- ini adalah repo baseline aktif TA,
- simulasi sudah kuat untuk evidence kuantitatif,
- hardware bench sudah kuat untuk evidence integrasi nyata,
- field validation masih bertahap,
- dokumentasi repo harus membantu pembaca membedakan antara **sudah tervalidasi**, **sedang diintegrasikan**, dan **belum final**.
