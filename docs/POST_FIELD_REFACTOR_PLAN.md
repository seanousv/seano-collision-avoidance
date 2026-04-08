# POST-FIELD REFACTOR PLAN
## SEANO Collision Avoidance — Rencana refactor setelah baseline lapangan terkunci

Dokumen ini bukan instruksi untuk refactor sekarang.

Justru tujuan dokumen ini adalah memastikan repo **tidak direfactor terlalu cepat** sebelum:

- baseline simulasi final jelas,
- baseline hardware final jelas,
- field evidence utama sudah terkumpul,
- dan nama file aktif sudah konsisten dipakai di laporan/slide/jurnal.

---

## 1. Prinsip utama

Sebelum field validation final terkunci:

- jangan rename launch aktif,
- jangan pecah struktur folder besar-besaran,
- jangan ganti nama file yang sudah disebut di laporan,
- jangan refactor hanya demi estetika.

Setelah baseline benar-benar terkunci, barulah refactor dilakukan secara terkendali.

---

## 2. Kandidat refactor paling penting nanti

## 2.1 Naming node takeover manager

Kandidat:
- `auto_controller_stub_node.py`

Alasan:
- nama `stub` terlalu lemah untuk runtime component yang sebenarnya aktif,
- isi file saat ini sudah bertindak sebagai takeover manager nyata.

Arah refactor setelah baseline lock:
- rename ke nama yang lebih representatif,
- pertahankan alias entrypoint sementara agar backward-compatible,
- perbarui README, launch, dan laporan secara serempak.

## 2.2 Pemisahan node yang terlalu padat

Kandidat:
- `risk_evaluator_node.py`

Alasan:
- file ini sudah menangani beberapa tanggung jawab sekaligus:
  - parsing detection,
  - tracking,
  - risk scoring,
  - state perception,
  - VQ/freeze integration,
  - HUD rendering.

Arah refactor setelah baseline lock:
- pisahkan risk core,
- pisahkan health/perception state,
- pisahkan HUD renderer.

Catatan:
- ini refactor kualitas struktur, bukan prioritas sebelum field evidence utama selesai.

## 2.3 Arsip launch legacy

Kandidat:
- launch yang sekarang masuk kelompok bench/debug/legacy.

Arah refactor:
- pindahkan ke folder `launch/legacy/` atau `launch/archive/` bila struktur package mengizinkan,
- atau minimal beri penamaan status yang lebih tegas di dokumentasi.

## 2.4 Parameter centralization

Arah refactor:
- parameter final yang benar-benar resmi dapat dipusatkan ke file YAML atau tabel konfigurasi yang lebih rapi,
- saat ini masih wajar banyak parameter tersebar di launch selama baseline aktif belum dikunci penuh.

---

## 3. Syarat sebelum refactor dimulai

Refactor besar baru layak dilakukan jika semua syarat ini sudah terpenuhi:

1. nama launch aktif final sudah diputuskan,
2. parameter aktif final sudah dikunci,
3. evidence simulasi final sudah stabil,
4. evidence hardware/field utama sudah terkumpul,
5. laporan TA tidak lagi berubah struktur besar,
6. slide seminar/sidang tidak lagi bergantung pada nama file lama.

---

## 4. Urutan refactor yang disarankan

Jika nanti refactor dilakukan, urutannya sebaiknya:

1. dokumentasikan baseline final dulu,
2. buat branch refactor terpisah,
3. ubah naming dan struktur secara kecil bertahap,
4. uji ulang `phase5` dan `phase7`,
5. pastikan topic aktif dan bag evidence tetap kompatibel,
6. baru rapikan file legacy.

---

## 5. Hal yang jangan dilakukan nanti

Bahkan setelah baseline final, tetap hindari:

- rename banyak file sekaligus tanpa branch khusus,
- ubah nama topic aktif tanpa peta migrasi,
- refactor node inti sambil sekaligus mengubah behavior kontrol,
- membersihkan legacy sebelum memastikan tidak ada ketergantungan laporan/jurnal ke file itu.

---

## 6. Ringkasan

Repo SEANO memang layak dirapikan lebih jauh, tetapi **waktu refactor yang tepat adalah setelah baseline lapangan final sudah terkunci**.

Sampai titik itu tercapai, strategi terbaik adalah:
- perkuat dokumentasi,
- perjelas baseline aktif,
- kumpulkan evidence,
- dan jaga stabilitas file aktif.
