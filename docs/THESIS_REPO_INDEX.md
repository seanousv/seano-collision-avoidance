# THESIS REPO INDEX

Dokumen ini adalah **peta masuk utama** untuk membaca repository collision avoidance SEANO dari sudut pandang tugas akhir.

Tujuannya bukan menggantikan README, tetapi memberi urutan baca yang paling efisien untuk:
- pembimbing,
- penguji,
- rekan tim,
- dan diri sendiri saat mendekati sidang atau pengujian lapangan.

---

## 1. Jika ingin memahami repo ini dalam 5 menit

Baca urutan berikut:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/LAUNCH_STATUS_MAP.md`
4. `docs/ACTIVE_RUNTIME_FILE_MATRIX.md`
5. `docs/ACTIVE_TOPIC_FLOW.md`

Urutan ini cukup untuk memahami:
- baseline aktif simulasi,
- baseline aktif hardware,
- file runtime inti,
- topic utama,
- dan alur mission-aware collision avoidance secara menyeluruh.

---

## 2. Jika ingin memahami posisi proyek TA saat ini

Baca:

1. `docs/THESIS_PROGRESS_STATUS.md`
2. `docs/VALIDATION_BOUNDARY.md`
3. `docs/CLAIMS_AND_EVIDENCE_MATRIX.md`
4. `docs/HARDWARE_BENCH_RESULTS.md`

Dokumen-dokumen ini menjawab:
- apa yang sudah cukup kuat untuk diklaim,
- apa yang masih sebatas evidence simulasi,
- apa yang sudah tervalidasi di bench hardware,
- dan apa yang masih menunggu pembuktian lapangan terkontrol.

---

## 3. Jika ingin menjalankan sistem

Baca:

1. `README.md`
2. `docs/RUNBOOK.md`
3. `docs/LAUNCH_STATUS_MAP.md`
4. `docs/BASELINE_PARAMETER_LOCK.md`

Dokumen-dokumen ini dipakai untuk:
- memilih launch yang benar,
- membaca parameter aktif utama,
- menjalankan simulasi,
- menjalankan bench hardware,
- dan menjaga agar baseline tidak berubah sembarangan.

---

## 4. Jika ingin audit file aktif yang benar-benar relevan untuk TA

Baca:

1. `docs/ACTIVE_RUNTIME_FILE_MATRIX.md`
2. `docs/ACTIVE_TOPIC_FLOW.md`
3. `docs/ARCHITECTURE.md`

Dokumen ini membantu menjawab pertanyaan sidang seperti:
- file mana yang benar-benar aktif,
- node mana yang punya peran inti,
- bagaimana topik mengalir dari kamera sampai autopilot,
- dan bagaimana state machine dipakai untuk evidence generation.

---

## 5. Jika ingin menilai batas klaim secara akademik

Baca:

1. `docs/VALIDATION_BOUNDARY.md`
2. `docs/CLAIMS_AND_EVIDENCE_MATRIX.md`
3. `docs/HARDWARE_BENCH_RESULTS.md`
4. `docs/PHASE6_RESULTS_SUMMARY.md`

Tujuannya agar repo ini tidak dibaca terlalu optimistis.

Interpretasi yang benar saat ini:
- simulasi sudah kuat untuk evidence kuantitatif baseline,
- hardware bench sudah kuat untuk evidence integrasi,
- field success penuh tetap harus dibuktikan secara terkontrol.

---

## 6. Jika ingin menyiapkan pengujian lapangan

Baca:

1. `docs/FIELD_TEST_EVIDENCE_CHECKLIST.md`
2. `docs/BASELINE_PARAMETER_LOCK.md`
3. `docs/HARDWARE_BENCH_RESULTS.md`
4. `docs/ACTIVE_TOPIC_FLOW.md`

Fokusnya:
- apa yang harus direkam,
- screenshot apa yang harus diambil,
- topic apa yang harus dipantau,
- dan apa yang harus dianggap sebagai berhasil parsial atau berhasil penuh.

---

## 7. Jika ingin refactor repo setelah field baseline terkunci

Baca:

1. `docs/POST_FIELD_REFACTOR_PLAN.md`
2. `docs/ACTIVE_RUNTIME_FILE_MATRIX.md`
3. `docs/LAUNCH_STATUS_MAP.md`

Prinsip penting:
- refactor besar ditunda sampai baseline pengujian lapangan stabil,
- file launch aktif jangan di-rename sebelum evidence final terkunci,
- audit file lama dilakukan setelah kebutuhan TA utama aman.

---

## 8. Ringkasan satu kalimat per dokumen inti

- `README.md` -> ringkasan repo dan baseline aktif
- `docs/ARCHITECTURE.md` -> arsitektur formal sistem
- `docs/LAUNCH_STATUS_MAP.md` -> peta launch aktif vs bench vs audit-later
- `docs/RUNBOOK.md` -> langkah operasional menjalankan sistem
- `docs/THESIS_PROGRESS_STATUS.md` -> posisi proyek TA saat ini
- `docs/VALIDATION_BOUNDARY.md` -> batas klaim yang boleh dan tidak boleh
- `docs/CLAIMS_AND_EVIDENCE_MATRIX.md` -> peta klaim ke evidence
- `docs/BASELINE_PARAMETER_LOCK.md` -> parameter lock dan parameter yang masih tunable
- `docs/ACTIVE_RUNTIME_FILE_MATRIX.md` -> file runtime aktif untuk TA
- `docs/ACTIVE_TOPIC_FLOW.md` -> alur topic ROS utama
- `docs/HARDWARE_BENCH_RESULTS.md` -> status hardware bench
- `docs/FIELD_TEST_EVIDENCE_CHECKLIST.md` -> checklist evidence pengujian lapangan
- `docs/POST_FIELD_REFACTOR_PLAN.md` -> rencana refactor sesudah baseline lapangan stabil

---

## 9. Prinsip pembacaan repo ini

Repo ini saat ini harus dibaca sebagai:
- **baseline aktif TA**,
- dengan dua jalur utama:
  - simulasi untuk evidence kuantitatif,
  - hardware untuk evidence integrasi dan persiapan uji nyata.

Repo ini **bukan** lagi sekadar workspace eksperimen bebas.
