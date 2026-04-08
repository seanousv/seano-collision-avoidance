# VALIDATION BOUNDARY
## Batas Klaim yang Boleh dan Belum Boleh Dibuat dari Baseline Saat Ini

Dokumen ini dipakai agar interpretasi hasil tetap kuat, jujur, dan tidak overclaim.

---

## 1. Prinsip Umum

Repo ini saat ini memiliki dua baseline validasi utama:

1. **simulation baseline**
   untuk evidence kuantitatif perilaku sistem.

2. **hardware bench baseline**
   untuk evidence integrasi pada platform target.

Kedua baseline ini **sangat penting**, tetapi **tidak memiliki makna validasi yang sama**.

---

## 2. Dari Simulation Baseline, Boleh Diklaim

Boleh diklaim bahwa sistem:
- memiliki state machine mission-aware yang berjalan,
- dapat masuk ke avoidance saat hazard muncul,
- dapat melakukan release,
- dapat melakukan rejoin,
- dapat memvalidasi repeated hazard behavior,
- dapat memvalidasi failsafe activation,
- dapat memvalidasi failsafe recovery,
- menghasilkan metrik kuantitatif yang bisa dianalisis.

---

## 3. Dari Simulation Baseline, Belum Boleh Diklaim

Belum boleh diklaim bahwa:
- performa simulasi identik dengan lapangan,
- robustness simulasi otomatis sama dengan robustness kamera nyata di air,
- threshold simulasi otomatis final untuk hardware.

---

## 4. Dari Hardware Bench Baseline, Boleh Diklaim

Boleh diklaim bahwa:
- stack aktif pada Jetson,
- kamera USB aktif,
- detector aktif,
- risk evaluator aktif,
- watchdog/failsafe aktif,
- command chain aktif,
- MAVROS aktif,
- FCU nyata terhubung,
- Mission Planner dapat divalidasi di jalur bench,
- monitoring raw / annotated / HUD dapat dipakai,
- baseline hardware saat ini sudah cukup kuat untuk dibaca sebagai **integration-level evidence** pada platform target.

---

## 5. Dari Hardware Bench Baseline, Belum Boleh Diklaim

Belum boleh diklaim bahwa:
- obstacle avoidance lapangan sudah final,
- release + rejoin lapangan sudah konsisten,
- mission completion setelah real obstacle avoidance sudah tervalidasi penuh,
- tuning dinamik kapal sudah final,
- kepatuhan penuh terhadap seluruh ketentuan COLREGs sudah terbukti,
- sistem sudah siap diklaim sebagai full on-water operational deployment.

---

## 6. Kalimat Framing yang Disarankan

Gunakan framing seperti ini pada laporan/presentasi:

> Simulation baseline provides the primary quantitative behavioral evidence, while the hardware bench baseline provides integration-level evidence on the target platform.

Versi Indonesia yang aman:

> Baseline simulasi menjadi sumber utama bukti perilaku kuantitatif sistem, sedangkan baseline hardware bench menjadi sumber utama bukti integrasi pada platform target.

---

## 7. Kesimpulan

Kekuatan proyek saat ini ada pada:
- kualitas evidence simulasi,
- kualitas integrasi hardware bench,
- konsistensi arsitektur mission-aware.

Batasnya ada pada:
- controlled field validation yang belum lengkap,
- parameter dinamik kapal yang belum final,
- sensitivitas perception nyata terhadap lingkungan lapangan.

---

## Companion Document

For a compact defense-facing mapping from claim to evidence, read:

- `docs/CLAIMS_AND_EVIDENCE_MATRIX.md`

This companion document is intended to keep wording in the report, seminar slides, and journal draft aligned with the current evidence level.
