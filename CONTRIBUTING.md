```md
# Kontribusi Repo SEANO Collision Avoidance

Repo ini dipakai tim SEANO untuk mengembangkan **Collision Avoidance (vision-only)**.
Mohon ikuti aturan berikut agar kerja bareng rapi dan tidak tabrakan.

> **Scope repo ini:** hanya modul CA (deteksi, tracking, risk assessment, keputusan manuver).
> Modul lain (dashboard, komunikasi, data logging, path planning, dll) **jangan** dimasukkan ke repo ini.

---

## 1) Identitas Git (WAJIB)

Semua anggota menggunakan email tim:
- Email: `seanousv@gmail.com`

Nama dibedakan per orang agar histori commit jelas, format:
- `SEANO | Naufal`
- `SEANO | Budi`
- `SEANO | Rani`

Set di dalam folder repo (supaya cuma berlaku untuk repo ini):
```bash
git config user.name "SEANO | NAMA_KAMU"
git config user.email "seanousv@gmail.com"

Di repo (WSL/Ubuntu):

```bash
git status
git add README.md CONTRIBUTING.md
git commit -m "docs: improve README and contributing"
git push