# Kontribusi Repo SEANO Collision Avoidance

Repo ini dipakai tim SEANO. Mohon ikuti aturan berikut agar kerja bareng rapi dan tidak tabrakan.

## 1) Identitas Git (wajib)
Semua anggota menggunakan email tim:
- Email: `seanousv@gmail.com`

Nama dibuat beda per orang agar histori commit jelas, contoh:
- `SEANO | Naufal`
- `SEANO | Budi`
- `SEANO | Rani`

Cek identitas (di folder repo):
```bash
git config user.name
git config user.email

## Scope repo ini
Repo ini khusus untuk modul **collision avoidance berbasis kamera (vision-only)**.
Fokus lain seperti dashboard, komunikasi, data logging, dan path planning dikerjakan di repo terpisah.
Mohon jangan memasukkan kode di luar scope CA ke repo ini.

git config user.name "SEANO | Haidar"
git config user.email "seanousv@gmail.com"

git checkout -b Haidar/collision-avoidance
docs: add contributing guide
git add .
git commit -m "docs: add contributing guide"
git push -u origin Haidar/collision-avoidance
