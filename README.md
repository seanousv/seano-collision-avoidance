# SEANO Collision Avoidance (ROS2 Humble) — USV Differential Thruster

Repo ini berisi modul **collision avoidance berbasis kamera** untuk USV SEANO, dibangun di atas **ROS 2 Humble**. Fokus utama repo: pipeline **perception → decision → safety → actuation** yang bisa diuji dulu di simulasi (**ArduPilot SITL ArduRover rover-skid**) dan nantinya dipindahkan ke hardware (**Jetson Orin Nano + CUAV X7+**).

Catatan desain penting:
- USV SEANO memakai **differential thruster (kiri–kanan)** tanpa rudder.
- Karena itu kontrol internal yang diprioritaskan adalah **left_cmd / right_cmd**, bukan “rudder”.
- Untuk simulasi, dipakai **ArduRover rover-skid** agar perilaku skid steering/differential lebih relevan.

---

## Struktur Repo


seano-collision-avoidance/
├── seano_ca_ws/
│ └── src/
│ └── seano_vision/ # package ROS2 utama (Python)
│ ├── seano_vision/ # node-node Python
│ ├── launch/ # launch files
│ ├── config/ # config kamera/param (jika ada)
│ ├── package.xml
│ ├── setup.py
│ └── setup.cfg
├── tools/ # opsional (mis. ardupilot dipisah)
├── requirements.txt # dependency AI (WSL aman; Jetson torch manual)
├── LICENSE
└── README.md


## Prasyarat (Development / Simulation)

Host:
- Windows + WSL2 Ubuntu 22.04

WSL:
- ROS 2 Humble
- MAVROS2 (ROS2 Humble)
- OpenCV + cv_bridge

Simulator/GCS:
- ArduPilot SITL (ArduRover + rover-skid)
- Mission Planner (Windows)

---

## Build ROS2 Workspace (WSL)

1) Clone repo:
```bash
git clone https://github.com/seanousv/seano-collision-avoidance.git
cd seano-collision-avoidance

Install paket yang umum dipakai (minimal):

sudo apt update
sudo apt install -y python3-opencv ros-humble-cv-bridge
sudo apt install -y ros-humble-launch ros-humble-launch-ros
sudo apt install -y ros-humble-mavros ros-humble-mavros-msgs
sudo apt install -y ros-humble-vision-msgs

Build:

cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

Patokan benar:

colcon build selesai tanpa error.

ros2 pkg list | grep seano_vision menampilkan package seano_vision.

Stack Kontrol SEANO (Konsep)

Tujuan stack ini: jalur kontrol deterministik dan aman.

Input manual: keyboard teleop → /seano/manual/left_cmd, /seano/manual/right_cmd

Input auto: dari AI/risk/planner → /seano/auto/left_cmd, /seano/auto/right_cmd

MUX: pilih manual vs auto → /seano/selected/*

Safety limiter: timeout + failsafe + clamp → /seano/left_cmd, /seano/right_cmd

Bridge: left/right → PWM RC override → /mavros/rc/override

Jalur ini sengaja dibuat “industri style”: input → mux → limiter → bridge → autopilot.

AI Environment (WSL x86_64)

Gunakan venv terpisah agar tidak mengganggu ROS2.

cd ~/seano-collision-avoidance

# Jika pernah error sebelumnya dan mau ulang bersih:
# rm -rf .venv_ai

python3 -m venv .venv_ai
source .venv_ai/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

# Test (patokan: TORCH_OK + MODEL_OK)
python -c "import torch; print('TORCH_OK', torch.__version__)"
python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print('MODEL_OK')"

Patokan benar:

Muncul TORCH_OK ...

Muncul MODEL_OK (download model boleh terjadi sekali)

Catatan:

Jangan jalankan install AI kalau storage WSL/Windows mepet. Torch bisa besar.

AI Environment (Jetson / aarch64) — Catatan Penting

requirements.txt sengaja dibuat agar tidak memaksa install torch via pip pada Jetson.
Urutan yang benar di Jetson:

Install PyTorch/torchvision dari rekomendasi JetPack/NVIDIA (bukan pip install torch biasa).

Baru install ultralytics dan dependency lainnya.

WSL2 Runbook (SITL + Mission Planner + MAVROS + SEANO)

Bagian ini yang paling penting untuk FASE 0: supaya demo/re-run selalu sama.

Peta Port Baseline

14550/UDP : Mission Planner (Windows) menerima MAVLink

14551/UDP : jalur MAVROS ↔ SITL (WSL)

5760/TCP : master MAVProxy/ArduPilot (internal)

Catatan WSL2:

IP Windows host dari sisi WSL2 bisa berubah tiap sesi.

Cara yang konsisten: ambil default gateway di WSL.

Terminal 1 (WSL) — Start SITL Rover Skid + Out ke MP & MAVROS
cd ~/tools/ardupilot

WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
echo "WIN_HOST_IP=$WIN_HOST_IP"

sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551

Patokan benar:

SITL + MAVProxy console/map muncul.

Di output terlihat --out udp:...:14550 dan --out udp:127.0.0.1:14551.

Mission Planner (Windows) — Connect

Pilih koneksi UDP

Port 14550

Connect

Patokan benar:

Vehicle terlihat normal (mode, parameter, map).

Telemetry jalan.

Terminal 2 (WSL) — Start MAVROS2
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551

Patokan benar:

/mavros/state menjadi connected: true.

Terminal 3 (WSL) — Monitor State
source /opt/ros/humble/setup.bash
ros2 topic echo /mavros/state

Patokan benar:

connected: true

mode terbaca (mis. MANUAL)

armed true/false sesuai kondisi

Terminal 4 (WSL) — Jalankan Stack SEANO
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision run_auto_stack.launch.py

Patokan benar:

Node-node start tanpa error fatal.

Failsafe heartbeat publish stabil.

Checklist Validasi Cepat (Patokan “Sudah Benar”)

MAVROS terhubung:

ros2 topic echo /mavros/state
# connected: true

Failsafe heartbeat stabil:

ros2 topic hz /ca/failsafe_active
# rate stabil (mis. ~10 Hz), data biasanya False saat aman

RC override ada:

ros2 topic echo /mavros/rc/override
# channel PWM berubah saat teleop / auto command aktif

Vehicle benar-benar bergerak (Mission Planner map/track berubah).

Troubleshooting (Masalah Paling Umum)

A) Mission Planner “Connect Failed”

Pastikan SITL mengirim ke IP Windows host dari WSL:
WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')

Pastikan SITL punya --out udp:${WIN_HOST_IP}:14550

B) /mavros/state connected: false

Pastikan SITL jalan dulu.

Pastikan SITL out ke udp:127.0.0.1:14551.

Pastikan MAVROS pakai:
udp://0.0.0.0:14551@127.0.0.1:14551

Jika restart SITL, restart MAVROS (supaya endpoint tidak “nyangkut”).

C) Kendaraan tidak bergerak padahal PWM keluar

Pastikan vehicle ARMED.

Pastikan mode menerima RC override (umumnya MANUAL/STEERING saat uji).

Pastikan limiter tidak mengunci STOP (cek log FAILSAFE_STOP).

Pastikan tidak ada publisher ganda yang menimpa command (ros2 topic info -v ...).

D) Topic “local_setup.bash not found”

Biasanya karena workspace belum build, atau ada auto-source salah path di ~/.bashrc.

Pastikan yang di-source adalah:
~/seano-collision-avoidance/seano_ca_ws/install/setup.bash (setelah build)

Kontribusi / Catatan Praktik

Jangan commit build/ install/ log/ colcon.

Untuk demo, usahakan selalu pakai runbook 4 terminal di atas.

License

MIT (lihat file LICENSE).
