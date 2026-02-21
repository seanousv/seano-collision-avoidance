Berikut versi `README.md` yang lebih profesional, rapi, dan siap kamu **copy-paste utuh**. Ini tetap 1 file saja.

Ganti seluruh isi `README.md` (root repo) dengan teks di bawah:

```md
# SEANO Collision Avoidance (ROS 2 Humble) — USV Differential Thruster

This repository contains the **camera-based collision avoidance** module for the SEANO Unmanned Surface Vehicle (USV).  
The system is developed in **ROS 2 Humble** and validated first in simulation using **ArduPilot SITL (ArduRover rover-skid)**, then ported to the target hardware (**Jetson Orin Nano + CUAV X7+**).

Key vehicle constraint (critical for control design):
- SEANO uses **differential thrusters** (left–right) **without a rudder**.
- Therefore, the preferred internal command format is **left_cmd / right_cmd** (not rudder).

---

## Contents

- [Repository Structure](#repository-structure)
- [Roadmap (TA Phases)](#roadmap-ta-phases)
- [Development Environment](#development-environment)
- [Build (ROS 2 Workspace)](#build-ros-2-workspace)
- [Control Stack Overview](#control-stack-overview)
- [AI Environment (WSL x86_64)](#ai-environment-wsl-x86_64)
- [AI Environment (Jetson aarch64)](#ai-environment-jetson-aarch64)
- [WSL2 Simulation Runbook (SITL + Mission Planner + MAVROS + SEANO)](#wsl2-simulation-runbook-sitl--mission-planner--mavros--seano)
- [Validation Checklist](#validation-checklist)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Repository Structure

```

seano-collision-avoidance/
├── seano_ca_ws/
│   └── src/
│       └── seano_vision/                 # main ROS 2 package (Python)
│           ├── seano_vision/             # ROS nodes
│           ├── launch/                   # launch files
│           ├── config/                   # optional configs
│           ├── package.xml
│           ├── setup.py
│           └── setup.cfg
├── tools/                                # optional tools (e.g., ardupilot kept separate)
├── requirements.txt                      # AI deps (WSL safe; Jetson torch is manual)
├── LICENSE
└── README.md

````

---

## Roadmap (TA Phases)

- **FASE 0 — Baseline / Reproducible Setup**  
  Runbook, ports, dependencies, and repository hygiene. Must be repeatable for demo and migration.

- **FASE 1 — Mature USV Control (Priority)**  
  Reliable actuation behavior (forward/turn/stop), consistent over repeated runs, visible in Mission Planner.

- **FASE 2+ — Vision & Avoidance Integration**  
  Camera pipeline → detection → risk → avoidance command → return-to-path → evaluation → Jetson porting.

This README focuses on closing **FASE 0** and enabling **FASE 1** testing.

---

## Development Environment

- Host OS: Windows
- Linux: **WSL2 Ubuntu 22.04**
- ROS 2: **Humble**
- Autopilot sim: **ArduPilot SITL**
- Vehicle model: **ArduRover `rover-skid`** (skid steering ≈ differential thrust behavior)
- MAVLink bridge: **MAVROS2**
- GCS: **Mission Planner (Windows)**

---

## Build (ROS 2 Workspace)

1) Clone:
```bash
git clone https://github.com/seanousv/seano-collision-avoidance.git
cd seano-collision-avoidance
````

2. Install minimal dependencies (WSL):

```bash
sudo apt update
sudo apt install -y \
  python3-opencv \
  ros-humble-cv-bridge \
  ros-humble-launch \
  ros-humble-launch-ros \
  ros-humble-mavros \
  ros-humble-mavros-msgs \
  ros-humble-vision-msgs
```

3. Build:

```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Expected result:

* `colcon build` completes successfully.
* `ros2 pkg list | grep seano_vision` shows `seano_vision`.

---

## Control Stack Overview

The control pipeline is designed to be deterministic and safe:

1. Manual teleop (keyboard):

* Publishes:
  `/seano/manual/left_cmd` (Float32), `/seano/manual/right_cmd` (Float32)

2. Auto command source (from AI / risk / planner):

* Publishes:
  `/seano/auto/left_cmd`, `/seano/auto/right_cmd`
  and a toggle (commonly) `/seano/auto_enable` (Bool)

3. Command MUX (manual vs auto):

* Outputs: `/seano/selected/left_cmd`, `/seano/selected/right_cmd`

4. Safety limiter + failsafe:

* Applies timeout, clamp, and safe-stop policy
* Outputs final: `/seano/left_cmd`, `/seano/right_cmd`

5. MAVROS RC override bridge:

* Converts left/right into PWM RC override
* Publishes: `/mavros/rc/override`

This separation is intentional: AI only needs to generate motion intent, while the lower layer ensures safe actuation.

---

## AI Environment (WSL x86_64)

Use a dedicated Python venv (recommended). Run from repo root:

```bash
cd ~/seano-collision-avoidance

# if you need a clean reinstall:
# rm -rf .venv_ai

python3 -m venv .venv_ai
source .venv_ai/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

# verification (expected: TORCH_OK + MODEL_OK)
python -c "import torch; print('TORCH_OK', torch.__version__)"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('MODEL_OK')"
```

Expected result:

* `TORCH_OK ...`
* `MODEL_OK`

---

## AI Environment (Jetson aarch64)

Do **not** install `torch` from standard pip wheels on Jetson. The recommended order:

1. Install PyTorch/torchvision following NVIDIA/JetPack guidance for your Jetson version.
2. Create venv and install ultralytics after torch is confirmed working:

```bash
python3 -m venv .venv_ai
source .venv_ai/bin/activate
python -m pip install -U pip setuptools wheel

# torch must already be installed & working here:
python -c "import torch; print('TORCH_OK', torch.__version__)"

pip install ultralytics --no-deps
pip install -r requirements.txt

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('MODEL_OK')"
```

---

## WSL2 Simulation Runbook (SITL + Mission Planner + MAVROS + SEANO)

Goal: a repeatable startup procedure that always works.

### Port Map (Baseline)

* `14550/UDP` : Mission Planner (Windows) listens here
* `14551/UDP` : MAVROS ↔ SITL (WSL)
* `5760/TCP`   : MAVProxy/ArduPilot master (internal)

WSL2 note:

* Windows host IP seen from WSL can change each session. Use the WSL default gateway.

---

### Terminal 1 (WSL) — Start SITL (Rover Skid) + Outputs

```bash
cd ~/tools/ardupilot

WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
echo "WIN_HOST_IP=$WIN_HOST_IP"

sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
```

Expected result:

* MAVProxy console and map open.
* Output lines show both `14550` (Mission Planner) and `14551` (MAVROS).

---

### Mission Planner (Windows) — Connect

* Connection type: **UDP**
* Port: **14550**
* Click **Connect**

Expected result:

* Vehicle status appears normally (mode, parameters, map/track).

---

### Terminal 2 (WSL) — Start MAVROS

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

Expected result:

* `/mavros/state` becomes `connected: true`.

---

### Terminal 3 (WSL) — Monitor MAVROS State

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mavros/state
```

Expected result:

* `connected: true`
* `mode` readable (e.g., MANUAL)
* `armed` reflects current state

---

### Terminal 4 (WSL) — Start SEANO Stack

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision run_auto_stack.launch.py
```

Expected result:

* Nodes start without fatal errors.
* Failsafe heartbeat publishes.

---

## Validation Checklist

Use this as “definition of done” for baseline setup:

1. MAVROS is connected:

```bash
ros2 topic echo /mavros/state
# connected: true
```

2. Failsafe heartbeat runs:

```bash
ros2 topic hz /ca/failsafe_active
# stable rate (e.g., ~10 Hz)
```

3. RC override is being published:

```bash
ros2 topic echo /mavros/rc/override
# PWM values change when teleop/auto is active
```

4. Vehicle movement is visible in Mission Planner:

* track changes, heading changes according to inputs

---

## Troubleshooting

### 1) Mission Planner “Connect Failed”

* Ensure SITL sends UDP to Windows host IP from WSL:
  `WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')`
* Ensure SITL includes:
  `--out udp:${WIN_HOST_IP}:14550`

### 2) `/mavros/state connected: false`

* Start SITL first.
* Ensure SITL sends to `udp:127.0.0.1:14551`.
* Ensure MAVROS uses:
  `udp://0.0.0.0:14551@127.0.0.1:14551`
* If SITL restarts, restart MAVROS too.

### 3) Vehicle does not move even though PWM is published

* Ensure vehicle is **ARMED**.
* Ensure mode accepts RC override (commonly **MANUAL/STEERING** for testing).
* Ensure safety limiter is not forcing STOP (look for `FAILSAFE_STOP` logs).
* Ensure there are no duplicate publishers overriding commands:
  `ros2 topic info -v <topic>`

### 4) `install/local_setup.bash not found`

* Workspace likely not built or wrong path is sourced.
* Correct file after build:
  `~/seano-collision-avoidance/seano_ca_ws/install/setup.bash`

---

## License

MIT License (see `LICENSE`).

````
