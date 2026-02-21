# SEANO Collision Avoidance (ROS 2 Humble) — USV Differential Thruster

This repository contains a **camera-based collision avoidance** module for the SEANO USV. The system is developed with **ROS 2 Humble**, validated first in simulation using **ArduPilot SITL (ArduRover rover-skid)**, and designed for deployment on **Jetson Orin Nano + CUAV X7+**.

Key vehicle constraint (critical for control design):
- SEANO uses **differential thrusters** (left–right) **without a rudder**.
- Therefore, the preferred internal motion command format is **left_cmd / right_cmd** (not rudder).

---

````md
## Table of Contents

- [Repository Structure](#repository-structure)
- [System Overview](#system-overview)
- [Prerequisites](#prerequisites)
- [Build (ROS 2 Workspace)](#build-ros-2-workspace)
- [AI Environment (WSL x86_64)](#ai-environment-wsl-x86_64)
- [AI Environment (Jetson aarch64)](#ai-environment-jetson-aarch64)
- [WSL2 Simulation Runbook](#wsl2-simulation-runbook)
- [Validation Checklist](#validation-checklist)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Repository Structure

```text
seano-collision-avoidance/
├── seano_ca_ws/                      # ROS 2 workspace (colcon)
│   └── src/
│       └── seano_vision/             # main ROS 2 package (ament_python)
│           ├── seano_vision/         # ROS nodes (Python)
│           ├── launch/               # launch files
│           ├── config/               # configuration (optional)
│           ├── resource/             # ament resources
│           ├── package.xml
│           ├── setup.py
│           └── setup.cfg
├── tools/                            # optional tools (kept separate from ROS workspace)
├── requirements.txt                  # AI deps (WSL-safe; Jetson torch manual)
├── LICENSE
└── README.md
````

Notes:

* `seano_ca_ws/` is the ROS 2 workspace built with `colcon`.
* `seano_vision/` contains the control/vision/bridge nodes used in the pipeline.
* `tools/` is optional and intended for external tooling (e.g., keeping ArduPilot outside the ROS workspace).

---

## System Overview

The control pipeline is designed to be deterministic and safe:

1. Manual teleoperation (keyboard):

* Publishes: `/seano/manual/left_cmd`, `/seano/manual/right_cmd`

2. Autonomous command source (AI/risk/planner):

* Publishes: `/seano/auto/left_cmd`, `/seano/auto/right_cmd`
* Toggle commonly used: `/seano/auto_enable` (Bool)

3. Command multiplexer (manual vs auto):

* Outputs: `/seano/selected/left_cmd`, `/seano/selected/right_cmd`

4. Safety limiter + failsafe:

* Enforces timeout, clamps, safe-stop policy
* Outputs final: `/seano/left_cmd`, `/seano/right_cmd`

5. MAVROS RC override bridge:

* Converts left/right to PWM RC override
* Publishes: `/mavros/rc/override`

This separation keeps the AI layer focused on “motion intent” while safety and actuation remain robust.

---

## Prerequisites

Host:

* Windows + WSL2 Ubuntu 22.04

WSL:

* ROS 2 Humble
* MAVROS2 (ROS2 Humble)
* OpenCV + cv_bridge

Simulator/GCS:

* ArduPilot SITL (ArduRover rover-skid)
* Mission Planner (Windows)

---

## Build (ROS 2 Workspace)

1. Clone:

```bash
git clone https://github.com/seanousv/seano-collision-avoidance.git
cd seano-collision-avoidance
```

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

3. Build workspace:

```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Expected output:

* `colcon build` completes without errors.
* `ros2 pkg list | grep seano_vision` shows `seano_vision`.

---

## AI Environment (WSL x86_64)

Use a dedicated venv to avoid affecting ROS.

```bash
cd ~/seano-collision-avoidance

# if you need a clean reinstall:
# rm -rf .venv_ai

python3 -m venv .venv_ai
source .venv_ai/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

python -c "import torch; print('TORCH_OK', torch.__version__)"
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('MODEL_OK')"
```

Expected output:

* `TORCH_OK ...`
* `MODEL_OK`

---

## AI Environment (Jetson aarch64)

Do **not** install `torch` from standard pip wheels on Jetson. Recommended order:

1. Install PyTorch/torchvision according to NVIDIA/JetPack guidance.
2. Then install ultralytics and the remaining dependencies:

```bash
python3 -m venv .venv_ai
source .venv_ai/bin/activate
python -m pip install -U pip setuptools wheel

python -c "import torch; print('TORCH_OK', torch.__version__)"

pip install ultralytics --no-deps
pip install -r requirements.txt

python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('MODEL_OK')"
```

---

## WSL2 Simulation Runbook

Goal: a repeatable startup procedure that always works.

### Port Map (Baseline)

* `14550/UDP`: Mission Planner (Windows)
* `14551/UDP`: MAVROS ↔ SITL (WSL)
* `5760/TCP`: MAVProxy/ArduPilot master (internal)

WSL2 note:

* Windows host IP seen from WSL changes between sessions.
* Use the WSL default gateway.

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

Expected output:

* MAVProxy console/map appears.
* SITL shows outputs to both `14550` and `14551`.

---

### Mission Planner (Windows) — Connect

* Connection type: UDP
* Port: 14550
* Click Connect

Expected output:

* Vehicle status appears normally (mode/parameters/map).

---

### Terminal 2 (WSL) — Start MAVROS

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

Expected output:

* `/mavros/state` becomes `connected: true`.

---

### Terminal 3 (WSL) — Monitor MAVROS State

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mavros/state
```

Expected output:

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

Expected output:

* Nodes start without fatal errors.
* Failsafe heartbeat publishes.

---

## Validation Checklist

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
# PWM changes when teleop/auto is active
```

4. Vehicle movement is visible in Mission Planner (map/track changes).

---

## Troubleshooting

### Mission Planner “Connect Failed”

* Ensure SITL sends to Windows host IP from WSL:
  `WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')`
* Ensure SITL includes:
  `--out udp:${WIN_HOST_IP}:14550`

### `/mavros/state connected: false`

* Start SITL first.
* Ensure SITL outputs to `udp:127.0.0.1:14551`.
* Ensure MAVROS uses:
  `udp://0.0.0.0:14551@127.0.0.1:14551`
* If SITL restarts, restart MAVROS too.

### Vehicle does not move although PWM is published

* Ensure vehicle is **ARMED**.
* Ensure mode accepts RC override (commonly **MANUAL/STEERING** for testing).
* Ensure the safety limiter is not forcing STOP (`FAILSAFE_STOP` logs).
* Ensure there are no duplicate publishers overriding commands:
  `ros2 topic info -v <topic>`

### `install/local_setup.bash not found`

* Workspace not built or wrong path sourced.
* Correct file after build:
  `~/seano-collision-avoidance/seano_ca_ws/install/setup.bash`

---

## License

MIT License

Copyright (c) 2026 seanousv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


---
