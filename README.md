[![ROS2 Humble CI](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml)
[![Release Drafter](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-brightgreen)](https://github.com/seanousv/seano-collision-avoidance/network/updates)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# SEANO Collision Avoidance (ROS 2 Humble) — USV Differential Thruster

This repository contains a **camera-based collision avoidance** module for the SEANO USV. The system is developed with **ROS 2 Humble**, validated first in simulation using **ArduPilot SITL (ArduRover rover-skid)**, and designed for deployment on **Jetson Orin Nano + CUAV X7+**.

Key vehicle constraint (critical for control design):
- SEANO uses **differential thrusters** (left–right) **without a rudder**.
- Therefore, the preferred internal motion command format is **left_cmd / right_cmd** (not rudder).

---

## Table of Contents

- [Documentation](#Documentation)
- [Repository Structure](#repository-structure)
- [System Overview](#system-overview)
- [Prerequisites](#prerequisites)
- [Build (ROS 2 Workspace)](#build-ros-2-workspace)
- [AI Environment (WSL x86_64)](#ai-environment-wsl-x86_64)
- [AI Environment (Jetson aarch64)](#ai-environment-jetson-aarch64)
- [CI](#ci)
---

## Documentation

- Architecture overview: `docs/ARCHITECTURE.md`
- Simulation runbook (WSL2 + SITL + Mission Planner + MAVROS + SEANO): `docs/RUNBOOK.md`
- Changelog: `CHANGELOG.md`
- Contributing: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Support: `SUPPORT.md`
- Citation: `CITATION.cff`


## Repository Structure
````md
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

## CI

This repository includes a ROS 2 Humble build workflow:

* Workflow: `.github/workflows/ros2_ci.yml`
* Status: shown via the CI badge above

