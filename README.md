[![ROS2 Humble CI](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml)
[![Release Drafter](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-brightgreen)](https://github.com/seanousv/seano-collision-avoidance/network/updates)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# SEANO Collision Avoidance (ROS 2 Humble) — USV Differential Thruster

Camera-based collision avoidance module for the SEANO USV.
Developed on **ROS 2 Humble**, validated first in **ArduPilot SITL (ArduRover rover-skid)**, and designed for deployment to **Jetson Orin Nano + CUAV X7+**.

## Active baseline status

Current active baseline of this repository:

- **Phase 0** — baseline and reproducible setup: completed
- **Phase 1** — differential-thrust control foundation: completed
- **Phase 5** — mission / avoid / rejoin integration: functional
- **Phase 6** — rosbag-based metrics extraction: active

The current runtime baseline is no longer only “control stack bring-up”.
It now supports:

- waypoint mission on autopilot
- temporary avoidance takeover
- release back to mission
- explicit `REJOIN` state
- synthetic-camera-based integration testing
- quantitative metrics extraction from rosbag

---

## Key vehicle constraint (critical for control design)

SEANO uses **differential thrusters** (left–right) without a rudder.

Therefore, the preferred internal motion command format is:

- `left_cmd`
- `right_cmd`

not rudder-based control.

This decision is used consistently across the control architecture.

---

## What this repository does

The intended mission behavior is:

1. Autopilot follows waypoint mission.
2. Camera / perception detects hazardous obstacle.
3. Collision avoidance performs temporary takeover.
4. Vehicle deviates from nominal path to avoid collision.
5. Takeover is released after hazard clears.
6. System enters `REJOIN`.
7. Vehicle returns to mission-following behavior and continues to the final waypoint.

The current implementation uses the following high-level state logic:

- `MISSION`
- `AVOID`
- `REJOIN`
- `FAILSAFE`

---

## Documentation

Primary documents:

- Architecture overview: `docs/ARCHITECTURE.md`
- Simulation runbook: `docs/RUNBOOK.md`
- Phase 6 test matrix and results template: `docs/PHASE6_TEST_MATRIX.md`
- Changelog: `CHANGELOG.md`
- Contributing: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`
- Support: `SUPPORT.md`
- Citation: `CITATION.cff`
- Code of conduct: `CODE_OF_CONDUCT.md`

Recommended reading order:

1. `docs/RUNBOOK.md`
2. `docs/ARCHITECTURE.md`
3. `docs/PHASE6_TEST_MATRIX.md`

---

````md
## Repository structure

```text
seano-collision-avoidance/
├── seano_ca_ws/                      # ROS 2 workspace (colcon)
│   └── src/
│       └── seano_vision/             # main ROS 2 package (ament_python)
│           ├── seano_vision/         # ROS nodes (Python)
│           ├── launch/               # launch files
│           ├── config/               # configuration
│           ├── resource/             # ament resources
│           ├── scripts/              # helper / metrics scripts
│           ├── package.xml
│           ├── setup.py
│           └── setup.cfg
├── docs/                             # architecture, runbook, test matrix
├── tools/                            # optional external tools
├── requirements.txt                  # AI deps (WSL-safe; Jetson torch manual)
├── LICENSE
└── README.md
````

Notes:

* `seano_ca_ws/` is the ROS 2 workspace built with `colcon`.
* `seano_vision/` contains the control, vision, bridge, and metrics nodes/scripts.
* `tools/` is optional and intended for external tooling.
* `docs/` contains the current active operational baseline documentation.

---

## Current system overview

The active runtime architecture is organized into layers:

### 1. Mission / Autopilot layer

* Mission Planner defines waypoint mission.
* ArduPilot executes mission in autopilot mode.

### 2. Perception / decision layer

* Camera source (USB / RTSP / synthetic)
* Detector
* Risk evaluator
* Watchdog / failsafe logic

### 3. Control layer

* manual control
* takeover control
* command selection
* safety limiting
* RC override bridge

### 4. Mode / state layer

* mission mode manager
* mission / avoid / rejoin / failsafe transitions
* mode restore enforcement

### 5. Evaluation layer

* rosbag recording
* Phase 6 metrics extraction
* Phase 6 aggregated result collection

---

## Core runtime pipeline

Main actuation path:

```text
manual/auto command
-> command_mux_node
-> actuator_safety_limiter_node
-> mavros_rc_override_bridge_node
-> /mavros/rc/override
-> ArduPilot
```

Mission / state path:

```text
/mavros/state
+ /seano/rc_override_enable
+ /ca/failsafe_active
-> mission_mode_manager_node
-> /ca/mode_manager_state
-> /ca/mode_manager_event
-> /mavros/set_mode
```

Perception / decision path:

```text
camera
-> detector
-> risk evaluator
-> /ca/command_safe
-> auto_controller_stub_node
-> takeover / release
```

---

## Runtime test modes

The current runtime baseline is divided into 3 practical modes.

### Case A — Mode Manager Only

Purpose:

* validate state machine only
* validate `MISSION -> AVOID -> REJOIN -> MISSION`
* no perception pipeline
* no takeover manager interference

### Case B — Takeover Logic Only

Purpose:

* validate hazard command -> takeover -> release
* still lightweight
* no full perception/watchdog load

### Case C — Full Integration (Synthetic Camera)

Purpose:

* validate integrated runtime
* does not depend on hardware camera
* suitable for WSL testing

Default perception runtime profile for Case C:

* `synthetic_light`

Additional profiles:

* `synthetic_watchdog`
* `full`

---

## Why synthetic camera is used now

A synthetic camera path is intentionally used in simulation-first testing because it:

* removes dependence on hardware camera during control/state integration
* reduces runtime load in WSL
* keeps perception chain testable
* allows faster iteration before USB camera migration

Hardware camera integration remains part of the next-stage migration, not the current simulation baseline.

---

## Prerequisites

### Host

* Windows + WSL2 Ubuntu 22.04

### WSL

* ROS 2 Humble
* MAVROS2
* OpenCV
* `cv_bridge`

### Simulator / GCS

* ArduPilot SITL (ArduRover rover-skid)
* Mission Planner (Windows)

---

## Build (ROS 2 workspace)

### 1. Clone

```bash
git clone https://github.com/seanousv/seano-collision-avoidance.git
cd seano-collision-avoidance
```

### 2. Install minimal dependencies (WSL)

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

### 3. Build workspace

```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Expected output:

* `colcon build` completes without errors
* `ros2 pkg list | grep seano_vision` shows `seano_vision`

---

## Quick start (current active runtime)

Detailed steps are documented in `docs/RUNBOOK.md`.
Short version:

### Terminal 1 — SITL

```bash
cd ~/tools/ardupilot
WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
```

### Mission Planner (Windows)

* Connection type: `UDP`
* Port: `14550`

### Terminal 2 — MAVROS

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

### Terminal 3 — Phase 5 integration

#### Case A

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=false \
  use_takeover_manager:=false
```

#### Case B

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=false \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

#### Case C (default)

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

Expected baseline behavior:

* MAVROS connected
* state manager transitions work
* Case C runs using synthetic camera without hardware camera dependency

---

## Phase 5 validation target

A valid Phase 5 run should show:

* `/mavros/state connected: true`
* `/ca/mode_manager_state` transitions:

  * `MISSION -> AVOID -> REJOIN -> MISSION`
* `/ca/mode_manager_event` contains:

  * `TAKEOVER_ON`
  * `TAKEOVER_OFF`
  * `REJOIN_START`
  * `REJOIN_DONE`

---

## Phase 6 metrics workflow

### 1. Record a run

Example:

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_rejoin_run_01 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

### 2. Extract metrics from one bag

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_metrics_from_bag.py \
  --bag ~/bags/phase6_rejoin_run_01
```

### 3. Aggregate multiple runs

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_collect_results.py \
  --root ~/bags
```

Expected outputs:

* per-bag metrics JSON:

  * `~/bags/<bag_name>/phase6_metrics.json`
* aggregated results:

  * `~/bags/phase6_summary/phase6_results.csv`
  * `~/bags/phase6_summary/phase6_results.md`

---

## Phase 6 metrics currently tracked

Per-bag metrics include:

* takeover segments
* takeover duration
* reaction time
* release time
* rejoin time
* failsafe rises
* mission-mode mismatch ratio

These metrics are intended to support the thesis evaluation chapter.

---

## AI environment (WSL x86_64)

Use a dedicated venv to avoid affecting ROS.

```bash
cd ~/seano-collision-avoidance

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

## AI environment (Jetson aarch64)

Do not install `torch` from standard pip wheels on Jetson.

Recommended order:

1. install PyTorch/torchvision according to NVIDIA / JetPack guidance
2. then install ultralytics and remaining dependencies

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

## Current limitations

Current active baseline already supports:

* temporary avoidance takeover
* release back to mission
* explicit `REJOIN`
* rosbag metrics extraction

Still limited:

* rejoin is still mission-resume oriented, not a fully independent path replanner
* synthetic camera is still the default simulation perception source
* hardware camera baseline is planned after simulation metrics stabilize

---

## Recommended next operational sequence

Recommended execution order:

1. validate Case A
2. validate Case B
3. validate Case C (`synthetic_light`)
4. record hazard/rejoin bags
5. record failsafe bags
6. aggregate results
7. migrate to USB camera
8. prepare hardware porting and controlled field test

---

## CI

This repository includes a ROS 2 Humble build workflow.

* Workflow: `.github/workflows/ros2_ci.yml`
* Status: shown via badge

---

## Citation and academic use

If this repository is referenced in academic work, use `CITATION.cff` as the canonical citation metadata.

---

## License

This project is released under the MIT License. See `LICENSE`.
