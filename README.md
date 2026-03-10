[![ROS2 Humble CI](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml)
[![Release Drafter](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-brightgreen)](https://github.com/seanousv/seano-collision-avoidance/network/updates)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


# SEANO Collision Avoidance (ROS 2 Humble)
## Camera-based collision avoidance for USV SEANO BIMA30

Camera-based collision avoidance module for the SEANO USV, developed on **ROS 2 Humble**, validated in **ArduPilot SITL (ArduRover rover-skid)**, and prepared for hardware deployment on **Jetson Orin Nano + CUAV X7+ + USB camera**.

---

## 1. Current active project baseline

The repository now has **two active baselines**:

### A. Active simulation baseline
Used for repeatable integration testing and thesis evidence extraction.

Current status:
- **Phase 0** — baseline and reproducible setup: completed
- **Phase 1** — differential-thrust control foundation: completed
- **Phase 5** — mission / avoid / rejoin integration: functional
- **Phase 6** — rosbag-based metrics extraction: active

Simulation baseline currently supports:
- waypoint mission on autopilot
- temporary avoidance takeover
- release back to mission
- explicit `REJOIN` state
- synthetic-camera-based integration testing
- quantitative metrics extraction from rosbag

### B. Active hardware bench baseline
Used for Jetson + CUAV X7+ + USB camera bench integration and field-test preparation.

Current status:
- MAVROS connection to CUAV X7+ validated
- USB camera source validated on Jetson
- detector path active
- risk evaluator active
- watchdog active
- browser-based monitoring active
- debug HUD `/ca/debug_image` active
- hardware orchestration available through Phase 7

Hardware bench baseline currently supports:
- Jetson runtime
- CUAV X7+ autopilot link
- USB camera perception source
- detector -> risk -> watchdog chain
- browser monitoring for raw / annotated / HUD streams
- Phase 7 integrated launch for hardware bring-up

---

## 2. Mission behavior target

The intended mission behavior is:

1. Autopilot follows waypoint mission.
2. Camera / perception detects hazardous obstacle.
3. Collision avoidance performs temporary takeover.
4. Vehicle deviates from nominal path to avoid collision.
5. Takeover is released after hazard clears.
6. System enters `REJOIN`.
7. Vehicle returns to mission-following behavior and continues to the final waypoint.

The current high-level state logic is:

- `MISSION`
- `AVOID`
- `REJOIN`
- `FAILSAFE`

---

## 3. Critical vehicle constraint

SEANO uses **differential thrusters** (left-right) and does **not** use a rudder.

Therefore, the preferred internal motion command format is:

- `left_cmd`
- `right_cmd`

not rudder-based control.

This design choice is applied consistently across the control stack.

---

## 4. What this repository contains

This repository contains the collision-avoidance stack for:

- camera input
- AI-based object detection
- risk evaluation
- takeover / release logic
- failsafe and watchdog logic
- mission-mode management
- MAVROS bridge to ArduPilot
- simulation evaluation workflow
- hardware bench integration workflow

---

## 5. Active launch map

### Main simulation launch
- `phase5_mission_avoid_integration.launch.py`
- purpose: SITL mission -> avoid -> rejoin -> mission

### Main hardware launch
- `phase7_cuav_usb_hardware.launch.py`
- purpose: Jetson + CUAV X7+ + USB camera + detector + risk + watchdog + control integration

### Bench / debug launches
- `phase2_camera_usb_test.launch.py`
- `demo_detect.launch.py`
- `demo_risk.launch.py`
- `demo_full_ca.launch.py`

Rule of thumb:
- **SITL main** -> `phase5_mission_avoid_integration.launch.py`
- **Hardware main** -> `phase7_cuav_usb_hardware.launch.py`
- **Detector check** -> `demo_detect.launch.py`
- **Risk check** -> `demo_risk.launch.py`

---

## 6. Repository structure
````markdown
```text
seano-collision-avoidance/
├── .vscode/                           # local workspace settings
├── docs/                              # documentation and operational references
├── seano_ca_ws/                       # ROS 2 workspace (colcon)
│   └── src/
│       └── seano_vision/
│           ├── config/                # runtime configuration
│           ├── launch/                # ROS 2 launch files
│           ├── resource/              # ament resources
│           ├── scripts/               # helper and metrics scripts
│           ├── seano_vision/          # main ROS nodes
│           ├── package.xml
│           ├── setup.cfg
│           └── setup.py
├── tools/                             # optional external tools
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
└── SUPPORT.md
````

---

## 7. Key documents

Primary documents:

* `docs/RUNBOOK.md`
  Main simulation runbook for SITL + Mission Planner + MAVROS + Phase 5 / Phase 6

* `docs/ARCHITECTURE.md`
  Current architecture overview

* `docs/PHASE6_TEST_MATRIX.md`
  Test matrix for metrics-oriented simulation runs

* `docs/PHASE6_RESULTS_SUMMARY.md`
  Phase 6 summary results

* `docs/LAUNCH_STATUS_MAP.md`
  Quick operational map for active launch files vs bench/debug launches

Recommended reading order:

1. `docs/LAUNCH_STATUS_MAP.md`
2. `docs/RUNBOOK.md`
3. `docs/ARCHITECTURE.md`
4. `docs/PHASE6_TEST_MATRIX.md`
5. `docs/PHASE6_RESULTS_SUMMARY.md`

---

## 8. Current system overview

The active runtime architecture is organized into layers.

### 1. Mission / Autopilot layer

* Mission Planner defines waypoint mission
* ArduPilot executes mission in autopilot mode
* CUAV X7+ is the active hardware autopilot target

### 2. Perception / decision layer

* camera source (synthetic / USB)
* detector
* risk evaluator
* watchdog / failsafe logic

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
* result aggregation scripts

### 6. Monitoring layer

* raw image stream
* annotated image stream
* debug HUD `/ca/debug_image`
* browser monitoring via `web_video_server`

---

## 9. Core runtime pipeline

### Actuation path

```text
manual/auto command
-> command_mux_node
-> actuator_safety_limiter_node
-> mavros_rc_override_bridge_node
-> /mavros/rc/override
-> ArduPilot
```

### Mission / state path

```text
/mavros/state + /seano/rc_override_enable + /ca/failsafe_active
-> mission_mode_manager_node
-> /ca/mode_manager_state
-> /ca/mode_manager_event
-> /mavros/set_mode
```

### Perception / decision path

```text
camera
-> detector
-> risk evaluator
-> /ca/command_safe
-> auto_controller_stub_node
-> takeover / release
```

---

## 10. Simulation runtime modes

The practical simulation baseline is centered on **Phase 5**.

### Case A — Mode Manager Only

Purpose:

* validate state machine only
* validate `MISSION -> AVOID -> REJOIN -> MISSION`
* no perception pipeline
* no takeover manager interference

### Case B — Takeover Logic Only

Purpose:

* validate hazard command -> takeover -> release
* lightweight runtime
* no full perception/watchdog load

### Case C — Full Integration (Synthetic Camera)

Purpose:

* validate integrated runtime
* avoid hardware camera dependency
* suitable for WSL testing

Default practical profiles:

* `synthetic_light`
* `synthetic_watchdog`
* `full`

---

## 11. Hardware runtime mode

The practical hardware baseline is centered on **Phase 7**.

### Phase 7 — Full hardware integration

Purpose:

* Jetson + CUAV X7+ + USB camera hardware bring-up
* detector -> risk -> watchdog -> control integration
* browser-based monitoring
* field-test preparation

Typical monitoring topics:

* `/seano/camera/image_raw_reliable`
* `/camera/image_annotated`
* `/ca/debug_image`

---

## 12. Prerequisites

### Simulation host

* Windows + WSL2 Ubuntu 22.04
* ROS 2 Humble
* MAVROS2
* Mission Planner on Windows
* ArduPilot SITL

### Hardware runtime

* Jetson Orin Nano
* ROS 2 Humble
* CUAV X7+
* USB camera
* MAVROS2
* browser monitoring via `web_video_server`

---

## 13. Build workspace

### Clone

```bash
git clone https://github.com/seanousv/seano-collision-avoidance.git
cd seano-collision-avoidance
```

### Build

```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select seano_vision --symlink-install
source install/setup.bash
```

Expected:

* build finishes without errors
* `seano_vision` is available
* latest launch and node changes are installed

---

## 14. Quick start — simulation

### Terminal 1 — SITL

```bash
cd ~/tools/ardupilot
WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
```

### Mission Planner (Windows)

* connection type: `UDP`
* port: `14550`

### Terminal 2 — MAVROS

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

### Terminal 3 — Phase 5

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true \
  ca_runtime_profile:=synthetic_watchdog \
  record:=true \
  bag_name:=sitl_main_run_01
```

Target:

* `/mavros/state` connected
* `MISSION -> AVOID -> REJOIN -> MISSION`
* rosbag recorded for metrics

---

## 15. Quick start — hardware bench / field preparation

### Terminal 1 — video server

```bash
cd ~/Seano_ws/resource_git/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run web_video_server web_video_server
```

### Terminal 2 — Phase 7

```bash
cd ~/Seano_ws/resource_git/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200 \
  record:=true \
  bag_name:=fieldtest_main_01
```

### Browser monitoring

Open:

* raw camera stream
* annotated image stream
* CA HUD stream

Typical topics:

* `/seano/camera/image_raw_reliable`
* `/camera/image_annotated`
* `/ca/debug_image`

---

## 16. Phase 6 metrics workflow

### Record run

```bash
ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_rejoin_run_01 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

### Extract one bag

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_metrics_from_bag.py \
  --bag ~/bags/phase6_rejoin_run_01
```

### Aggregate multiple bags

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_collect_results.py \
  --root ~/bags
```

Tracked metrics include:

* takeover segments
* takeover duration
* reaction time
* release time
* rejoin time
* failsafe rises
* mission-mode mismatch ratio

---

## 17. Current limitations

Current active baseline already supports:

* temporary avoidance takeover
* release back to mission
* explicit `REJOIN`
* rosbag metrics extraction
* hardware bench bring-up with Phase 7

Still limited:

* rejoin is mission-resume oriented, not a full path replanner
* detector performance still strongly affects hardware behavior
* field success depends on visibility, obstacle presentation, and runtime stability
* controlled field validation still needs to be expanded

---

## 18. Recommended operational sequence

### For simulation

1. validate SITL + MAVROS + Mission Planner
2. validate Phase 5 Case C
3. record hazard / rejoin runs
4. record failsafe runs
5. extract Phase 6 metrics
6. aggregate results

### For hardware

1. validate USB camera
2. validate detector
3. validate risk + HUD
4. validate Phase 7 dockside
5. validate AUTO waypoint without obstacle
6. validate obstacle run
7. validate avoidance + rejoin behavior
8. review rosbag and logs

---

## 19. CI

This repository includes:

* ROS 2 Humble CI workflow
* Release Drafter
* Dependabot configuration

---

## 20. Citation and academic use

If this repository is referenced in academic work, use `CITATION.cff` as the canonical citation metadata.

---

## 21. License

This project is released under the MIT License. See `LICENSE`.

````
