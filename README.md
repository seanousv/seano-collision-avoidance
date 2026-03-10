[![ROS2 Humble CI](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/ros2_ci.yml)
[![Release Drafter](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml/badge.svg)](https://github.com/seanousv/seano-collision-avoidance/actions/workflows/release-drafter.yml)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-brightgreen)](https://github.com/seanousv/seano-collision-avoidance/network/updates)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


# SEANO Collision Avoidance (ROS 2 Humble)
## Vision-Based Collision Avoidance for USV SEANO BIMA30

This repository contains a ROS 2 Humble–based collision avoidance stack for **SEANO BIMA30**, a differential-thruster unmanned surface vehicle (USV). The project is developed and validated in two parallel baselines:

- **Simulation baseline** using **ArduPilot SITL + MAVROS + Mission Planner**
- **Hardware baseline** using **Jetson + CUAV X7+ + USB camera**

The system goal is to support the following mission behavior:

1. The autopilot follows a waypoint mission in `AUTO`.
2. The vision stack detects a hazardous obstacle.
3. The collision avoidance system takes over temporarily.
4. The USV leaves the nominal mission path to avoid collision.
5. When the hazard is no longer present, takeover is released.
6. The system enters `REJOIN`.
7. The vehicle returns to mission-following behavior and continues to the next waypoint until the mission is complete.

---

# 1. Current Active Baselines

## 1.1 Simulation Baseline
The main simulation integration launch is:

- [`phase5_mission_avoid_integration.launch.py`](seano_ca_ws/src/seano_vision/launch/phase5_mission_avoid_integration.launch.py)

This baseline is used for:

- SITL mission execution
- temporary avoidance takeover
- release back to mission
- explicit `MISSION -> AVOID -> REJOIN -> MISSION` validation
- rosbag recording
- Phase 6 metrics extraction

Related documents:

- [Simulation Runbook](docs/RUNBOOK.md)
- [Phase 6 Test Matrix](docs/PHASE6_TEST_MATRIX.md)
- [Phase 6 Results Summary](docs/PHASE6_RESULTS_SUMMARY.md)

## 1.2 Hardware Baseline
The main hardware integration launch is:

- [`phase7_cuav_usb_hardware.launch.py`](seano_ca_ws/src/seano_vision/launch/phase7_cuav_usb_hardware.launch.py)

This baseline is used for:

- Jetson + CUAV X7+ integration
- USB camera bring-up
- detector -> risk -> watchdog runtime
- RC override and safety limiting
- browser-based monitoring
- dockside and field-test preparation

Related bench/debug launches:

- [`phase2_camera_usb_test.launch.py`](seano_ca_ws/src/seano_vision/launch/phase2_camera_usb_test.launch.py)
- [`demo_detect.launch.py`](seano_ca_ws/src/seano_vision/launch/demo_detect.launch.py)
- [`demo_risk.launch.py`](seano_ca_ws/src/seano_vision/launch/demo_risk.launch.py)
- [`demo_full_ca.launch.py`](seano_ca_ws/src/seano_vision/launch/demo_full_ca.launch.py)

---

# 2. Vehicle Constraint and Control Philosophy

SEANO BIMA30 uses **differential thrust**:

- left motor
- right motor
- no rudder

Therefore, the preferred internal motion command format is:

- `left_cmd`
- `right_cmd`

This is why the control architecture is built around left/right actuation rather than rudder steering.

---

# 3. Repository Map

````md
```text
seano-collision-avoidance/
├── .github/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── HARDWARE_BENCH_RESULTS.md
│   ├── LAUNCH_STATUS_MAP.md
│   ├── PHASE6_RESULTS_SUMMARY.md
│   ├── PHASE6_TEST_MATRIX.md
│   └── RUNBOOK.md
├── seano_ca_ws/
│   └── src/
│       └── seano_vision/
│           ├── config/
│           ├── launch/
│           ├── models/
│           ├── resource/
│           ├── scripts/
│           ├── seano_vision/
│           ├── package.xml
│           ├── setup.cfg
│           └── setup.py
├── tools/
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
├── SUPPORT.md
├── requirements.txt
└── requirements-dev.txt
````

Useful entry points:

* [Launch Status Map](docs/LAUNCH_STATUS_MAP.md)
* [Hardware Bench Results](docs/HARDWARE_BENCH_RESULTS.md)
* [Architecture](docs/ARCHITECTURE.md)
* [Simulation + Hardware Runbook](docs/RUNBOOK.md)
* [Main launch directory](seano_ca_ws/src/seano_vision/launch/)
* [Main Python nodes](seano_ca_ws/src/seano_vision/seano_vision/)
* [Metrics scripts](seano_ca_ws/src/seano_vision/scripts/)

---

# 4. Active Launch Selection Guide

## 4.1 Use This for Simulation

* [`phase5_mission_avoid_integration.launch.py`](seano_ca_ws/src/seano_vision/launch/phase5_mission_avoid_integration.launch.py)

## 4.2 Use This for Full Hardware / Field Test

* [`phase7_cuav_usb_hardware.launch.py`](seano_ca_ws/src/seano_vision/launch/phase7_cuav_usb_hardware.launch.py)

## 4.3 Use These for Bench / Debug

* camera only: [`phase2_camera_usb_test.launch.py`](seano_ca_ws/src/seano_vision/launch/phase2_camera_usb_test.launch.py)
* camera + detector: [`demo_detect.launch.py`](seano_ca_ws/src/seano_vision/launch/demo_detect.launch.py)
* camera + detector + risk: [`demo_risk.launch.py`](seano_ca_ws/src/seano_vision/launch/demo_risk.launch.py)
* full CA bench pipeline: [`demo_full_ca.launch.py`](seano_ca_ws/src/seano_vision/launch/demo_full_ca.launch.py)

---

# 5. Core Runtime Components

## 5.1 Perception / Decision

* [`camera_node.py`](seano_ca_ws/src/seano_vision/seano_vision/camera_node.py)
* [`detector_node.py`](seano_ca_ws/src/seano_vision/seano_vision/detector_node.py)
* [`risk_evaluator_node.py`](seano_ca_ws/src/seano_vision/seano_vision/risk_evaluator_node.py)
* [`watchdog_failsafe_node.py`](seano_ca_ws/src/seano_vision/seano_vision/watchdog_failsafe_node.py)

## 5.2 Control

* [`command_mux_node.py`](seano_ca_ws/src/seano_vision/seano_vision/command_mux_node.py)
* [`actuator_safety_limiter_node.py`](seano_ca_ws/src/seano_vision/seano_vision/actuator_safety_limiter_node.py)
* [`mavros_rc_override_bridge_node.py`](seano_ca_ws/src/seano_vision/seano_vision/mavros_rc_override_bridge_node.py)
* [`auto_controller_stub_node.py`](seano_ca_ws/src/seano_vision/seano_vision/auto_controller_stub_node.py)

## 5.3 Mission / Mode

* [`mission_mode_manager_node.py`](seano_ca_ws/src/seano_vision/seano_vision/mission_mode_manager_node.py)

---

# 6. Build Instructions

## 6.1 Clone

```bash
git clone https://github.com/seanousv/seano-collision-avoidance.git
cd seano-collision-avoidance
```

## 6.2 Build Workspace

```bash
cd seano_ca_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select seano_vision --symlink-install
source install/setup.bash
```

## 6.3 Verify

```bash
ros2 pkg list | grep seano_vision
```

Expected result:

* `seano_vision` is listed
* the package builds without errors

---

# 7. Simulation Workflow

This section is intentionally separate from hardware.

## 7.1 Purpose

Use simulation when you want to validate:

* autopilot mission execution
* mission / avoid / rejoin transitions
* takeover / release logic
* rosbag-based metrics extraction

## 7.2 Required Tools

* ArduPilot SITL
* Mission Planner (Windows)
* MAVROS
* ROS 2 Humble workspace

## 7.3 Startup Order

Always use this order:

1. ArduPilot SITL
2. Mission Planner
3. MAVROS
4. SEANO Phase 5 launch

## 7.4 Terminal 1 — Start SITL

```bash
cd ~/tools/ardupilot
WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
echo "WIN_HOST_IP=$WIN_HOST_IP"

sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
```

## 7.5 Mission Planner (Windows)

Connect with:

* Connection type: `UDP`
* Port: `14550`

## 7.6 Terminal 2 — Start MAVROS

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

## 7.7 Terminal 3 — Main Simulation Launch

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

## 7.8 Simulation Verification

```bash
ros2 topic echo /mavros/state -n 1
ros2 topic echo /ca/mode_manager_state
ros2 topic echo /ca/mode_manager_event
```

Expected behavior:

* `/mavros/state` shows `connected: true`
* the system can progress through:

  * `MISSION`
  * `AVOID`
  * `REJOIN`
  * back to `MISSION`

## 7.9 Lighter Simulation Modes

### Mode Manager Only

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=false \
  use_takeover_manager:=false
```

### Takeover Logic Only

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  use_ca_pipeline:=false \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

Manual trigger example:

```bash
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'TURN_RIGHT'}"
sleep 3
ros2 topic pub --once /ca/command_safe std_msgs/msg/String "{data: 'HOLD_COURSE'}"
```

## 7.10 Phase 6 Metrics

Record:

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase5_mission_avoid_integration.launch.py \
  record:=true \
  bag_name:=phase6_rejoin_run_01 \
  use_ca_pipeline:=true \
  use_takeover_manager:=true \
  master_enable_on_start:=true
```

Extract one bag:

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_metrics_from_bag.py \
  --bag ~/bags/phase6_rejoin_run_01
```

Aggregate multiple bags:

```bash
python3 ~/seano-collision-avoidance/seano_ca_ws/src/seano_vision/scripts/phase6_collect_results.py \
  --root ~/bags
```

---

# 8. Hardware Workflow

This section is intentionally separate from simulation.

## 8.1 Purpose

Use hardware workflow when you want to validate:

* Jetson runtime
* CUAV X7+ MAVROS connection
* USB camera perception chain
* detector / risk / watchdog runtime
* full hardware integration
* dockside and field preparation

## 8.2 Bench Progression

Recommended order:

1. camera only
2. camera + detector
3. camera + detector + risk
4. full CA bench
5. full hardware Phase 7
6. field test

## 8.3 Camera Only

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase2_camera_usb_test.launch.py
```

Verify:

```bash
ros2 topic list | grep image_raw
ros2 topic hz /seano/camera/image_raw_reliable
```

## 8.4 Camera + Detector

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision demo_detect.launch.py
```

Verify:

```bash
ros2 topic list | grep -E "image_raw_reliable|image_annotated|detections"
ros2 topic hz /seano/camera/image_raw_reliable
ros2 topic hz /camera/image_annotated
ros2 topic hz /camera/detections
```

## 8.5 Camera + Detector + Risk

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision demo_risk.launch.py
```

Verify:

```bash
ros2 topic list | grep -E "image_raw_reliable|image_annotated|detections|risk|command|debug_image"
ros2 topic echo /ca/risk
ros2 topic echo /ca/command
```

## 8.6 Full CA Bench Pipeline

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision demo_full_ca.launch.py
```

Verify:

```bash
ros2 topic list | grep -E "image_raw_reliable|image_annotated|detections|risk|command|debug_image|failsafe"
ros2 topic echo /ca/failsafe_active
```

## 8.7 Video Monitoring

Start video server:

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run web_video_server web_video_server
```

Typical monitoring URLs:

### Raw camera

```text
http://localhost:8080/stream?topic=/seano/camera/image_raw_reliable
```

### Annotated image

```text
http://localhost:8080/stream?topic=/camera/image_annotated
```

### HUD / debug image

```text
http://localhost:8080/stream?topic=/ca/debug_image
```

### Snapshot

```text
http://localhost:8080/snapshot?topic=/ca/debug_image
```

If VS Code port forwarding maps remote `8080` to local `8081`, use `8081` instead.

## 8.8 Full Hardware Main Launch

This is the main hardware run:

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200 \
  record:=true \
  bag_name:=phase7_fieldtest_01
```

## 8.9 Hardware Runtime Profiles

### Default practical profile

```bash
ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200 \
  ca_runtime_profile:=usb_watchdog
```

### Full perception profile

```bash
ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200 \
  ca_runtime_profile:=full
```

### Force master enable on start

```bash
ros2 launch seano_vision phase7_cuav_usb_hardware.launch.py \
  fcu_url:=/dev/ttyACM0:115200 \
  master_enable_on_start:=true
```

## 8.10 Hardware Verification

```bash
ros2 topic list | grep -E "image_raw_reliable|image_annotated|detections|debug_image|risk|command|failsafe|mode_manager"
ros2 topic hz /seano/camera/image_raw_reliable
ros2 topic hz /camera/image_annotated
ros2 topic hz /camera/detections
ros2 topic echo /ca/risk
ros2 topic echo /ca/command
ros2 topic echo /ca/watchdog_status
ros2 topic echo /ca/failsafe_active
ros2 topic echo /ca/mode_manager_state
ros2 topic echo /mavros/state -n 1
```

Expected behavior:

* camera stream is active
* annotated image is active
* HUD is active
* watchdog status is healthy
* MAVROS is connected
* risk and command topics are published

---

# 9. Recommended Operational Sequence

## 9.1 For Simulation

1. Start SITL
2. Connect Mission Planner
3. Start MAVROS
4. Start Phase 5
5. Validate mission / avoid / rejoin
6. Record bags
7. Extract metrics

## 9.2 For Hardware

1. Validate USB camera
2. Validate detector
3. Validate risk + HUD
4. Validate full CA bench
5. Start Phase 7
6. Verify dockside
7. Verify AUTO waypoint run
8. Verify obstacle run
9. Review bags and logs

---

# 10. Current Limitations

* `REJOIN` is currently mission-resume oriented, not a full global path planner.
* Real-world success still depends on obstacle visibility, lighting, and detector performance.
* Full field validation should always be preceded by bench verification.
* Active launch files should not be renamed or restructured until the baseline is locked.

---

# 11. Related Documents

* [Simulation + Hardware Runbook](docs/RUNBOOK.md)
* [Launch Status Map](docs/LAUNCH_STATUS_MAP.md)
* [Hardware Bench Results](docs/HARDWARE_BENCH_RESULTS.md)
* [Architecture](docs/ARCHITECTURE.md)
* [Phase 6 Test Matrix](docs/PHASE6_TEST_MATRIX.md)
* [Phase 6 Results Summary](docs/PHASE6_RESULTS_SUMMARY.md)

---

# 12. License

This project is released under the [MIT License](LICENSE).
