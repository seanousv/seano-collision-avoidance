# Support

This document explains how to get help and how to report problems effectively.

## Where to ask for help

### 1) Bug reports (something is broken)
Use GitHub Issues → **Bug report** template:
- Include your environment (WSL/Jetson, ROS2 version, MAVROS, ArduPilot/SITL).
- Include steps to reproduce and relevant logs.

### 2) Feature requests (new capability / improvement)
Use GitHub Issues → **Feature request** template:
- Explain the motivation and acceptance criteria.

### 3) Security issues
Do **not** open a public issue. Please follow the instructions in `SECURITY.md`.

## Before opening an issue (quick checks)
- Make sure you are on the latest `main` branch.
- Confirm CI status is passing on `main`.
- For simulation issues (WSL2):
  - Verify Mission Planner connection (UDP 14550).
  - Verify MAVROS connection (`/mavros/state` → `connected: true`).
  - Confirm there are no duplicate publishers on critical topics (`ros2 topic info -v <topic>`).

## Response expectations
This project is maintained on a best-effort basis. Issues with clear reproduction steps and logs will be prioritized.

## Contact
For non-public coordination, contact:
- Email: **seanousv@gmail.com**
