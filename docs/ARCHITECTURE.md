# Architecture Overview

This document describes the runtime architecture and ROS 2 interfaces used by the SEANO collision avoidance module.

## Goals

- Provide a deterministic actuation pipeline suitable for a differential-thrust USV (left/right thrusters).
- Keep autonomy logic (risk/decision) separated from safety and actuation.
- Enable simulation validation via ArduPilot SITL + MAVROS + GCS.

## High-level data flow

```mermaid
flowchart LR
  subgraph Inputs
    KBD["Keyboard Teleop"] --> M1["/seano/manual/left_cmd & /seano/manual/right_cmd"]
    AUTO["Autonomy Logic (risk/planner)"] --> A1["/seano/auto/left_cmd & /seano/auto/right_cmd"]
    EN["/seano/auto_enable (Bool)"] --> MUX
    FS["/ca/failsafe_active (Bool)"] --> LIM
  end

  M1 --> MUX["Command MUX"]
  A1 --> MUX
  MUX --> SEL["/seano/selected/left_cmd & /seano/selected/right_cmd"]

  SEL --> LIM["Safety Limiter"]
  LIM --> OUT["/seano/left_cmd & /seano/right_cmd"]

  OUT --> BR["RC Override Bridge"]
  BR --> RC["/mavros/rc/override"]
  RC --> FCU["ArduPilot FCU (SITL / HW)"]
````

## Core design decision: left/right as the primary control interface

The vehicle uses differential thrust without a rudder. The system therefore uses:

* `left_cmd` and `right_cmd` as the main internal command signals

This avoids ambiguity that can appear when using throttle + steering on skid/differential models.

## Nodes and responsibilities

### 1) `teleop_diff_thruster_node`

* Purpose: manual control using keyboard input (e.g., WASD)
* Publishes:

  * `/seano/manual/left_cmd` (std_msgs/Float32)
  * `/seano/manual/right_cmd` (std_msgs/Float32)

### 2) `command_mux_node`

* Purpose: selects manual vs autonomous command source
* Inputs:

  * `/seano/manual/left_cmd`, `/seano/manual/right_cmd`
  * `/seano/auto/left_cmd`, `/seano/auto/right_cmd`
  * `/seano/auto_enable` (std_msgs/Bool)
* Outputs:

  * `/seano/selected/left_cmd`
  * `/seano/selected/right_cmd`

### 3) `actuator_safety_limiter_node`

* Purpose: safety enforcement and failsafe policy
* Typical responsibilities:

  * command timeout handling
  * output clamp / range limiting
  * safe-stop behavior when failsafe is active or command is stale
* Inputs:

  * `/seano/selected/left_cmd`, `/seano/selected/right_cmd`
  * `/ca/failsafe_active` (std_msgs/Bool)
* Outputs:

  * `/seano/left_cmd`
  * `/seano/right_cmd`

### 4) `mavros_rc_override_bridge_node`

* Purpose: converts `left/right` into PWM RC override commands
* Inputs:

  * `/seano/left_cmd`, `/seano/right_cmd`
* Output:

  * `/mavros/rc/override` (mavros_msgs/OverrideRCIn)

## Topic reference

| Topic                       | Type                     | Direction | Notes                                    |
| --------------------------- | ------------------------ | --------- | ---------------------------------------- |
| `/seano/manual/left_cmd`    | std_msgs/Float32         | pub       | manual left command                      |
| `/seano/manual/right_cmd`   | std_msgs/Float32         | pub       | manual right command                     |
| `/seano/auto/left_cmd`      | std_msgs/Float32         | pub       | autonomy left command                    |
| `/seano/auto/right_cmd`     | std_msgs/Float32         | pub       | autonomy right command                   |
| `/seano/auto_enable`        | std_msgs/Bool            | pub       | enables autonomous selection in MUX      |
| `/seano/selected/left_cmd`  | std_msgs/Float32         | pub       | mux output (pre-limiter)                 |
| `/seano/selected/right_cmd` | std_msgs/Float32         | pub       | mux output (pre-limiter)                 |
| `/ca/failsafe_active`       | std_msgs/Bool            | pub       | safety state (heartbeat + fail triggers) |
| `/seano/left_cmd`           | std_msgs/Float32         | pub       | final left command (post limiter)        |
| `/seano/right_cmd`          | std_msgs/Float32         | pub       | final right command (post limiter)       |
| `/mavros/rc/override`       | mavros_msgs/OverrideRCIn | pub       | PWM override to FCU                      |

## Extension points

* Perception: camera node → detector → tracking (optional)
* Decision: risk evaluator / behavior logic publishes to `/seano/auto/*`
* Mission integration: higher-level state machine can switch `/seano/auto_enable` and determine when override should be active

## Simulation notes

* In WSL2 simulation, SITL and MAVROS run in Linux (WSL), while GCS (Mission Planner) runs in Windows.
* Validate runtime in layers:

  1. SITL + GCS connectivity
  2. MAVROS connectivity (`/mavros/state`)
  3. RC override publishing (`/mavros/rc/override`)
  4. vehicle response (map/track changes in GCS)

```
