# Runbook (WSL2 Simulation) — SITL + Mission Planner + MAVROS + SEANO

This runbook provides a repeatable startup procedure for the simulation stack.

## Baseline Port Map

- Mission Planner (Windows): `14550/UDP`
- MAVROS ↔ SITL (WSL): `14551/UDP`
- MAVProxy/ArduPilot master (internal): `5760/TCP`

WSL2 note:
- Windows host IP as seen from WSL can change each session.
- Always compute it from the WSL default gateway.

---

## Startup Order (Required)

Start components in this order:
1) ArduPilot SITL (WSL)
2) Mission Planner (Windows)
3) MAVROS2 (WSL)
4) SEANO ROS 2 stack (WSL)

If SITL is restarted, restart MAVROS as well.

---

## Terminal 1 (WSL) — Start ArduPilot SITL (ArduRover rover-skid)

```bash
cd ~/tools/ardupilot

WIN_HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
echo "WIN_HOST_IP=$WIN_HOST_IP"

sim_vehicle.py -v Rover -f rover-skid --console --map \
  --out udp:${WIN_HOST_IP}:14550 \
  --out udp:127.0.0.1:14551
````

Expected:

* MAVProxy console and map open.
* SITL outputs to both MP (14550) and MAVROS (14551).

---

## Mission Planner (Windows) — Connect

* Connection type: UDP
* Port: 14550
* Click Connect

Expected:

* Vehicle status visible (mode/params).
* Map shows vehicle.

---

## Terminal 2 (WSL) — Start MAVROS2

```bash
source /opt/ros/humble/setup.bash
ros2 launch mavros apm.launch fcu_url:=udp://0.0.0.0:14551@127.0.0.1:14551
```

Expected:

* `/mavros/state` becomes `connected: true`.

---

## Terminal 3 (WSL) — Monitor MAVROS state

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mavros/state
```

Expected:

* `connected: true`
* `mode` readable
* `armed` reflects current state

---

## Terminal 4 (WSL) — Start SEANO stack

```bash
cd ~/seano-collision-avoidance/seano_ca_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch seano_vision run_auto_stack.launch.py
```

Expected:

* Nodes start without fatal errors.
* Failsafe heartbeat publishes (`/ca/failsafe_active`).

---

## Validation Commands

### 1) MAVROS connected

```bash
ros2 topic echo /mavros/state
```

### 2) Failsafe heartbeat frequency

```bash
ros2 topic hz /ca/failsafe_active
```

### 3) RC override publishing

```bash
ros2 topic echo /mavros/rc/override
```

### 4) Duplicate publisher check (if behavior is inconsistent)

```bash
ros2 topic info -v /seano/auto/left_cmd
ros2 topic info -v /seano/auto/right_cmd
```

---

## Common Failure Modes

### Mission Planner cannot connect

* Confirm `WIN_HOST_IP` was computed via `ip route`.
* Confirm SITL includes `--out udp:${WIN_HOST_IP}:14550`.

### MAVROS connected: false

* Confirm SITL is running and outputs `udp:127.0.0.1:14551`.
* Confirm MAVROS uses:
  `udp://0.0.0.0:14551@127.0.0.1:14551`.
* Restart MAVROS after restarting SITL.

### Vehicle does not move while PWM changes

* Ensure vehicle is ARMED.
* Ensure mode accepts RC override (commonly MANUAL/STEERING during testing).
* Ensure safety limiter is not forcing STOP (`FAILSAFE_STOP` logs).
