#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# FASE 1 - Verification Script
# Cek cepat: publisher count, hz, dan output rc override.
#
# Cara pakai:
#   1) Pastikan SITL + MAVROS connected true + ARMED
#   2) Jalankan: ros2 launch seano_vision phase1_maneuver_test.launch.py
#   3) Jalankan script ini di terminal lain.

TOPICS=(
  "/seano/manual/left_cmd"
  "/seano/manual/right_cmd"
  "/seano/left_cmd"
  "/seano/right_cmd"
  "/mavros/rc/override"
)

echo "=== [FASE 1 VERIFY] Checking ROS environment ==="
if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 command not found. Source ROS2 + workspace dulu:"
  echo "  source /opt/ros/humble/setup.bash"
  echo "  source ~/seano-collision-avoidance/seano_ca_ws/install/setup.bash"
  exit 1
fi

echo
echo "=== [1] Publisher count (idealnya 1 publisher untuk tiap command topic) ==="
for t in "${TOPICS[@]}"; do
  echo
  echo "--- ros2 topic info -v $t"
  ros2 topic info -v "$t" || true
done

echo
echo "=== [2] Rate check (sampling ~6 detik) ==="
echo "--- hz /seano/manual/left_cmd (expect ~20 Hz)"
timeout 6 ros2 topic hz /seano/manual/left_cmd || true

echo "--- hz /seano/left_cmd (expect ~20 Hz setelah mux)"
timeout 6 ros2 topic hz /seano/left_cmd || true

echo
echo "=== [3] RC override sample (PWM harus berubah saat manuver) ==="
for i in 1 2 3 4 5; do
  echo "--- sample $i"
  timeout 2 ros2 topic echo /mavros/rc/override --once || true
done

echo
echo "=== DONE ==="
echo "Interpretasi:"
echo "- Publisher count: jangan sampai dobel di /seano/manual/* atau /seano/*_cmd."
echo "- Hz: jangan 0 Hz (stale). Idealnya sekitar 20 Hz."
echo "- RC override: PWM channel berubah (bukan 1500/1500 terus) saat FORWARD/TURN."
