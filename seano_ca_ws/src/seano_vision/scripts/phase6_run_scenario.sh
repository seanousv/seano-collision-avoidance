#!/usr/bin/env bash
set -euo pipefail

# Phase 6 Scenario Runner — SEANO CA
#
# Tujuan:
# - Menstandardkan trigger skenario hazard / rejoin / failsafe
# - Mengurangi publish manual yang tidak konsisten
# - Menjaga input uji tetap repeatable
#
# Skenario:
#   hazard_right
#   hazard_left
#   repeated_hazard
#   zigzag3
#   manual_takeover
#   manual_failsafe
#   failsafe_recover
#   hazard_then_failsafe
#
# Catatan:
# - Untuk skenario hazard_*:
#     jalankan launch dengan use_takeover_manager:=true
# - Untuk manual_takeover:
#     jalankan launch dengan use_takeover_manager:=false
# - Untuk manual_failsafe / failsafe_recover:
#     paling aman dijalankan saat tidak ada publisher lain yang terus-menurus
#     mengendalikan /ca/failsafe_active, atau gunakan mode uji terisolasi.

TOPIC_CMD="/ca/command_safe"
TOPIC_OVERRIDE="/seano/rc_override_enable"
TOPIC_FAILSAFE="/ca/failsafe_active"

HOLD_SEC="3"
GAP_SEC="2"
CLEAR_CMD="HOLD_COURSE"
SCENARIO=""
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage:
  bash phase6_run_scenario.sh <scenario> [options]

Scenarios:
  hazard_right
  hazard_left
  repeated_hazard
  zigzag3
  manual_takeover
  manual_failsafe
  failsafe_recover
  hazard_then_failsafe

Options:
  --hold <sec>       Durasi aktif hazard / takeover / failsafe (default: 3)
  --gap <sec>        Jeda antar-segmen (default: 2)
  --cmd-topic <t>    Topic command_safe (default: /ca/command_safe)
  --ovr-topic <t>    Topic rc_override_enable (default: /seano/rc_override_enable)
  --fs-topic <t>     Topic failsafe_active (default: /ca/failsafe_active)
  --dry-run          Print langkah tanpa publish
  -h, --help         Tampilkan bantuan

Examples:
  bash phase6_run_scenario.sh hazard_right
  bash phase6_run_scenario.sh repeated_hazard --hold 3 --gap 2
  bash phase6_run_scenario.sh manual_takeover --hold 3
  bash phase6_run_scenario.sh manual_failsafe --hold 3
  bash phase6_run_scenario.sh failsafe_recover --hold 3 --gap 2
  bash phase6_run_scenario.sh hazard_then_failsafe --hold 3 --gap 2
EOF
}

log() {
  echo "[phase6_run_scenario] $*"
}

need_ros() {
  if ! command -v ros2 >/dev/null 2>&1; then
    echo "ERROR: ros2 command not found."
    echo "Source ROS2 + workspace dulu:"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source ~/seano-collision-avoidance/seano_ca_ws/install/setup.bash"
    exit 1
  fi
}

pub_cmd() {
  local value="$1"
  log "publish ${TOPIC_CMD} = ${value}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  ros2 topic pub --once "${TOPIC_CMD}" std_msgs/msg/String "{data: '${value}'}"
}

pub_override() {
  local value="$1"
  log "publish ${TOPIC_OVERRIDE} = ${value}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  ros2 topic pub --once "${TOPIC_OVERRIDE}" std_msgs/msg/Bool "{data: ${value}}"
}

pub_failsafe() {
  local value="$1"
  log "publish ${TOPIC_FAILSAFE} = ${value}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  ros2 topic pub --once "${TOPIC_FAILSAFE}" std_msgs/msg/Bool "{data: ${value}}"
}

sleep_step() {
  local sec="$1"
  log "sleep ${sec}s"
  sleep "${sec}"
}

run_hazard_right() {
  pub_cmd "TURN_RIGHT"
  sleep_step "${HOLD_SEC}"
  pub_cmd "${CLEAR_CMD}"
}

run_hazard_left() {
  pub_cmd "TURN_LEFT"
  sleep_step "${HOLD_SEC}"
  pub_cmd "${CLEAR_CMD}"
}

run_repeated_hazard() {
  pub_cmd "TURN_RIGHT"
  sleep_step "${HOLD_SEC}"
  pub_cmd "${CLEAR_CMD}"
  sleep_step "${GAP_SEC}"

  pub_cmd "TURN_LEFT"
  sleep_step "${HOLD_SEC}"
  pub_cmd "${CLEAR_CMD}"
}

run_zigzag3() {
  pub_cmd "TURN_RIGHT"
  sleep_step "${HOLD_SEC}"
  pub_cmd "${CLEAR_CMD}"
  sleep_step "${GAP_SEC}"

  pub_cmd "TURN_LEFT"
  sleep_step "${HOLD_SEC}"
  pub_cmd "${CLEAR_CMD}"
  sleep_step "${GAP_SEC}"

  pub_cmd "TURN_RIGHT"
  sleep_step "${HOLD_SEC}"
  pub_cmd "${CLEAR_CMD}"
}

run_manual_takeover() {
  pub_override "true"
  sleep_step "${HOLD_SEC}"
  pub_override "false"
}

run_manual_failsafe() {
  pub_failsafe "true"
  sleep_step "${HOLD_SEC}"
  pub_failsafe "false"
}

run_failsafe_recover() {
  pub_failsafe "true"
  sleep_step "${HOLD_SEC}"
  pub_failsafe "false"
  sleep_step "${GAP_SEC}"
}

run_hazard_then_failsafe() {
  pub_cmd "TURN_RIGHT"
  sleep_step "${HOLD_SEC}"

  pub_failsafe "true"
  sleep_step "${HOLD_SEC}"

  pub_failsafe "false"
  sleep_step "${GAP_SEC}"

  pub_cmd "${CLEAR_CMD}"
}

# --------------------------
# Parse args
# --------------------------
if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SCENARIO="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hold)
      HOLD_SEC="$2"
      shift 2
      ;;
    --gap)
      GAP_SEC="$2"
      shift 2
      ;;
    --cmd-topic)
      TOPIC_CMD="$2"
      shift 2
      ;;
    --ovr-topic)
      TOPIC_OVERRIDE="$2"
      shift 2
      ;;
    --fs-topic)
      TOPIC_FAILSAFE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

need_ros

log "scenario=${SCENARIO} hold=${HOLD_SEC}s gap=${GAP_SEC}s dry_run=${DRY_RUN}"

case "${SCENARIO}" in
  hazard_right)
    run_hazard_right
    ;;
  hazard_left)
    run_hazard_left
    ;;
  repeated_hazard)
    run_repeated_hazard
    ;;
  zigzag3)
    run_zigzag3
    ;;
  manual_takeover)
    run_manual_takeover
    ;;
  manual_failsafe)
    run_manual_failsafe
    ;;
  failsafe_recover)
    run_failsafe_recover
    ;;
  hazard_then_failsafe)
    run_hazard_then_failsafe
    ;;
  *)
    echo "ERROR: unknown scenario: ${SCENARIO}"
    usage
    exit 1
    ;;
esac

log "done"
