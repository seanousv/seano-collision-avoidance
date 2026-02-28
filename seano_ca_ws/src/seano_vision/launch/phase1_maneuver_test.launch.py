#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _auto_shutdown_timer(context, *args, **kwargs):
    """
    Jika max_cycles > 0, buat timer shutdown berdasarkan durasi total manuver.
    Ini mencegah spam FAILSAFE_STOP setelah test selesai.
    """
    try:
        max_cycles = int(context.perform_substitution(LaunchConfiguration("max_cycles")).strip())
    except Exception:
        max_cycles = 0

    if max_cycles <= 0:
        return []

    # Ambil durasi stage dari launch args (float)
    def _f(name: str, default: float) -> float:
        try:
            return float(context.perform_substitution(LaunchConfiguration(name)).strip())
        except Exception:
            return default

    warmup = _f("warmup_s", 1.0)
    t_forward = _f("t_forward", 5.0)
    t_left = _f("t_turn_left", 0.9)
    t_right = _f("t_turn_right", 0.9)
    t_stop = _f("t_stop", 2.0)

    one_cycle = t_forward + t_left + t_right + t_stop
    total = warmup + (one_cycle * max_cycles) + 2.0  # +margin 2s

    return [
        LogInfo(msg=f"[phase1] max_cycles={max_cycles} -> auto shutdown in ~{total:.1f}s"),
        TimerAction(
            period=total,
            actions=[
                LogInfo(msg="[phase1] auto shutdown timer fired -> shutting down launch"),
                EmitEvent(event=Shutdown(reason="phase1 maneuver test completed (timer)")),
            ],
        ),
    ]


def generate_launch_description():
    # ---- topics ----
    manual_left = LaunchConfiguration("manual_left_topic")
    manual_right = LaunchConfiguration("manual_right_topic")
    auto_left = LaunchConfiguration("auto_left_topic")
    auto_right = LaunchConfiguration("auto_right_topic")
    auto_enable = LaunchConfiguration("auto_enable_topic")

    selected_left = LaunchConfiguration("selected_left_topic")
    selected_right = LaunchConfiguration("selected_right_topic")
    out_left = LaunchConfiguration("out_left_topic")
    out_right = LaunchConfiguration("out_right_topic")

    mavros_rc_override = LaunchConfiguration("mavros_rc_override_topic")

    # ---- maneuver tuning ----
    warmup_s = LaunchConfiguration("warmup_s")
    t_forward = LaunchConfiguration("t_forward")
    t_turn_left = LaunchConfiguration("t_turn_left")
    t_turn_right = LaunchConfiguration("t_turn_right")
    t_stop = LaunchConfiguration("t_stop")

    base_throttle = LaunchConfiguration("base_throttle")
    turn_delta = LaunchConfiguration("turn_delta")
    max_cycles = LaunchConfiguration("max_cycles")
    repeat = LaunchConfiguration("repeat")

    # ---- mux ----
    mux_rate_hz = LaunchConfiguration("mux_rate_hz")
    command_timeout_s = LaunchConfiguration("command_timeout_s")

    # ---- limiter ----
    limiter_loop_hz = LaunchConfiguration("limiter_loop_hz")
    input_timeout_s = LaunchConfiguration("input_timeout_s")
    failsafe_stale_is_active = LaunchConfiguration("failsafe_stale_is_active")
    allow_reverse = LaunchConfiguration("allow_reverse")

    # ---- bridge ----
    bridge_input_mode = LaunchConfiguration("bridge_input_mode")
    bridge_output_mode = LaunchConfiguration("bridge_output_mode")
    rc_steer_chan = LaunchConfiguration("rc_steer_chan")
    rc_throttle_chan = LaunchConfiguration("rc_throttle_chan")
    lr_to_steer_gain = LaunchConfiguration("lr_to_steer_gain")

    # 0) Test maneuver node
    test_node = Node(
        package="seano_vision",
        executable="test_maneuver_node",
        name="test_maneuver_node",
        output="screen",
        parameters=[
            {
                "manual_left_topic": manual_left,
                "manual_right_topic": manual_right,
                "auto_enable_topic": auto_enable,
                "pub_hz": 20.0,
                "enable_pub_hz": 2.0,
                "use_base_delta": True,
                "warmup_s": ParameterValue(warmup_s, value_type=float),
                "t_forward": ParameterValue(t_forward, value_type=float),
                "t_turn_left": ParameterValue(t_turn_left, value_type=float),
                "t_turn_right": ParameterValue(t_turn_right, value_type=float),
                "t_stop": ParameterValue(t_stop, value_type=float),
                "base_throttle": ParameterValue(base_throttle, value_type=float),
                "turn_delta": ParameterValue(turn_delta, value_type=float),
                "repeat": ParameterValue(repeat, value_type=bool),
                "max_cycles": ParameterValue(max_cycles, value_type=int),
            }
        ],
    )

    # 1) MUX
    mux = Node(
        package="seano_vision",
        executable="command_mux_node",
        name="command_mux_node",
        output="screen",
        parameters=[
            {
                "auto_left_topic": auto_left,
                "auto_right_topic": auto_right,
                "manual_left_topic": manual_left,
                "manual_right_topic": manual_right,
                "auto_enable_topic": auto_enable,
                "out_left_topic": selected_left,
                "out_right_topic": selected_right,
                "rate_hz": ParameterValue(mux_rate_hz, value_type=float),
                "command_timeout_s": ParameterValue(command_timeout_s, value_type=float),
                "fallback_to_manual": True,
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
                "output_min": 0.0,
                "output_max": 1.0,
            }
        ],
    )

    # 2) Limiter
    limiter = Node(
        package="seano_vision",
        executable="actuator_safety_limiter_node",
        name="actuator_safety_limiter_node",
        output="screen",
        parameters=[
            {
                "in_left_topic": selected_left,
                "in_right_topic": selected_right,
                "out_left_topic": out_left,
                "out_right_topic": out_right,
                "loop_hz": ParameterValue(limiter_loop_hz, value_type=float),
                "input_timeout_s": ParameterValue(input_timeout_s, value_type=float),
                "failsafe_stale_is_active": ParameterValue(
                    failsafe_stale_is_active, value_type=bool
                ),
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
            }
        ],
    )

    # 3) Bridge
    bridge = Node(
        package="seano_vision",
        executable="mavros_rc_override_bridge_node",
        name="mavros_rc_override_bridge_node",
        output="screen",
        parameters=[
            {
                "input_mode": bridge_input_mode,
                "output_mode": bridge_output_mode,
                "left_topic": out_left,
                "right_topic": out_right,
                "out_topic": mavros_rc_override,
                "rc_steer_chan": ParameterValue(rc_steer_chan, value_type=int),
                "rc_throttle_chan": ParameterValue(rc_throttle_chan, value_type=int),
                "lr_to_steer_gain": ParameterValue(lr_to_steer_gain, value_type=float),
                "test_enable": False,
            }
        ],
    )

    delayed_mux = TimerAction(period=0.6, actions=[mux])
    delayed_limiter = TimerAction(period=0.8, actions=[limiter])
    delayed_bridge = TimerAction(period=1.0, actions=[bridge])

    # Shutdown juga kalau test_node benar-benar exit (bonus)
    shutdown_on_test_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=test_node,
            on_exit=[
                LogInfo(msg="[phase1] test_maneuver_node exited -> shutting down launch"),
                EmitEvent(event=Shutdown(reason="phase1 maneuver test completed (process exit)")),
            ],
        )
    )

    return LaunchDescription(
        [
            # topics
            DeclareLaunchArgument("manual_left_topic", default_value="/seano/manual/left_cmd"),
            DeclareLaunchArgument("manual_right_topic", default_value="/seano/manual/right_cmd"),
            DeclareLaunchArgument("auto_left_topic", default_value="/seano/auto/left_cmd"),
            DeclareLaunchArgument("auto_right_topic", default_value="/seano/auto/right_cmd"),
            DeclareLaunchArgument("auto_enable_topic", default_value="/seano/auto_enable"),
            DeclareLaunchArgument("selected_left_topic", default_value="/seano/selected/left_cmd"),
            DeclareLaunchArgument(
                "selected_right_topic", default_value="/seano/selected/right_cmd"
            ),
            DeclareLaunchArgument("out_left_topic", default_value="/seano/left_cmd"),
            DeclareLaunchArgument("out_right_topic", default_value="/seano/right_cmd"),
            DeclareLaunchArgument("mavros_rc_override_topic", default_value="/mavros/rc/override"),
            # maneuver (default rapi)
            DeclareLaunchArgument("warmup_s", default_value="1.0"),
            DeclareLaunchArgument("t_forward", default_value="5.0"),
            DeclareLaunchArgument("t_turn_left", default_value="0.9"),
            DeclareLaunchArgument("t_turn_right", default_value="0.9"),
            DeclareLaunchArgument("t_stop", default_value="2.0"),
            DeclareLaunchArgument("base_throttle", default_value="0.45"),
            DeclareLaunchArgument("turn_delta", default_value="0.06"),
            DeclareLaunchArgument("max_cycles", default_value="0"),
            DeclareLaunchArgument("repeat", default_value="true"),
            # mux/limiter
            DeclareLaunchArgument("mux_rate_hz", default_value="20.0"),
            DeclareLaunchArgument("command_timeout_s", default_value="0.6"),
            DeclareLaunchArgument("limiter_loop_hz", default_value="20.0"),
            DeclareLaunchArgument("input_timeout_s", default_value="0.6"),
            DeclareLaunchArgument("failsafe_stale_is_active", default_value="false"),
            DeclareLaunchArgument("allow_reverse", default_value="false"),
            # bridge
            DeclareLaunchArgument("bridge_input_mode", default_value="left_right"),
            DeclareLaunchArgument("bridge_output_mode", default_value="rc_thr_steer"),
            DeclareLaunchArgument("rc_steer_chan", default_value="1"),
            DeclareLaunchArgument("rc_throttle_chan", default_value="3"),
            DeclareLaunchArgument("lr_to_steer_gain", default_value="0.6"),
            test_node,
            delayed_mux,
            delayed_limiter,
            delayed_bridge,
            # robust auto-exit
            shutdown_on_test_exit,
            OpaqueFunction(function=_auto_shutdown_timer),
        ]
    )
