#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ---------- Launch args ----------
    # Topics
    command_topic = LaunchConfiguration("command_topic")
    failsafe_active_topic = LaunchConfiguration("failsafe_active_topic")

    out_throttle_topic = LaunchConfiguration("out_throttle_topic")
    out_rudder_topic = LaunchConfiguration("out_rudder_topic")

    out_left_topic = LaunchConfiguration("out_left_topic")
    out_right_topic = LaunchConfiguration("out_right_topic")

    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    mavros_out_topic = LaunchConfiguration("mavros_out_topic")

    # Limiter behavior
    command_timeout_s = LaunchConfiguration("command_timeout_s")
    failsafe_stale_is_active = LaunchConfiguration("failsafe_stale_is_active")
    loop_hz = LaunchConfiguration("loop_hz")

    # Bridge behavior
    enable_bridge = LaunchConfiguration("enable_bridge")

    input_mode = LaunchConfiguration("input_mode")  # thr_steer | left_right | twist
    output_mode = LaunchConfiguration("output_mode")  # rc_thr_steer | rc_left_right

    # Channel mapping
    rc_steer_chan = LaunchConfiguration("rc_steer_chan")
    rc_throttle_chan = LaunchConfiguration("rc_throttle_chan")
    rc_left_chan = LaunchConfiguration("rc_left_chan")
    rc_right_chan = LaunchConfiguration("rc_right_chan")

    # PWM calibration
    pwm_neutral = LaunchConfiguration("pwm_neutral")
    pwm_fwd_max = LaunchConfiguration("pwm_fwd_max")
    pwm_rev_min = LaunchConfiguration("pwm_rev_min")
    allow_reverse = LaunchConfiguration("allow_reverse")

    pwm_steer_left = LaunchConfiguration("pwm_steer_left")
    pwm_steer_right = LaunchConfiguration("pwm_steer_right")

    pwm_output_min = LaunchConfiguration("pwm_output_min")
    pwm_output_max = LaunchConfiguration("pwm_output_max")

    # Mixer / twist
    diff_mix_gain = LaunchConfiguration("diff_mix_gain")
    twist_v_max = LaunchConfiguration("twist_v_max")
    twist_yaw_max = LaunchConfiguration("twist_yaw_max")

    # Safety in bridge
    bridge_timeout_s = LaunchConfiguration("bridge_timeout_s")
    pub_hz = LaunchConfiguration("pub_hz")
    pwm_slew_rate = LaunchConfiguration("pwm_slew_rate_us_per_s")
    log_period_s = LaunchConfiguration("log_period_s")

    # Test mode
    test_enable = LaunchConfiguration("test_enable")
    test_throttle = LaunchConfiguration("test_throttle")
    test_steer = LaunchConfiguration("test_steer")
    test_left = LaunchConfiguration("test_left")
    test_right = LaunchConfiguration("test_right")

    # ---------- Nodes ----------
    actuator_limiter = Node(
        package="seano_vision",
        executable="actuator_safety_limiter_node",
        name="actuator_safety_limiter_node",
        output="screen",
        parameters=[
            {
                "command_topic": command_topic,
                "failsafe_active_topic": failsafe_active_topic,
                "out_throttle_topic": out_throttle_topic,
                "out_rudder_topic": out_rudder_topic,
                "command_timeout_s": ParameterValue(command_timeout_s, value_type=float),
                "failsafe_stale_is_active": ParameterValue(
                    failsafe_stale_is_active, value_type=bool
                ),
                "loop_hz": ParameterValue(loop_hz, value_type=float),
                "publish_twist": False,
            }
        ],
    )

    mavros_bridge = Node(
        package="seano_vision",
        executable="mavros_rc_override_bridge_node",
        name="mavros_rc_override_bridge_node",
        output="screen",
        parameters=[
            {
                # topics
                "thr_topic": out_throttle_topic,
                "steer_topic": out_rudder_topic,
                "left_topic": out_left_topic,
                "right_topic": out_right_topic,
                "cmd_vel_topic": cmd_vel_topic,
                "out_topic": mavros_out_topic,
                # enable + modes
                "enable": ParameterValue(enable_bridge, value_type=bool),
                "input_mode": input_mode,
                "output_mode": output_mode,
                # channel mapping
                "rc_steer_chan": ParameterValue(rc_steer_chan, value_type=int),
                "rc_throttle_chan": ParameterValue(rc_throttle_chan, value_type=int),
                "rc_left_chan": ParameterValue(rc_left_chan, value_type=int),
                "rc_right_chan": ParameterValue(rc_right_chan, value_type=int),
                # pwm calibration
                "pwm_neutral": ParameterValue(pwm_neutral, value_type=int),
                "pwm_fwd_max": ParameterValue(pwm_fwd_max, value_type=int),
                "pwm_rev_min": ParameterValue(pwm_rev_min, value_type=int),
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
                "pwm_steer_left": ParameterValue(pwm_steer_left, value_type=int),
                "pwm_steer_right": ParameterValue(pwm_steer_right, value_type=int),
                "pwm_output_min": ParameterValue(pwm_output_min, value_type=int),
                "pwm_output_max": ParameterValue(pwm_output_max, value_type=int),
                # mixer / twist
                "diff_mix_gain": ParameterValue(diff_mix_gain, value_type=float),
                "twist_v_max": ParameterValue(twist_v_max, value_type=float),
                "twist_yaw_max": ParameterValue(twist_yaw_max, value_type=float),
                # safety + timing
                "command_timeout_s": ParameterValue(bridge_timeout_s, value_type=float),
                "pub_hz": ParameterValue(pub_hz, value_type=float),
                "pwm_slew_rate_us_per_s": ParameterValue(pwm_slew_rate, value_type=float),
                "log_period_s": ParameterValue(log_period_s, value_type=float),
                # test mode
                "test_enable": ParameterValue(test_enable, value_type=bool),
                "test_throttle": ParameterValue(test_throttle, value_type=float),
                "test_steer": ParameterValue(test_steer, value_type=float),
                "test_left": ParameterValue(test_left, value_type=float),
                "test_right": ParameterValue(test_right, value_type=float),
            }
        ],
    )

    return LaunchDescription(
        [
            # Topics
            DeclareLaunchArgument("command_topic", default_value="/ca/command"),
            DeclareLaunchArgument("failsafe_active_topic", default_value="/ca/failsafe_active"),
            DeclareLaunchArgument("out_throttle_topic", default_value="/seano/throttle_cmd"),
            DeclareLaunchArgument("out_rudder_topic", default_value="/seano/rudder_cmd"),
            DeclareLaunchArgument("out_left_topic", default_value="/seano/left_cmd"),
            DeclareLaunchArgument("out_right_topic", default_value="/seano/right_cmd"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("mavros_out_topic", default_value="/mavros/rc/override"),
            # Limiter
            DeclareLaunchArgument("command_timeout_s", default_value="5.0"),
            DeclareLaunchArgument("failsafe_stale_is_active", default_value="false"),
            DeclareLaunchArgument("loop_hz", default_value="20.0"),
            # Bridge
            DeclareLaunchArgument("enable_bridge", default_value="true"),
            DeclareLaunchArgument("input_mode", default_value="thr_steer"),
            DeclareLaunchArgument("output_mode", default_value="rc_thr_steer"),
            DeclareLaunchArgument("rc_steer_chan", default_value="1"),
            DeclareLaunchArgument("rc_throttle_chan", default_value="3"),
            DeclareLaunchArgument("rc_left_chan", default_value="1"),
            DeclareLaunchArgument("rc_right_chan", default_value="3"),
            DeclareLaunchArgument("pwm_neutral", default_value="1500"),
            DeclareLaunchArgument("pwm_fwd_max", default_value="1900"),
            DeclareLaunchArgument("pwm_rev_min", default_value="1100"),
            DeclareLaunchArgument("allow_reverse", default_value="false"),
            DeclareLaunchArgument("pwm_steer_left", default_value="1100"),
            DeclareLaunchArgument("pwm_steer_right", default_value="1900"),
            DeclareLaunchArgument("pwm_output_min", default_value="1000"),
            DeclareLaunchArgument("pwm_output_max", default_value="2000"),
            DeclareLaunchArgument("diff_mix_gain", default_value="1.0"),
            DeclareLaunchArgument("twist_v_max", default_value="1.0"),
            DeclareLaunchArgument("twist_yaw_max", default_value="1.0"),
            DeclareLaunchArgument("bridge_timeout_s", default_value="0.5"),
            DeclareLaunchArgument("pub_hz", default_value="20.0"),
            DeclareLaunchArgument("pwm_slew_rate_us_per_s", default_value="0.0"),
            DeclareLaunchArgument("log_period_s", default_value="2.0"),
            # Test mode
            DeclareLaunchArgument("test_enable", default_value="false"),
            DeclareLaunchArgument("test_throttle", default_value="0.30"),
            DeclareLaunchArgument("test_steer", default_value="0.0"),
            DeclareLaunchArgument("test_left", default_value="0.30"),
            DeclareLaunchArgument("test_right", default_value="0.30"),
            actuator_limiter,
            mavros_bridge,
        ]
    )
