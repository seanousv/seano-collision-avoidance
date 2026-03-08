#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _bool_by_profile(
    profile_lc: LaunchConfiguration, enabled_profiles: list[str]
) -> PythonExpression:
    profile_list = ", ".join([f"'{p}'" for p in enabled_profiles])
    return PythonExpression(
        [
            "'true' if '",
            profile_lc,
            f"' in [{profile_list}] else 'false'",
        ]
    )


def _str_by_profile(
    profile_lc: LaunchConfiguration,
    full_value: str,
    reduced_value: str,
) -> PythonExpression:
    return PythonExpression(
        [
            f"'{full_value}' if '",
            profile_lc,
            "' == 'full' else '",
            reduced_value,
            "'",
        ]
    )


def generate_launch_description():
    pkg_share = FindPackageShare("seano_vision")

    # ------------------------------------------------------------------
    # Common args
    # ------------------------------------------------------------------
    record = LaunchConfiguration("record")
    bag_name = LaunchConfiguration("bag_name")

    master_enable_on_start = LaunchConfiguration("master_enable_on_start")
    failsafe_stale_is_active = LaunchConfiguration("failsafe_stale_is_active")

    input_mode = LaunchConfiguration("input_mode")
    output_mode = LaunchConfiguration("output_mode")

    avoid_mode = LaunchConfiguration("avoid_mode")
    mission_mode_default = LaunchConfiguration("mission_mode_default")
    failsafe_mode = LaunchConfiguration("failsafe_mode")

    # ------------------------------------------------------------------
    # Hardware-oriented runtime profile
    #
    # usb_light     = USB camera + detector + risk
    # usb_watchdog  = usb_light + watchdog
    # full          = semua perception path aktif
    # ------------------------------------------------------------------
    ca_runtime_profile = LaunchConfiguration("ca_runtime_profile")

    # ------------------------------------------------------------------
    # Main toggles
    # ------------------------------------------------------------------
    use_ca_pipeline = LaunchConfiguration("use_ca_pipeline")
    use_takeover_manager = LaunchConfiguration("use_takeover_manager")
    use_mode_manager = LaunchConfiguration("use_mode_manager")

    # ------------------------------------------------------------------
    # Camera / CA include args
    # ------------------------------------------------------------------
    ca_camera_launch = LaunchConfiguration("ca_camera_launch")
    ca_image_topic = LaunchConfiguration("ca_image_topic")

    ca_use_camera = LaunchConfiguration("ca_use_camera")
    ca_use_detector = LaunchConfiguration("ca_use_detector")
    ca_use_waterline = LaunchConfiguration("ca_use_waterline")
    ca_use_fp_guard = LaunchConfiguration("ca_use_fp_guard")
    ca_use_fusion = LaunchConfiguration("ca_use_fusion")
    ca_use_vq = LaunchConfiguration("ca_use_vq")
    ca_use_freeze = LaunchConfiguration("ca_use_freeze")
    ca_use_risk = LaunchConfiguration("ca_use_risk")
    ca_use_watchdog = LaunchConfiguration("ca_use_watchdog")
    ca_use_ca_viewer = LaunchConfiguration("ca_use_ca_viewer")
    ca_use_wl_viewer = LaunchConfiguration("ca_use_wl_viewer")

    ca_det_sub_reliability = LaunchConfiguration("ca_det_sub_reliability")
    ca_det_pub_reliability = LaunchConfiguration("ca_det_pub_reliability")
    ca_det_qos_depth = LaunchConfiguration("ca_det_qos_depth")

    wd_startup_grace_s = LaunchConfiguration("wd_startup_grace_s")
    wd_start_in_failsafe = LaunchConfiguration("wd_start_in_failsafe")

    # ------------------------------------------------------------------
    # Mux / limiter tuning
    # ------------------------------------------------------------------
    mux_command_timeout_s = LaunchConfiguration("mux_command_timeout_s")

    limiter_input_timeout_s = LaunchConfiguration("limiter_input_timeout_s")
    limiter_failsafe_timeout_s = LaunchConfiguration("limiter_failsafe_timeout_s")
    limiter_loop_hz = LaunchConfiguration("limiter_loop_hz")

    # ------------------------------------------------------------------
    # Bridge / RC / PWM
    # ------------------------------------------------------------------
    rc_left_chan = LaunchConfiguration("rc_left_chan")
    rc_right_chan = LaunchConfiguration("rc_right_chan")
    rc_steer_chan = LaunchConfiguration("rc_steer_chan")
    rc_throttle_chan = LaunchConfiguration("rc_throttle_chan")

    pwm_neutral = LaunchConfiguration("pwm_neutral")
    pwm_fwd_max = LaunchConfiguration("pwm_fwd_max")
    pwm_rev_min = LaunchConfiguration("pwm_rev_min")
    pwm_steer_left = LaunchConfiguration("pwm_steer_left")
    pwm_steer_right = LaunchConfiguration("pwm_steer_right")
    pwm_output_min = LaunchConfiguration("pwm_output_min")
    pwm_output_max = LaunchConfiguration("pwm_output_max")

    allow_reverse = LaunchConfiguration("allow_reverse")
    bridge_pub_hz = LaunchConfiguration("bridge_pub_hz")
    bridge_command_timeout_s = LaunchConfiguration("bridge_command_timeout_s")
    bridge_pwm_slew_rate_us_per_s = LaunchConfiguration("bridge_pwm_slew_rate_us_per_s")

    # ------------------------------------------------------------------
    # Takeover manager tuning
    # ------------------------------------------------------------------
    cruise_speed = LaunchConfiguration("cruise_speed")
    turn_cmd = LaunchConfiguration("turn_cmd")
    diff_mix_gain = LaunchConfiguration("diff_mix_gain")

    # ------------------------------------------------------------------
    # CA include
    # ------------------------------------------------------------------
    ca_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "demo_full_ca.launch.py"])
        ),
        condition=IfCondition(use_ca_pipeline),
        launch_arguments={
            "camera_launch": ca_camera_launch,
            "image_topic": ca_image_topic,
            "use_camera": ca_use_camera,
            "use_detector": ca_use_detector,
            "use_waterline": ca_use_waterline,
            "use_fp_guard": ca_use_fp_guard,
            "use_fusion": ca_use_fusion,
            "use_vq": ca_use_vq,
            "use_freeze": ca_use_freeze,
            "use_risk": ca_use_risk,
            "use_watchdog": ca_use_watchdog,
            "use_ca_viewer": ca_use_ca_viewer,
            "use_wl_viewer": ca_use_wl_viewer,
            "det_sub_reliability": ca_det_sub_reliability,
            "det_pub_reliability": ca_det_pub_reliability,
            "det_qos_depth": ca_det_qos_depth,
            "wd_startup_grace_s": wd_startup_grace_s,
            "wd_start_in_failsafe": wd_start_in_failsafe,
        }.items(),
    )

    # ------------------------------------------------------------------
    # mux -> limiter -> bridge
    # ------------------------------------------------------------------
    mux = Node(
        package="seano_vision",
        executable="command_mux_node",
        name="command_mux_node",
        output="screen",
        parameters=[
            {
                "manual_left_topic": "/seano/manual/left_cmd",
                "manual_right_topic": "/seano/manual/right_cmd",
                "auto_left_topic": "/seano/auto/left_cmd",
                "auto_right_topic": "/seano/auto/right_cmd",
                "out_left_topic": "/seano/selected/left_cmd",
                "out_right_topic": "/seano/selected/right_cmd",
                "auto_enable_topic": "/seano/auto_enable",
                "fallback_to_manual": True,
                "command_timeout_s": ParameterValue(mux_command_timeout_s, value_type=float),
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
            }
        ],
    )

    limiter = Node(
        package="seano_vision",
        executable="actuator_safety_limiter_node",
        name="actuator_safety_limiter_node",
        output="screen",
        parameters=[
            {
                "in_left_topic": "/seano/selected/left_cmd",
                "in_right_topic": "/seano/selected/right_cmd",
                "out_left_topic": "/seano/left_cmd",
                "out_right_topic": "/seano/right_cmd",
                "failsafe_active_topic": "/ca/failsafe_active",
                "failsafe_stale_is_active": ParameterValue(
                    failsafe_stale_is_active, value_type=bool
                ),
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
                "input_timeout_s": ParameterValue(limiter_input_timeout_s, value_type=float),
                "failsafe_timeout_s": ParameterValue(limiter_failsafe_timeout_s, value_type=float),
                "loop_hz": ParameterValue(limiter_loop_hz, value_type=float),
                "reason_topic": "/seano/limiter_reason",
            }
        ],
    )

    bridge = Node(
        package="seano_vision",
        executable="mavros_rc_override_bridge_node",
        name="mavros_rc_override_bridge_node",
        output="screen",
        parameters=[
            {
                "input_mode": input_mode,
                "output_mode": output_mode,
                "left_topic": "/seano/left_cmd",
                "right_topic": "/seano/right_cmd",
                "out_topic": "/mavros/rc/override",
                "allow_reverse": ParameterValue(allow_reverse, value_type=bool),
                "override_enable_topic": "/seano/rc_override_enable",
                "override_enabled_default": False,
                "publish_release_when_disabled": True,
                # channel mapping untuk hardware final
                "rc_left_chan": ParameterValue(rc_left_chan, value_type=int),
                "rc_right_chan": ParameterValue(rc_right_chan, value_type=int),
                # tetap disediakan jika sewaktu-waktu fallback ke profile SITL
                "rc_steer_chan": ParameterValue(rc_steer_chan, value_type=int),
                "rc_throttle_chan": ParameterValue(rc_throttle_chan, value_type=int),
                # PWM
                "pwm_neutral": ParameterValue(pwm_neutral, value_type=int),
                "pwm_fwd_max": ParameterValue(pwm_fwd_max, value_type=int),
                "pwm_rev_min": ParameterValue(pwm_rev_min, value_type=int),
                "pwm_steer_left": ParameterValue(pwm_steer_left, value_type=int),
                "pwm_steer_right": ParameterValue(pwm_steer_right, value_type=int),
                "pwm_output_min": ParameterValue(pwm_output_min, value_type=int),
                "pwm_output_max": ParameterValue(pwm_output_max, value_type=int),
                "pwm_slew_rate_us_per_s": ParameterValue(
                    bridge_pwm_slew_rate_us_per_s, value_type=float
                ),
                "pub_hz": ParameterValue(bridge_pub_hz, value_type=float),
                "command_timeout_s": ParameterValue(bridge_command_timeout_s, value_type=float),
            }
        ],
    )

    # ------------------------------------------------------------------
    # takeover manager
    # ------------------------------------------------------------------
    takeover = Node(
        package="seano_vision",
        executable="auto_controller_stub_node",
        name="auto_controller_stub_node",
        output="screen",
        condition=IfCondition(use_takeover_manager),
        parameters=[
            {
                "command_topic": "/ca/command_safe",
                "failsafe_active_topic": "/ca/failsafe_active",
                "out_left_topic": "/seano/auto/left_cmd",
                "out_right_topic": "/seano/auto/right_cmd",
                "auto_enable_topic": "/seano/auto_enable",
                "rc_override_enable_topic": "/seano/rc_override_enable",
                "master_enable_topic": "/seano/auto_master_enable",
                "master_enable_on_start": ParameterValue(master_enable_on_start, value_type=bool),
                "cruise_speed": ParameterValue(cruise_speed, value_type=float),
                "turn_cmd": ParameterValue(turn_cmd, value_type=float),
                "diff_mix_gain": ParameterValue(diff_mix_gain, value_type=float),
            }
        ],
    )

    # ------------------------------------------------------------------
    # mission / mode manager
    # ------------------------------------------------------------------
    mode_mgr = Node(
        package="seano_vision",
        executable="mission_mode_manager_node",
        name="mission_mode_manager_node",
        output="screen",
        condition=IfCondition(use_mode_manager),
        parameters=[
            {
                "avoid_mode": avoid_mode,
                "mission_mode_default": mission_mode_default,
                "failsafe_mode": failsafe_mode,
                "switch_to_avoid_on_takeover": True,
                "restore_mode_on_release": True,
                "switch_to_failsafe_on_failsafe": True,
                "restore_after_failsafe_if_clear": True,
                "min_mode_switch_interval_s": 1.0,
            }
        ],
    )

    # ------------------------------------------------------------------
    # rosbag record
    # ------------------------------------------------------------------
    bag_dir = PathJoinSubstitution([EnvironmentVariable("HOME"), "bags"])
    bag_path = PathJoinSubstitution([bag_dir, bag_name])

    topics = [
        "/ca/command",
        "/ca/command_safe",
        "/ca/failsafe_active",
        "/ca/failsafe_reason",
        "/ca/mode",
        "/ca/watchdog_status",
        "/vision/freeze",
        "/vision/freeze_reason",
        "/seano/auto_master_enable",
        "/seano/auto_enable",
        "/seano/rc_override_enable",
        "/seano/manual/left_cmd",
        "/seano/manual/right_cmd",
        "/seano/auto/left_cmd",
        "/seano/auto/right_cmd",
        "/seano/selected/left_cmd",
        "/seano/selected/right_cmd",
        "/seano/left_cmd",
        "/seano/right_cmd",
        "/seano/limiter_reason",
        "/mavros/state",
        "/mavros/rc/override",
        "/mavros/rc/in",
        "/ca/mode_manager_state",
        "/ca/mode_manager_event",
    ]

    cond_record = IfCondition(PythonExpression(["('", record, "'.lower() == 'true')"]))

    bag_record = ExecuteProcess(
        condition=cond_record,
        cmd=["ros2", "bag", "record", "-o", bag_path, *topics],
        output="screen",
    )

    return LaunchDescription(
        [
            # ------------------------------------------------------------------
            # core
            # ------------------------------------------------------------------
            DeclareLaunchArgument("record", default_value="false"),
            DeclareLaunchArgument("bag_name", default_value="phase7_cuav_usb_e2e"),
            DeclareLaunchArgument("master_enable_on_start", default_value="false"),
            DeclareLaunchArgument("failsafe_stale_is_active", default_value="true"),
            DeclareLaunchArgument("use_ca_pipeline", default_value="true"),
            DeclareLaunchArgument("use_takeover_manager", default_value="true"),
            DeclareLaunchArgument("use_mode_manager", default_value="true"),
            # ------------------------------------------------------------------
            # hardware runtime profile
            # ------------------------------------------------------------------
            DeclareLaunchArgument(
                "ca_runtime_profile",
                default_value="full",
                description="usb_light | usb_watchdog | full",
            ),
            DeclareLaunchArgument(
                "ca_camera_launch",
                default_value="phase2_camera_usb_test.launch.py",
            ),
            DeclareLaunchArgument(
                "ca_image_topic",
                default_value="/seano/camera/image_raw_reliable",
            ),
            DeclareLaunchArgument(
                "ca_use_camera",
                default_value=_bool_by_profile(
                    ca_runtime_profile, ["usb_light", "usb_watchdog", "full"]
                ),
            ),
            DeclareLaunchArgument(
                "ca_use_detector",
                default_value=_bool_by_profile(
                    ca_runtime_profile, ["usb_light", "usb_watchdog", "full"]
                ),
            ),
            DeclareLaunchArgument(
                "ca_use_waterline",
                default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
            ),
            DeclareLaunchArgument(
                "ca_use_fp_guard",
                default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
            ),
            DeclareLaunchArgument(
                "ca_use_fusion",
                default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
            ),
            DeclareLaunchArgument(
                "ca_use_vq",
                default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
            ),
            DeclareLaunchArgument(
                "ca_use_freeze",
                default_value=_bool_by_profile(ca_runtime_profile, ["full"]),
            ),
            DeclareLaunchArgument(
                "ca_use_risk",
                default_value=_bool_by_profile(
                    ca_runtime_profile, ["usb_light", "usb_watchdog", "full"]
                ),
            ),
            DeclareLaunchArgument(
                "ca_use_watchdog",
                default_value=_bool_by_profile(ca_runtime_profile, ["usb_watchdog", "full"]),
            ),
            DeclareLaunchArgument("ca_use_ca_viewer", default_value="false"),
            DeclareLaunchArgument("ca_use_wl_viewer", default_value="false"),
            DeclareLaunchArgument("ca_det_sub_reliability", default_value="reliable"),
            DeclareLaunchArgument("ca_det_pub_reliability", default_value="reliable"),
            DeclareLaunchArgument("ca_det_qos_depth", default_value="10"),
            DeclareLaunchArgument(
                "wd_startup_grace_s",
                default_value=_str_by_profile(ca_runtime_profile, "4.0", "8.0"),
            ),
            DeclareLaunchArgument("wd_start_in_failsafe", default_value="false"),
            # ------------------------------------------------------------------
            # mux / limiter
            # ------------------------------------------------------------------
            DeclareLaunchArgument("mux_command_timeout_s", default_value="0.6"),
            DeclareLaunchArgument("limiter_input_timeout_s", default_value="0.6"),
            DeclareLaunchArgument("limiter_failsafe_timeout_s", default_value="2.0"),
            DeclareLaunchArgument("limiter_loop_hz", default_value="20.0"),
            # ------------------------------------------------------------------
            # bridge / hardware output
            # ------------------------------------------------------------------
            DeclareLaunchArgument("input_mode", default_value="left_right"),
            DeclareLaunchArgument("output_mode", default_value="rc_left_right"),
            DeclareLaunchArgument("rc_left_chan", default_value="1"),
            DeclareLaunchArgument("rc_right_chan", default_value="3"),
            DeclareLaunchArgument("rc_steer_chan", default_value="1"),
            DeclareLaunchArgument("rc_throttle_chan", default_value="3"),
            DeclareLaunchArgument("pwm_neutral", default_value="1500"),
            DeclareLaunchArgument("pwm_fwd_max", default_value="1900"),
            DeclareLaunchArgument("pwm_rev_min", default_value="1100"),
            DeclareLaunchArgument("pwm_steer_left", default_value="1100"),
            DeclareLaunchArgument("pwm_steer_right", default_value="1900"),
            DeclareLaunchArgument("pwm_output_min", default_value="1000"),
            DeclareLaunchArgument("pwm_output_max", default_value="2000"),
            DeclareLaunchArgument("allow_reverse", default_value="false"),
            DeclareLaunchArgument("bridge_pub_hz", default_value="20.0"),
            DeclareLaunchArgument("bridge_command_timeout_s", default_value="0.5"),
            DeclareLaunchArgument(
                "bridge_pwm_slew_rate_us_per_s",
                default_value="250.0",
            ),
            # ------------------------------------------------------------------
            # mission / mode policy
            # ------------------------------------------------------------------
            DeclareLaunchArgument("avoid_mode", default_value="MANUAL"),
            DeclareLaunchArgument("mission_mode_default", default_value="AUTO"),
            DeclareLaunchArgument("failsafe_mode", default_value="MANUAL"),
            # ------------------------------------------------------------------
            # takeover tuning
            # ------------------------------------------------------------------
            DeclareLaunchArgument("cruise_speed", default_value="0.30"),
            DeclareLaunchArgument("turn_cmd", default_value="0.55"),
            DeclareLaunchArgument("diff_mix_gain", default_value="0.7"),
            # ------------------------------------------------------------------
            # actions
            # ------------------------------------------------------------------
            ca_include,
            mux,
            limiter,
            bridge,
            takeover,
            mode_mgr,
            bag_record,
        ]
    )
