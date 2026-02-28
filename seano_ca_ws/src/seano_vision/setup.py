from glob import glob
import os
import sys

from setuptools import find_packages, setup

package_name = "seano_vision"


# colcon/ament kadang memanggil setup.py dengan argumen yang tidak didukung
# oleh distutils/setuptools tertentu. Kita buang argumen tsb agar build lanjut.
def _strip_flag(flag: str, takes_value: bool = False) -> None:
    while flag in sys.argv:
        i = sys.argv.index(flag)
        sys.argv.pop(i)
        if takes_value and i < len(sys.argv):
            # buang value setelah flag
            sys.argv.pop(i)


def _strip_flag_prefix(prefix: str, takes_value: bool = False) -> None:
    # handle bentuk: --flag=value
    new_argv = []
    skip_next = False
    for a in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if a.startswith(prefix + "="):
            continue
        if a == prefix:
            if takes_value:
                skip_next = True
            continue
        new_argv.append(a)
    sys.argv[:] = new_argv


# buang argumen yang terbukti bikin error di environment kamu
_strip_flag("--editable", takes_value=False)
_strip_flag("--build-directory", takes_value=True)
_strip_flag_prefix("--build-directory", takes_value=False)


def _files_recursive(folder: str):
    paths = glob(os.path.join(folder, "**", "*"), recursive=True)
    return [p for p in paths if os.path.isfile(p)]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models"), _files_recursive("models")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="seano",
    maintainer_email="seano@todo.todo",
    description="SEANO collision avoidance nodes (ROS2 Humble + MAVROS2 + ArduRover SITL).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mavros_rc_override_bridge_node = seano_vision.mavros_rc_override_bridge_node:main",
            "actuator_safety_limiter_node = seano_vision.actuator_safety_limiter_node:main",
            "camera_node = seano_vision.camera_node:main",
            "detector_node = seano_vision.detector_node:main",
            "false_positive_guard_node = seano_vision.false_positive_guard_node:main",
            "frame_freeze_detector_node = seano_vision.frame_freeze_detector_node:main",
            "multi_target_fusion_node = seano_vision.multi_target_fusion_node:main",
            "risk_evaluator_node = seano_vision.risk_evaluator_node:main",
            "time_sync_node = seano_vision.time_sync_node:main",
            "vision_quality_node = seano_vision.vision_quality_node:main",
            "watchdog_failsafe_node = seano_vision.watchdog_failsafe_node:main",
            "waterline_horizon_node = seano_vision.waterline_horizon_node:main",
            "teleop_diff_thruster_node = seano_vision.teleop_diff_thruster_node:main",
            "command_mux_node = seano_vision.command_mux_node:main",
            "thrsteer_to_auto_left_right_node = seano_vision.thrsteer_to_auto_left_right_node:main",
            "auto_controller_stub_node = seano_vision.auto_controller_stub_node:main",
            "test_maneuver_node = seano_vision.test_maneuver_node:main",
            "mission_mode_manager_node = seano_vision.mission_mode_manager_node:main",
        ],
    },
)
