from setuptools import setup, find_packages
import os
from glob import glob

package_name = "seano_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="seano",
    maintainer_email="seanousv@gmail.com",
    description="SEANO collision avoidance stack (perception, risk/decision, safety, MAVROS actuation bridge).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = seano_vision.camera_node:main",
            "detector_node = seano_vision.detector_node:main",
            "risk_evaluator_node = seano_vision.risk_evaluator_node:main",
            "multi_target_fusion_node = seano_vision.multi_target_fusion_node:main",
            "false_positive_guard_node = seano_vision.false_positive_guard_node:main",
            "frame_freeze_detector_node = seano_vision.frame_freeze_detector_node:main",
            "vision_quality_node = seano_vision.vision_quality_node:main",
            "waterline_horizon_node = seano_vision.waterline_horizon_node:main",
            "time_sync_node = seano_vision.time_sync_node:main",
            "watchdog_failsafe_node = seano_vision.watchdog_failsafe_node:main",
            "actuator_safety_limiter_node = seano_vision.actuator_safety_limiter_node:main",
            "mavros_rc_override_bridge_node = seano_vision.mavros_rc_override_bridge_node:main",
        ],
    },
)
