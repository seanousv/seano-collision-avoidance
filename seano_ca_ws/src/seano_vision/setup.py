from setuptools import find_packages, setup
import os
from glob import glob

package_name = "seano_vision"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "models"), glob("models/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SEANO",
    maintainer_email="seano@example.com",
    description="SEANO vision pipeline (camera, detector, collision avoidance).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = seano_vision.camera_node:main",
            "detector_node = seano_vision.detector_node:main",
            "risk_evaluator_node = seano_vision.risk_evaluator_node:main",
            "frame_freeze_detector_node = seano_vision.frame_freeze_detector_node:main",
            "vision_quality_node = seano_vision.vision_quality_node:main",
        ],
    },
)
