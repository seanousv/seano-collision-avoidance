from setuptools import setup, find_packages
from glob import glob
import os

package_name = "seano_vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["yolov8n.pt"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="seano",
    maintainer_email="seano@example.com",
    description="SEANO camera-only collision avoidance pipeline",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = seano_vision.camera_node:main",
            "detector_node = seano_vision.detector_node:main",
            "risk_evaluator_node = seano_vision.risk_evaluator_node:main",
            "viewer_node = seano_vision.viewer_node:main",
        ],
    },
)
