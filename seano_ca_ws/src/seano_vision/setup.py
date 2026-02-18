from setuptools import setup
import os
from glob import glob

package_name = 'seano_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        # install configs
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),

        # install models (kalau dipakai)
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seano',
    maintainer_email='seano@todo.todo',
    description='SEANO vision and collision avoidance stack',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = seano_vision.camera_node:main',
            'detector_node = seano_vision.detector_node:main',
            'risk_evaluator_node = seano_vision.risk_evaluator_node:main',
            'multi_target_fusion_node = seano_vision.multi_target_fusion_node:main',
            'false_positive_guard_node = seano_vision.false_positive_guard_node:main',
            'frame_freeze_detector_node = seano_vision.frame_freeze_detector_node:main',
            'vision_quality_node = seano_vision.vision_quality_node:main',
            'waterline_horizon_node = seano_vision.waterline_horizon_node:main',
            'time_sync_node = seano_vision.time_sync_node:main',
            'watchdog_failsafe_node = seano_vision.watchdog_failsafe_node:main',
            'actuator_safety_limiter_node = seano_vision.actuator_safety_limiter_node:main',
        ],
    },
)
