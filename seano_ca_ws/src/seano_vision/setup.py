from setuptools import setup
from glob import glob
import os

package_name = 'seano_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # ament index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # configs
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # models (kalau folder models ada)
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seano',
    maintainer_email='seano@todo.todo',
    description='SEANO Vision nodes (camera, detector, risk evaluator)',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = seano_vision.camera_node:main',
            'detector_node = seano_vision.detector_node:main',
            'risk_evaluator_node = seano_vision.risk_evaluator_node:main',
            # kalau kamu memang punya viewer_node.py dan mau bisa ros2 run:
            # 'viewer_node = seano_vision.viewer_node:main',
        ],
    },
)
