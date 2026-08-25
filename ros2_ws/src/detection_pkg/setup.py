import os

from setuptools import find_packages, setup

package_name = 'detection_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mars',
    maintainer_email='3501711400@qq.com',
    description='cup+mouse YOLOv8 目标检测 ROS2 发布节点',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detection_node = detection_pkg.detection_node:main',
        ],
    },
)
