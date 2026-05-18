from setuptools import find_packages, setup

package_name = 'rppg_bridge_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jeonga',
    maintainer_email='jeonga563@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'rppg_udp_bridge = rppg_bridge_pkg.rppg_udp_bridge_node:main',
            # 가짜 코드 실행을 위한 노드 추가
            'rppg_dummy_publisher = rppg_bridge_pkg.rppg_dummy_publisher_node:main',
        ],
    },
)
