from setuptools import find_packages, setup
from glob import glob

package_name = "drone_grid_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="drone-dev",
    maintainer_email="dev@example.com",
    description="ROS 2 closed loop for kaiwu drone-delivery agent_diy.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "grid_env_node = drone_grid_sim.nodes.grid_env_node:main",
            "policy_node = drone_grid_sim.nodes.policy_node:main",
            "motion_controller_node = drone_grid_sim.nodes.motion_controller_node:main",
            "viz_node = drone_grid_sim.nodes.viz_node:main",
        ],
    },
)
