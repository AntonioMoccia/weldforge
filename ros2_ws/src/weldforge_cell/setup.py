from glob import glob

from setuptools import find_packages, setup

package_name = "weldforge_cell"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Antonio Moccia",
    maintainer_email="moccia.ant@gmail.com",
    description="WeldForge nodes for a robotic welding cell",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planner_node = weldforge_cell.planner_node:main",
            "welder_io_node = weldforge_cell.welder_io_node:main",
            "execution_node = weldforge_cell.execution_node:main",
        ],
    },
)
