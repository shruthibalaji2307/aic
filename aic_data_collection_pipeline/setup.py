from setuptools import find_packages, setup

package_name = "aic_data_collection_pipeline"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools", "PyYAML>=6.0"],
    zip_safe=True,
    maintainer="AIC Participant",
    maintainer_email="you@example.com",
    description="Automation helpers for AIC LeRobot data collection workflows.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "aic-generate-qualification-config = aic_data_collection_pipeline.generate_qualification_config:main",
            "aic-run-lerobot-pipeline = aic_data_collection_pipeline.run_lerobot_pipeline:main",
        ],
    },
)
