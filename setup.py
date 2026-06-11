from setuptools import find_packages, setup

setup(
    name="softbody-insertion",
    packages=find_packages(include=["pih_rebuild", "pih_rebuild.*"]),
    include_package_data=True,
    package_data={
        "pih_rebuild": [
            "assets/mujoco/*.xml",
            "assets/mujoco/meshs/visual/*",
            "assets/mujoco/textures/*",
            "assets/urdf/*.urdf",
        ],
    },
    install_requires=[
        "gym>=0.22,<=0.23",
        "mujoco==2.3.7",
        "numpy<2",
        "scipy",
        "stable-baselines3==1.8.0",
        "tensorboard",
        "urdf-parser-py",
    ],
    extras_require={
        "tests": ["pytest-cov"],
        "codestyle": ["black", "isort"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
)
