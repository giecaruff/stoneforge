# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name='stoneforge',
    version="0.1.0",
    author="GIECAR - UFF",
    url="https://github.com/giecaruff/stoneforge",
    description="Geophysics equations, algorithms and methods",
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    python_requires='==3.8.8',
    install_requires=[
        "numpy==1.20.0",
        "pytest==6.2.2",
    ],
)
