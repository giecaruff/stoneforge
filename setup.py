# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3"
]

setup(
    name="stoneforge",
    version="0.1.6",
    author="GIECAR - UFF",
    url="https://github.com/giecaruff/stoneforge",
    description="Geophysics equations, algorithms and methods",
    long_description=open("README.md").read() + "\n\n" + open("CHANGELOG.txt").read(),
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires="==3.12.6",
    install_requires=[
        "numpy==1.26.4", # 05/02/2024
        "pytest==8.1.1", # 09/03/2024
        "scipy==1.13.0", # 02/04/2024
        "scikit-learn==1.4.2", # 09/04/2024
        #"notebook==7.1.3", # 18/04/2024
        "xgboost==2.0.3", # 19/12/2023
        "matplotlib==3.8.4", # 09/04/2024
        "pandas==2.2.2", # 10/04/2024
        "catboost==1.2.5", # 18/04/2024
        "lightgbm==4.3.0", # 26/01/2024
        #"auto-sklearn==0.12.5"
    ],
    long_description_content_type='text/markdown',
)
