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
    version="0.1.7.dev1",
    author="GIECAR - UFF",
    url="https://github.com/giecaruff/stoneforge",
    description="Geophysics equations, algorithms and methods",
    long_description=open("README.md").read() + "\n\n" + open("CHANGELOG.txt").read(),
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "dlisio>=1.0.3", # 21/01/2025
        "numpy>=1.26.4", # 05/02/2024
        "pytest==8.3.4", # 01/12/2024
        "scipy>=1.15.0", # 03/01/2025
        "scikit-learn>=1.6.0", # 09/12/2024
        "notebook==7.3.2", # 21/12/2024
        "xgboost>=2.1.3", # 26/11/2024
        "matplotlib>=3.10.0", # 14/12/2024
        "pandas>=2.2.3", # 20/09/2024
        "catboost>=1.2.7", # 07/09/2024
        "lightgbm==4.5.0", # 26/07/2024
        #"auto-sklearn==0.12.5"
    ],
    long_description_content_type='text/markdown',
)
