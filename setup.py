# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Education",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python ::3"
]

setup(
    name="stoneforge",
    version="0.1.4",
    author="GIECAR - UFF",
    url="https://github.com/giecaruff/stoneforge",
    description="Geophysics equations, algorithms and methods",
    long_description=open("README.md").read() + "\n\n" + open("CHANGELOG.txt").read(),
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8.8",
    install_requires=[
        "numpy>=1.21, <1.22.1",
        "pytest>=6.2.2, <6.2.5",
        "scipy>=1.4.1, <1.8.1",
        "scikit-learn>=0.22.1, <0.24.2",
        #"xgboost==1.4.0",
        "matplotlib>=3.5.0, <3.5.3",
        "jupyter==1.0.0",
        "dash>=1.20.0, <1.21.0",
        "plotly>=5.1.0, <5.2.1",
        "pandas==1.5.2",
        "catboost==1.0.6",
        "lightgbm==3.3.2",
        #"catboost==0.26.1"
        #"auto-sklearn==0.12.5"
        
        #"matplotlib", ### TO REMOVE BEFORE MERGE!!
        #"pandas", ### TO REMOVE BEFORE MERGE!!
        #"jupyter", ### TO REMOVE BEFORE MERGE!!
        #"seaborn" ### # TO REMOVE BEFORE MERGE!!
    ],
)