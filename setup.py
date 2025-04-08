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
    version="attr: stoneforge.__version__",
    author="attr: stoneforge.__author__",
    url="attr: stoneforge.__url__",
    description="attr: stoneforge.__description__",
    license="attr: stoneforge.__license__",
    long_description=open("README.md").read() + "\n\n" + open("CHANGELOG.txt").read(),
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10", # 01/03/2021
    install_requires=[
        "dlisio>=1.0.3",
        "numpy>=2.2.0",
        "pytest>=8.3.4",
        "scipy>=1.14.1",
        "scikit-learn>=1.6.1",
        "xgboost>=2.1.4",
        "matplotlib>=3.10.0",
        "pandas>=2.2.2",
        "lightgbm>=4.5.0",
        "notebook>=6.4.0",
        "ipympl>=0.8.4"
    ],
    long_description_content_type='text/markdown',
)