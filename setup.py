from setuptools import setup, find_packages

setup(
    name="sunflower",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib"
    ],  # Add any dependencies here
)