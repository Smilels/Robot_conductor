from setuptools import find_packages, setup

requirements = ["hydra-core==0.11.3", "pytorch-lightning==0.7.1"]

__version__ = "0.0.1"


setup(
    name="Robot_conductor",
    version=__version__,
    author="Shuang Li",
    author_email="sli@informatik.uni-hamburg.de",
    packages=find_packages(),
    install_requires=requirements,
)