import pathlib
import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8").read()


setup(install_requires=read("requirements.txt").strip().split("\n"),
      packages=find_packages()
      )