#!/usr/bin/env python

from setuptools import setup

setup(name='hmc',
      version='0.1',
      description='Decision tree based hierachical multi-classifier',
      author='David Warshaw',
      author_email='david.warshaw@gmail.com',
      py_modules=['hmc', 'datasets'],
      requires=['sklearn', 'numpy', 'pandas'])
