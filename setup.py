# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 09:08:25 2019

@author: gduscher
"""
from codecs import open
import os

import setuptools


here = os.path.abspath(os.path.dirname(__file__))


with open(os.path.join(here, 'pyTEMlib/__version__.py')) as f:
    __version__ = f.read().split("'")[1]
    
setuptools.setup(
    name="pyTEMlib",
    version="0.6.2019.3",
    author="Gerd Duscher",
    author_email="gduscher@utk.edu",
    description="pyTEM: TEM Data Quantification library through a Model Based Approach",
    #long_description=open("README.rst").read(),
    url="https://web.utk.edu/~gduscher/Quantifit/",
    packages=["pyTEMlib"],
    package_data={"pyTEMlib": ["data/*"]},
    install_requires=['scipy', 'numpy',  'pillow','pyqtgraph', 'simpleITK','ase','pyUSID'],#,'PyQt5> 1.0'],#
    classifiers=[
        "Development Status :: 2 - Pre-Alpha"
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pyTEMlib=pyTEMlib:main',
            ],
        },
    
)
