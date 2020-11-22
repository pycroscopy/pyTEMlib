# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 09:08:25 2019

@author: gduscher
"""
from codecs import open
import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'pyTEMlib/version.py')) as f:
    __version__ = f.read().split("'")[1]
    
setuptools.setup(
    name="pyTEMlib",
    version=__version__,
    author="Gerd Duscher",
    author_email="gduscher@utk.edu",
    description="pyTEM: TEM Data Quantification library through a Model Based Approach",
    #long_description=open("README.rst").read(),
    url="https://web.utk.edu/~gduscher/Quantifit/",
    packages=["pyTEMlib"],
    package_data={"pyTEMlib": ["data/*"]},
    install_requires=['scipy', 'numpy',  'pillow', 'simpleITK','ase','pyNSID'],#,'PyQt5> 1.0'],#
    tests_require=['pytest'],
    platforms=['Linux', 'Mac OSX', 'Windows 10/8.1/8/7'],
    # package_data={'sample':['dataset_1.dat']}
    test_suite='pytest',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Information Analysis'],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pyTEMlib=pyTEMlib:main',
            ],
        },
    
)
