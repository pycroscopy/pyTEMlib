pyTEMlib
========

.. image:: https://github.com/pycroscopy/pyTEMlib/workflows/build/badge.svg?branch=master
    :target: https://github.com/pycroscopy/pyTEMlib/actions?query=workflow%3Abuild
    :alt: GitHub Actions

.. image:: https://img.shields.io/pypi/v/pyTEMlib.svg
    :target: https://pypi.org/project/pyTEMlib/
    :alt: PyPI

.. image:: https://codecov.io/gh/pycroscopy/pyTEMlib/branch/master/graph/badge.svg?token=
    :target: https://codecov.io/gh/pycroscopy/pyTEMlib
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/l/pyTEMlib.svg
    :target: https://pypi.org/project/pyTEMlib/
    :alt: License

.. image:: http://pepy.tech/badge/pyTEMlib
    :target: http://pepy.tech/project/pyTEMlib
    :alt: Downloads

.. image:: https://zenodo.org/badge/138171750.svg
   :target: https://zenodo.org/badge/latestdoi/138171750
   :alt: DOI

pyTEMlib is a package to read and process various kind of data acquired with a (scanning) transmission electron microscope (STEM).

The package is written in pure python and depends on various other libraries.

All data, user input, and results are stored as `NSID-formatted <https://pycroscopy.github.io/pyNSID/nsid.html>`_ HDF5 files.

The data are all presented as `sidpy.Dataset <https://pycroscopy.github.io/sidpy/notebooks/00_basic_usage/create_dataset.html>`_ objects

Install pyTEMlib via pip as:

``python3 -m pip install  pyTEMlib``
 
This command is also built into the `example notebooks <https://github.com/pycroscopy/pyTEMlib/notebooks>`_
