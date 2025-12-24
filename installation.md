# Installation


## Preparing for pyTEMlib

[pyTEMlib](https://github.com/pycroscopy/pyTEMlib) requires many commonly used scientific and numeric python packages 
such as numpy, h5py etc.
To simplify the installation process, we recommend the installation of
[anaconda](https://www.anaconda.com/distribution/) which contains most of the prerequisite packages,
[conda](https://conda.io/docs/) - a package / environment manager,
as well as an [interactive development environment](https://en.wikipedia.org/wiki/Integrated_development_environment) - 
[Spyder](https://www.coursera.org/learn/python-programming-introduction/lecture/ywcuv/introduction-to-the-spyder-ide)


Please note that any other python installation will work too and that pypi will install all missing packages.

Compatibility

* pyTEMlib is compatible with python 3.9 onwards. Please raise an issue if you find a bug.
* We do not support 32 bit architectures
* We only support text that is UTF-8 compliant due to restrictions posed by HDF5

The notebooks in this section explain installation and basic usage of pyTEMlib to view microscope datasets