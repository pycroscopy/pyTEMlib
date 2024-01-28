# pyTEMlib

**Python framework for model based analysis of TEM/STEM data**

 What?
------
* The `pyTEMlib` package:

  * is a part of the [pycroscopy](https://github.com/pycroscopy) ecosystem based on [python](http://www.python.org/)
  * enables quantitative analysis through model based approach
  * provides routines for the analysis of diffraction, image and spectroscopic datasets
  * handles one, two, three, and four dimensional datasets
  * works in jupyter notebooks and in python programs.
  * provides dialog windows for metadata  and analysis input in jupyter notebooks and in python programs.

* The `pyTEMlib` covers:
  * **Diffraction**: Single and poly crystalline diffraction data and analysis in parallel and convergent  illumination
  * **Imaging**: Image analysis, atom detection and image stack registration.
  * **EELS**: It provides a framework for quantification of EELS spectra and spectrum images.


* Just as scipy uses numpy underneath, scientific packages like **pyTEMlib** use 
  * **sidpy**  format for dataset representation and 
  * **pyNSID** for all file-handling.
* Dialogs are based on  **ipython widgets**
* The packages **sidpy** and **pyNSID** use popular packages such as numpy, h5py, dask, matplotlib, etc. for most of 
  the storage, computation, and visualization.


Why?
-----
pyTEMlib originates in the need for teaching and the development of new techniques for TEM/STEM data analysis.
Please, see my lecture note(-books) for information on the background of analysis.


**1. Growing data sizes**
  * Cannot use desktop computers for analysis
  * *Need: High performance computing, storage resources and compatible, scalable file structures*

**2. Increasing data complexity**
  * Sophisticated imaging and spectroscopy modes resulting in 5,6,7... dimensional data
  * *Need: Robust software and generalized data formatting*

**3. Multiple file formats**
  * Different formats from each instrument. Proprietary in most cases
  * Incompatible for correlation
  * *Need: Open, instrument-independent data format*

**4. Expensive analysis software**
  * Software supplied with instruments often insufficient / incapable of custom analysis routines
  * Commercial software (Eg: Matlab, Origin..) are often prohibitively expensive.
  * *Need: Free, powerful, open source, user-friendly software*

**5. Closed science**
  * Analysis software and data not shared
  * No guarantees of reproducibility or traceability
  * *Need: open source data structures, file formats, centralized code and data repositories*

Who?
----
* We envision **pyTEMlib** to be a convenient package that facilitates all scientists to analyse data and develop new methods of anlysis, without being burdened with basic code functionality.
* This project is being led by staff members at Oak Ridge National Laboratory (ORNL), and professors at University of Tennessee, Knoxville
* We invite anyone interested to join our team to build better, free software for the scientific community
* Please visit our [credits and acknowledgements](./credits.html)_ page for more information.
* If you are interested in integrating our in your existing package, please [get in touch](./contact.html) with us.

Content
-------
```{tableofcontents}
```