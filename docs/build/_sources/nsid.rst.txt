NSID
====
**Suhas Somnath**

N-Dimensional Spectroscopy and Imaging Data (NSID)

Why not just use h5py?
----------------------
h5py does indeed provide all the functionality necessary to support NSID. However, a layer of convenience and standardization is still useful / necessary for few reasons:

1. To ensure that data (in memory) are always stored in the same standardized fashion. This would be a function like `pyUSID.hdf_utils.write_main_dataset() <https://pycroscopy.github.io/pyUSID/auto_examples/intermediate/plot_hdf_utils_write.html#write-main-dataset>`_ or a class like `pyUSID.ArrayTranslator <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_numpy_translator.html#sphx-glr-auto-examples-beginner-plot-numpy-translator-py>`_.
2. To make it easier to access relevant ancillary information from HDF5 datasets such as the dimensions, units, scales, etc. without needing to write a lot of h5py code. I anticpate that this may look like a class along the lines of `pyusid.USIDataset <https://pycroscopy.github.io/pyUSID/auto_examples/beginner/plot_usi_dataset.html>`_. However, this class may extend a ``dask.array`` object instead of a h5py.Dataset object for simplicity. ``xarray`` apparently extends pandas which is inappropriate for this application. However, packages like ``pint`` should ceratinly be used.
3. To simplify certain ancillary tasks like identify all NSID datasets in a given file, seamlessly reusing datasets representing dimensions / copying datasets, verifying whether a dataset is indeed NSID or not.
4. To facilitate embarrasingly parallel computations on datasets along the lines of `pyUSID.Process <https://pycroscopy.github.io/pyUSID/auto_examples/intermediate/plot_process.html#sphx-glr-auto-examples-intermediate-plot-process-py>`_. I would love to use dask to handle parallelization. However, HDF5 datasets are still not pickle-able. Therefore, Dask cannot operate on them. It is likely, that this framaework would rely on lower-level libraries like mpi4py

I expect that the package to support NSID would be far simpler than pyUSID since h5py provides the majority of the functionality inherently.

Strawman NSID specifications
----------------------------
Originally the NSID model was envisioned to be similar to USID in that it too would have a `Main Dataset` that is supported by `Ancillary Datasets` to provide reference information about each dimension. The ancillary datasets for each dimension would be `attached <http://docs.h5py.org/en/stable/high/dims.html>`_ to the `Main Dataset` using HDF5 `Dimension Scales <https://support.hdfgroup.org/HDF5/Tutor/h5dimscale.html>`_
However, I have since learnt that HDF5's Dimension Scales can capture the information that would have been stored in these ancillary datasets and can be attached to the `Main Dataset`.

Main Dataset
~~~~~~~~~~~~
The main data will be stored in an HDF5 dataset:

* **shape**: Arbitrary - matching the dimensionality of the data
* **dtype**: basic types like integer, float, and complex only. I am told that compound-valued datasets are not supported well in languages other than python. Therefore, such data should be broken up into simpler dtype datasets.
* **chunks**: Leave as default / do not specify anything.
* **compression**: Preferably do not use anything. If compression is indeed necessary, consider using `gzip`.
* **Dimension scales**: Every single dimension needs to have at least one scale attached to it with the name(s) of the dimension(s) as the `label`(s) for the scale. Normally, we would only have one dataset attached to each dimension. However, for example, if one of the reference axes was a color (a tuple of three integers), we would need to attach three datasets to the scale for the appropriate dimension.
* **Required Attributes**:

  * ``quantity``: `string`: Physical quantity that is contained in this dataset
  * ``units``: `string`: Units for this physical quantity
  * ``data_type``: `string : What kind of data this is. Example - image, image stack, video, hyperspectral image, etc.
  * ``modality``: `string : Experimental / simulation modality - scientific meaning of data. Example - photograph, TEM micrograph, SPM Force-Distance spectroscopy.
  * ``source``: `string : Source for dataset like the kind of instrument. One could go very deep here into either the algorithmic details if this is a result from analysis or the exact configurations for the instrument that generated this daatset. I am inclined to remove this attribute and have this expressed in the metadata alone.
  * ``nsid_version``: `string`: Version of the abstract NSID model.

Note tha we should take guidance from experts in schemas and ontologies on how best to represent the ``data_type`` and ``modality`` information.

Ancillary Datasets
~~~~~~~~~~~~~~~~~~
Each of the `N` dimensions corresponding to the `N`-dimensional `Main Dataset` would be an HDF5 dataset:

* **shape** - 1D only
* **dtype** - Simple data types like int, float, complex
* **Required attributes** -

  * ``quantity``: `string`: Physical quantity that is contained in this dataset
  * ``units``: `string`: units for the physical quantity
  * ``dimension_type``: `string`: Kind of dimension - 'position', 'spectral', 'reciprocal'

Metadata
~~~~~~~~
Strawman solution - Store the heirarchical metadata into heirarchical HDF5 groups within the same file as the `Main Dataset` and link the parent group the provides the metadata to the `Main Dataset`. Again, this requires feedback from experts in schemas and ontologies.

Multiple measurements in same file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A single HDF5 file can contain multiple HDF5 datasets. It is not necessary that all datasets be NSID-specific. Similarly, the heirarchical nature of HDF5 will allow the storage of multiple NSID measurements within the same HDF5 file. Strict restrictions will not be placed on how the datasets should be arranged. Users are free to use and are recommended to use the same guidelienes of `Measurement Groups <https://pycroscopy.github.io/USID/h5_usid.html#measurement-data>`_ and `Channels <https://pycroscopy.github.io/USID/usid_model.html#channels>`_ as defined in USID.

Data processing results in same file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We defined a `possible solution <https://pycroscopy.github.io/USID/h5_usid.html#tool-analysis-processing>`_ for capturing provenance between the source dataset and the results datasets. Briefly, results would be stored in a group whose name would be formatted as ``SourceDataset-ProcessingAlgorithmName_NumericIndex``. However, this solution does not work *elegantly* for certain situations:

* if multiple source datasets were used to produce a set of results datasets.
* if results are written into a different file.
* In general, the algorithm name was loosly defined.

Do get in touch if you know of a better solution

Existing solutions
------------------
A while ago, I had come across the following resources:

* From UIUC folks - a webpage called `electron microscopy data <https://emdatasets.com/format/>`_
* from the DREAM.3D folks - "`MXA: a customizable HDF5-based data format for multi-dimensional data sets <https://iopscience.iop.org/article/10.1088/0965-0393/18/6/065008>`_" by Michael Jackson
* From APS folks at Argonne - "`Scientific data exchange: a schema for HDF5-based storage of raw and analyzed data <https://onlinelibrary.wiley.com/doi/full/10.1107/S160057751401604X?sentby=iucr>`_" by Francesco de Carlo.

However, all of these were targeting a specific scientific sub-domain / modality. They were not as simple / general as pyNSID.

I am not sure if something like NSID or a python API like pyNSID exist now.
We would need to survey literature again for existing solutions to avoid duplicating efforts and for supporting an existing central effort.