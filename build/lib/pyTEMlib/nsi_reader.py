# -*- coding: utf-8 -*-
"""
Reader capable of reading one or all NSID datasets present in a given HDF5 file

Created on Fri May 22 16:29:25 2020

@author: Gerd Duscher, Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import numpy as np

# pyNSID function
import sidpy

from pyTEMlib.hdf_utils import get_all_main
if sys.version_info.major == 3:
    unicode = str


class NSIDReader(sidpy.Reader):

    def __init__(self, h5_object):
        """
        h5_object - hdf5 dataset
            specific Main dataset that needs to be read into a sidpy Dataset.
            # If no path is specified, read all available NSID Main datasets
        """

        if not isinstance(h5_object.file, h5py.File):
            raise TypeError('we can only read h5py datasets')

        super(NSIDReader, self).__init__(file_path=h5_object.file.name)

        self.dset = None
        self.main_datasets = []
        if isinstance(h5_object, h5py.Dataset):
            self.dset = h5_object
            self.h5_group = self.dset.parent

        elif isinstance(h5_object, h5py.Group):
            self.h5_group = h5_object
        else:
            raise TypeError('we can only read h5py datasets')

        # Find all main datasets is done in read as the file may change between readings
        # DO NOT close HDF5 file. Dask array will fail if you do so.
        # TODO: sidpy.Dataset may need the ability to close a HDF5 file
        # Perhaps this would be done by reading all contents into memory..

    @staticmethod
    def read_h5py_dataset(dset):

        if not isinstance(dset, h5py.Dataset):
            raise TypeError('can only read single Dataset, use read_all_in_group or read_all function instead')
        # create vanilla dask array
        dataset = sidpy.Dataset.from_array(np.array(dset))

        if 'title' in dset.attrs:
            dataset.title = dset.attrs['title']
        else:
            dataset.title = dset.name

        if 'units' in dset.attrs:
            dataset.units = dset.attrs['units']
        else:
            dataset.units = 'generic'

        if 'quantity' in dset.attrs:
            dataset.quantity = dset.attrs['quantity']
        else:
            dataset.quantity = 'generic'

        if 'data_type' in dset.attrs:
            dataset.data_type = dset.attrs['data_type']
        else:
            dataset.data_type = 'generic'

        if 'modality' in dset.attrs:
            dataset.modality = dset.attrs['modality']
        else:
            dataset.modality = 'generic'

        if 'source' in dset.attrs:
            dataset.source = dset.attrs['source']
        else:
            dataset.source = 'generic'

        dataset.axes = {}

        for dim in range(np.array(dset).ndim):
            try:
                label = dset.dims[dim].keys()[-1]

            except ValueError:
                print('dimension {} not NSID type using generic'.format(dim))
            name = dset.dims[dim][label].name
            dim_dict = {'quantity': 'generic', 'units': 'generic', 'dimension_type': 'generic'}
            h5_dim_dict = dict(dset.parent[name].attrs)
            if 'quantity' in h5_dim_dict:
                dim_dict['quantity'] = h5_dim_dict['quantity']
            else:
                if 'NAME' in h5_dim_dict:
                    param = h5_dim_dict['NAME'].decode("utf-8").split('[')
                    # print(param)
                    if len(param) == 1:
                        if param[0] == 'frame':
                            dim_dict['quantity'] = 'stack'
                            dim_dict['units'] = 'frame'
                            dim_dict['dimension_type'] = sidpy.DimensionTypes.TEMPORAL
                    elif len(param) == 2:
                        dim_dict['quantity'] = param[0]
                        dim_dict['units'] = param[1][0:-1]

                        if dim_dict['units'] == 'nm':
                            dim_dict['dimension_type'] = sidpy.DimensionTypes.SPATIAL
                        elif dim_dict['units'] == 'eV':
                            dim_dict['dimension_type'] = sidpy.DimensionTypes.SPECTRAL

            if 'units' in h5_dim_dict:
                dim_dict['units'] = h5_dim_dict['units']
            if 'dimension_type' in h5_dim_dict:
                dim_dict['dimension_type'] = h5_dim_dict['dimension_type']

            dim_dict.update(dict(dset.parent[name].attrs))

            dataset.set_dimension(dim, sidpy.Dimension(np.array(dset.parent[name][()]), name=dset.dims[dim].label,
                                                       quantity=dim_dict['quantity'], units=dim_dict['units'],
                                                       dimension_type=dim_dict['dimension_type']))

        if 'metadata' in dset.parent:
            dataset.metadata = sidpy.base.dict_utils.nest_dict(dict(dset.parent['metadata'].attrs))

        dataset.metadata.update(dict(dset.attrs))

        dataset.original_metadata = {}
        if 'original_metadata' in dset.parent:
            dataset.original_metadata = sidpy.base.dict_utils.nest_dict(dict(dset.parent['original_metadata'].attrs))

        # hdf5 information
        dataset.h5_dataset = dset

        return dataset

    def can_read(self):
        pass

    def read(self):
        if not isinstance(self.h5_group, h5py.Group):
            raise TypeError('This function needs to be initialised with a hdf5 group or dataset first')
        list_of_main = get_all_main(self.h5_group, verbose=False)

        """
        Go through each of the identified
        """
        list_of_datasets = []
        for dset in list_of_main:
            list_of_datasets.append(self.read_h5py_dataset(dset))

        return list_of_datasets

    def read_all_in_group(self, recursive=True):
        pass
