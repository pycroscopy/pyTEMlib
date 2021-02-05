#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

################################################################################
# Python class for reading Nion Swift files into sidpy Dataset
# and extracting all metadata
#
# Written by Gerd Duscher, UTK 2020
#
# Works for python 3
#
################################################################################
from __future__ import division, print_function, absolute_import, unicode_literals

import json
import struct
import h5py
# from warnings import warn
import sys
import numpy as np
import os

import sidpy

__all__ = ["NionReader", "version"]

version = '0.1beta'

debugLevel = 0  # 0=none, 1-3=basic, 4-5=simple, 6-10 verbose

if sys.version_info.major == 3:
    unicode = str

# ### utility functions ###


def parse_zip(fp):
    """
        Parse the zip file headers at fp
        :param fp: the file pointer from which to parse the zip file
        :return: A tuple of local files, directory headers, and end of central directory
        The local files are dictionary where the keys are the local file offset and the
        values are each a tuple consisting of the name, data position, data length, and crc32.
        The directory headers are a dictionary where the keys are the names of the files
        and the values are a tuple consisting of the directory header position, and the
        associated local file position.
        The end of central directory is a tuple consisting of the location of the end of
        central directory header and the location of the first directory header.
        This method will seek to location 0 of fp and leave fp at end of file.

        This function is copied from  nionswift/nion/swift/model/NDataHandler.py

    """
    local_files = {}
    dir_files = {}
    eocd = None
    fp.seek(0)
    while True:
        pos = fp.tell()
        signature = struct.unpack('I', fp.read(4))[0]
        if signature == 0x04034b50:
            fp.seek(pos + 14)
            crc32 = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 18)
            data_len = struct.unpack('I', fp.read(4))[0]
            fp.seek(pos + 26)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            name_bytes = fp.read(name_len)
            fp.seek(extra_len, os.SEEK_CUR)
            data_pos = fp.tell()
            fp.seek(data_len, os.SEEK_CUR)
            local_files[pos] = (name_bytes, data_pos, data_len, crc32)
        elif signature == 0x02014b50:
            fp.seek(pos + 28)
            name_len = struct.unpack('H', fp.read(2))[0]
            extra_len = struct.unpack('H', fp.read(2))[0]
            comment_len = struct.unpack('H', fp.read(2))[0]
            fp.seek(pos + 42)
            pos2 = struct.unpack('I', fp.read(4))[0]
            name_bytes = fp.read(name_len)
            fp.seek(pos + 46 + name_len + extra_len + comment_len)
            dir_files[name_bytes] = (pos, pos2)
        elif signature == 0x06054b50:
            fp.seek(pos + 16)
            pos2 = struct.unpack('I', fp.read(4))[0]
            eocd = (pos, pos2)
            break
        else:
            raise IOError()
    return local_files, dir_files, eocd


class NionReader(sidpy.Reader):

    def __init__(self, file_path, verbose=False):
        """
        file_path: filepath to dm3 file.
        """

        super().__init__(file_path)

        # initialize variables ##
        self.verbose = verbose
        self.__filename = file_path

        path, file_name = os.path.split(self.__filename)
        self.basename, self.extension = os.path.splitext(file_name)
        self.data_cube = None
        self.original_metadata = {}
        self.dimensions = []
        if 'ndata' in self.extension:

            # - open file for reading
            try:
                self.__f = open(self.__filename, "rb")
            except FileNotFoundError:
                raise FileNotFoundError('File not found')
            try:
                local_files, dir_files, eocd = parse_zip(self.__f)
            except IOError:
                raise IOError("File {} does not seem to be of Nion`s .ndata format".format(self.__filename))
            self.__f.close()
        elif self.extension == '.h5':
            try:
                fp = h5py.File(self.__filename, mode='a')
                if 'data' not in fp:
                    raise IOError("File {} does not seem to be of Nion`s .h5 format".format(self.__filename))
                fp.close()
            except IOError:
                raise IOError("File {} does not seem to be of Nion`s .h5 format".format(self.__filename))

    def read(self):
        if 'ndata' in self.extension:
            try:
                self.__f = open(self.__filename, "rb")
            except FileNotFoundError:
                raise FileNotFoundError('File not found')
            local_files, dir_files, eocd = parse_zip(self.__f)

            contains_data = b"data.npy" in dir_files
            contains_metadata = b"metadata.json" in dir_files
            file_count = contains_data + contains_metadata  # use fact that True is 1, False is 0

            self.__f.seek(local_files[dir_files[b"data.npy"][1]][1])

            self.data_cube = np.load(self.__f)

            json_pos = local_files[dir_files[b"metadata.json"][1]][1]
            json_len = local_files[dir_files[b"metadata.json"][1]][2]
            self.__f.seek(json_pos)
            json_properties = self.__f.read(json_len)

            self.original_metadata = json.loads(json_properties.decode("utf-8"))
            self.__f.close()
        elif self.extension == '.h5':
            # TODO: use lazy load for large datasets
            self.__f = h5py.File(self.__filename, 'a')
            if 'data' in self.__f:
                json_properties = self.__f['data'].attrs.get("properties", "")
                self.data_cube = self.__f['data'][:]
                self.original_metadata = json.loads(json_properties)

        self.get_dimensions()
        # Need to switch image dimensions in Nion format
        image_dims = []
        spectral_dims = []
        for dim, axis in enumerate(self.dimensions):
            # print(dim, axis)
            if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                image_dims.append(dim)
            if axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                spectral_dims.append(dim)

        # convert linescan to spectral_image
        if len(spectral_dims) == 1:
            if self.data_cube.ndim > 1:
                self.data_cube = self.data_cube.reshape(1, self.data_cube.shape[0], self.data_cube.shape[1])
                new_dims = [sidpy.Dimension([1], name='x', units='pixels',
                                            quantity='distance', dimension_type='spatial'),
                            sidpy.Dimension(np.arange(self.data_cube.shape[0]), name='y', units='pixels',
                                            quantity='distance', dimension_type='spatial'),
                            self.dimensions[spectral_dims[0]]]
                self.dimensions = new_dims

        if len(image_dims) == 2:
            self.data_cube = np.swapaxes(self.data_cube, image_dims[0], image_dims[1])
            temp = self.dimensions[image_dims[0]].copy()
            self.dimensions[image_dims[0]] = self.dimensions[image_dims[1]].copy()
            self.dimensions[image_dims[1]] = temp

        dataset = sidpy.Dataset.from_array(self.data_cube)

        dim_names = []
        for dim, axis in enumerate(self.dimensions):
            dataset.set_dimension(dim, axis)

        dataset.original_metadata = self.original_metadata
        if 'dimensional_calibrations' in dataset.original_metadata:
            for dim in dataset.original_metadata['dimensional_calibrations']:
                if dim['units'] == '':
                    dim['units'] = 'pixels'

        dataset.quantity = 'intensity'
        dataset.units = 'counts'
        if 'description' in dataset.original_metadata:
            dataset.title = dataset.original_metadata['description']['title']
        else:
            if 'title' in dataset.original_metadata:
                dataset.title = dataset.original_metadata['title']
            else:
                path, file_name = os.path.split(self.__filename)
                basename, extension = os.path.splitext(file_name)
                dataset.title = basename

        if 'data_source' in dataset.original_metadata:
            dataset.source = dataset.original_metadata['data_source']
        else:
            dataset.source = 'NionReader'

        self.set_data_type(dataset)
        dataset.modality = 'STEM data'
        dataset.h5_dataset = None

        return dataset

    def set_data_type(self, dataset):

        spectral_dim = False
        for axis in dataset._axes.values():
            if axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                spectral_dim = True

        if len(dataset.shape) > 3:
            raise NotImplementedError('Data_type not implemented yet')
        elif len(dataset.shape) == 3:
            if spectral_dim:
                dataset.data_type = 'spectral_image'
            else:
                dataset.data_type = 'IMAGE_STACK'
                for dim, axis in dataset._axes.items():
                    if axis.dimension_type != sidpy.DimensionType.SPATIAL:
                        dataset.set_dimension(dim, sidpy.Dimension(axis.values,
                                                                   name='frame',
                                                                   units='frame',
                                                                   quantity='stack',
                                                                   dimension_type=sidpy.DimensionType.TEMPORAL))
                        break

        elif len(dataset.shape) == 2:
            if spectral_dim:
                dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE
            else:
                dataset.data_type = sidpy.DataType.IMAGE
        elif len(dataset.shape) == 1:
            if spectral_dim:
                dataset.data_type = sidpy.DataType.SPECTRUM
            else:
                dataset.data_type = sidpy.DataType.LINE_PLOT

    def get_dimensions(self):
        dic = self.original_metadata

        reciprocal_name = 'u'
        spatial_name = 'x'

        if 'dimensional_calibrations' in dic:
            dimension_list = dic['dimensional_calibrations']
        elif 'spatial_calibrations' in dic:
            dimension_list = dic['spatial_calibrations']
        else:
            return

        for dim in range(len(dimension_list)):
            dimension_tags = dimension_list[dim]
            units = dimension_tags['units']
            values = (np.arange(self.data_cube.shape[int(dim)])-dimension_tags['offset']) * dimension_tags['scale']

            if 'eV' == units:
                self.dimensions.append(sidpy.Dimension(values, name='energy_loss', units=units,
                                                       quantity='energy-loss', dimension_type='spectral'))
            elif 'eV' in units:
                self.dimensions.append(sidpy.Dimension(values, name='energy', units=units,
                                                       quantity='energy', dimension_type='spectral'))
            elif '1/' in units or units in ['mrad', 'rad']:
                self.dimensions.append(sidpy.Dimension(values, name=reciprocal_name, units=units,
                                                       quantity='reciprocal distance',
                                                       dimension_type='reciprocal'))
                reciprocal_name = chr(ord(reciprocal_name) + 1)
            elif 'nm' in units:
                self.dimensions.append(sidpy.Dimension(values, name=spatial_name, units=units,
                                                       quantity='distance', dimension_type='spatial'))
                spatial_name = chr(ord(spatial_name) + 1)
            else:
                self.dimensions.append(sidpy.Dimension(values, name=f'generic_{dim}', units='generic',
                                                       quantity='generic', dimension_type='UNKNOWN'))

    def get_filename(self):
        return self.__filename

    filename = property(get_filename)

    def get_raw(self):
        return self.data

    data = property(get_raw)

    def get_tags(self):
        return self.original_metadata

    tags = property(get_tags)
