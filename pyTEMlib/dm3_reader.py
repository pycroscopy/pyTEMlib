#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Will move to Scope Reader

################################################################################
# Python class for reading GATAN DM3 (DigitalMicrograph) files
# and extracting all metadata
# --
# tested on EELS spectra, spectrum images and single-image files and image-stacks
# --
# based on the DM3_Reader plug-in (v 1.3.4) for ImageJ by Greg Jefferis <jefferis@stanford.edu>
# http://rsb.info.nih.gov/ij/plugins/DM3_Reader.html
# --
# Python adaptation: Pierre-Ivan Raynal <raynal@med.univ-tours.fr>
# http://microscopies.med.univ-tours.fr/
#
# Extended for EELS support by Gerd Duscher, UTK 2012
# Rewritten for integration of sidpy 2020
#
# Works for python 3
#
################################################################################
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import struct
import time
import numpy

import sys
import numpy as np
import os

import sidpy

version = '0.1beta'

debugLevel = 0  # 0=none, 1-3=basic, 4-5=simple, 6-10 verbose

if sys.version_info.major == 3:
    unicode = str


# ### utility functions ###

# ## binary data reading functions ###


def read_long(f):
    """Read 4 bytes as integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('>l', read_bytes)[0]


def read_64_long(f):
    read_bytes = f.read(8)
    return struct.unpack('>Q', read_bytes)[0]


def read_short(f):
    """Read 2 bytes as integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('>h', read_bytes)[0]


def read_byte(f):
    """Read 1 byte as integer in file f"""
    read_bytes = f.read(1)
    return struct.unpack('>b', read_bytes)[0]


def read_bool(f):
    """Read 1 byte as boolean in file f"""
    read_val = read_byte(f)
    return read_val != 0


def read_char(f):
    """Read 1 byte as char in file f"""
    read_bytes = f.read(1)
    return struct.unpack('c', read_bytes)[0]


def read_string(f, length=1):
    """Read len bytes as a string in file f"""
    read_bytes = f.read(length)
    str_fmt = '>' + str(length) + 's'
    return struct.unpack(str_fmt, read_bytes)[0]


def read_le_short(f):
    """Read 2 bytes as *little endian* integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('<h', read_bytes)[0]


def read_le_long(f):
    """Read 4 bytes as *little endian* integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<l', read_bytes)[0]


def read_leu_short(f):
    """Read 2 bytes as *little endian* unsigned integer in file f"""
    read_bytes = f.read(2)
    return struct.unpack('<H', read_bytes)[0]


def read_leu_long(f):
    """Read 4 bytes as *little endian* unsigned integer in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<L', read_bytes)[0]


def read_le_float(f):
    """Read 4 bytes as *little endian* float in file f"""
    read_bytes = f.read(4)
    return struct.unpack('<f', read_bytes)[0]


def read_le_double(f):
    """Read 8 bytes as *little endian* double in file f"""
    read_bytes = f.read(8)
    return struct.unpack('<d', read_bytes)[0]


# constants for encoded data types ##
SHORT = 2
LONG = 3
USHORT = 4
ULONG = 5
FLOAT = 6
DOUBLE = 7
BOOLEAN = 8
CHAR = 9
OCTET = 10
STRUCT = 15
STRING = 18
ARRAY = 20

# - association data type <--> reading function
readFunc = {
    SHORT: read_le_short,
    LONG: read_le_long,
    USHORT: read_leu_short,
    ULONG: read_leu_long,
    FLOAT: read_le_float,
    DOUBLE: read_le_double,
    BOOLEAN: read_bool,
    CHAR: read_char,
    OCTET: read_char,  # difference with char???
}

# other constants ##
IMGLIST = "root.ImageList."
OBJLIST = "root.DocumentObjectList."
MAXDEPTH = 64


# END constants ##

class DM3Reader(sidpy.Reader):
    debugLevel = -1

    """
    file_path: filepath to dm3 file.

    warn('This Reader will eventually be moved to the ScopeReaders package'
         '. Be prepared to change your import statements',
         FutureWarning)
    """

    def __init__(self, file_path, verbose=False):
        super().__init__(file_path)

        # initialize variables ##
        self.verbose = verbose
        self.__filename = file_path
        self.__chosen_image = -1

        # - open file for reading
        try:
            self.__f = open(self.__filename, 'rb')
        except FileNotFoundError:
            raise FileNotFoundError('File not found')

        # - create Tags repositories
        self.__stored_tags = {'DM': {}}

        # check if this is valid DM3 file
        is_dm = True
        # read header (first 3 4-byte int)
        # get version
        file_version = read_long(self.__f)

        # get indicated file size
        if file_version == 3:
            file_size = read_long(self.__f)
        elif file_version == 4:
            file_size = read_64_long(self.__f)
        else:
            is_dm = False
        # get byte-ordering
        le = read_long(self.__f)
        little_endian = (le == 1)

        if little_endian:
            self.endian_str = '>'
        else:
            self.endian_str = '<'

        # check file header, raise Exception if not DM3
        if not is_dm:
            raise TypeError("%s does not appear to be a DM3 or DM4 file." % os.path.split(self.__filename)[1])
        elif self.verbose:
            print("%s appears to be a DM3 file" % self.__filename)
        self.file_version = file_version
        self.file_size = file_size

        if self.verbose:
            print("Header info.:")
            print("- file version:", file_version)
            print("- le:", le)
            print("- file size:", file_size, "bytes")

        # don't read but close file
        self.__f.close()

    def read(self):
        try:
            self.__f = open(self.__filename, 'rb')
        except FileNotFoundError:
            raise FileNotFoundError('File not found')

        t1 = time.time()
        file_version = read_long(self.__f)

        # get indicated file size
        if file_version == 3:
            file_size = read_long(self.__f)
        elif file_version == 4:
            file_size = read_64_long(self.__f)
        else:
            is_dm = False
        # get byte-ordering
        le = read_long(self.__f)
        little_endian = (le == 1)

        if little_endian:
            self.endian_str = '>'
        else:
            self.endian_str = '<'

        # ... then read it

        self.__stored_tags = {'DM': {'file_version': file_version, 'file_size': file_size}}

        self.__read_tag_group(self.__stored_tags)

        if self.verbose:
            print("-- %s Tags read --" % len(self.__stored_tags))

        if self.verbose:
            t2 = time.time()
            print("| parse DM3 file: %.3g s" % (t2 - t1))

        path, file_name = os.path.split(self.__filename)
        basename, extension = os.path.splitext(file_name)
        dataset = sidpy.Dataset.from_array(self.data_cube, name=basename)
        self.__stored_tags['DM']['chosen_image'] = self.__chosen_image
        dataset.original_metadata = self.get_tags()

        self.set_dimensions(dataset)
        self.set_data_type(dataset)
        # convert linescan to spectral image
        if self.spectral_dim and dataset.ndim == 2:
            old_dataset = dataset.copy()
            meta = dataset.original_metadata.copy()
            basename = dataset.name
            data = np.array(dataset).reshape(dataset.shape[0], 1, dataset.shape[1])
            dataset = sidpy.Dataset.from_array(data, name=basename)
            dataset.original_metadata = meta
            dataset.set_dimension(0, old_dataset.dim_0)

            dataset.set_dimension(1, sidpy.Dimension([1], name='y', units='pixels',
                                                     quantity='distance', dimension_type='spatial'))
            dataset.set_dimension(2, old_dataset.dim_1)
            dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE  # 'linescan'


        dataset.quantity = 'intensity'
        dataset.units = 'counts'
        dataset.title = basename
        dataset.modality = 'generic'
        dataset.source = 'DM3Reader'
        dataset.original_metadata['DM']['full_file_name'] = self.__filename

        return dataset

    def set_data_type(self, dataset):
        spectral_dim = False
        # print(dataset._axes)
        for dim, axis in dataset._axes.items():
            if axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                spectral_dim = True
        self.spectral_dim = spectral_dim

        dataset.data_type = 'unknown'
        if 'ImageTags' in dataset.original_metadata['ImageList'][str(self.__chosen_image)]:
            image_tags = dataset.original_metadata['ImageList'][str(self.__chosen_image)]['ImageTags']
            if 'SI' in image_tags:
                if len(dataset.shape) == 3:
                    dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE
                else:
                    if spectral_dim:
                        dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE  # 'linescan'
                    else:
                        dataset.data_type = sidpy.DataType.IMAGE
                        dataset.metadata['image_type'] = 'survey image'

        if dataset.data_type == sidpy.DataType.UNKNOWN:
            if len(dataset.shape) > 3:
                raise NotImplementedError('Data_type not implemented yet')
            elif len(dataset.shape) == 3:
                if spectral_dim:
                    dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE
                else:
                    dataset.data_type = 'image_stack'
            elif len(dataset.shape) == 2:
                if spectral_dim:
                    basename = dataset.name
                    dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE
                else:
                    dataset.data_type = 'image'
            elif len(dataset.shape) == 1:
                if spectral_dim:
                    dataset.data_type = sidpy.DataType.SPECTRUM
                else:
                    dataset.data_type = sidpy.DataType.LINE_PLOT

    def set_dimensions(self, dataset):
        dimensions_dict = dataset.original_metadata['ImageList'][str(self.__chosen_image)]['ImageData']['Calibrations'][
            'Dimension']

        reciprocal_name = 'u'
        spatial_name = 'x'

        for dim, dimension_tags in dimensions_dict.items():
            # Fix annoying scale of spectrum_images in Zeiss  and SEM images
            if dimension_tags['Units'] == 'µm':
                dimension_tags['Units'] = 'nm'
                dimension_tags['Scale'] *= 1000.0

            if dimension_tags['Units'].strip() == '':
                units = 'counts'
            else:
                units = dimension_tags['Units']

            values = (np.arange(dataset.shape[int(dim)]) - dimension_tags['Origin']) * dimension_tags['Scale']

            if 'eV' == units:
                dataset.set_dimension(int(dim), sidpy.Dimension(values, name='energy_loss', units=units,
                                                                quantity='energy-loss',
                                                                dimension_type=sidpy.DimensionType.SPECTRAL))
            elif 'eV' in units:
                dataset.set_dimension(int(dim), sidpy.Dimension(values, name='energy', units=units,
                                                                quantity='energy',
                                                                dimension_type=sidpy.DimensionType.SPECTRAL))
            elif '1/' in units or units in ['mrad', 'rad']:
                dataset.set_dimension(int(dim), sidpy.Dimension(values, name=reciprocal_name, units=units,
                                                                quantity='reciprocal distance',
                                                                dimension_type=sidpy.DimensionType.RECIPROCAL))
                reciprocal_name = chr(ord(reciprocal_name) + 1)
            else:
                units = 'counts'
                dataset.set_dimension(int(dim), sidpy.Dimension(values, name=spatial_name, units=units,
                                                                quantity='distance',
                                                                dimension_type=sidpy.DimensionType.SPATIAL))
                spatial_name = chr(ord(spatial_name) + 1)

    # utility functions

    def __read_tag_group(self, tags):

        g_sorted = read_byte(self.__f)
        # is the group open?
        opened = read_byte(self.__f)
        # number of Tags
        if self.file_version == 3:
            n_tags = read_long(self.__f)
        else:
            n_tags = read_64_long(self.__f)

        # read Tags
        for i in range(n_tags):
            data = read_byte(self.__f)
            is_data = (data == 21)

            len_tag_label = read_short(self.__f)

            if len_tag_label > 0:
                tag_label = self.__f.read(len_tag_label).decode('latin-1')
            else:
                tag_label = '0'
                for key in tags:
                    if key.isdigit():
                        tag_label = str(int(key) + 1)
            if is_data:
                value = self.__read_any_data()
                tags[tag_label] = value
            else:
                tags[tag_label] = {}
                self.__read_tag_group(tags[tag_label])
        return 1

    def __encoded_type_size(self, et):
        # returns the size in bytes of the data type
        if et == 0:
            width = 0
        elif et in (BOOLEAN, CHAR, OCTET):
            width = 1
        elif et in (SHORT, USHORT):
            width = 2
        elif et in (LONG, ULONG, FLOAT):
            width = 4
        elif et == DOUBLE:
            width = 8
        else:
            # returns -1 for unrecognised types
            width = -1
        return width

    def __read_any_data(self):
        if self.file_version == 4:
            tag_byte_length = struct.unpack_from('>Q', self.__f.read(8))[0]
            # DM4 specifies this property as always big endian

        delim = read_string(self.__f, 4)
        if delim != b"%%%%":
            raise Exception(hex(self.__f.tell()) + ": Tag Type delimiter not %%%%")
        if self.file_version == 4:
            n_in_tag = read_64_long(self.__f)
        else:
            n_in_tag = read_long(self.__f)
        # higher level function dispatching to handling data types to other functions
        # - get Type category (short, long, array...)
        if self.file_version == 4:
            encoded_type = read_64_long(self.__f)
        else:
            encoded_type = read_long(self.__f)
        # - calc size of encoded_type
        et_size = self.__encoded_type_size(encoded_type)
        if et_size > 0:
            data = self.__read_native_data(encoded_type, et_size)
        elif encoded_type == STRING:
            string_size = read_long(self.__f)
            data = self.__read_string_data(string_size)
        elif encoded_type == STRUCT:
            struct_types = self.__read_struct_types()
            data = self.__read_struct_data(struct_types)
        elif encoded_type == ARRAY:
            array_types = self.__read_array_types()
            data = self.__read_array_data(array_types)
        else:
            raise Exception("rAnD, " + hex(self.__f.tell()) + ": Can't understand encoded type")
        return data

    def __read_native_data(self, encoded_type, et_size):
        # reads ordinary data types
        if encoded_type in readFunc.keys():
            val = readFunc[encoded_type](self.__f)
        else:
            raise Exception("rND, " + hex(self.__f.tell()) + ": Unknown data type " + str(encoded_type))
        return val

    def __read_string_data(self, string_size):
        # reads string data
        if string_size <= 0:
            r_string = ""
        else:
            # !!! *Unicode* string (UTF-16)... convert to Python unicode str
            r_string = read_string(self.__f, string_size)
            r_string = str(r_string, "utf_16_le")
        return r_string

    def __read_array_types(self):
        # determines the data types in an array data type
        array_type = read_long(self.__f)
        item_types = []
        if array_type == STRUCT:
            item_types = self.__read_struct_types()
        elif array_type == ARRAY:
            item_types = self.__read_array_types()
        else:
            item_types.append(array_type)
        return item_types

    def __read_array_data(self, array_types):
        # reads array data
        array_size = read_long(self.__f)
        item_size = 0
        encoded_type = 0
        for i in range(len(array_types)):
            encoded_type = int(array_types[i])
            et_size = self.__encoded_type_size(encoded_type)
            item_size += et_size
        buf_size = array_size * item_size

        if len(array_types) == 1 and encoded_type == USHORT \
                and array_size < 256:
            # treat as string
            val = self.__read_string_data(buf_size)
        else:
            # treat as binary data
            # - store data size and offset as tags
            val = self.__f.read(buf_size)
        return val

    def __read_struct_types(self):
        # analyses data types in a struct
        if self.file_version == 4:
            struct_name_length = read_64_long(self.__f)
            n_fields = read_64_long(self.__f)
        else:
            struct_name_length = read_long(self.__f)
            n_fields = read_long(self.__f)

        field_types = []
        name_length = 0
        for i in range(n_fields):
            if self.file_version == 4:
                name_length = read_64_long(self.__f)
                field_type = read_64_long(self.__f)
            else:
                name_length = read_long(self.__f)
                field_type = read_long(self.__f)
            field_types.append(field_type)
        return field_types

    def __read_struct_data(self, struct_types):
        # reads struct data based on type info in structType
        data = []
        for i in range(len(struct_types)):
            encoded_type = struct_types[i]
            et_size = self.__encoded_type_size(encoded_type)
            # get data
            data.append(self.__read_native_data(encoded_type, et_size))
        return data

    # ## END utility functions ###

    def get_filename(self):
        return self.__filename

    filename = property(get_filename)

    def get_tags(self):
        return self.__stored_tags

    tags = property(get_tags)

    def get_raw(self):
        """Extracts  data as np array"""

        # DataTypes for image data <--> PIL decoders
        data_types = {
            1: '<u2',  # 2 byte integer signed ("short")
            2: '<f4',  # 4 byte real (IEEE 754)
            3: '<c8',  # 8 byte complex (real, imaginary)
            4: '',  # ?
            # 4 byte packed complex (see below)
            5: (numpy.int16, {'real': (numpy.int8, 0), 'imaginary': (numpy.int8, 1)}),
            6: '<u1',  # 1 byte integer unsigned ("byte")
            7: '<i4',  # 4 byte integer signed ("long")
            # I do not have any dm3 file with this format to test it.
            8: '',  # rgb view, 4 bytes/pixel, unused, red, green, blue?
            9: '<i1',  # byte integer signed
            10: '<u2',  # 2 byte integer unsigned
            11: '<u4',  # 4 byte integer unsigned
            12: '<f8',  # 8 byte real
            13: '<c16',  # byte complex
            14: 'bool',  # 1 byte binary (ie 0 or 1)
            # Packed RGB. It must be a recent addition to the format because it does
            # not appear in http://www.microscopy.cen.dtu.dk/~cbb/info/dmformat/
            23: (numpy.float32,
                 {'R': ('<u1', 0), 'G': ('<u1', 1), 'B': ('<u1', 2), 'A': ('<u1', 3)}),
        }

        # find main image
        for key in self.__stored_tags['ImageList']:

            if key.isdigit():
                if 'ImageData' in self.__stored_tags['ImageList'][key]:
                    if 'Data' in self.__stored_tags['ImageList'][key]['ImageData'] \
                            and 'DataType' in self.__stored_tags['ImageList'][key]['ImageData'] \
                            and 'Dimensions' in self.__stored_tags['ImageList'][key]['ImageData']:
                        if int(key) > self.__chosen_image:
                            self.__chosen_image = int(key)
        if self.__chosen_image < 0:
            raise IOError('Did not find data in file')

        # get relevant Tags
        byte_data = self.__stored_tags['ImageList'][str(self.__chosen_image)]['ImageData']['Data']
        data_type = self.__stored_tags['ImageList'][str(self.__chosen_image)]['ImageData']['DataType']
        dimensions = self.__stored_tags['ImageList'][str(self.__chosen_image)]['ImageData']['Dimensions']

        # get shape from Dimensions
        shape = []
        for dim in dimensions:
            shape.append(dimensions[dim])

        # get data_type and reformat into numpy array
        dt = data_types[data_type]
        if dt == '':
            raise TypeError('The datatype is not supported')
        else:
            raw_data = numpy.frombuffer(byte_data, dtype=dt, count=numpy.cumprod(shape)[-1]).reshape(shape, order='F')
        # delete byte data in dictionary
        self.__stored_tags['ImageList'][str(self.__chosen_image)]['ImageData']['Data'] = 'read'
        return raw_data

    data_cube = property(get_raw)


if __name__ == '__main__':
    pass  # print "DM3lib v.%s"%version
