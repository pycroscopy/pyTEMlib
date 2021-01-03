"""file_tools: All tools to load and save data
"""
##################################
#
# 2018 01 31 Included Nion Swift files to be opened
# major revision 2020 09 to include sidpy and pyNSID data formats
#
##################################

import numpy as np
import h5py
import os
import sys


# =============================================
#   Include sidpy and other pyTEMlib Libraries                                      #
# =============================================
from .config_dir import config_path

# from .nsi_reader import NSIDReader
from .dm3_reader import DM3Reader
from .nion_reader import NionReader
import pyNSID
import sidpy


from .sidpy_tools import *

# Open/Save File dialog
try:
    from PyQt5 import QtGui, QtWidgets, QtCore
    QT_available = True
except ImportError:
    QT_available = False

if "google.colab" in sys.modules:
    QT_available = False

if QT_available:
    from .qt_sidpy_tools import *


Dimension = sidpy.Dimension
nest_dict = sidpy.base.dict_utils.nest_dict

get_slope = sidpy.base.num_utils.get_slope
__version__ = '10.30.2020'

flatten_dict = sidpy.dict_utils.flatten_dict


class FileWidget(object):
    """Widget to select directories or widgets from a list

    Works in google colab.
    The widgets converts the name of the nion file to the one in Nion's swift software,
    because it is otherwise incomprehensible

    Attributes
    ----------
    dir_name: str
        name of starting directory
    extension: list of str
        extensions of files to be listed  in widget

    Methods
    -------
    get_directory
    set_options
    get_file_name

    Example
    -------
    >>from google.colab import drive
    >>drive.mount("/content/drive")
    >>file_list = pyTEMlib.file_tools.FileWidget()
    next code cell:
    >>dataset = pyTEMlib.file_tools.open_file(file_list.file_name)

    """

    def __init__(self, dir_name=None, extension=['*']):
        self.save_path = False
        self.dir_dictionary = {}
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']

        if dir_name is None:
            dir_name = get_last_path()
            self.save_path = True
            self.get_directory(dir_name)
        elif os.path.isdir(dir_name):
            self.get_directory(dir_name)
            self.dir_name = dir_name
        else:
            self.dir_name = '.'
            self.get_directory(self.dir_name)

        self.dir_list == ['.']
        self.extensions = extension
        self.file_name = ''

        self.select_files = widgets.Select(
            options=self.dir_list,
            value=self.dir_list[0],
            description='Select file:',
            disabled=False,
            rows=10,
            layout=widgets.Layout(width='70%')
        )
        display(self.select_files)
        self.set_options()
        self.select_files.observe(self.get_file_name, names='value')

    def get_directory(self, directory=None):
        self.dir_name = directory
        self.dir_dictionary = {}
        self.dir_list = []
        self.dir_list = ['.', '..'] + os.listdir(directory)

    def set_options(self):
        self.dir_name = os.path.abspath(os.path.join(self.dir_name, self.dir_list[self.select_files.index]))
        dir_list = os.listdir(self.dir_name)

        file_list = []
        display_file_list = []
        directory_list = []

        for i in range(len(dir_list)):
            name = dir_list[i]
            full_name = os.path.join(self.dir_name, name)

            if os.path.isfile(full_name):
                size = os.path.getsize(full_name) * 2 ** -20
                basename, extension = os.path.splitext(name)
                if self.extensions[0] == 'hf5':
                    if extension in ['.hf5']:
                        file_list.append(dir_list[i])
                        display_file_list.append(f" {name}  - {size:.1f} MB")
                else:
                    file_list.append(dir_list[i])
                    if extension in ['.h5', '.ndata']:
                        reader = NionReader(full_name)
                        dataset_nion = reader.read()
                        display_file_list.append(f" {dataset_nion.title}{extension}  - {size:.1f} MB")
                    elif extension in ['.hf5']:
                        display_file_list.append(f" {name}  -- {size:.1f} MB")
                    else:
                        display_file_list.append(f' {name}  - {size:.1f} MB')
            else:
                directory_list.append(name)

        sort = np.argsort(directory_list)
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']
        for j in sort:
            self.display_list.append(f' * {directory_list[j]}')
            self.dir_list.append(directory_list[j])

        sort = np.argsort(display_file_list)

        for i, j in enumerate(sort):
            if '--' in dir_list[j]:
                self.display_list.append(f' {i:3} {display_file_list[j]}')
            else:
                self.display_list.append(f' {i:3}   {display_file_list[j]}')
            self.dir_list.append(file_list[j])

        self.dir_label = os.path.split(self.dir_name)[-1] + ':'
        self.select_files.options = self.display_list

    def get_file_name(self, b):

        if os.path.isdir(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.set_options()

        elif os.path.isfile(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.file_name = os.path.join(self.dir_name, self.dir_list[self.select_files.index])


####
#  General Open and Save Methods
####


def get_qt_app():
    """will start QT Application if not running yet and returns QApplication """

    # start qt event loop
    _instance = QtWidgets.QApplication.instance()
    if not _instance:
        # print('not_instance')
        _instance = QtWidgets.QApplication([])

    return _instance


def get_last_path():
    """Returns the path of the file last opened"""
    try:
        fp = open(config_path + '\\path.txt', 'r')
        path = fp.read()
        fp.close()
    except IOError:
        path = ''

    if len(path) < 2:
        path = '.'
    return path


def save_path(filename):
    """Save path of last opened file"""

    if len(filename) > 1:
        fp = open(config_path + '\\path.txt', 'w')
        path, fname = os.path.split(filename)
        fp.write(path)
        fp.close()
    else:
        path = '.'
    return path


def savefile_dialog(initial_file='*.hf5', file_types=None):
    """Opens a save dialog in QT and returns name of file. New now with initial file"""

    # Check whether QT is available
    if not QT_available:
        print('No QT dialog')
        return None
    else:
        if file_types is None:
            file_types = "All files (*)"
    try:
        get_qt_app()
    except BaseException:
        pass

    # Determine last path used
    path = get_last_path()

    filename = sidpy.io.interface_utils.savefile_dialog(initial_file, file_types=file_types, file_path=path)
    save_path(filename)

    if len(filename) > 3:
        h5_file_name = get_h5_filename(filename)
        return h5_file_name
    else:
        return ''


def openfile_dialog(file_types=None):  # , multiple_files=False):
    """Opens a File dialog which is used in open_file() function

    This function uses pyQt5.
    The app of the Gui has to be running for QT. Tkinter does not run on Macs at this point in time.
    In jupyter notebooks use %gui Qt early in the notebook.

    The file looks first for a path.txt file for the last directory you used.

    Parameters
    ----------
    file_types : string
        file type filter in the form of '*.hf5'


    Returns
    -------
    filename : string
        full filename with absolute path and extension as a string

    Example
    -------
    >> import file_tools as ft
    >> filename = ft.openfile_dialog()
    >> print(filename)

    """

    # determine file types by extension
    if file_types is None:
        file_types = 'TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;pyNSID files (*.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'
    elif file_types == 'pyNSID':
        file_types = 'pyNSID files (*.hf5);;TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'

        # file_types = [("TEM files",["*.dm*","*.hf*","*.ndata" ]),("pyUSID files","*.hf5"),("DM files","*.dm*"),
        # ("Nion files",["*.h5","*.ndata"]),("all files","*.*")]

    # Determine last path used
    path = get_last_path()
    _ = get_qt_app()

    filename = sidpy.io.interface_utils.openfile_dialog(file_types=file_types, file_path=path)

    save_path(filename)
    return filename


def open_file(filename=None,  h5_group=None):  # save_file=False,
    """Opens a file if the extension is .hf5, .ndata, .dm3 or .dm4

    If no filename is provided the QT open_file windows opens (if QT_available==True)
    Everything will be stored in a NSID style hf5 file.
    Subroutines used:
        - NSIDReader
        - nsid.write_
            - get_main_tags
            - get_additional tags

    Parameters
    ----------
    filename: str
        name of file to be opened, if filename is None, a QT file dialog will try to open
    h5_group: hd5py.Group
        not used yet #TODO: provide hook for usage of external chosen group

    Returns
    -------
    sidpy.Dataset
        sidpy dataset with location of hdf5 dataset as attribute

    """
    if filename is None:
        if QT_available:
            get_qt_app()
            filename = openfile_dialog()
        else:
            raise TypeError('filename must be provided if QT is not installed')
    else:
        if not isinstance(filename, str):
            raise TypeError('filename must be a non-empty string or None (to a QT open file dialog)')
        elif filename == '':
            raise TypeError('filename must be a non-empty string or None (to a QT open file dialog)')

    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)

    if extension == '.hf5':
        h5_file = h5py.File(filename, mode='a')

        h5_group = get_start_channel(h5_file)
        print()
        if 'nDim_Data' in h5_group:
            h5_dataset = h5_group['nDim_Data']

            h5_dataset.attrs['title'] = basename
            reader = pyNSID.NSIDReader(h5_dataset)
            dataset = reader.read_h5py_dataset(h5_dataset)
            dataset.h5_file = h5_file
        elif 'Raw_Data' in h5_group:
            dataset = read_old_h5group(h5_group)
            dataset.h5_dataset = h5_group['Raw_Data']
        else:
            reader = pyNSID.NSIDReader(h5_file['Measurement_000/Channel_000'])
            dataset = reader.read()[-1]
            dataset.h5_file = h5_file
        return dataset

    elif extension in ['.dm3', '.dm4', '.ndata', '.h5']:

        # tags = open_file(filename)
        if extension in ['.dm3', '.dm4']:
            reader = DM3Reader(filename)
        else:   # extension in ['.ndata', '.h5']:
            reader = NionReader(filename)

        path, file_name = os.path.split(filename)
        basename, _ = os.path.splitext(file_name)
        dset = reader.read()

        if extension in ['.dm3', '.dm4']:
            dset.title = basename.strip().replace('-', '_')
        dset.filename = basename.strip().replace('-', '_')
        dset.original_metadata = flatten_dict(dset.original_metadata)
        filename = os.path.join(path,  dset.title+extension)
        h5_filename = get_h5_filename(filename)
        h5_file = h5py.File(h5_filename, mode='a')

        if 'Measurement_000' in h5_file:
            print('could not write dataset to file, try saving it with ft.save()')
        else:
            if not isinstance(h5_group, h5py.Group):
                h5_group = h5_file.create_group('Measurement_000/Channel_000')
            dset.axes = dset._axes
            dset.attrs = {}
            h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dset, h5_group)
            dset.original_metadata = nest_dict(dset.original_metadata)

            dset.h5_dataset = h5_dataset
        return dset
    else:
        print('file type not handled yet.')
        return


def get_h5_filename(fname):
    """Determines file name of hdf5 file for newly converted data file"""

    path, filename = os.path.split(fname)
    basename, extension = os.path.splitext(filename)
    h5_file_name_original = os.path.join(path, basename + '.hf5')
    h5_file_name = h5_file_name_original

    if os.path.exists(os.path.abspath(h5_file_name_original)):
        count = 1
        h5_file_name = h5_file_name_original[:-4] + '-' + str(count) + '.hf5'
        while os.path.exists(os.path.abspath(h5_file_name)):
            count += 1
            h5_file_name = h5_file_name_original[:-4] + '-' + str(count) + '.hf5'

    if h5_file_name != h5_file_name_original:
        path, filename = os.path.split(h5_file_name)
        print('Cannot overwrite file. Using: ', filename)
    return str(h5_file_name)


def get_start_channel(h5_file):
    """ Legacy for get start channel"""

    DeprecationWarning('Depreciated: use function get_main_channel instead')
    return get_main_channel(h5_file)


def get_main_channel(h5_file):
    """Returns name of first channel group in hdf5-file"""

    current_channel = None
    if 'Measurement_000' in h5_file:
        if 'Measurement_000/Channel_000' in h5_file:
            current_channel = h5_file['Measurement_000/Channel_000']
    return current_channel


def h5_tree(input_object):
    """Just a wrapper for the sidpy function print_tree,

    so that sidpy does not have to be loaded in notebook

    """

    if isinstance(input_object, sidpy.Dataset):
        if not isinstance(input_object.h5_dataset, h5py.Dataset):
            raise ValueError('sidpy dataset does not have an associated h5py dataset')
        h5_file = input_object.h5_dataset.file
    elif isinstance(input_object, h5py.Dataset):
        h5_file = input_object.file
    elif isinstance(input_object, (h5py.Group, h5py.File)):
        h5_file = input_object
    else:
        raise TypeError('should be a h5py.object or sidpy Dataset')
    sidpy.hdf_utils.print_tree(h5_file)


def log_results(h5_group, dataset=None, attributes=None):
    """Log Results in hdf5-file

    Saves either a sidpy.Dataset or dictionary in a hdf5-file.
    The group for the result will consist of 'Log_ and a running index.
    That group will be placed in h5_group.

    Parameters
    ----------
    h5_group: hd5py.Group
        groups where result group are to be stored
    dataset: sidpy.Dataset or None
        sidpy dataset to be stored
    attributes: dict
        dictionary containing results that are not based on a sidpy.Dataset

    Returns
    -------
    log_group: hd5py.Group
        group in hdf5 file with results.

    """
    if isinstance(h5_group, sidpy.Dataset):
        h5_group = h5_group.h5_dataset
        if not isinstance(h5_group, h5py.Dataset):
            raise TypeError('Use h5_dataset of sidpy.Dataset is not a valid h5py.Dataset')
        h5_group = h5_group.parent.parent

    if not isinstance(h5_group, h5py.Group):
        raise TypeError('Need a valid h5py.Group for logging results')

    if dataset is None:
        log_group = sidpy.hdf.prov_utils.create_indexed_group(h5_group, 'Log_')
    else:
        log_group = pyNSID.hdf_io.write_results(h5_group, dataset=dataset)
        if hasattr(dataset, 'meta_data'):
            metadata = sidpy.dict_utils.flatten_dict(dataset.meta_data)
            metadata_group = log_group.create_group('meta_data')
            for key, item in metadata.items():
                metadata_group.attrs[key] = item
            if 'analysis' in dataset.meta_data:
                log_group['analysis'] = dataset.meta_data['analysis']

        dataset.h5_dataset = log_group[dataset.title.replace('-', '_')][dataset.title.replace('-', '_')]
    if attributes is not None:
        for key, item in attributes.items():
            if key not in log_group:
                log_group[key] = item
    return log_group


###############################################
# Support old pyTEM file format
###############################################

def read_old_h5group(current_channel):
    """Make a  sidpy.Dataset from pyUSID style hdf5 group

    Parameters
    ----------
        current_channel: h5_group

    Returns
    -------
        sidpy.Dataset
    """

    dim_dir = []
    if 'nDim_Data' in current_channel:
        h5_dataset = current_channel['nDim_Data']
        reader = pyNSID.NSIDReader(h5_dataset)
        dataset = reader.read_h5py_dataset(h5_dataset)
        dataset.h5_file = current_channel.file
        return dataset
    elif 'Raw_Data' in current_channel:
        if 'image_stack' in current_channel:
            sid_dataset = sidpy.Dataset.from_array(np.swapaxes(current_channel['image_stack'][()], 2, 0))
            dim_dir = ['SPATIAL', 'SPATIAL', 'TEMPORAL']
        elif 'data' in current_channel:
            sid_dataset = sidpy.Dataset.from_array(current_channel['data'][()])
            dim_dir = ['SPATIAL', 'SPATIAL']
        else:
            size_x = int(current_channel['spatial_size_x'][()])
            size_y = int(current_channel['spatial_size_y'][()])
            if 'spectral_size_x' in current_channel:
                size_s = int(current_channel['spectral_size_x'][()])
            else:
                size_s = 0
            data = np.reshape(current_channel['Raw_Data'][()], (size_x, size_y, size_s))
            sid_dataset = sidpy.Dataset.from_array(data)
            if size_x > 1:
                dim_dir.append('SPATIAL')
            if size_y > 1:
                dim_dir.append('SPATIAL')
            if size_s > 1:
                dim_dir.append('SPECTRAL')
        sid_dataset.h5_dataset = current_channel['Raw_Data']

    elif 'data' in current_channel:
        sid_dataset = sidpy.Dataset.from_array(current_channel['data'][()])
        dim_dir = ['SPATIAL', 'SPATIAL']
        sid_dataset.h5_dataset = current_channel['data']
    else:
        return

    if 'SPATIAL' in dim_dir:
        if 'SPECTRAL' in dim_dir:
            sid_dataset.data_type = sidpy.DataType.SPECTRAL_IMAGE
        elif 'TEMPORAL' in dim_dir:
            sid_dataset.data_type = sidpy.DataType.IMAGE_STACK
        else:
            sid_dataset.data_type = sidpy.DataType.IMAGE
    else:
        sid_dataset.data_type = sidpy.DataType.SPECTRUM

    sid_dataset.quantity = 'intensity'
    sid_dataset.units = 'counts'
    if 'analysis' in current_channel:
        sid_dataset.source = current_channel['analysis'][()]

    set_dimensions(sid_dataset, current_channel)

    return sid_dataset


def set_dimensions(dset, current_channel):
    """Attaches correct dimension from old pyTEMlib style.

    Parameters
    ----------
    dset: sidpy.Dataset
    current_channel: hdf5.Group
    """
    dim = 0
    if dset.data_type == sidpy.DataType.IMAGE_STACK:
        dset.set_dimension(dim, sidpy.Dimension(np.arange(dset.shape[dim]), name='frame',
                                                units='frame', quantity='stack',
                                                dimension_type='TEMPORAL'))
        dim += 1
    if 'IMAGE' in dset.data_type:

        if 'spatial_scale_x' in current_channel:
            scale_x = current_channel['spatial_scale_x'][()]
        else:
            scale_x = 1
        if 'spatial_units' in current_channel:
            units_x = current_channel['spatial_units'][()]
            if len(units_x) < 2:
                units_x = 'pixel'
        else:
            units_x = 'generic'
        if 'spatial_scale_y' in current_channel:
            scale_y = current_channel['spatial_scale_y'][()]
        else:
            scale_y = 0
        dset.set_dimension(dim, sidpy.Dimension('x', np.arange(dset.shape[dim])*scale_x,
                                                units=units_x, quantity='Length',
                                                dimension_type='SPATIAL'))
        dim += 1
        dset.set_dimension(dim, sidpy.Dimension('y', np.arange(dset.shape[dim])*scale_y,
                                                units=units_x, quantity='Length',
                                                dimension_type='SPATIAL'))
        dim += 1
    if dset.data_type in [sidpy.DataType.SPECTRUM, sidpy.DataType.SPECTRAL_IMAGE]:
        if 'spectral_scale_x' in current_channel:
            scale_s = current_channel['spectral_scale_x'][()]
        else:
            scale_s = 1.0
        if 'spectral_units_x' in current_channel:
            units_s = current_channel['spectral_units_x']
        else:
            units_s = 'eV'

        if 'spectral_offset_x' in current_channel:
            offset = current_channel['spectral_offset_x']
        else:
            offset = 0.0
        dset.set_dimension(dim, sidpy.Dimension(np.arange(dset.shape[dim]) * scale_s + offset,
                                                name='energy',
                                                units=units_s,
                                                quantity='energy_loss',
                                                dimension_type='SPECTRAL'))
