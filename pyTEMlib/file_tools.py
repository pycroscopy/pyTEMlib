##################################
#
# 2018 01 31 Included Nion Swift files to be opened
# major revision 2020 09 to include sidpy and pyNSID data formats
#
##################################

import numpy as np
import h5py
import os

# Open/Save File dialog
try:
    from PyQt5 import QtGui, QtWidgets, QtCore
    QT_available = True
except ImportError:
    QT_available = False

# =============================================================
#   Include sidpy and other pyTEMlib Libraries                                      #
# =============================================================
from .config_dir import config_path

from .nsi_reader import NSIDReader
from .dm3_reader import DM3Reader
from .nion_reader import NionReader
import pyNSID

import ipywidgets as widgets
from IPython.display import display

from .sidpy_tools import *
# import sys
# sys.path.insert(0, "../../sidpy/")
import sidpy


Dimension = sidpy.Dimension
nest_dict = sidpy.base.dict_utils.nest_dict

get_slope = sidpy.base.num_utils.get_slope
__version__ = '10.30.2020'

# TODO: new sidpy-version, uncomment and delete function below.
# flatten_dict = sidpy.dict_utils.flatten_dict


def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        if sep in k:
            k = k.replace(sep, '_')
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], dict):
                    for kk in v[i]:
                        items.append(('dim-'+kk+'-'+str(i), v[i][kk]))
                else:
                    if type(v) != bytes:
                        items.append((new_key, v))
        else:
            if type(v) != bytes:
                items.append((new_key, v))
    return dict(items)


####
#  General Open and Save Methods
####


def get_qt_app():
    """
    will start QT Application if not running yet

    :returns: QApplication

    """

    # start qt event loop
    _instance = QtWidgets.QApplication.instance()
    if not _instance:
        # print('not_instance')
        _instance = QtWidgets.QApplication([])

    return _instance


def get_last_path():
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
    if len(filename) > 1:
        fp = open(config_path + '\\path.txt', 'w')
        path, fname = os.path.split(filename)
        fp.write(path)
        fp.close()
    else:
        path = '.'
    return path


def set_directory():
    path = get_last_path()

    try:
        get_qt_app()
    except BaseException:
        pass

    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.ShowDirsOnly

    fname = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory", path, options=options))

    path = save_path(fname)

    return path


def savefile_dialog(initial_file='*.hf5', file_types=None):
    """
        Opens a save dialog in QT and returns an "*.hf5" file.
        New now with initial file
    """
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
    """
    Opens a File dialog which is used in open_file() function
    This function uses tkinter or pyQt5.
    The app of the Gui has to be running for QT so Tkinter is a safer bet.
    In jupyter notebooks use %gui Qt early in the notebook.


    The file looks first for a path.txt file for the last directory you used.

    Parameters
    ----------
    file_types : string of the file type filter


    Returns
    -------
    filename : full filename with absolute path and extension as a string

    Examples
    --------

    >> import file_tools as ft
    >>
    >> filename = ft.openfile_dialog()
    >>
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
    #
    save_path(filename)

    return filename


def open_file(filename=None, save_file=False, h5_group=None):
    """
    Opens a file if the extension is .hf5, .dm3 or .dm4
    If no filename is provided the qt open_file windows opens

    Everything will be stored in a NSID style hf5 file.

    Subroutines used:
        - NSIDReader
        - nsid.write_
            - get_main_tags
            - get_additional tags

    """
    get_qt_app()
    if filename is None:
        filename = openfile_dialog()
        if filename == '':
            return
    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)

    if extension == '.hf5':
        h5_file = h5py.File(filename, mode='a')

        h5_group = get_start_channel(h5_file)
        print()
        if 'nDim_Data' in h5_group:
            h5_dataset = h5_group['nDim_Data']

            h5_dataset.attrs['title'] = basename
            reader = NSIDReader(h5_dataset)
            dataset = reader.read_h5py_dataset(h5_dataset)
            dataset.h5_file = h5_file
        elif 'Raw_Data' in h5_group:
            dataset = read_old_h5group(h5_group)
            dataset.h5_dataset = h5_group['Raw_Data']
        else:
            reader = NSIDReader(h5_file['Measurement_000/Channel_000'])
            dataset = reader.read()[-1]
            dataset.h5_file = h5_file
        return dataset

    elif extension in ['.dm3', '.dm4', '.ndata', '.h5']:

        # tags = open_file(filename)
        if extension in ['.dm3', '.dm4']:
            reader = DM3Reader(filename)
        elif extension in ['.ndata', '.h5']:
            reader = NionReader(filename)
        else:
            IOError('problem')
        path, file_name = os.path.split(filename)
        basename, _ = os.path.splitext(file_name)
        dset = reader.read()
        dset.title = basename.strip().replace('-', '_')
        dset.filename = basename.strip().replace('-', '_')
        dset.original_metadata = flatten_dict(dset.original_metadata)

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
    return get_main_channel(h5_file)


def get_main_channel(h5_file):
    current_channel = None
    if 'Measurement_000' in h5_file:
        if 'Measurement_000/Channel_000' in h5_file:
            current_channel = h5_file['Measurement_000/Channel_000']
    return current_channel


def h5_tree(input):
    """
    Just a wrapper for the sidpy function print_tree,
    so that sidpy does not have to be loaded in notebook
    """
    if isinstance(input, sidpy.Dataset):
        if not isinstance(input.h5_dataset, h5py.Dataset):
            raise ValueError('sidpy dataset does not have an associated h5py dataset')
        h5_file = input.h5_dataset.file
    elif isinstance(input, h5py.Dataset):
        h5_file = input.file
    elif isinstance(input, (h5py.Group, h5py.File)):
        h5_file = input
    else:
        raise TypeError('should be a h5py.object or sidpy Dataset')
    sidpy.hdf_utils.print_tree(h5_file)


def log_results(h5_group, dataset=None, attributes=None):
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

        dataset.h5_dataset = log_group[dataset.title.replace('-', '_')]
    if attributes is not None:
        for key, item in attributes.items():
            if key not in log_group:
                log_group[key] = item

    return log_group


###############################################
# Support old pyTEM file format
###############################################

def read_old_h5group(current_channel):
    """
    make a  sidpy dataset from pyUSID style hdf5 group
    input
        current_channel: h5_group
    return
        sidpy Dataset
    """
    dim_dir = []
    if 'nDim_Data' in current_channel:
        h5_dataset = current_channel['nDim_Data']
        reader = NSIDReader(h5_dataset)
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
            sid_dataset.data_type = sidpy.DataTypes.SPECTRAL_IMAGE
        elif 'TEMPORAL' in dim_dir:
            sid_dataset.data_type = sidpy.DataTypes.IMAGE_STACK
        else:
            sid_dataset.data_type = sidpy.DataTypes.IMAGE
    else:

        sid_dataset.data_type = sidpy.DataTypes.SPECTRUM

    sid_dataset.quantity = 'intensity'
    sid_dataset.units = 'counts'
    if 'analysis' in current_channel:
        sid_dataset.source = current_channel['analysis'][()]

    set_dimensions(sid_dataset, current_channel)

    return sid_dataset


def set_dimensions(dset, current_channel):
    """
    Attaches correct dimension from old pyTEMlib style.
    Input:
    dset: sidpy Dataset
    current_channel: hdf5 group
    """
    dim = 0
    if dset.data_type == sidpy.DataTypes.IMAGE_STACK:
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
    if dset.data_type in [sidpy.DataTypes.SPECTRUM, sidpy.DataTypes.SPECTRAL_IMAGE]:
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
