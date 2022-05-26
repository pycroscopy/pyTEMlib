"""file_tools: All tools to load and save data

##################################

 2018 01 31 Included Nion Swift files to be opened
 major revision 2020 09 to include sidpy and pyNSID data formats
 2022 change to ase format for structures: this changed the default unit of length to Angstrom!!!

##################################
"""

import numpy as np
import h5py
import os
import pickle

# For structure files of various flavor for instance POSCAR
import ase.io
import ipyfilechooser

# =============================================
#   Include  pycroscopy libraries                                      #
# =============================================
import SciFiReaders
import pyNSID
import sidpy
import ipywidgets as widgets
from IPython.display import display

# =============================================
#   Include  pyTEMlib libraries                                      #
# =============================================
import pyTEMlib.crystal_tools
from .config_dir import config_path
from .sidpy_tools import *

QT_available = False
try:
    from pyTEMlib.file_tools_qt import *
    QT_available = True
except ImportError:
    print('QT Dialogs are not available')


Dimension = sidpy.Dimension

get_slope = sidpy.base.num_utils.get_slope
__version__ = '2022.3.3'


class FileWidget(object):
    """Widget to select directories or widgets from a list

    Works in google colab.
    The widget converts the name of the nion file to the one in Nion's swift software,
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

        self.dir_name = '.'
        if dir_name is None:
            self.dir_name = get_last_path()
            self.save_path = True
        elif os.path.isdir(dir_name):
            self.dir_name = dir_name

        self.get_directory(self.dir_name)
        self.dir_list = ['.']
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
        file_dict = update_directory_list(self.dir_name)

        sort = np.argsort(file_dict['directory_list'])
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']
        for j in sort:
            self.display_list.append(f" * {file_dict['directory_list'][j]}")
            self.dir_list.append(file_dict['directory_list'][j])

        sort = np.argsort(file_dict['display_file_list'])

        for i, j in enumerate(sort):
            if '--' in dir_list[j]:
                self.display_list.append(f" {i:3} {file_dict['display_file_list'][j]}")
            else:
                self.display_list.append(f" {i:3}   {file_dict['display_file_list'][j]}")
            self.dir_list.append(file_dict['file_list'][j])

        self.dir_label = os.path.split(self.dir_name)[-1] + ':'
        self.select_files.options = self.display_list

    def get_file_name(self, b):

        if os.path.isdir(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.set_options()

        elif os.path.isfile(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.file_name = os.path.join(self.dir_name, self.dir_list[self.select_files.index])


class ChooseDataset(object):
    """Widget to select dataset object """

    def __init__(self, input_object, show_dialog=True):
        if isinstance(input_object, sidpy.Dataset):
            if isinstance(input_object.h5_dataset, h5py.Dataset):
                self.current_channel = input_object.h5_dataset.parent
        elif isinstance(input_object, h5py.Group):
            self.current_channel = input_object
        elif isinstance(input_object, h5py.Dataset):
            self.current_channel = input_object.parent
        else:
            raise ValueError('Need hdf5 group or sidpy Dataset to determine image choices')
        self.dataset_names = []
        self.dataset_list = []
        self.dataset_type = None
        self.dataset = None
        self.reader = SciFiReaders.NSIDReader(self.current_channel.file.filename)

        self.get_dataset_list()
        self.select_image = widgets.Dropdown(options=self.dataset_names,
                                             value=self.dataset_names[0],
                                             description='select dataset:',
                                             disabled=False,
                                             button_style='')
        if show_dialog:
            display(self.select_image)

        self.select_image.observe(self.set_dataset, names='value')
        self.set_dataset(0)
        self.select_image.index = (len(self.dataset_names) - 1)

    def get_dataset_list(self):
        """ Get by Log number sorted list of datasets"""
        datasets = self.reader.read()
        order = []
        for dset in datasets:
            if self.dataset_type is None or dset.data_type == self.data_type:
                if 'Log' in dset.title:
                    position = dset.title.find('Log_') + 4
                    order.append(int(dset.title[position:position + 3])+1)
                else:
                    order.append(0)
        for index in np.argsort(order):
            dset = datasets[index]
            self.dataset_names.append('/'.join(dset.title.replace('-', '_').split('/')[-1:]))
            self.dataset_list.append(dset)

    def set_dataset(self, b):
        index = self.select_image.index
        self.dataset = self.dataset_list[index]
        # Find
        self.dataset.title = self.dataset.title.split('/')[-1]


def add_to_dict(file_dict, name):
    full_name = os.path.join(file_dict['directory'], name)
    basename, extension = os.path.splitext(name)
    size = os.path.getsize(full_name) * 2 ** -20
    display_name = name
    if len(extension) == 0:
        display_file_list = f' {name}  - {size:.1f} MB'
    elif extension[0] == 'hf5':
        if extension in ['.hf5']:
            display_file_list = f" {name}  - {size:.1f} MB"
    elif extension in ['.h5', '.ndata']:
        try:
            reader = SciFiReaders.NionReader(full_name)
            dataset_nion = reader.read()
            display_name = dataset_nion.title
            display_file_list = f" {display_name}{extension}  - {size:.1f} MB"
        except:
            display_file_list = f" {name}  - {size:.1f} MB"
    else:
        display_file_list = f' {name}  - {size:.1f} MB'
    file_dict[name] = {'display_string': display_file_list, 'basename': basename, 'extension': extension,
                       'size': size, 'display_name': display_name}


def update_directory_list(directory_name):
    dir_list = os.listdir(directory_name)

    if '.pyTEMlib.files.pkl' in dir_list:
        with open(os.path.join(directory_name, '.pyTEMlib.files.pkl'), 'rb') as f:
            file_dict = pickle.load(f)
        if directory_name != file_dict['directory']:
            print('directory moved since last time read')
            file_dict['directory'] = directory_name
        dir_list.remove('.pyTEMlib.files.pkl')
    else:
        file_dict = {'directory': directory_name}

    # add new files
    file_dict['file_list'] = []
    file_dict['display_file_list'] = []
    file_dict['directory_list'] = []

    for name in dir_list:
        if os.path.isfile(os.path.join(file_dict['directory'], name)):
            if name not in file_dict:
                add_to_dict(file_dict, name)
            file_dict['file_list'].append(name)
            file_dict['display_file_list'].append(file_dict[name]['display_string'])
        else:
            file_dict['directory_list'].append(name)
    remove_item = []

    # delete items of deleted files
    save_pickle = False

    for name in file_dict.keys():
        if name not in dir_list and name not in ['directory', 'file_list', 'directory_list', 'display_file_list']:
            remove_item.append(name)
        else:
            if 'extension' in file_dict[name]:
                save_pickle = True
    for item in remove_item:
        file_dict.pop(item)

    if save_pickle:
        with open(os.path.join(file_dict['directory'], '.pyTEMlib.files.pkl'), 'wb') as f:
            pickle.dump(file_dict, f)
    return file_dict


####
#  General Open and Save Methods
####



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

"""
class open_file_dialog(ipyfilechooser.FileChooser):
    def __init__(self, directory=None):
        if directory is None:
            directory = get_last_path()
        super().__init__(directory) 
        self._use_dir_icons = True

    def _apply_selection(self):
        super()._apply_selection()
        selected = os.path.join(
            self._selected_path,
            self._selected_filename
        )

        if os.path.isfile(selected):
            self._label.value = self._LBL_TEMPLATE.format(
                self._selected_filename,
                'blue'
            )
        else:
            self._label.value = self._LBL_TEMPLATE.format(
                self._selected_filename,
                'green'
            )
        save_path(selected)
        
    def _set_form_values(self, path: str, filename: str):
        ""Set the form values.""
        # Disable triggers to prevent selecting an entry in the Select
        # box from automatically triggering a new event.
        self._pathlist.unobserve(
            self._on_pathlist_select,
            names='value'
        )
        self._dircontent.unobserve(
            self._on_dircontent_select,
            names='value'
        )
        self._filename.unobserve(
            self._on_filename_change,
            names='value'
        )

        # In folder only mode zero out the filename
        if self._show_only_dirs:
            filename = ''

        # Set form values
        self._pathlist.options = ipyfilechooser.utils.get_subpaths(path)
        self._pathlist.value = path
        self._filename.value = filename

        # file/folder real names
        dircontent_real_names = ipyfilechooser.utils.get_dir_contents(
            path,
            show_hidden=self._show_hidden,
            prepend_icons=False,
            show_only_dirs=self._show_only_dirs,
            filter_pattern=self._filter_pattern
        )

        # file/folder 
        names
        dircontent_display_names = ipyfilechooser.utils.get_dir_contents(
            path,
            show_hidden=self._show_hidden,
            prepend_icons=self._use_dir_icons,
            show_only_dirs=self._show_only_dirs,
            filter_pattern=self._filter_pattern
        )
        dircontent_display_names = self.set_display_names(dircontent_real_names, dircontent_display_names)

        # Dict to map real names to display names
        self._map_name_to_disp = {
            real_name: disp_name
            for real_name, disp_name in zip(
                dircontent_real_names,
                dircontent_display_names
            )
        }

        # Dict to map display names to real names
        self._map_disp_to_name = {
            disp_name: real_name
            for real_name, disp_name in
            self._map_name_to_disp.items()
        }

        # Set _dircontent form value to display names
        self._dircontent.options = dircontent_display_names

        # If the value in the filename Text box equals a value in the
        # Select box and the entry is a file then select the entry.
        if ((filename in dircontent_real_names) and
                os.path.isfile(os.path.join(path, filename))):
            self._dircontent.value = self._map_name_to_disp[filename]
        else:
            self._dircontent.value = None

        # Reenable triggers again
        self._pathlist.observe(
            self._on_pathlist_select,
            names='value'
        )
        self._dircontent.observe(
            self._on_dircontent_select,
            names='value'
        )
        self._filename.observe(
            self._on_filename_change,
            names='value'
        )

        # Update the state of the select button
        if self._gb.layout.display is None:
            # Disable the select button if path and filename
            # - equal an existing folder in the current view
            # - equal the already selected values
            # - don't match the provided filter pattern(s)
            check1 = filename in dircontent_real_names
            check2 = os.path.isdir(os.path.join(path, filename))
            check3 = False
            check4 = False

            # Only check selected if selected is set
            if ((self._selected_path is not None) and
                    (self._selected_filename is not None)):
                selected = os.path.join(
                    self._selected_path,
                    self._selected_filename
                )
                check3 = os.path.join(path, filename) == selected

            # Ensure only allowed extensions are used
            if self._filter_pattern:
                check4 = not ipyfilechooser.utils.match_item(filename, self._filter_pattern)

            if (check1 and check2) or check3 or check4:
                self._select.disabled = True
            else:
                self._select.disabled = False
    
    def set_display_names(self, dircontent_real_names, dircontent_display_names):
        
        for i in range(len(dircontent_display_names)):
            name = dircontent_display_names[i]
            full_name = os.path.join(self._pathlist.value, dircontent_real_names[i])
            if os.path.isfile(full_name):
                size = os.path.getsize(full_name) * 2 ** -20
                basename, extension = os.path.splitext(name)
                if extension in ['.hf5']:
                    dircontent_display_names[i] = f" {dircontent_display_names[i]:50}  -- {size:.1f} MB"
                
                elif extension in ['.h5', '.ndata']:
                    try:
                        reader = SciFiReaders.NionReader(full_name)
                        dataset_nion = reader.read()
                        dircontent_display_names[i] = f"   {dataset_nion.title+extension:50}  - {size:.1f} MB"
                    except IOError:
                        dircontent_display_names[i] = dircontent_display_names[i]
                else:
                    dircontent_display_names[i] = dircontent_display_names[i]
            
        return dircontent_display_names"""

    
def open_file_dialog_qt(file_types=None):  # , multiple_files=False):
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
    """will start QT Application if not running yet and returns QApplication """
    try:
        from PyQt5 import QtGui, QtWidgets, QtCore
        QT_available = True
    except ImportError:
        QT_available = False
        
    # determine file types by extension
    if file_types is None:
        file_types = 'TEM files (*.dm3 *.emi *.ndata *.h5 *.hf5);;pyNSID files (*.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3 *.dm4);;Nion files (*.ndata *.h5);;All files (*)'
    elif file_types == 'pyNSID':
        file_types = 'pyNSID files (*.hf5);;TEM files (*.dm3 *.qf3 *.ndata *.h5 *.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3);;Nion files (*.ndata *.h5);;All files (*)'

        # file_types = [("TEM files",["*.dm*","*.hf*","*.ndata" ]),("pyNSID files","*.hf5"),("DM files","*.dm*"),
        # ("Nion files",["*.h5","*.ndata"]),("all files","*.*")]

    # Determine last path used
    path = get_last_path()
    _ = get_qt_app()
    if QT_available:
        filename = sidpy.io.interface_utils.openfile_dialog_QT(file_types=file_types, file_path=path)

        save_path(filename)
        return filename


def save_dataset(dataset, filename=None,  h5_group=None):
    """Saves a dataset to a file in pyNSID format
    Parameters
    ----------
    dataset: sidpy.Dataset
        the data
    filename: str
        name of file to be opened, if filename is None, a QT file dialog will try to open
    h5_group: hd5py.Group
        not used yet

    """
    if filename is None:
        filename = open_file_dialog_qt()
    h5_filename = get_h5_filename(filename)
    h5_file = h5py.File(h5_filename, mode='a')
    path, file_name = os.path.split(filename)
    basename, _ = os.path.splitext(file_name)

    if h5_group is None:
        if 'Measurement_000' in h5_file:
            h5_group = sidpy.hdf.prov_utils.create_indexed_group(h5_group, 'Measurement_')
        else:
            h5_group = h5_file.create_group('Measurement_000')

        if 'Channel_000' in h5_group:
            h5_group = sidpy.hdf.prov_utils.create_indexed_group(h5_group, 'Channel_')
        else:
            h5_group = h5_group.create_group('Channel_000')

    elif isinstance(h5_group, str):
        if h5_group not in h5_file:
            h5_group = h5_file.create_group(h5_group)
        else:

            if h5_group[-1] == '/':
                h5_group = h5_group[:-1]

            channel = h5_group.split('/')[-1]
            h5_group = h5_group[:-len(channel)]
            h5_group = sidpy.hdf.prov_utils.create_indexed_group(h5_group, 'Channel_')
    else:
        raise ValueError('h5_group needs to be string or None')
    dataset.original_metadata['original_title'] = dataset.title
    dataset.title = basename
    h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dataset, h5_group)
    dataset.h5_dataset = h5_dataset
    return h5_dataset


def open_file(filename=None,  h5_group=None, write_hdf_file=True):  # save_file=False,
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
        selected_file = open_file_dialog_qt()
        filename = selected_file
        
    else:
        if not isinstance(filename, str):
            raise TypeError('filename must be a non-empty string or None (to a QT open file dialog)')
        elif filename == '':
            raise TypeError('filename must be a non-empty string or None (to a QT open file dialog)')

    path, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)

    if extension == '.hf5':
        reader = SciFiReaders.NSIDReader(filename)
        datasets = reader.read()
        if len(datasets) < 1:
            print('no hdf5 dataset found in file')
            return
        else:
            if not write_hdf_file:
                datasets[0].h5_dataset.file.close()
            return datasets[0]

        """ 
        should go to no dataset found
        if 'Raw_Data' in h5_group:
            dataset = read_old_h5group(h5_group)
            dataset.h5_dataset = h5_group['Raw_Data']
        """
    

    elif extension in ['.dm3', '.dm4', '.ndata', '.ndata1', '.h5', '.emi']:

        # tags = open_file(filename)
        if extension in ['.dm3', '.dm4']:
            reader = SciFiReaders.DM3Reader(filename)
        elif extension == '.emi':
            try:
                import hyperspy.api as hs
            except ImportError:
                print('This file type needs hyperspy to be installed to be able to be read')
            s = hs.load(filename)
            dset = SciFiReaders.convert_hyperspy(s)

        else:   # extension in ['.ndata', '.h5']:
            reader = SciFiReaders.NionReader(filename)

        path, file_name = os.path.split(filename)
        basename, _ = os.path.splitext(file_name)
        if extension != '.emi':
            dset = reader.read()

        if extension in ['.dm3', '.dm4']:
            dset.title = (basename.strip().replace('-', '_')).split('/')[-1]
            if 'PageSetup' in dset.original_metadata:
                del dset.original_metadata['PageSetup']
            if 'ImageList' in dset.original_metadata:
                if '0' in dset.original_metadata['ImageList']:
                    if 'ImageData' in dset.original_metadata['ImageList']['0']:
                        if 'Data' in dset.original_metadata['ImageList']['0']['ImageData']:
                            del dset.original_metadata['ImageList']['0']['ImageData']['Data']
        dset.filename = basename.strip().replace('-', '_')
        # dset.original_metadata = flatten_dict(dset.original_metadata)

        if write_hdf_file:
            filename = os.path.join(path,  dset.title+extension)

            h5_filename = get_h5_filename(filename)
            h5_file = h5py.File(h5_filename, mode='a')

            if 'Measurement_000' in h5_file:
                print('could not write dataset to file, try saving it with ft.save()')
            else:
                if not isinstance(h5_group, h5py.Group):
                    h5_group = h5_file.create_group('Measurement_000/Channel_000')
                # dset.axes = dset._axes
                # dset.attrs = {}
                h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dset, h5_group)

                # dset.original_metadata = nest_dict(dset.original_metadata)

                dset.h5_dataset = h5_dataset
                # pyNSID.io.hdf_utils.make_nexus_compatible(h5_dataset)

        save_path(path)
        dset.structures = []
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
    h5_group: hd5py.Group, or sidpy.Dataset
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
            if 'analysis' in dataset.meta_data:
                log_group['analysis'] = dataset.meta_data['analysis']
        if hasattr(dataset, 'structures'):
            for structure in dataset.structures:
                h5_add_crystal_structure(log_group, structure)

        dataset.h5_dataset = log_group[dataset.title.replace('-', '_')][dataset.title.replace('-', '_')]
    if attributes is not None:
        for key, item in attributes.items():
            if not isinstance(item, dict):
                log_group[key] = attributes[key]
            else:
                log_group.create_group(key)
                sidpy.hdf.hdf_utils.write_simple_attrs(log_group[key], attributes[key])
    return log_group


def add_dataset(dataset, h5_group=None):
    """Write data to hdf5 file

    Parameters
    ----------
    dataset: sidpy.Dataset
        data to write to file
    h5_group: None, sidpy.Dataset, h5py.Group, h5py.Dataset, h5py.File
        identifier to which group the data are added (if None the dataset must have a valid h5_dataset)

    Returns
    -------
    log_group: h5py.Dataset
        reference the dataset has been written to. (is also stored in h5_dataset attribute of sidpy.Dataset)
    """

    if h5_group is None:
        if isinstance(dataset.h5_dataset, h5py.Dataset):
            h5_group = dataset.h5_dataset.parent.parent.parent
    if isinstance(h5_group, h5py.Dataset):
        h5_group = h5_group.parent.parent.parent
    elif isinstance(h5_group, sidpy.Dataset):
        h5_group = h5_group.h5_dataset.parent.parent.parent
    elif isinstance(h5_group, h5py.File):
        h5_group = h5_group['Measurement_000']

    if not isinstance(h5_group, h5py.Group):
        raise TypeError('Need a valid identifier for a hdf5 group to store data in')

    structures = []
    if hasattr(dataset, 'structures'):
        structures = dataset.structures.copy()
        del dataset.structures

    log_group = sidpy.hdf.prov_utils.create_indexed_group(h5_group, 'Channel_')
    h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dataset, log_group)

    if hasattr(dataset, 'meta_data'):
        if 'analysis' in dataset.meta_data:
            log_group['analysis'] = dataset.meta_data['analysis']

    for structure in structures:
        h5_add_crystal_structure(log_group, structure)

    dataset.h5_dataset = h5_dataset
    return h5_dataset


# ##
# Crystal Structure Read and Write
# ##
def read_poscar(file_name=None):
    """
    Open a POSCAR file from Vasp
    If no file name is provided an open file dialog to select a POSCAR file appears

    Parameters
    ----------
    file_name: str
        if None is provided an open file dialog will appear

    Return
    ------
    crystal: ase.Atoms
        crystal structure in ase format
    """

    if file_name is None:
        file_name = open_file_dialog_qt('POSCAR (POSCAR*.txt);;All files (*)')

    # use ase package to read file
    base = os.path.basename(file_name)
    base_name = os.path.splitext(base)[0]
    crystal = ase.io.read(file_name, format='vasp', parallel=False)

    # make dictionary and plot structure (not essential for further notebook)
    crystal.info = {'title':  base_name}
    return crystal


def read_cif(file_name=None, verbose=False):  # open file dialog to select cif file
    """
    Open a cif file
    If no file name is provided an open file dialog to select a cif file appears

    Parameters
    ----------
    file_name: str
        if None is provided an open file dialog will appear
    verbose: bool

    Return
    ------
    crystal: ase.Atoms
        crystal structure in ase format
    """

    if file_name is None:
        file_name = open_file_dialog_qt('cif (*.cif);;All files (*)')
    # use ase package to read file

    base = os.path.basename(file_name)
    base_name = os.path.splitext(base)[0]
    crystal = ase.io.read(file_name, format='cif', store_tags=True, parallel=False)

    # make dictionary and plot structure (not essential for further notebook)
    if crystal.info is None:
        crystal.info = {'title': base_name}
    crystal.info.update({'title': base_name})
    if verbose:
        print('Opened cif file for ', crystal.get_chemical_formula())

    return crystal


def h5_add_crystal_structure(h5_file, input_structure):
    """Write crystal structure to NSID file"""

    if isinstance(input_structure, ase.Atoms):
        crystal_tags = pyTEMlib.crystal_tools.get_dictionary(input_structure)
        if crystal_tags['metadata'] == {}:
            crystal_tags['metadata'] = {'title': input_structure.get_chemical_formula()}
    elif isinstance(input_structure, dict):
        crystal_tags = input_structure
    else:
        raise TypeError('Need a dictionary or an ase.Atoms object with ase installed')

    structure_group = sidpy.hdf.prov_utils.create_indexed_group(h5_file, 'Structure_')

    for key, item in crystal_tags.items():
        if not isinstance(item, dict):
            structure_group[key] = item

    if 'base' in crystal_tags:
        structure_group['relative_positions'] = crystal_tags['base']
    if 'title' in crystal_tags:
        structure_group['title'] = str(crystal_tags['title'])
        structure_group['_' + crystal_tags['title']] = str(crystal_tags['title'])

    # ToDo: Save all of info dictionary
    if 'metadata' in input_structure:
        structure_group.create_group('metadata')
        sidpy.hdf.hdf_utils.write_simple_attrs(structure_group['metadata'], input_structure['metadata'])

    h5_file.file.flush()
    return structure_group


def h5_add_to_structure(structure_group, crystal_tags):
    """add dictionary as structure group"""

    for key in crystal_tags:
        if key in structure_group.keys():
            print(key, ' not written; use new name')
        else:
            structure_group[key] = crystal_tags[key]


def h5_get_crystal_structure(structure_group):
    """Read crystal structure  from NSID file
    Any additional information will be read as dictionary into the info attribute of the ase.Atoms object

    Parameters
    ----------
    structure_group: h5py.Group
        location in hdf5 file to where the structure information is stored

    Returns
    -------
    atoms: ase.Atoms object
        crystal structure in ase format

    """

    crystal_tags = {'unit_cell': structure_group['unit_cell'][()],
                    'base': structure_group['relative_positions'][()],
                    'title': structure_group['title'][()]}
    if '2D' in structure_group:
        crystal_tags['2D'] = structure_group['2D'][()]
    elements = structure_group['elements'][()]
    crystal_tags['elements'] = []
    for e in elements:
        crystal_tags['elements'].append(e.astype(str, copy=False))

    atoms = pyTEMlib.crystal_tools.atoms_from_dictionary(crystal_tags)
    if 'metadata' in structure_group:
        atoms.info = sidpy.hdf.hdf_utils.h5_group_to_dict(structure_group)

    if 'zone_axis' in structure_group:
        atoms.info = {'experiment': {'zone_axis': structure_group['zone_axis'][()]}}
    # ToDo: Read all of info dictionary
    return atoms


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
        reader = pyNSID.NSIDReader(h5_dataset.file.filename)
        dataset = reader.read(h5_dataset)
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
