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

# For structure files of various flavor for instance POSCAR and other theory packages
import ase.io

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
from pyTEMlib.config_dir import config_path
from pyTEMlib.sidpy_tools import *

Qt_available = True
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except ModuleNotFoundError:
    print('Qt dialogs are not available')
    Qt_available = False

Dimension = sidpy.Dimension

__version__ = '2024.9.14'

from traitlets import Unicode, Bool, validate, TraitError
import ipywidgets 


@ipywidgets.register
class FileWidget2(ipywidgets.DOMWidget):
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

    def __init__(self, dir_name=None, extension=['*'], sum_frames=False):
        self.save_path = False
        self.dir_dictionary = {}
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']
        self.sum_frames = sum_frames

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
        self.datasets = {}
        self.dataset = None

        self.select_files = widgets.Select(
            options=self.dir_list,
            value=self.dir_list[0],
            description='Select file:',
            disabled=False,
            rows=10,
            layout=widgets.Layout(width='70%')
        )
        self.path_choice = widgets.Dropdown(options=['None'],
                                            value='None',
                                            description='directory:',
                                            disabled=False,
                                            layout=widgets.Layout(width='90%'))
        
        
        
        self.set_options()
        ui = widgets.VBox([self.path_choice, self.select_files])
        display(ui)
        
        self.select_files.observe(self.get_file_name, names='value')
        self.path_choice.observe(self.set_dir, names='value')



    def get_directory(self, directory=None):
        self.dir_name = directory
        self.dir_dictionary = {}
        self.dir_list = []
        self.dir_list = ['.', '..'] + os.listdir(directory)

    def set_dir(self, value=0):
        self.dir_name = self.path_choice.value
        self.select_files.index = 0
        self.set_options()


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
        
        path = self.dir_name
        old_path = ' '
        path_list = []
        while path != old_path:
            path_list.append(path)
            old_path = path
            path = os.path.split(path)[0]
        self.path_choice.options = path_list
        self.path_choice.value = path_list[0]

    def get_file_name(self, b):

        if os.path.isdir(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.set_options()

        elif os.path.isfile(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.file_name = os.path.join(self.dir_name, self.dir_list[self.select_files.index])

class FileWidget3(FileWidget2):
    def __init__(self, dir_name=None, extension=['*'], sum_frames=False):
        if dir_name is None:
            dir_name = get_last_path()
            self.save_path = True     
        super().__init__(dir_name=dir_name, extension=extension, sum_frames=sum_frames)
        
        select_button = widgets.Button(description='Select Main',
                                       layout=widgets.Layout(width='auto', grid_area='header'),
                                       style=widgets.ButtonStyle(button_color='lightblue'))
        
        add_button = widgets.Button(description='Add',
                                    layout=widgets.Layout(width='auto', grid_area='header'),
                                    style=widgets.ButtonStyle(button_color='lightblue'))
        
        self.dataset_list = ['None']
        self.loaded_datasets = widgets.Dropdown(options=self.dataset_list,
                                                value=self.dataset_list[0],
                                                description='loaded datasets:',
                                                disabled=False)
        
        ui = widgets.HBox([select_button, add_button, self.loaded_datasets])
        display(ui)
        
        
        select_button.on_click(self.select_main)
        add_button.on_click(self.add_dataset)
        self.loaded_datasets.observe(self.select_dataset)
    

    def select_dataset(self, value=0):
        key = self.loaded_datasets.value.split(':')[0]
        if key != 'None':
            self.selected_dataset = self.datasets[key]
            self.selected_key = key

    def select_main(self, value=0):
        self.datasets = {}
        self.dataset_list = []
        self.datasets = open_file(self.file_name, sum_frames=self.sum_frames)
        self.dataset_list = []
        for key in self.datasets.keys():
            self.dataset_list.append(f'{key}: {self.datasets[key].title}')
        self.loaded_datasets.options = self.dataset_list
        self.loaded_datasets.value = self.dataset_list[0]
        self.dataset = self.datasets[list(self.datasets.keys())[0]]
        self.selected_dataset = self.dataset

    def add_dataset(self, value=0):
        key = add_dataset_from_file(self.datasets, self.file_name, 'Channel')
        self.dataset_list.append(f'{key}: {self.datasets[key].title}')
        self.loaded_datasets.options = self.dataset_list
        self.loaded_datasets.value = self.dataset_list[-1]
    

@ipywidgets.register
class FileWidget(ipywidgets.DOMWidget):
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

    def __init__(self, dir_name=None, extension=['*'], sum_frames=False):
        self.save_path = False
        self.dir_dictionary = {}
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']
        self.sum_frames = sum_frames

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
        self.datasets = {}
        self.dataset = None

        self.select_files = widgets.Select(
            options=self.dir_list,
            value=self.dir_list[0],
            description='Select file:',
            disabled=False,
            rows=10,
            layout=widgets.Layout(width='70%')
        )
        
        select_button = widgets.Button(description='Select Main',
                                       layout=widgets.Layout(width='auto', grid_area='header'),
                                       style=widgets.ButtonStyle(button_color='lightblue'))
        
        add_button = widgets.Button(description='Add',
                                    layout=widgets.Layout(width='auto', grid_area='header'),
                                    style=widgets.ButtonStyle(button_color='lightblue'))
        
        self.path_choice = widgets.Dropdown(options=['None'],
                                            value='None',
                                            description='directory:',
                                            disabled=False,
                                            layout=widgets.Layout(width='90%'))
        self.dataset_list = ['None']
        self.loaded_datasets = widgets.Dropdown(options=self.dataset_list,
                                                value=self.dataset_list[0],
                                                description='loaded datasets:',
                                                disabled=False)
        
        self.set_options()
        ui = widgets.VBox([self.path_choice, self.select_files, widgets.HBox([select_button, add_button,
                                                                              self.loaded_datasets])])
        display(ui)
        
        self.select_files.observe(self.get_file_name, names='value')
        self.path_choice.observe(self.set_dir, names='value')

        select_button.on_click(self.select_main)
        add_button.on_click(self.add_dataset)
        self.loaded_datasets.observe(self.select_dataset)

    def select_main(self, value=0):
        self.datasets = {}
        #self.loaded_datasets.value = self.dataset_list[0]
        self.dataset_list = []
        self.datasets = open_file(self.file_name, sum_frames=self.sum_frames)
        self.dataset_list = []
        for key in self.datasets.keys():
            self.dataset_list.append(f'{key}: {self.datasets[key].title}')
        self.loaded_datasets.options = self.dataset_list
        self.loaded_datasets.value = self.dataset_list[0]
        self.debug = 5
        self.dataset = self.datasets[list(self.datasets.keys())[0]]
        self.debug = 6
        self.selected_dataset = self.dataset
        
    def add_dataset(self, value=0):
        key = add_dataset_from_file(self.datasets, self.file_name, 'Channel')
        self.dataset_list.append(f'{key}: {self.datasets[key].title}')
        self.loaded_datasets.options = self.dataset_list
        self.loaded_datasets.value = self.dataset_list[-1]

    def get_directory(self, directory=None):
        self.dir_name = directory
        self.dir_dictionary = {}
        self.dir_list = []
        self.dir_list = ['.', '..'] + os.listdir(directory)

    def set_dir(self, value=0):
        self.dir_name = self.path_choice.value
        self.select_files.index = 0
        self.set_options()

    def select_dataset(self, value=0):
        
        key = self.loaded_datasets.value.split(':')[0]
        if key != 'None':
            self.selected_dataset = self.datasets[key]
            self.selected_key = key

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
        
        path = self.dir_name
        old_path = ' '
        path_list = []
        while path != old_path:
            path_list.append(path)
            old_path = path
            path = os.path.split(path)[0]
        self.path_choice.options = path_list
        self.path_choice.value = path_list[0]

    def get_file_name(self, b):

        if os.path.isdir(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.set_options()

        elif os.path.isfile(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.file_name = os.path.join(self.dir_name, self.dir_list[self.select_files.index])


class ChooseDataset(object):
    """Widget to select dataset object """

    def __init__(self, input_object, show_dialog=True):
        self.datasets = None
        if isinstance(input_object, sidpy.Dataset):
            if isinstance(input_object.h5_dataset, h5py.Dataset):
                self.current_channel = input_object.h5_dataset.parent
        elif isinstance(input_object, h5py.Group):
            self.current_channel = input_object
        elif isinstance(input_object, h5py.Dataset):
            self.current_channel = input_object.parent
        elif isinstance(input_object, dict):
            self.datasets = input_object
        else:
            raise ValueError('Need hdf5 group or sidpy Dataset to determine image choices')
        self.dataset_names = []
        self.dataset_list = []
        self.dataset_type = None
        self.dataset = None
        if not isinstance(self.datasets, dict):
            self.reader = SciFiReaders.NSIDReader(self.current_channel.file.filename)
        else:
            self.reader = None
        self.get_dataset_list()
        if len(self.dataset_list) < 1:
            self.dataset_list = ['None']
        self.select_image = widgets.Dropdown(options=self.dataset_list,
                                             value=self.dataset_list[0],
                                             description='select dataset:',
                                             disabled=False)
        if show_dialog:
            display(self.select_image)

        self.select_image.observe(self.set_dataset, names='value')
        self.set_dataset(0)
        self.select_image.index = (len(self.dataset_names) - 1)

    def get_dataset_list(self):
        """ Get by Log number sorted list of datasets"""
        dataset_list = []
        if not isinstance(self.datasets, dict):
            dataset_list = self.reader.read()
            self.datasets = {}
            for dataset in dataset_list:
                self.datasets[dataset.title] = dataset
        order = []
        keys = []
        for title, dset in self.datasets.items():
            if isinstance(dset, sidpy.Dataset):
                if self.dataset_type is None or dset.data_type == self.data_type:
                    if 'Log' in title:
                        order.append(2)
                    else:
                        order.append(0)
                    keys.append(title)
        for index in np.argsort(order):
            self.dataset_names.append(keys[index])
            self.dataset_list.append(keys[index] + ': ' + self.datasets[keys[index]].title)

    def set_dataset(self, b):
        index = self.select_image.index
        if index < len(self.dataset_names):
            self.key = self.dataset_names[index]
            self.dataset = self.datasets[self.key]
            self.dataset.title = self.dataset.title.split('/')[-1]
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
    else:
        if not os.path.exists(path):
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


if Qt_available:
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

    # determine file types by extension
    if file_types is None:
        file_types = 'TEM files (*.dm3 *.dm4 *.emd *.ndata *.h5 *.hf5);;pyNSID files (*.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3 *.dm4);;Nion files (*.ndata *.h5);;All files (*)'
    elif file_types == 'pyNSID':
        file_types = 'pyNSID files (*.hf5);;TEM files (*.dm3 *.dm4 *.qf3 *.ndata *.h5 *.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3 *.dm4);;Nion files (*.ndata *.h5);;All files (*)'

        # file_types = [("TEM files",["*.dm*","*.hf*","*.ndata" ]),("pyNSID files","*.hf5"),("DM files","*.dm*"),
        # ("Nion files",["*.h5","*.ndata"]),("all files","*.*")]

    # Determine last path used
    path = get_last_path()

    if Qt_available:
        _ = get_qt_app()
        filename = sidpy.io.interface_utils.openfile_dialog_QT(file_types=file_types, file_path=path)
        save_path(filename)
        return filename


def save_file_dialog_qt(file_types=None):  # , multiple_files=False):
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

    # determine file types by extension
    if file_types is None:
        file_types = 'pyNSID files (*.hf5);;TEM files (*.dm3 *.dm4 *.qf3 *.ndata *.h5 *.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3 *.dm4);;Nion files (*.ndata *.h5);;All files (*)'
    elif file_types == 'TEM':
        file_types = 'TEM files (*.dm3 *.dm4 *.emd *.ndata *.h5 *.hf5);;pyNSID files (*.hf5);;QF files ( *.qf3);;' \
                     'DM files (*.dm3 *.dm4);;Nion files (*.ndata *.h5);;All files (*)'

    # Determine last path used
    path = get_last_path()

    if Qt_available:
        _ = get_qt_app()
        filename = sidpy.io.interface_utils.savefile_dialog(file_types=file_types, file_path=path)
        save_path(filename)
        return filename


def save_dataset(dataset, filename=None,  qt=False, h5_group=None):
    """ Saves a dataset to a file in pyNSID format
    Parameters
    ----------
    dataset: sidpy.Dataset
        the data
    filename: str
        name of file to be opened, if filename is None, a QT file dialog will try to open
    h5_group: hd5py.Group
        not used yet
    """
    if filename is None or qt==True:
        filename = save_file_dialog_qt()
    h5_filename = get_h5_filename(filename)
    h5_file = h5py.File(h5_filename, mode='a')
    path, file_name = os.path.split(filename)
    basename, _ = os.path.splitext(file_name)

    if isinstance(dataset, dict):
        h5_group = save_dataset_dictionary(h5_file, dataset)
        return h5_group

    elif isinstance(dataset, sidpy.Dataset):
        h5_dataset = save_single_dataset(h5_file, dataset, h5_group=h5_group)
        return h5_dataset.parent
    else:
        raise TypeError('Only sidpy.datasets or dictionaries can be saved with pyTEMlib')


def save_single_dataset(h5_file, dataset, h5_group=None):
    if h5_group is None:
        h5_measurement_group = sidpy.hdf.prov_utils.create_indexed_group(h5_file, 'Measurement_')
        h5_group = sidpy.hdf.prov_utils.create_indexed_group(h5_measurement_group, 'Channel_')

    elif isinstance(h5_group, str):
        if h5_group not in h5_file:
            h5_group = h5_file.create_group(h5_group)
        else:
            if h5_group[-1] == '/':
                h5_group = h5_group[:-1]

            channel = h5_group.split('/')[-1]
            h5_measurement_group = h5_group[:-len(channel)]
            h5_group = sidpy.hdf.prov_utils.create_indexed_group(h5_group, 'Channel_')
    else:
        raise ValueError('h5_group needs to be string or None')

    h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dataset, h5_group)
    dataset.h5_dataset = h5_dataset
    h5_dataset.file.flush()
    return h5_dataset


def save_dataset_dictionary(h5_file, datasets):
    h5_measurement_group = sidpy.hdf.prov_utils.create_indexed_group(h5_file, 'Measurement_')
    for key, dataset in datasets.items():
        if key[-1] == '/':
            key = key[:-1]
        if isinstance(dataset, sidpy.Dataset):
            h5_group = h5_measurement_group.create_group(key)
            h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dataset, h5_group)
            dataset.h5_dataset = h5_dataset
            h5_dataset.file.flush()
        elif isinstance(dataset, dict):
            sidpy.hdf.hdf_utils.write_dict_to_h5_group(h5_measurement_group, dataset, key)
        else:
            print('could not save item ', key, 'of dataset dictionary')
    return h5_measurement_group


def h5_group_to_dict(group, group_dict={}):
    if not isinstance(group, h5py.Group):
        raise TypeError('we need a h5py group to read from')
    if not isinstance(group_dict, dict):
        raise TypeError('group_dict needs to be a python dictionary')

    group_dict[group.name.split('/')[-1]] = dict(group.attrs)
    for key in group.keys():
        h5_group_to_dict(group[key], group_dict[group.name.split('/')[-1]])
    return group_dict


def read_annotation(image):
    if 'MAGE' not in image.data_type.name:
        return {}
    scale_x = np.abs(image.x[1]-image.x[0])
    scale_y = np.abs(image.y[1]-image.y[0])
    rec_scale = np.array([scale_x, scale_y,scale_x, scale_y])
    if 'DocumentObjectList' not in image.original_metadata:
        return {}
    if '0' not in image.original_metadata['DocumentObjectList']:
        return {}
    annotations = {}        
    tags = image.original_metadata['DocumentObjectList']['0']     
    for key in tags:
        if 'AnnotationGroupList' in key:
            an_tags = tags[key]
            for key2 in an_tags:
                if isinstance(an_tags[key2], dict):
                    if an_tags[key2]['AnnotationType'] == 13:  #type 'text'
                        annotations[key2] = {'type': 'text'}
                        if 'Label' in an_tags:
                            annotations[key2]['label'] = an_tags['Label']
                        rect = np.array(an_tags[key2]['Rectangle']) * rec_scale
                        annotations[key2]['position'] = [rect[1],rect[0]]
                        annotations[key2]['text'] = an_tags['Text'] 
                        
                    elif an_tags[key2]['AnnotationType']==6:
                        annotations[key2] = {'type': 'circle'}
                        if 'Label' in an_tags:
                            annotations[key2]['label'] = an_tags['Label']
                        rect = np.array(an_tags[key2]['Rectangle']) * rec_scale
                    
                        annotations[key2]['radius'] =rect[3]-rect[1]
                        annotations[key2]['position'] = [rect[1],rect[0]]
        
                    elif an_tags[key2]['AnnotationType'] == 23:
                        annotations[key2] = {'type':  'spectral_image'}
                        if 'Label' in an_tags[key2]:
                            annotations[key2]['label'] = an_tags[key2]['Label']
                        rect = np.array(an_tags[key2]['Rectangle']) * rec_scale
                        
                        annotations[key2]['width'] =rect[3]-rect[1]
                        annotations[key2]['height'] =rect[2]-rect[0]
                        annotations[key2]['position'] = [rect[1],rect[0]]
                        annotations[key2]['Rectangle'] = np.array(an_tags[key2]['Rectangle'])
    if len(annotations)>0:
        image.metadata['annotations'] = annotations                
    return annotations


def open_file(filename=None,  h5_group=None, write_hdf_file=False, sum_frames=False):  # save_file=False,
    """Opens a file if the extension is .emd, .mrc, .hf5, .ndata, .dm3 or .dm4

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
    write_hdf_file: bool
        set to false so that sidpy dataset will not be written to hf5-file automatically

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
            return {}
        else:
            if isinstance(datasets, dict):
                dataset_dict = datasets

            else:    
                dataset_dict = {}
                for index, dataset in enumerate(datasets):
                    title = str(dataset.title).split('/')[-1]
                    # dataset.title = str(dataset.title).split('/')[-1]
                    dataset_dict[title] = dataset
                    if index == 0:
                        file = datasets[0].h5_dataset.file
                        master_group = datasets[0].h5_dataset.parent.parent.parent
                for key in master_group.keys():
                    if key not in dataset_dict:
                        dataset_dict[key] = h5_group_to_dict(master_group[key])
                if not write_hdf_file:
                    file.close()
            for dset in dataset_dict.values():
                if isinstance(dset, sidpy.Dataset):
                    if 'Measurement' in dset.title:
                        dset.title = dset.title.split('/')[-1]
            return dataset_dict
    elif extension in ['.dm3', '.dm4', '.ndata', '.ndata1', '.h5', '.emd', '.emi', '.edaxh5', '.mrc']:
        # tags = open_file(filename)
        if extension in ['.dm3', '.dm4']:
            reader = SciFiReaders.DMReader(filename)

        elif extension in ['.emi']:
            try:
                import hyperspy.api as hs
                s = hs.load(filename)
                dataset_dict = {}
                spectrum_number = 0
                if not isinstance(s, list):
                    s = [s]
                for index, datum in enumerate(s):
                    dset = SciFiReaders.convert_hyperspy(datum)
                    if datum.data.ndim == 1:
                        dset.title = dset.title + f'_{spectrum_number}_Spectrum'
                        spectrum_number += 1
                    elif datum.data.ndim == 3:
                        dset.title = dset.title + '_SI'
                    dset = dset.T
                    dset.title = dset.title[11:]
                    dataset_dict[f'Channel_{index:03d}'] = dset
                return dataset_dict
            except ImportError:
                print('This file type needs hyperspy to be installed to be able to be read')
                return
        elif extension == '.emd':
            reader = SciFiReaders.EMDReader(filename, sum_frames=sum_frames)
            
        elif 'edax' in extension.lower():
            if 'h5' in extension:
                reader = SciFiReaders.EDAXReader(filename)

        elif extension in ['.ndata', '.h5']:
            reader = SciFiReaders.NionReader(filename)

        elif extension in ['.mrc']:
            reader = SciFiReaders.MRCReader(filename)

        else:
            raise NotImplementedError('extension not supported')

        path, file_name = os.path.split(filename)
        basename, _ = os.path.splitext(file_name)
        if extension != '.emi':
            dset = reader.read()

        if extension in ['.dm3', '.dm4']:
            title = (basename.strip().replace('-', '_')).split('/')[-1]
            if not isinstance(dset, dict):
                print('Please use new SciFiReaders Package for full functionality')
            if isinstance(dset, sidpy.Dataset):
                dset = {'Channel_000': dset}
            for key in dset:
                read_annotation(dset[key])
        if extension == '.emd':
            for key1 in dset:
                for key in dset[key1].original_metadata:
                    if key == 'Instrument':
                        model = dset[key1].original_metadata[key]['InstrumentModel']
                        id = dset[key1].original_metadata[key]['InstrumentId']
                        dset[key1].metadata['experiment']['instrument'] = model + str(id)
                    if key == 'Optics':
                        if 'LastMeasuredScreenCurrent' in dset[key1].original_metadata[key]:
                            dset[key1].metadata['experiment']['current'] = float(dset[key1].original_metadata[key]['LastMeasuredScreenCurrent'])
                    if key == 'Scan':
                        if 'DwellTime' in dset[key1].original_metadata[key]:
                            dset[key1].metadata['experiment']['pixel_time'] = float(dset[key1].original_metadata[key]['DwellTime'])
                        if 'FrameTime' in dset[key1].original_metadata[key]:
                            dset[key1].metadata['experiment']['exposure_time'] = float(dset[key1].original_metadata[key]['FrameTime'])
                    if  key == 'Sample':
                        if 'SampleDescription' in dset[key1].original_metadata[key]:
                            dset[key1].metadata['experiment']['sample'] = dset[key1].original_metadata[key]['SampleDescription']
                        if 'SampleId' in dset[key1].original_metadata[key]:
                            dset[key1].metadata['experiment']['sample_id'] = dset[key1].original_metadata[key]['SampleId']    
                    if key == 'Detectors':
                        if 'detector' in dset[key1].metadata['experiment']:
                            used_detector = dset[key1].metadata['experiment']['detector']
                            for detector in dset[key1].original_metadata[key].values():
                                if 'DetectorName' in detector:
                                    if used_detector in detector['DetectorName']: 
                                        if 'CollectionAngleRange' in detector:
                                            begin = detector['CollectionAngleRange']['begin']
                                            end = detector['CollectionAngleRange']['end']
                                            dset[key1].metadata['experiment']['collection_angle'] = float(begin)
                                            dset[key1].metadata['experiment']['collection_angle_end'] = float(end)
        if isinstance(dset, dict):
            dataset_dict = dset
            for dataset in dataset_dict.values():
                dataset.metadata['filename'] = filename

        elif isinstance(dset, list):
            if len(dset) < 1:
                print('no dataset found in file')
                return {}
            else:
                if 'PageSetup' in dset[0].original_metadata:
                    del dset[0].original_metadata['PageSetup']
                    dset[0].original_metadata['original_title'] = title
                dataset_dict = {}
                for index, dataset in enumerate(dset):
                    if extension == '.emi':
                        if 'experiment' in dataset.metadata:
                            if 'detector' in dataset.metadata['experiment']:
                                dataset.title = dataset.metadata['experiment']['detector']
                    dataset.filename = basename.strip()
                    # read_essential_metadata(dataset)
                    dataset.metadata['filename'] = filename
                    dataset_dict[f'Channel_{index:03}'] = dataset
        else:
            dset.filename = basename.strip().replace('-', '_')
            read_essential_metadata(dset)
            dset.metadata['filename'] = filename
            dataset_dict = {'Channel_000': dset}

        # Temporary Fix for dual eels spectra in dm files
        # Todo: Fic in ScifyReaders
        for dset in dataset_dict.values(): 
            if 'experiment' in dset.metadata:
                if 'single_exposure_time' in dset.metadata['experiment']:
                    dset.metadata['experiment']['exposure_time'] = dset.metadata['experiment']['number_of_frames'] * \
                                                                dset.metadata['experiment']['single_exposure_time']
        if write_hdf_file:
            h5_master_group = save_dataset(dataset_dict, filename=filename)

        save_path(filename)
        return dataset_dict
    else:
        print('file type not handled yet.')
        return


################################################################
# Read Functions
#################################################################

def read_essential_metadata(dataset):
    """Updates dataset.metadata['experiment'] with essential information read from original metadata

    This depends on whether it is originally a nion or a dm3 file
    """
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError("we need a sidpy.Dataset")
    experiment_dictionary = {}
    if 'metadata' in dataset.original_metadata:
        if 'hardware_source' in dataset.original_metadata['metadata']:
            experiment_dictionary = read_nion_image_info(dataset.original_metadata)
    if 'DM' in dataset.original_metadata:
        experiment_dictionary = read_dm3_info(dataset.original_metadata)
    if 'experiment' not in dataset.metadata:
        dataset.metadata['experiment'] = {}

    dataset.metadata['experiment'].update(experiment_dictionary)


def read_dm3_info(original_metadata):
    """Read essential parameter from original_metadata originating from a dm3 file"""
    if not isinstance(original_metadata, dict):
        raise TypeError('We need a dictionary to read')

    if 'DM' not in original_metadata:
        return {}
    if 'ImageTags' not in original_metadata:
        return {}
    exp_dictionary = original_metadata['ImageTags']
    experiment = {}
    if 'EELS' in exp_dictionary:
        if 'Acquisition' in exp_dictionary['EELS']:
            for key, item in exp_dictionary['EELS']['Acquisition'].items():
                if 'Exposure' in key:
                    _, units = key.split('(')
                    if units[:-1] == 's':
                        experiment['single_exposure_time'] = item
                if 'Integration' in key:
                    _, units = key.split('(')
                    if units[:-1] == 's':
                        experiment['exposure_time'] = item
                if 'frames' in key:
                    experiment['number_of_frames'] = item

        if 'Experimental Conditions' in exp_dictionary['EELS']:
            for key, item in exp_dictionary['EELS']['Experimental Conditions'].items():
                if 'Convergence' in key:
                    experiment['convergence_angle'] = item
                if 'Collection' in key:
                    # print(item)
                    # for val in item.values():
                    experiment['collection_angle'] = item
        if 'number_of_frames' not in experiment:
            experiment['number_of_frames'] = 1
        if 'exposure_time' not in experiment:
            if 'single_exposure_time' in experiment:
                experiment['exposure_time'] = experiment['number_of_frames'] * experiment['single_exposure_time']

    else:
        if 'Acquisition' in exp_dictionary:
            if 'Parameters' in exp_dictionary['Acquisition']:
                if 'High Level' in exp_dictionary['Acquisition']['Parameters']:
                    if 'Exposure (s)' in exp_dictionary['Acquisition']['Parameters']['High Level']:
                        experiment['exposure_time'] = exp_dictionary['Acquisition']['Parameters']['High Level'][
                            'Exposure (s)']

    if 'Microscope Info' in exp_dictionary:
        if 'Microscope' in exp_dictionary['Microscope Info']:
            experiment['microscope'] = exp_dictionary['Microscope Info']['Microscope']
        if 'Voltage' in exp_dictionary['Microscope Info']:
            experiment['acceleration_voltage'] = exp_dictionary['Microscope Info']['Voltage']

    return experiment


def read_nion_image_info(original_metadata):
    """Read essential parameter from original_metadata originating from a dm3 file"""
    if not isinstance(original_metadata, dict):
        raise TypeError('We need a dictionary to read')
    if 'metadata' not in original_metadata:
        return {}
    if 'hardware_source' not in original_metadata['metadata']:
        return {}
    if 'ImageScanned' not in original_metadata['metadata']['hardware_source']:
        return {}

    exp_dictionary = original_metadata['metadata']['hardware_source']['ImageScanned']
    experiment = exp_dictionary
    # print(exp_dictionary)
    if 'autostem' in exp_dictionary:
        pass


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
    The group for the result will consist of 'Log_' and a running index.
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
            for structure in dataset.structures.values():
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


def add_dataset_from_file(datasets, filename=None, key_name='Log', single_dataset=True):
    """Add dataset to datasets dictionary

    Parameters
    ----------
    dataset: dict 
        dictionary to write to file
    filename: str, default: None, 
        name of file to open, if None, adialog will appear
    key_name: str, default: 'Log'
        name for key in dictionary with running number being added

    Returns
    -------
    key_name: str
        actual last used name of dictionary key
    """

    datasets2 = open_file(filename=filename)
    first_dataset = datasets2[list(datasets2)[0]]
    if isinstance(first_dataset, sidpy.Dataset):
            
        index = 0
        for key in datasets.keys():
            if key_name in key:
                if int(key[-3:]) >= index:
                    index = int(key[-3:])+1
        if single_dataset:
            datasets[key_name+f'_{index:03}'] = first_dataset
        else:
            for dataset in datasets2.values():
                datasets[key_name+f'_{index:03}'] = dataset
                index += 1
            index -= 1
    else:
        return None       

    return f'{key_name}_{index:03}'


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


def h5_add_crystal_structure(h5_file, input_structure, name=None):
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
