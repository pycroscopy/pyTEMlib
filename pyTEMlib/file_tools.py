"""file_tools: All tools to load and save data

##################################

 2018 01 31 Included Nion Swift files to be opened
 major revision 2020 09 to include sidpy and pyNSID data formats
 2022 change to ase format for structures: this changed the default unit of length to Angstrom!!!

##################################
"""
import typing

import os
import pickle
import numpy as np
import h5py

# For structure files of various flavor for instance POSCAR and other theory packages
import ase.io

# =============================================
#   Include  pycroscopy libraries                                      #
# =============================================
import SciFiReaders
import pyNSID
import sidpy
import ipywidgets
import IPython

# =============================================
#   Include  pyTEMlib libraries                                      #
# =============================================
from . import crystal_tools
from .config_dir import config_path
from .file_reader import adorned_to_sidpy, read_old_h5group
from .version import __version__
Dimension = sidpy.Dimension

__version__ = '2025.8.07'


ChooseDataset = sidpy.ChooseDataset

class FileWidget(sidpy.FileWidget):
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
    >>datasets = file_list.datasets
    >>dataset = file_list.selected_dataset

    """
    def __init__(self, dir_name=None, extension=['*'], sum_frames=False):
        if dir_name is None:
            dir_name = get_last_path()
            self.save_path = True
        super().__init__(dir_name=dir_name, extension=extension, sum_frames=sum_frames)
        select_button = ipywidgets.Button(description='Select Main',
                                       layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                       style=ipywidgets.ButtonStyle(button_color='lightblue'))

        add_button = ipywidgets.Button(description='Add',
                                    layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                    style=ipywidgets.ButtonStyle(button_color='lightblue'))
        self.dataset_list = ['None']
        self.selected_dataset = None
        self.datasets = {}
        self.selected_key = ''
        self.loaded_datasets = ipywidgets.Dropdown(options=self.dataset_list,
                                                value=self.dataset_list[0],
                                                description='loaded datasets:',
                                                disabled=False)

        ui = ipywidgets.HBox([select_button, add_button, self.loaded_datasets])
        IPython.display.display(ui)
        select_button.on_click(self.select_main)
        add_button.on_click(self.add_dataset)
        self.loaded_datasets.observe(self.select_dataset)

    def select_dataset(self, value: int = 0):
        """Select a dataset from the dropdown."""
        key = self.loaded_datasets.value.split(':')[0]
        if key != 'None':
            self.selected_dataset = self.datasets[key]
            self.selected_key = key

    def select_main(self, value: int = 0):
        """Select the main dataset."""
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

    def add_dataset(self, value: int = 0):
        """Add another dataset to the list of loaded datasets."""
        key = add_dataset_from_file(self.datasets, self.file_name, 'Channel')
        self.dataset_list.append(f'{key}: {self.datasets[key].title}')
        self.loaded_datasets.options = self.dataset_list
        self.loaded_datasets.value = self.dataset_list[-1]


def add_to_dict(file_dict: dict, name: str):
    """Add a file to the dictionary with its metadata."""
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
            key = list(dataset_nion.keys())[0]
            display_name = dataset_nion[key].title
            display_file_list = f" {display_name}{extension}  - {size:.1f} MB"
        except:
            display_file_list = f" {name}  - {size:.1f} MB"
    else:
        display_file_list = f' {name}  - {size:.1f} MB'
    file_dict[name] = {'display_string': display_file_list, 'basename': basename,
                       'extension': extension, 'size': size, 'display_name': display_name}


def update_directory_list(directory_name: str) -> dict:
    """Update the directory list and return the file dictionary."""
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
        if name not in dir_list and name not in ['directory', 'file_list',
                                                 'directory_list', 'display_file_list']:
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

def get_last_path() -> str:
    """Returns the path of the file last opened"""
    try:
        with open(config_path + '\\path.txt', 'r', encoding='utf-8') as file:
            path = file.read()
    except IOError:
        path = ''

    if len(path) < 2:
        path = '.'
    else:
        if not os.path.exists(path):
            path = '.'
    return path


def save_path(filename: str) -> str:
    """Save path of last opened file"""

    if len(filename) > 1:
        with open(config_path + '\\path.txt', 'w', encoding='utf-8') as file:
            path, _ = os.path.split(filename)
            file.write(path)
    else:
        path = '.'
    return path


def save_dataset(dataset, filename,  h5_group=None):
    """ Saves a dataset to a file in pyNSID format
    Parameters
    ----------
    dataset: sidpy.Dataset
        the data
    filename: str
        name of file to be opened
    h5_group: hd5py.Group
        not used yet
    """
    h5_filename = get_h5_filename(filename)
    h5_file = h5py.File(h5_filename, mode='a')
    if isinstance(dataset, dict):
        h5_group = save_dataset_dictionary(h5_file, dataset)
        return h5_group
    if isinstance(dataset, sidpy.Dataset):
        h5_dataset = save_single_dataset(h5_file, dataset, h5_group=h5_group)
        return h5_dataset.parent
    
    raise TypeError('Only sidpy.datasets or dictionaries can be saved with pyTEMlib')


def save_single_dataset(h5_file, dataset, h5_group=None):
    """
    Saves a single sidpy.Dataset to an HDF5 file.
    """
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


def save_dataset_dictionary(h5_file: h5py.File, datasets: dict) -> h5py.Group:
    """Saves a dictionary of datasets to an HDF5 file."""
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
    """
    Converts an h5py group to a python dictionary.
    """
    if not isinstance(group, h5py.Group):
        raise TypeError('we need a h5py group to read from')
    if not isinstance(group_dict, dict):
        raise TypeError('group_dict needs to be a python dictionary')

    group_dict[group.name.split('/')[-1]] = dict(group.attrs)
    for key in group.keys():
        h5_group_to_dict(group[key], group_dict[group.name.split('/')[-1]])
    return group_dict


def read_dm_annotation(image: sidpy.Dataset) -> typing.Dict[str, typing.Any]:
    """
    Reads annotations from a sidpy.Dataset that originated from a dm3 file.
    """
    if 'MAGE' not in image.data_type.name:
        return {}
    dimensions = image.get_image_dims(return_axis=True)
    scale_x = np.abs(dimensions[0].slope)
    scale_y = np.abs(dimensions[1].slope)
    rec_scale = np.array([scale_x, scale_y, scale_x, scale_y])
    annotations = {}
    tags = image.original_metadata.get('DocumentObjectList', {}).get('0', {}).get('AnnotationGroupList', {})

    if not tags:
        return annotations

    for key in tags:
        if isinstance(tags[key], dict):
            if tags[key]['AnnotationType'] == 13:  #type 'text'
                annotations[key] = {'type': 'text'}
                annotations[key]['label'] = tags[key].get('Label', '')
                rect = np.array(tags[key]['Rectangle']) * rec_scale
                annotations[key]['position'] = [rect[1], rect[0]]
                annotations[key]['text'] = tags[key].get('Text', key)
            elif tags[key]['AnnotationType']==6:
                annotations[key] = {'type': 'circle'}
                annotations[key]['label'] = tags[key].get('Label', '')
                rect = np.array(tags[key]['Rectangle']) * rec_scale
                annotations[key]['radius'] = rect[3]-rect[1]
                annotations[key]['position'] = [rect[1],rect[0]]
            elif tags[key]['AnnotationType'] == 23:
                annotations[key] = {'type':  'spectral_image'}
                annotations[key]['label'] = tags[key].get('Label', '')
                rect = np.array(tags[key].get('Rectangle', [0 ,0, 0, 0])) * rec_scale
                annotations[key]['width'] = rect[3]-rect[1]
                annotations[key]['height'] = rect[2]-rect[0]
                annotations[key]['position'] = [rect[1],rect[0]]
                annotations[key]['Rectangle'] = np.array(tags[key].get('Rectangle', [0 ,0, 0, 0]))
    if annotations:
        image.metadata['annotations'] = annotations
    return annotations


def open_file(filename, write_hdf_file=False, sum_frames=False, sum_eds=True):
    """Opens a file if the extension is .emd, .mrc, .hf5, .ndata, .dm3 or .dm4

    Everything will be stored in a NSID style hf5 file.
    Subroutines used:
        - NSIDReader
        - nsid.write_
            - get_main_tags
            - get_additional tags

    Parameters
    ----------
    filename: str
        name of file to be opened
    h5_group: hd5py.Group
        not used yet #TODO: provide hook for usage of external chosen group
    write_hdf_file: bool
        set to false so that sidpy dataset will not be written to hf5-file automatically

    Returns
    -------
    sidpy.Dataset
        sidpy dataset with location of hdf5 dataset as attribute

    """
    if not isinstance(filename, str):
        raise TypeError('filename must be a non-empty string')
    if filename == '':
        raise TypeError('filename must be a non-empty string')

    _, file_name = os.path.split(filename)
    basename, extension = os.path.splitext(file_name)
    provenance = ''
    if extension == '.hf5':
        reader = SciFiReaders.NSIDReader(filename)
        datasets = reader.read()
        if len(datasets) < 1:
            print('no hdf5 dataset found in file')
            return {}
        if isinstance(datasets, dict):
            dataset_dict = datasets
        else:
            dataset_dict = {}
            for index, dataset in enumerate(datasets):
                title = str(dataset.title).rsplit('/', maxsplit=1)[-1]
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
                dset.add_provenance('pyTEMlib', 'open_file', version=__version__,
                                    linked_data='emi_converted_by_hyperspy')
                dataset_dict[f'Channel_{index:03d}'] = dset
            return dataset_dict
        except ImportError:
            print('This file type needs hyperspy to be installed to be able to be read')
            return
    elif extension == '.emd':
        reader = SciFiReaders.EMDReader(filename, sum_frames=sum_frames)
        provenance =  'SciFiReader.EMDReader'
    elif 'edax' in extension.lower():
        if 'h5' in extension:
            reader = SciFiReaders.EDAXReader(filename)
            provenance = 'SciFiReader.EDAXReader'

    elif extension in ['.ndata', '.h5']:
        reader = SciFiReaders.NionReader(filename)
        provenance = 'SciFiReader.NionReader'

    elif extension in ['.rto']:
        reader = SciFiReaders.BrukerReader(filename)
        provenance = 'SciFiReader.BrukerReader'

    elif extension in ['.mrc']:
        reader = SciFiReaders.MRCReader(filename)
        provenance = 'SciFiReader.MRCReader'

    else:
        raise NotImplementedError('extension not supported')

    _, file_name = os.path.split(filename)
    basename, _ = os.path.splitext(file_name)

    # ### Here we read the data into sidpy datasets
    if extension != '.emi':
        dset = reader.read()

    if extension in ['.dm3', '.dm4']:
        title = (basename.strip().replace('-', '_')).split('/')[-1]
        if not isinstance(dset, dict):
            print('Please use new SciFiReaders Package for full functionality')
        if isinstance(dset, sidpy.Dataset):
            dset = {'Channel_000': dset}

        for key in dset:
            read_dm_annotation(dset[key])

    elif extension == '.emd':
        if not sum_eds:
            return
        eds_keys = []
        for key, item in dset.items():
            if item.data_type.name in ['SPECTRUM', 'SPECTRAL_IMAGE']:     
                if ('SuperX' in item.title or 'UltraX' in item.title) and item.data_type.name in ['SPECTRUM', 'SPECTRAL_IMAGE']:
                    if item.title[-2:].isnumeric():
                        if item.title[-2].isdigit():
                            if len(eds_keys) == 0:
                                spectrum = item.copy()
                            else:
                                spectrum += item
                        eds_keys.append(key)       
        if eds_keys:
            spectrum.compute()
            spectrum.data_type = dset[eds_keys[0]].data_type
            if 'SuperX' in dset[eds_keys[0]].title:
                spectrum.title = 'EDS_SuperX'
            if 'UltraX' in dset[eds_keys[0]].title:
                spectrum.title = 'EDS_UltraX'
            spectrum.original_metadata = dset[eds_keys[0]].original_metadata.copy()
            spectrum.metadata = dset[eds_keys[0]].metadata.copy()

            for key in eds_keys:
                del dset[key]
            dset['SuperX'] = spectrum

    if isinstance(dset, dict):
        dataset_dict = dset
        for dataset in dataset_dict.values():
            dataset.add_provenance('pyTEMlib', 'open_file',
                                    version=__version__,
                                    linked_data=provenance)
            dataset.metadata['filename'] = filename

    elif isinstance(dset, list):
        DeprecationWarning('Update SciFiReaders, we do not support list of datasets anymore')
    else:
        dset.filename = basename.strip().replace('-', '_')
        read_essential_metadata(dset)
        dset.metadata['filename'] = filename
        dataset_dict = {'Channel_000': dset}

        # Temporary Fix for dual eels spectra in dm files
        # Todo: Fix in SciFiReaders
        for dset in dataset_dict.values():
            if 'experiment' in dset.metadata:
                exp_meta = dset.metadata['experiment']
                if 'single_exposure_time' in exp_meta:
                    exp_meta['exposure_time'] = exp_meta['number_of_frames'] * \
                                                 exp_meta['single_exposure_time']
        if write_hdf_file:
            save_dataset(dataset_dict, filename=filename)

        save_path(filename)
    return dataset_dict


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
    if dataset.original_metadata.get('metadata', {}).get('hardware_source'):
        experiment_dictionary = read_nion_image_info(dataset.original_metadata)
    if 'experiment' not in dataset.metadata:
        dataset.metadata['experiment'] = {}
    dataset.metadata['experiment'].update(experiment_dictionary)



def read_nion_image_info(original_metadata):
    """Read essential parameter from original_metadata originating from a dm3 file"""
    if not isinstance(original_metadata, dict):
        raise TypeError('We need a dictionary to read')
    metadata = original_metadata.get('metadata', {}).get('hardware_source', {})

    return  metadata.get('ImageScanned', {})


def get_h5_filename(fname):
    """Determines file name of hdf5 file for newly converted data file"""

    path, filename = os.path.split(fname)
    basename, _ = os.path.splitext(filename)
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
            for key, dataset in datasets2.items():
                print(key)
                if isinstance(dataset, sidpy.Dataset):
                    datasets[key_name+f'_{index:03}'] = dataset
                    index += 1
                else:
                    print(key)
                    datasets[key] = dataset
            index -= 1
    else:
        return None

    return f'{key_name}_{index:03}'


# ##
# Crystal Structure Read and Write
# ##
def read_poscar(file_name):
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

    # use ase package to read file
    base = os.path.basename(file_name)
    base_name = os.path.splitext(base)[0]
    crystal = ase.io.read(file_name, format='vasp', parallel=False)

    # make dictionary and plot structure (not essential for further notebook)
    crystal.info = {'title':  base_name}
    return crystal


def read_cif(file_name, verbose=False):  # open file dialog to select cif file
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

        crystal_tags = crystal_tools.get_dictionary(input_structure)
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
        sidpy.hdf.hdf_utils.write_simple_attrs(structure_group['metadata'],
                                               input_structure['metadata'])

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
    Any additional information will be read as dictionary into the 
    info attribute of the ase.Atoms object

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

    atoms = crystal_tools.atoms_from_dictionary(crystal_tags)
    if 'metadata' in structure_group:
        atoms.info = sidpy.hdf.hdf_utils.h5_group_to_dict(structure_group)

    if 'zone_axis' in structure_group:
        atoms.info = {'experiment': {'zone_axis': structure_group['zone_axis'][()]}}
    # ToDo: Read all of info dictionary
    return atoms
