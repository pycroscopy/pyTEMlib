"""utility functions for sidpy; will move to sidpy"""
import numpy as np
import sidpy
import h5py
import pyNSID
import os
import ipywidgets as widgets
from IPython.display import display


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
        self.reader = pyNSID.NSIDReader(self.current_channel.file.filename)

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


def get_dimensions_by_order(dims_in, dataset):
    """get dimension

    Parameters
    ----------
    dims_in: int or list of int
        the dimensions by numerical order
    dataset: sidpy.Dataset

    Returns
    -------
    dims_out: list of dimensions
    """

    if isinstance(dims_in, int):
        dims_in = [dims_in]
    dims_out = []
    for item in dims_in:
        if isinstance(item, int):
            if item in dataset._axes:
                dims_out.append([item, dataset._axes[item]])
    return dims_out


def get_dimensions_by_type(dims_in, dataset):
    """ get dimension by dimension_type name

    Parameters
    ----------
    dims_in: dimension_type or list of dimension_types
        the dimensions by numerical order
    dataset: sidpy.Dataset

    Returns
    -------
    dims_out: list of dimensions
    """

    if isinstance(dims_in, (str, sidpy.DimensionType)):
        dims_in = [dims_in]
    for i in range(len(dims_in)):
        if isinstance(dims_in[i], str):
            dims_in[i] = sidpy.DimensionType[dims_in[i].upper()]
    dims_out = []
    for dim, axis in dataset._axes.items():
        if axis.dimension_type in dims_in:
            dims_out.append([dim, dataset._axes[dim]])
    return dims_out


def make_dummy_dataset(value_type):
    """Make a dummy sidpy.Dataset """

    assert isinstance(value_type, sidpy.DataType)
    if type == sidpy.DataType.SPECTRUM:
        dataset = sidpy.Dataset.from_array(np.arange(100))
        dataset.data_type = 'spectrum'
        dataset.units = 'counts'
        dataset.quantity = 'intensity'

        dataset.set_dimension(0, sidpy.Dimension(np.arange(dataset.shape[0]) + 70, name='energy_scale'))
        dataset.dim_0.dimension_type = 'spectral'
        dataset.dim_0.units = 'eV'
        dataset.dim_0.quantity = 'energy loss'
    else:
        raise NotImplementedError('not implemented')
    return dataset


def plot(dataset):
    dataset.plot()


def get_image_dims(dataset):
    """Get all spatial dimensions"""

    image_dims = []
    for dim, axis in dataset._axes.items():
        if axis.dimension_type == sidpy.DimensionType.SPATIAL:
            image_dims.append(dim)
    return image_dims


def get_extent(dataset):
    """get extent to plot with matplotlib"""
    image_dims = get_image_dims(dataset)
    return dataset.get_extent(image_dims)


class FileWidget_sidpy(object):
    """Widget to select directories or widgets from a list

    Works in google colab.

    The widgets converts the name of the nion file to the one in Nion's swift software,
    because it is otherwise incomprehensible
    Use as:
    >>from google.colab import drive
    >>drive.mount("/content/drive")
    >>file_list = pyTEMlib.FileWidget()
    next code cell:
    >>dataset = pyTEMlib.open_file(file_list.file_name)

    """

    def __init__(self, dir_name='.', extension=['*']):

        if dir_name == '.':
            self.get_directory('.')
            """elif dir_name == None:
            fp = open(config_path+'\\path.txt','r')
            path = fp.read()
            fp.close()
            self.get_directory(path)
            self.dir_name = path
            """
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
                    # if extension in ['.h5', '.ndata']:
                    #    # tags = open_file(os.path.join(self.dir_name, name))
                    #    display_file_list.append(f" {tags['original_name']}{extension}  - {size:.1f} MB")
                    if extension in ['.hf5']:
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
