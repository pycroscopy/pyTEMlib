"""utility functions for sidpy; will move to sidpy"""
import numpy as np
import sidpy
import h5py
import pyNSID
import os
import ipywidgets as widgets
from IPython.display import display
import json


class EMDReader(sidpy.Reader):
    def __init__(self, file_path):
        """
        Creates an instance of EMDReader which can read one or more HDF5
        datasets formatted in the FEI Velox style EDM file

        We can read Images, and SpectrumStreams (SpectrumImages and Spectra).
        Please note that all original metadata are retained in each sidpy dataset.

        Parameters
        ----------
        file_path : str
            Path to a HDF5 file
        """

        super(EMDReader, self).__init__(file_path)
        self._h5_file = h5py.File(file_path, mode='r+')

        self.datasets = []
        self.data_array = None
        self.metadata = None
        self.number_of_frames = 1

    def read(self, eds_stream=False):
        """
        Reads all available datasets in FEI Velox style hdf5 files with .edm

        Parameters
        ----------
        eds_stream: boolean
            switch to return spectrum image (default - False) or original spectrum stream (True)

        Returns
        -------
        datasets: list of sidpy.Dataset objects
            Datasets present in the provided file
        """

        if 'Data' not in self._h5_file:
            raise TypeError('Velox EMD File is empty')
        for key in self._h5_file['Data']:
            if key == 'Image':
                for image_key in self._h5_file['Data']['Image']:
                    self.get_data('Data/Image/' + image_key)
                    self.get_image()
                    self.get_metadata(-1)
            elif key == 'SpectrumStream':
                for stream_key in self._h5_file['Data']['SpectrumStream']:
                    self.get_data('Data/SpectrumStream/' + stream_key)
                    self.get_eds(eds_stream)
                    self.get_metadata(-1)

        self.close()
        return self.datasets

    def get_data(self, image_key):
        self.data_array = self._h5_file[image_key]['Data']
        metadata_array = self._h5_file[image_key]['Metadata'][:, 0]
        metadata_string = metadata_array.tobytes().decode("utf-8")
        self.metadata = dict(json.loads(metadata_string.rstrip('\x00')))
        if 'AcquisitionSettings' in self._h5_file[image_key]:
            self.metadata['AcquisitionSettings'] = json.loads(self._h5_file[image_key]['AcquisitionSettings'][0])

    def get_eds(self, eds_stream=False):
        if 'AcquisitionSettings' not in self.metadata:
            eds_stream = True
        if eds_stream:
            self.datasets.append(sidpy.Dataset.from_array(self.data_array))
        else:
            data_array = self.get_eds_spectrum()
            if data_array.shape[0] == 1 and data_array.shape[1] == 1:
                data_array = np.array(data_array).flatten()
            self.datasets.append(sidpy.Dataset.from_array(data_array))
        # print(self.datasets[-1])

        self.datasets[-1].original_metadata = self.metadata

        detectors = self.datasets[-1].original_metadata['Detectors']
        if eds_stream:
            pass
        else:
            offset = 0.
            dispersion = 1.
            for detector in detectors.values():
                if self.metadata['BinaryResult']['Detector'] in detector['DetectorName']:
                    if 'OffsetEnergy' in detector:
                        offset = float(detector['OffsetEnergy'])
                    if 'Dispersion' in detector:
                        dispersion = float(detector['Dispersion'])

            self.datasets[-1].units = 'counts'
            self.datasets[-1].quantity = 'intensity'
            energy_scale = np.arange(self.datasets[-1].shape[-1]) * dispersion + offset

            if self.datasets[-1].ndim == 1:
                self.datasets[-1].data_type = 'spectrum'

                self.datasets[-1].set_dimension(0, sidpy.Dimension(energy_scale,
                                                                   name='energy_scale', units='eV',
                                                                   quantity='energy',
                                                                   dimension_type='spectral'))

            else:
                self.datasets[-1].data_type = 'spectral_image'
                self.datasets[-1].set_dimension(2, sidpy.Dimension(energy_scale,
                                                                   name='energy_scale', units='eV',
                                                                   quantity='energy',
                                                                   dimension_type='spectral'))
                scale_x = float(self.metadata['BinaryResult']['PixelSize']['width']) * 1e9
                scale_y = float(self.metadata['BinaryResult']['PixelSize']['height']) * 1e9

                self.datasets[-1].set_dimension(0, sidpy.Dimension(np.arange(self.datasets[-1].shape[0]) * scale_x,
                                                                   name='x', units='nm',
                                                                   quantity='distance',
                                                                   dimension_type='spatial'))
                self.datasets[-1].set_dimension(1, sidpy.Dimension(np.arange(self.datasets[-1].shape[1]) * scale_y,
                                                                   name='y', units='nm',
                                                                   quantity='distance',
                                                                   dimension_type='spatial'))

    def get_eds_spectrum(self):
        acquisition = self.metadata['AcquisitionSettings']
        size_x = 1
        size_y = 1
        if 'RasterScanDefinition' in acquisition:
            size_x = int(acquisition['RasterScanDefinition']['Width'])
            size_y = int(acquisition['RasterScanDefinition']['Height'])
        spectrum_size = int(acquisition['bincount'])

        self.number_of_frames = int(np.ceil((self.data_array[:, 0] == 65535).sum() / (size_x * size_y)))
        # print(size_x,size_y,number_of_frames)
        data = np.zeros((size_x * size_y, spectrum_size))
        # progress = tqdm(total=number_of_frames)
        pixel_number = 0
        frame = 0
        for value in self.data_array[:, 0]:
            if value == 65535:
                pixel_number += 1
                if pixel_number >= size_x * size_y:
                    pixel_number = 0
                    frame += 1
                    # print(frame)
                    # progress.update(1)
            else:
                data[pixel_number, value] += 1
        self.number_of_frames = frame
        return np.reshape(data, (size_x, size_y, spectrum_size))

    def get_image(self):

        scale_x = float(self.metadata['BinaryResult']['PixelSize']['width']) * 1e9
        scale_y = float(self.metadata['BinaryResult']['PixelSize']['height']) * 1e9

        if self.data_array.shape[2] == 1:
            self.datasets.append(sidpy.Dataset.from_array(self.data_array[:, :, 0]))
            self.datasets[-1].data_type = 'image'
            self.datasets[-1].set_dimension(0, sidpy.Dimension(np.arange(self.data_array.shape[0]) * scale_x,
                                                               name='x', units='nm',
                                                               quantity='distance',
                                                               dimension_type='spatial'))
            self.datasets[-1].set_dimension(1, sidpy.Dimension(np.arange(self.data_array.shape[1]) * scale_y,
                                                               name='y', units='nm',
                                                               quantity='distance',
                                                               dimension_type='spatial'))
        else:
             # Speedup copied from hyperspy.ioplugins.EMDReader.FEEMDReader

            data_array = np.empty(self.data_array.shape)
            self.data_array.read_direct(data_array)
            self.data_array = np.moveaxis(data_array, source=[0, 1, 2], destination=[2, 0, 1])
            
            self.data_array = np.moveaxis(self.data_array, source=[0, 1, 2], destination=[2, 0, 1])
            self.datasets.append(sidpy.Dataset.from_array(self.data_array))
            self.datasets[-1].rechunk([1, self.data_array.shape[0], self.data_array.shape[1]])
            self.datasets[-1].data_type = 'image_stack'
            self.datasets[-1].set_dimension(0, sidpy.Dimension(np.arange(self.data_array.shape[0]),
                                                               name='frame', units='frame',
                                                               quantity='time',
                                                               dimension_type='temporal'))
            self.datasets[-1].set_dimension(1, sidpy.Dimension(np.arange(self.data_array.shape[1]) * scale_x,
                                                               name='x', units='nm',
                                                               quantity='distance',
                                                               dimension_type='spatial'))
            self.datasets[-1].set_dimension(2, sidpy.Dimension(np.arange(self.data_array.shape[2]) * scale_y,
                                                               name='y', units='nm',
                                                               quantity='distance',
                                                               dimension_type='spatial'))
        self.datasets[-1].original_metadata = self.metadata

        self.datasets[-1].units = 'counts'
        self.datasets[-1].quantity = 'intensity'

    def get_metadata(self, index):
        metadata = self.datasets[index].original_metadata
        experiment = {'detector': metadata['BinaryResult']['Detector'],
                      'acceleration_voltage': float(metadata['Optics']['AccelerationVoltage']),
                      'microscope': metadata['Instrument']['InstrumentClass'],
                      'start_date_time': int(metadata['Acquisition']['AcquisitionStartDatetime']['DateTime'])}

        if metadata['Optics']['ProbeMode'] == "1":
            experiment['probe_mode'] = "convergent"
            if 'BeamConvergence' in metadata['Optics']:
                experiment['convergence_angle'] = float(metadata['Optics']['BeamConvergence'])
        else:  # metadata['Optics']['ProbeMode'] == "2":
            experiment['probe_mode'] = "parallel"
            experiment['convergence_angle'] = 0.0
        experiment['stage'] = {"holder": "",
                               "position": {"x": float(metadata['Stage']['Position']['x']),
                                            "y": float(metadata['Stage']['Position']['y']),
                                            "z": float(metadata['Stage']['Position']['z'])},
                               "tilt": {"alpha": float(metadata['Stage']['AlphaTilt']),
                                        "beta": float(metadata['Stage']['BetaTilt'])}}

        self.datasets[index].metadata['experiment'] = experiment

    def close(self):
        self._h5_file.close()


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
        # Find
        self.dataset.title = self.dataset.title.split('/')[-1]


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
