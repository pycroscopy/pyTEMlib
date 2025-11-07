"""file_reader:  tools to load other data files

##################################

2025 copied from file_tools

##################################
"""
import typing
import collections
import xml
import numpy as np


# =============================================
#   Include  pycroscopy libraries                                      #
# =============================================
import pyNSID
import sidpy

def etree_to_dict(element: xml.etree.ElementTree.Element) -> dict[str, typing.Any]:
    """Recursively converts an ElementTree object into a nested dictionary."""
    d = {element.tag: {} if element.attrib else None}
    children = list(element)
    if children:
        dd = collections.defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {element.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if element.attrib:
        d[element.tag].update(('@' + k, v) for k, v in element.attrib.items())
    if element.text:
        text = element.text.strip()
        if children or element.attrib:
            if text:
                d[element.tag]['#text'] = text
        else:
            d[element.tag] = text
    return d


def read_adorned_metadata(image: typing.Any) -> tuple[str, dict[str, typing.Any]]:
    """Extract metadata from an adorned image."""
    xml_str = image.metadata.metadata_as_xml
    root = xml.etree.ElementTree.fromstring(xml_str)
    metadata_dict = etree_to_dict(root)
    detector = 'detector'

    if 'Detectors' in metadata_dict['Metadata']:
        if 'ScanningDetector' in metadata_dict['Metadata']['Detectors']:
            detector = metadata_dict['Metadata']['Detectors']['ScanningDetector']['DetectorName']
        elif 'ImagingDetector' in metadata_dict['Metadata']['Detectors']:
            detector = metadata_dict['Metadata']['Detectors']['ImagingDetector']['DetectorName']
    segment = ''
    if 'CustomPropertyGroup' in  metadata_dict['Metadata']:
        if 'CustomProperties' in metadata_dict['Metadata']['CustomPropertyGroup']:
            for list_item in metadata_dict['Metadata']['CustomPropertyGroup']['CustomProperties']:
                if isinstance(list_item, dict):
                    for key in list_item:
                        for item in list_item[key]:
                            if '@name' in item:
                                if item['@name']==  'DetectorCommercialName':
                                    detector = item['@value']
                                if item['@name']== 'StemSegment':
                                    segment = '_'+item['@value']
    return detector+segment, metadata_dict['Metadata']


def get_metadata_from_adorned(ds: sidpy.Dataset):
    """Extract relevant metadata from adorned image metadata and add to sidpy.Dataset"""
    exp_meta = {}
    orig_meta = ds.original_metadata
    if 'Optics' in orig_meta:
        if 'LastMeasuredScreenCurrent' in orig_meta['Optics']:
            exp_meta['current'] = float(orig_meta['Optics']['LastMeasuredScreenCurrent'])
        if 'ConvergenceAngle' in orig_meta['Optics']:
            exp_meta['convergence_angle'] = float(orig_meta['Optics']['ConvergenceAngle'])
        if 'AccelerationVoltage' in orig_meta['Optics']:
            exp_meta['acceleration_voltage'] = float(orig_meta['Optics']['AccelerationVoltage'])
        if 'SpotIndex' in orig_meta['Optics']:
            exp_meta['spot_size'] = orig_meta['Optics']['SpotIndex']
    if 'StagesSettings' in orig_meta:
        if 'StagePosition' in orig_meta['StagesSettings']:
            exp_meta['stage_position'] = orig_meta['StagesSettings']['StagePosition']
    if 'Detectors' in orig_meta:
        if 'ScanningDetector' in orig_meta['Detectors']:
            exp_meta['detector'] = orig_meta['Detectors']['ScanningDetector']['DetectorName']
        elif 'ImagingDetector' in orig_meta['Detectors']:
            exp_meta['detector'] = orig_meta['Detectors']['ImagingDetector']['DetectorName']
            exp_meta['exposure_time'] = orig_meta['Detectors']['ImagingDetector']['ExposureTime']
    ds.metadata['experiment'] = exp_meta

def adorned_to_sidpy(images: typing.Union[list[typing.Any], typing.Any]
                     ) -> dict[str, sidpy.Dataset]:
    """
    Convert a list of adorned images to a dictionary of Sidpy datasets.
    Each dataset is created from the image data and adorned metadata.       
    The datasets are stored in a dictionary with keys 'Channel_000', 'Channel_001', etc.
    The dimensions of the datasets are set based on the image data shape and pixel sizes.
    The original metadata is also stored in the dataset.
    Args:           
        images (list or object): A list of adorned images or a single adorned image.
        Returns:    
        dict: A dictionary of Sidpy datasets, where each dataset corresponds to an image.
    """

    data_sets = {}
    if not isinstance(images, list):
        images = [images]
    for index, image in enumerate(images):
        name, original_metadata = read_adorned_metadata(image)
        data_sets[f'Channel_{index:03}'] = sidpy.Dataset.from_array(image.data.T, title=name)
        ds = data_sets[f'Channel_{index:03}']

        ds.original_metadata = original_metadata

        pixel_size_x_m = float(ds.original_metadata['BinaryResult']['PixelSize']['X']['#text'])
        pixel_size_y_m = float(ds.original_metadata['BinaryResult']['PixelSize']['Y']['#text'])
        pixel_size_x_nm = pixel_size_x_m * 1e9
        pixel_size_y_nm = pixel_size_y_m * 1e9
        if image.data.ndim == 3:
            ds.data_type = 'image_stack'
            ds.set_dimension(0, sidpy.Dimension(np.arange(image.data.shape[0]),
                                           name='frame', units='frame', quantity='Length',
                                           dimension_type='temporal'))
            ds.set_dimension(1, sidpy.Dimension(np.arange(image.data.shape[1]) * pixel_size_y_nm,
                                          name='y', units='nm', quantity='Length',
                                          dimension_type='spatial'))
            ds.set_dimension(2, sidpy.Dimension(np.arange(image.data.shape[2]) * pixel_size_x_nm,
                                          name='x', units='nm', quantity='Length',
                                          dimension_type='spatial'))
        else:
            ds.data_type = 'image'
            ds.set_dimension(0, sidpy.Dimension(np.arange(image.data.shape[0]) * pixel_size_y_nm,
                                          name='y', units='nm', quantity='Length',
                                          dimension_type='spatial'))
            ds.set_dimension(1, sidpy.Dimension(np.arange(image.data.shape[1]) * pixel_size_x_nm,
                                          name='x', units='nm', quantity='Length',
                                          dimension_type='spatial'))
        get_metadata_from_adorned(ds)
    return data_sets


###############################################
# Support old pyTEM file format
###############################################

def read_old_h5group(current_channel: typing.Any) -> typing.Optional[sidpy.Dataset]:
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
    if 'Raw_Data' in current_channel:
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
