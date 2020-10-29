
import numpy as np
import sidpy


def get_dimensions_by_order(dims_in, dataset):
    if isinstance(dims_in, int):
        dims_in = [dims_in]
    dims_out = []
    for item in dims_in:
        if isinstance(item, int):
            if item in dataset._axes:
                dims_out.append([item, dataset._axes[item]])
    return dims_out


def get_dimensions_by_type(dims_in, dataset):
    if isinstance(dims_in, (str, sidpy.DimensionTypes)):
        dims_in = [dims_in]
    for i in range(len(dims_in)):
        if isinstance(dims_in[i], str):
            dims_in[i] = sidpy.DimensionTypes[dims_in[i].upper()]
    dims_out = []
    for dim, axis in dataset._axes.items():
        if axis.dimension_type in dims_in:
            dims_out.append([dim, dataset._axes[dim]])
    return dims_out


def make_dummy_dataset(value_type):
    assert isinstance(value_type, sidpy.DataTypes)
    if type == sidpy.DataTypes.SPECTRUM:
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
    image_dims = []
    for dim, axis in dataset._axes.items():
        if axis.dimension_type == sidpy.DimensionTypes.SPATIAL:
            image_dims.append(dim)
    return image_dims


def get_extent(dataset):
    image_dims = get_image_dims(dataset)
    return dataset.get_extent(image_dims)
