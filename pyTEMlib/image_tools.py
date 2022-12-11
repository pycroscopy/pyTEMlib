"""
image_tools.py
by Gerd Duscher, UTK
part of pyTEMlib
MIT license except where stated differently
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.widgets as mwidgets
# from matplotlib.widgets import RectangleSelector

import sidpy
import pyTEMlib.file_tools as ft
import pyTEMlib.sidpy_tools
# import pyTEMlib.probe_tools

from tqdm.auto import trange, tqdm


# import itertools
from itertools import product

from scipy import fftpack
# from scipy import signal
from scipy.interpolate import interp1d  # , interp2d
import scipy.optimize as optimization

# Multidimensional Image library
import scipy.ndimage as ndimage
import scipy.constants as const

# from scipy.spatial import Voronoi, KDTree, cKDTree

import skimage

import skimage.registration as registration
# from skimage.feature import register_translation  # blob_dog, blob_doh
from skimage.feature import peak_local_max
# from skimage.measure import points_in_poly

# our blob detectors from the scipy image package
from skimage.feature import blob_log  # blob_dog, blob_doh

from sklearn.feature_extraction import image
from sklearn.utils.extmath import randomized_svd

_SimpleITK_present = True
try:
    import SimpleITK as sitk
except ImportError:
    sitk = False
    _SimpleITK_present = False

if not _SimpleITK_present:
    print('SimpleITK not installed; Registration Functions for Image Stacks not available\n' +
          'install with: conda install -c simpleitk simpleitk ')


# Wavelength in 1/nm
def get_wavelength(e0):
    """
    Calculates the relativistic corrected de Broglie wave length of an electron

    Parameters
    ----------
    e0: float
      acceleration voltage in volt

    Returns
    -------
    wave length in 1/nm
    """

    eV = const.e * e0
    return const.h/np.sqrt(2*const.m_e*eV*(1+eV/(2*const.m_e*const.c**2)))*10**9


def fourier_transform(dset):
    """
        Reads information into dictionary 'tags', performs 'FFT', and provides a smoothed FT and reciprocal
        and intensity limits for visualization.

        Parameters
        ----------
        dset: sidpy.Dataset
            image

        Returns
        -------
        fft_dset: sidpy.Dataset
            Fourier transform with correct dimensions

        Example
        -------
        >>> fft_dataset = fourier_transform(sidpy_dataset)
        >>> fft_dataset.plot()
    """

    assert isinstance(dset, sidpy.Dataset), 'Expected a sidpy Dataset'

    selection = []
    image_dim = []
    # image_dim = get_image_dims(sidpy.DimensionTypes.SPATIAL)

    if dset.data_type == sidpy.DataType.IMAGE_STACK:
        image_dim = dset.get_image_dims()
        stack_dim = dset.get_dimensions_by_type('TEMPORAL')

        if len(image_dim) != 2:
            raise ValueError('need at least two SPATIAL dimension for an image stack')

        for i in range(dset.dims):
            if i in image_dim:
                selection.append(slice(None))
            if len(stack_dim) == 0:
                stack_dim = i
                selection.append(slice(None))
            elif i in stack_dim:
                stack_dim = i
                selection.append(slice(None))
            else:
                selection.append(slice(0, 1))

        image_stack = np.squeeze(np.array(dset)[selection])
        new_image = np.sum(np.array(image_stack), axis=stack_dim)
    elif dset.data_type == sidpy.DataType.IMAGE:
        new_image = np.array(dset)
    else:
        return

    new_image = new_image - new_image.min()
    fft_transform = (np.fft.fftshift(np.fft.fft2(new_image)))

    image_dims = pyTEMlib.sidpy_tools.get_image_dims(dset)

    units_x = '1/' + dset._axes[image_dims[0]].units
    units_y = '1/' + dset._axes[image_dims[1]].units

    fft_dset = sidpy.Dataset.from_array(fft_transform)
    fft_dset.quantity = dset.quantity
    fft_dset.units = 'a.u.'
    fft_dset.data_type = 'IMAGE'
    fft_dset.source = dset.title
    fft_dset.modality = 'fft'

    fft_dset.set_dimension(0, sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(new_image.shape[0],
                                                                             d=ft.get_slope(dset.x.values))),

                                              name='u', units=units_x, dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))
    fft_dset.set_dimension(1, sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(new_image.shape[1],
                                                                             d=ft.get_slope(dset.y.values))),
                                              name='v', units=units_y, dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))

    return fft_dset


def power_spectrum(dset, smoothing=3):
    """
    Calculate power spectrum

    Parameters
    ----------
    dset: sidpy.Dataset
        image
    smoothing: int
        Gaussian smoothing

    Returns
    -------
    power_spec: sidpy.Dataset
        power spectrum with correct dimensions

    """

    fft_transform = fourier_transform(dset)  # dset.fft()
    fft_mag = np.abs(fft_transform)
    fft_mag2 = ndimage.gaussian_filter(fft_mag, sigma=(smoothing, smoothing), order=0)

    power_spec = fft_transform.like_data(np.log(1.+fft_mag2))

    # prepare mask
    x, y = np.meshgrid(power_spec.v.values, power_spec.u.values)
    mask = np.zeros(power_spec.shape)

    mask_spot = x ** 2 + y ** 2 > 1 ** 2
    mask = mask + mask_spot
    mask_spot = x ** 2 + y ** 2 < 11 ** 2
    mask = mask + mask_spot

    mask[np.where(mask == 1)] = 0  # just in case of overlapping disks

    minimum_intensity = np.array(power_spec)[np.where(mask == 2)].min() * 0.95
    maximum_intensity = np.array(power_spec)[np.where(mask == 2)].max() * 1.05
    power_spec.metadata = {'fft': {'smoothing': smoothing,
                                   'minimum_intensity': minimum_intensity, 'maximum_intensity': maximum_intensity}}
    power_spec.title = 'power spectrum ' + power_spec.source

    return power_spec

def diffractogram_spots(dset, spot_threshold):
    """Find spots in diffractogram and sort them by distance from center

    Uses blob_log from scipy.spatial

    Parameters
    ----------
    dset: sidpy.Dataset
        diffractogram
    spot_threshold: float
        threshold for blob finder

    Returns
    -------
    spots: numpy array
        sorted position (x,y) and radius (r) of all spots
    """

    # spot detection (for future reference there is no symmetry assumed here)
    data = np.array(np.log(1+np.abs(dset)))
    data = data - data.min()
    data = data/data.max()
    # some images are strange and blob_log does not work on the power spectrum
    try:
        spots_random = blob_log(data, max_sigma=5, threshold=spot_threshold)
    except ValueError:
        spots_random = peak_local_max(np.array(data.T), min_distance=3, threshold_rel=spot_threshold)
        spots_random = np.hstack(spots_random, np.zeros((spots_random.shape[0], 1)))
            
    print(f'Found {spots_random.shape[0]} reflections')

    # Needed for conversion from pixel to Reciprocal space
    rec_scale = np.array([ft.get_slope(dset.u.values), ft.get_slope(dset.v.values)])
    spots_random[:, :2] = spots_random[:, :2]*rec_scale+[dset.u.values[0], dset.v.values[0]]
    # sort reflections
    spots_random[:, 2] = np.linalg.norm(spots_random[:, 0:2], axis=1)
    spots_index = np.argsort(spots_random[:, 2])
    spots = spots_random[spots_index]
    # third row is angles
    spots[:, 2] = np.arctan2(spots[:, 0], spots[:, 1])
    return spots


def adaptive_fourier_filter(dset, spots, low_pass=3, reflection_radius=0.3):
    """
    Use spots in diffractogram for a Fourier Filter

    Parameters:
    -----------
    dset: sidpy.Dataset
        image to be filtered
    spots: np.ndarray(N,2)
        sorted spots in diffractogram in 1/nm
    low_pass:  float
        low pass filter in center of diffractogram in 1/nm
    reflection_radius:  float
        radius of masked reflections in 1/nm

    Output:
    -------
            Fourier filtered image
    """

    if not isinstance(dset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    fft_transform = fourier_transform(dset)

    # prepare mask
    x, y = np.meshgrid(fft_transform.v.values, fft_transform.u.values)
    mask = np.zeros(dset.shape)

    # mask reflections
    for spot in spots:
        mask_spot = (x - spot[1]) ** 2 + (y - spot[0]) ** 2 < reflection_radius ** 2  # make a spot
        mask = mask + mask_spot  # add spot to mask

    # mask zero region larger (low-pass filter = intensity variations)
    mask_spot = x ** 2 + y ** 2 < low_pass ** 2
    mask = mask + mask_spot
    mask[np.where(mask > 1)] = 1
    fft_filtered = np.array(fft_transform * mask)

    filtered_image = dset.like_data(np.fft.ifft2(np.fft.fftshift(fft_filtered)).real)
    filtered_image.title = 'Fourier filtered ' + dset.title
    filtered_image.source = dset.title
    filtered_image.metadata = {'analysis': 'adaptive fourier filtered', 'spots': spots,
                               'low_pass': low_pass, 'reflection_radius': reflection_radius}
    return filtered_image


def rotational_symmetry_diffractogram(spots):
    """ Test rotational symmetry of diffraction spots"""

    rotation_symmetry = []
    for n in [2, 3, 4, 6]:
        cc = np.array(
            [[np.cos(2 * np.pi / n), np.sin(2 * np.pi / n), 0], [-np.sin(2 * np.pi / n), np.cos(2 * np.pi / n), 0],
             [0, 0, 1]])
        sym_spots = np.dot(spots, cc)
        dif = []
        for p0, p1 in product(sym_spots[:, 0:2], spots[:, 0:2]):
            dif.append(np.linalg.norm(p0 - p1))
        dif = np.array(sorted(dif))

        if dif[int(spots.shape[0] * .7)] < 0.2:
            rotation_symmetry.append(n)
    return rotation_symmetry

#####################################################
# Registration Functions
#####################################################


def complete_registration(main_dataset, storage_channel=None):
    """Rigid and then non-rigid (demon) registration

    Performs rigid and then non-rigid registration, please see individual functions:
    - rigid_registration
    - demon_registration

    Parameters
    ----------
    main_dataset: sidpy.Dataset
        dataset of data_type 'IMAGE_STACK' to be registered
    storage_channel: h5py.Group
        optional - location in hdf5 file to store datasets

    Returns
    -------
    non_rigid_registered: sidpy.Dataset
    rigid_registered_dataset: sidpy.Dataset

    """

    if not isinstance(main_dataset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if main_dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    print('Rigid_Registration')

    rigid_registered_dataset = rigid_registration(main_dataset)
    if storage_channel is None:
        storage_channel = main_dataset.h5_dataset.parent.parent

    registration_channel = ft.log_results(storage_channel, rigid_registered_dataset)

    print('Non-Rigid_Registration')

    non_rigid_registered = demon_registration(rigid_registered_dataset)
    registration_channel = ft.log_results(storage_channel, non_rigid_registered)

    return non_rigid_registered, rigid_registered_dataset


def demon_registration(dataset, verbose=False):
    """
    Diffeomorphic Demon Non-Rigid Registration

    Depends on:
        simpleITK and numpy
    Please Cite: http://www.simpleitk.org/SimpleITK/project/parti.html
    and T. Vercauteren, X. Pennec, A. Perchant and N. Ayache
    Diffeomorphic Demons Using ITK\'s Finite Difference Solver Hierarchy
    The Insight Journal, http://hdl.handle.net/1926/510 2007

    Parameters
    ----------
    dataset: sidpy.Dataset
        stack of image after rigid registration and cropping
    verbose: boolean
        optional for increased output
    Returns
    -------
        dem_reg: stack of images with non-rigid registration

    Example
    -------
    dem_reg = demon_reg(stack_dataset, verbose=False)
    """

    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    dem_reg = np.zeros(dataset.shape)
    nimages = dataset.shape[0]
    if verbose:
        print(nimages)
    # create fixed image by summing over rigid registration

    fixed_np = np.average(np.array(dataset), axis=0)

    if not _SimpleITK_present:
        print('This feature is not available: \n Please install simpleITK with: conda install simpleitk -c simpleitk')

    fixed = sitk.GetImageFromArray(fixed_np)
    fixed = sitk.DiscreteGaussian(fixed, 2.0)

    # demons = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons = sitk.DiffeomorphicDemonsRegistrationFilter()

    demons.SetNumberOfIterations(200)
    demons.SetStandardDeviations(1.0)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    done = 0

    for i in trange(nimages):

        moving = sitk.GetImageFromArray(dataset[i])
        moving_f = sitk.DiscreteGaussian(moving, 2.0)
        displacement_field = demons.Execute(fixed, moving_f)
        out_tx = sitk.DisplacementFieldTransform(displacement_field)
        resampler.SetTransform(out_tx)
        out = resampler.Execute(moving)
        dem_reg[i, :, :] = sitk.GetArrayFromImage(out)

    print(':-)')
    print('You have successfully completed Diffeomorphic Demons Registration')

    demon_registered = dataset.like_data(dem_reg)
    demon_registered.title = 'Non-Rigid Registration'
    demon_registered.source = dataset.title

    demon_registered.metadata = {'analysis': 'non-rigid demon registration'}
    if 'input_crop' in dataset.metadata:
        demon_registered.metadata['input_crop'] = dataset.metadata['input_crop']
    if 'input_shape' in dataset.metadata:
        demon_registered.metadata['input_shape'] = dataset.metadata['input_shape']
    demon_registered.metadata['input_dataset'] = dataset.source
    return demon_registered


###############################
# Rigid Registration New 05/09/2020

def rigid_registration(dataset):
    """
    Rigid registration of image stack with sub-pixel accuracy

    Uses phase_cross_correlation from skimage.registration
    (we determine drift from one image to next)

    Parameters
    ----------
    dataset: sidpy.Dataset
        sidpy dataset with image_stack dataset

    Returns
    -------
    rigid_registered: sidpy.Dataset
        Registered Stack and drift (with respect to center image)
    """

    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Registration makes only sense for an image stack')

    frame_dim = dataset.get_dimensions_by_type(['temporal'])
    image_dims = dataset.get_dimensions_by_type(['spatial'])

    nopix = dataset.shape[image_dims[0]]
    nopiy = dataset.shape[image_dims[1]]
    nimages = dataset.shape[frame_dim[0]]

    print('Stack contains ', nimages, ' images, each with', nopix, ' pixels in x-direction and ', nopiy,
          ' pixels in y-direction')

    data_array = np.moveaxis(np.array(dataset), frame_dim[0], 0)

    fixed = np.array(data_array[0])
    fft_fixed = np.fft.fft2(fixed)

    relative_drift = [[0., 0.]]

    for i in trange(nimages):
        moving = np.array(data_array[i])
        fft_moving = np.fft.fft2(moving)
        if skimage.__version__[:4] == '0.16':
            raise DeprecationWarning('Old scikit image version does not work')
        else:
            shift = registration.phase_cross_correlation(fft_fixed, fft_moving, upsample_factor=1000, space='fourier')
        fft_fixed = fft_moving
        relative_drift.append(shift[0])

    rig_reg, drift = rig_reg_drift(data_array, relative_drift)

    crop_reg, input_crop = crop_image_stack(rig_reg, drift)

    rigid_registered = sidpy.Dataset.from_array(crop_reg)
    rigid_registered.set_dimension(0, dataset._axes[frame_dim[0]])
    rigid_registered.set_dimension(1, dataset._axes[image_dims[0]][input_crop[0]:input_crop[1]])
    rigid_registered.set_dimension(2, dataset._axes[image_dims[1]][input_crop[2]:input_crop[3]])
    rigid_registered.data_type = 'Image_stack'

    rigid_registered.title = 'Rigid Registration'
    rigid_registered.source = dataset.title
    rigid_registered.metadata = {'analysis': 'rigid sub-pixel registration', 'drift': drift,
                                 'input_crop': input_crop, 'input_shape': np.array(dataset.shape)[image_dims]}

    return rigid_registered


def rig_reg_drift(dset, rel_drift):
    """ Shifting images on top of each other

    Uses relative drift to shift images on top of each other,
    with center image as reference.
    Shifting is done with shift routine of ndimage from scipy.
    This function is used by rigid_registration routine

    Parameters
    ----------
    dset: sidpy.Dataset
        dataset with image_stack
    rel_drift:
        relative_drift from image to image as list of [shiftx, shifty]

    Returns
    -------
    stack: numpy array
    drift: list of drift in pixel
    """

    rig_reg = np.zeros(dset.shape)
    # absolute drift
    drift = np.array(rel_drift).copy()

    drift[0] = [0, 0]
    for i in range(drift.shape[0]):
        drift[i] = drift[i - 1] + rel_drift[i]
    center_drift = drift[int(drift.shape[0] / 2)]
    drift = drift - center_drift
    # Shift images
    for i in range(rig_reg.shape[0]):
        # Now we shift
        rig_reg[i, :, :] = ndimage.shift(dset[i], [drift[i, 0], drift[i, 1]], order=3)
    return rig_reg, drift


def crop_image_stack(rig_reg, drift):
    """Crop images in stack according to drift

    This function is used by rigid_registration routine

    Parameters
    ----------
    rig_reg: numpy array (N,x,y)
    drift: list (2,B)

    Returns
    -------
    numpy array
    """

    xpmin = int(-np.floor(np.min(np.array(drift)[:, 0])))
    xpmax = int(rig_reg.shape[1] - np.ceil(np.max(np.array(drift)[:, 0])))
    ypmin = int(-np.floor(np.min(np.array(drift)[:, 1])))
    ypmax = int(rig_reg.shape[2] - np.ceil(np.max(np.array(drift)[:, 1])))

    return rig_reg[:, xpmin:xpmax, ypmin:ypmax], [xpmin, xpmax, ypmin, ypmax]


class ImageWithLineProfile:
    """Image with line profile"""

    def __init__(self, data, extent, title=''):
        fig, ax = plt.subplots(1, 1)
        self.figure = fig
        self.title = title
        self.line_plot = False
        self.ax = ax
        self.data = data
        self.extent = extent
        self.ax.imshow(data, extent=extent)
        self.ax.set_title(title)
        self.line,  = self.ax.plot([0], [0], color='orange')  # empty line
        self.end_x = self.line.get_xdata()
        self.end_y = self.line.get_ydata()
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes != self.line.axes:
            return
        self.start_x = self.end_x
        self.start_y = self.end_y

        self.line.set_data([self.start_x, event.xdata], [self.start_y, event.ydata])
        self.line.figure.canvas.draw()

        self.end_x = event.xdata
        self.end_y = event.ydata

        self.update()

    def update(self):
        if not self.line_plot:
            self.line_plot = True
            self.figure.clear()
            self.ax = self.figure.subplots(2, 1)
            self.ax[0].imshow(self.data, extent=self.extent)
            self.ax[0].set_title(self.title)

            self.line,  = self.ax[0].plot([0], [0], color='orange')  # empty line
            self.line_plot, = self.ax[1].plot([], [], color='orange')
            self.ax[1].set_xlabel('distance [nm]')

        x0 = self.start_x
        x1 = self.end_x
        y0 = self.start_y
        y1 = self.end_y
        length_plot = np.sqrt((x1-x0)**2+(y1-y0)**2)

        num = length_plot*(self.data.shape[0]/self.extent[1])
        x = np.linspace(x0, x1, num)*(self.data.shape[0]/self.extent[1])
        y = np.linspace(y0, y1, num)*(self.data.shape[0]/self.extent[1])

        # Extract the values along the line, using cubic interpolation
        zi2 = ndimage.map_coordinates(self.data.T, np.vstack((x, y)))

        x_axis = np.linspace(0, length_plot, len(zi2))
        self.x = x_axis
        self.z = zi2

        self.line_plot.set_xdata(x_axis)
        self.line_plot.set_ydata(zi2)
        self.ax[1].set_xlim(0, x_axis.max())
        self.ax[1].set_ylim(zi2.min(), zi2.max())
        self.ax[1].draw()


def histogram_plot(image_tags):
    """interactive histogram"""
    nbins = 75
    color_map_list = ['gray', 'viridis', 'jet', 'hot']
    if 'minimum_intensity' not in image_tags:
        image_tags['minimum_intensity'] = image_tags['plotimage'].min()
    minimum_intensity = image_tags['minimum_intensity']
    if 'maximum_intensity' not in image_tags:
        image_tags['maximum_intensity'] = image_tags['plotimage'].max()
    data = image_tags['plotimage']
    vmin = image_tags['minimum_intensity']
    vmax = image_tags['maximum_intensity']
    if 'color_map' not in image_tags:
        image_tags['color_map'] = color_map_list[0]
    cmap = plt.cm.get_cmap(image_tags['color_map'])

    colors = cmap(np.linspace(0., 1., nbins))

    norm2 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    hist, bin_edges = np.histogram(data, np.linspace(vmin, vmax, nbins), density=True)

    width = bin_edges[1]-bin_edges[0]

    def onselect(vmin, vmax):
        ax1.clear()
        cmap = plt.cm.get_cmap(image_tags['color_map'])

        colors = cmap(np.linspace(0., 1., nbins))

        norm2 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        hist2, bin_edges2 = np.histogram(data, np.linspace(vmin, vmax, nbins), density=True)

        width2 = (bin_edges2[1]-bin_edges2[0])

        for i in range(nbins-1):
            histogram[i].xy = (bin_edges2[i], 0)
            histogram[i].set_height(hist2[i])
            histogram[i].set_width(width2)
            histogram[i].set_facecolor(colors[i])
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(0, hist2.max()*1.01)

        cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm2, orientation='horizontal')

        image_tags['minimum_intensity'] = vmin
        image_tags['maximum_intensity'] = vmax

    def onclick(event):
        global event2
        event2 = event
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.inaxes == ax1:
            if event.button == 3:
                ind = color_map_list.index(image_tags['color_map'])+1
                if ind == len(color_map_list):
                    ind = 0
                image_tags['color_map'] = color_map_list[ind]  # 'viridis'
                vmin = image_tags['minimum_intensity']
                vmax = image_tags['maximum_intensity']
            else:
                vmax = data.max()
                vmin = data.min()
        onselect(vmin, vmax)

    fig2 = plt.figure()

    ax = fig2.add_axes([0., 0.2, 0.9, 0.7])
    ax1 = fig2.add_axes([0., 0.15, 0.9, 0.05])

    histogram = ax.bar(bin_edges[0:-1], hist, width=width, color=colors, edgecolor='black', alpha=0.8)
    onselect(vmin, vmax)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm2, orientation='horizontal')

    rectprops = dict(facecolor='blue', alpha=0.5)

    span = mwidgets.SpanSelector(ax, onselect, 'horizontal', rectprops=rectprops)

    cid = fig2.canvas.mpl_connect('button_press_event', onclick)
    return span


def clean_svd(im, pixel_size=1, source_size=5):
    """De-noising of image by using first component of single value decomposition"""
    patch_size = int(source_size/pixel_size)
    if patch_size < 3:
        patch_size = 3
    patches = image.extract_patches_2d(im, (patch_size, patch_size))
    patches = patches.reshape(patches.shape[0], patches.shape[1]*patches.shape[2])

    num_components = 32

    u, s, v = randomized_svd(patches, num_components)
    u_im_size = int(np.sqrt(u.shape[0]))
    reduced_image = u[:, 0].reshape(u_im_size, u_im_size)
    reduced_image = reduced_image/reduced_image.sum()*im.sum()
    return reduced_image


def rebin(im, binning=2):
    """
    rebin an image by the number of pixels in x and y direction given by binning

    Parameter
    ---------
    image: numpy array in 2 dimensions

    Returns
    -------
    binned image as numpy array
    """
    if len(im.shape) == 2:
        return im.reshape((im.shape[0]//binning, binning, im.shape[1]//binning, binning)).mean(axis=3).mean(1)
    else:
        raise TypeError('not a 2D image')


def cart2pol(points):
    """Cartesian to polar coordinate conversion

    Parameters
    ---------
    points: float or numpy array
        points to be converted (Nx2)

    Returns
    -------
    rho: float or numpy array
        distance
    phi: float or numpy array
        angle
    """

    rho = np.linalg.norm(points[:, 0:2], axis=1)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return rho, phi


def pol2cart(rho, phi):
    """Polar to Cartesian coordinate conversion

    Parameters
    ----------
    rho: float or numpy array
        distance
    phi: float or numpy array
        angle

    Returns
    -------
    x: float or numpy array
        x coordinates of converted points(Nx2)
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def xy2polar(points, rounding=1e-3):
    """ Conversion from carthesian to polar coordinates

    the angles and distances are sorted by r and then phi
    The indices of this sort is also returned

    Parameters
    ----------
    points: numpy array
        number of points in axis 0 first two elements in axis 1 are x and y
    rounding: int
        optional rounding in significant digits

    Returns
    -------
    r, phi, sorted_indices
    """

    r, phi = cart2pol(points)

    phi = phi-phi.min()  # only positive angles
    r = (np.floor(r/rounding))*rounding  # Remove rounding error differences

    sorted_indices = np.lexsort((phi, r))  # sort first by r and then by phi
    r = r[sorted_indices]
    phi = phi[sorted_indices]

    return r, phi, sorted_indices


def cartesian2polar(x, y, grid, r, t, order=3):
    """Transform cartesian grid to polar grid

    Used by warp
    """

    rr, tt = np.meshgrid(r, t)

    new_x = rr*np.cos(tt)
    new_y = rr*np.sin(tt)

    ix = interp1d(x, np.arange(len(x)))
    iy = interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    return ndimage.map_coordinates(grid, np.array([new_ix, new_iy]), order=order).reshape(new_x.shape)


def warp(diff, center):
    """Convert diffraction pattern to polar coordinates"""

    # Define original polar grid
    nx = diff.shape[0]
    ny = diff.shape[1]

    x = np.linspace(1, nx, nx, endpoint=True)-center[1]
    y = np.linspace(1, ny, ny, endpoint=True)-center[0]
    z = np.abs(diff)

    # Define new polar grid
    nr = min([center[0], center[1], diff.shape[0]-center[0], diff.shape[1]-center[1]])-1
    nt = 360*3

    r = np.linspace(1, nr, nr)
    t = np.linspace(0., np.pi, nt, endpoint=False)
    return cartesian2polar(x, y, z, r, t, order=3).T


def calculate_ctf(wavelength, cs, defocus, k):
    """ Calculate Contrast Transfer Function

    everything in nm

    Parameters
    ----------
    wavelength: float
        deBroglie wavelength of electrons
    cs: float
        spherical aberration coefficient
    defocus: float
        defocus
    k: numpy array
        reciprocal scale

    Returns
    -------
    ctf: numpy array
        contrast transfer function

    """
    ctf = np.sin(np.pi*defocus*wavelength*k**2+0.5*np.pi*cs*wavelength**3*k**4)
    return ctf


def calculate_scherzer(wavelength, cs):
    """
    Calculate the Scherzer defocus. Cs is in mm, lambda is in nm

    # Input and output in nm
    """

    scherzer = -1.155*(cs*wavelength)**0.5  # in m
    return scherzer


def calibrate_image_scale(fft_tags, spots_reference, spots_experiment):
    """depreciated get change of scale from comparison of spots to Bragg angles """
    gx = fft_tags['spatial_scale_x']
    gy = fft_tags['spatial_scale_y']

    dist_reference = np.linalg.norm(spots_reference, axis=1)
    distance_experiment = np.linalg.norm(spots_experiment, axis=1)

    first_reflections = abs(distance_experiment - dist_reference.min()) < .2
    print('Evaluate ', first_reflections.sum(), 'reflections')
    closest_exp_reflections = spots_experiment[first_reflections]

    def func(params, xdata, ydata):
        dgx, dgy = params
        return np.sqrt((xdata * dgx) ** 2 + (ydata * dgy) ** 2) - dist_reference.min()

    x0 = [1.001, 0.999]
    [dg, sig] = optimization.leastsq(func, x0, args=(closest_exp_reflections[:, 0], closest_exp_reflections[:, 1]))
    return dg


def align_crystal_reflections(spots, crystals):
    """ Depreciated - use diffraction spots"""

    crystal_reflections_polar = []
    angles = []
    exp_r, exp_phi = cart2pol(spots)  # just in polar coordinates
    spots_polar = np.array([exp_r, exp_phi])

    for i in range(len(crystals)):
        tags = crystals[i]
        r, phi, indices = xy2polar(tags['allowed']['g'])  # sorted by r and phi , only positive angles
        # we mask the experimental values that are found already
        angle = 0.

        angle_i = np.argmin(np.abs(exp_r - r[1]))
        angle = exp_phi[angle_i] - phi[0]
        angles.append(angle)  # Determine rotation angle

        crystal_reflections_polar.append([r, angle + phi, indices])
        tags['allowed']['g_rotated'] = pol2cart(r, angle + phi)
        for spot in tags['allowed']['g']:
            dif = np.linalg.norm(spots[:, 0:2]-spot[0:2], axis=1)
            # print(dif.min())
            if dif.min() < 1.5:
                ind = np.argmin(dif)

    return crystal_reflections_polar, angles


# Deconvolution
def decon_lr(o_image, probe,  verbose=False):
    """
    # This task generates a restored image from an input image and point spread function (PSF) using
    # the algorithm developed independently by Lucy (1974, Astron. J. 79, 745) and Richardson
    # (1972, J. Opt. Soc. Am. 62, 55) and adapted for HST imagery by Snyder
    # (1990, in Restoration of HST Images and Spectra, ST ScI Workshop Proceedings; see also
    # Snyder, Hammoud, & White, JOSA, v. 10, no. 5, May 1993, in press).
    # Additional options developed by Rick White (STScI) are also included.
    #
    # The Lucy-Richardson method can be derived from the maximum likelihood expression for data
    # with a Poisson noise distribution. Thus, it naturally applies to optical imaging data such as HST.
    # The method forces the restored image to be positive, in accord with photon-counting statistics.
    #
    # The Lucy-Richardson algorithm generates a restored image through an iterative method. The essence
    # of the iteration is as follows: the (n+1)th estimate of the restored image is given by the nth estimate
    # of the restored image multiplied by a correction image. That is,
    #
    #                            original data
    #       image    = image    ---------------  * reflect(PSF)
    #            n+1        n     image * PSF
    #                                  n

    # where the *'s represent convolution operators and reflect(PSF) is the reflection of the PSF, i.e.
    # reflect((PSF)(x,y)) = PSF(-x,-y). When the convolutions are carried out using fast Fourier transforms
    # (FFTs), one can use the fact that FFT(reflect(PSF)) = conj(FFT(PSF)), where conj is the complex conjugate
    # operator.
    """

    if len(o_image) < 1:
        return o_image

    if o_image.shape != probe.shape:
        print('Weirdness ', o_image.shape, ' != ', probe.shape)

    probe_c = np.ones(probe.shape, dtype=np.complex64)
    probe_c.real = probe

    error = np.ones(o_image.shape, dtype=np.complex64)
    est = np.ones(o_image.shape, dtype=np.complex64)
    source = np.ones(o_image.shape, dtype=np.complex64)
    source.real = o_image

    response_ft = fftpack.fft2(probe_c)

    ap_angle = o_image.metadata['experiment']['convergence_angle'] / 1000.0  # now in rad

    e0 = float(o_image.metadata['experiment']['acceleration_voltage'])

    wl = get_wavelength(e0)
    o_image.metadata['experiment']['wavelength'] = wl

    over_d = 2 * ap_angle / wl

    dx = o_image.x[1]-o_image.x[0]
    dk = 1.0 / float(o_image.x[-1])  # last value of x-axis is field of view
    screen_width = 1 / dx

    aperture = np.ones(o_image.shape, dtype=np.complex64)
    # Mask for the aperture before the Fourier transform
    n = o_image.shape[0]
    size_x = o_image.shape[0]
    size_y = o_image.shape[1]
    app_ratio = over_d / screen_width * n

    theta_x = np.array(-size_x / 2. + np.arange(size_x))
    theta_y = np.array(-size_y / 2. + np.arange(size_y))
    t_xv, t_yv = np.meshgrid(theta_x, theta_y)

    tp1 = t_xv ** 2 + t_yv ** 2 >= app_ratio ** 2
    aperture[tp1.T] = 0.
    # print(app_ratio, screen_width, dk)

    progress = tqdm(total=500)
    # de = 100
    dest = 100
    i = 0
    while abs(dest) > 0.0001:  # or abs(de)  > .025:
        i += 1
        error_old = np.sum(error.real)
        est_old = est.copy()
        error = source / np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est) * response_ft)))
        est = est * np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(error) * np.conjugate(response_ft))))
        # est = est_old * est
        # est =  np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est)*fftpack.fftshift(aperture) )))

        error_new = np.real(np.sum(np.power(error, 2))) - error_old
        dest = np.sum(np.power((est - est_old).real, 2)) / np.sum(est) * 100
        # print(np.sum((est.real - est_old.real)* (est.real - est_old.real) )/np.sum(est.real)*100 )

        if error_old != 0:
            de = error_new / error_old * 1.0
        else:
            de = error_new

        if verbose:
            print(
                ' LR Deconvolution - Iteration: {0:d} Error: {1:.2f} = change: {2:.5f}%, {3:.5f}%'.format(i, error_new,
                                                                                                          de,
                                                                                                          abs(dest)))
        if i > 500:
            dest = 0.0
            print('terminate')
        progress.update(1)
    progress.write(f"converged in {i} iterations")
    # progress.close()
    print('\n Lucy-Richardson deconvolution converged in ' + str(i) + '  iterations')
    est2 = np.real(fftpack.ifft2(fftpack.fft2(est) * fftpack.fftshift(aperture)))
    out_dataset = o_image.like_data(est2)
    out_dataset.title = 'Lucy Richardson deconvolution'
    out_dataset.data_type = 'image'
    return out_dataset
