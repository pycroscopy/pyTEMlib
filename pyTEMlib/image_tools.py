"""
image_tools.py
by Gerd Duscher, UTK
part of pyTEMlib
MIT license except where stated differently
"""

import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.widgets as mwidgets
# from matplotlib.widgets import RectangleSelector

import sidpy
import pyTEMlib.file_tools as ft
import pyTEMlib.sidpy_tools

from tqdm.auto import trange, tqdm

# import itertools
from itertools import product

from scipy import fftpack
import scipy
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
from sklearn.cluster import DBSCAN

from collections import Counter

# center diff function
from skimage.filters import threshold_otsu, sobel
from scipy.optimize import leastsq
from sklearn.cluster import DBSCAN

from ase.build import fcc110

from scipy.ndimage import rotate
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve


_SimpleITK_present = True
try:
    import SimpleITK as sitk
except ImportError:
    sitk = False
    _SimpleITK_present = False

if not _SimpleITK_present:
    print('SimpleITK not installed; Registration Functions for Image Stacks not available\n' +
          'install with: conda install -c simpleitk simpleitk ')


def get_atomic_pseudo_potential(fov, atoms, size=512, rotation=0):
    # Big assumption: the atoms are not near the edge of the unit cell
    # If any atoms are close to the edge (ex. [0,0]) then the potential will be clipped
    # before calling the function, shift the atoms to the center of the unit cell

    pixel_size = fov / size
    max_size = int(size * np.sqrt(2) + 1)  # Maximum size to accommodate rotation

    # Create unit cell potential
    positions = atoms.get_positions()[:, :2]
    atomic_numbers = atoms.get_atomic_numbers()
    unit_cell_size = atoms.cell.cellpar()[:2]

    unit_cell_potential = np.zeros((max_size, max_size))
    for pos, atomic_number in zip(positions, atomic_numbers):
        x = pos[0] / pixel_size
        y = pos[1] / pixel_size
        atom_width = 0.5  # Angstrom
        gauss_width = atom_width/pixel_size # important for images at various fov.  Room for improvement with theory
        gauss = pyTEMlib.probe_tools.make_gauss(max_size, max_size, width = gauss_width, x0=x, y0=y)
        unit_cell_potential += gauss * atomic_number  # gauss is already normalized to 1

    # Create interpolation function for unit cell potential
    x_grid = np.linspace(0, fov * max_size / size, max_size)
    y_grid = np.linspace(0, fov * max_size / size, max_size)
    interpolator = RegularGridInterpolator((x_grid, y_grid), unit_cell_potential, bounds_error=False, fill_value=0)

    # Vectorized computation of the full potential map with max_size
    x_coords, y_coords = np.meshgrid(np.linspace(0, fov, max_size), np.linspace(0, fov, max_size), indexing="ij")
    xtal_x = x_coords % unit_cell_size[0]
    xtal_y = y_coords % unit_cell_size[1]
    potential_map = interpolator((xtal_x.ravel(), xtal_y.ravel())).reshape(max_size, max_size)

    # Rotate and crop the potential map
    potential_map = rotate(potential_map, rotation, reshape=False)
    center = potential_map.shape[0] // 2
    potential_map = potential_map[center - size // 2:center + size // 2, center - size // 2:center + size // 2]

    potential_map = scipy.ndimage.gaussian_filter(potential_map,3)

    return potential_map

def convolve_probe(ab, potential):
    # the pixel sizes should be the exact same as the potential
    final_sizes = potential.shape

    # Perform FFT-based convolution
    pad_height = pad_width = potential.shape[0] // 2
    potential = np.pad(potential, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    probe, A_k, chi = pyTEMlib.probe_tools.get_probe(ab, potential.shape[0], potential.shape[1],  scale = 'mrad', verbose= False)
    

    convolved = fftconvolve(potential, probe, mode='same')

    # Crop to original potential size
    start_row = pad_height
    start_col = pad_width
    end_row = start_row + final_sizes[0]
    end_col = start_col + final_sizes[1]

    image = convolved[start_row:end_row, start_col:end_col]   

    return probe, image


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


def fourier_transform(dset: sidpy.Dataset) -> sidpy.Dataset:
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

        for i in range(dset.ndim):
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
                                                                             d=dset.x[1]-dset.x[0])),

                                              name='u', units=units_x, dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))
    fft_dset.set_dimension(1, sidpy.Dimension(np.fft.fftshift(np.fft.fftfreq(new_image.shape[1],
                                                                             d=dset.y[1]- dset.y[0])),
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


def diffractogram_spots(dset, spot_threshold, return_center=True, eps=0.1):
    """Find spots in diffractogram and sort them by distance from center

    Uses blob_log from scipy.spatial

    Parameters
    ----------
    dset: sidpy.Dataset
        diffractogram
    spot_threshold: float
        threshold for blob finder
    return_center: bool, optional
        return center of image if true
    eps: float, optional
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
    image_dims = dset.get_image_dims(return_axis=True)
    rec_scale = np.array([image_dims[0].slope, image_dims[1].slope])
    
    spots_random[:, :2] = spots_random[:, :2]*rec_scale+[dset.u.values[0], dset.v.values[0]]
    # sort reflections
    spots_random[:, 2] = np.linalg.norm(spots_random[:, 0:2], axis=1)
    spots_index = np.argsort(spots_random[:, 2])
    spots = spots_random[spots_index]
    # third row is angles
    spots[:, 2] = np.arctan2(spots[:, 0], spots[:, 1])

    center = [0, 0]

    if return_center:
        points = spots[:, 0:2]

        # Calculate the midpoints between all points
        reshaped_points = points[:, np.newaxis, :]
        midpoints = (reshaped_points + reshaped_points.transpose(1, 0, 2)) / 2.0
        midpoints = midpoints.reshape(-1, 2)

        # Find the most dense cluster of midpoints
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(midpoints)
        cluster_counter = Counter(labels)
        largest_cluster_label = max(cluster_counter, key=cluster_counter.get)
        largest_cluster_points = midpoints[labels == largest_cluster_label]

        # Average of these midpoints must be the center
        center = np.mean(largest_cluster_points, axis=0)

    return spots, center


def center_diffractogram(dset, return_plot = True, smoothing = 1, min_samples = 10, beamstop_size = 0.1):
    try:
        diff = np.array(dset).T.astype(np.float16)
        diff[diff < 0] = 0
        threshold = threshold_otsu(diff)
        binary = (diff > threshold).astype(float)
        smoothed_image = ndimage.gaussian_filter(binary, sigma=smoothing) # Smooth before edge detection
        smooth_threshold = threshold_otsu(smoothed_image)
        smooth_binary = (smoothed_image > smooth_threshold).astype(float)

        # add a circle to mask the beamstop
        x, y = np.meshgrid(np.arange(dset.shape[0]), np.arange(dset.shape[1]))
        circle = (x - dset.shape[0] / 2) ** 2 + (y - dset.shape[1] / 2) ** 2 < (beamstop_size * dset.shape[0] / 2) ** 2
        smooth_binary[circle] = 1
        
        # Find the edges using the Sobel operator
        edges = sobel(smooth_binary)
        edge_points = np.argwhere(edges)

        # Use DBSCAN to cluster the edge points
        db = DBSCAN(eps=10, min_samples=min_samples).fit(edge_points)
        labels = db.labels_
        if len(set(labels)) == 1:
            raise ValueError("DBSCAN clustering resulted in only one group, check the parameters.")

        # Get the largest group of edge points
        unique, counts = np.unique(labels, return_counts=True)
        counts = dict(zip(unique, counts))
        largest_group = max(counts, key=counts.get)
        edge_points = edge_points[labels == largest_group]

        # Fit a circle to the diffraction ring
        def calc_distance(c, x, y):
            Ri = np.sqrt((x - c[0])**2 + (y - c[1])**2)
            return Ri - Ri.mean()
        x_m = np.mean(edge_points[:, 1])
        y_m = np.mean(edge_points[:, 0])
        center_guess = x_m, y_m
        center, ier = leastsq(calc_distance, center_guess, args=(edge_points[:, 1], edge_points[:, 0]))
        mean_radius = np.mean(calc_distance(center, edge_points[:, 1], edge_points[:, 0])) + np.sqrt((edge_points[:, 1] - center[0])**2 + (edge_points[:, 0] - center[1])**2).mean()
    
    finally:
        if return_plot:
            fig, ax = plt.subplots(1, 5, figsize=(14, 4), sharex=True, sharey=True)
            ax[0].set_title('Diffractogram')
            ax[0].imshow(dset.T, cmap='viridis')
            ax[1].set_title('Otsu Binary Image')
            ax[1].imshow(binary, cmap='gray')
            ax[2].set_title('Smoothed Binary Image')
            ax[2].imshow(smoothed_image, cmap='gray')

            ax[3].set_title('Smoothed Binary Image')
            ax[3].imshow(smooth_binary, cmap='gray')
            ax[4].set_title('Edge Detection and Fitting')
            ax[4].imshow(edges, cmap='gray')
            ax[4].scatter(center[0], center[1], c='r', s=10)
            circle = plt.Circle(center, mean_radius, color='red', fill=False)
            ax[4].add_artist(circle)
            for axis in ax:
                axis.axis('off')
            fig.tight_layout()
          
    return center


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

    print(rigid_registered_dataset)
    rigid_registered_dataset.data_type = 'IMAGE_STACK'
    print('Non-Rigid_Registration')

    non_rigid_registered = demon_registration(rigid_registered_dataset)
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
    demon_registered.metadata['experiment'] = dataset.metadata['experiment'].copy()
    if 'input_crop' in dataset.metadata:
        demon_registered.metadata['input_crop'] = dataset.metadata['input_crop']
    if 'input_shape' in dataset.metadata:
        demon_registered.metadata['input_shape'] = dataset.metadata['input_shape']
    demon_registered.metadata['input_dataset'] = dataset.source
    demon_registered.data_type = 'IMAGE_STACK'
    return demon_registered


###############################
# Rigid Registration New 05/09/2020

def rigid_registration(dataset, sub_pixel=True):
    """
    Rigid registration of image stack with pixel accuracy

    Uses simple cross_correlation
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
    
    frame_dim = []
    spatial_dim = []
    selection = []
    
    for i, axis in dataset._axes.items():
        if axis.dimension_type.name == 'SPATIAL':
            spatial_dim.append(i)
            selection.append(slice(None))
        else:
            frame_dim.append(i)
            selection.append(slice(0, 1))
    
    if len(spatial_dim) != 2:
        print('need two spatial dimensions')
    if len(frame_dim) != 1:
        print('need one frame dimensions')
    
    nopix = dataset.shape[spatial_dim[0]]
    nopiy = dataset.shape[spatial_dim[1]]
    nimages = dataset.shape[frame_dim[0]]
    
    print('Stack contains ', nimages, ' images, each with', nopix, ' pixels in x-direction and ', nopiy,
          ' pixels in y-direction')
    
    fixed = dataset[tuple(selection)].squeeze().compute()
    fft_fixed = np.fft.fft2(fixed)
    
    relative_drift = [[0., 0.]]
    
    for i in trange(nimages):
        selection[frame_dim[0]] = slice(i, i+1)
        moving = dataset[tuple(selection)].squeeze().compute()
        fft_moving = np.fft.fft2(moving)
        if sub_pixel:
            shift = skimage.registration.phase_cross_correlation(fft_fixed, fft_moving, upsample_factor=1000,
                                                                 space='fourier')[0]
        else:    
            image_product = fft_fixed * fft_moving.conj()
            cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
            shift = np.array(ndimage.maximum_position(cc_image.real))-cc_image.shape[0]/2
        fft_fixed = fft_moving
        relative_drift.append(shift)
    rig_reg, drift = rig_reg_drift(dataset, relative_drift)
    crop_reg, input_crop = crop_image_stack(rig_reg, drift)
    
    rigid_registered = sidpy.Dataset.from_array(crop_reg, 
                                                title='Rigid Registration', 
                                                data_type='IMAGE_STACK',
                                                quantity=dataset.quantity,
                                                units=dataset.units)
    rigid_registered.title = 'Rigid_Registration'
    rigid_registered.source = dataset.title
    rigid_registered.metadata = {'analysis': 'rigid sub-pixel registration', 'drift': drift,
                                 'input_crop': input_crop, 'input_shape': dataset.shape[1:]}
    rigid_registered.metadata['experiment'] = dataset.metadata['experiment'].copy()
    rigid_registered.set_dimension(0, sidpy.Dimension(np.arange(rigid_registered.shape[0]), 
                                          name='frame', units='frame', quantity='time',
                                          dimension_type='temporal'))
    
    array_x = dataset._axes[spatial_dim[0]][input_crop[0]:input_crop[1]].values
    rigid_registered.set_dimension(1, sidpy.Dimension(array_x,
                                          'x', units='nm', quantity='Length',
                                          dimension_type='spatial'))
    array_y = dataset._axes[spatial_dim[1]][input_crop[2]:input_crop[3]].values
    rigid_registered.set_dimension(2, sidpy.Dimension(array_y,
                                          'y', units='nm', quantity='Length',
                                          dimension_type='spatial'))
    rigid_registered.data_type = 'IMAGE_STACK'
    return rigid_registered.rechunk({0: 'auto', 1: -1, 2: -1})


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

    frame_dim = []
    spatial_dim = []
    selection = []

    for i, axis in dset._axes.items():
        if axis.dimension_type.name == 'SPATIAL':
            spatial_dim.append(i)
            selection.append(slice(None))
        else:
            frame_dim.append(i)
            selection.append(slice(0, 1))

    if len(spatial_dim) != 2:
        print('need two spatial dimensions')
    if len(frame_dim) != 1:
        print('need one frame dimensions')

    rig_reg = np.zeros([dset.shape[frame_dim[0]], dset.shape[spatial_dim[0]], dset.shape[spatial_dim[1]]])

    # absolute drift
    drift = np.array(rel_drift).copy()
    
    drift[0] = [0, 0]
    for i in range(1, drift.shape[0]):
        drift[i] = drift[i - 1] + rel_drift[i]
    center_drift = drift[int(drift.shape[0] / 2)]
    drift = drift - center_drift
    # Shift images
    for i in range(rig_reg.shape[0]):
        selection[frame_dim[0]] = slice(i, i+1)
        # Now we shift
        rig_reg[i, :, :] = ndimage.shift(dset[tuple(selection)].squeeze().compute(),
                                         [drift[i, 0], drift[i, 1]], order=3)
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


class LineSelector(matplotlib.widgets.PolygonSelector):
    def __init__(self, ax, onselect, line_width=1, **kwargs):
        super().__init__(ax, onselect, **kwargs)
        bounds = ax.viewLim.get_points()
        np.max(bounds[0])
        self.line_verts = np.array([[np.max(bounds[1])/2, np.max(bounds[0])/5], [np.max(bounds[1])/2,
                                                                                 np.max(bounds[0])/5+1],
                                    [np.max(bounds[1])/5, np.max(bounds[0])/2], [np.max(bounds[1])/5,
                                                                                 np.max(bounds[0])/2]])
        self.verts = self.line_verts
        self.line_width = line_width

    def set_linewidth(self, line_width=None):
        if line_width is not None:
            self.line_width = line_width

        m = -(self.line_verts[0, 1]-self.line_verts[3, 1])/(self.line_verts[0, 0]-self.line_verts[3, 0])
        c = 1/np.sqrt(1+m**2)
        s = c*m
        self.line_verts[1] = [self.line_verts[0, 0]+self.line_width*s, self.line_verts[0, 1]+self.line_width*c]
        self.line_verts[2] = [self.line_verts[3, 0]+self.line_width*s, self.line_verts[3, 1]+self.line_width*c]
        
        self.verts = self.line_verts.copy()

    def onmove(self, event):
        super().onmove(event)
        if np.max(np.linalg.norm(self.line_verts-self.verts, axis=1)) > 1:
            self.moved_point = np.argmax(np.linalg.norm(self.line_verts-self.verts, axis=1))
            
            self.new_point = self.verts[self.moved_point]
            moved_point = int(np.floor(self.moved_point/2)*3)
            self.moved_point = moved_point
            self.line_verts[moved_point] = self.new_point
            self.set_linewidth()

def get_profile(dataset, line, spline_order=-1):
    """
    This function extracts a line profile from a given dataset. The line profile is a representation of the data values 
    along a specified line in the dataset. This function works for both image and spectral image data types.

    Args:
        dataset (sidpy.Dataset): The input dataset from which to extract the line profile.
        line (list): A list specifying the line along which the profile should be extracted.
        spline_order (int, optional): The order of the spline interpolation to use. Default is -1, which means no interpolation.

    Returns:
        profile_dataset (sidpy.Dataset): A new sidpy.Dataset containing the line profile.


    """
    xv, yv = get_line_selection_points(line)
    if dataset.data_type.name == 'IMAGE':
        image_dims = dataset.get_image_dims(return_axis=True)
        xv /= image_dims[0].slope
        yv /= image_dims[1].slope
        profile = scipy.ndimage.map_coordinates(np.array(dataset), [xv, yv])
        
        profile_dataset = sidpy.Dataset.from_array(profile.sum(axis=0))
        profile_dataset.data_type='spectrum'
        profile_dataset.units = dataset.units
        profile_dataset.quantity = dataset.quantity
        profile_dataset.set_dimension(0, sidpy.Dimension(np.linspace(xv[0,0], xv[-1,-1], profile_dataset.shape[0]), 
                                                  name='x', units=dataset.x.units, quantity=dataset.x.quantity,
                                                  dimension_type='spatial'))

        profile_dataset

    if dataset.data_type.name == 'SPECTRAL_IMAGE':
        spectral_axis = dataset.get_spectral_dims(return_axis=True)[0]
        if spline_order > -1:
            xv, yv, zv = get_line_selection_points_interpolated(line, z_length=dataset.shape[2])
            profile = scipy.ndimage.map_coordinates(np.array(dataset), [xv, yv, zv], order=spline_order)
            profile = profile.sum(axis=0)
            profile = np.stack([profile, profile], axis=1)
            start = xv[0, 0, 0]
        else:
            profile = get_line_profile(np.array(dataset), xv, yv, len(spectral_axis))
            start = xv[0, 0]
        print(profile.shape)
        profile_dataset = sidpy.Dataset.from_array(profile)
        profile_dataset.data_type='spectral_image'
        profile_dataset.units = dataset.units
        profile_dataset.quantity = dataset.quantity
        profile_dataset.set_dimension(0, sidpy.Dimension(np.arange(profile_dataset.shape[0])+start, 
                                                  name='x', units=dataset.x.units, quantity=dataset.x.quantity,
                                                  dimension_type='spatial'))
        profile_dataset.set_dimension(1, sidpy.Dimension([0, 1], 
                                                  name='y', units=dataset.x.units, quantity=dataset.x.quantity,
                                                  dimension_type='spatial'))
        
        profile_dataset.set_dimension(2, spectral_axis)
    return profile_dataset



def get_line_selection_points_interpolated(line, z_length=1):
    
    start_point = line.line_verts[3]
    right_point = line.line_verts[0]
    low_point = line.line_verts[2]

    if start_point[0] > right_point[0]:
        start_point = line.line_verts[0]
        right_point = line.line_verts[3]
        low_point = line.line_verts[1]
    m = (right_point[1] - start_point[1]) / (right_point[0] - start_point[0])
    length_x = int(abs(start_point[0]-right_point[0]))
    length_v = int(np.linalg.norm(start_point-right_point))
    
    linewidth = int(abs(start_point[1]-low_point[1]))
    x = np.linspace(0,length_x, length_v)
    y = np.linspace(0,linewidth, line.line_width)
    if z_length > 1:
        z = np.linspace(0, z_length, z_length)
        xv, yv, zv = np.meshgrid(x, y, np.arange(z_length))
        x = np.atleast_2d(x).repeat(z_length, axis=0).T
        y = np.atleast_2d(y).repeat(z_length, axis=0).T
    else:
        xv, yv = np.meshgrid(x, y)
    
    
    yv = yv + x*m + start_point[1] 
    xv = (xv.swapaxes(0,1) -y*m ).swapaxes(0,1) + start_point[0]

    if z_length > 1:
        return xv, yv, zv
    else:   
        return xv, yv


def get_line_selection_points(line):
    
    start_point = line.line_verts[3]
    right_point = line.line_verts[0]
    low_point = line.line_verts[2]

    if start_point[0] > right_point[0]:
        start_point = line.line_verts[0]
        right_point = line.line_verts[3]
        low_point = line.line_verts[1]
    m = (right_point[1] - start_point[1]) / (right_point[0] - start_point[0])
    length_x = int(abs(start_point[0]-right_point[0]))
    length_v = int(np.linalg.norm(start_point-right_point))
    
    linewidth = int(abs(start_point[1]-low_point[1]))
    x = np.linspace(0,length_x, length_v)
    y = np.linspace(0,linewidth, line.line_width)
    xv, yv = np.meshgrid(x, y)
    
    yy = yv +x*m+start_point[1] 
    xx = (xv.T -y*m ).T + start_point[0]
    
    return xx, yy


def get_line_profile(data, xv, yv, z_length):
    profile = np.zeros([len(xv[0]), 2, z_length])
    for index_x in range(xv.shape[1]):
        for  index_y  in range(xv.shape[0]):
            x = int(xv[index_y, index_x])
            y = int(yv[index_y, index_x])
            if x< data.shape[0] and x>0 and y < data.shape[1] and y>0:
                profile[index_x, 0] +=data[x, y]
    return profile
     

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
    patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size))
    patches = patches.reshape(patches.shape[0], patches.shape[1]*patches.shape[2])

    num_components = 32

    u, s, v = randomized_svd(patches, num_components)
    u_im_size = int(np.sqrt(u.shape[0]))
    reduced_image = u[:, 0].reshape(u_im_size, u_im_size)
    reduced_image = reduced_image/reduced_image.sum()*im.sum()
    if isinstance(im, sidpy.Dataset):
        reduced_image = im.like_data(reduced_image)
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

    phi = phi  # %np.pi # only positive angles
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
    """Takes a diffraction pattern (as a sidpy dataset)and warps it to a polar grid"""

    # Define original polar grid
    nx = np.shape(diff)[0]
    ny = np.shape(diff)[1]

    # Define center pixel
    pix2nm = np.gradient(diff.u.values)[0]

    x = np.linspace(1, nx, nx, endpoint=True)-center[0]
    y = np.linspace(1, ny, ny, endpoint=True)-center[1]
    z = diff

    # Define new polar grid
    nr = int(min([center[0], center[1], diff.shape[0]-center[0], diff.shape[1]-center[1]])-1)
    nt = 360 * 3

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


def get_rotation(experiment_spots, crystal_spots):
    """Get rotation by comparing spots in diffractogram to diffraction Bragg spots

    Parameter
    ---------
    experiment_spots: numpy array (nx2)
        positions (in 1/nm) of spots in diffractogram
    crystal_spots: numpy array (nx2)
        positions (in 1/nm) of Bragg spots according to kinematic scattering theory

    """

    r_experiment, phi_experiment = cart2pol(experiment_spots)
    
    # get crystal spots of same length and sort them by angle as well
    r_crystal, phi_crystal, crystal_indices = xy2polar(crystal_spots)
    angle_index = np.argmin(np.abs(r_experiment-r_crystal[1]))
    rotation_angle = phi_experiment[angle_index] % (2*np.pi) - phi_crystal[1]
    print(phi_experiment[angle_index])
    st = np.sin(rotation_angle)
    ct = np.cos(rotation_angle)
    rotation_matrix = np.array([[ct, -st], [st, ct]])

    return rotation_matrix, rotation_angle


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
    o_image = o_image - o_image.min()
    image_mult = o_image.max()
    o_image = o_image / o_image.max()
    source.real = o_image

    response_ft = fftpack.fft2(probe_c)

    

    ap_angle = o_image.metadata['experiment']['convergence_angle']
    if ap_angle > .1:
        ap_angle /= 1000  # now in rad

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
    while abs(dest) > 0.001:  # or abs(de)  > .025:
        i += 1
        error_old = np.sum(error.real)
        est_old = est.copy()
        error = source / np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est) * response_ft)))
        est = est * np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(error) * np.conjugate(response_ft))))
        
        error_new = np.real(np.sum(np.power(error, 2))) - error_old
        dest = np.sum(np.power((est - est_old).real, 2)) / np.sum(est) * 100
        
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
    print('\n Lucy-Richardson deconvolution converged in ' + str(i) + '  iterations')
    est2 = np.real(fftpack.ifft2(fftpack.fft2(est) * fftpack.fftshift(aperture)))*image_mult
    out_dataset = o_image.like_data(est2)
    out_dataset.title = 'Lucy Richardson deconvolution'
    out_dataset.data_type = 'image'
    return out_dataset

