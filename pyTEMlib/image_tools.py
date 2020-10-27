##################################
#
# image_tools.py
# by Gerd Duscher, UTK
# part of pyTEMlib
# MIT license except where stated differently
#
###############################
import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.patches import Polygon  # plotting of polygons -- graph rings

import matplotlib.widgets as mwidgets
from matplotlib.widgets import RectangleSelector

import pickle
import json
import struct

import sidpy
from .file_tools import *

import sys


import itertools
from itertools import product

from scipy import fftpack
from scipy import signal
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import leastsq
import scipy.optimize as optimization

# Multidimensional Image library
import scipy.ndimage as ndimage
import scipy.constants as const

import scipy.spatial as sp
from scipy.spatial import Voronoi, KDTree, cKDTree

from skimage.feature import peak_local_max
# our blob detectors from the scipy image package
from skimage.feature import blob_log  # blob_dog, blob_doh

from sklearn.feature_extraction import image
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans

_SimpleITK_present = True
try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    _SimpleITK_present = False

if not _SimpleITK_present:
    print('SimpleITK not installed; Registration Functions for Image Stacks not available')


# Wavelength in 1/nm
def get_wavelength(e0):
    """
    Calculates the relativistic corrected de Broglie wave length of an electron

    Input:
    ------
        acceleration voltage in volt
    Output:
    -------
        wave length in 1/nm
    """

    eV = const.e * e0
    return const.h/np.sqrt(2*const.m_e*eV*(1+eV/(2*const.m_e*const.c**2)))*10**9


def fourier_transform(dset):
    """
        Reads information into dictionary 'tags', performs 'FFT', and provides a smoothed FT and reciprocal
        and intensity limits for visualization.

        Input
        -----
        dset: sidp Dataset

        Usage
        -----

        fft_dataset = fourier_transform(sidpy_dataset)
        fft+dataset.plot()
    """

    assert isinstance(dset, sidpy.Dataset), 'Expected a sidpy Dataset'

    selection = []
    image_dim = []
    # image_dim = get_image_dims(sidpy.DimensionTypes.SPATIAL)
    if dset.data_type == sidpy.DataTypes.IMAGE_STACK:
        for dim, axis in dset._axes.items():
            if axis.dimension_type == sidpy.DimensionTypes.SPATIAL:
                selection.append(slice(None))
                image_dim.append(dim)
            elif axis.dimension_type == sidpy.DimensionTypes.TEMPORAL or len(dset) == 3:
                selection.append(slice(None))
                stack_dim = dim
            else:
                selection.append(slice(0, 1))
        if len(image_dim) != 2:
            raise ValueError('need at least two SPATIAL dimension for an image stack')
        image_stack = np.squeeze(np.array(dset)[selection])
        image = np.sum(np.array(image_stack), axis=stack_dim)
    elif dset.data_type.upper() == 'IMAGE':
        image = np.array(dset)
    else:
        return

    image = image - image.min()
    fft_transform = (np.fft.fftshift(np.fft.fft2(image)))

    image_dims = get_image_dims(dset)
    extent = dset.get_extent(image_dims)
    scale_x = 1 / abs(extent[1] - extent[0])
    scale_y = 1 / abs(extent[2] - extent[3])

    units_x = '1/' + dset._axes[image_dims[0]].units
    units_y = '1/' + dset._axes[image_dims[1]].units

    fft_dset = sidpy.Dataset.from_array(fft_transform)
    fft_dset.quantity = dset.quantity
    fft_dset.units = 'a.u.'
    fft_dset.data_type = 'IMAGE'
    fft_dset.source = dset.title
    fft_dset.modality = 'fft'
    fft_dset.set_dimension(0, sidpy.Dimension((np.arange(fft_dset.shape[0]) - fft_dset.shape[0] / 2) * scale_x,
                                              name='u', units=units_x, dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))
    fft_dset.set_dimension(1, sidpy.Dimension((np.arange(fft_dset.shape[1]) - fft_dset.shape[1] / 2) * scale_y,
                                              name='v', units=units_y, dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))

    return fft_dset


def power_spectrum(dset, smoothing=3):
    """
    Calculate power spectrum

    Input:
    ======
            channel: channnel in h5f file with image content

    Output:
    =======
            tags: dictionary with
                ['data']: fourier transformed image
                ['axis']: scale of reciprocal image
                ['power_spectrum']: power_spectrum
                ['FOV']: field of view for extent parameter in plotting
                ['minimum_intensity']: suggested minimum intensity for plotting
                ['maximum_intensity']: suggested maximum intensity for plotting

    """
    fft_transform = fourier_transform(dset)
    fft_mag = np.abs(fft_transform)
    fft_mag2 = ndimage.gaussian_filter(fft_mag, sigma=(smoothing, smoothing), order=0)

    power_spec = fft_transform.like_data(np.log(1.+fft_mag2))

    # prepare mask

    x, y = np.meshgrid(power_spec.u.values, power_spec.v.values)
    mask = np.zeros(power_spec.shape)

    mask_spot = x ** 2 + y ** 2 > 1 ** 2
    mask = mask + mask_spot
    mask_spot = x ** 2 + y ** 2 < 11 ** 2
    mask = mask + mask_spot

    mask[np.where(mask == 1)] = 0  # just in case of overlapping disks

    minimum_intensity = np.log2(1 + fft_mag2)[np.where(mask == 2)].min() * 0.95
    maximum_intensity = np.log2(1 + fft_mag2)[np.where(mask == 2)].max() * 1.05
    power_spec.metadata = {'smoothing': smoothing,
                           'minimum_intensity': minimum_intensity, 'maximum_intensity': maximum_intensity}
    power_spec.title = 'power spectrum ' + power_spec.source

    return power_spec


def diffractogram_spots(dset, spot_threshold):
    """
    Find spots in diffractogram and sort them by distance from center

    Input:
    ======
            fft_tags: dictionary with
                ['spatial_***']: information of scale of fourier pattern
                ['data']: power_spectrum
            spot_threshold: threshold for blob finder
    Output:
    =======
            spots: numpy array with sorted position (x,y) and radius (r) of all spots

    """
    # Needed for conversion from pixel to Reciprocal space
    # we'll have to switch x- and y-coordinates due to the differences in numpy and matrix
    center = np.array([int(dset.shape[0]/2.), int(dset.shape[1]/2.), 1])
    rec_scale = np.array([get_slope(dset.u.values), get_slope(dset.v.values), 1])

    # spot detection ( for future referece there is no symmetry assumed here)
    data = np.array(dset).T
    data = (data - data.min()) / data.max()
    spots_random = (blob_log(data, max_sigma=5, threshold=spot_threshold) - center) * rec_scale
    print(f'Found {spots_random.shape[0]} reflections')

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

    Input:
    ======
            image:  image to be filtered
            tags: dictionary with
                ['spatial_***']: information of scale of fourier pattern
                ['spots']: sorted spots in diffractogram in 1/nm
            low_pass: low pass filter in center of diffractogrm

    Output:
    =======
            Fourier filtered image
    """
    # prepare mask

    fft_transform = fourier_transform(dset)
    x, y = np.meshgrid(fft_transform.u.values, fft_transform.v.values)
    mask = np.zeros(dset.shape)

    # mask reflections
    # reflection_radius = 0.3 # in 1/nm
    for spot in spots:
        mask_spot = (x - spot[0]) ** 2 + (y - spot[1]) ** 2 < reflection_radius ** 2  # make a spot
        mask = mask + mask_spot  # add spot to mask

    # mask zero region larger (low-pass filter = intensity variations)
    # low_pass = 3 # in 1/nm
    mask_spot = x ** 2 + y ** 2 < low_pass ** 2
    mask = mask + mask_spot
    mask[np.where(mask > 1)] = 1
    fft_filtered = fft_transform * mask

    filtered_image = dset.like_data(np.fft.ifft2(np.fft.fftshift(fft_filtered)).real)
    filtered_image.title = 'Fourier filtered ' + dset.title
    filtered_image.source = dset.title
    filtered_image.metadata = {'analysis': 'adaptive fourier filtered', 'spots': spots,
                               'low_pass': low_pass, 'reflection_radius': reflection_radius}

    return filtered_image


def rotational_symmetry_diffractogram(spots):
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


def complete_registration(main_dataset):
    rigid_registered_dataset = rigid_registration(main_dataset)

    registration_channel = log_results(current_channel, rig_reg_dataset)

    print('Non-Rigid_Registration')

    non_rigid_registered = demon_registration(rigid_registered_dataset)

    registration_channel = log_results(current_channel, non_rigid_registered)

    return registration_channel


def demon_registration(dataset, verbose=False):
    """
    Diffeomorphic Demon Non-Rigid Registration
    Usage:
    ------
    dem_reg = demon_reg(cube, verbose = False)

    Input:
        cube: stack of image after rigid registration and cropping
    Output:
        dem_reg: stack of images with non-rigid registration

    Depends on:
        simpleITK and numpy

    Please Cite: http://www.simpleitk.org/SimpleITK/project/parti.html
    and T. Vercauteren, X. Pennec, A. Perchant and N. Ayache
    Diffeomorphic Demons Using ITK\'s Finite Difference Solver Hierarchy
    The Insight Journal, http://hdl.handle.net/1926/510 2007
    """

    dem_reg = np.zeros(dataset.shape)
    nimages = dataset.shape[0]
    if verbose:
        print(nimages)
    # create fixed image by summing over rigid registration

    fixed_np = np.average(np.array(dataset), axis=0)

    fixed = sITK.GetImageFromArray(fixed_np)
    fixed = sITK.DiscreteGaussian(fixed, 2.0)

    # demons = sITK.SymmetricForcesDemonsRegistrationFilter()
    demons = sITK.DiffeomorphicDemonsRegistrationFilter()

    demons.SetNumberOfIterations(200)
    demons.SetStandardDeviations(1.0)

    resampler = sITK.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sITK.sitkBSpline)
    resampler.SetDefaultPixelValue(0)

    done = 0

    if QT_available:
        progress = QtWidgets.QProgressDialog("Non-Rigid Registration.", "Abort", 0, nimages)
        progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # progress.setWindowModality(Qt.WindowModal);
        progress.show()

    for i in range(nimages):

        if QT_available:
            progress.setValue(i)
            Qt.QApplication.processEvents()
        else:
            if done < int((i + 1) / nimages * 50):
                done = int((i + 1) / nimages * 50)
                sys.stdout.write('\r')
                # progress output :
                sys.stdout.write("[%-50s] %d%%" % ('*' * done, 2 * done))
                sys.stdout.flush()

        moving = sITK.GetImageFromArray(cube[i])
        moving_f = sITK.DiscreteGaussian(moving, 2.0)
        displacement_field = demons.Execute(fixed, moving_f)
        out_tx = sITK.DisplacementFieldTransform(displacement_field)
        resampler.SetTransform(out_tx)
        out = resampler.Execute(moving)
        dem_reg[i, :, :] = sITK.GetArrayFromImage(out)
        # print('image ', i)

    if QT_available:
        progress.setValue(nimages)

    print(':-)')
    print('You have successfully completed Diffeomorphic Demons Registration')

    demon_registered = dataset.like_data(dem_reg)
    demon_registered.title = 'Non-Rigid Registration'
    demon_registered.source = dataset.title

    demon_registered.metadata = {'analysis': 'non-rigid demon registration'}
    if 'boundaries' in dataset.metadata:
        demon_registered.metadata['boundaries'] = dataset.metadata['boundaries']

    return demon_registered


###############################
# Rigid Registration New 05/09/2020

def rigid_registration(dataset):
    """
    Rigid registration of image stack with sub-pixel accuracy
    used phase_cross_correlation from skimage.registration
    (we determine drift from one image to next)

    Input hdf5 group with image_stack dataset

    Output Registered Stack and drift (with respect to center image)

    """

    nopix = dataset.shape[1]
    nopiy = dataset.shape[2]
    nimages = dataset.shape[0]

    print('Stack contains ', nimages, ' images, each with', nopix, ' pixels in x-direction and ', nopiy,
          ' pixels in y-direction')
    fixed = np.array(dataset[0])
    fft_fixed = np.fft.fft2(fixed)

    relative_drift = [[0., 0.]]
    done = 0

    if QT_available:
        progress = QtWidgets.QProgressDialog("Rigid Registration.", "Abort", 0, nimages)
        progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # progress.setWindowModality(Qt.WindowModal);
        progress.show()

    for i in range(nimages):
        if QT_available:
            progress.setValue(i)
            Qt.QApplication.processEvents()
        else:
            if done < int((i + 1) / nimages * 50):
                done = int((i + 1) / nimages * 50)
                sys.stdout.write('\r')
                # progress output :
                sys.stdout.write("[%-50s] %d%%" % ('*' * done, 2 * done))
                sys.stdout.flush()

        moving = np.array(dataset[i])
        fft_moving = np.fft.fft2(moving)
        if skimage.__version__[:4] == '0.16':
            shift = register_translation(fft_fixed, fft_moving, upsample_factor=1000, space='fourier')
        else:
            shift = registration.phase_cross_correlation(fft_fixed, fft_moving, upsample_factor=1000, space='fourier')

        fft_fixed = fft_moving
        # print(f'Image number {i:2}  xshift =  {shift[0][0]:6.3f}  y-shift =  {shift[0][1]:6.3f}')

        relative_drift.append(shift[0])
    if QT_available:
        progress.setValue(nimages)
    rig_reg, drift = rig_reg_drift(dataset, relative_drift)

    crop_reg, boundaries = crop_image_stack(rig_reg, drift)

    rigid_registered = dataset.like_data(crop_reg)
    rigid_registered.title = 'Rigid Registration'
    rigid_registered.source = dataset.title
    rigid_registered.metadata = {'analysis': 'rigid sub-pixel registration', 'drift': drift, 'boundaries': boundaries}

    return rigid_registered


def rig_reg_drift(dset, rel_drift):
    """
    Uses relative drift to shift images ontop of each other
    Shifting is done with shift routine of ndimage from scipy

    is used by Rigid_Registration routine

    Input image_channel with image_stack numpy array
    relative_drift from image to image as list of [shiftx, shifty]

    output stack and drift
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
    """
    ## Crop images
    """
    xpmin = int(-np.floor(np.min(np.array(drift)[:, 0])))
    xpmax = int(rig_reg.shape[1] - np.ceil(np.max(np.array(drift)[:, 0])))
    ypmin = int(-np.floor(np.min(np.array(drift)[:, 1])))
    ypmax = int(rig_reg.shape[2] - np.ceil(np.max(np.array(drift)[:, 1])))

    return rig_reg[:, xpmin:xpmax, ypmin:ypmax], [xpmin, xpmax, ypmin, ypmax]


class ImageWithLineProfile:
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
    nbins = 75
    minbin = 0.
    maxbin = 1.
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


def find_atoms(im, tags, verbose=False):
    
    if 'rel_blob_size' not in tags:
        tags['rel_blob_size'] = .4  # between 0 and 1 nromally around 0.5
        tags['source_size'] = 0.06  # in nm gives the size of the atoms or resolution
        tags['nearest_neighbours'] = 7  # up to this number nearest neighbours are evaluated (normally 7)
        tags['threshold'] = 0.15  # between 0.01 and 0.1
        tags['rim_size'] = 2  # size of rim in multiples of source size
        
    rel_blob_size = tags['rel_blob_size']  # between 0 and 1 nromaly around 0.5
    source_size = tags['source_size']  # in nm gives the size of the atoms
    nearest_neighbours = tags['nearest_neighbours']  # up to this number nearest neighbours are evaluated (normally 7)
    threshold = tags['threshold']  # between 0.01 and 0.1 
    rim_size = tags['rim_size']  # sizeof rim in multiples of resolution
    pixel_size = tags['pixel_size']
                      
    rim_width = rim_size*source_size/pixel_size
    
    # Get a noise free image: reduced
    # pixel_size = FOV/im.shape[0]
    reduced_image = clean_svd(im, pixel_size=pixel_size, source_size=source_size)

    reduced_image = reduced_image-reduced_image.min()
    reduced_image = reduced_image/reduced_image.max()

    tags['reduced_image'] = reduced_image
    patch_size = im.shape[0]-reduced_image.shape[0]
    tags['patch_size'] = patch_size
    print(f' Use {patch_size} x {patch_size} pixels for image-patch of atoms')

    # Find atoms    
    thresh = reduced_image.std()*threshold
    blobs = blob_log(np.array(reduced_image), max_sigma=source_size/pixel_size, threshold=thresh)
    
    atoms = []
    
    for blob in blobs:
        y, x, r = blob
        if r > patch_size*rel_blob_size:
            atoms.append([x+patch_size/2, y+patch_size/2, r])

    atoms = np.array(atoms)
    # Determine Rim atoms
    rim_left = np.where(atoms[:, 0] < rim_width)[0].flatten()
    rim_right = np.where(atoms[:, 0] > im.shape[0]-rim_width)[0].flatten()
    rim_up = np.where(atoms[:, 1] < rim_width)[0].flatten()
    rim_down = np.where(atoms[:, 1] > im.shape[1]-rim_width)[0].flatten()

    rim_indices = np.concatenate((rim_left, rim_right, rim_up, rim_down))

    rim_atoms = list(np.unique(rim_indices))
    mid_atoms_list = np.setdiff1d(np.arange(len(atoms)), rim_atoms)
    
    mid_atoms = np.array(atoms)[mid_atoms_list]
    if verbose:
        print(f'Evaluated {len(mid_atoms)} atom positions, out of {len(atoms)} atoms')
    tags['atoms'] = atoms
    tags['mid_atoms'] = mid_atoms
    tags['rim_atoms'] = rim_atoms
    tags['number_of_atoms'] = len(atoms)
    tags['number_of_evaluated_atoms'] = len(mid_atoms)
    return tags


def atoms_clustering(atoms, mid_atoms, number_of_clusters=3, nearest_neighbours=7):
    # get distances
    nn_tree = cKDTree(np.array(atoms)[:, 0:2])

    distances, indices = nn_tree.query(np.array(mid_atoms)[:, 0:2], nearest_neighbours)

    # CLustering
    k_means = KMeans(n_clusters=number_of_clusters, random_state=0)  # Fixing the RNG in kmeans
    k_means.fit(distances)
    clusters = k_means.predict(distances)
    return clusters, distances, indices


def voronoi(atoms, tags):
    im = tags['image']
    vor = Voronoi(np.array(atoms)[:, 0:2])  # Plot it:
    rim_vertices = []
    for i in range(len(vor.vertices)):
        if (vor.vertices[i, 0:2] < 0).any() or (vor.vertices[i, 0:2] > im.shape[0]-5).any():
            rim_vertices.append(i)
    rim_vertices = set(rim_vertices)
    mid_vertices = list(set(np.arange(len(vor.vertices))).difference(rim_vertices))

    mid_regions = []
    for region in vor.regions:  # Check all Voronoi polygons
        if all(x in mid_vertices for x in region) and len(region) > 1:  # we get a lot of rim (-1) and empty and regions
            mid_regions.append(region)
    tags['atoms']['voronoi'] = vor
    tags['atoms']['voronoi_vertices'] = vor.vertices
    tags['atoms']['voronoi_regions'] = vor.regions
    tags['atoms']['voronoi_midVerticesIndices'] = mid_vertices
    tags['atoms']['voronoi_midVertices'] = vor.vertices[mid_vertices]
    tags['atoms']['voronoi_midRegions'] = mid_regions


def clean_svd(im, pixel_size=1, source_size=5):
    patch_size = int(source_size/pixel_size)
    if patch_size < 3:
        patch_size = 3
    print(patch_size)
    
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
    
    Input:
    ======
            image: numpy array in 2 dimensions
    
    Output:
    =======
            binned image 
    """
    if len(im.shape) == 2:
        return im.reshape((im.shape[0]//binning, binning, im.shape[1]//binning, binning)).mean(axis=3).mean(1)
    else:
        print('not a 2D image')
        return im


def cart2pol(points):
    rho = np.linalg.norm(points[:, 0:2], axis=1)
    phi = np.arctan2(points[:, 1], points[:, 0])
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def xy2polar(points, rounding=1e-3):
    """
    Conversion from carthesian to polar coordinates
    
    the angles and distances are sorted by r and then phi
    The indices of this sort is also returned
    
    input points: numpy array with number of points in axis 0 first two elements in axis 1 are x and y
    
    optional rounding in significant digits 
    
    returns r,phi, sorted_indices
    """
    
    r, phi = cart2pol(points)
    
    phi = phi-phi.min()  # only positive angles
    r = (np.floor(r/rounding))*rounding  # Remove rounding error differences

    sorted_indices = np.lexsort((phi, r))  # sort first by r and then by phi
    r = r[sorted_indices]
    phi = phi[sorted_indices]
    
    return r, phi, sorted_indices
            

def cartesian2polar(x, y, grid, r, t, order=3):

    rr, tt = np.meshgrid(r, t)

    new_x = rr*np.cos(tt)
    new_y = rr*np.sin(tt)

    ix = interp1d(x, np.arange(len(x)))
    iy = interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    return ndimage.map_coordinates(grid, np.array([new_ix, new_iy]), order=order).reshape(new_x.shape)


def warp(diff, center):
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
    """
    ctf = np.sin(np.pi*defocus*wavelength*k**2+0.5*np.pi*cs*wavelength**3*k**4)
    return ctf


def calculate_scherzer(wavelength, cs):
    """
    Calculate the Scherzer defocus. Cs is in mm, lambda is in nm
    # EInput and output in nm
    """
    scherzer = -1.155*(cs*wavelength)**0.5  # in m
    return scherzer


def calibrate_image_scale(fft_tags, spots_reference, spots_experiment):
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
    dg, sig = optimization.leastsq(func, x0, args=(closest_exp_reflections[:, 0], closest_exp_reflections[:, 1]))
    return dg


def align_crystal_reflections(spots, crystals):
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


def plot_image(tags):
    if 'axis' in tags:
        pixel_size = tags['axis']['0']['scale']
        units = tags['axis']['0']['units']
    elif 'pixel_size' not in tags:
        pixel_size = 1
        units = 'px'
    else:
        pixel_size = tags['pixel_size']
        units = 'nm'
    
        
    image = tags['data'].T
    FOV = pixel_size*image.shape[0]
    plt.imshow(image, cmap='gray', extent=(0, FOV, 0, FOV))
    if 'basename' in tags:
        plt.title(tags['basename'])

    plt.show()



def makechi1( phi, theta,wl,ab, C1include)  :
    """
    ###
    # Aberration function chi without defocus
    ###
    """
    t0 = np.power(theta,1)/1*(  float(ab['C01a']) * np.cos(1*phi)
                                + float(ab['C01b']) * np.sin(1*phi))

    if C1include == 1: #First and second terms
        t1 = np.power(theta,2)/2*(  ab['C10']
                                    + ab['C12a'] *np.cos(2*phi) 
                                    + ab['C12b'] *np.sin(2*phi))
    elif C1include == 2:#Second terms only
        t1 = np.power(theta,2)/2*(  ab['C12a'] *np.cos(2*phi) 
                                    + ab['C12b'] *np.sin(2*phi))
    else: # none for zero
        t1 = t0*0.
    t2 = np.power(theta,3)/3*(  ab['C21a'] * np.cos(1*phi)
                                + ab['C21b'] * np.sin(1*phi)
                                + ab['C23a'] * np.cos(3*phi)
                                + ab['C23b'] * np.sin(3*phi) )

    t3 = np.power(theta,4)/4*(  ab['C30']
                                + ab['C32a'] * np.cos(2*(phi))
                                + ab['C32b'] * np.sin(2*(phi))
                                + ab['C34a'] * np.cos(4*(phi))
                                + ab['C34b'] * np.sin(4*(phi)) )

    t4 = np.power(theta,5)/5*(  ab['C41a'] * np.cos(1*phi)
                                + ab['C41b'] * np.sin(1*phi)
                                + ab['C43a'] * np.cos(3*phi)
                                + ab['C43b'] * np.sin(3*phi)
                                + ab['C45a'] * np.cos(5*phi)
                                + ab['C45b'] * np.sin(5*phi) )

    t5 = np.power(theta,6)/6*(  ab['C50']
                                + ab['C52a'] * np.cos(2*phi)
                                + ab['C52b'] * np.sin(2*phi)
                                + ab['C54a'] * np.cos(4*phi)
                                + ab['C54b'] * np.sin(4*phi)
                                + ab['C56a'] * np.cos(6*phi)
                                + ab['C56b'] * np.sin(6*phi) )



    chi = t0 + t1+t2+t3+t4+t5
    if 'C70' in ab:
        chi += np.power(theta,8)/8*(   ab['C70'])
    return chi*2*np.pi/wl #np.power(theta,6)/6*(  ab['C50'] )



def Probe2( ab, sizeX, sizeY, tags, verbose= False):     
    """
    **********************************************
    * This function creates a incident STEM probe 
    * at position (0,0)
    * with parameters given in ab dictionary
    *
    * The following Abberation functions are being used:
    * 1) ddf = Cc*dE/E  but not  + Cc2*(dE/E)^2,    
    *    Cc, Cc2 = chrom. Abber. (1st, 2nd order) [1]
    * 2) chi(qx,qy) = (2*pi/lambda)*{0.5*C1*(qx^2+qy^2)+
    *                 0.5*C12a*(qx^2-qy^2)+
    *                 C12b*qx*qy+
    *                 C21a/3*qx*(qx^2+qy^2)+
    *                 ... 
    *                 +0.5*C3*(qx^2+qy^2)^2
    *                 +0.125*C5*(qx^2+qy^2)^3
    *                 ... (need to finish)
    *
    *
    *    qx = acos(kx/K), qy = acos(ky/K) 
    *
    * References:
    * [1] J. Zach, M. Haider, 
    *    "Correction of spherical and Chromatic Abberation 
    *     in a low Voltage SEM", Optik 98 (3), 112-118 (1995)
    * [2] O.L. Krivanek, N. Delby, A.R. Lupini,
    *    "Towards sub-Angstroem Electron Beams", 
    *    Ultramicroscopy 78, 1-11 (1999)
    *
    *********************************************'''
    ####
    # Internally reciprocal lattice vectors in 1/nm or rad.
    # All calculations of chi in angles.
    # All aberration coefficients in nm
    """  

    if 'FOV' not in ab:
        if 'FOV' not in tags:
            print(' need field of view in tags ' )
        else:
            ab['FOV'] = tags['FOV']

    if 'convAngle' not in ab:
        ab['convAngle'] = 30 # in mrad

    ApAngle=ab['convAngle']/1000.0 # in rad

    E0= ab['EHT'] = float( ab['EHT'])  # acceleration voltage in eV

    defocus = ab['C10'] 


    if 'C01a' not in ab:
        ab['C01a'] = 0.
    if 'C01b' not in ab:
        ab['C01b'] = 0.

    if 'C50' not in ab:
        ab['C50'] = 0.
    if 'C70' not in ab:
        ab['C70'] = 0.

    if 'Cc' not in ab:
        ab['Cc'] = 1.3e6            #// Cc in  nm


    def get_wl():
        h=6.626*10**-34
        m0=9.109*10**-31
        eV=1.602*10**-19*E0
        C=2.998*10**8
        return h/np.sqrt(2*m0*eV*(1+eV/(2*m0*C**2)))*10**9

    wl=get_wl()
    if verbose:
        print('Acceleration voltage {0:}kV => wavelength {1:.2f}pm'.format(int(E0/1000),wl*1000) )
    ab['wavelength'] = wl


    ## Reciprocal plane in 1/nm
    dk = 1/ab['FOV']
    kx = np.array(dk*(-sizeX/2.+ np.arange(sizeX)))
    ky = np.array(dk*(-sizeY/2.+ np.arange(sizeY)))
    Txv, Tyv = np.meshgrid(kx, ky)

    # define reciprocal plane in angles
    phi =  np.arctan2(Txv, Tyv)
    theta = np.arctan2(np.sqrt(Txv**2 + Tyv**2),1/wl)

    ## calculate chi but omit defocus
    chi = np.fft.ifftshift (makechi1(phi,theta,wl,ab, 2))
    probe = np.zeros((sizeX, sizeY))


    ## Aperture function 
    mask = theta >= ApAngle

    ## Calculate probe with Cc

    for i in range(len(ab['zeroLoss'])):
        df = ab['C10'] + ab['Cc']* ab['zeroEnergy'][i]/E0
        if verbose:
            print('defocus due to Cc: {0:.2f} nm with weight {1:.2f}'.format(df,ab['zeroLoss'][i]))
        # Add defocus
        chi2 = chi + np.power(theta,2)/2*(df)
        #Calculate exponent of - i * chi
        chiT = np.fft.ifftshift (np.vectorize(complex)(np.cos(chi2), -np.sin(chi2)) )
        ## Aply aperture function
        chiT[mask] = 0.
        ## inverse fft of aberration function
        i2  = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift (chiT)))
        ## add intensities
        probe = probe + np.real(i2 * np.conjugate(i2)).T*ab['zeroLoss'][i]

    ab0={}
    for key in ab:
        ab0[key] = 0.
    chiIA = np.fft.fftshift (makechi1(phi,theta,wl,ab0, 0))#np.ones(chi2.shape)*2*np.pi/wl
    chiI = np.ones((sizeY, sizeX))
    chiI[mask]=0.
    i2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift (chiI)))
    ideal = np.real(i2 * np.conjugate(i2))

    probeF = np.fft.fft2(probe,probe.shape)+1e-12
    idealF = np.fft.fft2(ideal,probe.shape)
    fourier_space_division = idealF/probeF
    probeR = (np.fft.ifft2(fourier_space_division,probe.shape))


    return probe/sum(ab['zeroLoss']), np.real(probeR)


    


def DeconLR(  Oimage, probe, tags, verbose = False):
    """
    
    
    # This task generates a restored image from an input image and point spread function (PSF) using the algorithm developed independently by Lucy (1974, Astron. J. 79, 745) and Richardson (1972, J. Opt. Soc. Am. 62, 55) and adapted for HST imagery by Snyder (1990, in Restoration of HST Images and Spectra, ST ScI Workshop Proceedings; see also Snyder, Hammoud, & White, JOSA, v. 10, no. 5, May 1993, in press). Additional options developed by Rick White (STScI) are also included.
    #
    # The Lucy-Richardson method can be derived from the maximum likelihood expression for data with a Poisson noise distribution. Thus, it naturally applies to optical imaging data such as HST. The method forces the restored image to be positive, in accord with photon-counting statistics.
    #
    # The Lucy-Richardson algorithm generates a restored image through an iterative method. The essence of the iteration is as follows: the (n+1)th estimate of the restored image is given by the nth estimate of the restored image multiplied by a correction image. That is,
    #
    #                            original data
    #       image    = image    ---------------  * reflect(PSF) 
    #            n+1        n     image * PSF
    #                                  n

    # where the *'s represent convolution operators and reflect(PSF) is the reflection of the PSF, i.e. reflect((PSF)(x,y)) = PSF(-x,-y). When the convolutions are carried out using fast Fourier transforms (FFTs), one can use the fact that FFT(reflect(PSF)) = conj(FFT(PSF)), where conj is the complex conjugate operator. 
    """
    
    if len(Oimage) < 1:
        return Oimage
    
    if Oimage.shape != probe.shape:
        print('Weirdness ',Oimage.shape,' != ',probe.shape)

    probeC = np.ones((probe.shape), dtype = np.complex64)
    probeC.real = probe


    error = np.ones((Oimage.shape), dtype = np.complex64)
    est = np.ones((Oimage.shape), dtype = np.complex64)
    source= np.ones((Oimage.shape), dtype = np.complex64)
    source.real = Oimage
    
    responseFT =fftpack.fft2(probeC)



    
    if 'ImageScanned' in tags:
        ab = tags['ImageScanned']
    elif 'aberrations' in  tags:
        ab = tags['aberrations']
    if 'convAngle' not in ab:
        ab['convAngle'] = 30
    ApAngle=ab['convAngle']/1000.0
    
    E0=  float( ab['EHT'])  

    def get_wl(E0):
        h=6.626*10**-34
        m0=9.109*10**-31
        eV=1.602*10**-19*E0
        C=2.998*10**8
        return h/np.sqrt(2*m0*eV*(1+eV/(2*m0*C**2)))*10**9

    wl=get_wl(E0)
    ab['wavelength'] = wl
    
    over_d = 2* ApAngle / wl

    dx = tags['pixel_size']
    dk = 1.0/ float(tags['FOV'])
    ScreenWidth = 1/dx

    
    aperture = np.ones((Oimage.shape), dtype = np.complex64)
    # Mask for the aperture before the Fourier transform
    N = Oimage.shape[0]
    sizeX = Oimage.shape[0]
    sizeY = Oimage.shape[1]
    App_ratio = over_d/ScreenWidth*N

    
    Thetax = np.array((-sizeX/2.+ np.arange(sizeX)))
    Thetay = np.array((-sizeY/2.+ np.arange(sizeY)))
    Txv, Tyv = np.meshgrid(Thetax, Thetay)

    tp1 = Txv**2 + Tyv**2 >= (App_ratio)**2
    aperture[tp1.T] = 0.
    print( App_ratio, ScreenWidth, dk)

    
    

    
    dE = 100
    dest = 100
    i=0
    while abs(dest) > 0.0001 :#or abs(dE)  > .025:
        i += 1

        error_old = np.sum(error.real)
        est_old = est.copy()
        error = source / np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est)*responseFT)))
        est = est * np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(error)*np.conjugate(responseFT))))
        #est = est_old * est
        #est =  np.real(fftpack.fftshift(fftpack.ifft2(fftpack.fft2(est)*fftpack.fftshift(aperture) )))
    
        error_new = np.real(np.sum(np.power(error,2)))-error_old
        dest = np.sum(np.power((est - est_old).real,2))/np.sum(est)*100
        #print(np.sum((est.real - est_old.real)* (est.real - est_old.real) )/np.sum(est.real)*100 )

        if error_old!=0:
            dE = error_new / error_old *1.0
            
        else:
            dE = error_new
    
        if verbose:
            print(' LR Deconvolution - Iteration: {0:d} Error: {1:.2f} = change: {2:.5f}%, {3:.5f}%'.format(i,error_new,dE,abs(dest)))
    
        if i > 1000:
            dE = dest =  0.0
            print('terminate')
    
    print('\n Lucy-Richardson deconvolution converged in '+str(i)+ '  Iterations')
    est2 =  np.real(fftpack.ifft2(fftpack.fft2(est)*fftpack.fftshift(aperture) ))
    #plt.imshow(np.real(np.log10(np.abs(fftpack.fftshift(fftpack.fft2(est)))+1)+aperture), origin='lower',)
    #plt.show()
    print(est2.shape)
    return est2


##########################################
# Functions Used
##########################################
 
def MakeProbeG(sizeX,sizeY,widthi,xi,yi):
    sizeX = (sizeX/2)
    sizeY = (sizeY/2)
    width = 2*widthi**2
    x, y = np.mgrid[-sizeX:sizeX, -sizeY:sizeY]
    g = np.exp(-((x-xi)**2/float(width)+(y-yi)**2/float(width)))
    probe = g/g.sum()
        
    return probe

def MakeLorentz(sizeX,sizeY,width,xi,yi):
    sizeX = np.floor(sizeX/2)
    sizeY = np.floor(sizeY/2)
    gamma = width
    x, y = np.mgrid[-sizeX:sizeX, -sizeY:sizeY]
    g = gamma/(2*np.pi)/ np.power( ((x-xi)**2+(y-yi)**2+gamma**2),1.5)
    probe = g/g.sum()
        
    return probe



def ZLPWeight():
    x = np.linspace(-0.5,.9, 29)
    y = [0.0143,0.0193,0.0281,0.0440,0.0768,0.1447,0.2785,0.4955,0.7442,0.9380,1.0000,0.9483,0.8596,0.7620,0.6539,0.5515,0.4478,0.3500,0.2683,0.1979,0.1410,0.1021,0.0752,0.0545,0.0401,0.0300,0.0229,0.0176,0.0139]
    
    return (x,y)


    ##
    # All atom detection is done here
    # Everything is in unit of pixel!!
    ##


def findatoms(image, tags):
    """
    ######################################
    # Find atoms 
    ######################################
    """
    
    image = image-image.min()
    image = image/image.max()

    if 'sigma_min' not in tags:
        tags['sigma_min'] = 0.1
    if 'resolution' not in tags:
        tags['resolution'] = 0.1

    if 'ROIsize' not in tags:
        tags['ROIsize'] = 100.


    res = tags['resolution'] / tags['pixel_size']#* tags['ROIsize']/100.
    print('res',res)
    coordinates = peak_local_max(image, min_distance=int(res/2), threshold_rel=tags['sigma_min'], exclude_border =True)
    print('coor',len( coordinates))
    """
       peak_local_max(image, min_distance=10, threshold_abs=0, threshold_rel=0.1,
                   exclude_border=True, indices=True, num_peaks=np.inf,
                   footprint=None, labels=None):
    
        Find peaks in an image, and return them as coordinates or a boolean array.
        Peaks are the local maxima in a region of `2 * min_distance + 1
        
        (i.e. peaks are separated by at least `min_distance`).
        NOTE: If peaks are flat (i.e. multiple adjacent pixels have identical
        intensities), the coordinates of all such pixels are returned.
        """
        



    
    # We calculate the radius in pixel of a round area in which atoms are evaluated
    sc = tags['pixel_size']
    r= tags['resolution']/sc*tags['ROIsize']/100./2.
    tags['radius'] = r

    #######################################
    # Now we determine intensity #
    #######################################

    ###
    # Make a circular mask for integration of atom intensity
    ###
    rr = int(r+0.5)
    mask = np.zeros((2*rr+1,2*rr+1))
    
    for i in range (2*rr+1):
        for j  in range (2*rr+1):
            if (i-rr)**2+(j-rr)**2<rr**2+0.1:
                mask[i,j]=1

                
                
        
                    
    ###
    # Determine  pixel position and intensity  of all atoms 
    ###
    atoms = []
    for i in range(len( coordinates)):
        x,y = coordinates[i]
        
        if x>rr and y>rr and x<image.shape[1]-rr and y<image.shape[0]-rr:
            
            area = image[x-rr:x+rr+1,y-rr:y+rr+1]
            arr = area*mask
            atoms.append((x,y,rr, arr.sum(), arr.max()))

    print(' Detected ', len(atoms), ' atoms')
    atoms.sort()
    return atoms

# sort corners in counter-clockwise direction
def TurningFunction(corners,points):
    # calculate centroid of the polygon
    corners1 = np.array(points[corners])
    corners2 = np.roll(corners1,1)
    
    corners0 = np.roll(corners1,-1)

    v= corners1-corners0
    an = (np.arctan2(v[:,0],v[:,1]) + 2.0 * np.pi)% (2.0 * np.pi)/np.pi*180
    print(corners1)
    print('an',an,v)
    print(4*180/6)

    angles = []
    for i in range(len(corners1)):
        A = corners1[i] - corners0[i]
        B = corners1[i] - corners2[i] 
        num = np.dot(A, B)
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        angles.append(np.arccos(num/denom) * 180 / np.pi)
        
    return angles


    
 
def PolygonSort2(corners,points):
    """
    # sort corners in counter-clockwise direction
    input:
            corners are indices in points array
            points is list or array of points
    output: 
            cornersWithAngles
    """
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in points[corners])) / n
    cy = float(sum(y for x, y in points[corners])) / n
    
    # create a new list of corners which includes angles
    # angles from the positive x axis
    cornersWithAngles = []
    for i in corners:
        x,y = points[i]       
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi)% (2.0 * np.pi)
        cornersWithAngles.append([i, np.degrees(an)])
    
    #sort it using the angles
    cornersWithAngles.sort(key = lambda tup: tup[1])
    

    return  cornersWithAngles


def PolygonsInner(indices, points):
    pp = np.array(points)[indices,:]
    # Determine inner angle of polygon
    # Generate second array which is shifted by one 
    pp2 = np.roll(pp,1,axis=0)
    # and subtract it from former: this is now a list of vectors
    p_vectors = pp-pp2

    #angles of vectors with respect to positive x-axis
    ang = np.arctan2(p_vectors[:,1],p_vectors[:,0])/np.pi*180+360 % 360
    # shift angles array by one  
    ang2 = np.roll(ang,-1,axis=0)

    #difference of angles is outer angle but we want the inner (inner + outer = 180) 
    inner_angles = (180-(ang2-ang)+360 )% 360
    
    return inner_angles


 # sort corners in counter-clockwise direction
def PolygonSort(corners):
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    
    # create a new list of corners which includes angles
    cornersWithAngles = []
    for x, y in corners:
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi)% (2.0 * np.pi)
        cornersWithAngles.append((x, y, np.degrees(an)))
    
    #sort it using the angles
    cornersWithAngles.sort(key = lambda tup: tup[2])

    return  cornersWithAngles




def PolygonArea(corners):
    """
    # Area of Polygon using Shoelace formula
    # http://en.wikipedia.org/wiki/Shoelace_formula
    # FB - 20120218
    # corners must be ordered in clockwise or counter-clockwise direction
    """
    n = len(corners) # of corners
    area = 0.0
    C_x =0
    C_y =0
    for i in range(n):
        j = (i + 1) % n
        nn = corners[i][0] * corners[j][1] - corners[j][0] * corners[i][1]
        area += nn
        C_x +=  (corners[i][0] + corners[j][0])*nn
        C_y +=  (corners[i][1] + corners[j][1])*nn

    area = abs(area) / 2.0

    # centeroid or arithmetic mean 
    C_x = C_x/(6*area) 
    C_y = C_y/(6* area)

    return (area), C_x, C_y


def PolygonAngles( corners):
    angles = []
    # calculate centroid of the polygon
    n = len(corners) # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of angles
    #print (cx,cy)
    for x, y in corners:
        an = (np.atan2(y - cy, x - cx) + 2.0 * np.pi)% (2.0 * np.pi)
        angles.append((np.degrees(an)))

    return angles






def voronoi_tags(vor):
    sym = {}
    sym['voronoi'] = vor
    sym['vertices'] = vor.vertices #(ndarray of double, shape (nvertices, ndim)) Coordinates of the Voronoi vertices.
    sym['ridge_points'] = vor.ridge_points #ridge_points 	(ndarray of ints, shape (nridges, 2)) Indices of the points between which each Voronoi ridge lies.
    sym['ridge_vertices'] = vor.ridge_vertices #ridge_vertices 	(list of list of ints, shape (nridges, *)) Indices of the Voronoi vertices forming each Voronoi ridge.
    sym['regions'] = vor.regions #regions 	(list of list of ints, shape (nregions, *)) Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi diagram.
    sym['point_region'] = vor.point_region #point_region 	(list of ints, shape (npoints)) Index of the Voronoi region for each input point. If qhull option “Qc” was not specified, the list will contain -1 for points that are not associated with a Voronoi region.

    
    points = vor.points
    nnTree = KDTree(points)
    
    rim = []
    regions=[]

    ###
    # We get all the vertice length
  
    lengths =[]
    for vertice in  vor.ridge_vertices:
        if not(-1 in vertice):
            p1 = vor.vertices[vertice[0]]
            p2 = vor.vertices[vertice[1]]
            lengths.append(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2))
                
    sym['lengths'] = lengths
    sym['median lengths'] = np.median(lengths)
    sym['Min Voronoi Edge'] = np.median(lengths)/1.5
    #print ('median lengths', np.median(lengths))
    #print ('Min Voronoi Edge',np.median(lengths)/1.5)
    cornersHist=[]
    nnHist = []
    nnDistHist =[]
    angleHist = []
    areaHist=[]
    deviationHist =[]

    for i, region in enumerate(vor.point_region):
        x,y = points[i]
        sym[str(i)]={}
        vertices = vor.regions[region]

        ###
        # We get all the rim atoms
        ###

        #if all(v >= 0  and all(vor.vertices[v] >0) and all(vor.vertices[v]<tags['data'].shape[0]) for v in vertices):
        if all(v >= 0  and all(vor.vertices[v] >0) for v in vertices):
             # finite regions only now
            # negative and too large vertices (corners) are excluded

            regions.append(vertices)
            poly = []
            for v in vertices:
                poly.append(vor.vertices[v])

            area, cx,cy = PolygonArea(poly)
            cx = abs(cx)
            cy = abs(cy)

            angles = PolygonAngles(poly)
            angleHist.append(angles)
            areaHist.append(area)
            deviationHist.append(np.sqrt((cx-x)**2+ (cy-y)**2))

            sym[str(i)]['xy'] = [x, y]
            sym[str(i)]['geometric'] = [cx, cy]
            sym[str(i)]['area'] = area

            sym[str(i)]['angles'] = angles
            sym[str(i)]['off center'] = [cx-x, cy-y]

            sym[str(i)]['position'] = 'inside'
            sym[str(i)]['corner'] = vertices
            sym[str(i)]['vertices']=poly
            sym[str(i)]['corners'] = len(vertices)
            cornersHist.append(len(vertices))
            nn = 0
            nnVor = []
            length = []
            for j in range(len(vertices)):
                k = (j+1) % len(vertices)
                p1 = vor.vertices[vertices[j]]
                p2 = vor.vertices[vertices[k]]
                leng = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2)
                length.append(leng)
                sym[str(i)]['length'] = length
                if leng > sym['Min Voronoi Edge']:        
                    nn +=1
                    nnVor.append(vertices[j])
                sym[str(i)]['length'] = length    
            nnP = nnTree.query(points[i],k = nn+1)
            sym [str(i)]['neighbors'] = []
            sym [str(i)]['nn Distance'] = []
            sym [str(i)]['nn']=nn
            if nn>0:
                nnHist.append(nn)   
                for j in range (1,len(nnP[0])):
                    sym [str(i)]['nn Distance'].append(nnP[0][j])
                    sym [str(i)]['neighbors'].append(nnP[1][j])
                    nnDistHist.append(nnP[0][j])
            else:
                rim.append(i)
                sym[str(i)]['position'] = 'rim'
                sym[str(i)]['corners'] = 0
                print('weird nn determination',i)

        else:
            rim.append(i)
            sym[str(i)]['position'] = 'rim'
            sym[str(i)]['corners'] = 0
            sym[str(i)]['xy'] = [x, y]


    sym['average corners']= np.median(cornersHist)
    sym['average area']= np.median(areaHist)
    sym['num atoms at rim']= len(rim)
    sym['num voronoi']= len(points)-len(rim)
    sym['Median Coordination']= np.median(nnHist)
    sym['Median NN Distance']= np.median(nnDistHist)

    sym['Hist corners']= (cornersHist)
    sym['Hist area']= areaHist
    sym['atoms at rim']= (rim)
    sym['Hist Coordination']= (nnHist)
    sym['Hist NN Distance']= (nnDistHist)
    sym['Hist deviation']= (deviationHist)


    return sym
    #print ('average corners', np.median(cornersHist))


def defineSymmetry(tags):

    #make dictionary to store
    if 'symmetry' in tags:
        tags['symmetry'].clear()

    tags['symmetry'] = {}
    sym = tags['symmetry']
    if 'latticeType' in tags:
        latticeTypes = ['None', 'Find Lattice',  'hexagonal', 'honeycomb', 'square', 'square centered',
                'diamond', 'fcc']
        sym['lattice']=latticeTypes[tags['latticeType']]

    sym['number of atoms'] = len(self.tags['atoms'])
    
    points = []
    for i in range(sym['number of atoms']):            
        sym[str(i)] = {}
        sym[str(i)]['index']= i
        sym[str(i)]['x'] = self.tags['atoms'] [i][0]
        sym[str(i)]['y'] = self.tags['atoms'] [i][1]
        sym[str(i)]['intensity'] = self.tags['atoms'] [i][3]
        sym[str(i)]['maximum'] = self.tags['atoms'] [i][4]
        sym[str(i)]['position'] = 'inside'
        sym[str(i)]['Z'] = 0
        sym[str(i)]['Name'] = 'undefined'
        sym[str(i)]['Column'] = -1
        
        points.append([int(sym[str(i)]['x']+0.5),int(sym[str(i)]['y']+0.5)])

    self.points = points.copy()
    


def voronoi2(tags, atoms):

    sym = tags['symmetry']
    points = []
    
    for i in range(sym['number of atoms']):            
        points.append([int(sym[str(i)]['x']+0.5),int(sym[str(i)]['y']+0.5)])


    #points = np.array(atoms[:][0:2])
    vor = sp.Voronoi(points)

    
    sym['voronoi'] = vor

    nnTree = sp.KDTree(points)
    
    rim = []
    regions=[]

    ###
    # We get all the vertice length
  
    lengths =[]
    for vertice in  vor.ridge_vertices:
        if all(v >= 0 for v in vertice):
            p1 = vor.vertices[vertice[0]]
            p2 = vor.vertices[vertice[1]]
            lengths.append(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2))
                
    sym['lengths'] = lengths
    sym['median lengths'] = np.median(lengths)
    sym['Min Voronoi Edge'] = np.median(lengths)/1.5
    #print ('median lengths', np.median(lengths))
    #print ('Min Voronoi Edge',np.median(lengths)/1.5)
    cornersHist=[]
    nnHist = []
    nnDistHist =[]
    angleHist = []
    areaHist=[]
    deviationHist =[]

    for i, region in enumerate(vor.point_region):
        x,y = points[i]

        vertices = vor.regions[region]

        ###
        # We get all the rim atoms
        ###

        if all(v >= 0  and all(vor.vertices[v] >0) and all(vor.vertices[v]<tags['data'].shape[0]) for v in vertices):
            # finite regions only now
            # negative and too large vertices (corners) are excluded

            regions.append(vertices)
            poly = []
            for v in vertices:
                poly.append(vor.vertices[v])

            area, cx,cy = PolygonArea(poly)
            cx = abs(cx)
            cy = abs(cy)

            angles = PolygonAngles(poly)
            angleHist.append(angles)
            areaHist.append(area)
            deviationHist.append(np.sqrt((cx-x)**2+ (cy-y)**2))

            sym[str(i)]['xy'] = [x, y]
            sym[str(i)]['geometric'] = [cx, cy]
            sym[str(i)]['area'] = area

            sym[str(i)]['angles'] = angles
            sym[str(i)]['off center'] = [cx-x, cy-y]

            sym[str(i)]['position'] = 'inside'
            sym[str(i)]['corner'] = vertices
            sym[str(i)]['vertices']=poly
            sym[str(i)]['corners'] = len(vertices)
            cornersHist.append(len(vertices))
            nn = 0
            nnVor = []
            length = []
            for j in range(len(vertices)):
                k = (j+1) % len(vertices)
                p1 = vor.vertices[vertices[j]]
                p2 = vor.vertices[vertices[k]]
                leng = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1] )**2)
                length.append(leng)
                sym[str(i)]['length'] = length
                if leng > sym['Min Voronoi Edge']:        
                    nn +=1
                    nnVor.append(vertices[j])
                sym[str(i)]['length'] = length    
            nnP = nnTree.query(points[i],k = nn+1)
            sym [str(i)]['neighbors'] = []
            sym [str(i)]['nn Distance'] = []
            sym [str(i)]['nn']=nn
            if nn>0:
                nnHist.append(nn)   
                for j in range (1,len(nnP[0])):
                    sym [str(i)]['nn Distance'].append(nnP[0][j])
                    sym [str(i)]['neighbors'].append(nnP[1][j])
                    nnDistHist.append(nnP[0][j])
            else:
                rim.append(i)
                sym[str(i)]['position'] = 'rim'
                sym[str(i)]['corners'] = 0
                print('weird nn determination',i)

        else:
            rim.append(i)
            sym[str(i)]['position'] = 'rim'
            sym[str(i)]['corners'] = 0
            sym[str(i)]['xy'] = [x, y]


    sym['average corners']= np.median(cornersHist)
    sym['average area']= np.median(areaHist)
    sym['num atoms at rim']= len(rim)
    sym['num voronoi']= len(points)-len(rim)
    sym['Median Coordination']= np.median(nnHist)
    sym['Median NN Distance']= np.median(nnDistHist)

    sym['Hist corners']= (cornersHist)
    sym['Hist area']= areaHist
    sym['atoms at rim']= (rim)
    sym['Hist Coordination']= (nnHist)
    sym['Hist NN Distance']= (nnDistHist)
    sym['Hist deviation']= (deviationHist)



    #print ('average corners', np.median(cornersHist))

def intensity_area(image,atoms, radius):
    rr = int(radius+0.5) # atom radius
    print('using radius ',rr, 'pixels')
    
    pixels = np.linspace(0,2*rr,2*rr+1)-rr
    x,y = np.meshgrid(pixels,pixels);
    mask = (x**2+y**2) < rr**2 #
    intensity_area = []
    for i in range(len( atoms)):
        
        x = int(atoms[i][1]   ) 
        y = int(atoms[i][0]   ) 
        area = image[x-rr:x+rr+1,y-rr:y+rr+1]
        if area.shape == mask.shape:
            intensity_area.append((area*mask).sum() )
        else:
            intensity_area.append(-1)
    return intensity_area

def Gauss_2D(params, ydata):
    width = int(ydata.shape[0]/2)
    Gauss_width = params[0]
    x0 = params[1]
    y0 = params[2]
    inten = params[3]

    x, y = np.mgrid[-width:width+1, -width:width+1]


    return np.exp(-((x-x0)**2 + (y-y0)**2) /2./ Gauss_width**2)*inten
def Gauss_difference (params,  xdata, ydata):
    #self.img1b.setImage(gauss)
    gauss = Gauss_2D(params, ydata)
    return (ydata - gauss).flatten()

def atomRefine(image, atoms, radius, MaxInt = 0,MinInt = 0, maxDist = 4):
    """
        fits a Gaussian in a blob
    """
    rr = int(radius+0.5) # atom radius
    print('using radius ',rr, 'pixels')
    
    pixels = np.linspace(0,2*rr,2*rr+1)-rr
    x,y = np.meshgrid(pixels,pixels);
    mask = (x**2+y**2) < rr**2 #

    guess  = [rr*2, 0.0, 0.0 , 1]    
    
    sym = {}
    sym['number_of_atoms'] = len(atoms)
    
    volume = []
    position = []
    intensity_area = []
    maximum_area = []
    newatoms = []
    Gauss_width = []
    Gauss_amplitude = []
    Gauss_intensity = []
    
    done = 0
    for i in range(len(atoms)):
        if done < int((i+1)/len(atoms)*50):
            done = int((i+1)/len(atoms)*50)
            sys.stdout.write('\r')
            # progress output :
            sys.stdout.write("[%-50s] %d%%" % ('='*done, 2*done))
            sys.stdout.flush()
        
        y,x = atoms[i][0:2]
        x = int(x)
        y = int(y)
        append = False
        
        
        area = image[x-rr:x+rr+1,y-rr:y+rr+1]
       
        append = False
        
        if (x-rr) < 0 or y-rr <0 or x+rr+1 > image.shape[0] or y+rr+1 > image.shape[1]:
            position.append(-1)
            intensity_area.append(0) 
            maximum_area.append(0)
        else:
            position.append(1)
            intensity_area.append((area*mask).sum() )
            maximum_area.append((area*mask).max())
            
        if MaxInt>0:
            if area.sum()< MaxInt:                    
                if area.sum() > MinInt:
                    append = True
        elif area.sum()> MinInt:
            append = True
        
        pout = [0,0,0,0]
        if append:
            if (x-rr) < 0 or y-rr <0 or x+rr+1 > image.shape[0] or y+rr+1 > image.shape[1]:
                pass
            else:
                pout, res =  leastsq(Gauss_difference, guess, args=(area, area))
                
            if (abs(pout[1])> maxDist) or (abs(pout[2])> maxDist):
                pout = [0,0,0,0]
    
        volume.append(2* np.pi * pout[3] * pout[0]*pout[0])

        newatoms.append([y+pout[2], x+pout[1]])# ,pout[0],  volume)) #,pout[3]))
        if (all(v == 0 for v in pout)):
            Gauss_intensity.append(0.)
        else:
            Gauss_intensity.append((Gauss_2D(pout, area)*mask).sum() )
        Gauss_width.append(pout[0])
        Gauss_amplitude.append(pout[3])
    
    
    sym['inside'] = position
    sym['intensity_area'] = intensity_area 
    sym['maximum_area'] = maximum_area
    sym['atoms'] = newatoms
    sym['Gauss_width'] = Gauss_width
    sym['Gauss_amplitude'] = Gauss_amplitude
    sym['Gauss_intensity'] = Gauss_intensity
    sym['Gauss_volume'] = volume
    
    return sym

def Fourier_transform(current_channel,data):# = image_channel
    # spatial data
    tags = dict(current_channel.attrs)
    out_tags = {}
    basename = current_channel['title'][()]
    
    sizeX = current_channel['spatial_size_x'][()]
    sizeY = current_channel['spatial_size_y'][()]
    scaleX = current_channel['spatial_scale_x'][()]
    scaleY = current_channel['spatial_scale_y'][()]
    basename = current_channel['title'][()]

    FOV_x = sizeX*scaleX
    FOV_y = sizeY*scaleY
    
    image = data- data.min()
    fft_mag = (np.abs((np.fft.fftshift(np.fft.fft2(image)))))
    
    out_tags['Magnitude']=fft_mag

    ## pixel_size in recipical space
    rec_scale_x = 1/FOV_x  
    rec_scale_y = 1/FOV_y 

    ## Field of View (FOV) in recipical space please note: rec_FOV_x = 1/(scaleX*2)
    rec_FOV_x = rec_scale_x * sizeX /2.
    rec_FOV_y = rec_scale_y * sizeY /2.
    print(rec_FOV_x , 1/(scaleX*2))


    ## Field ofView (FOV) in recipical space
    rec_extend = (-rec_FOV_x,rec_FOV_x-rec_scale_x,rec_FOV_y-rec_scale_y,-rec_FOV_y)

    out_tags['spatial_size_x']=sizeX
    out_tags['spatial_size_y']=sizeY
    out_tags['spatial_scale_x']=rec_scale_x
    out_tags['spatial_scale_y']=rec_scale_y
    out_tags['spatial_origin_x']=sizeX/2.
    out_tags['spatial_origin_y']=sizeY/2.
    out_tags['title']=out_tags['basename']=basename
    out_tags['FOV_x']=rec_FOV_x
    out_tags['FOV_y']=rec_FOV_y
    out_tags['extent']=rec_extend
    
    
    # We need some smoothing (here with a Gaussian)
    smoothing = 3
    fft_mag2 = ndimage.gaussian_filter(fft_mag, sigma=(smoothing, smoothing), order=0)
    #fft_mag2 = np.log2(1+fft_mag2)

    out_tags['data'] = out_tags['Magnitude_smoothed']=fft_mag2
    #prepare mask
    pixelsy = (np.linspace(0,image.shape[0]-1,image.shape[0])-image.shape[0]/2)* rec_scale_x
    pixelsx = (np.linspace(0,image.shape[1]-1,image.shape[1])-image.shape[1]/2)* rec_scale_y
    x,y = np.meshgrid(pixelsx,pixelsy);
    mask = np.zeros(image.shape)

    mask_spot = x**2+y**2 > 1**2 
    mask = mask + mask_spot
    mask_spot = x**2+y**2 < 11**2 
    mask = mask + mask_spot

    mask[np.where(mask==1)]=0 # just in case of overlapping disks

    minimum_intensity = np.log2(1+fft_mag2)[np.where(mask==2)].min()*0.95
    #minimum_intensity = np.mean(fft_mag3)-np.std(fft_mag3)
    maximum_intensity = np.log2(1+fft_mag2)[np.where(mask==2)].max()*1.05
    #maximum_intensity =  np.mean(fft_mag3)+np.std(fft_mag3)*2
    out_tags['minimum_intensity']=minimum_intensity
    out_tags['maximum_intensity']=maximum_intensity
    
    return out_tags



def find_Bragg(fft_tags, spot_threshold = 0 , verbose = False):
    if spot_threshold ==0:
        spot_threshold = 0.05#(fft_tags['maximum_intensity']*10)
    
    # we'll have to switch x and ycoordonates
    center = np.array([int(fft_tags['spatial_origin_y']), int(fft_tags['spatial_origin_x']),1] )
    rec_scale = np.array([fft_tags['spatial_scale_y'], fft_tags['spatial_scale_x'],1])
    data = fft_tags['data'].T
    data = (data-data.min())/data.max()
    spots_random =  (blob_log(data,  max_sigma= 5 , threshold=spot_threshold)-center)*rec_scale
    
    if verbose:
        print(f'found {len(spots_random)} Bragg spots with threshold of {spot_threshold}')
    spots_random[:,2] = np.linalg.norm(spots_random[:,0:2], axis=1)
    spots_index = np.argsort(spots_random[:,2])
    
    spots = spots_random[spots_index]
    spots[:,2] = np.arctan2(spots[:,0], spots[:,1])
    return spots
   

