"""
image_tools.py
by Gerd Duscher, UTK
part of pyTEMlib
MIT license except where stated differently

This version is build on top of pycroscopy.image package of the pycrocsopy ecosysgtem.
"""

import numpy as np

import scipy
import skimage
import sklearn
import matplotlib
import matplotlib.pylab as plt

import sidpy
import pyTEMlib

## import all function of the image package of pycroscopy
from .image import *
from .image.image_utilities import pol2cart, cart2pol, xy2polar


def get_atomic_pseudo_potential(fov, atoms, size=512, rotation=0):
    """Big assumption: the atoms are not near the edge of the unit cell
    # If any atoms are close to the edge (ex. [0,0]) then the potential will be clipped
    # before calling the function, shift the atoms to the center of the unit cell
    """

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
        # important for images at various fov.  Room for improvement with theory
        gauss_width = atom_width/pixel_size
        gauss = pyTEMlib.probe_tools.make_gauss(max_size, max_size,
                                                width=gauss_width,
                                                x0=x, y0=y)
        unit_cell_potential += gauss * atomic_number  # gauss is already normalized to 1

    # Create interpolation function for unit cell potential
    x_grid = np.linspace(0, fov * max_size / size, max_size)
    y_grid = np.linspace(0, fov * max_size / size, max_size)
    interpolator = scipy.interpolate.RegularGridInterpolator((x_grid, y_grid),
                                                             unit_cell_potential,
                                                             bounds_error=False,
                                                             fill_value=0)
    # Vectorized computation of the full potential map with max_size
    x_coords, y_coords = np.meshgrid(np.linspace(0, fov, max_size),
                                     np.linspace(0, fov, max_size),
                                     indexing="ij")
    xtal_x = x_coords % unit_cell_size[0]
    xtal_y = y_coords % unit_cell_size[1]
    potential_map = interpolator((xtal_x.ravel(), xtal_y.ravel())).reshape(max_size, max_size)

    # Rotate and crop the potential map
    potential_map = scipy.ndimage.rotate(potential_map, rotation, reshape=False)
    center = potential_map.shape[0] // 2
    potential_map = potential_map[center - size // 2:center + size // 2,
                                  center - size // 2:center + size // 2]
    potential_map = scipy.ndimage.gaussian_filter(potential_map,3)

    return potential_map

def convolve_probe(ab, potential):
    """ Convolve probe with potential using FFT based convolution"""
    # the pixel sizes should be the exact same as the potential
    final_sizes = potential.shape

    # Perform FFT-based convolution
    pad_height = pad_width = potential.shape[0] // 2
    potential = np.pad(potential, ((pad_height, pad_height),
                                   (pad_width, pad_width)), mode='constant')

    probe, _, _ = pyTEMlib.probe_tools.get_probe(ab, potential.shape[0],
                                                 potential.shape[1],
                                                 scale='mrad', verbose=False)
    convolved = scipy.signal.fftconvolve(potential, probe, mode='same')

    # Crop to original potential size
    start_row = pad_height
    start_col = pad_width
    end_row = start_row + final_sizes[0]
    end_col = start_col + final_sizes[1]

    image = convolved[start_row:end_row, start_col:end_col]
    return probe, image

def get_wavelength(e0):
    """
    Calculates the relativistic corrected de Broglie wave length of an electron
    # Wavelength in 1/nm
    Parameters
    ----------
    e0: float
      acceleration voltage in volt

    Returns
    -------
    wave length in 1/nm
    """
    ev = scipy.constants.e * e0
    h = scipy.constants.h
    m = scipy.constants.m_e
    c = scipy.constants.c
    return h / np.sqrt(2 * m * ev * (1 + ev / (2 * m * c**2))) * 10**9

def fourier_transform(dset: sidpy.Dataset) -> sidpy.Dataset:
    """
        Reads information into dictionary 'tags', performs 'FFT', 
        and provides a smoothed FT and reciprocal
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
    image_dims = dset.get_image_dims(return_axis=True)
    if dset.data_type.name == 'IMAGE_STACK':
        stack_dim = dset.get_dimensions_by_type('TEMPORAL')

        if len(image_dims) != 2:
            raise ValueError('need at least two SPATIAL dimension for an image stack')

        for i in range(dset.ndim):
            if i in image_dims:
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
    elif dset.data_type.name == 'IMAGE':
        new_image = np.array(dset)
    else:
        return

    new_image = new_image - new_image.min()
    fft_transform = (np.fft.fftshift(np.fft.fft2(np.array(new_image))))

    image_dims = dset.get_image_dims(return_axis=True)

    units_x = '1/' + image_dims[0].units
    units_y = '1/' + image_dims[1].units
    fft_dset = sidpy.Dataset.from_array(fft_transform)
    fft_dset.quantity = dset.quantity
    fft_dset.units = 'a.u.'
    fft_dset.data_type = 'IMAGE'
    fft_dset.source = dset.title
    fft_dset.modality = 'fft'
    axis = np.fft.fftshift(np.fft.fftfreq(new_image.shape[0], d=dset.x[1]-dset.x[0]))
    fft_dset.set_dimension(0, sidpy.Dimension(axis,
                                              name='u', units=units_x,
                                              dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))
    axis = np.fft.fftshift(np.fft.fftfreq(new_image.shape[1], d=dset.y[1]-dset.y[0]))
    fft_dset.set_dimension(1, sidpy.Dimension(axis,
                                              name='v', units=units_y,
                                              dimension_type='RECIPROCAL',
                                              quantity='reciprocal_length'))
    return fft_dset


def center_diffractogram(dset, return_plot = True, smoothing = 1,
                         min_samples = 10, beamstop_size = 0.1):
    """Find center of diffractogram by fitting a circle to the diffraction ring"""
    mean_radius = 0
    try:
        diff = np.array(dset).T.astype(np.float16)
        diff[diff < 0] = 0
        threshold = skimage.filters.threshold_otsu(diff)
        binary = (diff > threshold).astype(float)
        # Smooth before edge detection
        smoothed_image = scipy.ndimage.gaussian_filter(binary, sigma=smoothing)
        smooth_threshold = skimage.filters.threshold_otsu(smoothed_image)
        smooth_binary = (smoothed_image > smooth_threshold).astype(float)

        # add a circle to mask the beamstop
        x, y = np.meshgrid(np.arange(dset.shape[0]), np.arange(dset.shape[1]))
        radius = (x - dset.shape[0] / 2) ** 2 + (y - dset.shape[1] / 2) ** 2
        circle = radius < (beamstop_size * dset.shape[0] / 2) ** 2
        smooth_binary[circle] = 1

        # Find the edges using the Sobel operator
        edges = skimage.filters.sobel(smooth_binary)
        edge_points = np.argwhere(edges)

        # Use DBSCAN to cluster the edge points
        db = sklearn.cluster.DBSCAN(eps=10, min_samples=min_samples).fit(edge_points)
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
            ri = np.sqrt((x - c[0])**2 + (y - c[1])**2)
            return ri - ri.mean()
        x_m = np.mean(edge_points[:, 1])
        y_m = np.mean(edge_points[:, 0])
        center_guess = x_m, y_m
        center, _ = scipy.optimize.leastsq(calc_distance, center_guess,
                                             args=(edge_points[:, 1], edge_points[:, 0]))
        mean_radius = (np.mean(calc_distance(center, edge_points[:, 1], edge_points[:, 0]))
                       + np.sqrt((edge_points[:, 1] - center[0])**2
                                 + (edge_points[:, 0] - center[1])**2).mean())
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
        self.x = 0
        self.y = 0
        self.z = 0
        self.start_x =  self.end_x
        self.start_y = self.end_y
        self.moved_point = [0, 0]
        self.new_point = [0, 0]
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
        """ Update line profile"""
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
        zi2 = scipy.ndimage.map_coordinates(self.data.T, np.vstack((x, y)))

        x_axis = np.linspace(0, length_plot, len(zi2))
        self.x = x_axis
        self.z = zi2

        self.line_plot.set_xdata(x_axis)
        self.line_plot.set_ydata(zi2)
        self.ax[1].set_xlim(0, x_axis.max())
        self.ax[1].set_ylim(zi2.min(), zi2.max())
        self.ax[1].draw()


class LineSelector(matplotlib.widgets.PolygonSelector):
    """ Line selector with adjustable line width"""
    def __init__(self, ax, onselect, line_width=1, **kwargs):
        super().__init__(ax, onselect, **kwargs)
        bounds = ax.viewLim.get_points()
        np.max(bounds[0])
        self.line_verts = np.array([[np.max(bounds[1])/2, np.max(bounds[0])/5],
                                    [np.max(bounds[1])/2, np.max(bounds[0])/5+1],
                                    [np.max(bounds[1])/5, np.max(bounds[0])/2],
                                    [np.max(bounds[1])/5, np.max(bounds[0])/2]])
        self.verts = self.line_verts
        self.line_width = line_width

    def set_linewidth(self, line_width=None):
        """ Set the line width of the line selector"""
        if line_width is not None:
            self.line_width = line_width

        m = -(self.line_verts[0, 1]-self.line_verts[3, 1])/(self.line_verts[0, 0]
                                                            -self.line_verts[3, 0])
        c = 1/np.sqrt(1+m**2)
        s = c*m
        self.line_verts[1] = [self.line_verts[0, 0]+self.line_width*s,
                              self.line_verts[0, 1]+self.line_width*c]
        self.line_verts[2] = [self.line_verts[3, 0]+self.line_width*s,
                              self.line_verts[3, 1]+self.line_width*c]
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
    This function extracts a line profile from a given dataset. 
    The line profile is a representation of the data values 
    along a specified line in the dataset. 
    This function works for both image and spectral image data types.

    Args:
        dataset (sidpy.Dataset): The input dataset from which to extract the line profile.
        line (list): A list specifying the line along which the profile should be extracted.
        spline_order (int, optional): The order of the spline interpolation to use. 
        Default is -1, which means no interpolation.

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
        profile_dataset.set_dimension(0, sidpy.Dimension(np.linspace(xv[0,0], xv[-1,-1],
                                                                     profile_dataset.shape[0]),
                                                  name='x', units=dataset.x.units,
                                                  quantity=dataset.x.quantity,
                                                  dimension_type='spatial'))

    if dataset.data_type.name == 'SPECTRAL_IMAGE':
        spectral_axis = dataset.get_spectral_dims(return_axis=True)[0]
        if spline_order > -1:
            xv, yv, zv = get_line_selection_points_interpolated(line, z_length=dataset.shape[2])
            profile = scipy.ndimage.map_coordinates(np.array(dataset), [xv, yv, zv],
                                                    order=spline_order)
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
                                                  name='x', units=dataset.x.units,
                                                  quantity=dataset.x.quantity,
                                                  dimension_type='spatial'))
        profile_dataset.set_dimension(1, sidpy.Dimension([0, 1],
                                                  name='y', units=dataset.x.units,
                                                  quantity=dataset.x.quantity,
                                                  dimension_type='spatial'))

        profile_dataset.set_dimension(2, spectral_axis)
    return profile_dataset


def get_line_selection_points_interpolated(line, z_length=1):
    """ Get line selection points from line selector with interpolation in z direction"""
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
    """ Get line selection points from line selector"""
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
    """ Get line profile from data array"""
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
    if 'maximum_intensity' not in image_tags:
        image_tags['maximum_intensity'] = image_tags['plotimage'].max()
    data = image_tags['plotimage']
    vmin = image_tags['minimum_intensity']
    vmax = image_tags['maximum_intensity']
    if 'color_map' not in image_tags:
        image_tags['color_map'] = color_map_list[0]

    cmap = plt.cm.get_cmap(image_tags['color_map'])
    colors = cmap(np.linspace(0., 1., nbins))
    norm2 = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    hist, bin_edges = np.histogram(data, np.linspace(vmin, vmax, nbins), density=True)

    width = bin_edges[1]-bin_edges[0]

    def onselect(vmin, vmax):
        """on select update histogram"""
        ax1.clear()
        cmap = plt.cm.get_cmap(image_tags['color_map'])
        colors = cmap(np.linspace(0., 1., nbins))
        norm2 = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        hist2, bin_edges2 = np.histogram(data, np.linspace(vmin, vmax, nbins), density=True)

        width2 = bin_edges2[1]-bin_edges2[0]
        for i in range(nbins-1):
            histogram[i].xy = (bin_edges2[i], 0)
            histogram[i].set_height(hist2[i])
            histogram[i].set_width(width2)
            histogram[i].set_facecolor(colors[i])
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(0, hist2.max()*1.01)

        # cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
        # norm=norm2, orientation='horizontal')

        image_tags['minimum_intensity'] = vmin
        image_tags['maximum_intensity'] = vmax

    def onclick(event):
        """on click change color map"""
        # global event2
        # event2 = event
        button_click = 'double' if event.dblclick else 'single'
        print(f"{button_click} click: button={event.button},"
              + f" x={event.x}, y={event.y}, xdata={event.xdata}, ydata={event.ydata}")
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

    histogram = ax.bar(bin_edges[0:-1], hist, width=width, color=colors,
                       edgecolor='black', alpha=0.8)
    onselect(vmin, vmax)
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm2, orientation='horizontal')

    rectprops = dict(facecolor='blue', alpha=0.5)

    span = matplotlib.widgets.SpanSelector(ax, onselect, 'horizontal', props=rectprops)

    cid = fig2.canvas.mpl_connect('button_press_event', onclick)
    return span


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
    r_crystal, phi_crystal, _ = xy2polar(crystal_spots)
    angle_index = np.argmin(np.abs(r_experiment-r_crystal[1]))
    rotation_angle = phi_experiment[angle_index] % (2*np.pi) - phi_crystal[1]
    print(phi_experiment[angle_index])
    st = np.sin(rotation_angle)
    ct = np.cos(rotation_angle)
    rotation_matrix = np.array([[ct, -st], [st, ct]])

    return rotation_matrix, rotation_angle


def calibrate_image_scale(fft_tags, spots_reference, spots_experiment):
    """depreciated get change of scale from comparison of spots to Bragg angles """

    dist_reference = np.linalg.norm(spots_reference, axis=1)
    distance_experiment = np.linalg.norm(spots_experiment, axis=1)

    first_reflections = abs(distance_experiment - dist_reference.min()) < .2
    print('Evaluate ', first_reflections.sum(), 'reflections')
    closest_exp_reflections = spots_experiment[first_reflections]

    def func(params, xdata, ydata):
        dgx, dgy = params
        return np.sqrt((xdata * dgx) ** 2 + (ydata * dgy) ** 2) - dist_reference.min()

    x0 = [1.001, 0.999]
    [dg, _] = scipy.optimize.leastsq(func, x0, args=(closest_exp_reflections[:, 0],
                                                       closest_exp_reflections[:, 1]))
    return dg


def align_crystal_reflections(spots, crystals):
    """ Depreciated - use diffraction spots"""

    crystal_reflections_polar = []
    angles = []
    exp_r, exp_phi = cart2pol(spots)  # just in polar coordinates

    for tags in crystals:
        # sorted by r and phi , only positive angles
        r, phi, indices = xy2polar(tags['allowed']['g'])
        # we mask the experimental values that are found already
        angle = 0.

        angle_i = np.argmin(np.abs(exp_r - r[1]))
        angle = exp_phi[angle_i] - phi[0]
        angles.append(angle)  # Determine rotation angle

        crystal_reflections_polar.append([r, angle + phi, indices])
        tags['allowed']['g_rotated'] = pol2cart(r, angle + phi)
        """for spot in tags['allowed']['g']:
            dif = np.linalg.norm(spots[:, 0:2]-spot[0:2], axis=1)
        """
    return crystal_reflections_polar, angles
