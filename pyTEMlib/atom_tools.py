""" Atom detection

All atom detection is done here
Everything is in unit of pixel!!

Author: Gerd Duscher

part of pyTEMlib


a pycroscopy package
"""

import numpy as np

import skimage.feature
import sklearn.cluster
import scipy.spatial
import scipy.optimize

import sidpy
import pyTEMlib

def find_atoms(image, atom_size=0.1, threshold=0.):
    """ Find atoms is a simple wrapper for blob_log in skimage.feature

    Parameters
    ----------
    image: sidpy.Dataset
        the image to find atoms
    atom_size: float
        visible size of atom blob diameter in nm gives minimal distance between found blobs
    threshold: float
        threshold for blob finder; (usually between 0.001 and 1.0) 
        for threshold <= 0 we use the RMS contrast

    Returns
    -------
    atoms: numpy array(Nx3)
        atoms positions and radius of blob
    """

    if not isinstance(image, sidpy.Dataset):
        raise TypeError('We need a sidpy.Dataset')
    if image.data_type.name != 'IMAGE':
        raise TypeError('We need sidpy.Dataset of sidpy.Datatype: IMAGE')
    if not isinstance(atom_size, (float, int)):
        raise TypeError('atom_size parameter has to be a number')
    if not isinstance(threshold, float):
        raise TypeError('threshold parameter has to be a float number')

    scale_x = np.unique(np.gradient(image.dim_0.values))[0]
    im = np.array(image-image.min())
    im = im/im.max()
    if threshold <= 0.:
        threshold = np.std(im)
    atoms = skimage.feature.blob_log(im, max_sigma=atom_size/scale_x, threshold=threshold)
    return atoms


def atoms_clustering(atoms, mid_atoms, number_of_clusters=3, nearest_neighbours=7):
    """ A wrapper for sklearn.cluster kmeans clustering of atoms.

    Parameters
    ----------
    atoms: list or np.array (Nx2)
        list of all atoms
    mid_atoms: list or np.array (Nx2)
        atoms to be evaluated
    number_of_clusters: int
        number of clusters to sort (ini=3)
    nearest_neighbours: int
        number of nearest neighbours evaluated

    Returns
    -------
    clusters, distances, indices: numpy arrays
    """

    # get distances
    nn_tree = scipy.spatial.KDTree(np.array(atoms)[:, 0:2])

    distances, indices = nn_tree.query(np.array(mid_atoms)[:, 0:2], nearest_neighbours)

    # Clustering with fixed RNG in kmeans
    k_means = sklearn.cluster.KMeans(n_clusters=number_of_clusters, random_state=0)
    k_means.fit(distances)
    clusters = k_means.predict(distances)

    return clusters, distances, indices


def gauss_difference(params, area):
    """
    Difference between part of an image and a Gaussian
    This function is used int he atom refine function of pyTEMlib

    Parameters
    ----------
    params: list
        list of Gaussian parameters [width, position_x, position_y, intensity]
    area:  numpy array
        2D matrix = part of an image

    Returns
    -------
    numpy array: flattened array of difference

    """
    gauss = pyTEMlib.probe_tools.make_gauss(area.shape[0], area.shape[1],
                                             width=params[0], x0=params[1],
                                             y0=params[2], intensity=params[3])
    return (area - gauss).flatten()


def atom_refine(image, atoms, radius, max_int=0, min_int=0, max_dist=4):
    """Fits a Gaussian in a blob of an image

    Parameters
    ----------
    image: np.array or sidpy Dataset
    atoms: list or np.array
        positions of atoms
    radius: float
        radius of circular mask to define fitting of Gaussian
    max_int: float
        optional - maximum intensity to be considered for fitting 
        (to exclude contaminated areas for example)
    min_int: float
        optional - minimum intensity to be considered for fitting
        (to exclude contaminated holes for example)
    max_dist: float
        optional - maximum distance of movement of Gaussian during fitting

    Returns
    -------
    sym: dict
        dictionary containing new atom positions and other output
    """
    atom_radius = int(radius + 0.5)  # atom radius

    pixels = np.linspace(0, 2 * atom_radius, 2 * atom_radius + 1) - atom_radius
    x, y = np.meshgrid(pixels, pixels)
    mask = (x ** 2 + y ** 2) < atom_radius**2

    guess = [atom_radius * 2, 0.0, 0.0, 1]

    volume = []
    position = []
    intensities = []
    maximum_area = []
    new_atoms = []
    gauss_width = []
    gauss_amplitude = []
    gauss_intensity = []

    for atom in atoms:
        x = int(atom[0])
        y = int(atom[1])
        area = image[x - atom_radius:x + atom_radius + 1, y - atom_radius:y + atom_radius + 1]
        append = False

        if (atom_radius < x < image.shape[0] - atom_radius and
            atom_radius < y < image.shape[1] - atom_radius):
            position.append(1)
            intensities.append((area * mask).sum())
            maximum_area.append((area * mask).max())
        else:  # atom on rim
            position.append(-1)
            intensities.append(-1.)
            maximum_area.append(-1.)

        if max_int > 0 and min_int < area.sum() < max_int:
            append = True
        elif area.sum() > min_int:
            append = True

        pout = [0, 0, 0, 0]
        if append:
            if position[-1] > 0:
                [pout, _] = scipy.optimize.leastsq(gauss_difference, guess, args=area)
            if (abs(pout[1]) > max_dist) or (abs(pout[2]) > max_dist):
                pout = [0, 0, 0, 0]

        volume.append(2 * np.pi * pout[3] * pout[0] * pout[0])

        new_atoms.append([x + pout[1], y + pout[2]])  # ,pout[0],  volume)) #,pout[3]))
        if all(v == 0 for v in pout):
            gauss_intensity.append(0.)
        else:
            gauss = pyTEMlib.probe_tools.make_gauss(area.shape[0], area.shape[1], width=pout[0],
                                                    x0=pout[1], y0=pout[2], intensity=pout[3])
            gauss_intensity.append((gauss * mask).sum())
        gauss_width.append(pout[0])
        gauss_amplitude.append(pout[3])

    sym = {'number_of_atoms': len(atoms),
           'inside': position,
           'intensity_area': intensities,
           'maximum_area': maximum_area,
           'atoms': new_atoms,
           'gauss_width': gauss_width,
           'gauss_amplitude': gauss_amplitude,
           'gauss_intensity': gauss_intensity,
           'gauss_volume': volume}

    return sym


def intensity_area(image, atoms, radius):
    """
    integrated intensity of atoms in an image with a mask around each atom of radius radius
    """
    atom_radius = int(radius + 0.5)  # atom radius
    print('using radius ', atom_radius, 'pixels')

    pixels = np.linspace(0, 2 * atom_radius, 2 * atom_radius + 1) - atom_radius
    x, y = np.meshgrid(pixels, pixels)
    mask = np.array((x ** 2 + y ** 2) < atom_radius ** 2)
    intensities = []
    for atom in atoms:
        x = int(atom[1])
        y = int(atom[0])
        area = image[x - atom_radius:x + atom_radius + 1, y - atom_radius:y + atom_radius + 1]
        if area.shape == mask.shape:
            intensities.append((area * mask).sum())
        else:
            intensities.append(-1)
    return intensities
