#
# All atom detection is done here
# Everything is in unit of pixel!!
#

import numpy as np
import sys

from skimage.feature import peak_local_max
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import scipy.optimize as optimization

from .probe_tools import *
from .file_tools import *


def find_atoms(image, tags):
    """
    Find atoms - old please do not use
    """
    image = image - image.min()
    image = image / image.max()

    if 'sigma_min' not in tags:
        tags['sigma_min'] = 0.1
    if 'resolution' not in tags:
        tags['resolution'] = 0.1

    if 'ROIsize' not in tags:
        tags['ROIsize'] = 100.

    res = tags['resolution'] / tags['pixel_size']  # * tags['ROIsize']/100.
    print('res', res)
    coordinates = peak_local_max(image, min_distance=int(res / 2), threshold_rel=tags['sigma_min'], exclude_border=True)
    print('coor', len(coordinates))
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
    r = tags['resolution'] / sc * tags['ROIsize'] / 100. / 2.
    tags['radius'] = r

    #######################################
    # Now we determine intensity #
    #######################################

    ###############
    # Make a circular mask for integration of atom intensity
    ################
    rr = int(r + 0.5)
    mask = np.zeros((2 * rr + 1, 2 * rr + 1))

    for i in range(2 * rr + 1):
        for j in range(2 * rr + 1):
            if (i - rr) ** 2 + (j - rr) ** 2 < rr ** 2 + 0.1:
                mask[i, j] = 1

    ###
    # Determine  pixel position and intensity  of all atoms
    ###
    atoms = []
    for i in range(len(coordinates)):
        x, y = coordinates[i]

        if rr < x < image.shape[1] - rr and rr < y < image.shape[0] - rr:
            area = image[x - rr:x + rr + 1, y - rr:y + rr + 1]
            arr = area * mask
            atoms.append((x, y, rr, arr.sum(), arr.max()))

    print(' Detected ', len(atoms), ' atoms')
    atoms.sort()
    return atoms


def atoms_clustering(atoms, mid_atoms, number_of_clusters=3, nearest_neighbours=7):
    """
    A wrapper for scipy kmeans clustering of atoms.
    """
    # get distances
    nn_tree = cKDTree(np.array(atoms)[:, 0:2])

    distances, indices = nn_tree.query(np.array(mid_atoms)[:, 0:2], nearest_neighbours)

    # Clustering
    k_means = KMeans(n_clusters=number_of_clusters, random_state=0)  # Fixing the RNG in kmeans
    k_means.fit(distances)
    clusters = k_means.predict(distances)
    return clusters, distances, indices


def gauss_difference(params, area):
    """
    Difference between part of an image and a Gaussian
    This function is used int he atom refine function of pyTEMlib

    Input:
    params: list of Gaussian parameters [width, position_x, position_y, intensity]
    area:  numpy array of 2D matrix = part of an image

    Output:
    numpy array: flattened array of difference

    """
    gauss = make_gauss(area.shape[0], area.shape[1], width=params[0], x0=params[1], y0=params[2], intensity=params[3])
    return (area - gauss).flatten()


def atom_refine(image, atoms, radius, max_int=0, min_int=0, max_dist=4):
    """
        fits a Gaussian in a blob of an image

    Input:
    image: np.array or sidpy Dataset
    atoms: positions of atoms
    radius: radius of circular mask to define fitting of Gaussian

    optional
    max_int: maximum intensity to be considered for fitting (to exclude contaminated areas for example)
    min_int: minimum intensity to be considered for fitting (to exclude contaminated holes for example)
    max_dist: maximum distance of movement of Gaussian during fitting

    Output:
    dictionary: contains new atom positions and other output such as intensity of the fitted Gaussian
    """
    rr = int(radius + 0.5)  # atom radius
    print('using radius ', rr, 'pixels')

    pixels = np.linspace(0, 2 * rr, 2 * rr + 1) - rr
    x, y = np.meshgrid(pixels, pixels)
    mask = (x ** 2 + y ** 2) < rr ** 2

    guess = [rr * 2, 0.0, 0.0, 1]

    sym = {'number_of_atoms': len(atoms)}

    volume = []
    position = []
    intensities = []
    maximum_area = []
    new_atoms = []
    gauss_width = []
    gauss_amplitude = []
    gauss_intensity = []
    if QT_available:
        progress = QtWidgets.QProgressDialog("Refine Atom Positions", "Abort", 0, len(atoms))
        progress.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # progress.setWindowModality(Qt.WindowModal);
        progress.show()

    done = 0
    for i in range(len(atoms)):
        if QT_available:
            progress.setValue(i)
            progress.processEvents()
        else:
            if done < int((i + 1) / len(atoms) * 50):
                done = int((i + 1) / len(atoms) * 50)
                sys.stdout.write('\r')
                # progress output :
                sys.stdout.write("[%-50s] %d%%" % ('=' * done, 2 * done))
                sys.stdout.flush()

        x, y = atoms[i][0:2]
        x = int(x)
        y = int(y)

        area = image[x - rr:x + rr + 1, y - rr:y + rr + 1]

        append = False

        if (x - rr) < 0 or y - rr < 0 or x + rr + 1 > image.shape[0] or y + rr + 1 > image.shape[1]:
            position.append(-1)
            intensities.append(0)
            maximum_area.append(0)
        else:
            position.append(1)
            intensities.append((area * mask).sum())
            maximum_area.append((area * mask).max())

        if max_int > 0:
            if area.sum() < max_int:
                if area.sum() > min_int:
                    append = True
        elif area.sum() > min_int:
            append = True

        pout = [0, 0, 0, 0]
        if append:
            if (x - rr) < 0 or y - rr < 0 or x + rr + 1 > image.shape[0] or y + rr + 1 > image.shape[1]:
                pass
            else:
                [pout, _] = optimization.leastsq(gauss_difference, guess, args=(area, area))

            if (abs(pout[1]) > max_dist) or (abs(pout[2]) > max_dist):
                pout = [0, 0, 0, 0]

        volume.append(2 * np.pi * pout[3] * pout[0] * pout[0])

        new_atoms.append([x + pout[1], y + pout[2]])  # ,pout[0],  volume)) #,pout[3]))
        if all(v == 0 for v in pout):
            gauss_intensity.append(0.)
        else:
            gauss = make_gauss(area.shape[0], area.shape[1], width=pout[0], x0=pout[1], y0=pout[2], intensity=pout[3])
            gauss_intensity.append((gauss * mask).sum())
        gauss_width.append(pout[0])
        gauss_amplitude.append(pout[3])

    sym['inside'] = position
    sym['intensity_area'] = intensities
    sym['maximum_area'] = maximum_area
    sym['atoms'] = new_atoms
    sym['gauss_width'] = gauss_width
    sym['gauss_amplitude'] = gauss_amplitude
    sym['gauss_intensity'] = gauss_intensity
    sym['gauss_volume'] = volume

    return sym


def intensity_area(image, atoms, radius):
    """
    integrated intensity of atoms in an image with a mask around each atom of radius radius
    """
    rr = int(radius + 0.5)  # atom radius
    print('using radius ', rr, 'pixels')

    pixels = np.linspace(0, 2 * rr, 2 * rr + 1) - rr
    x, y = np.meshgrid(pixels, pixels)
    mask = (x ** 2 + y ** 2) < rr ** 2
    intensities = []
    for i in range(len(atoms)):
        x = int(atoms[i][1])
        y = int(atoms[i][0])
        area = image[x - rr:x + rr + 1, y - rr:y + rr + 1]
        if area.shape == mask.shape:
            intensities.append((area * mask).sum())
        else:
            intensities.append(-1)
    return intensities
