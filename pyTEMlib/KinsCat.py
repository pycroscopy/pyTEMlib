"""
KinsCat
Kinematic Scattering Theory
Copyright by Gerd Duscher

The University of Tennessee, Knoxville
Department of Materials Science & Engineering

Sources:
   Scattering Theory:
   Zuo and Spence, "Advanced TEM", 2017

   Spence and Zuo, Electron Microdiffraction, Plenum 1992

   Atomic Form Factor:
       Kirkland: Advanced Computing in Electron Microscopy 2nd edition
       Appendix C

Units:
    everything is in SI units, except length is given in nm.

Usage:
    See the notebooks for examples of these routines

All the input and output is done through a dictionary
"""

# numerical packages used
import numpy as np
import scipy.constants as const
import itertools
from scipy import spatial

import sidpy
# plotting package used
import matplotlib.pylab as plt  # basic plotting
import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
from matplotlib.patches import Circle  # , Ellipse, Rectangle
from matplotlib.collections import PatchCollection

import ase.io
import pyTEMlib.file_tools as ft
from pyTEMlib.crystal_tools import *
import os


_version_ = 0.5

print('Using KinsCat library version ', _version_, ' by G.Duscher')
_spglib_present = True
try:
    import spglib
except ModuleNotFoundError:
    _spglib_present = False

if not _spglib_present:
    print('spglib not installed; Symmetry functions of spglib disabled')
else:
    print('Symmetry functions of spglib enabled')


def read_poscar(filename):
    print('read_poscar and read_cif moved to file_tools, \n'
          'please use that library in the future!')
    ft.read_poscar(filename)


# Some Structure Determination Routines
def example(verbose=True):
    """
    same as Zuo_fig_3_18
    """
    return Zuo_fig_3_18(verbose=verbose)


def Zuo_fig_3_18(verbose=True):
    """
    Input for Figure 3.18 in Zuo and Spence \"Advanced TEM\", 2017

    This input acts as an example as well as a reference

    Returns:
        dictionary: tags is the dictionary of all input and output paramter needed to reproduce that figure.
    """

    # INPUT
    # Create Silicon structure (Could be produced with Silicon routine)
    if verbose:
        print('Sample Input for Figure 3.18 in Zuo and Spence \"Advanced TEM\", 2017')
    tags = {'crystal_name': 'Silicon'}
    if verbose:
        print('tags[\'crystal\'] = ', tags['crystal_name'])
    a = 0.514  # nm

    tags['lattice_parameter_nm'] = a
    if verbose:
        print('tags[\'lattice_parameter_nm\'] =', tags['lattice_parameter_nm'])
    tags['unit_cell'] = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    if verbose:
        print('tags[\'unit_cell\'] =', tags['unit_cell'])
    tags['elements'] = list(itertools.repeat('Si', 8))
    if verbose:
        print('tags[\'elements\'] =', tags['elements'])
    base = [(0., 0., 0.), (0.5, 0.0, 0.5), (0.5, 0.5, 0.), (0., 0.5, 0.5)]
    tags['base'] = np.array(base + (np.array(base) + (.25, .25, .25)).tolist())
    if verbose:
        print('tags[\'base\'] =', tags['base'])

    # Define Experimental Conditions
    tags['convergence_angle_mrad'] = 7

    tags['acceleration_voltage_V'] = 101.6*1000.0  # V
    if verbose:
        print('tags[\'acceleration_voltage_V\'] =', tags['acceleration_voltage_V'])

    tags['convergence_angle_mrad'] = 7.1  # mrad;  0 is parallel illumination
    if verbose:
        print('tags[\'convergence_angle_mrad\'] =', tags['convergence_angle_mrad'])

    tags['zone_hkl'] = np.array([-2, 2, 1])  # incident neares zone axis: defines Laue Zones!!!!
    if verbose:
        print('tags[\'zone_hkl\'] =', tags['zone_hkl'])
    tags['mistilt'] = np.array([0, 0, 0])  # mistilt in degrees
    if verbose:
        print('tags[\'mistilt\'] =', tags['mistilt'])

    # Define Simulation Parameters

    tags['Sg_max'] = .2  # 1/nm  maximum allowed excitation error
    if verbose:
        print('tags[\'Sg_max\'] =', tags['Sg_max'])

    tags['hkl_max'] = 9   # Highest evaluated Miller indices
    if verbose:
        print('tags[\'hkl_max\'] =', tags['hkl_max'])

        print('##################')
        print('# Output Options #')
        print('##################')

    # Output options

    tags['background'] = 'black'  # 'white'  'grey'
    if verbose:
        print('tags[\'background\'] =', tags['background'], '# \'white\',  \'grey\' ')
    tags['color map'] = 'plasma'
    if verbose:
        print('tags[\'color map\'] =', tags['color map'], '#,\'cubehelix\',\'Greys\',\'jet\' ')

    tags['plot HOLZ'] = 1
    if verbose:
        print('tags[\'plot HOLZ\'] =', tags['plot HOLZ'])
    tags['plot HOLZ excess'] = 1
    if verbose:
        print('tags[\'plot HOLZ excess\'] =', tags['plot HOLZ excess'])
    tags['plot Kikuchi'] = 1
    if verbose:
        print('tags[\'plot Kikuchi\'] =', tags['plot Kikuchi'])
    tags['plot reflections'] = 1
    if verbose:
        print('tags[\'plot reflections\'] =', tags['plot reflections'])

    tags['label HOLZ'] = 0
    if verbose:
        print('tags[\'label HOLZ\'] =', tags['label HOLZ'])
    tags['label Kikuchi'] = 0
    if verbose:
        print('tags[\'label Kikuchi\'] =', tags['label Kikuchi'])
    tags['label reflections'] = 0
    if verbose:
        print('tags[\'label reflections\'] =', tags['label reflections'])

    tags['label color'] = 'black'
    if verbose:
        print('tags[\'label color\'] =', tags['label color'])
    tags['label size'] = 10
    if verbose:
        print('tags[\'label size\'] =', tags['label size'])

    tags['color Laue Zones'] = ['red',  'blue', 'green', 'blue', 'green']   # , 'green', 'red'] #for OLZ give a sequence
    if verbose:
        print('tags[\'color Laue Zones\'] =', tags['color Laue Zones'], ' #[\'red\', \'blue\', \'lightblue\']')

    tags['color Kikuchi'] = 'green'
    if verbose:
        print('tags[\'color Kikuchi\'] =', tags['color Kikuchi'])
    tags['linewidth HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    if verbose:
        print('tags[\'linewidth HOLZ\'] =', tags['linewidth HOLZ'], '# -1: linewidth according to intensity '
                                                                    '(structure factor F^2)')
    tags['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2
    if verbose:
        print('tags[\'linewidth Kikuchi\'] =', tags['linewidth Kikuchi'], '# -1: linewidth according to intensity '
                                                                          '(structure factor F^2)')

    tags['color reflections'] = 'intensity'  # 'Laue Zone'
    if verbose:
        print('tags[\'color reflections\'] =', tags['color reflections'], '#\'Laue Zone\' ')
    tags['color zero'] = 'white'  # 'None', 'white', 'blue'
    if verbose:
        print('tags[\'color zero\'] =', tags['color zero'], '#\'None\', \'white\', \'blue\' ')
    tags['color ring zero'] = 'None'  # 'Red' #'white' #, 'None'
    if verbose:
        print('tags[\'color ring zero\'] =', tags['color ring zero'], '#\'None\', \'white\', \'Red\' ')

        print('########################')
        print('# End of Example Input #')
        print('########################\n\n')
    return tags


def zone_mistilt(zone, angles):
    """ Rotation of zone axis by mistilt

    Parameters
    ----------
    zone: list or numpy array of int
        zone axis in Miller indices
    angles: ist or numpy array of float
        list of mistilt angles in degree

    Returns
    -------
    new_zone_axis: np.ndarray (3)
        new tilted zone axis

    """
    if not isinstance(angles, (np.ndarray, list)):
        raise TypeError('angles must be a list of float of length 3')
    if len(angles) != 3:
        raise TypeError('angles must be a list of float of length 3')
    if not isinstance(zone, (np.ndarray, list)):
        raise TypeError('Miller indices must be a list of int of length 3')
    alpha, beta, gamma = np.radians(angles)

    # first we rotate alpha about x axis
    c, s = np.cos(alpha), np.sin(alpha)
    rotx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # second we rotate beta about y axis
    c, s = np.cos(beta), np.sin(beta)
    roty = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # third we rotate gamma about z-axis
    c, s = np.cos(gamma), np.sin(gamma)
    rotz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return np.dot(np.dot(np.dot(zone, rotx), roty), rotz)


def get_symmetry(unit_cell, base, atoms, verbose=True):
    """
    Symmetry analysis with spglib

    spglib must be installed
    """
    if _spglib_present:
        if verbose:
            print('#####################')
            print('# Symmetry Analysis #')
            print('#####################')

        atomic_number = []
        for i in range(len(atoms)):
            a = atoms[i]
            b = base[i]
            atomic_number.append(electronFF[a]['Z'])
            if verbose:
                print(f'{i+1}: {atomic_number[i]} = {2} : [{base[i][0]:.2f}, {base[i][1]:.2f}, {base[i][2]:.2f}]')

        lattice = (unit_cell, base, atomic_number)
        spgroup = spglib.get_spacegroup(lattice)
        sym = spglib.get_symmetry(lattice)

        if verbose:
            print("  Spacegroup  is %s." % spgroup)
            print('  Crystal has {0} symmetry operation'.format(sym['rotations'].shape[0]))

        p_lattice, p_positions, p_numbers = spglib.find_primitive(lattice, symprec=1e-5)
        print("\n########################\n #Basis vectors of primitive Cell:")
        for i in range(3):
            print('[{0:.4f}, {1:.4f}, {2:.4f}]'.format(p_lattice[i][0], p_lattice[i][1], p_lattice[i][2]))

        print('There {0} atoms and {1} species in primitive unit cell:'.format(len(p_positions), p_numbers))
    else:
        print('spglib is not installed')

    return True


def ball_and_stick(tags, extend=1, max_bond_length=0.):
    """Calculates the data to plot a ball and stick model

    Parameters
    ----------
    tags: dict
        dictionary containing the 'unit_cell', 'base' and 'elements' tags.

    extend: 1 or 3 integers
        The *extend* argument scales the effective cell in which atoms
        will be included. It must either be a list of three integers or a single
        integer scaling all 3 directions.  By setting this value to one,
        all  corner and edge atoms will be included in the returned cell.
        This will of cause make the returned cell non-repeatable, but this is
        very useful for visualisation.

    max_bond_length: 1 float
        The max_bond_length argument defines the distance for which a bond will be shown.
        If max_bond_length is zero, the tabulated atom radii will be used.

    Returns
    -------
    corners,balls, Z, bonds: lists
        These lists can be used to plot the
    unit cell:
        for x, y, z in corners:
            l=mlab.plot3d( x,y,z, tube_radius=0.002)
    bonds:
        for x, y, z in bonds:
            mlab.plot3d( x,y,z, tube_radius=0.02)
    and atoms:
        for i,atom in enumerate(balls):
            mlab.points3d(atom[0],atom[1],atom[2],
                          scale_factor=0.1,##ks.vdw_radii[Z[i]]/5,
                          resolution=20,
                          color=tuple(ks.jmol_colors [Z[i]]),
                          scale_mode='none')

    Please note that you'll need the *Z* list for coloring, or for radii that depend on elements
    """

    # Check in which form extend is given
    if isinstance(extend, int):
        extend = [extend]*3

    extend = np.array(extend, dtype=int)

    # Make the x,y, and z multiplicators
    if len(extend) == 3:
        x = np.linspace(0, extend[0], extend[0]+1)
        y = np.linspace(0, extend[1], extend[1]+1)
        z = np.linspace(0, extend[2], extend[2]+1)
    else:
        print('wrong extend parameter')
        return

    # Check whether this is the right kind of dictionary
    if 'unit_cell' not in tags:
        return
    cell = tags['unit_cell']

    # Corners and Outline of unit cell
    h = (0, 1)
    corner_vectors = np.dot(np.array(list(itertools.product(h, h, h))), cell)
    trace = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 7], [6, 7], [6, 4], [1, 5], [2, 6], [3, 7]]
    corners = []
    for s, e in trace:
        corners.append([*zip(corner_vectors[s], corner_vectors[e])])

    # ball position and elements in supercell
    super_cell = np.array(list(itertools.product(x, y, z)))  # all evaluated Miller indices

    pos = np.add(super_cell, np.array(tags['base'])[:, np.newaxis])

    atomic_number = []
    for i in range(len(tags['elements'])):
        atomic_number.append(electronFF[tags['elements'][i]]['Z'])

    # List of bond lengths taken from electronFF database below
    bond_lengths = []
    for atom in tags['elements']:
        bond_lengths.append(electronFF[atom]['bond_length'][0])

    # extend list of atomic numbers
    zpp = []
    for z in atomic_number:
        zpp.append([z]*pos.shape[1])
    zpp = np.array(zpp).flatten()

    # reshape supercell atom positions
    pos = pos.reshape((pos.shape[0]*pos.shape[1], pos.shape[2]))

    # Make a mask that excludes all atoms outside of super cell
    maskz = pos[:, 2] <= extend[2]
    masky = pos[:, 1] <= extend[1]
    maskx = pos[:, 0] <= extend[0]
    mask = np.logical_and(maskx, np.logical_and(masky, maskz))   # , axis=1)

    # Only use balls and elements inside super cell
    balls = np.dot(pos[mask], cell)
    atomic_number = zpp[mask]

    # Get maximum bond length from list of bond -lengths taken from electronFF database
    if max_bond_length == 0:
        max_bond_length = np.median(bond_lengths)/5
    # Check nearest neighbours within max_bond_length
    tree = spatial.KDTree(balls)
    nearest_neighbours = np.array(tree.query(balls, k=8, distance_upper_bound=max_bond_length))

    # Calculate bonds
    bonds = []
    bond_indices = []
    for i in range(nearest_neighbours.shape[1]):
        for j in range(nearest_neighbours.shape[2]):
            if nearest_neighbours[0, i, j] < max_bond_length:
                if nearest_neighbours[0, i, j] > 0:
                    # print(atoms[i],atoms[int(bonds[1,i,j])],bonds[:,i,j])
                    bonds.append([*zip(balls[i], balls[int(nearest_neighbours[1, i, j])])])
                    bond_indices.append([i, int(nearest_neighbours[1, i, j])])
    return corners, balls, atomic_number, bonds


def plot_unitcell_mayavi(tags):
    """Makes a 3D plot of crystal structure

    Parameters
    ----------
    tags: dict
        Dictionary with tags: 'unit_cell, 'elements', 'base'

    Returns
    -------
    3D plot

    Dependencies
    ------------
    ball_and_stick function of KinsCat
    mlab of mayavi
    """

    from mayavi import mlab

    # Make sure "max_bond_length" and "extend" variables are initialized
    if 'max_bond_length' not in tags:
        max_bond_length = 0.
    else:
        max_bond_length = tags['max_bond_length']

    if 'extend' not in tags:
        extend = 1
    else:
        extend = tags['extend']

    # get balls, sticks and atomic numbers for colors and sizes
    corners, balls, atomic_number, bonds = ball_and_stick(tags, extend=extend, max_bond_length=max_bond_length)

    print('Now plotting')
    fig = mlab.figure(1, bgcolor=(0, 0, 0), size=(350, 350))

    mlab.clf()  # clear figure

    # parallel projection
    mlab.gcf().scene.parallel_projection = True
    mlab.gcf().scene.camera.parallel_scale = 5

    # plot unit cell
    for x, y, z in corners:
        ll = mlab.plot3d(x, y, z, tube_radius=0.002)

    # plot bonds as sticks
    for x, y, z in bonds:
        mlab.plot3d(x, y, z, tube_radius=0.02)

    # plot atoms
    for i, atom in enumerate(balls):
        mlab.points3d(atom[0], atom[1], atom[2],
                      scale_factor=0.1,  # ks.vdw_radii[Z[i]]/50,
                      resolution=20,
                      color=tuple(jmol_colors[atomic_number[i]]),
                      scale_mode='none')

    # parallel projection
    mlab.gcf().scene.parallel_projection = True
    mlab.gcf().scene.camera.parallel_scale = 5
    # show plot
    mlab.show()


def plot_unitcell(tags):
    """
    Simple plot of unit cell
    """

    if 'max_bond_length' not in tags:
        max_bond_length = 0.
    else:
        max_bond_length = tags['max_bond_length']

    if 'extend' not in tags:
        extend = 1
    else:
        extend = tags['extend']

    corners, balls, atomic_number, bonds = ball_and_stick(tags, extend=extend, max_bond_length=max_bond_length)

    maximum_position = balls.max()*1.05
    maximum_x = balls[:, 0].max()
    maximum_y = balls[:, 1].max()
    maximum_z = balls[:, 2].max()

    balls = balls - [maximum_x/2, maximum_y/2, maximum_z/2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # draw unit_cell
    for x, y, z in corners:
        ax.plot3D(x-maximum_x/2, y-maximum_y/2, z-maximum_z/2, color="blue")

    # draw bonds
    for x, y, z in bonds:
        ax.plot3D(x-maximum_x/2, y-maximum_y/2, z-maximum_z/2, color="black", linewidth=4)  # , tube_radius=0.02)

    # draw atoms
    for i, atom in enumerate(balls):
        ax.scatter(atom[0], atom[1], atom[2],
                   color=tuple(jmol_colors[atomic_number[i]]),
                   alpha=1.0, s=50)
    maximum_position = balls.max()*1.05
    ax.set_proj_type('ortho')

    ax.set_zlim(-maximum_position/2, maximum_position/2)
    ax.set_ylim(-maximum_position/2, maximum_position/2)
    ax.set_xlim(-maximum_position/2, maximum_position/2)

    if 'name' in tags:
        ax.set_title(tags['name'])

    ax.set_xlabel('x [nm]')
    ax.set_ylabel('y [nm]')
    ax.set_zlabel('z [nm]')


# The metric tensor of the lattice.
def metric_tensor(matrix):
    """
    The metric tensor of the lattice.

    Usage:
        metric_tensor(unit_cell)
    """
    metric_tensor2 = np.dot(matrix, matrix.T)
    return metric_tensor2


def vector_norm(g):
    """ Length of vector

    depreciated - use np.linalg.norm
    """
    g = np.array(g)
    return np.sqrt(g[:, 0]**2+g[:, 1]**2+g[:, 2]**2)


def make_pretty_labels(hkls, hex_label=False):
    """Make pretty labels

    Parameters
    ----------
    hkls: np.ndarray
        a numpy array with all the Miller indices to be labeled
    hex_label: boolean - optional
        if True this will make for Miller indices.

    Returns
    -------
    hkl_label: list
        list of labels in Latex format
    """
    hkl_label = []
    for i in range(len(hkls)):
        h, k, l = np.array(hkls)[i]

        if h < 0:
            h_string = r'[$\bar {'+str(int(-h))+'},'
        else:
            h_string = r'[$\bar {'+str(int(h))+'},'
        if k < 0:
            k_string = r'\bar {'+str(int(-k))+'},'
        else:
            k_string = str(int(k))+','
        if hex_label:
            ii = -(h+k)
            if ii < 0:
                k_string = k_string + r'\bar {'+str(int(-ii))+'},'
            else:
                k_string = k_string + str(int(ii))+','
        if l < 0:
            l_string = r'\bar {'+str(int(-l))+'} $]'
        else:
            l_string = str(int(l))+'} $]'
        label = h_string+k_string+l_string
        hkl_label.append(label)
    return hkl_label


def get_wavelength(e0):
    """
    Calculates the relativistic corrected de Broglie wavelength of an electron in nm

    Input:
    ------
        acceleration voltage in volt
    Output:
    -------
        wave length in nm
    """
    if not isinstance(e0, (int, float)):
        raise TypeError('Acceleration voltage has to be a real number')
    eV = const.e * e0
    return const.h/np.sqrt(2*const.m_e*eV*(1+eV/(2*const.m_e*const.c**2)))*10**9


# Determine rotation matrix to tilt zone axis onto z-axis
# We determine spherical coordinates to do that
def get_rotation_matrix(zone, verbose=False):
    """Calculates the rotation matrix to rotate the zone axis parallel to the cartasian z-axis.

     We use spherical coordinates to first rotate around the z-axis and then around the y-axis.
     This makes it easier to apply additional tilts, than to use the cross product to determine a single rotation
     axis (Rodrigues Formula)

     We start from the dot product of zone axis  and unit cell will accomplish that.


     Parameters
     ----------
     zone: list of int or np.ndarray of length 3
        axis has to be in cartesian coordinates.

    Returns
    -------
    rotation_matrix: np.ndarray(3,3)
    theta: float (degrees)
    phi: float (degrees)
    """

    # spherical coordinates of zone
    zone = np.array(zone)
    r = np.sqrt((zone*zone).sum())
    theta = np.arccos(zone[2]/r)
    if zone[0] < 0:
        theta = -theta
    if zone[0] == 0:
        phi = np.pi/2
    else:
        phi = (np.arctan(zone[1]/zone[0]))

    if verbose:
        print('Rotation theta ', np.degrees(theta), ' phi ', np.degrees(phi))
    # unit = np.array([[1, 0, 0],[0,1, 0],[0, 0,1]])

    # first we rotate phi about z-axis
    c, s = np.cos(phi), np.sin(phi)
    rotz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # second we rotate theta about y axis
    c, s = np.cos(theta), np.sin(theta)
    roty = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # the rotation now makes zone-axis coincide with plane normal
    return np.dot(rotz, roty), np.degrees(theta), np.degrees(phi)


def check_sanity(tags):
    """
    Check sanity of input parameters
    """
    stop = False
    if 'unit_cell' not in tags:
        print(' No unit_cell defined')
        stop = True
    elif 'base' not in tags:
        print(' No base defined')
        stop = True
    elif 'elements' not in tags:
        print(' No atoms defined')
        stop = True
    elif 'acceleration_voltage_V' not in tags:
        print(' No acceleration_voltage_V defined')
        stop = True
    elif 'zone_hkl' not in tags:
        print(' No zone_hkl defined')
        stop = True
    elif 'Sg_max' not in tags:
        print(' No Sg_max defined')
        stop = True
    elif 'hkl_max' not in tags:
        print(' No hkl_max defined')
        stop = True

    elif 'crystal_name' not in tags:
        tags['crystal_name'] = 'undefined'
        print('tags[\'crystal\'] = \'undefined\'')
    elif 'mistilt' not in tags:
        tags['mistilt'] = [0., 0., 0.]
        print('tags[\'mistilt\'] = [0., 0., 0.]')
    elif 'convergence_angle_mrad' not in tags:
        tags['convergence_angle_mrad'] = 0.
        print('tags[\'convergence_angle_mrad\'] = 0')

    return not stop


def ring_pattern_calculation(tags, verbose=False):
    """
    Calculate the ring diffraction pattern of a crystal structure

    Parameters
    ----------
    tags: dict
        dictionary of crystal structure

    Returns
    -------
    tags: dict
        dictionary with diffraction information added

    """
    # Reciprocal Lattice
    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    reciprocal_unit_cell = np.linalg.inv(tags['unit_cell']).T  # transposed of inverted unit_cell

    # INPUT
    hkl_max = 7   # maximum allowed Miller index

    acceleration_voltage = 200.0*1000.0  # V
    wave_length = get_wavelength(acceleration_voltage)

    h = np.linspace(-hkl_max, hkl_max, 2*hkl_max+1)    # all to be evaluated single Miller Index
    hkl = np.array(list(itertools.product(h, h, h)))  # all to be evaluated Miller indices
    g_hkl = np.dot(hkl, reciprocal_unit_cell)

    # Calculate Structure Factors

    structure_factors = []

    base = np.dot(tags['base'], tags['unit_cell'])  # transformation from relative to Cartesian coordinates
    for j in range(len(g_hkl)):
        F = 0
        for b in range(len(base)):
            # Atomic form factor for element and momentum change (g vector)
            f = feq(tags['elements'][b], np.linalg.norm(g_hkl[j]))
            F += f * np.exp(-2*np.pi*1j*(g_hkl[j]*base[b]).sum())
        structure_factors.append(F)
    F = structure_factors = np.array(structure_factors)

    # Allowed reflections have a non zero structure factor F (with a  bit of numerical error)
    allowed = np.absolute(structure_factors) > 0.001

    distances = np.linalg.norm(g_hkl, axis=1)

    if verbose:
        print(f' Of the evaluated {hkl.shape[0]} Miller indices {allowed.sum()} are allowed. ')
    # We select now all the
    zero = distances == 0.
    allowed = np.logical_and(allowed, np.logical_not(zero))

    F = F[allowed]
    g_hkl = g_hkl[allowed]
    hkl = hkl[allowed]
    distances = distances[allowed]

    sorted_allowed = np.argsort(distances)

    distances = distances[sorted_allowed]
    hkl = hkl[sorted_allowed]
    F = F[sorted_allowed]

    # How many have unique distances and what is their multiplicity
    unique, indices = np.unique(distances, return_index=True)

    if verbose:
        print(f' Of the {allowed.sum()} allowed Bragg reflections there are {len(unique)} families of reflections.')

    intensity = np.absolute(F[indices]**2*(np.roll(indices, -1)-indices))
    if verbose:
        print('\n index \t  hkl \t      1/d [1/nm]       d [pm]     F     multip.  intensity')
    family = []
    out_tags = {}
    for j in range(len(unique)-1):
        i = indices[j]
        i2 = indices[j+1]
        family.append(hkl[i+np.argmax(hkl[i:i2].sum(axis=1))])
        index = '{'+f'{family[j][0]:.0f} {family[j][1]:.0f} {family[j][2]:.0f}'+'}'
        if verbose:
            print(f'{i:3g}\t {index} \t  {distances[i]:.2f}  \t {1/distances[i]*1000:.0f} \t {np.absolute(F[i]):.2f},'
                  f' \t  {indices[j+1]-indices[j]:3g} \t {intensity[j]:.2f}')
        out_tags[index] = {}
        out_tags[index]['reciprocal_distance'] = distances[i]
        out_tags[index]['real_distance'] = 1/distances[i]
        out_tags[index]['F'] = np.absolute(F[i])
        out_tags[index]['multiplicity'] = indices[j+1]-indices[j]
        out_tags[index]['intensity'] = intensity[j]
    return out_tags


def kinematic_scattering(tags, verbose=False):
    """
        All kinematic scattering calculation

        Calculates Bragg spots, Kikuchi lines, excess, and deficient HOLZ lines

        Parameters
        ----------
        tags: dict
            dictionary with crystal structure:
            'unit_cell', 'base' 'elements'
            and with experimental parameters:
            'acceleration_voltage_V', 'zone_hkl', 'Sg_max', 'hkl_max'
            Optional parameters are:
            'mistilt', convergence_angle_mrad', and 'crystal_name'
        verbose:  boolean
            True will give extended output of the calculation

        Returns
        -------
        dict:
            There are three sub_dictionaries:
            ['allowed'], ['forbidden'], and ['HOLZ']
            ['allowed'] and ['forbidden'] dictionaries contain:
                ['Sg'], ['hkl'], ['g'], ['structure factor'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], ['HHOLZ'], ['label'], and ['Laue_zone']
            the ['HOLZ'] dictionary contains:
                ['slope'], ['distance'], ['theta'], ['g deficient'], ['g excess'], ['hkl'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], and  ['HHOLZ']
            Please note that the Kikuchi lines are the HOLZ lines of ZOLZ

            There are also a few parameters stored in the main dictionary:
                ['wave_length_nm'], ['reciprocal_unit_cell'], ['inner_potential_V'], ['incident_wave_vector'],
                ['volume'], ['theta'], ['phi'], and ['incident_wave_vector_vacuum']
    """

    if not check_sanity(tags):
        print('Input is not complete, stopping')
        print('Try \'example()\' for example input')
        return

    tags['wave_length_nm'] = get_wavelength(tags['acceleration_voltage_V'])

    # ###########################################
    # reciprocal_unit_cell
    # ###########################################
    unit_cell = np.array(tags['unit_cell'])
    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    reciprocal_unit_cell = np.linalg.inv(unit_cell).T  # transposed of inverted unit_cell
    tags['reciprocal_unit_cell'] = reciprocal_unit_cell

    if verbose:
        print('reciprocal_unit_cell')
        print(np.round(reciprocal_unit_cell, 3))

    # ###########################################
    # Incident wave vector K0 in vacuum and material
    # ###########################################

    # Incident wave vector K0 in vacuum and material
    U0 = 0
    for i in range(len(tags['elements'])):
        a = tags['elements'][i]
        U0 += feq(a, 0)*0.023933754

    tags['volume'] = np.linalg.det(tags['unit_cell'])
    volume = tags['volume']*1000  # Needs to be in Angstrom for form factors

    AngstromConversion = 1.0e10  # So [1A (in m)] * AngstromConversion = 1
    NanometerConversion = 1.0e9

    ScattFacToVolts = (const.h**2)*(AngstromConversion**2)/(2*np.pi*const.m_e*const.e)*volume
    U0 = U0*ScattFacToVolts
    tags['inner_potential_A'] = U0
    tags['inner_potential_V'] = U0*ScattFacToVolts
    if verbose:
        print('The inner potential is {0:.3f}kV'.format(tags['inner_potential_V']/1000))

    # Calculating incident wave vector magnitude 'K0' in material
    wl = tags['wave_length_nm']
    tags['incident_wave_vector_vacuum'] = 1/wl

    K0 = tags['incident_wave_vector'] = np.sqrt(1/wl**2 - (U0/volume*100.))  # 1/nm

    tags['convergence_angle_nm-1'] = K0*np.sin(tags['convergence_angle_mrad']/1000.)

    if verbose:
        print('Magnitude of incident wave vector in material {0:.1f} 1/nm and vacuum {1:.1f} 1/nm'.format(K0, 1/wl))
        print('The convergence angle of {0}mrad = {1:.2f} 1/nm'.format(tags['convergence_angle_mrad'],
                                                                       tags['convergence_angle_nm-1']))

    # ############
    # Rotate
    # ############

    # first we take care of mistilt: zone axis is then in fractional Miller indices
    zone = tags['zone'] = zone_mistilt(tags['zone_hkl'], tags['mistilt'])

    # zone axis in global coordinate system
    zone_vector = np.dot(zone, reciprocal_unit_cell)

    rotation_matrix, theta, phi = get_rotation_matrix(zone_vector, verbose=False)

    if verbose:
        print('Rotation angles are {0:.1f} deg and {1:.1f} deg'.format(theta, phi))
    tags['theta'] = theta
    tags['phi'] = phi

    # rotate incident wave vector
    w_vector = np.dot(zone_vector, rotation_matrix)
    K0_unit_vector = w_vector / np.linalg.norm(w_vector)  # incident unit wave vector
    K0_vector = K0_unit_vector*K0                         # incident  wave vector

    if verbose:
        print('Center of Ewald sphere ', K0_vector)

    # #######################
    # Find all Miller indices whose reciprocal point lays near the Ewald sphere with radius K0
    # within a maximum excitation error Sg
    # #######################

    hkl_max = tags['hkl_max']
    Sg_max = tags['Sg_max']  # 1/nm  maximum allowed excitation error

    h = np.linspace(-hkl_max, hkl_max, 2*hkl_max+1)  # all evaluated single Miller Indices
    hkl = np.array(list(itertools.product(h, h, h)))  # all evaluated Miller indices
    g = np.dot(hkl, reciprocal_unit_cell)  # all evaluated reciprocal_unit_cell points
    g_norm = np.linalg.norm(g, axis=1)   # length of all vectors
    not_zero = g_norm > 0
    g = g[not_zero]  # zero reflection will make problems further on, so we exclude it.
    g_norm = g_norm[not_zero]
    hkl = hkl[not_zero]
    g_non_rot = g
    g = np.dot(g, rotation_matrix)

    # #######################
    # Calculate excitation errors for all reciprocal_unit_cell points
    # #######################

    # Zuo and Spence, 'Adv TEM', 2017 -- Eq 3:14
    # S=(K0**2-np.linalg.norm(g - K0_vector, axis=1)**2)/(2*K0)
    gMz = g - K0_vector

    in_sqrt = gMz[:, 2]**2 + np.linalg.norm(gMz, axis=1)**2 - K0**2
    in_sqrt[in_sqrt < 0] = 0.
    S = -gMz[:, 2] - np.sqrt(in_sqrt)

    # #######################
    # Determine reciprocal_unit_cell points with excitation error less than the maximum allowed one: Sg_max
    # #######################

    reflections = abs(S) < Sg_max   # This is now a boolean array with True for all possible reflections

    Sg = S[reflections]
    g_hkl = g[reflections]
    g_hkl_non_rot = g_non_rot[reflections]
    hkl = hkl[reflections]
    g_norm = g_norm[reflections]

    if verbose:
        print('Of the {0} tested reciprocal_unit_cell points, {1} have an excitation error less than {2:.2f} 1/nm'.
              format(len(g), len(g_hkl), Sg_max))

    # #################################
    # Calculate Structure Factors
    # ################################

    structure_factors = []
    """for j  in range(len(g_hkl)):
        F = 0
        for b in range(len(tags['base'])):
            f = feq(tags['elements'][b],np.linalg.norm(g_hkl[j]))
            #F += f * np.exp(-2*np.pi*1j*(hkl*tags['base'][b]).sum()) # may only work for cubic Gerd
            F += f * np.exp(-2*np.pi*1j*(g_hkl_non_rot*np.dot(tags['base'][b],unit_cell)).sum())


        structure_factors.append(F)

    F = structure_factors = np.array(structure_factors)
    """
    base = np.dot(tags['base'], tags['unit_cell'])  # transformation from relative to Cartesian coordinates
    for j in range(len(g_hkl)):
        F = 0
        for b in range(len(base)):
            f = feq(tags['elements'][b], g_norm[j])  # Atomic form factor for element and momentum change (g vector)
            F += f * np.exp(-2*np.pi*1j*(g_hkl_non_rot[j]*base[b]).sum())
        structure_factors.append(F)
    F = structure_factors = np.array(structure_factors)

    # ###########################################
    # Sort reflection in allowed and forbidden #
    # ###########################################

    allowed = np.absolute(F) > 0.000001    # allowed within numerical error

    if verbose:
        print('Of the {0} possible reflection {1} are allowed.'.format(hkl.shape[0], allowed.sum()))

    # information of allowed reflections
    Sg_allowed = Sg[allowed]
    hkl_allowed = hkl[allowed][:]
    g_allowed = g_hkl[allowed, :]
    F_allowed = F[allowed]
    g_norm_allowed = g_norm[allowed]

    tags['allowed'] = {}
    tags['allowed']['Sg'] = Sg_allowed
    tags['allowed']['hkl'] = hkl_allowed
    tags['allowed']['g'] = g_allowed
    tags['allowed']['structure factor'] = F_allowed

    # information of forbidden reflections
    forbidden = np.logical_not(allowed)
    Sg_forbidden = Sg[forbidden]
    hkl_forbidden = hkl[forbidden]
    g_forbidden = g_hkl[forbidden]

    tags['forbidden'] = {}
    tags['forbidden']['Sg'] = Sg_forbidden
    tags['forbidden']['hkl'] = hkl_forbidden
    tags['forbidden']['g'] = g_forbidden

    # ##########################
    # Make pretty labels
    # ##########################
    hkl_label = make_pretty_labels(hkl_allowed)
    tags['allowed']['label'] = hkl_label

    # hkl_label = make_pretty_labels(hkl_forbidden)
    # tags['forbidden']['label'] = hkl_label

    # ###########################
    # Calculate Intensities (of allowed reflections)
    # ###########################

    intensities = np.absolute(F_allowed)**2

    tags['allowed']['intensities'] = intensities

    # ###########################
    # Calculate Laue Zones (of allowed reflections)
    # ###########################
    # Below is the expression given in most books.
    # However, that would only work for orthogonal crystal systems
    # Laue_Zone = abs(np.dot(hkl_allowed,tags['zone_hkl']))  # works only for orthogonal systems

    # This expression works for all crystal systems
    # Remember we have already tilted, and so the dot product is trivial and gives only the z-component.
    length_zone_axis = np.linalg.norm(np.dot(tags['zone_hkl'], tags['unit_cell']))
    Laue_Zone = abs(np.floor(g_allowed[:, 2]*length_zone_axis+0.5))

    tags['allowed']['Laue_Zone'] = Laue_Zone

    ZOLZ_forbidden = abs(np.floor(g_forbidden[:, 2]*length_zone_axis+0.5)) == 0

    tags['forbidden']['Laue_Zone'] = ZOLZ_forbidden
    ZOLZ = Laue_Zone == 0
    FOLZ = Laue_Zone == 1
    SOLZ = Laue_Zone == 2
    HOLZ = Laue_Zone > 0
    HOLZp = Laue_Zone > 2

    tags['allowed']['ZOLZ'] = ZOLZ
    tags['allowed']['FOLZ'] = FOLZ
    tags['allowed']['SOLZ'] = SOLZ
    tags['allowed']['HOLZ'] = HOLZ
    tags['allowed']['HOLZ_plus'] = tags['allowed']['HHOLZ'] = HOLZp

    if verbose:
        print(' There are {0} allowed reflections in the zero order Laue Zone'.format(ZOLZ.sum()))
        print(' There are {0} allowed reflections in the first order Laue Zone'.format((Laue_Zone == 1).sum()))
        print(' There are {0} allowed reflections in the second order Laue Zone'.format((Laue_Zone == 2).sum()))
        print(' There are {0} allowed reflections in the other higher order Laue Zones'.format((Laue_Zone > 2).sum()))

    if verbose == 2:
        print(' hkl  \t Laue zone \t Intensity (*1 and \t log) \t length \n')
        for i in range(len(hkl_allowed)):
            print(' {0} \t {1} \t {2:.3f} \t  {3:.3f} \t  {4:.3f}   '.format(hkl_allowed[i], g_allowed[i],
                                                                             intensities[i], np.log(intensities[i]+1),
                                                                             g_norm_allowed[i]))

    # ##########################
    # Dynamically Activated forbidden reflections
    # ##########################

    double_diffraction = (np.sum(np.array(list(itertools.combinations(hkl_allowed[ZOLZ], 2))), axis=1))

    dynamical_allowed = []
    still_forbidden = []
    for i, hkl in enumerate(hkl_forbidden):
        if ZOLZ_forbidden[i]:
            if hkl.tolist() in double_diffraction.tolist():
                dynamical_allowed.append(i)
            else:
                still_forbidden.append(i)
    tags['forbidden']['dynamically_activated'] = dynamical_allowed
    tags['forbidden']['forbidden'] = dynamical_allowed
    if verbose:
        print('Length of zone axis vector in real space {0} nm'.format(np.round(length_zone_axis, 3)))
        print(f'There are {len(dynamical_allowed)} forbidden but dynamical activated diffraction spots:')
        # print(tags['forbidden']['hkl'][dynamical_allowed])

    # ###################################
    # Calculate HOLZ and Kikuchi Lines #
    # ###################################

    # Dynamic Correction

    # Equation Spence+Zuo 3.86a
    gamma_1 = - 1./(2.*K0) * (intensities / (2.*K0*Sg_allowed)).sum()
    # print('gamma_1',gamma_1)

    # Equation Spence+Zuo 3.84
    Kg = K0 - K0*gamma_1/(g_allowed[:, 2]+1e-15)
    Kg[ZOLZ] = K0

    # print(Kg, Kg.shape)

    # Calculate angle between K0 and deficient cone vector
    # For dynamic calculations K0 is replaced by Kg
    Kg[:] = K0
    dtheta = np.arcsin(g_norm_allowed/Kg/2.)-np.arcsin(np.abs(g_allowed[:, 2])/g_norm_allowed)

    # calculate length of distance of deficient cone to K0 in ZOLZ plane
    gd_length = 2*np.sin(dtheta/2)*K0

    # Calculate nearest point of HOLZ and Kikuchi lines
    g_closest = g_allowed.copy()
    g_closest = g_closest*(gd_length/np.linalg.norm(g_closest, axis=1))[:, np.newaxis]

    g_closest[:, 2] = 0.

    # calculate and save line in Hough space coordinates (distance and theta)
    slope = g_closest[:, 0]/(g_closest[:, 1]+1e-10)
    distance = gd_length
    theta = np.arctan2(g_allowed[:, 0], g_allowed[:, 1])

    tags['HOLZ'] = {}
    tags['HOLZ']['slope'] = slope
    # a line is now given by

    tags['HOLZ']['distance'] = distance
    tags['HOLZ']['theta'] = theta

    tags['HOLZ']['g deficient'] = g_closest
    tags['HOLZ']['g excess'] = g_closest+g_allowed

    tags['HOLZ']['ZOLZ'] = ZOLZ
    tags['HOLZ']['HOLZ'] = HOLZ
    tags['HOLZ']['FOLZ'] = FOLZ
    tags['HOLZ']['SOLZ'] = SOLZ
    tags['HOLZ']['HHOLZ'] = HOLZp  # even higher HOLZ

    tags['HOLZ']['hkl'] = tags['allowed']['hkl']
    tags['HOLZ']['intensities'] = intensities

    if verbose:
        print('KinsCat\'s  \"Kinematic_Scattering\" finished')


def plotSAED(tags, gray=False):
    """
    Plot SAED Pattern of single crystal
    """

    saed = tags.copy()
    saed['convergence_angle_nm-1'] = 0

    saed['background'] = 'white'   # 'white'  'grey'
    saed['color map'] = 'plasma'  # ,'cubehelix'#'Greys'#'plasma'
    saed['color reflections'] = 'ZOLZ'

    if gray:
        saed['color map'] = 'gray'
        saed['background'] = '#303030'  # 'darkgray'
        saed['color reflections'] = 'intensity'
    saed['plot HOLZ'] = 0
    saed['plot HOLZ excess'] = 0
    saed['plot Kikuchi'] = 0
    saed['plot reflections'] = 1

    saed['label HOLZ'] = 0
    saed['label Kikuchi'] = 0
    saed['label reflections'] = 0

    saed['label color'] = 'white'
    saed['label size'] = 10

    saed['color Laue Zones'] = ['red',  'blue', 'green', 'blue', 'green']  # , 'green', 'red'] #for OLZ give a sequence
    saed['color zero'] = 'red'  # 'None' #'white'
    saed['color ring zero'] = 'None'  # 'Red' #'white' #, 'None'
    saed['width ring zero'] = 2

    plot_diffraction_pattern(saed, True)


def plotKikuchi(tags, grey=False):
    """
    Plot Kikuchi Pattern
    """
    Kikuchi = tags.copy()

    Kikuchi['background'] = 'black'   # 'white'  'grey'
    Kikuchi['color map'] = 'plasma'   # ,'cubehelix'#'Greys'#'plasma'
    Kikuchi['color reflections'] = 'intensity'

    Kikuchi['plot HOLZ'] = 0
    Kikuchi['plot HOLZ excess'] = 0
    Kikuchi['plot Kikuchi'] = 1
    Kikuchi['plot reflections'] = 1

    Kikuchi['label HOLZ'] = 0
    Kikuchi['label Kikuchi'] = 0
    Kikuchi['label reflections'] = 0

    Kikuchi['label color'] = 'white'
    Kikuchi['label size'] = 10

    Kikuchi['color Kikuchi'] = 'green'
    Kikuchi['linewidth HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    Kikuchi['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2
    Kikuchi['color Laue Zones'] = ['red',  'blue', 'green', 'blue', 'green']  # , 'green', 'red']
    # #for OLZ give a sequence
    Kikuchi['color zero'] = 'white'  # 'None' #'white'
    Kikuchi['color ring zero'] = 'None'  # 'Red' #'white' #, 'None'
    Kikuchi['width ring zero'] = 2

    plot_diffraction_pattern(Kikuchi, True)


def plotHOLZ(tags, grey=False):
    """
    Plot HOLZ Pattern
    """
    holz = tags.copy()

    holz['background'] = 'black'   # 'white'  'grey'
    holz['color map'] = 'plasma'   # ,'cubehelix'#'Greys'#'plasma'
    holz['color reflections'] = 'intensity'

    holz['plot HOLZ'] = 1
    holz['plot HOLZ excess'] = 0
    holz['plot Kikuchi'] = 1
    holz['plot reflections'] = 0

    holz['label HOLZ'] = 0
    holz['label Kikuchi'] = 0
    holz['label reflections'] = 0

    holz['label color'] = 'white'
    holz['label size'] = 10

    holz['color Kikuchi'] = 'green'
    holz['linewidth HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    holz['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2

    holz['color Laue Zones'] = ['red',  'blue', 'green']  # , 'green', 'red'] #for OLZ give a sequence
    holz['color zero'] = 'white'   # 'None' #'white'
    holz['color ring zero'] = 'Red'  # 'Red' #'white' #, 'None'
    holz['width ring zero'] = 2

    plot_diffraction_pattern(holz, True)


def plotCBED(tags, grey=False):
    """
    Plot CBED Pattern
    """
    cbed = tags.copy()

    cbed['background'] = 'black'   # 'white'  'grey'
    cbed['color map'] = 'plasma'   # ,'cubehelix'#'Greys'#'plasma'
    cbed['color reflections'] = 'intensity'

    cbed['plot HOLZ'] = 1
    cbed['plot HOLZ excess'] = 1
    cbed['plot Kikuchi'] = 1
    cbed['plot reflections'] = 1

    cbed['label HOLZ'] = 0
    cbed['label Kikuchi'] = 0
    cbed['label reflections'] = 0

    cbed['label color'] = 'white'
    cbed['label size'] = 10

    cbed['color Kikuchi'] = 'green'
    cbed['linewidth HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    cbed['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2

    cbed['color reflections'] = 'intensity'

    cbed['color Laue Zones'] = ['red',  'blue', 'green']   # , 'green', 'red'] #for OLZ give a sequence
    cbed['color zero'] = 'white'  # 'None' #'white'
    cbed['color ring zero'] = 'Red'  # 'Red' #'white' #, 'None'
    cbed['width ring zero'] = 2

    plot_diffraction_pattern(cbed, True)


# #######################
# Plot HOLZ Pattern #
# #######################
def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles.

    Similar to plt.scatter, but the size of circles are in data scale.

    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : ~matplotlib.collections.PathCollection

    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()

    License
    -------
    This code is under [The BSD 3-Clause License] (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_)
               for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def plot_diffraction_pattern(tags, grey=False):
    """
    Plot any diffraction pattern based on content in dictionary
    """
    if 'plot_rotation' not in tags:
        tags['plot_rotation'] = 0.

    if 'new_plot' not in tags:
        tags['new_plot'] = True

    if tags['new_plot']:
        fig = plt.figure()
        ax = fig.add_subplot(111, facecolor=tags['background'])

        diffraction_pattern(tags, grey)

        plt.axis('equal')
        if 'plot FOV' in tags:
            x = tags['plot FOV']  # in 1/nm
            plt.xlim(-x, x)
            plt.ylim(-x, x)
        plt.title(tags['crystal_name'])
        plt.show()
    else:
        diffraction_pattern(tags, grey)


def diffraction_pattern(tags, grey=False):
    """
    Determines how to plot diffraction pattern from kinematic scattering data.

    Parameters:
    -----------
    tags: dict
        dictionary of kinematic scattering data
    grey: boolean optional
        plot in gray scale

    Returns:
    --------
    tags: dict
        dictionary that now contains all information of how to plot any diffraction pattern
    plot: matplotlib figure
    """

    # Get information from dictionary
    HOLZ = tags['allowed']['HOLZ']
    ZOLZ = tags['allowed']['ZOLZ']

    Laue_Zone = tags['allowed']['Laue_Zone']

    if 'label' in tags['allowed']:
        label = tags['allowed']['label']

    angle = np.radians(tags['plot_rotation'])  # mrad
    c = np.cos(angle)
    s = np.sin(angle)
    r_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # HOLZ and Kikuchi lines coordinates in Hough space
    gd = np.dot(tags['HOLZ']['g deficient'], r_mat)
    ge = np.dot(tags['HOLZ']['g excess'], r_mat)

    points = np.dot(tags['allowed']['g'], r_mat)

    theta = tags['HOLZ']['theta']+angle

    intensity = tags['allowed']['intensities']
    radius = tags['convergence_angle_nm-1']
    tags['Bragg'] = {}
    tags['Bragg']['points'] = points
    tags['Bragg']['intensity'] = intensity
    tags['Bragg']['radius'] = radius

    if radius < 0.1:
        radiusI = 2
    else:
        radiusI = radius
    # Beginning and ends of HOLZ lines
    maxlength = radiusI * 1.3
    Hxp = gd[:, 0] + maxlength * np.cos(np.pi-theta)
    Hyp = gd[:, 1] + maxlength * np.sin(np.pi-theta)
    Hxm = gd[:, 0] - maxlength * np.cos(np.pi-theta)
    Hym = gd[:, 1] - maxlength * np.sin(np.pi-theta)
    tags['HOLZ lines'] = {}
    tags['HOLZ lines']['Hxp'] = Hxp
    tags['HOLZ lines']['Hyp'] = Hyp
    tags['HOLZ lines']['Hxm'] = Hxm
    tags['HOLZ lines']['Hym'] = Hym

    # Beginning and ends of excess HOLZ lines
    maxlength = radiusI * 0.8
    Exp = ge[:, 0] + maxlength * np.cos(np.pi-theta)
    Eyp = ge[:, 1] + maxlength * np.sin(np.pi-theta)
    Exm = ge[:, 0] - maxlength * np.cos(np.pi-theta)
    Eym = ge[:, 1] - maxlength * np.sin(np.pi-theta)
    tags['excess HOLZ lines'] = {}
    tags['excess HOLZ lines']['Exp'] = Exp
    tags['excess HOLZ lines']['Eyp'] = Eyp
    tags['excess HOLZ lines']['Exm'] = Exm
    tags['excess HOLZ lines']['Eym'] = Eym

    # Beginning and ends of HOLZ lines
    maxlength = 20
    Kxp = gd[:, 0] + maxlength * np.cos(np.pi-theta)
    Kyp = gd[:, 1] + maxlength * np.sin(np.pi-theta)
    Kxm = gd[:, 0] - maxlength * np.cos(np.pi-theta)
    Kym = gd[:, 1] - maxlength * np.sin(np.pi-theta)
    tags['Kikuchi lines'] = {}
    tags['Kikuchi lines']['Kxp'] = Kxp
    tags['Kikuchi lines']['Kyp'] = Kyp
    tags['Kikuchi lines']['Kxm'] = Kxm
    tags['Kikuchi lines']['Kym'] = Kym

    intensity_Kikuchi = intensity*4./intensity[ZOLZ].max()
    if len(intensity[tags['HOLZ']['HOLZ']]) > 1:
        intensity_HOLZ = intensity*4./intensity[tags['HOLZ']['HOLZ']].max()*.75
        tags['HOLZ lines']['intensity_HOLZ'] = intensity_HOLZ
    tags['Kikuchi lines']['intensity_Kikuchi'] = intensity_Kikuchi
    # #######
    # Plot  #
    # #######
    cms = mpl.cm
    # cm = cms.plasma#jet#, cms.gray, cms.autumn]
    cm = plt.get_cmap(tags['color map'])

    fig = plt.gcf()
    ax = plt.gca()

    if 'plot image' in tags:
        l = -tags['plot image FOV']/2+tags['plot shift x']
        r = tags['plot image FOV']/2+tags['plot shift x']
        t = -tags['plot image FOV']/2+tags['plot shift y']
        b = tags['plot image FOV']/2+tags['plot shift y']
        ax.imshow(tags['plot image'], extent=(l, r, t, b))
        print('image')

    ix = np.argsort((points**2).sum(axis=1))
    # print(tags['allowed']['hkl'][ix])
    p = points[ix]
    inten = intensity[ix]
    tags['Bragg']['points'] = p
    tags['Bragg']['intensity'] = inten

    Lauecolor = []
    for i in range(int(Laue_Zone.max())+1):
        if i < len(tags['color Laue Zones']):
            Lauecolor.append(tags['color Laue Zones'][i])
        else:
            Lauecolor.append(tags['color Laue Zones'][-1])

    if tags['plot reflections']:
        if radius < 0.1:
            if tags['color reflections'] == 'intensity':
                ax.scatter(points[:, 0], points[:, 1], c=np.log(intensity), cmap=cm, s=20)
            else:
                for i in range(len(Laue_Zone)):
                    color = Lauecolor[int(Laue_Zone[i])]
                    ax.scatter(points[i, 0], points[i, 1], c=color, cmap=cm, s=20)

            ax.scatter(0, 0, c=tags['color zero'], s=100)
            radius = 2
        else:
            ix = np.argsort((points**2).sum(axis=1))
            p = points[ix]
            inten = intensity[ix]
            if tags['color reflections'] == 'intensity':
                circles(p[:, 0], p[:, 1], s=radius, c=np.log(inten+1), cmap=cm, alpha=0.9, edgecolor='')
            else:
                for i in range(len(Laue_Zone)):
                    color = Lauecolor[int(Laue_Zone[i])]
                    circles(p[i, 0], p[i, 1], s=radius, c=color, cmap=cm, alpha=0.9, edgecolor='')

    if not tags['color zero'] == 'None':
        circle = plt.Circle((0, 0), radius, color=tags['color zero'])
        ax.add_artist(circle)

    for i in range(len(Hxp)):
        if tags['HOLZ']['HOLZ'][i]:
            color = Lauecolor[int(Laue_Zone[i])]
            if tags['plot HOLZ']:
                # plot HOLZ lines
                ax.plot((Hxp[i], Hxm[i]), (Hyp[i], Hym[i]), c=color, linewidth=intensity_HOLZ[i])
            if tags['plot HOLZ excess']:
                ax.plot((Exp[i], Exm[i]), (Eyp[i], Eym[i]), c=color, linewidth=intensity_HOLZ[i])

                if tags['label HOLZ']:  # Add indices
                    ax.text(Hxp[i], Hyp[i], label[i], fontsize=10)
                    ax.text(Exp[i], Eyp[i], label[i], fontsize=10)
        else:
            # Plot Kikuchi lines
            if tags['plot Kikuchi']:
                ax.plot((Kxp[i], Kxm[i]), (Kyp[i], Kym[i]), c=tags['color Kikuchi'], linewidth=intensity_Kikuchi[i])
                if tags['label Kikuchi']:  # Add indices
                    ax.text(Kxp[i], Kyp[i], label[i], fontsize=tags['label size'], color=tags['label color'])
                tags['Kikuchi lines']

    if not (tags['color ring zero'] == 'None'):
        ring = plt.Circle((0, 0), radius, color=tags['color ring zero'], fill=False, linewidth=2)
        ax.add_artist(ring)
        print(ring)
    return tags


def feq(element, q):
    """Atomic form factor parametrized in 1/Angstrom but converted to 1/nm

    The atomic form factor is from Kirkland: Advanced Computing in Electron Microscopy 2nd edition, Appendix C.
    From Appendix C of Kirkland, "Advanced Computing in Electron Microscopy", 2nd ed.
    Calculation of electron form factor for specific q:
    Using equation Kirkland C.15

    Parameters
    ----------
    element: string
        element name
    q: float
        magnitude of scattering vector in 1/nm -- (=> exp(-i*g.r), physics negative convention)

    Returns
    -------
    fL+fG: float
        atomic scattering vector
    """

    if not isinstance(element, str):
        raise TypeError('Element has to be a string')
    if element not in electronFF:
        if len(element) > 2:
            raise TypeError('Please use standard convention for element abbreviation with not more than two letters')
        else:
            raise TypeError('Element {element} not known to electron diffraction should')
    if not isinstance(q, (float, int)):
        raise TypeError('Magnitude of scattering vector has to be a number of type float')

    q = q/10
    # q is now magnitude of scattering vector in 1/A -- (=> exp(-i*g.r), physics negative convention)
    param = electronFF[element]
    fL = 0
    fG = 0
    for i in range(3):
        fL += param['fa'][i]/(q**2 + param['fb'][i])
        fG += param['fc'][i]*np.exp(-q**2 * param['fd'][i])

    # Conversion factor from scattering factors to volts. h^2/(2pi*m0*e), see e.g. Kirkland eqn. C.5
    # !NB RVolume is already in A unlike RPlanckConstant
    # ScattFacToVolts=(PlanckConstant**2)*(AngstromConversion**2)/(2*np.pi*ElectronMass*ElectronCharge)
    return fL+fG  # * ScattFacToVolts
