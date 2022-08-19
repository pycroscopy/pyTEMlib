"""
kinematic_scattering
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
    everything is in SI units, except length which is given in Angstrom.

Usage:
    See the notebooks for examples of these routines

All the input and output is done through a ase.Atoms object and the dictionary in the info attribute
"""

# numerical packages used
import numpy as np
import scipy.constants as const
import itertools

# plotting package used
import matplotlib.pylab as plt  # basic plotting

import pyTEMlib.file_tools as ft
from pyTEMlib.crystal_tools import *
from pyTEMlib.diffraction_plot import *

_version_ = "0.2022.1.0"

print('Using kinematic_scattering library version {_version_ }  by G.Duscher')

inputKeys = ['acceleration_voltage_V', 'zone_hkl', 'Sg_max', 'hkl_max']
optional_inputKeys = ['crystal', 'lattice_parameter_nm', 'convergence_angle_mrad', 'mistilt', 'thickness',
                      'dynamic correction', 'dynamic correction K0']


def read_poscar(filename):
    print('read_poscar and read_cif moved to file_tools, \n'
          'please use that library in the future!')
    ft.read_poscar(filename)


def example(verbose=True):
    """
    same as Zuo_fig_3_18
    """
    print('\n##########################')
    print('# Start of Example Input #')
    print('##########################\n')
    print('Define only mandatory input: ', inputKeys)
    print(' Kinematic diffraction routine will set optional input : ', optional_inputKeys)

    return Zuo_fig_3_18(verbose=verbose)


def Zuo_fig_3_18(verbose=True):
    """
    Input for Figure 3.18 in Zuo and Spence \"Advanced TEM\", 2017

    This input acts as an example as well as a reference

    Parameters:
    -----------
    verbose: boolean:
        optional to see output
    Returns:
    -------
        atoms: ase.Atoms
            Silicon crystal structure
        e
        dictionary: tags is the dictionary of all input and output parameter needed to reproduce that figure.
    """

    # INPUT
    # Create Silicon structure (Could be produced with Silicon routine)
    if verbose:
        print('Sample Input for Figure 3.18 in Zuo and Spence \"Advanced TEM\", 2017')
    import ase
    import ase.build
    a = 5.14  # A
    atoms = ase.build.bulk('Si', 'diamond', a=a, cubic=True)

    experiment = {'acceleration_voltage_V': 99.2 * 1000.0,  # V
                  'convergence_angle_mrad': 7.15,  # mrad;
                  'zone_hkl': np.array([-2, 2, 1]),
                  'mistilt': np.array([0, 0, 0]),  # mistilt in degrees
                  'Sg_max': .03,  # 1/A  maximum allowed excitation error
                  'hkl_max': 9  # Highest evaluated Miller indices
                  }
    # Define Experimental Conditions
    if verbose:
        print('###########################')
        print('# Experimental Conditions #')
        print('###########################')

        for key, value in experiment.items():
            print(f'tags[\'{key}\'] =', value)

        print('##################')
        print('# Output Options #')
        print('##################')

    # Output options
    output = {'background': 'black',  # 'white'  'grey'
              'color_map': 'plasma',
              'plot_HOLZ': True,
              'plot_HOLZ_excess': True,
              'plot_Kikuchi': True,
              'plot_reflections': True,
              'label_HOLZ': False,
              'label_Kikuchi': False,
              'label_reflections': False,
              'label_color': 'black',
              'label_size': 10,
              'color_Laue_Zones': ['red', 'blue', 'green', 'blue', 'green'],  # for OLZ give a sequence
              'color_Kikuchi': 'green',
              'linewidth_HOLZ': -1,  # -1: linewidth according to intensity (structure factor F^2)
              'linewidth_Kikuchi': -1,  # -1: linewidth according to intensity (structure factor F^2)
              'color_reflections': 'intensity',  # 'Laue Zone'
              'color_zero': 'white',  # 'None', 'white', 'blue'
              'color_ring_zero': 'None'  # 'Red' #'white' #, 'None'
              }

    if verbose:
        for key, value in output.items():
            print(f'tags[\'{key}\'] =', value)
        print('########################')
        print('# End of Example Input #')
        print('########################\n\n')

    if atoms.info is None:
        atoms.info = {}
    atoms.info['experimental'] = experiment
    atoms.info['output'] = output

    return atoms


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

    # first we rotate alpha about x-axis
    c, s = np.cos(alpha), np.sin(alpha)
    rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    # second we rotate beta about y-axis
    c, s = np.cos(beta), np.sin(beta)
    rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # third we rotate gamma about z-axis
    c, s = np.cos(gamma), np.sin(gamma)
    rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return np.dot(np.dot(np.dot(zone, rot_x), rot_y), rot_z)


def get_metric_tensor(matrix):
    """The metric tensor of the lattice."""
    metric_tensor2 = np.dot(matrix, matrix.T)
    return metric_tensor2


def vector_norm(g):
    """ Length of vector

    depreciated - use np.linalg.norm
    """
    g = np.array(g)
    return np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2 + g[:, 2] ** 2)


def get_wavelength(acceleration_voltage):
    """
    Calculates the relativistic corrected de Broglie wavelength of an electron in Angstrom

    Parameter:
    ---------
    acceleration_voltage: float
        acceleration voltage in volt
    Returns:
    -------
    wavelength: float
        wave length in Angstrom
    """
    if not isinstance(acceleration_voltage, (int, float)):
        raise TypeError('Acceleration voltage has to be a real number')
    eU = const.e * acceleration_voltage
    return const.h/np.sqrt(2*const.m_e*eU*(1+eU/(2*const.m_e*const.c**2)))*10**10


def find_nearest_zone_axis(tags):
    """Test all zone axis up to a maximum of hkl_max"""
    
    hkl_max = 5
    # Make all hkl indices
    h = np.linspace(-hkl_max, hkl_max, 2 * hkl_max + 1)  # all evaluated single Miller Indices
    hkl = np.array(list(itertools.product(h, h, h)))  # all evaluated Miller indices

    # delete [0,0,0]
    index = int(len(hkl) / 2)
    zones_hkl = np.delete(hkl, index, axis=0)  # delete [0,0,0]

    # make zone axis in reciprocal space
    zones_g = np.dot(zones_hkl, tags['reciprocal_unit_cell'])  # all evaluated reciprocal_unit_cell points

    # make zone axis in microscope coordinates of reciprocal space
    zones_g = np.dot(zones_g, tags['rotation_matrix'])  # rotate these reciprocal_unit_cell points

    # calculate angles with z-axis
    zones_g_norm = vector_norm(zones_g)
    z_axis = np.array([0, 0, 1])

    zones_angles = np.abs(np.arccos(np.dot((zones_g.T / zones_g_norm).T, z_axis)))

    # get smallest angle
    smallest = (zones_angles - zones_angles.min()) < 0.001
    if smallest.sum() > 1:  # multiples of Miller index of zone axis have same angle
        zone = zones_hkl[smallest]
        zone_index = abs(zone).sum(axis=1)
        ind = zone_index.argmin()
        zone_hkl = zone[ind]
    else:
        zone_hkl = zones_hkl[smallest][0]

    tags['nearest_zone_axis'] = zone_hkl

    # get other zone axes up to 5 degrees away
    others = np.logical_not(smallest)
    next_smallest = (zones_angles[others]) < np.deg2rad(5.)
    ind = np.argsort((zones_angles[others])[next_smallest])
    
    tags['next_nearest_zone_axes'] = ((zones_hkl[others])[next_smallest])[ind]

    return zone_hkl


def find_angles(zone):
    """Microscope stage coordinates of zone"""

    # rotation around y-axis
    r = np.sqrt(zone[1] ** 2 + zone[2] ** 2)
    alpha = np.arctan(zone[0] / r)
    if zone[2] < 0:
        alpha = np.pi - alpha
    # rotation around x-axis
    if zone[2] == 0:
        beta = np.pi / 2 * np.sign(zone[1])
    else:
        beta = (np.arctan(zone[1] / zone[2]))
    return alpha, beta


def stage_rotation_matrix(alpha, beta):
    """ Microscope stage coordinate system """
    
    # FIRST we rotate beta about x-axis
    c, s = np.cos(beta), np.sin(beta)
    rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    # second we rotate alpha about y-axis
    c, s = np.cos(alpha), np.sin(alpha)
    rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    return np.dot(rot_x, rot_y)


# ##################
# Determine rotation matrix to tilt zone axis onto z-axis
# We determine spherical coordinates to do that
# ##################

def get_rotation_matrix(tags):
    """zone axis in global coordinate system"""
    
    zone_hkl = tags['zone_hkl']
    zone = np.dot(zone_hkl, tags['reciprocal_unit_cell'])

    # angle of zone with Z around x,y:
    alpha, beta = find_angles(zone)

    alpha = alpha + tags['mistilt_alpha']
    beta = beta + tags['mistilt_beta']

    tags['y-axis rotation alpha'] = alpha
    tags['x-axis rotation beta'] = beta

    tags['rotation_matrix'] = rotation_matrix = stage_rotation_matrix(alpha, -beta)

    # the rotation now makes z-axis coincide with plane normal

    zone_nearest = find_nearest_zone_axis(tags)
    tags['nearest_zone_axis'] = zone_nearest

    # tilt angles of coordinates of nearest zone
    zone_nearest = np.dot(zone_nearest, tags['reciprocal_unit_cell'])

    alpha_nearest, beta_nearest = find_angles(zone_nearest)

    # calculate mistilt of nearest zone axis
    tags['mistilt_nearest_zone alpha'] = alpha - alpha_nearest
    tags['mistilt_nearest_zone beta'] = beta - beta_nearest

    tags['nearest_zone_axes'] = {}
    tags['nearest_zone_axes']['0'] = {}
    tags['nearest_zone_axes']['0']['hkl'] = tags['nearest_zone_axis']
    tags['nearest_zone_axes']['0']['mistilt_alpha'] = alpha - alpha_nearest
    tags['nearest_zone_axes']['0']['mistilt_beta'] = beta - beta_nearest

    # find polar coordinates of next nearest zones
    tags['nearest_zone_axes']['amount'] = len(tags['next_nearest_zone_axes']) + 1

    for i in range(len(tags['next_nearest_zone_axes'])):
        zone_n = tags['next_nearest_zone_axes'][i]
        tags['nearest_zone_axes'][str(i + 1)] = {}
        tags['nearest_zone_axes'][str(i + 1)]['hkl'] = zone_n

        zone_near = np.dot(zone_n, tags['reciprocal_unit_cell'])
        # zone_near_g = np.dot(zone_near,rotation_matrix)

        tags['nearest_zone_axes'][str(i + 1)]['g'] = zone_near
        alpha_nearest, beta_nearest = find_angles(zone_near)

        tags['nearest_zone_axes'][str(i + 1)]['mistilt_alpha'] = alpha - alpha_nearest
        tags['nearest_zone_axes'][str(i + 1)]['mistilt_beta'] = beta - beta_nearest
        # print('other' , i, np.rad2deg([alpha, alpha_nearest, beta, beta_nearest]))

    return rotation_matrix


def check_sanity(atoms, verbose_level=0):
    """
    Check sanity of input parameters
    """
    stop = False
    output = atoms.info['output']
    tags = atoms.info['experimental']
    for key in ['acceleration_voltage_V']:
        if key not in tags:
            print(f'Necessary parameter {key} not defined')
            stop = True
    if 'SpotPattern' not in output:
        output['SpotPattern'] = False
    if output['SpotPattern']:
        if 'zone_hkl' not in tags:
            print(' No zone_hkl defined')
            stop = True
        if 'Sg_max' not in tags:
            print(' No Sg_max defined')
            stop = True
    if 'hkl_max' not in tags:
        print(' No hkl_max defined')
        stop = True

    if stop:
        print('Input is not complete, stopping')
        print('Try \'example()\' for example input')
        return False
    ############################################
    # Check optional input
    ############################################

    if output['SpotPattern']:
        if 'mistilt_alpha degree' not in tags:
            # mistilt is in microscope coordinates
            tags['mistilt_alpha'] = tags['mistilt_alpha degree'] = 0.0
            if verbose_level > 0:
                print('Setting undefined input:  tags[\'mistilt_alpha\'] = 0.0 ')
        else:
            tags['mistilt_alpha'] = np.deg2rad(tags['mistilt_alpha degree'])

        if 'mistilt_beta degree' not in tags:
            # mistilt is in microscope coordinates
            tags['mistilt_beta'] = tags['mistilt_beta degree'] = 0.0
            if verbose_level > 0:
                print('Setting undefined input:  tags[\'mistilt_beta\'] = 0.0')
        else:
            tags['mistilt_beta'] = np.deg2rad(tags['mistilt_beta degree'])

        if 'convergence_angle_mrad' not in tags:
            tags['convergence_angle_mrad'] = 0.
            if verbose_level > 0:
                print('Setting undefined input: tags[\'convergence_angle_mrad\'] = 0')

        if 'thickness' not in tags:
            tags['thickness'] = 0.
            if verbose_level > 0:
                print('Setting undefined input: tags[\'thickness\'] = 0')
        if 'dynamic correction' not in tags:
            tags['dynamic correction'] = 0.
            if verbose_level > 0:
                print('Setting undefined input: tags[\'dynamic correction\'] = False')
        if 'dynamic correction K0' not in tags:
            tags['dynamic correction K0'] = 0.
            if verbose_level > 0:
                print('Setting undefined input: tags[\'dynamic correction k0\'] = False')
    return not stop


def scattering_matrix(tags, verbose_level=1):
    """ Scattering matrix"""
    if not check_sanity(tags, verbose_level):
        return
    # ##
    # Pair distribution Function
    # ##
    unit_cell = np.array(tags['unit_cell'])
    base = tags['base']

    atom_coordinates = np.dot(base, unit_cell)

    n = 20
    x = np.linspace(-n, n, 2 * n + 1)  # all evaluated multiples of x
    xyz = np.array(list(itertools.product(x, x, x)))  # all evaluated multiples in all direction

    mat = np.dot(xyz, unit_cell)  # all evaluated unit_cells

    atom = {}

    for i in range(len(atom_coordinates)):
        distances = np.linalg.norm(mat + atom_coordinates[i], axis=1)
        if i == 0:
            all_distances = distances
        else:
            all_distances = np.append(all_distances, distances)
        unique, counts = np.unique(distances, return_counts=True)

        atom[str(i)] = dict(zip(unique, counts))
        print(atom[str(i)])

        all_distances = np.append(all_distances, distances)
        unique, counts = np.unique(all_distances, return_counts=True)

        plt.plot(unique, counts)
        plt.show()


def ring_pattern_calculation(atoms, verbose=False):
    """
    Calculate the ring diffraction pattern of a crystal structure

    Parameters
    ----------
    atoms: Crystal
        crystal structure
    verbose: verbose print-outs
        set to False
    Returns
    -------
    tags: dict
        dictionary with diffraction information added
    """
    
    # Check sanity
    if not check_sanity(atoms, verbose):
        return

    tags = atoms.info['experimental']
    # wavelength
    tags['wave_length'] = get_wavelength(tags['acceleration_voltage_V'])

    #  volume of unit_cell
    unit_cell = atoms.cell.array
    metric_tensor = get_metric_tensor(unit_cell)  # converts hkl to g vectors and back
    tags['metric_tensor'] = metric_tensor
    # volume_unit_cell = np.sqrt(np.linalg.det(metric_tensor))

    # reciprocal_unit_cell
    
    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    reciprocal_unit_cell = atoms.cell.reciprocal()  # np.linalg.inv(unit_cell).T  # transposed of inverted unit_cell
    tags['reciprocal_unit_cell'] = reciprocal_unit_cell
    # inverse_metric_tensor = get_metric_tensor(reciprocal_unit_cell)

    hkl_max = tags['hkl_max']

    h = np.linspace(-hkl_max, hkl_max, 2 * hkl_max + 1)  # all evaluated single Miller Indices
    hkl = np.array(list(itertools.product(h, h, h)))  # all evaluated Miller indices
    
    # delete [0,0,0]
    index_center = int(len(hkl) / 2)
    hkl = np.delete(hkl, index_center, axis=0)  # delete [0,0,0]

    g_hkl = np.dot(hkl, reciprocal_unit_cell)  # all evaluated reciprocal_unit_cell points

    ##################################
    # Calculate Structure Factors
    #################################

    structure_factors = []
    for j in range(len(g_hkl)):
        F = 0
        for b in range(len(atoms)):
            f = feq(atoms[b].symbol, np.linalg.norm(g_hkl[j]))
            F += f * np.exp(-2 * np.pi * 1j * (hkl[j] * atoms.get_scaled_positions()[b]).sum())

        structure_factors.append(F)

    F = structure_factors = np.array(structure_factors)

    # Sort reflection in allowed and forbidden #
    
    allowed = np.absolute(F) > 0.000001  # allowed within numerical error

    if verbose:
        print('Of the {0} possible reflection {1} are allowed.'.format(hkl.shape[0], allowed.sum()))

    # information of allowed reflections
    hkl_allowed = hkl[allowed][:]
    g_allowed = g_hkl[allowed, :]
    F_allowed = F[allowed]
    g_norm_allowed = vector_norm(g_allowed)  # length of all vectors = 1/

    ind = np.argsort(g_norm_allowed)
    g_norm_sorted = g_norm_allowed[ind]
    hkl_sorted = hkl_allowed[ind][:]
    F_sorted = F_allowed[ind]

    unique, counts = np.unique(np.around(g_norm_sorted, decimals=5), return_counts=True)
    if verbose:
        print('Of the {0} allowed reflection {1} have unique distances.'.format(allowed.sum(), len(unique)))

    reflections_d = []
    reflections_m = []
    reflections_F = []

    start = 0
    for i in range(len(unique)):
        end = start + counts[i]
        hkl_max = np.argmax(hkl_sorted[start:end].sum(axis=1))

        reflections_d.append(g_norm_sorted[start])
        reflections_m.append(hkl_sorted[start + hkl_max])
        reflections_F.append(F_sorted[start])  # :end].sum())

        start = end

    if verbose:
        print('\n\n [hkl]  \t 1/d [1/nm] \t d [nm] \t F^2 ')
        for i in range(len(unique)):
            print(' {0} \t {1:.2f} \t         {2:.4f} \t {3:.2f} '
                  .format(reflections_m[i], unique[i]*10., 1 / unique[i]/10., np.real(reflections_F[i]) ** 2))

    atoms.info['Ring_Pattern'] = {}
    atoms.info['Ring_Pattern']['allowed'] = {}
    atoms.info['Ring_Pattern']['allowed']['hkl'] = reflections_m
    atoms.info['Ring_Pattern']['allowed']['g norm'] = unique
    atoms.info['Ring_Pattern']['allowed']['structure factor'] = reflections_F
    atoms.info['Ring_Pattern']['allowed']['multiplicity'] = counts

    atoms.info['Ring_Pattern']['profile_x'] = np.linspace(0, unique.max(), 2048)
    step_size = atoms.info['Ring_Pattern']['profile_x'][1]
    intensity = np.zeros(2048)
    x_index = [(unique / step_size + 0.5).astype(int)]
    intensity[x_index] = np.array(np.real(reflections_F)) * np.array(np.real(reflections_F))
    atoms.info['Ring_Pattern']['profile_y delta'] = intensity

    def gaussian(xx, pp):
        s1 = pp[2] / 2.3548
        prefactor = 1.0 / np.sqrt(2 * np.pi * s1 ** 2)
        y = (pp[1] * prefactor) * np.exp(-(xx - pp[0]) ** 2 / (2 * s1 ** 2))
        return y

    if 'thickness' in tags:
        if tags['thickness'] > 0:
            x = np.linspace(-1024, 1023, 2048) * step_size
            p = [0.0, 1, 2 / tags['thickness']]

            gauss = gaussian(x, p)
            intensity = np.convolve(np.array(intensity), np.array(gauss), mode='same')
    atoms.info['Ring_Pattern']['profile_y'] = intensity

    # Make pretty labels
    hkl_allowed = reflections_m
    hkl_label = make_pretty_labels(hkl_allowed)
    atoms.info['Ring_Pattern']['allowed']['label'] = hkl_label


def get_dynamically_allowed(atoms, verbose=False):
    if not isinstance(atoms, ase.Atoms):
        print('we need an ase atoms object as input')
    if 'diffraction' not in atoms.info:
        print('Run the kinematic_scattering function first')

    # Dynamically Allowed Reflection

    dif = atoms.info['diffraction']
    hkl_allowed = dif['allowed']['hkl']
    hkl_forbidden = dif['forbidden']['hkl']
    indices = range(len(hkl_allowed))
    combinations = [list(x) for x in itertools.permutations(indices, 2)]
    hkl_forbidden = hkl_forbidden.tolist()
    dynamically_allowed = np.zeros(len(hkl_forbidden), dtype=bool)
    for [i, j] in combinations:
        possible = (hkl_allowed[i] + hkl_allowed[j]).tolist()
        if possible in hkl_forbidden:
            dynamically_allowed[hkl_forbidden.index(possible)] = True
    dif['forbidden']['dynamically_allowed'] = dynamically_allowed

    if verbose:
        print(f"Of the {len(hkl_forbidden)} forbidden reflection {dynamically_allowed.sum()} "
              f"can be dynamically activated.")
        # print(dif['forbidden']['hkl'][dynamically_allowed])


def kinematic_scattering(atoms, verbose=False):
    """
        All kinematic scattering calculation

        Calculates Bragg spots, Kikuchi lines, excess, and deficient HOLZ lines

        Parameters
        ----------
        atoms: ase.Atoms
            object with crystal structure:
            and with experimental parameters in info attribute:
            'acceleration_voltage_V', 'zone_hkl', 'Sg_max', 'hkl_max'
            Optional parameters are:
            'mistilt', convergence_angle_mrad', and 'crystal_name'
            verbose = True will give extended output of the calculation
        verbose: boolean
            default is False

        Returns
        -------
        atoms:
            There are three sub_dictionaries in info attribute:
            ['allowed'], ['forbidden'], and ['HOLZ']
            ['allowed'] and ['forbidden'] dictionaries contain:
                ['Sg'], ['hkl'], ['g'], ['structure factor'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], ['HHOLZ'], ['label'], and ['Laue_zone']
            the ['HOLZ'] dictionary contains:
                ['slope'], ['distance'], ['theta'], ['g_deficient'], ['g_excess'], ['hkl'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], and  ['HHOLZ']
            Please note that the Kikuchi lines are the HOLZ lines of ZOLZ

            There are also a few parameters stored in the main dictionary:
                ['wave_length_nm'], ['reciprocal_unit_cell'], ['inner_potential_V'], ['incident_wave_vector'],
                ['volume'], ['theta'], ['phi'], and ['incident_wave_vector_vacuum']
    """

    # Check sanity
    if atoms.info is None:
        atoms.info = {'output': {}, 'experimental': {}}
    elif 'output' in atoms.info:
        output = atoms.info['output']
    else:
        output = atoms.info['output'] = {}

    output['SpotPattern'] = True

    if 'experimental' not in atoms.info:
        tags = atoms.info['experimental'] = {}

    if not check_sanity(atoms):
        print('Input is not complete, stopping')
        print('Try \'example()\' for example input')
        return

    tags = atoms.info['experimental']
    
    tags['wave_length'] = get_wavelength(tags['acceleration_voltage_V'])
    
    # ###########################################
    # reciprocal_unit_cell 
    # ###########################################
    unit_cell = atoms.cell.array
    tags['unit_cell'] = unit_cell
    metric_tensor = get_metric_tensor(unit_cell)  # converts hkl to g vectors and back
    tags['metric_tensor'] = metric_tensor
    volume_unit_cell = atoms.cell.volume

    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    reciprocal_unit_cell = atoms.cell.reciprocal()  # np.linalg.inv(unit_cell).T  # transposed of inverted unit_cell
    tags['reciprocal_unit_cell'] = reciprocal_unit_cell
    inverse_metric_tensor = get_metric_tensor(reciprocal_unit_cell)

    # ###########################################
    # Incident wave vector k0 in vacuum and material 
    # ###########################################

    # Incident wave vector K0 in vacuum and material 
    u0 = 0.0  # in (Ang)
    # atom form factor of zero reflection angle is the inner potential in 1/A
    for i in range(len(atoms)):
        u0 += feq(atoms[i].symbol, 0.0)

    angstrom_conversion = 1.0e10  # So [1A (in m)] * angstrom_conversion = 1
    # NanometerConversion = 1.0e9

    scattering_factor_to_volts = (const.h ** 2) * (1e10 ** 2) / (2 * np.pi * const.m_e * const.e) * volume_unit_cell
    tags['inner_potential_V'] = u0 * scattering_factor_to_volts
    if verbose:
        print(f'The inner potential is {u0:.1f} V')

    # Calculating incident wave vector magnitude 'k0' in material
    wl = tags['wave_length']
    tags['incident_wave_vector_vacuum'] = 1 / wl
    
    k_0 = tags['incident_wave_vector'] = np.sqrt(1 / wl**2 + u0/volume_unit_cell)  # 1/Ang

    tags['convergence_angle_A-1'] = k_0*np.sin(tags['convergence_angle_mrad']/1000.)

    if verbose:
        print(f"Using an acceleration voltage of {tags['acceleration_voltage_V']/1000:.1f}kV")
        print(f'Magnitude of incident wave vector in material: {k_0:.4f} 1/Ang and in vacuum: {1/wl:.4f} 1/Ang')
        print(f"Which is an wave length of {1/k_0 * 100.:.3f} pm in the material and {wl * 100.:.3f} pm "
              f"in the vacuum")
        print(f"The convergence angle of {tags['convergence_angle_mrad']:.1f}mrad "
              f"= {tags['convergence_angle_A-1']:.2f} 1/A")
        print(f"Magnitude of incident wave vector in material: {k_0:.1f} 1/A which is a wavelength {100/k_0:.3f} pm")

    # ############
    # Rotate
    # ############
    
    # get rotation matrix to rotate zone axis onto z-axis
    rotation_matrix = get_rotation_matrix(tags)
    
    if verbose:
        print(f"Rotation alpha {np.rad2deg(tags['y-axis rotation alpha']):.1f} degree, "
              f" beta {np.rad2deg(tags['x-axis rotation beta']):.1f} degree")
        print(f"from zone axis {tags['zone_hkl']}")
        print(f"Tilting {1} by {np.rad2deg(tags['mistilt_alpha']):.2f} " 
              f" in alpha and {np.rad2deg(tags['mistilt_beta']):.2f} in beta direction results in :")
        # list(tags['zone_hkl'])
        #
        # print(f"zone axis {list(tags['nearest_zone_axis'])} with a mistilt of "
        #      f"{np.rad2deg(tags['mistilt_nearest_zone alpha']):.2f} in alpha "
        #      f"and {np.rad2deg(tags['mistilt_nearest_zone beta']):.2f} in beta direction")
        nearest = tags['nearest_zone_axes']
        print('Next nearest zone axes are:')
        for i in range(1, nearest['amount']):
            print(f"{nearest[str(i)]['hkl']}: mistilt: {np.rad2deg(nearest[str(i)]['mistilt_alpha']):6.2f}, "
                  f"{np.rad2deg(nearest[str(i)]['mistilt_beta']):6.2f}")
    # rotate incident wave vector 
    k0_unit_vector = np.array([0, 0, 1])  # incident unit wave vector
    k0_vector = k0_unit_vector * k_0  # incident  wave vector
    cent = k0_vector  # center of Ewald sphere

    if verbose:
        print('Center of Ewald sphere ', k0_vector)

    # #######################
    # Find all Miller indices whose reciprocal point lies near the Ewald sphere with radius k_0
    # within a maximum excitation error Sg
    # #######################
    
    hkl_max = tags['hkl_max']
    Sg_max = tags['Sg_max']  # 1/Ang  maximum allowed excitation error
    
    h = np.linspace(-hkl_max, hkl_max, 2*hkl_max+1)  # all evaluated single Miller Indices
    hkl = np.array(list(itertools.product(h, h, h)))  # all evaluated Miller indices
    g_non_rot = np.dot(hkl, reciprocal_unit_cell)  # all evaluated reciprocal_unit_cell points
    g_norm = np.linalg.norm(g_non_rot, axis=1)   # length of all vectors
    not_zero = g_norm > 0
    g_non_rot = g_non_rot[not_zero]  # zero reflection will make problems further on, so we exclude it.
    g_norm = g_norm[not_zero]
    hkl_all = hkl[not_zero]
    g = np.dot(g_non_rot, rotation_matrix)
    
    # #######################
    # Calculate excitation errors for all reciprocal_unit_cell points
    # #######################
    
    # Zuo and Spence, 'Adv TEM', 2017 -- Eq 3:14
    S = (k_0**2-np.linalg.norm(g - k0_vector, axis=1)**2)/(2*k_0)
    
    # g_mz = g - k0_vector
    # in_sqrt = g_mz[:, 2]**2 + np.linalg.norm(g_mz, axis=1)**2 - k_0**2
    # in_sqrt[in_sqrt < 0] = 0.
    # S = -g_mz[:, 2] - np.sqrt(in_sqrt)
    
    # #######################
    # Determine reciprocal_unit_cell points with excitation error less than the maximum allowed one: Sg_max
    # #######################
    
    reflections = abs(S) < Sg_max   # This is now a boolean array with True for all possible reflections
    
    Sg = S[reflections]
    g_hkl = g[reflections]
    g_hkl_non_rot = g_non_rot[reflections]
    hkl = hkl_all[reflections]
    g_norm = g_norm[reflections]

    if verbose:
        print('Of the {0} tested reciprocal_unit_cell points, {1} have an excitation error less than {2:.2f} 1/nm'.
              format(len(g), len(g_hkl), Sg_max))
        
    # #################################
    # Calculate Structure Factors
    # ################################
    
    structure_factors = []
    for j in range(len(g_hkl)):
        F = 0
        for b in range(len(atoms)):
            f = feq(atoms[b].symbol, g_norm[j])  # Atomic form factor for element and momentum change (g vector)
            F += f * np.exp(-2*np.pi*1j*(g_hkl_non_rot[j]*atoms.positions[b]).sum())        
        structure_factors.append(F)
    F = structure_factors = np.array(structure_factors)
    
    # ###########################################
    # Sort reflection in allowed and forbidden #
    # ###########################################
    
    allowed = np.absolute(F) > 0.000001    # allowed within numerical error 

    if verbose:
        print('Of the {0} possible reflection {1} are allowed.'.format(hkl.shape[0], allowed.sum()))
        
    # information of allowed reflections
    s_g_allowed = Sg[allowed]
    hkl_allowed = hkl[allowed][:]
    g_allowed = g_hkl[allowed, :]
    F_allowed = F[allowed]
    g_norm_allowed = g_norm[allowed]

    atoms.info['diffraction'] = {}
    dif = atoms.info['diffraction']
    dif['allowed'] = {}
    dif['allowed']['Sg'] = s_g_allowed
    dif['allowed']['hkl'] = hkl_allowed
    dif['allowed']['g'] = g_allowed
    dif['allowed']['structure factor'] = F_allowed
    
    # Calculate Extinction Distance  Reimer 7.23
    # - makes only sense for non-zero F

    xi_g = np.real(np.pi * volume_unit_cell * k_0 / F_allowed)

    # Calculate Intensity of beams  Reimer 7.25
    if 'thickness' not in tags:
        tags['thickness'] = 0.
    thickness = tags['thickness']
    if thickness > 0.1:
        I_g = np.real(np.pi ** 2 / xi_g ** 2 * np.sin(np.pi * thickness * s_g_allowed) ** 2 / (np.pi * s_g_allowed)**2)
        dif['allowed']['Ig'] = I_g

    dif['allowed']['intensities'] = intensities = np.real(F_allowed) ** 2
    
    # Calculate Extinction Distance  Reimer 7.23
    # - makes only sense for non-zero F

    xi_g = np.real(np.pi * volume_unit_cell * k_0 / F_allowed)

    # ###########################
    # Calculate Intensities (of allowed reflections)
    # ###########################
    
    # Calculate Intensity of beams  Reimer 7.25
    if 'thickness' not in tags:
        tags['thickness'] = 0.
    thickness = tags['thickness']
    if thickness > 0.1:
        I_g = np.real(np.pi ** 2 / xi_g ** 2 * np.sin(np.pi * thickness * s_g_allowed) ** 2 / (np.pi * s_g_allowed)**2)
        dif['allowed']['Ig'] = I_g

    dif['allowed']['intensities'] = intensities = np.real(F_allowed) ** 2
    
    # information of forbidden reflections
    forbidden = np.logical_not(allowed)
    Sg_forbidden = Sg[forbidden]
    hkl_forbidden = hkl[forbidden]
    g_forbidden = g_hkl[forbidden]
    F_forbidden = F[forbidden]
    
    dif['forbidden'] = {}
    dif['forbidden']['Sg'] = Sg_forbidden
    dif['forbidden']['hkl'] = hkl_forbidden
    dif['forbidden']['g'] = g_forbidden

    # ##########################
    # Make pretty labels
    # ##########################
    hkl_label = make_pretty_labels(hkl_allowed)
    dif['allowed']['label'] = hkl_label
    hkl_label = make_pretty_labels(hkl_forbidden)
    dif['forbidden']['label'] = hkl_label
    
    # Center of Laue Circle
    laue_circle = np.dot(tags['nearest_zone_axis'], tags['reciprocal_unit_cell'])
    laue_circle = np.dot(laue_circle, rotation_matrix)
    laue_circle = laue_circle / np.linalg.norm(laue_circle) * k_0
    laue_circle[2] = 0

    dif['Laue_circle'] = laue_circle
    if verbose:
        print('Laue_circle', laue_circle)

    # ###########################
    # Calculate Laue Zones (of allowed reflections)
    # ###########################
    # Below is the expression given in most books.
    # However, that would only work for orthogonal crystal systems
    # Laue_Zone = abs(np.dot(hkl_allowed,tags['zone_hkl']))  # works only for orthogonal systems

    # This expression works for all crystal systems
    # Remember we have already tilted, and so the dot product is trivial and gives only the z-component.
    length_zone_axis = np.linalg.norm(np.dot(tags['zone_hkl'], tags['unit_cell']))
    laue_zone = abs(np.dot(hkl_allowed, tags['nearest_zone_axis']))
    dif['allowed']['Laue_Zone'] = laue_zone

    ZOLZ_forbidden = abs(np.floor(g_forbidden[:, 2]*length_zone_axis+0.5)) == 0
    
    dif['forbidden']['Laue_Zone'] = ZOLZ_forbidden
    ZOLZ = laue_zone == 0
    FOLZ = laue_zone == 1
    SOLZ = laue_zone == 2
    HOLZ = laue_zone > 0
    HOLZp = laue_zone > 2

    dif['allowed']['ZOLZ'] = ZOLZ
    dif['allowed']['FOLZ'] = FOLZ
    dif['allowed']['SOLZ'] = SOLZ
    dif['allowed']['HOLZ'] = HOLZ
    dif['allowed']['HOLZ_plus'] = dif['allowed']['HHOLZ'] = HOLZp
    
    if verbose:
        print(' There are {0} allowed reflections in the zero order Laue Zone'.format(ZOLZ.sum()))
        print(' There are {0} allowed reflections in the first order Laue Zone'.format((laue_zone == 1).sum()))
        print(' There are {0} allowed reflections in the second order Laue Zone'.format((laue_zone == 2).sum()))
        print(' There are {0} allowed reflections in the other higher order Laue Zones'.format((laue_zone > 2).sum()))
        
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
    dif['forbidden']['dynamically_activated'] = dynamical_allowed
    dif['forbidden']['forbidden'] = dynamical_allowed
    if verbose:
        print('Length of zone axis vector in real space {0} nm'.format(np.round(length_zone_axis, 3)))
        print(f'There are {len(dynamical_allowed)} forbidden but dynamical activated diffraction spots:')
        # print(tags['forbidden']['hkl'][dynamical_allowed])

    # ###################################
    # Calculate HOLZ and Kikuchi Lines #
    # ###################################

    # Dynamic Correction

    # Equation Spence+Zuo 3.86a
    gamma_1 = - 1./(2.*k_0) * (intensities / (2.*k_0*s_g_allowed)).sum()
    # print('gamma_1',gamma_1)

    # Equation Spence+Zuo 3.84
    Kg = k_0 - k_0*gamma_1/(g_allowed[:, 2]+1e-15)
    Kg[ZOLZ] = k_0
    
    # Calculate angle between K0 and deficient cone vector
    # For dynamic calculations K0 is replaced by Kg
    Kg[:] = k_0
    d_theta = np.arcsin(g_norm_allowed/Kg/2.)-np.arcsin(np.abs(g_allowed[:, 2])/g_norm_allowed)

    # calculate length of distance of deficient cone to K0 in ZOLZ plane
    gd_length = 2*np.sin(d_theta/2)*k_0
    
    # Calculate nearest point of HOLZ and Kikuchi lines
    g_closest = g_allowed.copy()
    g_closest = g_closest*(gd_length/np.linalg.norm(g_closest, axis=1))[:, np.newaxis]
    
    g_closest[:, 2] = 0.
    
    # calculate and save line in Hough space coordinates (distance and theta)
    slope = g_closest[:, 0]/(g_closest[:, 1]+1e-10)
    distance = gd_length
    theta = np.arctan2(g_allowed[:, 0], g_allowed[:, 1])
    
    dif['HOLZ'] = {}
    dif['HOLZ']['slope'] = slope
    # a line is now given by
    
    dif['HOLZ']['distance'] = distance
    dif['HOLZ']['theta'] = theta
    
    dif['HOLZ']['g_deficient'] = g_closest
    dif['HOLZ']['g_excess'] = g_closest+g_allowed
    
    dif['HOLZ']['ZOLZ'] = ZOLZ
    dif['HOLZ']['HOLZ'] = HOLZ
    dif['HOLZ']['FOLZ'] = FOLZ
    dif['HOLZ']['SOLZ'] = SOLZ
    dif['HOLZ']['HHOLZ'] = HOLZp  # even higher HOLZ
    
    dif['HOLZ']['hkl'] = dif['allowed']['hkl']
    dif['HOLZ']['intensities'] = intensities

    ####################################
    # Calculate HOLZ and Kikuchi Lines #
    ####################################

    tags_kikuchi = tags.copy()
    tags_kikuchi['mistilt_alpha'] = 0
    tags_kikuchi['mistilt_beta'] = 0

    for i in range(1):  # tags['nearest_zone_axes']['amount']):

        zone_tags = tags['nearest_zone_axes'][str(i)]
        tags_kikuchi['zone_hkl'] = zone_tags['hkl']
        if verbose:
            print('Calculating Kikuchi lines for zone: ', zone_tags['hkl'])

        tags_kikuchi['Laue_circle'] = laue_circle
        # Rotate to nearest zone axis
        rotation_matrix = get_rotation_matrix(tags_kikuchi)

        g_kikuchi_all = np.dot(g_non_rot, rotation_matrix)

        ZOLZ = abs(g_kikuchi_all[:, 2]) < .1

        g_kikuchi = g_kikuchi_all[ZOLZ]
        S = (k_0**2-np.linalg.norm(g_kikuchi - k0_vector, axis=1)**2)/(2*k_0)
        reflections = abs(S) < .01  # This is now a boolean array with True for all possible reflections
        g_kikuchi = g_kikuchi[reflections]
        hkl_kikuchi = (hkl_all[ZOLZ])[reflections]

        structure_factors = []
        for j in range(len(g_kikuchi)):
            F = 0
            for b in range(len(atoms)):
                f = feq(atoms[b].symbol, np.linalg.norm(g_kikuchi[j]))
                F += f * np.exp(-2 * np.pi * 1j * (g_kikuchi[j] * atoms.positions[b]).sum())
            structure_factors.append(F)

        F = np.array(structure_factors)

        allowed_kikuchi = np.absolute(F) > 0.000001

        g_kikuchi = g_kikuchi[allowed_kikuchi]
        hkl_kikuchi = hkl_kikuchi[allowed_kikuchi]

        gd2 = g_kikuchi / 2.
        gd2[:, 2] = 0.

        # calculate and save line in Hough space coordinates (distance and theta)
        slope2 = gd2[:, 0] / (gd2[:, 1] + 1e-20)
        distance2 = np.sqrt(gd2[:, 0] * gd2[:, 0] + gd2[:, 1] * gd2[:, 1])
        theta2 = np.arctan(slope2)

        dif['Kikuchi'] = {}
        dif['Kikuchi']['slope'] = slope2
        dif['Kikuchi']['distance'] = distance2
        dif['Kikuchi']['theta'] = theta2
        dif['Kikuchi']['hkl'] = hkl_kikuchi
        dif['Kikuchi']['g_hkl'] = g_kikuchi
        dif['Kikuchi']['g_deficient'] = gd2
        dif['Kikuchi']['min_dist'] = gd2 + laue_circle

    if verbose:
        print('pyTEMlib\'s  \"kinematic_scattering\" finished')


def kinematic_scattering2(atoms, verbose=False):
    """
        All kinematic scattering calculation

        Calculates Bragg spots, Kikuchi lines, excess, and deficient HOLZ lines

        Parameters
        ----------
        atoms: ase.Atoms
            object with crystal structure:
            and with experimental parameters in info attribute:
            'acceleration_voltage_V', 'zone_hkl', 'Sg_max', 'hkl_max'
            Optional parameters are:
            'mistilt', convergence_angle_mrad', and 'crystal_name'
            verbose = True will give extended output of the calculation
        verbose: boolean
            default is False

        Returns
        -------
        ato,s:
            There are three sub_dictionaries in info attribute:
            ['allowed'], ['forbidden'], and ['HOLZ']
            ['allowed'] and ['forbidden'] dictionaries contain:
                ['Sg'], ['hkl'], ['g'], ['structure factor'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], ['HHOLZ'], ['label'], and ['Laue_zone']
            the ['HOLZ'] dictionary contains:
                ['slope'], ['distance'], ['theta'], ['g_deficient'], ['g_excess'], ['hkl'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], and  ['HHOLZ']
            Please note that the Kikuchi lines are the HOLZ lines of ZOLZ

            There are also a few parameters stored in the main dictionary:
                ['wave_length_nm'], ['reciprocal_unit_cell'], ['inner_potential_V'], ['incident_wave_vector'],
                ['volume'], ['theta'], ['phi'], and ['incident_wave_vector_vacuum']
    """

    # Check sanity
    if atoms.info is None:
        atoms.info = {'output': {}, 'experimental': {}}
    elif 'output' in atoms.info:
        output = atoms.info['output']
    else:
        output = atoms.info['output'] = {}

    output['SpotPattern'] = True

    if 'experimental' not in atoms.info:
        tags = atoms.info['experimental'] = {}

    if not check_sanity(atoms):
        print('Input is not complete, stopping')
        print('Try \'example()\' for example input')
        return

    tags = atoms.info['experimental']

    # wavelength
    tags['wave_length'] = get_wavelength(tags['acceleration_voltage_V'])

    #  volume of unit_cell
    unit_cell = atoms.cell.array
    metric_tensor = get_metric_tensor(unit_cell)  # converts hkl to g vectors and back
    tags['metric_tensor'] = metric_tensor
    volume_unit_cell = atoms.cell.volume

    # reciprocal_unit_cell

    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    reciprocal_unit_cell = atoms.cell.reciprocal()  # np.linalg.inv(unit_cell).T  # transposed of inverted unit_cell
    tags['reciprocal_unit_cell'] = reciprocal_unit_cell
    inverse_metric_tensor = get_metric_tensor(reciprocal_unit_cell)

    if verbose:
        print('reciprocal_unit_cell')
        print(np.round(reciprocal_unit_cell, 3))

    ############################################
    # Incident wave vector k0 in vacuum and material
    ############################################

    u0 = 0.0  # in (Ang)
    # atom form factor of zero reflection angle is the inner potential in 1/A
    for i in range(len(atoms)):
        u0 += feq(atoms[i].symbol, 0.0)

    scattering_factor_to_volts = (const.h ** 2) * (1e10 ** 2) / (2 * np.pi * const.m_e * const.e) * volume_unit_cell

    tags['inner_potential_V'] = u0 * scattering_factor_to_volts
    if verbose:
        print(f'The inner potential is {u0:.1f} V')

    # Calculating incident wave vector magnitude 'k0' in material
    wl = tags['wave_length']
    tags['incident_wave_vector_vacuum'] = 1 / wl

    k0 = tags['incident_wave_vector'] = np.sqrt(1 / wl**2 + u0)  # 1/Ang

    tags['convergence_angle_A-1'] = k0 * np.sin(tags['convergence_angle_mrad'] / 1000.)
    if verbose:
        print(f"Using an acceleration voltage of {tags['acceleration_voltage_V']/1000:.1f}kV")
        print(f'Magnitude of incident wave vector in material: {k0:.1f} 1/Ang and in vacuum: {1/wl:.1f} 1/Ang')
        print(f"Which is an wave length of {1 / k0 * 100.:.3f} pm in the material and {wl * 100.:.3f} pm "
              f"in the vacuum")
        print(f"The convergence angle of {tags['convergence_angle_mrad']:.1f}mrad "
              f"= {tags['convergence_angle_A-1']:.2f} 1/A")
        print(f"Magnitude of incident wave vector in material: {k0:.1f} 1/A which is a wavelength {100/k0:.3f} pm")

    # ############
    # Rotate
    # ############

    # get rotation matrix to rotate zone axis onto z-axis
    rotation_matrix = get_rotation_matrix(tags)

    if verbose:
        print(f"Rotation alpha {np.rad2deg(tags['y-axis rotation alpha']):.1f} degree, "
              f" beta {np.rad2deg(tags['x-axis rotation beta']):.1f} degree")
        print(f"from zone axis {tags['zone_hkl']}")
        print(f"Tilting {1} by {np.rad2deg(tags['mistilt_alpha']):.2f} " 
              f" in alpha and {np.rad2deg(tags['mistilt_beta']):.2f} in beta direction results in :")
        # list(tags['zone_hkl'])
        #
        # print(f"zone axis {list(tags['nearest_zone_axis'])} with a mistilt of "
        #      f"{np.rad2deg(tags['mistilt_nearest_zone alpha']):.2f} in alpha "
        #      f"and {np.rad2deg(tags['mistilt_nearest_zone beta']):.2f} in beta direction")
        nearest = tags['nearest_zone_axes']
        print('Next nearest zone axes are:')
        for i in range(1, nearest['amount']):
            print(f"{nearest[str(i)]['hkl']}: mistilt: {np.rad2deg(nearest[str(i)]['mistilt_alpha']):6.2f}, "
                  f"{np.rad2deg(nearest[str(i)]['mistilt_beta']):6.2f}")
    k0_unit_vector = np.array([0, 0, 1])  # incident unit wave vector
    k0_vector = k0_unit_vector * k0  # incident  wave vector
    cent = k0_vector  # center of Ewald sphere

    if verbose:
        print('Center of Ewald sphere ', cent)

    # Find all Miller indices whose reciprocal point lays near the Ewald sphere with radius k0
    # within a maximum excitation error Sg
    hkl_max = tags['hkl_max']
    Sg_max = tags['Sg_max']  # 1/A  maximum allowed excitation error

    h = np.linspace(-hkl_max, hkl_max, 2 * hkl_max + 1)  # all evaluated single Miller indices
    hkl = np.array(list(itertools.product(h, h, h)))  # all evaluated Miller indices
    g_non_rot = np.dot(hkl, reciprocal_unit_cell)  # all evaluated reciprocal_unit_cell points

    g = np.dot(g_non_rot, rotation_matrix)  # rotate these reciprocal_unit_cell points
    g_norm = vector_norm(g)  # length of all vectors
    not_zero = g_norm > 0
    g = g[not_zero]  # zero reflection will make problems further on, so we exclude it.
    g_non_rot = g_non_rot[not_zero]
    g_norm = g_norm[not_zero]
    hkl = hkl[not_zero]

    # Calculate excitation errors for all reciprocal_unit_cell points
    # Zuo and Spence, 'Adv TEM', 2017 -- Eq 3:14
    S = (k0 ** 2 - vector_norm(g - cent) ** 2) / (2 * k0)
    g_mz = g - k0_vector
    in_sqrt = g_mz[:, 2]**2 + np.linalg.norm(g_mz, axis=1)**2 - k0**2
    in_sqrt[in_sqrt < 0] = 0.
    S2 = -g_mz[:, 2] - np.sqrt(in_sqrt)

    # Determine reciprocal_unit_cell points with excitation error less than the maximum allowed one: Sg_max

    reflections = abs(S) < Sg_max  # This is now a boolean array with True for all possible reflections
    hkl_all = hkl.copy()
    s_g = S[reflections]
    g_hkl = g[reflections]
    
    hkl = hkl[reflections]
    g_hkl_non_rot = g_non_rot[reflections]
    g_norm = g_norm[reflections]

    if verbose:
        print(f"Of the {len(g)} tested reciprocal_unit_cell points, {len(g_hkl)} "
              f"have an excitation error less than {Sg_max:.2f} 1/Angstrom")

    # Calculate Structure Factors
    base = atoms.positions
    structure_factors = []
    for j in range(len(g_hkl)):
        F = 0
        for b in range(len(atoms)):
            f = feq(atoms[b].symbol, np.linalg.norm(g_hkl[j]))
            F += f * np.exp(-2 * np.pi * 1j * (g_hkl_non_rot[j] * atoms.positions[b]).sum())

        structure_factors.append(F)

    F = structure_factors = np.array(structure_factors)

    # Sort reflection in allowed and forbidden #
    allowed = np.absolute(F) > 0.000001  # allowed within numerical error

    if verbose:
        print(f"Of the {hkl.shape[0]} possible reflection {allowed.sum()} are allowed.")

    # information of allowed reflections
    s_g_allowed = s_g[allowed]
    hkl_allowed = hkl[allowed][:]
    g_allowed = g_hkl[allowed, :]
    F_allowed = F[allowed]
    g_norm_allowed = g_norm[allowed]

    atoms.info['diffraction'] = {}
    dif = atoms.info['diffraction']
    dif['allowed'] = {}
    dif['allowed']['Sg'] = s_g_allowed
    dif['allowed']['hkl'] = hkl_allowed
    dif['allowed']['g'] = g_allowed
    dif['allowed']['structure factor'] = F_allowed

    # Calculate Extinction Distance  Reimer 7.23
    # - makes only sense for non-zero F

    xi_g = np.real(np.pi * volume_unit_cell * k0 / F_allowed)

    # Calculate Intensity of beams  Reimer 7.25
    if 'thickness' not in tags:
        tags['thickness'] = 0.
    thickness = tags['thickness']
    if thickness > 0.1:
        I_g = np.real(np.pi ** 2 / xi_g ** 2 * np.sin(np.pi * thickness * s_g_allowed) ** 2 / (np.pi * s_g_allowed)**2)
        dif['allowed']['Ig'] = I_g

    dif['allowed']['intensities'] = intensities = np.real(F_allowed) ** 2

    # information of forbidden reflections
    forbidden = np.logical_not(allowed)
    s_g_forbidden = s_g[forbidden]
    hkl_forbidden = hkl[forbidden]
    g_forbidden = g_hkl[forbidden]
    F_forbidden = F[forbidden]

    dif['forbidden'] = {}
    dif['forbidden']['Sg'] = s_g_forbidden
    dif['forbidden']['hkl'] = hkl_forbidden.copy()
    dif['forbidden']['g'] = g_forbidden
    
    # Make pretty labels
    hkl_label = make_pretty_labels(hkl_allowed)
    dif['allowed']['label'] = hkl_label
    hkl_label = make_pretty_labels(hkl_forbidden)
    dif['forbidden']['label'] = hkl_label

    # Dynamically Allowed Reflection
    """
    indices = range(len(hkl_allowed))
    combinations = [list(x) for x in itertools.permutations(indices, 2)]
    hkl_forbidden = hkl_forbidden.tolist()
    dynamically_allowed = np.zeros(len(hkl_forbidden), dtype=bool)
    for [i, j] in combinations:
        possible = (hkl_allowed[i] + hkl_allowed[j]).tolist()
        if possible in hkl_forbidden:
            dynamically
            _allowed[hkl_forbidden.index(possible)] = True
    dif['forbidden']['dynamically_allowed'] = dynamically_allowed
   
    if verbose:
        print(f"Of the {g_forbidden.shape[0]} forbidden reflection {dif['dynamically_allowed']['g'].shape[0]} "
              f"can be dynamically activated.")
        # print(dif['forbidden']['hkl'][dynamically_allowed])
    """

    # Center of Laue Circle
    laue_circle = np.dot(tags['nearest_zone_axis'], tags['reciprocal_unit_cell'])
    laue_circle = np.dot(laue_circle, rotation_matrix)
    laue_circle = laue_circle / np.linalg.norm(laue_circle) * k0
    laue_circle[2] = 0

    dif['Laue_circle'] = laue_circle
    if verbose:
        print('Laue_circle', laue_circle)

    # ###########################
    # Calculate Laue Zones (of allowed reflections)
    # ###########################
    # Below is the expression given in most books.
    # However, that would only work for orthogonal crystal systems
    # Laue_Zone = abs(np.dot(hkl_allowed,tags['zone_hkl']))  # works only for orthogonal systems

    # This expression works for all crystal systems
    # Remember we have already tilted, and so the dot product is trivial and gives only the z-component.

    Laue_Zone = abs(np.dot(hkl_allowed, tags['nearest_zone_axis']))
    dif['allowed']['Laue_Zone'] = Laue_Zone

    ZOLZ = Laue_Zone == 0
    FOLZ = Laue_Zone == 1
    SOLZ = Laue_Zone == 2
    HOLZ = Laue_Zone > 2

    dif['allowed']['ZOLZ'] = ZOLZ
    dif['allowed']['FOLZ'] = FOLZ
    dif['allowed']['SOLZ'] = SOLZ
    dif['allowed']['HOLZ'] = HOLZ

    if verbose:
        print(' There are {0} allowed reflections in the zero order Laue Zone'.format(ZOLZ.sum()))
        print(' There are {0} allowed reflections in the first order Laue Zone'.format((Laue_Zone == 1).sum()))
        print(' There are {0} allowed reflections in the second order Laue Zone'.format((Laue_Zone == 2).sum()))
        print(' There are {0} allowed reflections in the higher order Laue Zone'.format((Laue_Zone > 2).sum()))

    if verbose:
        print(' hkl  \t Laue zone \t Intensity (*1 and \t log) \t length \n')
        for i in range(len(hkl_allowed)):
            print(' {0} \t {1} \t {2:.3f} \t  {3:.3f} \t  {4:.3f}   '.format(hkl_allowed[i],
                                                                             g_allowed[i], intensities[i],
                                                                             np.log(intensities[i] + 1),
                                                                             g_norm_allowed[i]))

    ####################################
    # Calculate HOLZ and Kikuchi Lines #
    ####################################

    tags_new_zone = tags.copy()
    tags_new_zone['mistilt_alpha'] = 0
    tags_new_zone['mistilt_beta'] = 0

    for i in range(1):  # tags['nearest_zone_axes']['amount']):

        zone_tags = tags['nearest_zone_axes'][str(i)]

        if verbose:
            print('Calculating Kikuchi lines for zone: ', zone_tags['hkl'])

        laue_circle = np.dot(zone_tags['hkl'], tags['reciprocal_unit_cell'])
        laue_circle = np.dot(laue_circle, rotation_matrix)
        laue_circle = laue_circle / np.linalg.norm(laue_circle) * k0
        laue_circle[2] = 0

        zone_tags['Laue_circle'] = laue_circle
        # Rotate to nearest zone axis

        tags_new_zone['zone_hkl']

        theta = -(zone_tags['mistilt_alpha'])
        phi = -(zone_tags['mistilt_beta'])

        # first we rotate phi about z-axis
        c, s = np.cos(phi), np.sin(phi)
        rot_z = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        # second we rotate theta about y-axis
        c, s = np.cos(theta), np.sin(theta)
        rot_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        # the rotation now makes z-axis coincide with plane normal

        rotation_matrix2 = np.dot(rot_z, rot_y)

        g_kikuchi_all = np.dot(g, rotation_matrix2)
        ZOLZ = abs(g_kikuchi_all[:, 2]) < 1

        g_kikuchi = g_kikuchi_all[ZOLZ]
        S = (k0 ** 2 - vector_norm(g_kikuchi - np.array([0, 0, k0])) ** 2) / (2 * k0)
        reflections = abs(S) < 0.1  # This is now a boolean array with True for all possible reflections
        g_hkl_kikuchi2 = g_kikuchi[reflections]
        hkl_kikuchi2 = (hkl_all[ZOLZ])[reflections]

        structure_factors = []
        for j in range(len(g_hkl_kikuchi2)):
            F = 0
            for b in range(len(atoms)):
                f = feq(atoms[b].symbol, np.linalg.norm(g_hkl_kikuchi2[j]))
                F += f * np.exp(-2 * np.pi * 1j * (g_hkl_kikuchi2[j] * atoms.positions[b]).sum())

            structure_factors.append(F)

        F = np.array(structure_factors)

        allowed_kikuchi = np.absolute(F) > 0.000001

        g_hkl_kikuchi = g_hkl_kikuchi2[allowed_kikuchi]
        hkl_kikuchi = hkl_kikuchi2[allowed_kikuchi]

        gd2 = g_hkl_kikuchi / 2.
        gd2[:, 2] = 0.

        # calculate and save line in Hough space coordinates (distance and theta)
        slope2 = gd2[:, 0] / (gd2[:, 1] + 1e-20)
        distance2 = np.sqrt(gd2[:, 0] * gd2[:, 0] + gd2[:, 1] * gd2[:, 1])
        theta2 = np.arctan(slope2)

        dif['Kikuchi'] = {}
        dif['Kikuchi']['slope'] = slope2
        dif['Kikuchi']['distance'] = distance2
        dif['Kikuchi']['theta'] = theta2
        dif['Kikuchi']['hkl'] = hkl_kikuchi
        dif['Kikuchi']['g_hkl'] = g_hkl_kikuchi
        dif['Kikuchi']['g_deficient'] = gd2
        dif['Kikuchi']['min dist'] = gd2 + laue_circle

    k_g = k0

    # Dynamic Correction
    # Does not correct ZOLZ lines !!!!
    # Equation Spence+Zuo 3.86a
    if 'dynamic correction' in tags:
        if tags['dynamic correction']:
            gamma_1 = - 1. / (2. * k0) * (intensities / (2. * k0 * s_g_allowed)).sum()
            if verbose:
                print('Dynamic correction gamma_1: ', gamma_1)

            # Equation Spence+Zuo 3.84
            k_g = k0 - k0 * gamma_1 / g_allowed[:, 2]

    # k_g = np.dot( [0,0,k0],  rotation_matrix)
    # Calculate angle between k0 and deficient cone vector
    # For dynamic calculations k0 is replaced by k_g
    d_theta = np.arcsin(g_norm_allowed / k_g / 2.) - np.arcsin(np.abs(g_allowed[:, 2]) / g_norm_allowed)

    # calculate length of distance of deficient cone to k0 in ZOLZ plane
    gd_length = 2 * np.sin(d_theta / 2) * k0

    # Calculate nearest point of HOLZ and Kikuchi lines
    gd = g_allowed.copy()
    gd[:, 0] = -gd[:, 0] * gd_length / g_norm_allowed
    gd[:, 1] = -gd[:, 1] * gd_length / g_norm_allowed
    gd[:, 2] = 0.

    # calculate and save line in Hough space coordinates (distance and theta)
    slope = gd[:, 0] / (gd[:, 1] + 1e-20)
    distance = gd_length
    theta = np.arctan(slope)

    dif['HOLZ'] = {}
    dif['HOLZ']['slope'] = slope
    # a line is now given by
    dif['HOLZ']['distance'] = distance
    dif['HOLZ']['theta'] = theta
    dif['HOLZ']['g_deficient'] = gd
    dif['HOLZ']['g_excess'] = gd + g_allowed
    dif['HOLZ']['g_allowed'] = g_allowed.copy()

    dif['HOLZ']['ZOLZ'] = ZOLZ
    dif['HOLZ']['HOLZ'] = np.logical_not(ZOLZ)
    dif['HOLZ']['FOLZ'] = FOLZ
    dif['HOLZ']['SOLZ'] = SOLZ
    dif['HOLZ']['HHOLZ'] = HOLZ  # even higher HOLZ

    dif['HOLZ']['hkl'] = dif['allowed']['hkl']
    dif['HOLZ']['intensities'] = intensities

    print('done')


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
            h_string = r'[$\bar {' + str(int(-h)) + '},'
        else:
            h_string = r'[$\bar {' + str(int(h)) + '},'
        if k < 0:
            k_string = r'\bar {' + str(int(-k)) + '},'
        else:
            k_string = str(int(k)) + ','
        if hex_label:
            ii = -(h + k)
            if ii < 0:
                k_string = k_string + r'\bar {' + str(int(-ii)) + '},'
            else:
                k_string = k_string + str(int(ii)) + ','
        if l < 0:
            l_string = r'\bar {' + str(int(-l)) + '} $]'
        else:
            l_string = str(int(l)) + '} $]'
        label = h_string + k_string + l_string
        hkl_label.append(label)
    return hkl_label


def feq(element, q):
    """Atomic form factor parametrized in 1/Angstrom but converted to 1/Angstrom

    The atomic form factor is from Kirkland: Advanced Computing in Electron Microscopy 2nd edition, Appendix C.
    From Appendix C of Kirkland, "Advanced Computing in Electron Microscopy", 3Ard ed.
    Calculation of electron form factor for specific q:
    Using equation Kirkland C.15

    Parameters
    ----------
    element: string
        element name
    q: float
        magnitude of scattering vector in 1/Angstrom -- (=> exp(-i*g.r), physics negative convention)

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

    # q is in magnitude of scattering vector in 1/A -- (=> exp(-i*g.r), physics negative convention)
    param = electronFF[element]
    f_lorentzian = 0
    f_gauss = 0
    for i in range(3):
        f_lorentzian += param['fa'][i]/(q**2 + param['fb'][i])
        f_gauss += param['fc'][i]*np.exp(-q**2 * param['fd'][i])

    # Conversion factor from scattering factors to volts. h^2/(2pi*m0*e), see e.g. Kirkland eqn. C.5
    # !NB RVolume is already in A unlike RPlanckConstant
    # scattering_factor_to_volts=(PlanckConstant**2)*(AngstromConversion**2)/(2*np.pi*ElectronMass*ElectronCharge)
    return f_lorentzian+f_gauss  # * scattering_factor_to_volts
