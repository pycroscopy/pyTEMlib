"""
basic diffraction tools
part of diffraction tools in pyTEMlib
by Gerd Duscher

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
from warnings import warn
import itertools

import numpy as np
import scipy
import matplotlib.pylab as plt  # basic plotting

import ase
import ase.build

import pyTEMlib
from pyTEMlib.crystal_tools import electronFF
from ..utilities import get_wavelength, get_rotation_matrix


inputKeys = ['acceleration_voltage', 'zone_hkl', 'Sg_max', 'hkl_max']
optional_input_keys = ['crystal', 'lattice_parameter_nm', 'convergence_angle_mrad',
                       'mistilt', 'thickness', 'dynamic correction', 'dynamic correction K0']


def read_poscar(filename):
    """ Deprecated - use pyTEMlib.file_tools.read_poscar"""
    print('read_poscar and read_cif moved to file_tools, \n'
          'please use that library in the future!')
    pyTEMlib.file_tools.read_poscar(filename)


def example(verbose=True):
    """
    same as zuo_fig_3_18
    """
    print('\n##########################')
    print('# Start of Example Input #')
    print('##########################\n')
    print('Define only mandatory input: ', inputKeys)
    print(' Kinematic diffraction routine will set optional input : ', optional_input_keys)

    return zuo_fig_3_18(verbose=verbose)


def zuo_fig_3_18(verbose=True):
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
        atoms.info['output'] = output: dictionary:
            dictionary of all input and output parameter needed to reproduce that figure.
    """

    # INPUT
    # Create Silicon structure (Could be produced with Silicon routine)
    if verbose:
        print('Sample Input for Figure 3.18 in Zuo and Spence \"Advanced TEM\", 2017')
    a = 5.14  # A
    atoms = ase.build.bulk('Si', 'diamond', a=a, cubic=True)

    experiment = {'acceleration_voltage': 99.2 * 1000.0,  # V
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
              'color_Laue_Zones': ['red', 'blue', 'green', 'blue', 'green'],  
              'color_Kikuchi': 'green',
              'linewidth_HOLZ': -1,  # -1: linewidth according to intensity (structure factor F^2)
              'linewidth_Kikuchi': -1,  # linewidth according to intensity (structure factor F^2)
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
    if not isinstance(zone, (np.ndarray, list)):
        raise TypeError('Miller indices must be a list of int of length 3')

    rotation_matrix = get_rotation_matrix(angles)
    return np.dot(zone, rotation_matrix)


def get_metric_tensor(matrix):
    """The metric tensor of the lattice."""
    metric_tensor2 = np.dot(matrix, matrix.T)
    return metric_tensor2


def vector_norm(g):
    """ Length of vector
    depreciated - use np.linalg.norm
    """
    return np.linalg.norm(g)


def intensity_with_thickness(thickness, atoms, k0_magnitude, f_allowed, sg):
    """ Calculate intensity of diffracted beams according to Reimer&Kohl equ.7.25 

    Parameters:
    ----------
    thickness: float
        thickness of sample in Angstrom
    atoms: ase.Atoms
        structure of crystal
    k0_magnitude: float
        incident wave vector in 1/A
    f_allowed: np.array (1dim)
        structure factors (non-zero =  allowed)
    sg: np.array (1dm)
        excitation errors 
    Returns:
    --------
    i_g: np.array (1dim)
        intensities of g vectors at excitation error sg
    """

    if thickness < 0:
        return None
    # Calculate Extinction Distance  Reimer 7.23
    # - makes only sense for non-zero structure_factor
    xi_g = np.real(np.pi * atoms.cell.volume * k0_magnitude / f_allowed)
    s_eff = np.sqrt(sg**2 + xi_g**-2)

    i_g = np.real(np.pi**2 / xi_g**2 * np.sin(np.pi * s_eff * thickness)**2 / (np.pi * s_eff)**2)
    return i_g


def get_all_miller_indices(hkl_max):
    """ Get all Miller indices up to hkl_max"""
    h = np.linspace(-hkl_max, hkl_max, 2 * hkl_max + 1)  # all evaluated single Miller Indices
    hkl = np.array(list(itertools.product(h, h, h)))  # all evaluated Miller indices

    # delete [0,0,0]
    index_center = np.where(np.linalg.norm(hkl, axis=1) < 1e-6)[0]
    hkl = np.delete(hkl, index_center, axis=0)  # delete [0,0,0]
    return hkl


def get_all_g_vectors(hkl_max, atoms):
    """ Get all reflections up to a maximum Miller index"""
    hkl = get_all_miller_indices(hkl_max)
    g = np.dot(hkl, atoms.cell.reciprocal())  # all evaluated reciprocal lattice points
    return g, hkl

def get_structure_factors(atoms, g_hkl):
    """ Get structure factors for given reciprocal lattice points g_hkl"""

    form_factor = np.zeros((len(atoms.positions), g_hkl.shape[0]))
    for symbol in np.unique(atoms.symbols):
        atom_positions = atoms.symbols==symbol
        form_factor[(atom_positions)] = get_form_factor(symbol, np.linalg.norm(g_hkl, axis=1))
    structure_factors = calculate_structure_factors(np.array(g_hkl),
                                                    form_factor,
                                                    np.array(atoms.positions))
    return structure_factors # np.array(structure_factors).sum(axis=0)

def calculate_structure_factors(all_g, form_factor, atom_positions):
    """ Calculate structure factors for given reciprocal lattice points g_hkl"""
    structure_factors = np.zeros(len(all_g))
    for i, g_i in enumerate(all_g):
        struct_factor = 0.0+0.0*1j
        for j, f_j in enumerate(form_factor):
            f_q_j = f_j[i]
            r_j = atom_positions[j]
            struct_factor += f_q_j * (np.exp(-2*np.pi*1j*np.dot(g_i, r_j))).sum()
        structure_factors[i] = (struct_factor*struct_factor.conj()).real/(form_factor[:,i]).sum()
    return np.array(structure_factors)


def get_structure_factors_old(atoms, g_hkl):
    """ Calculate structure factors for given reciprocal lattice points g_hkl"""
    g_norm = np.linalg.norm(g_hkl, axis=1)
    structure_factors = []
    for j, g in enumerate(g_hkl):
        structure_factor = 0
        for atom in atoms:
            # Atomic form factor for element and momentum change (g vector)
            f = feq(atom.symbol, g_norm[j])
            structure_factor += f * np.exp(-2*np.pi*1j*(g*atom.position).sum())
        structure_factors.append(structure_factor)
    return np.array(structure_factors)


def get_inner_potential(atoms):
    """ inner potential in Volts """
    u_0 = 0  # in (Ang)
    # atom form factor of zero reflection angle is the inner potential in 1/A
    for atom in atoms:
        u_0 += get_form_factor(atom.symbol, 0)
    scattering_factor_to_volts = ((scipy.constants.h*1e10)**2
                                  / (2 * np.pi * scipy.constants.m_e * scipy.constants.e)
                                  / atoms.cell.volume)
    return u_0 * scattering_factor_to_volts

def get_incident_wave_vector(atoms, acceleration_voltage, verbose=False):
    """ Incident wave vector K0 in vacuum and material"""
    u0 = 0.0  # in (Ang)
    # atom form factor of zero reflection angle is the
    # inner potential in 1/A
    for atom in atoms:
        u0 += get_form_factor(atom.symbol, 0.0)

    e = scipy.constants.elementary_charge
    h = scipy.constants.Planck
    m0 = scipy.constants.electron_mass

    volume_unit_cell = atoms.cell.volume
    if verbose:
        scattering_factor_to_volts = (h**2) * (1e10**2) / (2 * np.pi * m0 * e) * volume_unit_cell
        print(f'The inner potential is {u0* scattering_factor_to_volts:.1f} V')

    # Calculating incident wave vector magnitude 'k0' in material
    wavelength = get_wavelength(acceleration_voltage, unit='A')  # in Angstrom
    incident_wave_vector = np.sqrt(1 / wavelength**2 + u0/volume_unit_cell)  # 1/Ang
    return incident_wave_vector


def ewald_sphere_center(acceleration_voltage, atoms, zone_hkl):
    """ Ewald sphere center in 1/Angstrom """
    incident_wave_vector = get_incident_wave_vector(atoms, acceleration_voltage)
    center = np.dot(zone_hkl, atoms.cell.reciprocal())
    center = center / np.linalg.norm(center) * incident_wave_vector
    return center


def get_cylinder_coordinates (zone_hkl, g, k0_magnitude):
    """ Get cylindrical coordinates of g vectors around zone axis"""
    theta, phi = find_angles(zone_hkl)
    rotation_matrix = get_rotation_matrix([-phi, theta, 0], in_radians=True)
    center_rotated = [0, 0, k0_magnitude]

    g_rotated = np.dot(g, rotation_matrix)
    return  np.stack([np.arccos((g_rotated[:, 2]+k0_magnitude)
                                /np.linalg.norm(g_rotated+center_rotated, axis=1)),
                      np.arctan2(g_rotated[:, 1], g_rotated[:, 0]),
                      g_rotated[:, 2],
                      np.linalg.norm(g, axis=1)],
                     axis=-1)


def find_nearest_zone_axis(tags):
    """Test all zone axis up to a maximum of hkl_max"""

    hkl_max = 5
    # Make all hkl indices
    zones_hkl = get_all_miller_indices(hkl_max)

    # make zone axis in reciprocal space
    # all evaluated reciprocal_unit_cell points
    zones_g = np.dot(zones_hkl, tags['reciprocal_unit_cell'])

    # make zone axis in microscope coordinates of reciprocal space
    zones_g = np.dot(zones_g, tags['rotation_matrix'])  # rotate these reciprocal_unit_cell points

    # calculate angles with z-axis
    zones_g_norm = vector_norm(zones_g)
    z_axis = np.array([0, 0, 1])

    zones_angles = np.abs(np.arccos(np.dot((np.array(zones_g).T / zones_g_norm).T, z_axis)))

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
        beta = np.arctan(zone[1] / zone[2])
    return alpha, beta


def stage_rotation_matrix(alpha, beta, gamma=0.):
    """ Microscope stage coordinate system """

    # FIRST we rotate beta about x-axis
    angles = [beta, alpha, gamma]
    return get_rotation_matrix(angles, in_radians=True)


# ##################
# Determine rotation matrix to tilt zone axis onto z-axis
# We determine spherical coordinates to do that
# ##################


def get_zone_rotation(tags):
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
    for key in ['acceleration_voltage']:
        if key not in tags:
            print(f'Necessary parameter {key} not defined')
            stop = True
    output.setdefault('SpotPattern', False)
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
        tags['mistilt_alpha'] = np.deg2rad(tags.get('mistilt_alpha degree', 0.0))
        tags['mistilt_beta'] = np.deg2rad(tags.get('mistilt_beta degree', 0.0))
        tags.setdefault('convergence_angle_mrad', 0.)
        tags.setdefault('thickness', 0.)
        tags.setdefault('dynamic correction', 0.)
        tags.setdefault('dynamic correction K0', 0.)

        if verbose_level > 0:
            print('Input: ')
            print(f"tags['mistilt_alpha'] = {tags['mistilt_alpha']*1000:.2f}mrad")
            print(f"tags['mistilt_beta'] = {tags['mistilt_beta']*1000:.2f}mrad")
            print(f"tags['convergence_angle_mrad'] = {tags['convergence_angle_mrad']}")
            print(f"tags['thickness'] = {tags['thickness']}")
            print(f"tags['dynamic correction'] = {tags['dynamic correction']}")
            print(f"tags['dynamic correction K0'] = {tags['dynamic correction K0']}")
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

    n = 20
    x = np.linspace(-n, n, 2 * n + 1)  # all evaluated multiples of x
    xyz = np.array(list(itertools.product(x, x, x)))  # all evaluated multiples in all direction

    mat = np.dot(xyz, unit_cell)  # all evaluated unit_cells

    atom = {}
    all_distances = np.array([])
    for i, coordinate in enumerate(np.dot(base, unit_cell)):
        distances = np.linalg.norm(mat + coordinate, axis=1)
        unique, counts = np.unique(distances, return_counts=True)

        atom[str(i)] = dict(zip(unique, counts))
        print(atom[str(i)])

        all_distances = np.append(all_distances, distances)
    unique, counts = np.unique(all_distances, return_counts=True)

    plt.plot(unique, counts)
    plt.show()


def gaussian(xx, pp):
    """ Gaussian function"""
    s1 = pp[2] / 2.3548
    prefactor = 1.0 / np.sqrt(2 * np.pi * s1 ** 2)
    y = (pp[1] * prefactor) * np.exp(-(xx - pp[0]) ** 2 / (2 * s1 ** 2))
    return y


def get_unit_cell(atoms, tags):
    """ Unit cell and reciprocal unit cell"""
    unit_cell = atoms.cell.array
    tags['unit_cell'] = unit_cell
    # converts hkl to g vectors and back
    tags['metric_tensor'] = get_metric_tensor(unit_cell)

    # We use the linear algebra package of numpy to invert the unit_cell "matrix"
    tags['reciprocal_unit_cell'] = atoms.cell.reciprocal()


def output_verbose(atoms, tags):
    """ Verbose output of experimental parameters"""
    print('Experimental Parameters:')
    print(f"Acceleration Voltage: {tags.get('acceleration_voltage', 0)*1000:.1f} kV")
    print(f"Convergence Angle: {tags.get('convergence_angle_mrad', None):.2f} mrad",
          f" = {tags.get('convergence_angle_A-1', None):.2f} 1/Ang")
    print(f"Wavelength: {tags.get('wave_length', 0)*1000:.3f} pm")
    print(f"Incident Wave Vector: {tags.get('incident_wave_vector', 0)*10} 1/nm in material ",
          f"; in vacumm: {1/tags.get('wave_length', 0):.4f} 1/nm")
    print("\n Rotation to tilt zone axis onto z-axis:")
    print(f"Rotation alpha {np.rad2deg(tags.get('y-axis rotation alpha', None)):.1f} degree, "
              f" beta {np.rad2deg(tags.get('x-axis rotation beta', None)):.1f} degree")
    print(f"from zone axis {tags.get('zone_hkl', None)}")
    print(f"Tilting {1} by {np.rad2deg(tags.get('mistilt_alpha', None)):.2f} "
          f" in alpha and {np.rad2deg(tags.get('mistilt_beta', None)):.2f}" +
          " in beta direction results in :")
    nearest = tags.get('nearest_zone_axes', {})
    print('Next nearest zone axes are:')
    for i in range(1, nearest.get('amount', 0)):
        print(f"{nearest[str(i)]['hkl']}: mistilt:",
                f" {np.rad2deg(nearest[str(i)]['mistilt_alpha']):6.2f}, "
                f"{np.rad2deg(nearest[str(i)]['mistilt_beta']):6.2f}")
    print('Center of Ewald sphere ', tags.get('k0_vector', None))
    dif = atoms.info.get('diffraction', {})
    print('Center or Laue circle', dif.get('Laue_circle', None))


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
    """Atomic form factor parametrized in 1/Angstrom"""
    warn('feq is deprecated, please use form_factor instead', DeprecationWarning, stacklevel=2)
    return get_form_factor(element, q)


def get_form_factor(element, q):
    """Atomic form factor parametrized in 1/Angstrom but converted to 1/Angstrom

    The atomic form factor is from reKirkland: Advanced Computing in 
    Electron Microscopy 2nd edition, Appendix C.
    From Appendix C of Kirkland, "Advanced Computing in Electron Microscopy", 3rd ed.
    Calculation of electron form factor for specific q:
    Using equation Kirkland C.15

    Parameters
    ----------
    element: string
        element name
    q: float or numpy.ndarray
        magnitude(s) of scattering vector(s) in 1/Angstrom -- (=> exp(-i*g.r), 
        physics negative convention)

    Returns
    -------
    fL+fG: float
        atomic scattering vector
    """

    if not isinstance(element, str):
        raise TypeError('Element has to be a string')
    if element not in electronFF:
        if len(element) > 2:
            raise TypeError('Please use standard convention for element ',
                            'abbreviation with not more than two letters')
        raise TypeError('Element {element} not known to electron diffraction should')
    if isinstance(q, float):
        q = np.array([q])

    if not isinstance(q, np.ndarray):
        raise TypeError('Magnitude of scattering vector has to be a number',
                        ' or numpy array of type float')

    # q is in magnitude of scattering vector in 1/A -- (=> exp(-i*g.r),
    # physics negative convention)
    parameter_dict = electronFF[element]
    f_parameter = np.array([parameter_dict[key] for key in ['fa', 'fb', 'fc', 'fd']])
    q = (np.array([q, q, q])).T
    f = ((f_parameter[0]/(q**2 + f_parameter[1])).sum(axis=1)
         + (f_parameter[2]*np.exp(-q**2 * f_parameter[3])).sum(axis=1))
    # for i in range(3):
    #    f_lorentzian += param['fa'][i]/(q**2 + param['fb'][i])
    #    f_gauss += param['fc'][i]*np.exp(-q**2 * param['fd'][i])

    # Conversion factor from scattering factors to volts. h^2/(2pi*m0*e),
    # see e.g. Kirkland eqn. C.5
    # !NB RVolume is already in A unlike RPlanckConstant
    # scattering_factor_to_volts=(PlanckConstant**2)*(AngstromConversion**2)
    #                             /(2*np.pi*ElectronMass*ElectronCharge)
    return f  # f_lorentzian + f_gauss  # * scattering_factor_to_volts
