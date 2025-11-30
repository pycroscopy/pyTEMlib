"""
# ###############################################################
# Utility Functions for spectroscopy data in pyTEMlib
# ################################################################
"""
import typing
import warnings
import numpy as np
from numba import jit
import scipy
import sidpy

from .xrpa_x_sections import x_sections

ELECTRON_REST_ENERGY = 5.10998918e5  # electron rest energy in eV

major_edges = ['K1', 'L3', 'M5', 'N5']

all_edges = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5',
             'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2',
             'O3', 'O4', 'O5', 'O6', 'O7', 'P1', 'P2', 'P3']

shell_occupancy = {'K1': 2, 'L1': 2, 'L2': 2, 'L3': 4, 'M1': 2, 'M2': 2, 'M3': 4, 'M4': 4, 'M5': 6,
                   'N1': 2, 'N2': 2, 'N3': 4, 'N4': 4, 'N5': 6, 'N6': 6, 'N7': 8, 'O1': 2, 'O2': 2,
                   'O3': 4, 'O4': 4, 'O5': 6, 'O6': 6, 'O7': 8, 'O8': 8, 'O9': 10}


first_close_edges = ['K1', 'L3', 'M5', 'M3', 'N5', 'N3']

elements = [' ', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
            'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']

def get_wavelength(acceleration_voltage: float) -> float:
    """
    Calculates the relativistic corrected de Broglie wavelength of an electron in meter

    Parameter:
    ---------
    acceleration_voltage: float
        acceleration voltage in volt
    Returns:
    -------
    wavelength: float
        wave length in meter
    """
    if not isinstance(acceleration_voltage, (int, float)):
        raise TypeError('Acceleration voltage has to be a real number')

    ev = acceleration_voltage * scipy.constants.elementary_charge
    h = scipy.constants.Planck
    m0 = scipy.constants.electron_mass
    c = scipy.constants.speed_of_light
    wavelength = h / np.sqrt(2 * m0 * ev * (1 + (ev / (2 * m0 * c ** 2))))
    return wavelength


def get_wave_length(acceleration_voltage: float) -> float:
    """Deprecated function, use get_wavelength instead"""
    warnings.warn("get_wave_length is deprecated, use get_wavelength instead",
                  DeprecationWarning,
                  stacklevel=2)
    return get_wavelength(acceleration_voltage)


def depth_of_focus(acceleration_voltage: float, convergence_angle: float) -> float:
    """calculate depth of focus

    Parameters
    ----------
    acceleration_voltage : float
        acceleration voltage in eV
    convergence_angle : float
        convergence angle in radians

    Returns
    -------
    float
        depth of focus in meters
    """

    wavelength = get_wavelength(acceleration_voltage)
    return wavelength / convergence_angle**2


def current_to_number_of_electrons(current: float) -> float:
    """convert current in Ampere to number of electrons per second

    Parameters
    ----------
    current : float
        current in Ampere

    Returns
    -------
    float
        number of electrons per second
    """
    return current / scipy.constants.elementary_charge



def effective_collection_angle(energy_scale: np.ndarray,
                               alpha: float,
                               beta: float,
                               beam_ev: float) -> float:
    """Calculates the effective collection angle in mrad:

    Translate from original Fortran program
    Calculates the effective collection angle in mrad:
    Parameter
    ---------
    energy_scale: numpy array
        first and last energy loss of spectrum in eV
    alpha: float
        convergence angle in mrad
    beta: float
        collection  angle in mrad
    beamKV: float
        acceleration voltage in V

    Returns
    -------
    eff_beta: float
        effective collection angle in mrad

    # function y = effbeta(ene,  alpha, beta, beam_kv) Note Pierre uses keV
    #
    #       This program computes etha(alpha,beta), that is the collection
    #       efficiency associated to the following geometry :
    #
    #       alpha = half angle of illumination  (0 -> pi/2)
    #       beta  = half angle of collection    (0 -> pi/2)
    #                                           (pi/2 = 1570.795 mrad)
    #
    #           A constant angular distribution of incident electrons is assumed
    #       for any incident angle (-alpha,alpha). These electrons imping the
    #       target and a single energy-loss event occurs, with a characteristic
    #       angle theta-e (relativistic). The angular distribution of the
    #       electrons after the target is analytically derived.
    #           This program integrates this distribution from theta=0 up to
    #       theta=beta with an adjustable angular step.
    #           This program also computes beta* which is the theoretical
    #       collection angle which would give the same value of etha(alpha,beta)
    #       with a parallel incident beam.
    #
    #       subroutines and function subprograms required
    #       ---------------------------------------------
    #       none
    #
    #       comments
    #       --------
    #
    #       The following parameters are asked as input :
    #        accelerating voltage (kV), energy loss range (eV) for the study,
    #        energy loss step (eV) in this range, alpha (mrad), beta (mrad).
    #       The program returns for each energy loss step :
    #        alpha (mrad), beta (mrad), theta-e (relativistic) (mrad),
    #        energy loss (eV), etha (#), beta * (mrad)
    #
    #       author :
    #       --------
    #       Pierre TREBBIA
    #       US 41 : "Microscopie Electronique Analytique Quantitative"
    #       Laboratoire de Physique des Solides, Bat. 510
    #       Universite Paris-Sud, F91405 ORSAY Cedex
    #       Phone : (33-1) 69 41 53 68
    #
    """
    if beam_ev == 0:
        beam_ev = 100.0 * 1e3

    if alpha == 0:
        return beta

    if beta == 0:
        return alpha

    alpha = alpha * 0.001  # rad
    beta = beta * 0.001  # rad
    z7 = 500.0  # number of integration steps to be modified at will

    #       main loop on energy loss
    for zx in range(int(energy_scale[0]), int(energy_scale[-1]), 100):
        # ! zx = current energy loss
        eta = 0.0
        # x0 = relativistic theta-e
        x0 = float(zx) * (beam_ev + 511060.) / (beam_ev * (beam_ev + 1022120.))
        dtheta = (beta - 0.1 * np.sqrt((x0**2 + alpha**2))) / 500  # integration steps
    #
    #        calculation of the analytical expression
    #
    for zi in range(1, int(z7)):
        theta = 0.1 * np.sqrt((x0**2 + alpha**2)) + dtheta * float(zi)
        x5 = theta**2
        x6 = 4. * x5 * x0 * x0
        x7 = (x0**2 + alpha**2) - x5
        eta += 2. * theta * dtheta * np.log((np.sqrt(x7**2 + x6) + x7) / (2. * x0**2))
    # addition of the central contribution
    eta = eta + (x0**2 + alpha**2) / 100. * np.log(1. + alpha**2/x0**2)
    # normalisation
    eta = eta / alpha * alpha * np.log(1. + np.pi**2 / (4. * x0**2))
    #
    #        correction by geometrical factor (beta/alpha)**2
    #
    if beta < alpha:
        x5 = alpha / beta
        eta = eta * x5**2

    #  etha2 = eta * 100.
    #
    #        calculation of beta *
    #
    x6 = np.power((1. + (1. + np.pi**2 / (4. * x0**2))), eta)
    x7 = x0 * np.sqrt(x6 - 1.)
    beta = x7 * 1000.  # in mrad

    return beta

def set_default_metadata(current_dataset: sidpy.Dataset) -> None:
    """sets default metadata for the dataset"""

    if 'experiment' not in current_dataset.metadata:
        current_dataset.metadata['experiment'] = {}
    if 'convergence_angle' not in current_dataset.metadata['experiment']:
        current_dataset.metadata['experiment']['convergence_angle'] = 30
    if 'collection_angle' not in current_dataset.metadata['experiment']:
        current_dataset.metadata['experiment']['collection_angle'] = 50
    if 'acceleration_voltage' not in current_dataset.metadata['experiment']:
        current_dataset.metadata['experiment']['acceleration_voltage'] = 200000


def lorentz(x, center, amplitude, width):
    """ Lorentzian Function """
    lorentz_peak = 0.5 * width / np.pi / ((x - center)**2 + (width / 2)**2)
    return amplitude * lorentz_peak / lorentz_peak.max()

@jit
def gauss(x, p):  # p[0]==mean, p[1]= amplitude p[2]==fwhm,
    """Gaussian Function

        p[0]==mean, p[1]= amplitude p[2]==fwhm
        area = np.sqrt(2* np.pi)* p[1] * np.abs(p[2] / 2.3548)
        FWHM = 2 * np.sqrt(2 np.log(2)) * sigma = 2.3548 * sigma
        sigma = FWHM/3548
    """
    if p[2] == 0:
        return x * 0.
    return p[1] * np.exp(-(x - p[0])**2 / (2.0 * (p[2] / 2.3548)**2))

def get_atomic_number(z):
    """Returns the atomic number independent of input as a string or number"""
    return get_z(z)

def get_z(z: typing.Union[int, str]) -> int:
    """Returns the atomic number independent of input as a string or number

    Parameter
    ---------
    z: int, str
        atomic number of chemical symbol (0 if not valid)
    Return:
    ------
    z_out: int
        atomic number
    """
    z_out = 0
    if str(z).isdigit():
        z_out = int(z)
    elif isinstance(z, str):
        z_out = elements.index(z)
    else:
        raise TypeError('A valid element string or number is required')
    return z_out


def get_element_symbol(z: typing.Union[int, str]) -> str:
    """Returns the element symbol independent of input as a string or number

    Parameter
    ---------
    z: int, str
        atomic number of chemical symbol (0 if not valid)
    Return:
    ------
    symbol: str
        element symbol
    """
    z_num = get_z(z)
    if 0 < z_num < len(elements):
        return elements[int(z_num)]
    return ''


def get_x_sections(z: int=0) -> dict:
    """Reads X-ray fluorescent cross-sections from a dictionary.

    Parameters
    ----------
    z: int
        atomic number if zero all cross-sections will be returned

    Returns
    -------
    dictionary
        cross-section of an element or of all elements if z = 0
    """
    if z < 1:
        return x_sections
    z = str(z)
    if z in x_sections:
        return x_sections[z]
    return {}


def get_spectrum(dataset, x=0, y=0, bin_x=1, bin_y=1):
    """
    Extracts a spectrum from a sidpy.Dataset object
    Parameter
    ---------
    dataset: sidpy.Dataset object
        contains spectrum or spectrum image
    x: int default = 0
        x position of spectrum image
    y: int default = 0
        y position of spectrum
    bin_x: int default = 1
        binning of spectrum image in x-direction
    bin_y: int default = 1
        binning of spectrum image in y-direction

    Returns:
    --------
    spectrum: sidpy.Dataset object

    """
    if dataset.data_type.name == 'SPECTRUM':
        spectrum = dataset.copy()
    else:
        image_dims = dataset.get_image_dims()
        x = min(x, dataset.shape[image_dims[0]] - bin_x)
        y = min(y, dataset.shape[image_dims[1]] - bin_y)
        selection = []
        dimensions = dataset.get_dimension_types()
        for dim, dimension_type in enumerate(dimensions):
            # print(dim, axis.dimension_type)
            if dimension_type == 'SPATIAL':
                if dim == image_dims[0]:
                    selection.append(slice(x, x + bin_x))
                else:
                    selection.append(slice(y, y + bin_y))
            elif dimension_type == 'SPECTRAL':
                selection.append(slice(None))
            elif dimension_type == 'CHANNEL':
                selection.append(slice(None))
            else:
                selection.append(slice(0, 1))
        spectrum = dataset[tuple(selection)].mean(axis=tuple(image_dims))
        spectrum.squeeze().compute()
        spectrum.data_type = 'Spectrum'
    return spectrum

def second_derivative(dataset: sidpy.Dataset) -> None:
    """Calculates second derivative of a sidpy.dataset"""
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0]
    if dataset.data_type.name == 'SPECTRAL_IMAGE':
        spectrum = dataset.view.get_spectrum()
    else:
        spectrum = np.array(dataset)
    spec = scipy.ndimage.gaussian_filter(spectrum, 3)
    dispersion = energy_scale.slope
    second_dif = np.roll(spec, -3) - 2 * spec + np.roll(spec, +3)
    second_dif[:3] = 0
    second_dif[-3:] = 0

    # find if there is a strong edge at high energy_scale
    noise_level = 2. * np.std(second_dif[3:50])
    [indices, _] = scipy.signal.find_peaks(second_dif, noise_level)
    width = max(50 / dispersion, 50)
    start_end_noise = int(len(energy_scale) - width)
    for index in indices[::-1]:
        if index > start_end_noise:
            start_end_noise = index - 70

    # noise_level_start = sensitivity * np.std(second_dif[3:50])
    # noise_level_end = sensitivity * np.std(second_dif[start_end_noise: start_end_noise + 50])
    # slope = (noise_level_end - noise_level_start) / (len(energy_scale) - 400)
    # noise_level = noise_level_start #+ np.arange(len(energy_scale)) * slope
    return second_dif , noise_level

def get_rotation_matrix(angles, in_radians=False):
    """ Rotation of zone axis by mistilt

        Parameters
        ----------
        angles: ist or numpy array of float
            list of mistilt angles (default in degrees)
        in_radians: boolean default False
            default is angles in degrees

        Returns
        -------
        rotation_matrix: np.ndarray (3x3)
            rotation matrix in 3d
        """

    if not isinstance(angles, (np.ndarray, list)):
        raise TypeError('angles must be a list of float of length 3')
    if len(angles) != 3:
        raise TypeError('angles must be a list of float of length 3')

    if in_radians:
        alpha, beta, gamma = angles
    else:
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
    return np.dot(np.dot(rot_x, rot_y), rot_z)
