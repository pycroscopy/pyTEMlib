"""
Dynamic Scattering Library for Multi-Slice Calculations

author: Gerd Duscher
"""

import numpy as np
import scipy.constants
import scipy.special

import pyTEMlib.kinematic_scattering as ks         # kinematic scattering Library


def potential_1dim(element, r):
    """ Calculates the projected potential of an atom of element

    The projected potential will be in units of V nm^2,
    however, internally we will use Angstrom instead of nm!
    The basis for these calculations are the atomic form factors of Kirkland 2𝑛𝑑 edition
    following the equation in  Appendix C page 252.

    Parameter
    ---------
    element: str
        name of 'element
    r: numpy array [nxn]
        impact parameters (distances from atom position) in nm

    Returns
    -------
    numpy array (nxn)
        projected potential in units of V nm^2
    """

    # get elementary constants
    a0 = scipy.constants.value('Bohr radius') * 1e10  # in Angstrom
    rydberg_div_e = scipy.constants.value('Rydberg constant times hc in eV')  # in V
    e0 = 2 * rydberg_div_e * scipy.constants.value('Bohr radius') * 1e10      # now in V A

    pre_factor = 2 * np.pi ** 2 * a0 * e0

    param = ks.electronFF[element]  # parametrized form factors
    f_lorentz = r * 0  # Lorentzian term
    f_gauss = r * 0  # Gaussian term
    for i in range(3):
        f_lorentz += param['fa'][i] * scipy.special.k0(2 * np.pi * r * np.sqrt(param['fb'][i]))
        f_gauss += param['fc'][i] / param['fd'][i] * np.exp(-np.pi ** 2 * r ** 2 / param['fd'][i])
    f_lorentz[0, 0] = f_lorentz[0, 1]
    # / 100 is  conversion from V Angstrom^2 to V  nm^2
    return pre_factor * (2 * f_lorentz + f_gauss)  # V Angstrom^2


def potential_2dim(element, nx, ny, n_cell_x, n_cell_y, lattice_parameter, base):
    """Make a super-cell with potentials

    Limitation is that we only place atom potential with single pixel resolution
    """
    n_cell_x = int(2 ** np.log2(n_cell_x))
    n_cell_y = int(2 ** np.log2(n_cell_y))

    pixel_size = lattice_parameter / (nx / n_cell_x)

    a_nx = a_ny = int(1 / pixel_size)
    x, y = np.mgrid[0:a_nx, 0:a_ny] * pixel_size
    a = int(nx / n_cell_x)
    r = x ** 2 + y ** 2

    atom_potential = potential_1dim(element, r)

    potential = np.zeros([nx, ny])

    atom_potential_corner = np.zeros([nx, ny])
    atom_potential_corner[0:a_nx, 0:a_ny] = atom_potential
    atom_potential_corner[nx - a_nx:, 0:a_ny] = np.flip(atom_potential, axis=0)
    atom_potential_corner[0:a_nx, ny - a_ny:] = np.flip(atom_potential, axis=1)
    atom_potential_corner[nx - a_nx:, ny - a_ny:] = np.flip(np.flip(atom_potential, axis=0), axis=1)

    unit_cell_base = np.array(base) * a
    unit_cell_base = np.array(unit_cell_base, dtype=int)

    for pos in unit_cell_base:
        potential = potential + np.roll(atom_potential_corner, shift=np.array(pos), axis=[0, 1])

    for column in range(int(np.log2(n_cell_x))):
        potential = potential + np.roll(potential, shift=2 ** column * a, axis=1)
    for row in range(int(np.log2(n_cell_y))):
        potential = potential + np.roll(potential, shift=2 ** row * a, axis=0)

    return potential


def interaction_parameter(acceleration_voltage):
    """Calculates interaction parameter sigma

    Parameter
    ---------
    acceleration_voltage: float
        acceleration voltage in volt

    Returns
    -------
    interaction parameter: float
        interaction parameter (dimensionless)
    """
    e0 = 510998.95  # m_0 c^2 in eV

    wavelength = ks.get_wavelength(acceleration_voltage)
    e = acceleration_voltage

    return 2. * np.pi / (wavelength * e) * (e0 + e) / (2. * e0 + e)


def get_transmission(potential, acceleration_voltage):
    """ Get transmission function

    has to be multiplied in real space with wave function

    Parameter
    ---------
    potential: numpy array (nxn)
        potential of a layer
    acceleration_voltage: float
        acceleration voltage in V

    Returns
    -------
    complex numpy array (nxn)
    """

    sigma = interaction_parameter(acceleration_voltage)

    return np.exp(1j * sigma * potential)


def get_propagator(size_in_pixel, delta_z, number_layers, wavelength, field_of_view, bandwidth_factor, verbose=True):
    """Get propagator function

    has to be convoluted with wave function after transmission

    Parameter
    ---------
    size_in_pixel: int
        number of pixels of one axis in square image
    delta_z: float
        distance between layers
    number_layers: int
        number of layers to make a propagator
    wavelength: float
        wavelength of incident electrons
    field_of_view: float
        field of view of image
    bandwidth_factor: float
        relative bandwidth to avoid anti-aliasing

    Returns
    -------
    propagator: complex numpy array (layers x size_in_pixel x size_in_pixel)

    """

    k2max = size_in_pixel / field_of_view / 2. * bandwidth_factor
    print(k2max)
    if verbose:
        print(f"Bandwidth limited to a real space resolution of {1.0 / k2max * 1000} pm")
        print(f"   (= {wavelength * k2max * 1000.0:.2f} mrad)  for symmetrical anti-aliasing.")
    k2max = k2max * k2max

    kx, ky = np.mgrid[-size_in_pixel / 2:size_in_pixel / 2, -size_in_pixel / 2:size_in_pixel / 2] / field_of_view
    k_square = kx ** 2 + ky ** 2
    k_square[k_square > k2max] = 0  # bandwidth limiting

    if verbose:
        temp = np.zeros([size_in_pixel, size_in_pixel])
        temp[k_square > 0] = 1
        print(f"Number of symmetrical non-aliasing beams = {temp.sum():.0f}")

    propagator = np.zeros([number_layers, size_in_pixel, size_in_pixel], dtype=complex)
    for i in range(number_layers):
        propagator[i] = np.exp(-1j * np.pi * wavelength * k_square * delta_z[i])

    return propagator


def multi_slice(wave, number_of_unit_cell_z, number_layers, transmission, propagator):
    """Multi-Slice Calculation

    The wave function will be changed iteratively

    Parameters
    ----------
    wave: complex numpy array (nxn)
        starting wave function
    number_of_unit_cell_z: int
        this gives the thickness in multiples of c lattice parameter
    number_layers: int
        number of layers per unit cell
    transmission: complex numpy array
        transmission function
    propagator: complex numpy array
        propagator function

    Returns
    -------
    complex numpy array
    """

    for i in range(number_of_unit_cell_z):
        for layer in range(number_layers):
            wave = wave * transmission[layer]  # transmission  - real space
            wave = np.fft.fft2(wave)
            wave = wave * propagator[layer]  # propagation; propagator is defined in reciprocal space
            wave = np.fft.ifft2(wave)  # back to real space
    return wave


def make_chi(theta, phi, aberrations):
    """
    ###
    # Aberration function chi
    ###
    phi and theta are meshgrids of the angles in polar coordinates.
    aberrations is a dictionary with the aberrations coefficients
    Attention: an empty aberration dictionary will give you a perfect aberration
    """

    chi = np.zeros(theta.shape)
    for n in range(6):  # First Sum up to fifth order
        term_first_sum = np.power(theta, n + 1) / (n + 1)  # term in first sum

        second_sum = np.zeros(theta.shape)  # second Sum initialized with zeros
        for m in range((n + 1) % 2, n + 2, 2):
            # print(n, m)

            if m > 0:
                if f'C{n}{m}a' not in aberrations:  # Set non existent aberrations coefficient to zero
                    aberrations[f'C{n}{m}a'] = 0.
                if f'C{n}{m}b' not in aberrations:
                    aberrations[f'C{n}{m}b'] = 0.

                # term in second sum
                second_sum = second_sum + aberrations[f'C{n}{m}a'] * np.cos(m * phi) + aberrations[
                    f'C{n}{m}b'] * np.sin(m * phi)
            else:
                if f'C{n}{m}' not in aberrations:  # Set non existent aberrations coefficient to zero
                    aberrations[f'C{n}{m}'] = 0.

                # term in second sum
                second_sum = second_sum + aberrations[f'C{n}{m}']
        chi = chi + term_first_sum * second_sum * 2 * np.pi / aberrations['wavelength']

    return chi


def objective_lens_function(ab, nx, ny, field_of_view, aperture_size=10):
    """Objective len function to be convoluted with exit wave to derive image function

    Parameter:
    ----------
    ab: dict
        aberrations in nm should at least contain defocus (C10), and spherical aberration (C30)
    nx: int
        number of pixel in x direction
    ny: int
        number of pixel in y direction
    field_of_view: float
        field of view of potential
    wavelength: float
        wavelength in nm
    aperture_size: float
        aperture size in 1/nm

    Returns:
    --------
    object function: numpy array (nx x ny)
    extent: list
    """

    wavelength = ab['wavelength']
    # Reciprocal plane in 1/nm
    dk = 1 / field_of_view
    t_xv, t_yv = np.mgrid[int(-nx / 2):int(nx / 2), int(-ny / 2):int(ny / 2)] * dk

    # define reciprocal plane in angles
    phi = np.arctan2(t_yv, t_xv)
    theta = np.arctan2(np.sqrt(t_xv ** 2 + t_yv ** 2), 1 / wavelength)

    mask = theta < aperture_size * wavelength

    # calculate chi
    chi = make_chi(theta, phi, ab)

    extent = [-nx / 2 * dk, nx / 2 * dk, -nx / 2 * dk, nx / 2 * dk]
    return np.exp(-1j * chi) * mask, extent
