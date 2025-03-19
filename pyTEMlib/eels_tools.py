"""
eels_tools
Model based quantification of electron energy-loss data
Copyright by Gerd Duscher

The University of Tennessee, Knoxville
Department of Materials Science & Engineering

Sources:
   M. Tian et al.

Units:
    everything is in SI units, except length is given in nm and angles in mrad.

Usage:
    See the notebooks for examples of these routines

All the input and output is done through a dictionary which is to be found in the meta_data
attribute of the sidpy.Dataset

Update by Austin Houston, UTK 12-2023 : Parallization of spectrum images
"""
import typing
from typing import Union
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy import constants
from scipy import interpolate
from scipy.interpolate import interp1d, splrep
from scipy.signal import peak_prominences
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, leastsq

from numba import jit, float64

import requests

# ## And we use the image tool library of pyTEMlib
from pyTEMlib.xrpa_x_sections import x_sections

import sidpy
from sidpy.proc.fitter import SidFitter

# we have a function called find peaks - is it necessary?
# or could we just use scipy.signal import find_peaks

major_edges = ['K1', 'L3', 'M5', 'N5']
all_edges = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2',
             'O3', 'O4', 'O5', 'O6', 'O7', 'P1', 'P2', 'P3']
first_close_edges = ['K1', 'L3', 'M5', 'M3', 'N5', 'N3']

elements = [' ', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
            'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']


# kroeger_core(e_data,a_data,eps_data,ee,thick, relativistic =True)
# kroeger_core2(e_data,a_data,eps_data,acceleration_voltage_kev,thickness, relativistic =True)
# get_wave_length(e0)

# plot_dispersion(plotdata, units, a_data, e_data, title, max_p, ee, ef = 4., ep= 16.8, Es = 0, IBT = [])
# drude(tags, e, ep, ew, tnm, eb)
# drude(ep, eb, gamma, e)
# drude_lorentz(epsInf,leng, ep, eb, gamma, e, Amplitude)
# zl_func( p,  x)
# ###############################################################
# Utility Functions
# ################################################################

def get_wave_length(e0):
    """get deBroglie wavelength of electron accelerated by energy (in eV) e0"""

    ev = constants.e * e0
    return constants.h / np.sqrt(2 * constants.m_e * ev * (1 + ev / (2 * constants.m_e * constants.c ** 2)))


def effective_collection_angle(energy_scale, alpha, beta, beam_kv):
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

    # function y = effbeta(ene,  alpha, beta, beam_kv)
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
    if beam_kv == 0:
        beam_kv = 100.0

    if alpha == 0:
        return beta

    if beta == 0:
        return alpha

    z1 = beam_kv  # eV
    z2 = energy_scale[0]
    z3 = energy_scale[-1]
    z4 = 100.0

    z5 = alpha * 0.001  # rad
    z6 = beta * 0.001  # rad
    z7 = 500.0  # number of integration steps to be modified at will

    #       main loop on energy loss
    #
    for zx in range(int(z2), int(z3), int(z4)):  # ! zx = current energy loss
        eta = 0.0
        x0 = float(zx) * (z1 + 511060.) / (z1 * (z1 + 1022120.))  # x0 = relativistic theta-e
        x1 = np.pi / (2. * x0)
        x2 = x0 * x0 + z5 * z5
        x3 = z5 / x0 * z5 / x0
        x4 = 0.1 * np.sqrt(x2)
        dtheta = (z6 - x4) / z7
    #
    #        calculation of the analytical expression
    #
    for zi in range(1, int(z7)):
        theta = x4 + dtheta * float(zi)
        x5 = theta * theta
        x6 = 4. * x5 * x0 * x0
        x7 = x2 - x5
        x8 = np.sqrt(x7 * x7 + x6)
        x9 = (x8 + x7) / (2. * x0 * x0)
        x10 = 2. * theta * dtheta * np.log(x9)
        eta = eta + x10

    eta = eta + x2 / 100. * np.log(1. + x3)  # addition of the central contribution
    x4 = z5 * z5 * np.log(1. + x1 * x1)  # normalisation
    eta = eta / x4
    #
    #        correction by geometrical factor (beta/alpha)**2
    #
    if z6 < z5:
        x5 = z5 / z6
        eta = eta * x5 * x5

    etha2 = eta * 100.
    #
    #        calculation of beta *
    #
    x6 = np.power((1. + x1 * x1), eta)
    x7 = x0 * np.sqrt(x6 - 1.)
    beta = x7 * 1000.  # in mrad

    return beta


def set_default_metadata(current_dataset: sidpy.Dataset) -> None:

    if 'experiment' not in current_dataset.metadata:
        current_dataset.metadata['experiment'] = {}
    if 'convergence_angle' not in current_dataset.metadata['experiment']:
        current_dataset.metadata['experiment']['convergence_angle'] = 30
    if 'collection_angle' not in current_dataset.metadata['experiment']:
        current_dataset.metadata['experiment']['collection_angle'] = 50
    if 'acceleration_voltage' not in current_dataset.metadata['experiment']:
        current_dataset.metadata['experiment']['acceleration_voltage'] = 200000

###

# ###############################################################
# Peak Fit Functions
# ################################################################


def residuals_smooth(p, x, y, only_positive_intensity):
    """part of fit"""

    err = (y - model_smooth(x, p, only_positive_intensity))
    return err


def model_smooth(x, p, only_positive_intensity=False):
    """part of fit"""

    y = np.zeros(len(x))

    number_of_peaks = int(len(p) / 3)
    for i in range(number_of_peaks):
        if only_positive_intensity:
            p[i * 3 + 1] = abs(p[i * 3 + 1])
        p[i * 3 + 2] = abs(p[i * 3 + 2])
        if p[i * 3 + 2] > abs(p[i * 3]) * 4.29193 / 2.0:
            p[i * 3 + 2] = abs(p[i * 3]) * 4.29193 / 2.  # ## width cannot extend beyond zero, maximum is FWTM/2

        y = y + gauss(x, p[i * 3:])

    return y

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
    else:
        return p[1] * np.exp(-(x - p[0]) ** 2 / (2.0 * (p[2] / 2.3548) ** 2))


def lorentz(x, center, amplitude, width):
    """ Lorentzian Function """
    lorentz_peak = 0.5 * width / np.pi / ((x - center) ** 2 + (width / 2) ** 2)
    return amplitude * lorentz_peak / lorentz_peak.max()

def zero_loss_function(x, p):
    return zl_func(x, *p)

def zl_func(x, center1, amplitude1, width1, center2, amplitude2, width2):
    """ zero loss function as product of two lorentzians """
    return lorentz(x, center1, amplitude1, width1) * lorentz(x, center2, amplitude2, width2)


def zl(x, p, p_zl):
    """zero-loss function"""
    p_zl_local = p_zl.copy()
    p_zl_local[2] += p[0]
    p_zl_local[5] += p[0]
    zero_loss = zl_func(x, p_zl_local)
    return p[1] * zero_loss / zero_loss.max()


def get_channel_zero(spectrum: np.ndarray, energy: np.ndarray, width: int = 8):
    """Determin shift of energy scale according to zero-loss peak position
    
    This function assumes that the zero loss peak is the maximum of the spectrum. 
    """

    zero = scipy.signal.find_peaks(spectrum/np.max(spectrum), height=0.98)[0][0]
    width = int(width/2)
    x = np.array(energy[int(zero-width):int(zero+width)])
    y = np.array(spectrum[int(zero-width):int(zero+width)]).copy()

    y[np.nonzero(y <= 0)] = 1e-12

    p0 = [energy[zero], spectrum.max(), .5]  # Initial guess is a normal distribution

    def errfunc(pp, xx, yy):
        return (gauss(xx, pp) - yy) / np.sqrt(yy)  # Distance to the target function
    
    [p1, _] = leastsq(errfunc, np.array(p0[:]), args=(x, y))
    fit_mu, area, fwhm = p1

    return fwhm, fit_mu


def get_zero_loss_energy(dataset):

    spectrum = dataset.sum(axis=tuple(range(dataset.ndim - 1)))

    startx = scipy.signal.find_peaks(spectrum/np.max(spectrum), height=0.98)[0][0]

    end = startx + 3
    start = startx - 3
    for i in range(10):
        if spectrum[startx - i] < 0.3 * spectrum[startx]:
            start = startx - i
        if spectrum[startx + i] < 0.3 * spectrum[startx]:
            end = startx + i
    if end - start < 7:
        end = startx + 4
        start = startx - 4
    width = int((end-start)/2+0.5)

    energy = dataset.get_spectral_dims(return_axis=True)[0].values

    if dataset.ndim == 1:  # single spectrum
        _, shifts = get_channel_zero(np.array(dataset), energy, width)
        shifts = np.array([shifts])
    elif dataset.ndim == 2:  # line scan
        shifts = np.zeros(dataset.shape[:1])
        for x in range(dataset.shape[0]):
            _, shifts[x] = get_channel_zero(dataset[x, :], energy, width)
    elif dataset.ndim == 3:  # spectral image
        shifts = np.zeros(dataset.shape[:2])
        for x in range(dataset.shape[0]):
            for y in range(dataset.shape[1]):
                _, shifts[x, y] = get_channel_zero(dataset[x, y, :], energy, width)
    return shifts


def shift_energy(dataset: sidpy.Dataset, shifts: np.ndarray) -> sidpy.Dataset:
    """ Align zero-loss peaks of any spectral sidpy dataset """

    new_si = dataset.copy()
    new_si *= 0.0

    image_dims = dataset.get_image_dims()
    if len(image_dims) == 0:
        image_dims =[0]
    if len(image_dims) != shifts.ndim:
        raise TypeError('array of energy shifts have to have same dimension as dataset')
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('This function needs a sidpy Dataset to shift energy scale')
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values
    if dataset.ndim == 1:  # single spectrum
        tck = interpolate.splrep(np.array(energy_scale - shifts), np.array(dataset), k=1, s=0)
        new_si[:] = interpolate.splev(energy_scale, tck, der=0)
        new_si.data_type = 'Spectrum'
    elif dataset.ndim == 2:  # line scan
        for x in range(dataset.shape[0]):
            tck = interpolate.splrep(np.array(energy_scale - shifts[x]), np.array(dataset[x, :]), k=1, s=0)
            new_si[x, :] = interpolate.splev(energy_scale, tck, der=0)
    elif dataset.ndim == 3:  # spectral image
        for x in range(dataset.shape[0]):
            for y in range(dataset.shape[1]):
                tck = interpolate.splrep(np.array(energy_scale - shifts[x, y]), np.array(dataset[x, y]), k=1, s=0)
                new_si[x, y, :] = interpolate.splev(energy_scale, tck, der=0)

    return new_si


def align_zero_loss(dataset: sidpy.Dataset) -> sidpy.Dataset:
    """
    Shifts the energy axis of the input dataset to be aligned with the zero-loss peak.

    Parameters:
    -----------
    dataset : sidpy.Dataset
        The input dataset containing the energy axis to be aligned.

    Returns:
    --------
    sidpy.Dataset
        The dataset with the energy axis shifted to align the zero-loss peak.

    """
    shifts = get_zero_loss_energy(dataset)
    # print(shifts, dataset)
    new_si = shift_energy(dataset, shifts)    
    new_si.metadata.update({'zero_loss': {'shifted': shifts}})
    return new_si

from numba import jit

def get_zero_losses(energy, z_loss_params):
    z_loss_dset = np.zeros((z_loss_params.shape[0], z_loss_params.shape[1], energy.shape[0]))
    for x in range(z_loss_params.shape[0]):
        for y in range(z_loss_params.shape[1]):
            z_loss_dset[x, y] +=  zl_func(energy, *z_loss_params[x, y])
    return z_loss_dset




def get_resolution_functions(dataset: sidpy.Dataset, startFitEnergy: float=-1, endFitEnergy: float=+1,
                             n_workers: int=1, n_threads: int=8):
    """
    Analyze and fit low-loss EELS data within a specified energy range to determine zero-loss peaks.

    This function processes a low-loss EELS dataset from transmission electron microscopy (TEM) data, 
    focusing on a specified energy range for analyzing and fitting the spectrum. 
    It determines fitting parameters and applies these to extract zero-loss peak information 
    from the dataset. The function handles both 2D and 3D datasets.

    Parameters:
    -----------
        dataset (sidpy.Dataset): The dataset containing TEM spectral data.
        startFitEnergy (float): The start energy of the fitting window.
        endFitEnergy (float): The end energy of the fitting window.
        n_workers (int, optional): The number of workers for parallel processing (default is 1).
        n_threads (int, optional): The number of threads for parallel processing (default is 8).

    Returns:
    --------
        tuple: A tuple containing:
            - z_loss_dset (sidpy.Dataset): The dataset with added zero-loss peak information.
            - z_loss_params (numpy.ndarray): Array of parameters used for the zero-loss peak fitting.

    Raises:
    -------
        ValueError: If the input dataset does not have the expected dimensions or format.

    Notes:
    ------
        - The function expects `dset` to have specific dimensionalities and will raise an error if they are not met.
        - Parallel processing is employed to enhance performance, particularly for large datasets.
    """
    energy = dataset.get_spectral_dims(return_axis=True)[0].values
    start_fit_pixel = np.searchsorted(energy, startFitEnergy)
    end_fit_pixel = np.searchsorted(energy, endFitEnergy)
    guess_width = (endFitEnergy - startFitEnergy)/2
    if end_fit_pixel - start_fit_pixel < 5:
        start_fit_pixel -= 2
        end_fit_pixel += 2
    
    def get_good_guess(zl_func, energy, spectrum):
        popt, pcov = curve_fit(zl_func, energy, spectrum,
                               p0=[0, guess_amplitude, guess_width,
                                   0, guess_amplitude, guess_width])
        return popt

    fit_energy = energy[start_fit_pixel:end_fit_pixel]
    # get a good guess for the fit parameters
    if len(dataset.shape) == 3:
        fit_dset = dataset[:, :, start_fit_pixel:end_fit_pixel]
        guess_amplitude = np.sqrt(fit_dset.max())
        guess_params = get_good_guess(zl_func, fit_energy, fit_dset.sum(axis=(0, 1))/fit_dset.shape[0]/fit_dset.shape[1])
    elif len(dataset.shape) == 2:
        fit_dset = dataset[:, start_fit_pixel:end_fit_pixel]
        fit_energy = energy[start_fit_pixel:end_fit_pixel]
        guess_amplitude = np.sqrt(fit_dset.max())
        guess_params = get_good_guess(zl_func, fit_energy, fit_dset.sum(axis=0)/fit_dset.shape[0])
    elif len(dataset.shape) == 1:
        fit_dset = dataset[start_fit_pixel:end_fit_pixel]
        fit_energy = energy[start_fit_pixel:end_fit_pixel]
        guess_amplitude = np.sqrt(fit_dset.max())
        guess_params = get_good_guess(zl_func, fit_energy, fit_dset)
        z_loss_dset = dataset.copy()
        z_loss_dset *= 0.0
        z_loss_dset += zl_func(energy, *guess_params)
        if 'zero_loss' not in z_loss_dset.metadata:
            z_loss_dset.metadata['zero_loss'] = {}
        z_loss_dset.metadata['zero_loss'].update({'startFitEnergy': startFitEnergy,
                                                  'endFitEnergy': endFitEnergy,
                                                  'fit_parameter': guess_params,
                                                  'original_low_loss': dataset.title})
        return z_loss_dset
    else:
        print('Error: need a spectrum or spectral image sidpy dataset')
        print('Not dset.shape = ', dataset.shape)
        return None

    # define guess function for SidFitter
    def guess_function(xvec, yvec):
        return guess_params
    
    # apply to all spectra
    zero_loss_fitter = SidFitter(fit_dset, zl_func, num_workers=n_workers, guess_fn=guess_function, threads=n_threads,
                                 return_cov=False, return_fit=False, return_std=False, km_guess=False, num_fit_parms=6)
    
    [z_loss_params] = zero_loss_fitter.do_fit()
    z_loss_dset = dataset.copy()
    z_loss_dset *= 0.0

    #energy_grid = np.broadcast_to(energy.reshape((1, 1, -1)), (z_loss_dset.shape[0],
    #                                                           z_loss_dset.shape[1], energy.shape[0]))
    #z_loss_peaks = zl_func(energy_grid, *z_loss_params)
    z_loss_params = np.array(z_loss_params)
    z_loss_dset += get_zero_losses(np.array(energy), np.array(z_loss_params))
    
    shifts = z_loss_params[:, :, 0] * z_loss_params[:, :, 3]
    widths = z_loss_params[:, :, 2] * z_loss_params[:, :, 5]

    z_loss_dset.metadata['zero_loss'].update({'startFitEnergy': startFitEnergy,
                                              'endFitEnergy': endFitEnergy,
                                              'fit_parameter': z_loss_params,
                                              'original_low_loss': dataset.title})


    return z_loss_dset


def drude(energy_scale, peak_position, peak_width, gamma):
    """dielectric function according to Drude theory"""

    eps = (1 - (peak_position ** 2 - peak_width * energy_scale * 1j) /
           (energy_scale ** 2 + 2 * energy_scale * gamma * 1j))  # Mod drude term
    return eps


def drude_lorentz(eps_inf, leng, ep, eb, gamma, e, amplitude):
    """dielectric function according to Drude-Lorentz theory"""

    eps = eps_inf
    for i in range(leng):
        eps = eps + amplitude[i] * (1 / (e + ep[i] + gamma[i] * 1j) - 1 / (e - ep[i] + gamma[i] * 1j))
    return eps


def get_plasmon_losses(energy, params):
    dset = np.zeros((params.shape[0], params.shape[1], energy.shape[0]))
    for x in range(params.shape[0]):
        for y in range(params.shape[1]):
            dset[x, y] +=  energy_loss_function(energy, params[x, y])
    return dset


def fit_plasmon(dataset: Union[sidpy.Dataset, np.ndarray], startFitEnergy: float, endFitEnergy: float,  number_workers: int = 4, number_threads: int = 8) -> Union[sidpy.Dataset, np.ndarray]:
    """
    Fit plasmon peak positions and widths in a TEM dataset using a Drude model.

    This function applies the Drude model to fit plasmon peaks in a dataset obtained 
    from transmission electron microscopy (TEM). It processes the dataset to determine 
    peak positions, widths, and amplitudes within a specified energy range. The function 
    can handle datasets with different dimensions and offers parallel processing capabilities.

    Parameters:
        dataset: sidpy.Dataset or numpy.ndarray
            The dataset containing TEM spectral data.
        startFitEnergy: float
            The start energy of the fitting window.
        endFitEnergy: float
            The end energy of the fitting window.
        plot_result: bool, optional
            If True, plots the fitting results (default is False).
        number_workers: int, optional
            The number of workers for parallel processing (default is 4).
        number_threads: int, optional
            The number of threads for parallel processing (default is 8).

    Returns:
        fitted_dataset: sidpy.Dataset or numpy.ndarray
            The dataset with fitted plasmon peak parameters. The dimensions and 
            format depend on the input dataset.

    Raises:
        ValueError: If the input dataset does not have the expected dimensions or format.

    Notes:
        - The function uses the Drude model to fit plasmon peaks.
        - The fitting parameters are peak position (Ep), peak width (Ew), and amplitude (A).
        - If `plot_result` is True, the function plots Ep, Ew, and A as separate subplots.
    """
    # define Drude function for plasmon fitting

    anglog, T, _ = angle_correction(dataset)
    def energy_loss_function(E: np.ndarray, Ep: float, Ew: float, A: float) -> np.ndarray:

        eps = 1 - Ep**2/(E**2+Ew**2) + 1j * Ew * Ep**2/E/(E**2+Ew**2)
        elf = (-1/eps).imag  
        return A*elf
    
    # define window for fitting
    energy = dataset.get_spectral_dims(return_axis=True)[0].values
    start_fit_pixel = np.searchsorted(energy, startFitEnergy)
    end_fit_pixel = np.searchsorted(energy, endFitEnergy)

    # rechunk dataset
    if dataset.ndim == 3:
        dataset = dataset.rechunk(chunks=(1, 1, -1))
        fit_dset = dataset[:, :, start_fit_pixel:end_fit_pixel]
    elif dataset.ndim == 2:
        dataset = dataset.rechunk(chunks=(1, -1))
        fit_dset = dataset[:, start_fit_pixel:end_fit_pixel]
    else:
        fit_dset = np.array(dataset[start_fit_pixel:end_fit_pixel]/ anglog[start_fit_pixel:end_fit_pixel])
        guess_pos = np.argmax(fit_dset)
        guess_amplitude = fit_dset[guess_pos]
        guess_width =(endFitEnergy-startFitEnergy)/4
        guess_pos = energy[start_fit_pixel+guess_pos]
        
        if guess_width >8:
            guess_width=8
        try:
            popt, pcov = curve_fit(energy_loss_function, energy[start_fit_pixel:end_fit_pixel], fit_dset,
                                p0=[guess_pos, guess_width, guess_amplitude])
        except:
            end_fit_pixel = np.searchsorted(energy, 30)
            fit_dset = np.array(dataset[start_fit_pixel:end_fit_pixel]/ anglog[start_fit_pixel:end_fit_pixel])
            try:
                popt, pcov = curve_fit(energy_loss_function, energy[start_fit_pixel:end_fit_pixel], fit_dset,
                                p0=[guess_pos, guess_width, guess_amplitude])
            except:
                popt=[0,0,0]
        
        plasmon = dataset.like_data(energy_loss_function(energy, popt[0], popt[1], popt[2]))
        plasmon *= anglog
        start_plasmon = np.searchsorted(energy, 0)+1
        plasmon[:start_plasmon] = 0.
        
        epsilon = drude(energy, popt[0], popt[1], 1) * popt[2]
        epsilon[:start_plasmon] = 0.
        
        plasmon.metadata['plasmon'] = {'parameter': popt, 'epsilon':epsilon}
        return plasmon
    
    # if it can be parallelized:
    fitter = SidFitter(fit_dset, energy_loss_function, num_workers=number_workers,
                       threads=number_threads, return_cov=False, return_fit=False, return_std=False,
                       km_guess=False, num_fit_parms=3)
    [fit_parameter] = fitter.do_fit()

    plasmon_dset = dataset * 0.0
    fit_parameter = np.array(fit_parameter)
    plasmon_dset += get_plasmon_losses(np.array(energy), fit_parameter)
    if 'plasmon' not in plasmon_dset.metadata:
        plasmon_dset.metadata['plasmon'] = {}
    plasmon_dset.metadata['plasmon'].update({'startFitEnergy': startFitEnergy,
                                              'endFitEnergy': endFitEnergy,
                                              'fit_parameter': fit_parameter,
                                              'original_low_loss': dataset.title})

    return plasmon_dset


def angle_correction(spectrum):

    acceleration_voltage = spectrum.metadata['experiment']['acceleration_voltage']
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0]
    # eff_beta = effective_collection_angle(energy_scale, spectrum.metadata['experiment']['convergence_angle'],
    #                                     spectrum.metadata['experiment']['collection_angle'],acceleration_voltage)
   
    
    epc = energy_scale.slope  # input('ev per channel : ');

    alpha = spectrum.metadata['experiment']['convergence_angle']  # input('Alpha (mrad) : ');
    beta = spectrum.metadata['experiment']['collection_angle']# input('Beta (mrad) : ');
    e = energy_scale.values
    e0 = acceleration_voltage/1000 # input('E0 (keV) : ');

    T = 1000.0*e0*(1.+e0/1022.12)/(1.0+e0/511.06)**2  # %eV # equ.5.2a or Appendix E p 427 
    
    tgt=e0*(1.+e0/1022.)/(1+e0/511.);
    thetae=(e+1e-6)/tgt; # % avoid NaN for e=0
    # %     A2,B2,T2 ARE SQUARES OF ANGLES IN RADIANS**2
    a2=alpha*alpha*1e-6 + 1e-7;  # % avoid inf for alpha=0
    b2=beta*beta*1e-6;
    t2=thetae*thetae*1e-6;
    eta1=np.sqrt((a2+b2+t2)**2-4*a2*b2)-a2-b2-t2;
    eta2=2.*b2*np.log(0.5/t2*(np.sqrt((a2+t2-b2)**2+4.*b2*t2)+a2+t2-b2));
    eta3=2.*a2*np.log(0.5/t2*(np.sqrt((b2+t2-a2)**2+4.*a2*t2)+b2+t2-a2));
    eta=(eta1+eta2+eta3)/a2/np.log(4./t2);
    f1=(eta1+eta2+eta3)/2./a2/np.log(1.+b2/t2);
    f2=f1;
    if(alpha/beta>1):
        f2=f1*a2/b2;

    bstar=thetae*np.sqrt(np.exp(f2*np.log(1.+b2/t2))-1.);
    anglog = f2
    """
    b = eff_beta/1000.0 # %rad
    e0 = acceleration_voltage/1000.0 # %keV
    T = 1000.0*e0*(1.+e0/1022.12)/(1.0+e0/511.06)**2  # %eV # equ.5.2a or Appendix E p 427 
    tgt = 1000*e0*(1022.12 + e0)/(511.06 + e0)  # %eV  Appendix E p 427 

    the = energy_scale/tgt # varies with energy loss! # Appendix E p 427 
    anglog = np.log(1.0+ b*b/the/the)
    # 2 * T = m_0 v**2 !!!  a_0 = 0.05292 nm  epc is for sum over I0
    """
    return anglog,   (np.pi*0.05292* T / 2.0)/epc, bstar[0],

def energy_loss_function(energy_scale: np.ndarray, p: np.ndarray, anglog=1) -> np.ndarray:
    eps = 1 - p[0]**2/(energy_scale**2+p[1]**2) + 1j * p[1] * p[0]**2/energy_scale/(energy_scale**2+p[1]**2)
    elf = (-1/eps).imag
    return elf*p[2]*anglog

def inelatic_mean_free_path(E_p, spectrum):
    acceleration_voltage = spectrum.metadata['experiment']['acceleration_voltage']
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values
    
    e0 = acceleration_voltage/1000.0 # %keV

    eff_beta = effective_collection_angle(energy_scale, spectrum.metadata['experiment']['convergence_angle'],
                                         spectrum.metadata['experiment']['collection_angle'],acceleration_voltage)
    beta = eff_beta/1000.0 # %rad
    
    T = 1000.0*e0*(1.+e0/1022.12)/(1.0+e0/511.06)**2  # %eV # equ.5.2a or Appendix E p 427 
    tgt = 1000*e0*(1022.12 + e0)/(511.06 + e0)  # %eV  Appendix E p 427 
    theta_e = E_p/tgt # varies with energy loss! # Appendix E p 427
    
    # 2 * T = m_0 v**2 !!!  
    a_0 = 0.05292 # nm 
    imfp = 4*T*a_0/E_p/np.log(1+beta**2/theta_e**2)

    return imfp, theta_e


def multiple_scattering(energy_scale: np.ndarray, p: list, core_loss=False)-> np.ndarray:
    p = np.abs(p)
    tmfp = p[3]
    if core_loss:
        dif = 1
    else:
        dif = 16
    LLene = np.linspace(1, 2048-1,2048)/dif
    
    SSD = energy_loss_function(LLene, p)
    ssd  = np.fft.fft(SSD)
    ssd2 = ssd.copy()
    
    ### sum contribution from each order of scattering:
    PSD = np.zeros(len(LLene))
    for order in range(15):
        # This order convoluted spectum 
        # convoluted SSD is SSD2
        SSD2 = np.fft.ifft(ssd).real
    
        # scale right (could be done better? GERD) 
        # And add this order to final spectrum
        PSD += SSD2*abs(sum(SSD)/sum(SSD2)) / scipy.special.factorial(order+1)*np.power(tmfp, (order+1))*np.exp(-tmfp) #using equation 4.1 of egerton ed2
        
        # next order convolution
        ssd = ssd * ssd2
    
    PSD /=tmfp*np.exp(-tmfp)
    BGDcoef = scipy.interpolate.splrep(LLene, PSD, s=0)   
    msd = scipy.interpolate.splev(energy_scale, BGDcoef)
    start_plasmon = np.searchsorted(energy_scale, 0)+1
    msd[:start_plasmon] = 0. 
    return msd

def fit_multiple_scattering(dataset: Union[sidpy.Dataset, np.ndarray], startFitEnergy: float, endFitEnergy: float,pin=None, number_workers: int = 4, number_threads: int = 8) -> Union[sidpy.Dataset, np.ndarray]:
    """
    Fit multiple scattering of plasmon peak in a TEM dataset.

    
    Parameters:
        dataset: sidpy.Dataset or numpy.ndarray
            The dataset containing TEM spectral data.
        startFitEnergy: float
            The start energy of the fitting window.
        endFitEnergy: float
            The end energy of the fitting window.
        number_workers: int, optional
            The number of workers for parallel processing (default is 4).
        number_threads: int, optional
            The number of threads for parallel processing (default is 8).

    Returns:
        fitted_dataset: sidpy.Dataset or numpy.ndarray
            The dataset with fitted plasmon peak parameters. The dimensions and 
            format depend on the input dataset.

    Raises:
        ValueError: If the input dataset does not have the expected dimensions or format.

    Notes:
        - The function uses the Drude model to fit plasmon peaks.
        - The fitting parameters are peak position (Ep), peak width (Ew), and amplitude (A).
        - If `plot_result` is True, the function plots Ep, Ew, and A as separate subplots.
    """
    

    # define window for fitting
    energy = dataset.get_spectral_dims(return_axis=True)[0].values
    start_fit_pixel = np.searchsorted(energy, startFitEnergy)
    end_fit_pixel = np.searchsorted(energy, endFitEnergy)

    def errf_multi(p, y, x):
        elf = multiple_scattering(x, p)
        err = y - elf
        #print (p,sum(np.abs(err)))
        return np.abs(err) # /np.sqrt(y)

    if pin is None:
        pin = np.array([9,1,.7, 0.3])

    
    fit_dset = np.array(dataset[start_fit_pixel:end_fit_pixel])
    popt, lsq = leastsq(errf_multi, pin, args=(fit_dset, energy[start_fit_pixel:end_fit_pixel]), maxfev=2000)
    
    multi = dataset.like_data(multiple_scattering(energy, popt))
    

    multi.metadata['multiple_scattering'] = {'parameter': popt}
    return multi

    

def drude_simulation(dset, e, ep, ew, tnm, eb):
    """probabilities of dielectric function eps relative to zero-loss integral (i0 = 1)

    Gives probabilities of dielectric function eps relative to zero-loss integral (i0 = 1) per eV
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011

    # Given the plasmon energy (ep), plasmon fwhm (ew) and binding energy(eb),
    # this program generates:
    # EPS1, EPS2 from modified Eq. (3.40), ELF=Im(-1/EPS) from Eq. (3.42),
    # single scattering from Eq. (4.26) and SRFINT from Eq. (4.31)
    # The output is e, ssd into the file drude.ssd (for use in Flog etc.)
    # and e,eps1 ,eps2 into drude.eps (for use in Kroeger etc.)
    # Gives probabilities relative to zero-loss integral (i0 = 1) per eV
    # Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    # Version 10.11.26

    """
    energy_scale = dset.get_spectral_dims(return_axis=True)[0].values

    epc = energy_scale[1] - energy_scale[0]  # input('ev per channel : ');
    
    b = dset.metadata['collection_angle'] / 1000.  # rad
    epc = dset.energy_scale[1] - dset.energy_scale[0]  # input('ev per channel : ');
    e0 = dset.metadata['acceleration_voltage'] / 1000.  # input('incident energy e0(kev) : ');

    # effective kinetic energy: T = m_o v^2/2,
    t = 1000.0 * e0 * (1. + e0 / 1022.12) / (1.0 + e0 / 511.06) ** 2  # eV # equ.5.2a or Appendix E p 427
    
    # 2 gamma T
    tgt = 1000 * e0 * (1022.12 + e0) / (511.06 + e0)  # eV  Appendix E p 427
    
    rk0 = 2590 * (1.0 + e0 / 511.06) * np.sqrt(2.0 * t / 511060)
    
    os = e[0]
    ew_mod = eb
    tags = dset.metadata
   
    eps = 1 - (ep ** 2 - ew_mod * e * 1j) / (e ** 2 + 2 * e * ew * 1j)  # Mod drude term
    
    eps[np.nonzero(eps == 0.0)] = 1e-19
    elf = np.imag(-1 / eps)

    the = e / tgt  # varies with energy loss! # Appendix E p 427
    # srfelf = 4..*eps2./((1+eps1).^2+eps2.^2) - elf; %equivalent
    srfelf = np.imag(-4. / (1.0 + eps)) - elf  # for 2 surfaces
    angdep = np.arctan(b / the) / the - b / (b * b + the * the)
    srfint = angdep * srfelf / (3.1416 * 0.05292 * rk0 * t)  # probability per eV
    anglog = np.log(1.0 + b * b / the / the)
    i0 = dset.sum()  # *tags['counts2e']

    # 2 * t = m_0 v**2 !!!  a_0 = 0.05292 nm
    volint = abs(tnm / (np.pi * 0.05292 * t * 2.0) * elf * anglog)  # S equ 4.26% probability per eV
    volint = volint * i0 / epc  # S probability per channel
    ssd = volint  # + srfint;

    if e[0] < -1.0:
        xs = int(abs(-e[0] / epc))

        ssd[0:xs] = 0.0
        volint[0:xs] = 0.0
        srfint[0:xs] = 0.0

        # if os <0:
        p_s = np.trapz(e, srfint)  # 2 surfaces but includes negative Begrenzung contribution.
        p_v = abs(np.trapz(e, abs(volint / tags['spec'].sum())))  # integrated volume probability
        p_v = (volint / i0).sum()  # our data have he same epc and the trapez formula does not include
        lam = tnm / p_v  # does NOT depend on free-electron approximation (no damping).
        lamfe = 4.0 * 0.05292 * t / ep / np.log(1 + (b * tgt / ep) ** 2)  # Eq.(3.44) approximation

        tags['eps'] = eps
        tags['lam'] = lam
        tags['lamfe'] = lamfe
        tags['p_v'] = p_v

    return ssd  # /np.pi


def kroeger_core(e_data, a_data, eps_data, acceleration_voltage_kev, thickness, relativistic=True):
    """This function calculates the differential scattering probability

     .. math::
        \\frac{d^2P}{d \\Omega d_e}
    of the low-loss region for total loss and volume plasmon loss

    Args:
       e_data (array): energy scale [eV]
       a_data (array): angle or momentum range [rad]
       eps_data (array) dielectric function
       acceleration_voltage_kev (float): acceleration voltage [keV]
       thickness (float): thickness in nm
       relativistic (boolean): relativistic correction

    Returns:
       P (numpy array 2d): total loss probability
       p_vol (numpy array 2d): volume loss probability

       return P, P*scale*1e2,p_vol*1e2, p_simple*1e2
    """

    # $d^2P/(dEd\Omega) = \frac{1}{\pi^2 a_0 m_0 v^2} \Im \left[ \frac{t\mu^2}{\varepsilon \phi^2 } \right]
    """
    # Internally everything is calculated in si units
    # acceleration_voltage_kev = 200 #keV
    # thick = 32.0*10-9 # m

    """
    a_data = np.array(a_data)
    e_data = np.array(e_data)
    # adjust input to si units
    wavelength = get_wave_length(acceleration_voltage_kev * 1e3)  # in m
    thickness = thickness * 1e-9  # input thickness now in m

    # Define constants
    # ec = 14.4;
    m_0 = constants.value(u'electron mass')  # REST electron mass in kg
    # h = constants.Planck  # Planck's constant
    hbar = constants.hbar

    c = constants.speed_of_light  # speed of light m/s
    bohr = constants.value(u'Bohr radius')  # Bohr radius in meters
    e = constants.value(u'elementary charge')  # electron charge in Coulomb
    # print('hbar =', hbar ,' [Js] =', hbar/e ,'[ eV s]')

    # Calculate fixed terms of equation
    va = 1 - (511. / (511. + acceleration_voltage_kev)) ** 2  # acceleration_voltage_kev is incident energy in keV
    v = c * np.sqrt(va)

    if relativistic:
        beta = v / c  # non-relativistic for =1
        gamma = 1. / np.sqrt(1 - beta ** 2)
    else:
        beta = 1
        gamma = 1  # set = 1 to correspond to E+B & Siegle

    momentum = m_0 * v * gamma  # used for xya, E&B have no gamma

    # ##### Define mapped variables

    # Define independent variables E, theta
    [energy, theta] = np.meshgrid(e_data + 1e-12, a_data)
    # Define CONJUGATE dielectric function variable eps
    [eps, _] = np.meshgrid(np.conj(eps_data), a_data)

    # ##### Calculate lambda in equation EB 2.3
    theta2 = theta ** 2 + 1e-15

    theta_e = energy * e / momentum / v  # critical angle

    lambda2 = theta2 - eps * theta_e ** 2 * beta ** 2  # Eq 2.3

    lambd = np.sqrt(lambda2)
    if (np.real(lambd) < 0).any():
        print(' error negative lambda')

    # ##### Calculate lambda0 in equation EB 2.4
    # According to Kröger real(lambda0) is defined as positive!

    phi2 = lambda2 + theta_e ** 2  # Eq. 2.2
    lambda02 = theta2 - theta_e ** 2 * beta ** 2  # eta=1 Eq 2.4
    lambda02[lambda02 < 0] = 0
    lambda0 = np.sqrt(lambda02)
    if not (np.real(lambda0) >= 0).any():
        print(' error negative lambda0')

    de = thickness * energy * e / (2.0 * hbar * v)  # Eq 2.5
    xya = lambd * de / theta_e  # used in Eqs 2.6, 2.7, 4.4

    lplus = lambda0 * eps + lambd * np.tanh(xya)  # eta=1 %Eq 2.6
    lminus = lambda0 * eps + lambd / np.tanh(xya)  # eta=1 %Eq 2.7

    mue2 = 1 - (eps * beta ** 2)  # Eq. 4.5
    phi20 = lambda02 + theta_e ** 2  # Eq 4.6
    phi201 = theta2 + theta_e ** 2 * (1 - (eps + 1) * beta ** 2)  # eta=1, eps-1 in E+b Eq.(4.7)

    # Eq 4.2
    a1 = phi201 ** 2 / eps
    a2 = np.sin(de) ** 2 / lplus + np.cos(de) ** 2 / lminus
    a = a1 * a2

    # Eq 4.3
    b1 = beta ** 2 * lambda0 * theta_e * phi201
    b2 = (1. / lplus - 1. / lminus) * np.sin(2. * de)
    b = b1 * b2

    # Eq 4.4
    c1 = -beta ** 4 * lambda0 * lambd * theta_e ** 2
    c2 = np.cos(de) ** 2 * np.tanh(xya) / lplus
    c3 = np.sin(de) ** 2 / np.tanh(xya) / lminus
    c = c1 * (c2 + c3)

    # Put all the pieces together...
    p_coef = e / (bohr * np.pi ** 2 * m_0 * v ** 2)

    p_v = thickness * mue2 / eps / phi2

    p_s1 = 2. * theta2 * (eps - 1) ** 2 / phi20 ** 2 / phi2 ** 2  # ASSUMES eta=1
    p_s2 = hbar / momentum
    p_s3 = a + b + c

    p_s = p_s1 * p_s2 * p_s3

    # print(p_v.min(),p_v.max(),p_s.min(),p_s.max())
    # Calculate P and p_vol (volume only)
    dtheta = a_data[1] - a_data[0]
    scale = np.sin(np.abs(theta)) * dtheta * 2 * np.pi

    p = p_coef * np.imag(p_v - p_s)  # Eq 4.1
    p_vol = p_coef * np.imag(p_v) * scale

    # lplus_min = e_data[np.argmin(np.real(lplus), axis=1)]
    # lminus_min = e_data[np.argmin(np.imag(lminus), axis=1)]

    p_simple = p_coef * np.imag(1 / eps) * thickness / (theta2 + theta_e ** 2) * scale
    # Watch it: eps is conjugated dielectric function

    return p, p * scale * 1e2, p_vol * 1e2, p_simple * 1e2  # ,lplus_min,lminus_min


#################################################################
# CORE - LOSS functions
#################################################################

def get_z(z: Union[int, str]) -> int:
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
    x_sections = get_x_sections()

    z_out = 0
    if str(z).isdigit():
        z_out = int(z)
    elif isinstance(z, str):
        for key in x_sections:
            if x_sections[key]['name'].lower() == z.lower():  # Well one really should know how to write elemental
                z_out = int(key)
    else:
        raise TypeError('A string or number is required')
    return z_out


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
    else:
        z = str(z)
        if z in x_sections:
            return x_sections[z]
        else:
            return 0


def list_all_edges(z: Union[str, int]=0, verbose=False)->list[str, dict]:
    """List all ionization edges of an element with atomic number z

    Parameters
    ----------
    z: int
        atomic number
    verbose: bool, optional
        more info if set to True

    Returns
    -------
    out_string: str
        string with all major edges in energy range
    """

    element = str(get_z(z))
    x_sections = get_x_sections()
    out_string = ''
    if verbose:
        print('Major edges')
    edge_list = {x_sections[element]['name']: {}}
    
    for key in all_edges:
        if key in x_sections[element]:
            if 'onset' in x_sections[element][key]:
                if verbose:
                    print(f" {x_sections[element]['name']}-{key}: {x_sections[element][key]['onset']:8.1f} eV ")
                out_string = out_string + f" {x_sections[element]['name']}-{key}: " \
                                          f"{x_sections[element][key]['onset']:8.1f} eV /n"
                edge_list[x_sections[element]['name']][key] = x_sections[element][key]['onset']
    return out_string, edge_list


def find_all_edges(edge_onset: float, maximal_chemical_shift: float=5.0, major_edges_only: bool=False) -> str:
    """Find all (major and minor) edges within an energy range

    Parameters
    ----------
    edge_onset: float
        approximate energy of ionization edge
    maximal_chemical_shift: float, default = 5eV
        range of energy window around edge_onset to look for major edges
    major_edges_only: boolean, default = False
        only major edges are considered if True
    Returns
    -------
    text: str
        string with all edges in energy range

    """

    text = ''
    x_sections = get_x_sections()
    for element in x_sections:
        for key in x_sections[element]:
            if isinstance(x_sections[element][key], dict):
                if 'onset' in x_sections[element][key]:
                    if abs(x_sections[element][key]['onset'] - edge_onset) < maximal_chemical_shift:
                        # print(element, x_sections[element]['name'], key, x_sections[element][key]['onset'])
                        new_text = f"\n {x_sections[element]['name']:2s}-{key}: " \
                                        f"{x_sections[element][key]['onset']:8.1f} eV "
                        if major_edges_only:
                            if key in major_edges:
                                text += new_text
                        else:
                            text += new_text

    return text         

def find_associated_edges(dataset: sidpy.Dataset) -> None:
    onsets = []
    edges = []
    if 'core_loss' in dataset.metadata:
        if'edges' in dataset.metadata['core_loss']:
            for key, edge in dataset.metadata['core_loss']['edges'].items():
                if key.isdigit():
                    """for sym in edge['all_edges']:  # TODO: Could be replaced with exclude
                        onsets.append(edge['all_edges'][sym]['onset'] + edge['chemical_shift'])
                        edges.append([key, f"{edge['element']}-{sym}", onsets[-1]])
                    """
                    onsets.append(edge['onset'])
                    dataset.metadata['core_loss']['edges'][key]['associated_peaks'] = {}
            if 'peak_fit' in dataset.metadata:
                p = dataset.metadata['peak_fit']['peak_out_list']
                for key, peak in enumerate(p):
                    distances  = (onsets-peak[0])*-1
                    distances[distances < -0.3] = 1e6
                    if np.min(distances) < 50:
                        index = np.argmin(distances)
                        dataset.metadata['core_loss']['edges'][str(index)]['associated_peaks'][key] = peak

                        
                """for key, peak in dataset.metadata['peak_fit']['peaks'].items():
                    if key.isdigit():
                        distance = dataset.get_spectral_dims(return_axis=True)[0].values[-1]
                        index = -1
                        for ii, onset in enumerate(onsets):
                            if onset < peak['position'] < onset+post_edge:
                                if distance > np.abs(peak['position'] - onset):
                                    distance = np.abs(peak['position'] - onset)  # TODO: check whether absolute is good
                                    distance_onset = peak['position'] - onset
                                    index = ii
                        if index >= 0:
                            peak['associated_edge'] = edges[index][1]  # check if more info is necessary
                            peak['distance_to_onset'] = distance_onset
                """
    
def find_white_lines(dataset: sidpy.Dataset) -> dict:
    white_lines_out ={'sum': {}, 'ratio': {}}
    white_lines = []
    if 'peak_fit' in dataset.metadata:
        peaks = dataset.metadata['peak_fit']['peaks']
    else:
        return
    for index, edge in dataset.metadata['core_loss']['edges'].items():
        if index.isdigit():
            if 'associated_peaks' in edge:
                peaks = edge['associated_peaks']
                
                if edge['symmetry'][-2:] in ['L3', 'M5']:
                    if 'L3' in edge['all_edges']:
                        end_range1 = edge['all_edges']['L2']['onset'] + edge['chemical_shift']
                        end_range2 = edge['all_edges']['L2']['onset']*2 - edge['all_edges']['L3']['onset'] + edge['chemical_shift'] 
                        white_lines = ['L3', 'L2']
                    elif 'M5' in edge['all_edges']:
                        end_range1 = edge['all_edges']['M4']['onset'] + edge['chemical_shift']
                        end_range2 = edge['all_edges']['M4']['onset']*2 - edge['all_edges']['M5']['onset'] + edge['chemical_shift'] 
                        white_lines = ['M5', 'M4']
                    else:
                        return
                    white_line_areas = [0., 0.]
                    for key, peak in peaks.items():
                        if str(key).isdigit():
                            if peak[0] < end_range1:
                                white_line_areas[0] += np.sqrt(2 * np.pi) * peak[1] * np.abs(peak[2]/np.sqrt(2 * np.log(2)))
                            elif peak[0] < end_range2:
                                white_line_areas[1] += np.sqrt(2 * np.pi) * peak[1] * np.abs(peak[2]/np.sqrt(2 * np.log(2)))

                    edge['white_lines'] = {white_lines[0]: white_line_areas[0], white_lines[1]: white_line_areas[1]}
                    
                    reference_counts = edge['areal_density']*dataset.metadata['core_loss']['xsections'][int(index)].sum()
                    white_lines_out['sum'][f"{edge['element']}-{white_lines[0]}+{white_lines[1]}"] = (white_line_areas[0] + white_line_areas[1])/reference_counts
                    white_lines_out['ratio'][f"{edge['element']}-{white_lines[0]}/{white_lines[1]}"] = white_line_areas[0] / white_line_areas[1]
    return white_lines_out


    """white_line_ratios = {}
    white_line_sum = {}
    for sym, area in white_lines.items():
        if sym[-2:] in ['L2', 'M4', 'M2']:
            if area > 0 and f"{sym[:-1]}{int(sym[-1]) + 1}" in white_lines:
                if white_lines[f"{sym[:-1]}{int(sym[-1]) + 1}"] > 0:
                    white_line_ratios[f"{sym}/{sym[-2]}{int(sym[-1]) + 1}"] = area / white_lines[
                        f"{sym[:-1]}{int(sym[-1]) + 1}"]
                    white_line_sum[f"{sym}+{sym[-2]}{int(sym[-1]) + 1}"] = (
                                area + white_lines[f"{sym[:-1]}{int(sym[-1]) + 1}"])

                    areal_density = 1.
                    if 'edges' in dataset.metadata:
                        for key, edge in dataset.metadata['edges'].items():
                            if key.isdigit():
                                if edge['element'] == sym.split('-')[0]:
                                    areal_density = edge['areal_density']
                                    break
                    white_line_sum[f"{sym}+{sym[-2]}{int(sym[-1]) + 1}"] /= areal_density

        dataset.metadata['peak_fit']['white_lines'] = white_lines
        dataset.metadata['peak_fit']['white_line_ratios'] = white_line_ratios
        dataset.metadata['peak_fit']['white_line_sums'] = white_line_sum
    """    

def second_derivative(dataset: sidpy.Dataset, sensitivity: float=2.5) -> None:
    """Calculates second derivative of a sidpy.dataset"""

    dim = dataset.get_spectral_dims()
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
    width = 50 / dispersion
    if width < 50:
        width = 50
    start_end_noise = int(len(energy_scale) - width)
    for index in indices[::-1]:
        if index > start_end_noise:
            start_end_noise = index - 70

    noise_level_start = sensitivity * np.std(second_dif[3:50])
    noise_level_end = sensitivity * np.std(second_dif[start_end_noise: start_end_noise + 50])
    #slope = (noise_level_end - noise_level_start) / (len(energy_scale) - 400)
    #noise_level = noise_level_start #+ np.arange(len(energy_scale)) * slope
    return second_dif , noise_level



def find_edges(dataset: sidpy.Dataset, sensitivity: float=2.5) -> None:
    """find edges within a sidpy.Dataset"""

    dim = dataset.get_spectral_dims()
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values

    second_dif, noise_level = second_derivative(dataset, sensitivity=sensitivity)

    [indices, peaks] = scipy.signal.find_peaks(second_dif, noise_level)

    peaks['peak_positions'] = energy_scale[indices]
    peaks['peak_indices'] = indices
    edge_energies = [energy_scale[50]]
    edge_indices = []

    [indices, _] = scipy.signal.find_peaks(-second_dif, noise_level)
    minima = energy_scale[indices]

    for peak_number in range(len(peaks['peak_positions'])):
        position = peaks['peak_positions'][peak_number]
        if position - edge_energies[-1] > 20:
            impossible = minima[minima < position]
            impossible = impossible[impossible > position - 5]
            if len(impossible) == 0:
                possible = minima[minima > position]
                possible = possible[possible < position + 5]
                if len(possible) > 0:
                    edge_energies.append((position + possible[0])/2)
                    edge_indices.append(np.searchsorted(energy_scale, (position + possible[0])/2))

    selected_edges = []
    for peak in edge_indices:
        if 525 < energy_scale[peak] < 533:
            selected_edges.append('O-K1')
        else:
            selected_edge = ''
            edges = find_all_edges(energy_scale[peak], 20, major_edges_only=True)
            edges = edges.split('\n')
            minimum_dist = 100.
            for edge in edges[1:]:
                edge = edge[:-3].split(':')
                name = edge[0].strip()
                energy = float(edge[1].strip())
                if np.abs(energy - energy_scale[peak]) < minimum_dist:
                    minimum_dist = np.abs(energy - energy_scale[peak])
                    selected_edge = name

            if selected_edge != '':
                selected_edges.append(selected_edge)

    return selected_edges


def assign_likely_edges(edge_channels: Union[list, np.ndarray], energy_scale: np.ndarray):
    edges_in_list = []
    result = {}
    for channel in edge_channels: 
        if channel not in edge_channels[edges_in_list]:
            shift = 5
            element_list = find_all_edges(energy_scale[channel], maximal_chemical_shift=shift, major_edges_only=True)
            while len(element_list) < 1:
                shift += 1
                element_list = find_all_edges(energy_scale[channel], maximal_chemical_shift=shift, major_edges_only=True)

            if len(element_list) > 1:
                while len(element_list) > 0:
                    shift-=1
                    element_list = find_all_edges(energy_scale[channel], maximal_chemical_shift=shift, major_edges_only=True)
                element_list = find_all_edges(energy_scale[channel], maximal_chemical_shift=shift+1, major_edges_only=True)
            element = (element_list[:4]).strip()
            z = get_z(element)
            result[element] =[]
            _, edge_list = list_all_edges(z)

            for peak in edge_list:
                for edge in edge_list[peak]:
                    possible_minor_edge = np.argmin(np.abs(energy_scale[edge_channels]-edge_list[peak][edge]))
                    if np.abs(energy_scale[edge_channels[possible_minor_edge]]-edge_list[peak][edge]) < 3:
                        #print('nex', next_e)
                        edges_in_list.append(possible_minor_edge)
                        
                        result[element].append(edge)
                    
    return result


def auto_id_edges(dataset):
    edge_channels = identify_edges(dataset)
    dim = dataset.get_spectral_dims()
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values
    found_edges = assign_likely_edges(edge_channels, energy_scale)
    return found_edges


def identify_edges(dataset: sidpy.Dataset, noise_level: float=2.0):
    """
    Using first derivative to determine edge onsets
    Any peak in first derivative higher than noise_level times standard deviation will be considered
    
    Parameters
    ----------
    dataset: sidpy.Dataset
        the spectrum
    noise_level: float
        ths number times standard deviation in first derivative decides on whether an edge onset is significant
        
    Return
    ------
    edge_channel: numpy.ndarray
    
    """
    dim = dataset.get_spectral_dims()
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0]
    dispersion = energy_scale.slope
    
    spec = scipy.ndimage.gaussian_filter(dataset, 3/dispersion)  # smooth with 3eV wideGaussian

    first_derivative = spec - np.roll(spec, +2) 
    first_derivative[:3] = 0
    first_derivative[-3:] = 0

    # find if there is a strong edge at high energy_scale
    noise_level = noise_level*np.std(first_derivative[3:50])
    [edge_channels, _] = scipy.signal.find_peaks(first_derivative, noise_level)
    
    return edge_channels


def add_element_to_dataset(dataset: sidpy.Dataset, z: Union[int, str]):
    """
    """
    # We check whether this element is already in the
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0]

    zz = get_z(z)
    if 'edges' not in dataset.metadata:
         dataset.metadata['edges'] = {'model': {}, 'use_low_loss': False}
    index = 0
    for key, edge in dataset.metadata['edges'].items():
        if key.isdigit():
            index += 1
            if 'z' in edge:
                if zz == edge['z']:
                    index = int(key)
                    break

    major_edge = ''
    minor_edge = ''
    all_edges = {}
    x_section = get_x_sections(zz)
    edge_start = 10  # int(15./ft.get_slope(self.energy_scale)+0.5)
    for key in x_section:
        if len(key) == 2 and key[0] in ['K', 'L', 'M', 'N', 'O'] and key[1].isdigit():
            if energy_scale[edge_start] < x_section[key]['onset'] < energy_scale[-edge_start]:
                if key in ['K1', 'L3', 'M5', 'M3']:
                    major_edge = key
                
                all_edges[key] = {'onset': x_section[key]['onset']}

    if major_edge != '':
        key = major_edge
    elif minor_edge != '':
        key = minor_edge
    else:
        print(f'Could not find no edge of {zz} in spectrum')
        return False

    
    if str(index) not in dataset.metadata['edges']:
        dataset.metadata['edges'][str(index)] = {}

    start_exclude = x_section[key]['onset'] - x_section[key]['excl before']
    end_exclude = x_section[key]['onset'] + x_section[key]['excl after']

    dataset.metadata['edges'][str(index)] = {'z': zz, 'symmetry': key, 'element': elements[zz],
                              'onset': x_section[key]['onset'], 'end_exclude': end_exclude,
                              'start_exclude': start_exclude}
    dataset.metadata['edges'][str(index)]['all_edges'] = all_edges
    dataset.metadata['edges'][str(index)]['chemical_shift'] = 0.0
    dataset.metadata['edges'][str(index)]['areal_density'] = 0.0
    dataset.metadata['edges'][str(index)]['original_onset'] = dataset.metadata['edges'][str(index)]['onset']
    return True


def make_edges(edges_present: dict, energy_scale: np.ndarray, e_0:float, coll_angle:float, low_loss:np.ndarray=None)->dict:
    """Makes the edges dictionary for quantification

    Parameters
    ----------
    edges_present: list
        list of edges
    energy_scale: numpy array
        energy scale on which to make cross-section
    e_0: float
        acceleration voltage (in V)
    coll_angle: float
        collection angle in mrad
    low_loss: numpy array with same length as energy_scale
        low_less spectrum with which to convolve the cross-section (default=None)

    Returns
    -------
    edges: dict
        dictionary with all information on cross-section
    """
    x_sections = get_x_sections()
    edges = {}
    for i, edge in enumerate(edges_present):
        element, symmetry = edge.split('-')
        z = 0
        for key in x_sections:
            if element == x_sections[key]['name']:
                z = int(key)
        edges[i] = {}
        edges[i]['z'] = z
        edges[i]['symmetry'] = symmetry
        edges[i]['element'] = element

    for key in edges:
        xsec = x_sections[str(edges[key]['z'])]
        if 'chemical_shift' not in edges[key]:
            edges[key]['chemical_shift'] = 0
        if 'symmetry' not in edges[key]:
            edges[key]['symmetry'] = 'K1'
        if 'K' in edges[key]['symmetry']:
            edges[key]['symmetry'] = 'K1'
        elif 'L' in edges[key]['symmetry']:
            edges[key]['symmetry'] = 'L3'
        elif 'M' in edges[key]['symmetry']:
            edges[key]['symmetry'] = 'M5'
        else:
            edges[key]['symmetry'] = edges[key]['symmetry'][0:2]

        edges[key]['original_onset'] = xsec[edges[key]['symmetry']]['onset']
        edges[key]['onset'] = edges[key]['original_onset'] + edges[key]['chemical_shift']
        edges[key]['start_exclude'] = edges[key]['onset'] - xsec[edges[key]['symmetry']]['excl before']
        edges[key]['end_exclude'] = edges[key]['onset'] + xsec[edges[key]['symmetry']]['excl after']

    edges = make_cross_sections(edges, energy_scale, e_0, coll_angle, low_loss)

    return edges

def fit_dataset(dataset: sidpy.Dataset):
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0]
    if 'fit_area' not in dataset.metadata['edges']:
        dataset.metadata['edges']['fit_area'] = {}
    if 'fit_start' not in dataset.metadata['edges']['fit_area']:
        dataset.metadata['edges']['fit_area']['fit_start'] = energy_scale[50]
    if 'fit_end' not in dataset.metadata['edges']['fit_area']:
        dataset.metadata['edges']['fit_area']['fit_end'] = energy_scale[-2]
    dataset.metadata['edges']['use_low_loss'] = False
        
    if 'experiment' in dataset.metadata:
        exp = dataset.metadata['experiment']
        if 'convergence_angle' not in exp:
            raise ValueError('need a convergence_angle in experiment of metadata dictionary ')
        alpha = exp['convergence_angle']
        beta = exp['collection_angle']
        beam_kv = exp['acceleration_voltage']
        energy_scale = dataset.get_spectral_dims(return_axis=True)[0]
        eff_beta = effective_collection_angle(energy_scale, alpha, beta, beam_kv)
        edges = make_cross_sections(dataset.metadata['edges'], np.array(energy_scale), beam_kv, eff_beta)
        dataset.metadata['edges'] = fit_edges2(dataset, energy_scale, edges)
        areal_density = []
        elements = []
        for key in edges:
            if key.isdigit():  # only edges have numbers in that dictionary
                elements.append(edges[key]['element'])
                areal_density.append(edges[key]['areal_density'])
        areal_density = np.array(areal_density)
        out_string = '\nRelative composition: \n'
        for i, element in enumerate(elements):
            out_string += f'{element}: {areal_density[i] / areal_density.sum() * 100:.1f}%  '

        print(out_string)


def auto_chemical_composition(dataset:sidpy.Dataset)->None:

    found_edges = auto_id_edges(dataset)
    for key in found_edges:
        add_element_to_dataset(dataset, key)
    fit_dataset(dataset)


def make_cross_sections(edges:dict, energy_scale:np.ndarray, e_0:float, coll_angle:float, low_loss:np.ndarray=None)->dict:
    """Updates the edges dictionary with collection angle-integrated X-ray photo-absorption cross-sections

    """
    for key in edges:
        if str(key).isdigit():
            if edges[key]['z'] <1:
                break
            edges[key]['data'] = xsec_xrpa(energy_scale, e_0 / 1000., edges[key]['z'], coll_angle,
                                           edges[key]['chemical_shift']) / 1e10  # from barnes to 1/nm^2
            if low_loss is not None:
                low_loss = np.roll(np.array(low_loss), 1024 - np.argmax(np.array(low_loss)))
                edges[key]['data'] = scipy.signal.convolve(edges[key]['data'], low_loss/low_loss.sum(), mode='same')

            edges[key]['onset'] = edges[key]['original_onset'] + edges[key]['chemical_shift']
            edges[key]['X_section_type'] = 'XRPA'
            edges[key]['X_section_source'] = 'pyTEMlib'

    return edges


def power_law(energy: np.ndarray, a:float, r:float)->np.ndarray:
    """power law for power_law_background"""
    return a * np.power(energy, -r)


def power_law_background(spectrum:np.ndarray, energy_scale:np.ndarray, fit_area:list, verbose:bool=False):
    """fit of power law to spectrum """

    # Determine energy window  for background fit in pixels
    startx = np.searchsorted(energy_scale, fit_area[0])
    endx = np.searchsorted(energy_scale, fit_area[1])

    x = np.array(energy_scale)[startx:endx]
    y = np.array(spectrum)[startx:endx].flatten()

    # Initial values of parameters
    p0 = np.array([1.0E+20, 3])

    # background fitting
    def bgdfit(pp, yy, xx):
        err = yy - power_law(xx, pp[0], pp[1])
        return err

    [p, _] = leastsq(bgdfit, p0, args=(y, x), maxfev=2000)

    background_difference = y - power_law(x, p[0], p[1])
    background_noise_level = std_dev = np.std(background_difference)
    if verbose:
        print(f'Power-law background with amplitude A: {p[0]:.1f} and exponent -r: {p[1]:.2f}')
        print(background_difference.max() / background_noise_level)

        print(f'Noise level in spectrum {std_dev:.3f} counts')

    # Calculate background over the whole energy scale
    background = power_law(energy_scale, p[0], p[1])
    return background, p


def cl_model(xx, pp, number_of_edges, xsec):
    """ core loss model for fitting"""
    yy = pp[0] *  xx**pp[1] +  pp[2] + pp[3]* xx + pp[4] * xx * xx
    for i in range(number_of_edges):
        pp[i+5] = np.abs(pp[i+5])
        yy = yy + pp[i+5] * xsec[i, :]
    return yy


def fit_edges2(spectrum, energy_scale, edges):
    """fit edges for quantification"""

    dispersion = energy_scale[1] - energy_scale[0]
    # Determine fitting ranges and masks to exclude ranges
    mask = np.ones(len(spectrum))

    background_fit_start = edges['fit_area']['fit_start']
    if edges['fit_area']['fit_end'] > energy_scale[-1]:
        edges['fit_area']['fit_end'] = energy_scale[-1]
    background_fit_end = edges['fit_area']['fit_end']

    startx = np.searchsorted(energy_scale, background_fit_start)
    endx = np.searchsorted(energy_scale, background_fit_end)
    mask[0:startx] = 0.0
    mask[endx:-1] = 0.0
    for key in edges:
        if key.isdigit():
            if edges[key]['start_exclude'] > background_fit_start + dispersion:
                if edges[key]['start_exclude'] < background_fit_end - dispersion * 2:
                    if edges[key]['end_exclude'] > background_fit_end - dispersion:
                        # we need at least one channel to fit.
                        edges[key]['end_exclude'] = background_fit_end - dispersion
                    startx = np.searchsorted(energy_scale, edges[key]['start_exclude'])
                    if startx < 2:
                        startx = 1
                    endx = np.searchsorted(energy_scale, edges[key]['end_exclude'])
                    mask[startx: endx] = 0.0

    ########################
    # Background Fit
    ########################
    bgd_fit_area = [background_fit_start, background_fit_end]
    background, [A, r] = power_law_background(spectrum, energy_scale, bgd_fit_area, verbose=False)

    #######################
    # Edge Fit
    #######################
    x = energy_scale
    blurred = gaussian_filter(spectrum, sigma=5)

    y = blurred  # now in probability
    y[np.where(y < 1e-8)] = 1e-8

    xsec = []
    number_of_edges = 0
    for key in edges:
        if key.isdigit():
            xsec.append(edges[key]['data'])
            number_of_edges += 1
    xsec = np.array(xsec)


    def model(xx, pp):
        yy = pp[0] *  xx**pp[1] +  pp[2] + pp[3]* xx + pp[4] * xx * xx
        for i in range(number_of_edges):
            pp[i+5] = np.abs(pp[i+5])
            yy = yy + pp[i+5] * xsec[i, :]
        return yy

    def residuals(pp, xx, yy):
        err = np.abs((yy - model(xx, pp)) * mask)  / np.sqrt(np.abs(y))
        return err

    scale = y[100]
    pin = np.array([A,-r, 10., 1., 0.00] + [scale/5] * number_of_edges)
    [p, _] = leastsq(residuals, pin, args=(x, y))

    for key in edges:
        if key.isdigit():
            edges[key]['areal_density'] = p[int(key)+5]
    # print(p)
    edges['model'] = {}
    edges['model']['background'] = ( p[0] * np.power(x, -p[1])+ p[2]+ x**p[3] +  p[4] * x * x)
    edges['model']['background-poly_0'] = p[2]
    edges['model']['background-poly_1'] = p[3]
    edges['model']['background-poly_2'] = p[4]
    edges['model']['background-A'] = p[0]
    edges['model']['background-r'] = p[1]
    edges['model']['spectrum'] = model(x, p)
    edges['model']['blurred'] = blurred
    edges['model']['mask'] = mask
    edges['model']['fit_parameter'] = p
    edges['model']['fit_area_start'] = edges['fit_area']['fit_start']
    edges['model']['fit_area_end'] = edges['fit_area']['fit_end']
    edges['model']['xsec'] = xsec
    return edges

    
def core_loss_model(energy_scale, pp, number_of_edges, xsec):
    """ core loss model for fitting"""
    xx = np.array(energy_scale)
    yy = pp[0] *  xx**pp[1] +  pp[2] + pp[3]* xx + pp[4] * xx * xx
    for i in range(number_of_edges):
        pp[i+5] = np.abs(pp[i+5])
        yy = yy + pp[i+5] * xsec[i, :]
    return yy



def fit_edges(spectrum, energy_scale, region_tags, edges):
    """fit edges for quantification"""

    # Determine fitting ranges and masks to exclude ranges
    mask = np.ones(len(spectrum))

    background_fit_end = energy_scale[-1]
    for key in region_tags:
        end = region_tags[key]['start_x'] + region_tags[key]['width_x']

        startx = np.searchsorted(energy_scale, region_tags[key]['start_x'])
        endx = np.searchsorted(energy_scale, end)

        if key == 'fit_area':
            mask[0:startx] = 0.0
            mask[endx:-1] = 0.0
        else:
            mask[startx:endx] = 0.0
            if region_tags[key]['start_x'] < background_fit_end:  # Which is the onset of the first edge?
                background_fit_end = region_tags[key]['start_x']

    ########################
    # Background Fit
    ########################
    bgd_fit_area = [region_tags['fit_area']['start_x'], background_fit_end]
    background, [A, r] = power_law_background(spectrum, energy_scale, bgd_fit_area, verbose=False)

    #######################
    # Edge Fit
    #######################
    x = energy_scale
    blurred = gaussian_filter(spectrum, sigma=5)

    y = blurred  # now in probability
    y[np.where(y < 1e-8)] = 1e-8

    xsec = []
    number_of_edges = 0
    for key in edges:
        if key.isdigit():
            xsec.append(edges[key]['data'])
            number_of_edges += 1
    xsec = np.array(xsec)

    def model(xx, pp):
        yy = background + pp[6] + pp[7] * xx + pp[8] * xx * xx
        for i in range(number_of_edges):
            pp[i] = np.abs(pp[i])
            yy = yy + pp[i] * xsec[i, :]
        return yy

    def residuals(pp, xx, yy):
        err = np.abs((yy - model(xx, pp)) * mask)  # / np.sqrt(np.abs(y))
        return err

    scale = y[100]
    pin = np.array([scale / 5, scale / 5, scale / 5, scale / 5, scale / 5, scale / 5, -scale / 10, 1.0, 0.001])
    [p, _] = leastsq(residuals, pin, args=(x, y))

    for key in edges:
        if key.isdigit():
            edges[key]['areal_density'] = p[int(key) - 1]

    edges['model'] = {}
    edges['model']['background'] = (background + p[6] + p[7] * x + p[8] * x * x)
    edges['model']['background-poly_0'] = p[6]
    edges['model']['background-poly_1'] = p[7]
    edges['model']['background-poly_2'] = p[8]
    edges['model']['background-A'] = A
    edges['model']['background-r'] = r
    edges['model']['spectrum'] = model(x, p)
    edges['model']['blurred'] = blurred
    edges['model']['mask'] = mask
    edges['model']['fit_parameter'] = p
    edges['model']['fit_area_start'] = region_tags['fit_area']['start_x']
    edges['model']['fit_area_end'] = region_tags['fit_area']['start_x'] + region_tags['fit_area']['width_x']

    return edges



def get_spectrum(dataset, x=0, y=0, bin_x=1, bin_y=1):
    """
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
        if x > dataset.shape[image_dims[0]] - bin_x:
            x = dataset.shape[image_dims[0]] - bin_x
        if y > dataset.shape[image_dims[1]] - bin_y:
            y = dataset.shape[image_dims[1]] - bin_y
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

def find_peaks(dataset, energy_scale):  #, fit_start, fit_end, sensitivity=2):
    """find peaks in spectrum"""

    peaks, prop = scipy.signal.find_peaks(np.abs(dataset)+1, width=5)
    results_half = scipy.signal.peak_widths(np.abs(dataset)+1, peaks, rel_height=0.5)[0]
    
    disp = energy_scale[1] - energy_scale[0]    
    if len(peaks) > 0:
        p_in = np.ravel([[energy_scale[peaks[i]], dataset[peaks[i]], results_half[i]*disp] for i in range(len(peaks))])
    return p_in  # model, p_in

def nothing():
    pass
    """
    if dataset.data_type.name == 'SPECTRAL_IMAGE':
        if hasattr(dataset.view, 'get_spectrum'):
            spectrum = dataset.view.get_spectrum()
        else:
            spectrum = np.array(dataset[0,0])

    else:
        spectrum = np.array(dataset)

    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values

    """
    
    
    """
    start_channel = np.searchsorted(energy_scale, fit_start)
    end_channel = np.searchsorted(energy_scale, fit_end)
    peaks = []
    for index in indices:
        if start_channel < index < end_channel:
            peaks.append(index - start_channel)

    if 'model' in dataset.metadata:
        model = dataset.metadata['model']

    elif energy_scale[0] > 0:
        if 'edges' not in dataset.metadata:
            return
        if 'model' not in dataset.metadata['edges']:
            return
        model = dataset.metadata['edges']['model']['spectrum']

    else:
        model = np.zeros(len(energy_scale))

    energy_scale = energy_scale[start_channel:end_channel]

    difference = np.array(spectrum - model)[start_channel:end_channel]
    fit = np.zeros(len(energy_scale))
    if len(peaks) > 0:
        p_in = np.ravel([[energy_scale[i], difference[i], .7] for i in peaks])
    """
    


def find_maxima(y, number_of_peaks):
    """ find the first most prominent peaks

    peaks are then sorted by energy

    Parameters
    ----------
    y: numpy array
        (part) of spectrum
    number_of_peaks: int

    Returns
    -------
    numpy array
        indices of peaks
    """
    blurred2 = gaussian_filter(y, sigma=2)
    peaks, _ = scipy.signal.find_peaks(blurred2)
    prominences = peak_prominences(blurred2, peaks)[0]
    prominences_sorted = np.argsort(prominences)
    peaks = peaks[prominences_sorted[-number_of_peaks:]]

    peak_indices = np.argsort(peaks)
    return peaks[peak_indices]

# 
def model3(x, p, number_of_peaks, peak_shape, p_zl, pin=None, restrict_pos=0, restrict_width=0):
    """ model for fitting low-loss spectrum"""
    if pin is None:
        pin = p

    # if len([restrict_pos]) == 1:
    #    restrict_pos = [restrict_pos]*number_of_peaks
    # if len([restrict_width]) == 1:
    #    restrict_width = [restrict_width]*number_of_peaks
    y = np.zeros(len(x))

    for i in range(number_of_peaks):
        index = int(i * 3)
        if restrict_pos > 0:
            if p[index] > pin[index] * (1.0 + restrict_pos):
                p[index] = pin[index] * (1.0 + restrict_pos)
            if p[index] < pin[index] * (1.0 - restrict_pos):
                p[index] = pin[index] * (1.0 - restrict_pos)

        p[index + 1] = abs(p[index + 1])
        # print(p[index + 1])
        p[index + 2] = abs(p[index + 2])
        if restrict_width > 0:
            if p[index + 2] > pin[index + 2] * (1.0 + restrict_width):
                p[index + 2] = pin[index + 2] * (1.0 + restrict_width)

        if peak_shape[i] == 'Lorentzian':
            y = y + lorentz(x, p[index:])
        elif peak_shape[i] == 'zl':

            y = y + zl(x, p[index:], p_zl)
        else:
            y = y + gauss(x, p[index:])
    return y


def sort_peaks(p, peak_shape):
    """sort fitting parameters by peak position"""
    number_of_peaks = int(len(p) / 3)
    p3 = np.reshape(p, (number_of_peaks, 3))
    sort_pin = np.argsort(p3[:, 0])

    p = p3[sort_pin].flatten()
    peak_shape = np.array(peak_shape)[sort_pin].tolist()

    return p, peak_shape


def add_peaks(x, y, peaks, pin_in=None, peak_shape_in=None, shape='Gaussian'):
    """ add peaks to fitting parameters"""
    if pin_in is None:
        return
    if peak_shape_in is None:
        return

    pin = pin_in.copy()

    peak_shape = peak_shape_in.copy()
    if isinstance(shape, str):  # if peak_shape is only a string make a list of it.
        shape = [shape]

    if len(shape) == 1:
        shape = shape * len(peaks)
    for i, peak in enumerate(peaks):
        pin.append(x[peak])
        pin.append(y[peak])
        pin.append(.3)
        peak_shape.append(shape[i])

    return pin, peak_shape

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
    else:
        return p[1] * np.exp(-(x - p[0]) ** 2 / (2.0 * (p[2] / 2.3548) ** 2))


@jit
def gmm(x, p):
    y = np.zeros(len(x))
    number_of_peaks= int(len(p)/3)
    for i in range(number_of_peaks):
        index = i*3
        p[index + 1] = p[index + 1]
        # print(p[index + 1])
        p[index + 2] = abs(p[index + 2])
        y = y + gauss(x, p[index:index+3])
    return y

@jit
def residuals3(pp, xx, yy):
    err = (yy - gmm(xx, pp))
    return err

def gaussian_mixture_model(dataset, p_in=None):
    peak_model = None
    if isinstance(dataset, sidpy.Dataset):
        if dataset.data_type.name == 'SPECTRAL_IMAGE':
            if hasattr(dataset.view, 'get_spectrum'):
                spectrum = dataset.view.get_spectrum()
            else:
                spectrum = dataset[0,0]
            spectrum.data_type == 'SPECTRUM'
        else:
            spectrum = dataset
        spectrum.data_type = 'SPECTRUM'
        energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values
    else:
        spectrum = np.array(dataset)
        energy_scale = np.arange(len(spectrum))
    spectrum = np.array(spectrum)    
    #spectrum -= np.min(spectrum)-1
    if p_in is None:
        p_in = find_peaks(spectrum, energy_scale)
    
    p = fit_gmm(energy_scale, np.array(spectrum), list(p_in))
    
    peak_model = gmm(energy_scale, p)
    return peak_model, p

def fit_gmm(x, y, pin):
    """fit a Gaussian mixture model to a spectrum"""
    [p, _] = leastsq(residuals3, pin, args=(x, y),maxfev = 10000)
    return p    

              
def fit_model(x, y, pin, number_of_peaks, peak_shape, p_zl, restrict_pos=0, restrict_width=0):
    """model for fitting low-loss spectrum"""

    pin_original = pin.copy()

    

    [p, _] =  scipy.optimize.leastsq(residuals3, pin, args=(x, y),maxfev = 19400)
    # p2 = p.tolist()
    # p3 = np.reshape(p2, (number_of_peaks, 3))
    # sort_pin = np.argsort(p3[:, 0])

    # p = p3[sort_pin].flatten()
    # peak_shape = np.array(peak_shape)[sort_pin].tolist()

    return p, peak_shape



def plot_dispersion(plotdata, units, a_data, e_data, title, max_p, ee, ef=4., ep=16.8, es=0, ibt=[]):
    """Plot loss function """

    [x, y] = np.meshgrid(e_data + 1e-12, a_data[1024:2048] * 1000)

    z = plotdata
    lev = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 4.9]) * max_p / 5

    wavelength = get_wave_length(ee)
    q = a_data[1024:2048] / (wavelength * 1e9)  # in [1/nm]
    scale = np.array([0, a_data[-1], e_data[0], e_data[-1]])
    ev2hertz = constants.value('electron volt-hertz relationship')

    if units[0] == 'mrad':
        units[0] = 'scattering angle [mrad]'
        scale[1] = scale[1] * 1000.
        light_line = constants.c * a_data  # for mrad
    elif units[0] == '1/nm':
        units[0] = 'scattering vector [1/nm]'
        scale[1] = scale[1] / (wavelength * 1e9)
        light_line = 1 / (constants.c / ev2hertz) * 1e-9

    if units[1] == 'eV':
        units[1] = 'energy loss [eV]'

    if units[2] == 'ppm':
        units[2] = 'probability [ppm]'
    if units[2] == '1/eV':
        units[2] = 'probability [eV$^{-1}$ srad$^{-1}$]'

    alpha = 3. / 5. * ef / ep

    ax2 = plt.gca()
    fig2 = plt.gcf()
    im = ax2.imshow(z.T, clim=(0, max_p), origin='lower', aspect='auto', extent=scale)
    co = ax2.contour(y, x, z, levels=lev, colors='k', origin='lower')
    # ,extent=(-ang*1000.,ang*1000.,e_data[0],e_data[-1]))#, vmin = p_vol.min(), vmax = 1000)

    fig2.colorbar(im, ax=ax2, label=units[2])

    ax2.plot(a_data, light_line, c='r', label='light line')
    # ax2.plot(e_data*light_line*np.sqrt(np.real(eps_data)),e_data, color='steelblue',
    # label='$\omega = c q \sqrt{\epsilon_2}$')

    # ax2.plot(q, Ep_disp, c='r')
    ax2.plot([11.5 * light_line, 0.12], [11.5, 11.5], c='r')

    ax2.text(.05, 11.7, 'surface plasmon', color='r')
    ax2.plot([0.0, 0.12], [16.8, 16.8], c='r')
    ax2.text(.05, 17, 'volume plasmon', color='r')
    ax2.set_xlim(0, scale[1])
    ax2.set_ylim(0, 20)
    # Interband transitions
    ax2.plot([0.0, 0.25], [4.2, 4.2], c='g', label='interband transitions')
    ax2.plot([0.0, 0.25], [5.2, 5.2], c='g')
    ax2.set_ylabel(units[1])
    ax2.set_xlabel(units[0])
    ax2.legend(loc='lower right')


def xsec_xrpa(energy_scale, e0, z, beta, shift=0):
    """ Calculate momentum-integrated cross-section for EELS from X-ray photo-absorption cross-sections.

    X-ray photo-absorption cross-sections from NIST.
    Momentum-integrated cross-section for EELS according to Egerton Ultramicroscopy 50 (1993) 13-28 equation (4)

    Parameters
    ----------
    energy_scale: numpy array
        energy scale of spectrum to be analyzed
    e0: float
        acceleration voltage in keV
    z: int
        atomic number of element
    beta: float
        effective collection angle in mrad
    shift: float
        chemical shift of edge in eV
    """
    beta = beta * 0.001  # collection half angle theta [rad]
    # theta_max = self.parent.spec[0].convAngle * 0.001  # collection half angle theta [rad]
    dispersion = energy_scale[1] - energy_scale[0]

    x_sections = get_x_sections(z)
    enexs = x_sections['ene']
    datxs = x_sections['dat']

    # enexs = enexs[:len(datxs)]

    #####
    # Cross Section according to Egerton Ultramicroscopy 50 (1993) 13-28 equation (4)
    #####

    # Relativistic correction factors
    t = 511060.0 * (1.0 - 1.0 / (1.0 + e0 / 511.06) ** 2) / 2.0
    gamma = 1 + e0 / 511.06
    a = 6.5  # e-14 *10**14
    b = beta

    theta_e = enexs / (2 * gamma * t)

    g = 2 * np.log(gamma) - np.log((b ** 2 + theta_e ** 2) / (b ** 2 + theta_e ** 2 / gamma ** 2)) - (
            gamma - 1) * b ** 2 / (b ** 2 + theta_e ** 2 / gamma ** 2)
    datxs = datxs * (a / enexs / t) * (np.log(1 + b ** 2 / theta_e ** 2) + g) / 1e8

    datxs = datxs * dispersion  # from per eV to per dispersion
    coeff = splrep(enexs, datxs, s=0)  # now in areal density atoms / m^2
    xsec = np.zeros(len(energy_scale))
    # shift = 0# int(ek -onsetXRPS)#/dispersion
    lin = interp1d(enexs, datxs, kind='linear')  # Linear instead of spline interpolation to avoid oscillations.
    if energy_scale[0] < enexs[0]:
        start = np.searchsorted(energy_scale, enexs[0])+1
    else:
        start = 0
    xsec[start:] = lin(energy_scale[start:] - shift)

    return xsec


##########################
# EELS Database
##########################


def read_msa(msa_string):
    """read msa formated file"""
    parameters = {}
    y = []
    x = []
    # Read the keywords
    data_section = False
    msa_lines = msa_string.split('\n')

    for line in msa_lines:
        if data_section is False:
            if len(line) > 0:
                if line[0] == "#":
                    try:
                        key, value = line.split(': ')
                        value = value.strip()
                    except ValueError:
                        key = line
                        value = None
                    key = key.strip('#').strip()

                    if key != 'SPECTRUM':
                        parameters[key] = value
                    else:
                        data_section = True
        else:
            # Read the data

            if len(line) > 0 and line[0] != "#" and line.strip():
                if parameters['DATATYPE'] == 'XY':
                    xy = line.replace(',', ' ').strip().split()
                    y.append(float(xy[1]))
                    x.append(float(xy[0]))
                elif parameters['DATATYPE'] == 'Y':
                    print('y')
                    data = [
                        float(i) for i in line.replace(',', ' ').strip().split()]
                    y.extend(data)
    parameters['data'] = np.array(y)
    if 'XPERCHAN' in parameters:
        parameters['XPERCHAN'] = str(parameters['XPERCHAN']).split(' ')[0]
        parameters['OFFSET'] = str(parameters['OFFSET']).split(' ')[0]
        parameters['energy_scale'] = np.arange(len(y)) * float(parameters['XPERCHAN']) + float(parameters['OFFSET'])
    return parameters


def get_spectrum_eels_db(formula=None, edge=None, title=None, element=None):
    """
    get spectra from EELS database
    chemical formula and edge is accepted.
    Could expose more of the search parameters
    """
    valid_edges = ['K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N4,5', 'O2,3', 'O4,5']
    if edge is not None and edge not in valid_edges:
        print('edge should be a in ', valid_edges)

    spectrum_type = None
    title = title
    author = None
    element = element
    min_energy = None
    max_energy = None
    resolution = None
    min_energy_compare = "gt"
    max_energy_compare = "lt",
    resolution_compare = "lt"
    max_n = -1
    monochromated = None
    order = None
    order_direction = "ASC"
    verify_certificate = True
    # Verify arguments

    if spectrum_type is not None and spectrum_type not in {'coreloss', 'lowloss', 'zeroloss', 'xrayabs'}:
        raise ValueError("spectrum_type must be one of \'coreloss\', \'lowloss\', "
                         "\'zeroloss\', \'xrayabs\'.")
    # valid_edges = ['K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N4,5', 'O2,3', 'O4,5']

    params = {
        "type": spectrum_type,
        "title": title,
        "author": author,
        "edge": edge,
        "min_energy": min_energy,
        "max_energy": max_energy,
        "resolution": resolution,
        "resolution_compare": resolution_compare,
        "monochromated": monochromated,
        "formula": formula,
        'element': element,
        "min_energy_compare": min_energy_compare,
        "max_energy_compare": max_energy_compare,
        "per_page": max_n,
        "order": order,
        "order_direction": order_direction,
    }

    request = requests.get('http://api.eelsdb.eu/spectra', params=params, verify=True)
    # spectra = []
    jsons = request.json()
    if "message" in jsons:
        # Invalid query, EELSdb raises error.
        raise IOError(
            "Please report the following error to the HyperSpy developers: "
            "%s" % jsons["message"])
    reference_spectra = {}
    for json_spectrum in jsons:
        download_link = json_spectrum['download_link']
        # print(download_link)
        msa_string = requests.get(download_link, verify=verify_certificate).text
        # print(msa_string[:100])
        parameters = read_msa(msa_string)
        if 'XPERCHAN' in parameters:
            reference_spectra[parameters['TITLE']] = parameters
            print(parameters['TITLE'])
    print(f'found {len(reference_spectra.keys())} spectra in EELS database)')

    return reference_spectra
