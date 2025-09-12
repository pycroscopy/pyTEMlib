"""
zero-loss tools part of eels tools in pyTEMlib
"""

import numpy as np
import scipy

import sidpy
from sidpy.proc.fitter import SidFitter

from ..utilities import lorentz, gauss


def zero_loss_function(x, p):
    """ zero-loss function as product of two lorentzians """
    return zl_func(x, *p)


def zl_func(x, center1, amplitude1, width1, center2, amplitude2, width2):
    """ zero loss function as product of two lorentzians """
    zero_loss = lorentz(x, center1, amplitude1, width1) * lorentz(x, center2, amplitude2, width2)
    return zero_loss


def zl(x, p, p_zl):
    """zero-loss function"""
    p_zl_local = p_zl.copy()
    p_zl_local[2] += p[0]
    p_zl_local[5] += p[0]
    zero_loss = zero_loss_function(x, p_zl_local)
    return p[1] * zero_loss / zero_loss.max()


def get_channel_zero(spectrum: np.ndarray, energy: np.ndarray, width: int = 8):
    """Determine shift of energy scale according to zero-loss peak position
    
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

    [p1, _] = scipy.optimize.leastsq(errfunc, np.array(p0[:]), args=(x, y))
    fit_mu, _, fwhm = p1

    return fwhm, fit_mu


def get_zero_loss_energy(dataset: sidpy.Dataset) -> np.ndarray:
    """ Determine zero-loss peaks of EELS spectral sidpy dataset """
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
        tck = scipy.interpolate.splrep(np.array(energy_scale - shifts), np.array(dataset), k=1, s=0)
        new_si[:] = scipy.interpolate.splev(energy_scale, tck, der=0)
        new_si.data_type = 'Spectrum'
    elif dataset.ndim == 2:  # line scan
        for x in range(dataset.shape[0]):
            tck = scipy.interpolate.splrep(np.array(energy_scale - shifts[x]),
                                           np.array(dataset[x, :]), k=1, s=0)
            new_si[x, :] = scipy.interpolate.splev(energy_scale, tck, der=0)
    elif dataset.ndim == 3:  # spectral image
        for x in range(dataset.shape[0]):
            for y in range(dataset.shape[1]):
                tck = scipy.interpolate.splrep(np.array(energy_scale - shifts[x, y]),
                                         np.array(dataset[x, y]), k=1, s=0)
                new_si[x, y, :] = scipy.interpolate.splev(energy_scale, tck, der=0)

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


def get_zero_losses(energy, z_loss_params):
    """Calculate zero-loss peaks for a given energy range and parameters."""
    z_loss_dset = np.zeros((z_loss_params.shape[0], z_loss_params.shape[1], energy.shape[0]))
    for x in range(z_loss_params.shape[0]):
        for y in range(z_loss_params.shape[1]):
            z_loss_dset[x, y] +=  zl_func(energy, *z_loss_params[x, y])
    return z_loss_dset




def get_resolution_functions(dataset: sidpy.Dataset, start_fit_energy: float=-1,
                             end_fit_energy: float=+1,
                             n_workers: int=1, n_threads: int=8):
    """
    Analyze and fit low-loss EELS data within a specified energy range to determine zero-loss peaks.

    This function processes a low-loss EELS dataset from transmission electron microscopy 
    (TEM) data,  focusing on a specified energy range for analyzing and fitting the spectrum.
    It determines fitting parameters and applies these to extract zero-loss peak information 
    from the dataset. The function handles both 2D and 3D datasets.

    Parameters:
    -----------
        dataset (sidpy.Dataset): The dataset containing TEM spectral data.
        start_fit_energy (float): The start energy of the fitting window.
        end_fit_energy (float): The end energy of the fitting window.
        n_workers (int, optional): The number of workers for parallel processing (default is 1).
        n_threads (int, optional): The number of threads for parallel processing (default is 8).

    Returns:
    --------
        tuple: A tuple containing:
            - z_loss_dset (sidpy.Dataset): The dataset with added zero-loss peak information.
            - z_loss_params (numpy.ndarray): Array of parameters used 
              for the zero-loss peak fitting.

    Raises:
    -------
        ValueError: If the input dataset does not have the expected dimensions or format.

    Notes:
    ------
        - The function expects `dset` to have specific dimensionalities and will raise an error 
            if they are not met.
        - Parallel processing is employed to enhance performance, particularly for large datasets.
    """
    energy = dataset.get_spectral_dims(return_axis=True)[0].values
    start_fit_pixel = np.searchsorted(energy, start_fit_energy)
    end_fit_pixel = np.searchsorted(energy, end_fit_energy)
    guess_width = (end_fit_pixel - start_fit_pixel)/2
    if end_fit_pixel - start_fit_pixel < 5:
        start_fit_pixel -= 2
        end_fit_pixel += 2

    def get_good_guess(zl_peak, energy, spectrum):
        popt, _ = scipy.optimize.curve_fit(zl_peak, energy, spectrum,
                                              p0=[0, guess_amplitude, guess_width,
                                                  0, guess_amplitude, guess_width])
        return popt

    fit_energy = energy[start_fit_pixel:end_fit_pixel]
    # get a good guess for the fit parameters
    if len(dataset.shape) == 3:
        fit_dset = dataset[:, :, start_fit_pixel:end_fit_pixel]
        guess_amplitude = np.sqrt(fit_dset.max())
        image_size = fit_dset.shape[0]/fit_dset.shape[1]
        guess_params = get_good_guess(zl_func, fit_energy,
                                      fit_dset.sum(axis=(0, 1))/image_size)
    elif len(dataset.shape) == 2:
        fit_dset = dataset[:, start_fit_pixel:end_fit_pixel]
        fit_energy = energy[start_fit_pixel:end_fit_pixel]
        guess_amplitude = np.sqrt(fit_dset.max())
        guess_params = get_good_guess(zl_func, fit_energy,
                                      fit_dset.sum(axis=0)/fit_dset.shape[0])
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
        tags = {'start_fit_energy': start_fit_energy,
                'end_fit_energy': end_fit_energy,
                'fit_parameter': guess_params,
                'original_low_loss': dataset.title}
        z_loss_dset.metadata['zero_loss'].update(tags)
        return z_loss_dset
    else:
        print('Error: need a spectrum or spectral image sidpy dataset')
        print('Not dset.shape = ', dataset.shape)
        return None

    # define guess function for SidFitter
    def guess_function(xvec, yvec):
        return guess_params

    # apply to all spectra
    zero_loss_fitter = SidFitter(fit_dset, zl_func, num_workers=n_workers,
                                 guess_fn=guess_function, threads=n_threads,
                                 return_cov=False, return_fit=False,
                                 return_std=False, km_guess=False, num_fit_parms=6)
    [z_loss_params] = zero_loss_fitter.do_fit()
    z_loss_dset = dataset.copy()
    z_loss_dset *= 0.0
    z_loss_params = np.array(z_loss_params)
    z_loss_dset += get_zero_losses(np.array(energy), np.array(z_loss_params))

    # shifts = z_loss_params[:, :, 0] * z_loss_params[:, :, 3]
    # widths = z_loss_params[:, :, 2] * z_loss_params[:, :, 5]
    tags = {'start_fit_energy': start_fit_energy,
            'end_fit_energy': end_fit_energy,
            'fit_parameter': z_loss_params,
            'original_low_loss': dataset.title}
    z_loss_dset.metadata['zero_loss'].update(tags)
    return z_loss_dset
