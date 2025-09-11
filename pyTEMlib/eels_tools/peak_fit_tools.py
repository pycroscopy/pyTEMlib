"""
peak_fit-tools of eels_tools
Model based quantification of electron energy-loss data
Copyright by Gerd Duscher

The University of Tennessee, Knoxville
Department of Materials Science & Engineering


Units:
    everything is in SI units, except length is given in nm and angles in mrad.

Usage:
    See the notebooks for examples of these routines

All the input and output is done through a dictionary which is to be found in the meta_data
attribute of the sidpy.Dataset
"""
import numpy as np
from numba import jit

import scipy
import sidpy


# ###############################################################
# Peak Fit Functions
# ################################################################

def residuals_smooth(p: np.ndarray,
                     x: np.ndarray,
                     y: np.ndarray,
                     only_positive_intensity: bool) -> np.ndarray:
    """part of fit"""
    err = y - model_smooth(x, p, only_positive_intensity)
    return err


def model_smooth(x: np.ndarray,
                 p: np.ndarray,
                 only_positive_intensity: bool = False) -> np.ndarray:
    """part of fit"""

    y = np.zeros(len(x))
    number_of_peaks = int(len(p) / 3)
    for i in range(number_of_peaks):
        if only_positive_intensity:
            p[i * 3 + 1] = abs(p[i * 3 + 1])
        p[i * 3 + 2] = abs(p[i * 3 + 2])
        if p[i * 3 + 2] > abs(p[i * 3]) * 4.29193 / 2.0:
            # width cannot extend beyond zero, maximum is FWTM/2
            p[i * 3 + 2] = abs(p[i * 3]) * 4.29193 / 2.0
        y = y + gauss(x, p[i * 3:])
    return y

# ###############################################################
# Gaussian Mixing Model Functions
# ################################################################

@jit
def gauss(x, p):
    """Gaussian Function

        p[0]==mean, p[1]= amplitude p[2]==fwhm
        area = np.sqrt(2* np.pi)* p[1] * np.abs(p[2] / 2.3548)
        FWHM = 2 * np.sqrt(2 np.log(2)) * sigma = 2.3548 * sigma
        sigma = FWHM/3548
    """
    if p[2] == 0:
        return x * 0.
    return p[1] * np.exp(-(x - p[0])**2 / (2.0 * (p[2] / 2.3548)**2))


@jit
def gmm(x, p):
    """Gaussian Mixture Model"""    
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
    """Residuals for Gaussian Mixture Model"""
    err = yy - gmm(xx, pp)
    return err


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
    blurred2 = scipy.ndimage.gaussian_filter(y, sigma=2)
    peaks, _ = scipy.signal.find_peaks(blurred2)
    prominences =  scipy.signal.peak_prominences(blurred2, peaks)[0]
    prominences_sorted = np.argsort(prominences)
    peaks = peaks[prominences_sorted[-number_of_peaks:]]

    peak_indices = np.argsort(peaks)
    return peaks[peak_indices]


def find_peaks(dataset, energy_scale):  #, fit_start, fit_end, sensitivity=2):
    """find peaks in spectrum"""

    peaks, _ = scipy.signal.find_peaks(np.abs(dataset)+1, width=5)
    results_half = scipy.signal.peak_widths(np.abs(dataset)+1, peaks, rel_height=0.5)[0]
    disp = energy_scale[1] - energy_scale[0]
    p_in = []
    if len(peaks) > 0:
        p_in = np.ravel([[energy_scale[peaks[i]], dataset[peaks[i]],
                          results_half[i]*disp] for i in range(len(peaks))])
    return p_in  # model, p_in


def gaussian_mixture_model(dataset, p_in=None):
    """Fit a Gaussian mixture model to a spectrum or a spectrum image"""
    peak_model = None
    if isinstance(dataset, sidpy.Dataset):
        if dataset.data_type.name == 'SPECTRAL_IMAGE':
            if hasattr(dataset.view, 'get_spectrum'):
                spectrum = dataset.view.get_spectrum()
            else:
                spectrum = dataset[0,0]
            spectrum.data_type = 'SPECTRUM'
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
    [p, _] = scipy.optimize.leastsq(residuals3, pin, args=(x, y),maxfev = 10000)
    return p


def sort_peaks(p, peak_shape):
    """sort fitting parameters by peak position"""
    number_of_peaks = int(len(p) / 3)
    p3 = np.reshape(p, (number_of_peaks, 3))
    sort_pin = np.argsort(p3[:, 0])

    p = p3[sort_pin].flatten()
    peak_shape = np.array(peak_shape)[sort_pin].tolist()

    return p, peak_shape
