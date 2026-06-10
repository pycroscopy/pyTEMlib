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
def gauss(x: np.ndarray, 
          p: list[float]) -> np.ndarray:
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
        p[index + 2] = abs(p[index + 2])
        y = y + gauss(x, p[index:index+3])
    return y

@jit
def residuals3(pp: list[float],
               xx: np.ndarray,
               yy: np.ndarray) -> np.ndarray:
    """Residuals for Gaussian Mixture Model"""
    err = yy - gmm(xx, pp)
    return err


def find_maxima(y: np.ndarray, 
                number_of_peaks: int) -> np.ndarray:
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


def find_peaks(dataset: sidpy.Dataset| np.ndarray,
               energy_scale: np.ndarray) -> list[list[float]]:
    """find peaks in spectrum"""

    peaks, _ = scipy.signal.find_peaks(np.abs(dataset)+1, width=5)
    results_half = scipy.signal.peak_widths(np.abs(dataset)+1, peaks, rel_height=0.5)[0]
    disp = energy_scale[1] - energy_scale[0]
    p_in = []
    if len(peaks) > 0:
        p_in = np.ravel([[energy_scale[peaks[i]], dataset[peaks[i]],
                          results_half[i]*disp] for i in range(len(peaks))])
    return p_in  # model, p_in

def find_relevant_peaks(spectrum, number_of_peaks=9):
    """Find relevant peaks in the gaussian mixture model."""
    peak_dict = spectrum.metadata['peak_fit']
    model = peak_dict['peak_model']
    number_of_peaks = min(number_of_peaks, len(peak_dict['peak_gmm_list']))
    noise_level = np.std((spectrum-model)[300:])/10
    print(noise_level)
    peaks = {'peaks': {}}
    new_number_of_peaks = 0
    peak_dict['peak_out_list'] = []
    for i in range(number_of_peaks):
        p = peak_dict['peak_gmm_list'][i]
        if abs(p[1])>noise_level:
            peak_dict['peak_out_list'].append(p)
            new_peak = {'position': p[0], 'amplitude': p[1], 'width': p[2],
                        'type': 'Gauss', 'asymmetry': 0}
            peaks['peaks'][str(new_number_of_peaks)] = new_peak
            new_number_of_peaks += 1
    return new_number_of_peaks
    
    
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

def fit_peaks(spectrum):
    """Fit spectrum with peaks given in peaks dictionary"""
    
    peak_dict = spectrum.metadata['peak_fit']
    model = peak_dict['start_model']
    full_energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values
    start_channel = np.searchsorted(full_energy_scale, peak_dict['fit_area']['fit_start'])
    end_channel = np.searchsorted(full_energy_scale, peak_dict['fit_area']['fit_end'])

    energy_scale = full_energy_scale[start_channel:end_channel]
    # select the core loss model if it exists. Otherwise, we will fit to the full spectrum.
    # if we have a core loss model we will only fit the difference
    # between the model and the data.
    diff = np.array(spectrum[start_channel:end_channel] - model[start_channel:end_channel])
    p_in =  peak_dict['peak_out_list'] #peak_gmm_list[:]
    # find the optimum fitting parameters

    [p_out, _] = scipy.optimize.leastsq(residuals3,
                                        np.array(p_in, dtype=np.float64),
                                        args=(energy_scale, diff))  # , False))
    # construct the fit data from the optimized parameters
    peak_model = gmm(full_energy_scale, p_out)  # , False)

    if 'parameter' not in spectrum.metadata['peak_fit']:
        if spectrum.data_type.name == 'SPECTRUM':
            spectrum.metadata['peak_fit']['parameter'] = np.zeros([1,1, len(p_out)])
        else:
            spectrum.metadata['peak_fit']['parameter'] = np.zeros([spectrum.shape[0],
                                                                   spectrum.shape[1],
                                                                       len(p_out)])
    peak_dict['peaks'] = {}
    for index in range(int(len(p_out)/3)):
        p_index = index *3
        peak_dict['peaks'][index] = {'position': p_out[p_index],
                                       'amplitude': p_out[p_index+1],
                                       'width': p_out[p_index+2],
                                       'type': 'Gauss',
                                       'associated_edge': ''}    
    spectrum.metadata['peak_fit']['peak_model'] = peak_model
    print(p_out)
    p_out = np.reshape(p_out, [len(p_out) // 3, 3])
    spectrum.metadata['peak_fit']['peak_out_list']  = p_out
    #spectrum.metadata['peak_fit']['peaks'] = peaks.copy()

    
    
def get_gmm(spectrum, start_model=None):
    if start_model is not None:
        peak_model, peak_out_list = gaussian_mixture_model(spectrum-start_model)
    else:
        peak_model, peak_out_list = gaussian_mixture_model(spectrum)
    new_list = np.reshape(peak_out_list, [len(peak_out_list) // 3, 3])
    area = np.sqrt(2 * np.pi) * np.abs(new_list[:, 1])
    area *= np.abs(new_list[:, 2] / np.sqrt(2 * np.log(2)))
    arg_list = np.argsort(area)[::-1]
    area = area[arg_list]
    peak_out_list = new_list[arg_list]
    
    number_of_peaks = np.searchsorted(area * -1, -np.average(area))
    spectrum.metadata.setdefault('peak_fit', {})
    spectrum.metadata['peak_fit']['start_model'] = spectrum.metadata['core_loss']['model']['spectrum']
    spectrum.metadata['peak_fit']['peak_model'] = peak_model
    spectrum.metadata['peak_fit']['peak_gmm_list'] = peak_out_list
    
    return number_of_peaks
