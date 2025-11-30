"""
eds_tools
Model based quantification of energy-dispersive X-ray spectroscopy data
Copyright by Gerd Duscher

The University of Tennessee, Knoxville
Department of Materials Science & Engineering

Sources:
   
Units:
    everything is in SI units, except length is given in nm and angles in mrad.
Usage:
    See the notebooks for examples of these routines

All the input and output is done through a dictionary which is to be found in the meta_data
attribute of the sidpy.Dataset
"""

import os
import csv
import json
import xml

import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.interpolate  # use interp1d,
import scipy.optimize  # leastsq  # least square fitting routine fo scipy
import sklearn  # .mixture import GaussianMixture

import sidpy

import pyTEMlib
import pyTEMlib.file_reader
from .utilities import elements as elements_list
from .eds_xsections import quantify_cross_section, quantification_k_factors
from .config_dir import config_path



def detector_response(dataset):
    """
    Calculate the detector response for the given dataset based on its metadata.

    Parameters:
    - dataset: A sidpy.Dataset object containing the spectral data and metadata.

    Returns:
    - A numpy array representing the detector efficiency across the energy scale.
    """
    tags = dataset.metadata['EDS']

    energy_scale = dataset.get_spectral_dims(return_axis=True)[0]
    if 'start_channel' not in tags['detector']:
        tags['detector']['start_channel'] = np.searchsorted(energy_scale, 100)

    start = tags['detector']['start_channel']
    detector_efficiency = np.zeros(len(dataset))
    detector_efficiency[start:] += get_detector_response(tags, energy_scale[start:])
    tags['detector']['detector_efficiency'] = detector_efficiency
    return detector_efficiency

def get_absorption(z, thickness, energy_scale):
    """ Calculate absorption for material with atomic number z and thickness t in m"""
    x_sections = pyTEMlib.eels_tools.get_x_sections()
    photoabsorption = x_sections[str(z)]['dat']/1e10/x_sections[str(z)]['photoabs_to_sigma']
    lin = scipy.interpolate.interp1d(x_sections[str(z)]['ene'], photoabsorption, kind='linear')
    mu = lin(energy_scale) * x_sections[str(z)]['nominal_density']*100  #1/cm -> 1/m
    return np.exp(-mu * thickness)


def get_detector_response(detector_definition, energy_scale):
    """
    Calculates response of Si drift detector for EDS spectrum background based 
    on detector parameters

    Parameters:
    ----------
    detector_definition: dictionary
        definition of detector
    energy_scale: numpy array (1 dim)
        energy scale of spectrum should start at about 100eV!!

    Return:
    -------
    response: numpy array with length(energy_scale)
        detector response

    Example
    -------

    tags ={}
    tags['acceleration_voltage'] = 200000

    tags['detector'] ={}

    ## layer thicknesses of common materials in EDS detectors in m
    tags['detector']['layers'] = {13: {'thickness':= 0.05*1e-6, 'Z': 13, 'element': 'Al'},
                                  6: {'thickness':= 0.15*1e-6, 'Z': 6, 'element': 'C'}
                                  }
    tags['detector']['SiDeadThickness'] = .13 *1e-6  # in m
    tags['detector']['SiLiveThickness'] = 0.05  # in m
    tags['detector']['detector_area'] = 30 * 1e-6 #in m2
    tags['detector']['energy_resolution'] = 125  # in eV
    tags['detector']['start_energy'] = 120  # in eV
    tags['detector']['start_channel'] = np.searchsorted(spectrum.energy_scale.values,120)

    energy_scale = np.linspace(.01, 20, 1199)*1000 # i eV
    start = np.searchsorted(spectrum.energy, 100)
    energy_scale = spectrum.energy[start:]
    detector_Efficiency= pyTEMlib.eds_tools.detector_response(tags, spectrum.energy[start:])

    p = np.array([1, 37, .3])/10000*3
    E_0= 200000
    background = np.zeros(len(spectrum))
    bremsstrahlung =  p[0] + p[1]*(E_0-energy_scale)/energy_scale 
    bremsstrahlung += p[2]*(E_0-energy_scale)**2/energy_scale
    background[start:] = detector_Efficiency * bremsstrahlung

    plt.figure()
    plt.plot(spectrum.energy, spectrum, label = 'spec')
    plt.plot(spectrum.energy, background, label = 'background')
    plt.show()

    """
    response = np.ones(len(energy_scale))

    for layer in detector_definition['detector']['layers'].values():
        if layer['Z'] != 14:
            response *= get_absorption(layer['Z'], layer['thickness'], energy_scale)
    if 'SiDeadThickness' in  detector_definition['detector']:
        response *= get_absorption(14, detector_definition['detector']['SiDeadThickness'],
                                   energy_scale)
    if 'SiLiveThickness' in  detector_definition['detector']:
        response *= 1-get_absorption(14, detector_definition['detector']['SiLiveThickness'],
                                     energy_scale)
    return response


def detect_peaks(dataset, minimum_number_of_peaks=30, prominence=10):
    """
    Detect peaks in the given spectral dataset.

    Parameters:
    -----------
    - dataset: A sidpy.Dataset object containing the spectral data.
    - minimum_number_of_peaks: The minimum number of peaks to detect.
    - prominence: The prominence threshold for peak detection.

    Returns:
    --------
    - An array of indices representing the positions of detected peaks in the spectrum.
    """
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('Needs an sidpy dataset')
    if not dataset.data_type.name == 'SPECTRUM':
        raise TypeError('Need a spectrum')

    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values
    if 'EDS' not in dataset.metadata:
        dataset.metadata['EDS'] = {}
    if 'detector' not in dataset.metadata['EDS']:
        raise ValueError('No detector information found, add detector dictionary to metadata')

    if 'energy_resolution' not in dataset.metadata['EDS']['detector']:
        dataset.metadata['EDS']['detector']['energy_resolution'] = 138
        print('Using energy resolution of 138 eV')
    if 'start_channel' not in dataset.metadata['EDS']['detector']:
        dataset.metadata['EDS']['detector']['start_channel'] =  np.searchsorted(energy_scale, 100)

    resolution = dataset.metadata['EDS']['detector']['energy_resolution']
    start = dataset.metadata['EDS']['detector']['start_channel']
    ## we use half the width of the resolution for smearing
    width = int(np.ceil(resolution/(energy_scale[1]-energy_scale[0])/2)+1)
    new_spectrum =  scipy.signal.savgol_filter(np.array(dataset)[start:], width, 2)

    minor_peaks, _  = scipy.signal.find_peaks(new_spectrum, prominence=prominence)

    while len(minor_peaks) > minimum_number_of_peaks:
        prominence+=10
        minor_peaks, _  = scipy.signal.find_peaks(new_spectrum, prominence=prominence)
    return np.array(minor_peaks)+start

def peaks_element_correlation(spectrum, minor_peaks):
    """
    Identify elements present in the spectrum based on detected minor peaks.

    Parameters:
    - spectrum: A sidpy.Dataset object containing the spectral data.
    - minor_peaks: An array of indices representing the positions of
                   minor peaks in the spectrum.

    Returns:
    - A list of element symbols identified in the spectrum.
    """
    if not isinstance(spectrum, sidpy.Dataset):
        raise TypeError(' Need a sidpy dataset')
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values
    element_list = set()
    peaks = minor_peaks[np.argsort(spectrum[minor_peaks])]
    accounted_peaks = set()
    for i, peak in reversed(list(enumerate(peaks))):
        for z in range(5, 82):
            if i in accounted_peaks:
                continue
            edge_info  = pyTEMlib.eels_tools.get_x_sections(z)
            # element = edge_info['name']
            lines = edge_info.get('lines', {})
            if abs(lines.get('K-L3', {}).get('position', 0) - energy_scale[peak]) <40:
                element_list.add(edge_info['name'])
                for key, line in lines.items():
                    dist = np.abs(energy_scale[peaks]-line.get('position', 0))
                    if key[0] == 'K' and np.min(dist)< 40:
                        ind = np.argmin(dist)
                        accounted_peaks.add(ind)
            # This is a special case for boron and carbon
            elif abs(lines.get('K-L2', {}).get('position', 0) - energy_scale[peak]) <30:
                accounted_peaks.add(i)
                element_list.add(edge_info['name'])

            if abs(lines.get('L3-M5', {}).get('position', 0) - energy_scale[peak]) <50:
                element_list.add(edge_info['name'])
                for key, line in edge_info['lines'].items():
                    dist = np.abs(energy_scale[peaks]-line.get('position', 0))
                    if key[0] == 'L' and np.min(dist)< 40 and line['weight'] > 0.01:
                        ind = np.argmin(dist)
                        accounted_peaks.add(ind)
    return list(element_list)


def get_elements(spectrum, minimum_number_of_peaks=10, verbose=False):
    """ Get the elments in a EDS spectrum 
    Parameters:
    -----------
    minimum_number_of_peaks: int
        approximate number of peaks in spectrum

    Returns:
    -------
    elements: list
        list of all elements found    
    """
    if not isinstance(spectrum, sidpy.Dataset):
        raise TypeError(' Need a sidpy dataset')
    if not isinstance(minimum_number_of_peaks, int):
        raise TypeError(' Need an integer for minimum_number_of_peaks')

    minor_peaks = detect_peaks(spectrum, minimum_number_of_peaks=minimum_number_of_peaks)

    keys = list(spectrum.metadata['EDS'].keys())
    for key in keys:
        if len(key) < 3:
            del spectrum.metadata['EDS'][key]

    elements = peaks_element_correlation(spectrum, minor_peaks)
    if verbose:
        print(elements)
    spectrum.metadata['EDS'].update(get_x_ray_lines(spectrum, elements))
    return elements

def get_x_ray_lines(spectrum, element_list):
    """
    Analyze the given spectrum to identify and characterize the X-ray emission lines
    associated with the specified elements.

    Parameters:
    - spectrum: A sidpy.Dataset object containing the spectral data.
    - elements: A list of element symbols (e.g., ['Fe', 'Cu']) to look for in the spectrum.

    Returns:
    - A dictionary where each key is an element symbol and each value is another dictionary
      containing information about the X-ray lines detected for that element.
    
    alpha_k = 1e6
    alpha_l = 6.5e7
    alpha_m = 8*1e8  # 2.2e10
    # My Fit
    alpha_K = .9e6
    alpha_l = 6.e7
    alpha_m = 6*1e8 #  2.2e10
    # omega_K = Z**4/(alpha_K+Z**4)
    # omega_L = Z**4/(alpha_l+Z**4)
    # omega_M = Z**4/(alpha_m+Z**4)
    """

    out_tags = {}
    x_sections = pyTEMlib.xrpa_x_sections.x_sections
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values
    for element in element_list:
        atomic_number = pyTEMlib.eds_tools.elements_list.index(element)
        out_tags[element] ={'Z': atomic_number}
        lines = pyTEMlib.xrpa_x_sections.x_sections.get(str(atomic_number), {}).get('lines', {})
        if not lines:
            break
        line_dict = {'K': {'lines': [],
                           'main': None,
                           'weight': 0}, 
                     'L': {'lines': [],
                           'main': None,
                           'weight': 0}, 
                     'M': {'lines': [],
                           'main': None,
                           'weight': 0}}

        for key, line in lines.items():
            if key[0] in line_dict:
                if line['position'] < energy_scale[-1]:
                    line_dict[key[0]]['lines'].append(key)
                    if line['weight'] > line_dict[key[0]]['weight']:
                        line_dict[key[0]]['weight'] = line['weight']
                        line_dict[key[0]]['main'] = key

        for key, family in line_dict.items():
            if family['weight'] > 0:
                out_tags[element].setdefault(f'{key}-family', {}).update(family)
                position = x_sections[str(atomic_number)]['lines'][family['main']]['position']
                height = spectrum[np.searchsorted(energy_scale, position)].compute()
                out_tags[element][f'{key}-family']['height'] = height/family['weight']
                z = str(atomic_number)
                for key in family['lines']:
                    out_tags[element][f'{key[0]}-family'][key] = x_sections[z]['lines'][key]
        spectrum.metadata.setdefault('EDS', {}).update(out_tags)
    return out_tags


def get_fwhm(energy: float, energy_ref: float, fwhm_ref: float) -> float:
    """ Calculate FWHM of Gaussians"""
    return np.sqrt(2.5*(energy-energy_ref)+fwhm_ref**2)


def gaussian(energy_scale: np.ndarray, mu: float, fwhm: float) -> np.ndarray:
    """ Gaussian function"""
    sig = fwhm/2/np.sqrt(2*np.log(2))
    return np.exp(-np.power(np.array(energy_scale) - mu, 2.) / (2 * np.power(sig, 2.)))


def get_peak(energy: float, energy_scale: np.ndarray,
             energy_ref: float = 5895.0, fwhm_ref: float = 136) -> np.ndarray:
    """ Generate a normalized Gaussian peak for a given energy."""
    # all energies in eV
    fwhm  = get_fwhm(energy, energy_ref, fwhm_ref)
    gauss = gaussian(energy_scale, energy, fwhm)

    return gauss /(gauss.sum()+1e-12)


def initial_model_parameter(spectrum):
    """ Initialize model parameters based on the spectrum's metadata.""" 
    tags = spectrum.metadata['EDS']
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0]
    p = []
    peaks = []
    keys = []
    for element, lines in tags.items():
        if 'K-family' in lines:
            model = np.zeros(len(energy_scale))
            for line, info in lines['K-family'].items():
                if line[0] == 'K':
                    model += get_peak(info['position'], energy_scale)*info['weight']
            lines['K-family']['peaks'] = model  /model.sum()  # *lines['K-family']['probability']

            p.append(lines['K-family']['height'] / lines['K-family']['peaks'].max())
            peaks.append(lines['K-family']['peaks'])
            keys.append(element+':K-family')
        if 'L-family' in lines:
            model = np.zeros(len(energy_scale))
            for line, info in lines['L-family'].items():
                if line[0] == 'L':
                    model += get_peak(info['position'], energy_scale)*info['weight']
            lines['L-family']['peaks'] = model  /model.sum() # *lines['L-family']['probability']
            p.append(lines['L-family']['height'] / lines['L-family']['peaks'].max())
            peaks.append(lines['L-family']['peaks'])
            keys.append(element+':L-family')
        if 'M-family' in lines:
            model = np.zeros(len(energy_scale))
            for line, info in lines['M-family'].items():
                if line[0] == 'M':
                    model += get_peak(info['position'], energy_scale)*info['weight']
            peaks_max = lines['M-family'].get('peaks', np.zeros(3)).max()
            model_normalized = model / model.sum()*lines['M-family'].get('probability', 0.0)
            lines['M-family']['peaks'] = model_normalized
            if peaks_max >0:
                p.append(lines['M-family']['height'] / peaks_max)
            else:
                p.append(0)
            peaks.append(lines['M-family']['peaks'])
            keys.append(element+':M-family')

    p.extend([1e7, 1e-3, 1500, 20])
    return np.array(peaks), np.array(p), keys

def get_model(spectrum):
    """
    Construct the model spectrum from the metadata in the given spectrum object.

    Parameters:
    - spectrum: The spectrum object containing metadata and spectral data.

    Returns:
    - model: The constructed model spectrum as a numpy array.
    """
    model = np.zeros(len(np.array(spectrum)))
    for key in spectrum.metadata['EDS']:
        if isinstance(spectrum.metadata['EDS'][key], dict) and key in elements_list:
            for family in spectrum.metadata['EDS'][key]:
                if '-family' in family:
                    intensity  = spectrum.metadata['EDS'][key][family].get('areal_density', 0)
                    peaks = spectrum.metadata['EDS'][key][family].get('peaks', np.zeros(len(model)))
                    if peaks.sum() <0.1:
                        print('no intensity',key, family)
                    model += peaks * intensity

    if 'detector_efficiency' in spectrum.metadata['EDS']['detector'].keys():
        detector_efficiency = spectrum.metadata['EDS']['detector']['detector_efficiency']
    else:
        detector_efficiency = None
    e_0 = spectrum.metadata['experiment']['acceleration_voltage']
    pp = spectrum.metadata['EDS']['bremsstrahlung']
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values

    if detector_efficiency is not None:
        bremsstrahlung = (pp[-3] + pp[-2] * (e_0 - energy_scale) / energy_scale +
                          pp[-1] * (e_0 - energy_scale) ** 2 / energy_scale)
        model += bremsstrahlung
        model *= detector_efficiency
    return model

def fit_model(spectrum, use_detector_efficiency=False):
    """
    Fit the EDS spectrum using a model composed of elemental peaks and bremsstrahlung background.

    Parameters:
    - spectrum: The EDS spectrum to fit.
    - elements: List of elements to consider in the fit.
    - use_detector_efficiency: Whether to include detector efficiency in the model.

    Returns:
    - peaks: The fitted peak shapes.
    - p: The fitted parameters.
    """
    peaks, pin, _ = initial_model_parameter(spectrum)

    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values

    if 'detector' in spectrum.metadata['EDS'].keys():
        start = spectrum.metadata['EDS'].get('detector', {}).get('start_channel', 120)
        spectrum.metadata['EDS']['detector']['start_channel'] = np.searchsorted(energy_scale, start)
        if use_detector_efficiency:
            efficiency = spectrum.metadata['EDS']['detector'].get('detector_efficiency', [])
            if not isinstance(efficiency, (list, np.ndarray)):
                if len(efficiency) != len(spectrum):
                    efficiency = detector_response(spectrum)
        else:
            use_detector_efficiency = False
    else:
        print('need detector information to fit spectrum')
        return None, None

    e_0 = spectrum.metadata.get('experiment', {}).get('acceleration_voltage', 0.)

    def residuals(pp, yy):
        """ residuals for fit"""
        model = np.zeros(len(yy))
        for i in range(len(pp)-4):
            model += peaks[i]*pp[i]
        if use_detector_efficiency:
            bremsstrahlung = (pp[-3] + pp[-2] * (e_0 - energy_scale) / energy_scale +
                              pp[-1] * (e_0 - energy_scale)**2 / energy_scale)
            model += bremsstrahlung
            model *= efficiency
        err = np.abs(yy - model)  # /np.sqrt(np.abs(yy[start:])+1e-12)
        return err

    y = np.array(spectrum)  # .compute()
    [p, _] = scipy.optimize.leastsq(residuals, pin, args=(y,), maxfev=10000)

    update_fit_values(spectrum.metadata['EDS'], peaks, p)
    return np.array(peaks), np.array(p)


def update_fit_values(out_tags, peaks, p):
    """
    Update the out_tags dictionary with the fitted peak shapes and parameters.

    Parameters:
    - out_tags: Dictionary containing the initial tags for each element and line family.
    - peaks: Array of fitted peak shapes.
    - p: Array of fitted parameters.
    """
    index = 0
    for lines in out_tags.values():
        if 'K-family' in lines:
            lines['K-family']['areal_density'] = p[index]
            lines['K-family']['peaks'] = peaks[index]
            index += 1
        if 'L-family' in lines:
            lines['L-family']['areal_density'] = p[index]
            lines['L-family']['peaks'] = peaks[index]
            index += 1
        if 'M-family' in lines:
            lines['M-family']['areal_density'] =p[index]
            lines['M-family']['peaks'] = peaks[index]
            index += 1
    out_tags['bremsstrahlung'] = p[-4:]


def get_phases(dataset, mode='kmeans', number_of_phases=4):
    """
    Perform phase segmentation on the dataset using the specified clustering mode.

    Parameters:
    - dataset: The dataset to be segmented.
    - mode: The clustering mode to use ('kmeans' or other).
    - number_of_phases: The number of phases (clusters) to identify.

    Returns:
    None. The results are stored in the dataset's metadata.
    """
    x_vec = np.array(dataset).reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
    x_vec = np.divide(x_vec.T, x_vec.sum(axis=1)).T
    if mode != 'kmeans':
        #choose number of components
        gmm = sklearn.mixture.GaussianMixture(n_components=number_of_phases, covariance_type="full")

        gmm_results = gmm.fit(np.array(x_vec)) #we can intelligently fold the data and perform GM
        gmm_labels = gmm_results.fit_predict(x_vec)

        dataset.metadata['gaussian_mixing_model'] = {'map': gmm_labels.reshape(dataset.shape[0],
                                                                               dataset.shape[1]),
                                                    'covariances': gmm.covariances_,
                                                    'weights': gmm.weights_,
                                                    'means':  gmm_results.means_}
    else:
        km = sklearn.cluster.KMeans(number_of_phases, n_init =10) #choose number of clusters
        km_results = km.fit(np.array(x_vec)) #we can intelligently fold the data and perform Kmeans
        dataset.metadata['kmeans'] = {'map': km_results.labels_.reshape(dataset.shape[0],
                                                                        dataset.shape[1]),
                                      'means': km_results.cluster_centers_}

def plot_phases(dataset, image=None, survey_image=None):
    """
    Plot the phase maps and corresponding spectra from the dataset.

    Parameters:
    - dataset: The dataset containing phase information.
    - image: Optional. The image to overlay the phase map on.
    - survey_image: Optional. A survey image to display alongside the phase maps.
    """
    if survey_image is not None:
        ncols = 3
    else:
        ncols = 2
    axis_index = 0
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize = (10,3))
    if survey_image is not None:
        im = axes[0].imshow(survey_image.T)
        axis_index += 1

    if 'kmeans' not in dataset.metadata:
        raise ValueError('No phase information found, run get_phases first')
    phase_spectra = dataset.metadata['kmeans']['means']
    map_data = dataset.metadata['kmeans']['map']

    cmap = plt.get_cmap('jet', len(phase_spectra))
    im = axes[axis_index].imshow(image.T,cmap='gray')
    im = axes[axis_index].imshow(map_data.T, cmap=cmap,vmin=np.min(map_data) - 0.5,
                          vmax=np.max(map_data) + 0.5,alpha=0.2)

    cbar = fig.colorbar(im, ax=axes[axis_index])
    cbar.ax.set_yticks(np.arange(0, len(phase_spectra) ))
    cbar.ax.set_ylabel("GMM Phase", fontsize = 14)
    axis_index += 1
    for index, spectrum in enumerate(phase_spectra):
        axes[axis_index].plot(dataset.energy/1000, spectrum, color = cmap(index), label=str(index))
        axes[axis_index].set_xlabel('energy (keV)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig


def plot_lines(eds_quantification: dict, axis: plt.Axes):
    """
    Plot EDS line strengths on the given matplotlib axis.

    Parameters:
    - eds_quantification: A dictionary containing EDS line data.
    - axis: A matplotlib Axes object where the lines will be plotted.
    """
    colors = plt.get_cmap('Dark2').colors # jet(np.linspace(0, 1, 10))

    index = 0
    for key, lines in eds_quantification.items():
        color = colors[index % len(colors)]
        if 'K-family' in lines:
            intensity = lines['K-family']['height']
            for line in lines['K-family']:
                if line[0] == 'K':
                    pos = lines['K-family'][line]['position']
                    axis.plot([pos,pos], [0, intensity*lines['K-family'][line]['weight']],
                              color=color)
                    if line == lines['K-family']['main']:
                        axis.text(pos,0, key+'\n'+line, verticalalignment='top', color=color)

        if 'L-family' in lines:
            intensity = lines['L-family']['height']
            for line in lines['L-family']:
                if line[0] == 'L':
                    pos = lines['L-family'][line]['position']
                    axis.plot([pos,pos], [0, intensity*lines['L-family'][line]['weight']],
                              color=color)
                    if line in [lines['L-family']['main'], 'L3-M5', 'L3-N5', 'L1-M3']:
                        axis.text(pos,0, key+'\n'+line, verticalalignment='top', color=color)

        if 'M-family' in lines:
            intensity = lines['M-family']['height']
            for line in lines['M-family']:
                if line[0] == 'M':
                    pos = lines['M-family'][line]['position']
                    axis.plot([pos,pos],
                              [0, intensity*lines['M-family'][line]['weight']],
                              color=color)
                    if line in [lines['M-family']['main'], 'M5-N7', 'M4-N6']:
                        axis.text(pos,0, key+'\n'+line, verticalalignment='top', color=color)

        index +=1
        index = index % 10


def get_eds_xsection(x_section, energy_scale, start_bgd, end_bgd):
    """
    Calculate the EDS cross-section by subtracting the background and zeroing out
    values outside the specified energy range.
    The processed cross-section data with background removed 
    and values outside the energy range set to zero.

    Parameters:
    - x_section: The raw cross-section data.
    - energy_scale: The energy scale corresponding to the cross-section data.
    - start_bgd: The start energy for background calculation.
    - end_bgd: The end energy for background calculation.

    Returns:
    - cross_section_core: np.array
    """
    background = pyTEMlib.eels_tools.power_law_background(x_section, energy_scale,
                                                          [start_bgd, end_bgd], verbose=False)
    cross_section_core = x_section- background[0]
    cross_section_core[cross_section_core < 0] = 0.0
    cross_section_core[energy_scale < end_bgd] = 0.0
    return cross_section_core


def add_k_factors(element_dict, element, k_factors):
    """Add k-factors to element dictionary."""
    family = element_dict.get('K-family', {})
    line = k_factors.get(element, {}).get('Ka1', False)
    if not line:
        line = k_factors.get(element, {}).get('Ka2', False)
    if not line:
        family = element_dict.get('L-family', {})
        line = k_factors.get(element, {}).get('La1', False)
        family['k_factor'] = float(line)
        print('using L k-factor for', element)
    if not line:
        family = element_dict.get('M-family', False)
        line = k_factors.get(element, {}).get('Ma1', False)
        if line:
            print('using M k-factor for', element)
    family['k_factor'] = float(line)


def quantify_eds(spectrum, quantification_dict=None, mask=None ):
    """Calculate quantification for EDS spectrum with either k-factors or cross sections."""

    for key in spectrum.metadata['EDS']:
        element = 0
        if isinstance(spectrum.metadata['EDS'][key], dict) and key in elements_list:
            element = spectrum.metadata['EDS'][key].get('Z', 0)
        if element < 1:
            continue
        if quantification_dict is None:
            quantification_dict = {}

        edge_info = pyTEMlib.eels_tools.get_x_sections(element)
        spectrum.metadata['EDS'][key]['atomic_weight'] = edge_info['atomic_weight']
        spectrum.metadata['EDS'][key]['nominal_density'] = edge_info['nominal_density']

        for family, item in edge_info['fluorescent_yield'].items():
            if spectrum.metadata['EDS'][key].get(f"{family}-family", {}):
                spectrum.metadata['EDS'][key][f"{family}-family"]['fluorescent_yield'] = item
        if quantification_dict.get('metadata', {}).get('type', '') == 'k_factor':
            k_factors = quantification_dict.get('table', {})
            add_k_factors(spectrum.metadata['EDS'][key], key, k_factors)
    if quantification_dict is None:
        print('using cross sections for quantification')
        quantify_cross_section(spectrum, mask=mask)
    elif not isinstance(quantification_dict, dict):
        pass
    elif quantification_dict.get('metadata', {}).get('type', '') == 'k_factor':
        print('using k-factors for quantification')
        quantification_k_factors(spectrum, mask=mask)  # , quantification_dict['table'],
    elif quantification_dict.get('metadata', {}).get('type', '') == 'cross_section':
        print('using cross sections for quantification')
        quantify_cross_section(spectrum, quantification_dict['table'], mask)
    else:
        print('using cross sections for quantification')
        quantify_cross_section(spectrum, mask=mask)


def read_esl_k_factors(filename, reduced=False):
    """ Read k-factors from esl file."""
    k_factors = {}
    if not os.path.isfile(filename):
        print('k-factor file not found', filename)
        return None, 'k_factors_Bruker_15keV.json'
    tree = xml.etree.ElementTree.parse(filename)
    root = tree.getroot()
    k_dict = pyTEMlib.file_reader.etree_to_dict(root)
    k_dict = k_dict.get('TRTStandardLibrary', {})
    k_factor_dict = (k_dict.get('ClassInstance', {}).get('CliffLorimerFactors', {}))
    for index, item in enumerate(k_factor_dict.get('K_Factors', '').split(',')):
        if index < 84:
            if item.strip() != '0':
                k_factors[elements_list[index]] = {'Ka1': float(item)}
            else:
                k_factors[elements_list[index]] = {}
    for index, item in enumerate(k_factor_dict.get('L_Factors', '').split(',')):
        if index < 84:
            if item.strip() != '0':
                k_factors[elements_list[index]]['La1'] =  float(item)
    for index, item in enumerate(k_factor_dict.get('M_Factors', '').split(',')):
        if index < 84:
            if item.strip() != '0':
                k_factors[elements_list[index]]['Ma1'] =  float(item)
    primary = int(float(k_dict.get('ClassInstance', {}).get('Header', {}).get('PrimaryEnergy', 0)))
    name = f'k_factors_Bruker_{primary}keV.json'
    metadata = {'origin': 'pyTEMlib',
                'source_file': filename,
                'reduced': reduced,
                'version': pyTEMlib.__version__,
                'type': 'k-factors',
                'spectroscopy': 'EDS',
                'acceleration_voltage': primary,
                'microscope': 'Bruker',
                'name': name}
    return k_factors, metadata


def get_absorption_correction(spectrum, thickness=50):
    """
    Calculate absorption correction for all elements in the spectrum based on thickness t in nm
    Updates the element in spectrum.metadata['EDS']['GUI'] dictionary
    Parameters:
    - spectrum: A sidpy.Dataset object containing the spectral data and metadata.
    - t: Thickness in nm
    Returns:
        None
    """
    start_channel = np.searchsorted(spectrum.energy_scale.values, 120)
    absorption = spectrum.energy_scale.values[start_channel:]*0.
    take_off_angle = spectrum.metadata['EDS']['detector'].get('ElevationAngle', 0)
    path_length = thickness *2 / np.cos(take_off_angle) * 1e-9 # /2?    in m
    count = 1
    for element, lines in spectrum.metadata['EDS']['GUI'].items():
        if element in elements_list:
            part = lines['atom%']/100
            if part > 0.01:
                count += 1
                absorption += get_absorption(pyTEMlib.utilities.get_atomic_number(element),
                                            path_length*part,
                                            spectrum.energy_scale[start_channel:])

    for element, lines in spectrum.metadata['EDS']['GUI'].items():
        symmetry =  lines['symmetry']
        peaks = []
        if symmetry in spectrum.metadata['EDS'][element]:
            peaks = spectrum.metadata['EDS'][element][symmetry].get('peaks', [])
        if len(peaks) > 0:
            peaks = peaks[start_channel:]
            lines['absorption'] = (peaks * absorption / count).sum()
            lines['thickness'] = thickness
        else:
            lines['absorption'] = 1.0
            lines['thickness'] = 0.0


def apply_absorption_correction(spectrum, thickness):
    """ 
    Apply Absorption Correction to Quantification
    Updates the element in spectrum.metadata['EDS']['GUI'] dictionary
    Parameters:
    - spectrum: A sidpy.Dataset object containing the spectral data and metadata.
    - thickness: Thickness in nm
    Returns:
        None
    """
    get_absorption_correction(spectrum, thickness)

    atom_sum = 0.
    weight_sum = 0.
    for lines in spectrum.metadata['EDS']['GUI'].values():
        atom_sum += lines.get('atom%', 0) / lines.get('absorption', 1)
        weight_sum += lines.get('weight%', 0) / lines.get('absorption', 1)
    for lines in spectrum.metadata['EDS']['GUI'].values():
        absorb = lines.get('absorption', 1)
        lines['corrected-atom%'] = lines.get('atom%', 0) / absorb / atom_sum * 100
        lines['corrected-weight%'] = lines.get('weight%', 0) / absorb / weight_sum * 100


def read_csv_k_factors(filename, reduced=True):
    """ Read k-factors from csv file of ThermoFisher TEMs."""
    k_factors = {}
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        start = True
        for row in reader:
            if start:
                k_column = row.index('K-factor')
                start = False
            else:
                element, line = row[0].split('-')
                if element not in k_factors:
                    k_factors[element] = {}
                if reduced:
                    if line[-1:] == '1':
                        k_factors[element][line] = row[k_column]
                else:
                    k_factors[element][line] = row[k_column]
    metadata = {'origin': 'pyTEMlib',
                'source_file': filename,
                'reduced': reduced,
                'microscope': 'ThermoFisher', 
                'acceleration_voltage': 200000,
                'version': pyTEMlib.__version__,
                'type': 'k-factors',
                'spectroscopy': 'EDS',
                'name': 'k_factors_Thermo_200keV.json'}
    return k_factors, metadata


def convert_k_factor_file(file_name, reduced=True, new_name=None):
    """ Convert k-factor file to a dictionary."""
    if not os.path.isfile(file_name):
        print('k-factor file not found', file_name)
        return None
    _, filename = os.path.split(file_name)
    _, extension = os.path.splitext(filename)
    if extension == '.csv':
        k_factors, metadata = read_csv_k_factors(file_name, reduced=reduced)
    elif extension == '.esl':
        k_factors, metadata = read_esl_k_factors(file_name)
    else:
        print('unknown k-factor file format', extension)
        return None
    if new_name is None:
        new_name = metadata['name']
    write_k_factors(k_factors, metadata, file_name=new_name)
    return k_factors, metadata


def get_k_factor_files():
    """ Get list of k-factor files in the .pyTEMlib folder."""
    k_factor_files = []
    for file_name in os.listdir(config_path):
        if 'k_factors' in file_name:
            k_factor_files.append(file_name)
    return k_factor_files


def write_k_factors(k_factors, metadata, file_name='k_factors.json'):
    """ Write k-factors to a json file."""
    file_name = os.path.join(config_path, file_name)
    save_dict = {"table" : k_factors, "metadata" : metadata}
    with open(file_name, "w", encoding='utf-8') as json_file:
        json.dump(save_dict, json_file, indent=4, encoding='utf-8')


def read_k_factors(file_name='k_factors.json'):
    """ Read k-factors from a json file."""
    if not os.path.isfile(os.path.join(config_path, file_name)):
        print('k-factor file not found', file_name)
        return None
    with open(os.path.join(config_path, file_name), 'r', encoding='utf-8') as json_file:
        table, metadata = json.load(json_file)
    return table, metadata


def load_k_factors(reduced=True):
    """ Load k-factors from csv files in the .pyTEMlib folder."""
    k_factors = {}
    metadata = {}
    data_path = os.path.join(os.path.expanduser('~'), '.pyTEMlib')
    for file_name in os.listdir(data_path):
        if 'k-factors' in file_name:
            path = os.path.join(data_path, file_name)
            k_factors, metadata = read_csv_k_factors(path, reduced=reduced)
            metadata['type'] = 'k_factor'
    return {'table': k_factors, 'metadata': metadata}
