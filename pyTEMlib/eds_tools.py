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

import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.interpolate  # use interp1d,
import scipy.optimize  # leastsq  # least square fitting routine fo scipy
import sklearn  # .mixture import GaussianMixture

import sidpy

import pyTEMlib

from .config_dir import config_path
elements_list = pyTEMlib.utilities.elements
shell_occupancy = pyTEMlib.utilities.shell_occupancy



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
    x_sections = pyTEMlib.eels_tools.get_x_sections()

    def get_absorption(z, t):
        photoabsorption = x_sections[str(z)]['dat']/1e10/x_sections[str(z)]['photoabs_to_sigma']
        lin = scipy.interpolate.interp1d(x_sections[str(z)]['ene'], photoabsorption, kind='linear')
        mu = lin(energy_scale) * x_sections[str(z)]['nominal_density']*100.  #1/cm -> 1/m
        return np.exp(-mu * t)

    for layer in detector_definition['detector']['layers'].values():
        if layer['Z'] != 14:
            response *= get_absorption(layer['Z'], layer['thickness'])
    if 'SiDeadThickness' in  detector_definition['detector']:
        response *= get_absorption(14, detector_definition['detector']['SiDeadThickness'])
    if 'SiLiveThickness' in  detector_definition['detector']:
        response *= 1-get_absorption(14, detector_definition['detector']['SiLiveThickness'])
    return response


def detect_peaks(dataset, minimum_number_of_peaks=30, prominence=10):
    """
    Detect peaks in the given spectral dataset.

    Parameters:
    - dataset: A sidpy.Dataset object containing the spectral data.
    - minimum_number_of_peaks: The minimum number of peaks to detect.
    - prominence: The prominence threshold for peak detection.

    Returns:
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

def find_elements(spectrum, minor_peaks):
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
    elements = set()
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
                elements.add(edge_info['name'])
                for key, line in lines.items():
                    dist = np.abs(energy_scale[peaks]-line.get('position', 0))
                    if key[0] == 'K' and np.min(dist)< 40:
                        ind = np.argmin(dist)
                        accounted_peaks.add(ind)
            # This is a special case for boron and carbon
            elif abs(lines.get('K-L2', {}).get('position', 0) - energy_scale[peak]) <30:
                accounted_peaks.add(i)
                elements.add(edge_info['name'])

            if abs(lines.get('L3-M5', {}).get('position', 0) - energy_scale[peak]) <50:
                elements.add(edge_info['name'])
                for key, line in edge_info['lines'].items():
                    dist = np.abs(energy_scale[peaks]-line.get('position', 0))
                    if key[0] == 'L' and np.min(dist)< 40 and line['weight'] > 0.01:
                        ind = np.argmin(dist)
                        accounted_peaks.add(ind)
    return list(elements)


def get_x_ray_lines(spectrum, elements):
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
    for element in elements:
        atomic_number = pyTEMlib.eds_tools.elements_list.index(element)
        out_tags[element] ={'Z': atomic_number}
        lines = pyTEMlib.xrpa_x_sections.x_sections.get(str(atomic_number), {}).get('lines', {})
        if not lines:
            break
        k_weight = 0
        k_main = 'None'
        k_lines = []
        l_weight = 0
        l_main = 'None'
        l_lines = []
        m_weight = 0
        m_main = 'None'
        m_lines  = []

        for key, line in lines.items():
            if 'K' == key[0]:
                if line['position'] < energy_scale[-1]:
                    k_lines.append(key)
                    if line['weight'] > k_weight:
                        k_weight = line['weight']
                        k_main = key
            if 'L' == key[0]:
                if line['position'] < energy_scale[-1]:
                    l_lines.append(key)
                    if line['weight'] > l_weight:
                        l_weight = line['weight']
                        l_main = key
            if 'M' == key[0]:
                if line['position'] < energy_scale[-1]:
                    m_lines .append(key)
                    if line['weight'] > m_weight:
                        m_weight = line['weight']
                        m_main = key

        if k_weight > 0:
            out_tags[element]['K-family'] = {'main': k_main, 'weight': k_weight, 'lines': k_lines}
            position = x_sections[str(atomic_number)]['lines'][k_main]['position']
            height = spectrum[np.searchsorted(energy_scale, position)].compute()
            out_tags[element]['K-family']['height'] = height/k_weight
            for key in k_lines:
                out_tags[element]['K-family'][key] = x_sections[str(atomic_number)]['lines'][key]
        if l_weight > 0:
            out_tags[element]['L-family'] = {'main': l_main, 'weight': l_weight, 'lines': l_lines}
            position = x_sections[str(atomic_number)]['lines'][l_main]['position']
            height = spectrum[np.searchsorted(energy_scale, position)].compute()
            out_tags[element]['L-family']['height'] = height/l_weight
            for key in l_lines:
                out_tags[element]['L-family'][key] = x_sections[str(atomic_number)]['lines'][key]
        if m_weight > 0:
            out_tags[element]['M-family'] = {'main': m_main, 'weight': m_weight, 'lines': m_lines }
            position = x_sections[str(atomic_number)]['lines'][m_main]['position']
            height = spectrum[np.searchsorted(energy_scale, position)].compute()
            out_tags[element]['M-family']['height'] = height/m_weight
            for key in m_lines :
                out_tags[element]['M-family'][key] = x_sections[str(atomic_number)]['lines'][key]

        xs = get_eds_cross_sections(atomic_number)
        if 'K' in xs and 'K-family' in out_tags[element]:
            out_tags[element]['K-family']['probability'] = xs['K']
        if 'L' in xs and 'L-family' in out_tags[element]:
            out_tags[element]['L-family']['probability'] = xs['L']
        if 'M' in xs and 'M-family' in out_tags[element]:
            out_tags[element]['M-family']['probability'] = xs['M']

    if 'EDS' not in spectrum.metadata:
        spectrum.metadata['EDS'] = {}
    spectrum.metadata['EDS'].update(out_tags)
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
            lines['M-family']['peaks'] = model  /model.sum()*lines['M-family']['probability']
            p.append(lines['M-family']['height'] / lines['M-family']['peaks'].max())
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
        model += detector_efficiency * bremsstrahlung

    return model

def fit_model(spectrum, elements, use_detector_efficiency=False):
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
            model *= efficiency
            bremsstrahlung = (pp[-3] + pp[-2] * (e_0 - energy_scale) / energy_scale +
                              pp[-1] * (e_0 - energy_scale)**2 / energy_scale)
            model += efficiency * bremsstrahlung
        err = np.abs(yy - model)  # /np.sqrt(np.abs(yy[start:])+1e-12)
        return err

    y = np.array(spectrum)  # .compute()
    [p, _] = scipy.optimize.leastsq(residuals, pin, args=(y,))

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


def get_eds_cross_sections(z, acceleration_voltage=200000):
    """
    Calculate the EDS cross sections for a given atomic number and acceleration voltage.

    Parameters:
    - z: Atomic number of the element.
    - acceleration_voltage: Acceleration voltage in volts (default is 200,000 V).

    Returns:
    - eds_cross_sections: Dictionary containing the calculated cross sections for various edges.
    """
    energy_scale = np.arange(1,40000)
    x_section = pyTEMlib.eels_tools.xsec_xrpa(energy_scale,
                                             acceleration_voltage/1000.,
                                             z, 400.)
    edge_info = pyTEMlib.eels_tools.get_x_sections(z)

    eds_cross_sections = {}
    x_yield = edge_info['total_fluorescent_yield']
    if 'K' in x_yield:
        start_bgd = edge_info['K1']['onset'] * 0.8
        end_bgd = edge_info['K1']['onset']  - 5
        if start_bgd > end_bgd:
            start_bgd = end_bgd-100
        if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
            eds_xsection = get_eds_xsection(x_section, energy_scale, start_bgd, end_bgd)
            eds_xsection[eds_xsection<0] = 0.
            start_sum = np.searchsorted(energy_scale, edge_info['K1']['onset'])
            end_sum = min(start_sum+600, len(x_section)-1)
            eds_cross_sections['K1'] = eds_xsection[start_sum:end_sum].sum()
            eds_cross_sections['K'] = eds_xsection[start_sum:end_sum].sum() * x_yield['K']

    if 'L3' in x_yield:
        start_bgd = edge_info['L3']['onset'] * 0.8
        end_bgd = edge_info['L3']['onset']  - 5
        if start_bgd > end_bgd:
            start_bgd = end_bgd-100
        if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
            eds_xsection = get_eds_xsection(x_section, energy_scale, start_bgd, end_bgd)
            eds_xsection[eds_xsection<0] = 0.
            start_sum = np.searchsorted(energy_scale, edge_info['L3']['onset'])
            end_sum = min(start_sum+600, len(x_section)-1)
            end_sum = min(end_sum, np.searchsorted(energy_scale, edge_info['K1']['onset']) - 10)
            eds_cross_sections['L'] = eds_xsection[start_sum:end_sum].sum()
            l1_channel =  np.searchsorted(energy_scale, edge_info['L1']['onset'])
            m_start = start_sum-100
            if m_start < 2:
                m_start = start_sum-20
            l3_rise = (np.max(x_section[m_start: l1_channel-10])-
                       np.min(x_section[m_start: l1_channel-10]))
            l1_rise = (np.max(x_section[l1_channel-10: l1_channel+100])-
                       np.min(x_section[l1_channel-10: l1_channel+100]))
            l1_ratio = l1_rise/l3_rise

            eds_cross_sections['L1'] = l1_ratio * eds_cross_sections['L']
            eds_cross_sections['L2'] = eds_cross_sections['L']*(1-l1_ratio)*1/3
            eds_cross_sections['L3'] = eds_cross_sections['L']*(1-l1_ratio)*2/3
            eds_cross_sections['yield_L1'] = x_yield['L1']
            eds_cross_sections['yield_L2'] = x_yield['L2']
            eds_cross_sections['yield_L3'] = x_yield['L3']

            eds_cross_sections['L'] = (eds_cross_sections['L1']*x_yield['L1']+
                                       eds_cross_sections['L2']*x_yield['L2']+
                                       eds_cross_sections['L3']*x_yield['L3'])
            # eds_cross_sections['L'] /= 8
    if 'M5' in x_yield:
        start_bgd = edge_info['M5']['onset'] * 0.8
        end_bgd = edge_info['M5']['onset']  - 5
        if start_bgd > end_bgd:
            start_bgd = end_bgd-100
        if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
            eds_xsection = get_eds_xsection(x_section, energy_scale, start_bgd, end_bgd)
            eds_xsection[eds_xsection<0] = 0.
            start_sum = np.searchsorted(energy_scale, edge_info['M5']['onset'])
            end_sum = start_sum + 600
            end_sum = min(end_sum, np.searchsorted(energy_scale, edge_info['L3']['onset']) - 10)
            eds_cross_sections['M'] = eds_xsection[start_sum:end_sum].sum()
            #print(edge_info['M5']['onset'] - edge_info['M1']['onset'])
            l3_channel =  np.searchsorted(energy_scale, edge_info['M3']['onset'])
            m1_channel =  np.searchsorted(energy_scale, edge_info['M1']['onset'])
            m5_rise = (np.max(x_section[start_sum-100: l3_channel-10])-
                       np.min(x_section[start_sum-100: l3_channel-10]))
            m3_rise = (np.max(x_section[l3_channel-10: m1_channel-10])-
                       np.min(x_section[l3_channel-10: m1_channel-10]))
            m1_rise = (np.max(x_section[m1_channel-10: m1_channel+100])-
                       np.min(x_section[m1_channel-10: m1_channel+100]))
            m1_ratio = m1_rise/m5_rise
            m3_ratio = m3_rise/m5_rise
            m5_ratio = 1-(m1_ratio+m3_ratio)
            #print(m1_ratio, m3_ratio, 1-(m1_ratio+m3_ratio))
            eds_cross_sections['M1'] = m1_ratio * eds_cross_sections['M']
            eds_cross_sections['M2'] = m3_ratio * eds_cross_sections['M']*1/3
            eds_cross_sections['M3'] = m3_ratio * eds_cross_sections['M']*2/3
            eds_cross_sections['M4'] = m5_ratio * eds_cross_sections['M']*2/5
            eds_cross_sections['M5'] = m5_ratio * eds_cross_sections['M']*3/5
            eds_cross_sections['yield_M1'] = x_yield['M1']
            eds_cross_sections['yield_M2'] = x_yield['M2']
            eds_cross_sections['yield_M3'] = x_yield['M3']
            eds_cross_sections['yield_M4'] = x_yield['M4']
            eds_cross_sections['yield_M5'] = x_yield['M5']
            eds_cross_sections['M'] = (eds_cross_sections['M1']*x_yield['M1']+
                                       eds_cross_sections['M2']*x_yield['M2']+
                                       eds_cross_sections['M3']*x_yield['M3']+
                                       eds_cross_sections['M4']*x_yield['M4']+
                                       eds_cross_sections['M5']*x_yield['M5'])
            #eds_cross_sections['M'] /= 18
    return eds_cross_sections


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


def get_eds_line_strength(z: int,
                          acceleration_voltage: float,
                          max_ev: int = 60000
                          ) -> dict[str, dict[str, float]]:
    """Get EDS line strength for a given element."""
    energy_scale = np.arange(10, max_ev, 1)
    edge_info = pyTEMlib.eels_tools.get_x_sections(z)
    eds_cross_sections = {'_element': {'atomic_weight': edge_info['atomic_weight'],
                                      'name': edge_info['name'],
                                      'nominal_density': edge_info['nominal_density']}}
    x_section = pyTEMlib.eels_tools.xsec_xrpa(energy_scale, acceleration_voltage, z, 1000. )
    if 'K1' in edge_info:
        start_bgd = edge_info['K1']['onset'] * 0.8
        if edge_info['K1']['onset'] - start_bgd >100:
            start_bgd = edge_info['K1']['onset'] - 100
        end_bgd = edge_info['K1']['onset'] - 5
        eds_xsection = get_eds_xsection(x_section, energy_scale, start_bgd, end_bgd)
        eds_cross_sections['K'] = {'x-section': eds_xsection[int(end_bgd) : int(end_bgd)+300].sum(),
                                   'strength':  eds_xsection[int(end_bgd) : int(end_bgd)+300].sum()}
        fluorescent_yield = edge_info.get('fluorescent_yield', {}).get('K', 0)
        if fluorescent_yield > 0:
            eds_cross_sections['K']['fluorescent_yield'] = fluorescent_yield
        eds_cross_sections['K']['strength'] *= fluorescent_yield
    if 'L3' in edge_info:
        if edge_info['L3']['onset'] > 100:
            start_bgd = edge_info['L3']['onset'] * 0.8
            if edge_info['L3']['onset'] - start_bgd >100:
                start_bgd = edge_info['L3']['onset'] - 100
            end_bgd = edge_info['L3']['onset'] - 5
            eds_xsection = get_eds_xsection(x_section, energy_scale, start_bgd, end_bgd)
            area = eds_xsection[int(end_bgd) : int(end_bgd)+300].sum()
            eds_cross_sections['L'] = {'x-section': area,
                                       'strength':  area}
            fluorescent_yield = edge_info.get('fluorescent_yield', {}).get('L', 0)
            if fluorescent_yield > 0:
                eds_cross_sections['L']['fluorescent_yield'] = fluorescent_yield
            eds_cross_sections['L']['strength'] *= fluorescent_yield
    if 'M5' in edge_info:
        if(edge_info['M5']['onset']) >100:
            start_bgd = edge_info['M5']['onset'] * 0.8
            if edge_info['M5']['onset'] - start_bgd >100:
                start_bgd = edge_info['M5']['onset'] - 100
            end_bgd = edge_info['M5']['onset'] - 5
            eds_xsection = get_eds_xsection(x_section, energy_scale, start_bgd, end_bgd)
            eds_cross_sections['M'] = {
                'x-section': eds_xsection[int(end_bgd) : int(end_bgd)+300].sum(),
                'strength':  eds_xsection[int(end_bgd) : int(end_bgd)+300].sum()
            }
            fluorescent_yield = edge_info.get('fluorescent_yield', {}).get('M', 0)
            if fluorescent_yield > 0:
                eds_cross_sections['M']['fluorescent_yield'] = fluorescent_yield
            eds_cross_sections['M']['strength'] *= fluorescent_yield
    return eds_cross_sections


def quantify_EDS(spectrum, k_factors=None, mask=None ):
    """Quantify EDS spectrum."""
    if k_factors is None:
        k_factors = {}
    acceleration_voltage = spectrum.metadata.get('experiment', {}
                                                 ).get('acceleration_voltage', 0.)
    print('quantifying EDS at', acceleration_voltage, 'V')
    for key in spectrum.metadata['EDS']:
        element = 0
        if isinstance(spectrum.metadata['EDS'][key], dict) and key in elements_list:
            element = spectrum.metadata['EDS'][key].get('Z', 0)
        if element < 1:
            continue
        tags = get_eds_line_strength(spectrum.metadata['EDS'][key]['Z'],
                                        acceleration_voltage, max_ev=60000 )
        spectrum.metadata['EDS'][key]['atomic_weight'] = tags['_element']['atomic_weight']
        spectrum.metadata['EDS'][key]['nominal_density'] = tags['_element']['nominal_density']

        line_tags = tags.get('K', False)
        family = spectrum.metadata['EDS'][key].get('K-family', False)
        line = k_factors.get(key, {}).get('Ka1', False)
        if not line:
            line = k_factors.get(key, {}).get('Ka2', False)
        if line_tags and family:
            family.update(line_tags)
        if line_tags and family and line:
            family['k_factor'] = float(line)

        line_tags = tags.get('L', False)
        family = spectrum.metadata['EDS'][key].get('L-family', False)
        line = k_factors.get(key, {}).get('La1', False)
        if line_tags and family:
            family.update(line_tags)
        if line_tags and family and line:
            family['k_factor'] = float(line)

        line_tags = tags.get('M', False)
        family = spectrum.metadata['EDS'][key].get('M-family', False)
        line = k_factors.get(key, {}).get('Ma1', False)
        if line_tags and family:
            family.update(line_tags)
        if line_tags and family and line:
            family['k_factor'] = float(line)
    quantification_k_factors(spectrum, mask=mask)


def quantification_k_factors(spectrum, mask=None):
    """Calculate quantification for EDS spectrum with k-factors."""
    tags = {}
    if not isinstance(mask, list) or mask is None:
        mask = []
    atom_sum = 0.
    weight_sum  = 0.
    for key in spectrum.metadata['EDS']:
        intensity = 0.
        k_factor = 0.
        if key in mask + ['detector', 'quantification']:
            pass
        elif isinstance(spectrum.metadata['EDS'][key], dict) and key in elements_list:
            family = spectrum.metadata['EDS'][key].get('GUI', {}).get('symmetry', None)
            if family is None:
                if 'K-family' in spectrum.metadata['EDS'][key]:
                    family = 'K-family'
                elif 'L-family' in spectrum.metadata['EDS'][key]:
                    family = 'L-family'
                elif 'M-family' in spectrum.metadata['EDS'][key]:
                    family = 'M-family'
            spectrum.metadata['EDS']['GUI'][key] = {'symmetry': family}

            intensity = spectrum.metadata['EDS'][key][family]['areal_density']
            k_factor = spectrum.metadata['EDS'][key][family]['k_factor']

            atomic_weight = spectrum.metadata['EDS'][key]['atomic_weight']
            tags[key] =  {'atom%': intensity*k_factor/ atomic_weight,
                          'weight%': (intensity*k_factor) ,
                          'k_factor': k_factor,
                          'intensity': intensity}
            atom_sum += intensity*k_factor/ atomic_weight
            weight_sum += intensity*k_factor
        tags['sums'] = {'atom%': atom_sum, 'weight%': weight_sum}

    spectrum.metadata['EDS']['quantification'] = tags
    eds_dict = spectrum.metadata['EDS']
    for key in eds_dict['quantification']:
        if key != 'sums':
            tags = eds_dict['quantification']
            out_string = f"{key:2}: {tags[key]['atom%']/tags['sums']['atom%']*100:.2f} at%"
            out_string += f" {tags[key]['weight%']/tags['sums']['weight%']*100:.2f} wt%"
            if key in eds_dict['GUI']:
                eds_dict['GUI'][key]['atom%'] = tags[key]['atom%']/tags['sums']['atom%']*100
                eds_dict['GUI'][key]['weight%'] = tags[key]['weight%']/tags['sums']['weight%']*100
    print('excluded from quantification ', mask)

import pyTEMlib.file_reader
import xml
from pyTEMlib.utilities import elements

def read_esl_k_factors(filename):
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
                k_factors[elements[index]] = {'Ka1': float(item)}
            else:
                k_factors[elements[index]] = {}
    for index, item in enumerate(k_factor_dict.get('L_Factors', '').split(',')):
        if index < 84:
            if item.strip() != '0':
                k_factors[elements[index]]['La1'] =  float(item)
    for index, item in enumerate(k_factor_dict.get('M_Factors', '').split(',')):
        if index < 84:
            if item.strip() != '0':
                k_factors[elements[index]]['Ma1'] =  float(item)
    primary = int(float(k_dict.get('ClassInstance', {}).get('Header', {}).get('PrimaryEnergy', 0)))
    name = f'k_factors_Bruker_{primary}keV.json'
    return k_factors, name


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
    return k_factors, 'k_factors_Thermo_200keV.json'


def convert_k_factor_file(file_name, reduced=True, new_name=None):
    """ Convert k-factor file to a dictionary."""
    if not os.path.isfile(file_name):
        print('k-factor file not found', file_name)
        return None
    path, filename = os.path.split(file_name)
    name, extension = os.path.splitext(filename)
    if extension == '.csv':
        k_factors, name = read_csv_k_factors(file_name, reduced=reduced)
    elif extension == '.esl':
        k_factors, name = read_esl_k_factors(file_name)
    else:
        print('unknown k-factor file format', extension)
        return None
    if new_name is None:
        new_name = name
    write_k_factors(k_factors, file_name=new_name)
    return k_factors, new_name


def get_k_factor_files():
    """ Get list of k-factor files in the .pyTEMlib folder."""
    k_factor_files = []
    for file_name in os.listdir(config_path):
        if 'k_factors' in file_name:
            k_factor_files.append(file_name)
    return k_factor_files


def write_k_factors(k_factors, file_name='k_factors.json'):
    """ Write k-factors to a json file."""
    file_name = os.path.join(config_path, file_name)
    with open(file_name, 'w', newline='', encoding='utf-8') as json_file:
        json.dump(k_factors, json_file, indent=4)


def read_k_factors(file_name='k_factors.json'):
    """ Read k-factors from a json file."""
    if not os.path.isfile(os.path.join(config_path, file_name)):
        print('k-factor file not found', file_name)
        return None
    with open(os.path.join(config_path, file_name), 'r', encoding='utf-8') as json_file:
        k_factors = json.load(json_file)
    return k_factors


def load_k_factors(reduced=True):
    """ Load k-factors from csv files in the .pyTEMlib folder."""
    k_factors = {}
    config_path = os.path.join(os.path.expanduser('~'), '.pyTEMlib')
    for file_name in os.listdir(config_path):
        if 'k-factors' in file_name:
            path = os.path.join(config_path, file_name)
            k_factors = read_csv_k_factors(path, reduced=reduced)
    return k_factors
