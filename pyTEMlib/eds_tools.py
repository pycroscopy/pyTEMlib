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

import numpy as np
import matplotlib.pyplot as plt
import os,csv

import scipy
import scipy.interpolate  # use interp1d,
import scipy.optimize  # leastsq  # least square fitting routine fo scipy
import sklearn  # .mixture import GaussianMixture

import sidpy
import pyTEMlib.eels_tools
import pyTEMlib.xrpa_x_sections

elements_list = pyTEMlib.eels_tools.elements

shell_occupancy = {'K1': 2, 'L1': 2, 'L2': 2, 'L3': 4, 'M1': 2, 'M2': 2, 'M3': 4, 'M4': 4, 'M5': 6,
                   'N1': 2, 'N2': 2, 'N3': 4, 'N4': 4, 'N5': 6, 'N6': 6, 'N7': 8,
                   'O1': 2, 'O2': 2, 'O3': 4, 'O4': 4, 'O5': 6, 'O6': 6, 'O7': 8, 'O8': 8, 'O9': 10}


def detector_response(dataset):
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
    Calculates response of Si drift detector for EDS spectrum background based on detector parameters

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
    background[start:] = detector_Efficiency * (p[0] + p[1]*(E_0-energy_scale)/energy_scale + p[2]*(E_0-energy_scale)**2/energy_scale)


    plt.figure()
    plt.plot(spectrum.energy, spectrum, label = 'spec')
    plt.plot(spectrum.energy, background, label = 'background')
    plt.show()

    """
    response = np.ones(len(energy_scale))
    x_sections = pyTEMlib.eels_tools.get_x_sections()
    
    def get_absorption(Z, t):
        photoabsorption = x_sections[str(Z)]['dat']/1e10/x_sections[str(Z)]['photoabs_to_sigma']
        lin = scipy.interpolate.interp1d(x_sections[str(Z)]['ene'], photoabsorption, kind='linear')
        mu = lin(energy_scale) * x_sections[str(Z)]['nominal_density']*100.  #1/cm -> 1/m
        return np.exp(-mu * t)
    
    for key, layer in detector_definition['detector']['layers'].items():
        if layer['Z'] != 14:
            response *= get_absorption(layer['Z'], layer['thickness'])
    if 'SiDeadThickness' in  detector_definition['detector']:
        response *= get_absorption(14, detector_definition['detector']['SiDeadThickness'])
    if 'SiLiveThickness' in  detector_definition['detector']:
        response *= 1-get_absorption(14, detector_definition['detector']['SiLiveThickness'])
    return response


def detect_peaks(dataset, minimum_number_of_peaks=30, prominence=10):
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('Needs an sidpy dataset')
    if not dataset.data_type.name == 'SPECTRUM':
        raise TypeError('Need a spectrum')

    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values
    if 'EDS' not in dataset.metadata:
        dataset.metadata['EDS'] = {}
    if 'detector' not in dataset.metadata['EDS']:
        if 'energy_resolution' not in dataset.metadata['EDS']['detector']:
            dataset.metadata['EDS']['detector']['energy_resolution'] = 138
            print('Using energy resolution of 138 eV')
        if 'start_channel' not in dataset.metadata['EDS']['detector']:
            dataset.metadata['EDS']['detector']['start_channel'] =  np.searchsorted(energy_scale, 100)
    resolution = dataset.metadata['EDS']['detector']['energy_resolution']

    start = dataset.metadata['EDS']['detector']['start_channel']
    ## we use half the width of the resolution for smearing
    width = int(np.ceil(resolution/(energy_scale[1]-energy_scale[0])/2)+1)
    new_spectrum =  scipy.signal.savgol_filter(np.array(dataset)[start:], width, 2) ## we use half the width of the resolution for smearing

    minor_peaks, _  = scipy.signal.find_peaks(new_spectrum, prominence=prominence)
    
    while len(minor_peaks) > minimum_number_of_peaks:
        prominence+=10
        minor_peaks, _  = scipy.signal.find_peaks(new_spectrum, prominence=prominence)
    return np.array(minor_peaks)+start

def find_elements(spectrum, minor_peaks):
    if not isinstance(spectrum, sidpy.Dataset):
        raise TypeError(' Need a sidpy dataset')
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0]
    elements = []
    peaks = minor_peaks[np.argsort(spectrum[minor_peaks])]
    accounted_peaks = []
    for i, peak in reversed(list(enumerate(peaks))):
        for z in range(5, 82):
            edge_info  = pyTEMlib.eels_tools.get_x_sections(z)
            element = edge_info['name']
            if 'lines' not in edge_info:
                pass
            elif 'K-L3' in edge_info['lines']: 
               if abs(edge_info['lines']['K-L3']['position']- energy_scale.values[peak]) <40:
                    if i not in accounted_peaks:
                        accounted_peaks.append(i)
                        if edge_info['name'] not in elements:
                            elements.append(edge_info['name'])
                    for line in edge_info['lines'].keys():
                        if line[0] == 'K':
                            if np.min(np.abs(energy_scale.values[peaks]-edge_info['lines'][line]['position']))< 50:
                                ind = np.argmin(np.abs(energy_scale.values[peaks]-edge_info['lines'][line]['position']))
                                if ind not in accounted_peaks:
                                    accounted_peaks.append(ind)
            elif 'K-L2' in edge_info['lines']:
                if abs(edge_info['lines']['K-L2']['position']- energy_scale.values[peak]) <30:
                    found = True
                    accounted_peaks .append(i)
                    if  edge_info['name'] not in elements:
                        elements.append(edge_info['name'])
            
            if 'L3-M5' in edge_info['lines']:            
                if abs(edge_info['lines']['L3-M5']['position']- energy_scale.values[peak]) <40:
                    if edge_info['name'] not in elements:
                        if i not in accounted_peaks:
                            accounted_peaks.append(i)
                            elements.append(edge_info['name'])
                        for line in edge_info['lines'].keys():
                            if line[0] == 'L': 
                                    #if edge_info['lines'][line]['weight'] > 0.01:
                                    if np.min(np.abs(energy_scale.values[peaks]-edge_info['lines'][line]['position']))< 50:
                                        ind = np.argmin(np.abs(energy_scale.values[peaks]-edge_info['lines'][line]['position']))
                                        if ind not in accounted_peaks:
                                            accounted_peaks.append(ind)
    return elements
                                        

def get_x_ray_lines(spectrum, elements):
    out_tags = {}
    alpha_K = 1e6
    alpha_L = 6.5e7
    alpha_M = 8*1e8  # 2.2e10
    # My Fit
    alpha_K = .9e6
    alpha_L = 6.e7
    alpha_M = 6*1e8 #  2.2e10
    # omega_K = Z**4/(alpha_K+Z**4)
    # omega_L = Z**4/(alpha_L+Z**4)
    # omega_M = Z**4/(alpha_M+Z**4)
    x_sections = pyTEMlib.xrpa_x_sections.x_sections
    energy_scale = np.array(spectrum.get_spectral_dims(return_axis=True)[0].values)
    for element in elements:
        atomic_number = pyTEMlib.eds_tools.elements_list.index(element)
        out_tags[element] ={'Z': atomic_number}
        lines = pyTEMlib.xrpa_x_sections.x_sections[str(atomic_number)]['lines']
        K_weight = 0
        K_main = 'None'
        K_lines = []
        L_weight = 0
        L_main = 'None'
        L_lines = []
        M_weight = 0
        M_main = 'None'
        M_lines = []
        
        for key, line in lines.items():
            if 'K' == key[0]:
                if line['position'] < energy_scale[-1]:
                        K_lines.append(key)
                        if line['weight'] > K_weight:
                            K_weight = line['weight']
                            K_main = key
            if 'L' == key[0]:
                if line['position'] < energy_scale[-1]:
                        L_lines.append(key)
                        if line['weight'] > L_weight:
                            L_weight = line['weight']
                            L_main = key
            if 'M' == key[0]:
                if line['position'] < energy_scale[-1]:
                        M_lines.append(key)
                        if line['weight'] > M_weight:
                            M_weight = line['weight']
                            M_main = key
            
        if K_weight > 0:
            out_tags[element]['K-family'] = {'main': K_main, 'weight': K_weight, 'lines': K_lines}
            height = spectrum[np.searchsorted(energy_scale, x_sections[str(atomic_number)]['lines'][K_main]['position'] )].compute()
            out_tags[element]['K-family']['height'] = height/K_weight
            for key in K_lines:
                out_tags[element]['K-family'][key] = pyTEMlib.xrpa_x_sections.x_sections[str(atomic_number)]['lines'][key]
        if L_weight > 0:     
            out_tags[element]['L-family'] = {'main': L_main, 'weight': L_weight, 'lines': L_lines}
            height = spectrum[np.searchsorted(energy_scale, x_sections[str(atomic_number)]['lines'][L_main]['position'] )].compute()
            out_tags[element]['L-family']['height'] = height/L_weight
            for key in L_lines:
                out_tags[element]['L-family'][key] = x_sections[str(atomic_number)]['lines'][key]
        if M_weight > 0:
            out_tags[element]['M-family'] = {'main': M_main, 'weight': M_weight, 'lines': M_lines}
            height = spectrum[np.searchsorted(energy_scale, x_sections[str(atomic_number)]['lines'][M_main]['position'] )].compute()
            out_tags[element]['M-family']['height'] = height/M_weight
            for key in M_lines:
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


def getFWHM(E, E_ref, FWHM_ref):
    return np.sqrt(2.5*(E-E_ref)+FWHM_ref**2)

def gaussian(energy_scale, mu, FWHM):
    sig = FWHM/2/np.sqrt(2*np.log(2))
    return np.exp(-np.power(np.array(energy_scale) - mu, 2.) / (2 * np.power(sig, 2.)))

def get_peak(E, energy_scale):
    E_ref = 5895.0
    FWHM_ref = 136 #eV
    FWHM  = getFWHM(E, E_ref, FWHM_ref)
    gauss = gaussian(energy_scale, E, FWHM)

    return gauss /(gauss.sum()+1e-12)


def initial_model_parameter(spectrum):
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
    model = np.zeros(len(np.array(spectrum)))
    for key in spectrum.metadata['EDS']:
        if isinstance(spectrum.metadata['EDS'][key], dict):
            for family in spectrum.metadata['EDS'][key]:
                if '-family' in family:
                    intensity  = spectrum.metadata['EDS'][key][family]['areal_density']
                    peaks = spectrum.metadata['EDS'][key][family]['peaks']
                    model += peaks * intensity

    if 'detector_efficiency' in spectrum.metadata['EDS']['detector'].keys():
        detector_efficiency = spectrum.metadata['EDS']['detector']['detector_efficiency']
    else:
        detector_efficiency = None
    E_0 = spectrum.metadata['experiment']['acceleration_voltage']
    pp = spectrum.metadata['EDS']['bremsstrahlung']
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values

    if detector_efficiency is not None:
        # bremsstrahlung = pp[-4] / (energy_scale + pp[-3] * energy_scale**2 + pp[-2] * energy_scale**.5) - pp[-1]
        bremsstrahlung = pp[-3] + pp[-2] * (E_0 - energy_scale) / energy_scale + pp[-1] * (E_0 - energy_scale) ** 2 / energy_scale
        model += detector_efficiency * bremsstrahlung
   
    return model

def fit_model(spectrum, elements, use_detector_efficiency=False):
    out_tags = get_x_ray_lines(spectrum, elements)
    peaks, pin, keys = initial_model_parameter(spectrum)

    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values
    
    if 'detector' in spectrum.metadata['EDS'].keys():
        if 'start_channel' not in spectrum.metadata['EDS']['detector']:
            spectrum.metadata['EDS']['detector']['start_channel'] = np.searchsorted(energy_scale, 120)
        if 'detector_efficiency' in spectrum.metadata['EDS']['detector'].keys():
            if use_detector_efficiency:
                detector_efficiency = spectrum.metadata['EDS']['detector']['detector_efficiency']
        else:
            use_detector_efficiency = False
    else:
        print('need detector information to fit spectrum')
        return
    start = 0 #.spectrum.metadata['EDS']['detector']['start_channel']
    # energy_scale = energy_scale[start:]

    E_0= spectrum.metadata['experiment']['acceleration_voltage']

    def residuals(pp, yy):
        #get_model(peaks, pp, detector_efficiency=None)
        model = np.zeros(len(yy))
        for i in range(len(pp)-4):
            model += peaks[i]*pp[i]
        # pp[-3:] = np.abs(pp[-3:])

        if use_detector_efficiency:
            model *= detector_efficiency
            # bremsstrahlung = pp[-4] / (energy_scale + pp[-3] * energy_scale**2 + pp[-2] * energy_scale**.5) - pp[-1]
            bremsstrahlung = pp[-3] + pp[-2] * (E_0 - energy_scale) / energy_scale + pp[-1] * (E_0 - energy_scale) ** 2 / energy_scale
            model += detector_efficiency * bremsstrahlung
            #(pp[-3] + pp[-2] * (E_0 - energy_scale) / energy_scale +
            #                                                pp[-1] * (E_0-energy_scale) ** 2 / energy_scale))

        err = np.abs((yy - model))  # /np.sqrt(np.abs(yy[start:])+1e-12)

        return err

    y = np.array(spectrum)  # .compute()
    [p, _] = scipy.optimize.leastsq(residuals, pin, args=(y))

    # print(pin[-6:], p[-6:])

    update_fit_values(out_tags, peaks, p)


    if 'EDS' not in spectrum.metadata:
        spectrum.metadata['EDS'] = {}
    spectrum.metadata['EDS'].update(out_tags)

    return np.array(peaks), np.array(p)


def update_fit_values(out_tags, peaks, p):
    index = 0
    for element, lines in out_tags.items():
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
    energy_scale = np.arange(1,20000)
    Xsection = pyTEMlib.eels_tools.xsec_xrpa(energy_scale, acceleration_voltage/1000., z, 400.)
    edge_info = pyTEMlib.eels_tools.get_x_sections(z)

    
    eds_cross_sections = {}
    Xyield = edge_info['total_fluorescent_yield']
    if 'K' in Xyield:
            start_bgd = edge_info['K1']['onset'] * 0.8
            end_bgd = edge_info['K1']['onset']  - 5
            if start_bgd > end_bgd:
                start_bgd = end_bgd-100
            if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
                eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
                eds_xsection[eds_xsection<0] = 0.
                start_sum = np.searchsorted(energy_scale, edge_info['K1']['onset'])
                end_sum = start_sum+600
                if end_sum> len(Xsection):
                    end_sum = len(Xsection)-1
                eds_cross_sections['K1'] = eds_xsection[start_sum:end_sum].sum() 
                eds_cross_sections['K'] = eds_xsection[start_sum:end_sum].sum() * Xyield['K']
    
    if 'L3' in Xyield:
            start_bgd = edge_info['L3']['onset'] * 0.8
            end_bgd = edge_info['L3']['onset']  - 5
            if start_bgd > end_bgd:
                start_bgd = end_bgd-100
            if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
                eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
                eds_xsection[eds_xsection<0] = 0.
                start_sum = np.searchsorted(energy_scale, edge_info['L3']['onset'])
                end_sum = start_sum+600
                if end_sum> len(Xsection):
                    end_sum = len(Xsection)-1
                if end_sum >np.searchsorted(energy_scale, edge_info['K1']['onset'])-10:
                    end_sum = np.searchsorted(energy_scale, edge_info['K1']['onset'])-10
                eds_cross_sections['L'] = eds_xsection[start_sum:end_sum].sum() 
                L1_channel =  np.searchsorted(energy_scale, edge_info['L1']['onset'])
                m_start = start_sum-100
                if m_start < 2:
                    m_start = start_sum-20
                l3_rise = np.max(Xsection[m_start: L1_channel-10])-np.min(Xsection[m_start: L1_channel-10])
                l1_rise = np.max(Xsection[L1_channel-10: L1_channel+100])-np.min(Xsection[L1_channel-10: L1_channel+100])
                l1_ratio = l1_rise/l3_rise
                
                eds_cross_sections['L1'] = l1_ratio * eds_cross_sections['L']
                eds_cross_sections['L2'] = eds_cross_sections['L']*(1-l1_ratio)*1/3
                eds_cross_sections['L3'] = eds_cross_sections['L']*(1-l1_ratio)*2/3
                eds_cross_sections['yield_L1'] = Xyield['L1']
                eds_cross_sections['yield_L2'] = Xyield['L2']
                eds_cross_sections['yield_L3'] = Xyield['L3']

                eds_cross_sections['L'] = eds_cross_sections['L1']*Xyield['L1']+eds_cross_sections['L2']*Xyield['L2']+eds_cross_sections['L3']*Xyield['L3']
                # eds_cross_sections['L'] /= 8
    if 'M5' in Xyield:
            start_bgd = edge_info['M5']['onset'] * 0.8
            end_bgd = edge_info['M5']['onset']  - 5
            if start_bgd > end_bgd:
                start_bgd = end_bgd-100
            if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
                eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
                eds_xsection[eds_xsection<0] = 0.
                start_sum = np.searchsorted(energy_scale, edge_info['M5']['onset'])
                end_sum = start_sum+600
                if end_sum > np.searchsorted(energy_scale, edge_info['L3']['onset'])-10:
                    end_sum = np.searchsorted(energy_scale, edge_info['L3']['onset'])-10
                eds_cross_sections['M'] = eds_xsection[start_sum:end_sum].sum() 
                #print(edge_info['M5']['onset'] - edge_info['M1']['onset'])
                M3_channel =  np.searchsorted(energy_scale, edge_info['M3']['onset'])
                M1_channel =  np.searchsorted(energy_scale, edge_info['M1']['onset'])
                m5_rise = np.max(Xsection[start_sum-100: M3_channel-10])-np.min(Xsection[start_sum-100: M3_channel-10])
                m3_rise = np.max(Xsection[M3_channel-10: M1_channel-10])-np.min(Xsection[M3_channel-10: M1_channel-10])
                m1_rise = np.max(Xsection[M1_channel-10: M1_channel+100])-np.min(Xsection[M1_channel-10: M1_channel+100])
                m1_ratio = m1_rise/m5_rise
                m3_ratio = m3_rise/m5_rise
                m5_ratio = 1-(m1_ratio+m3_ratio)
                #print(m1_ratio, m3_ratio, 1-(m1_ratio+m3_ratio))
                eds_cross_sections['M1'] = m1_ratio * eds_cross_sections['M']
                eds_cross_sections['M2'] = m3_ratio * eds_cross_sections['M']*1/3
                eds_cross_sections['M3'] = m3_ratio * eds_cross_sections['M']*2/3
                eds_cross_sections['M4'] = m5_ratio * eds_cross_sections['M']*2/5
                eds_cross_sections['M5'] = m5_ratio * eds_cross_sections['M']*3/5
                eds_cross_sections['yield_M1'] = Xyield['M1']
                eds_cross_sections['yield_M2'] = Xyield['M2']
                eds_cross_sections['yield_M3'] = Xyield['M3']
                eds_cross_sections['yield_M4'] = Xyield['M4']
                eds_cross_sections['yield_M5'] = Xyield['M5']
                eds_cross_sections['M'] = eds_cross_sections['M1']*Xyield['M1']+eds_cross_sections['M2']*Xyield['M2']+eds_cross_sections['M3']*Xyield['M3'] \
                                            +eds_cross_sections['M4']*Xyield['M4']+eds_cross_sections['M5']*Xyield['M5']
                #eds_cross_sections['M'] /= 18
    return eds_cross_sections


def get_phases(dataset, mode='kmeans', number_of_phases=4):
    X_vec = np.array(dataset).reshape(dataset.shape[0]*dataset.shape[1], dataset.shape[2])
    X_vec = np.divide(X_vec.T, X_vec.sum(axis=1)).T
    if mode != 'kmeans':
        gmm = sklearn.mixture.GaussianMixture(n_components=number_of_phases, covariance_type="full") #choose number of components

        gmm_results = gmm.fit(np.array(X_vec)) #we can intelligently fold the data and perform GM
        gmm_labels = gmm_results.fit_predict(X_vec)

        dataset.metadata['gaussian_mixing_model'] = {'map': gmm_labels.reshape(dataset.shape[0], dataset.shape[1]),
                                                    'covariances': gmm.covariances_,
                                                    'weights': gmm.weights_,
                                                    'means':  gmm_results.means_}
    else:
        km = sklearn.cluster.KMeans(number_of_phases, n_init =10) #choose number of clusters
        km_results = km.fit(np.array(X_vec)) #we can intelligently fold the data and perform Kmeans
        dataset.metadata['kmeans'] = {'map': km_results.labels_.reshape(dataset.shape[0], dataset.shape[1]),
                                      'means': km_results.cluster_centers_}

def plot_phases(dataset, image=None, survey_image=None):
    if survey_image is not None:
        ncols = 3
    else:
        ncols = 2
    axis_index = 0
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize = (10,3))
    if survey_image is not None:
        im = axes[0].imshow(survey_image.T)
        axis_index += 1
    #if 'gaussian_mixing_model' in dataset.metadata:
    #    phase_spectra = dataset.metadata['gaussian_mixing_model']['means']
    #   map = dataset.metadata['gaussian_mixing_model']['map']
    #
    if 'kmeans' in dataset.metadata:
        phase_spectra = dataset.metadata['kmeans']['means']
        map = dataset.metadata['kmeans']['map']

    cmap = plt.get_cmap('jet', len(phase_spectra))
    im = axes[axis_index].imshow(image.T,cmap='gray')
    im = axes[axis_index].imshow(map.T, cmap=cmap,vmin=np.min(map) - 0.5,
                          vmax=np.max(map) + 0.5,alpha=0.2)
    
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

    colors = plt.get_cmap('Dark2').colors # jet(np.linspace(0, 1, 10))
    index = 0
    for key, lines in eds_quantification.items():
        color = colors[index]
        if 'K-family' in lines:
            intensity = lines['K-family']['height']
            for line in lines['K-family']:
                if line[0] == 'K':
                    pos = lines['K-family'][line]['position']
                    axis.plot([pos,pos], [0, intensity*lines['K-family'][line]['weight']], color=color)
                    if line == lines['K-family']['main']:      
                        axis.text(pos,0, key+'\n'+line, verticalalignment='top', color=color)

        if 'L-family' in lines:
            intensity = lines['L-family']['height']
            for line in lines['L-family']:
                if line[0] == 'L':
                    pos = lines['L-family'][line]['position']
                    axis.plot([pos,pos], [0, intensity*lines['L-family'][line]['weight']], color=color)
                    if line in [lines['L-family']['main'], 'L3-M5', 'L3-N5', 'L1-M3']:            
                        axis.text(pos,0, key+'\n'+line, verticalalignment='top', color=color)

        if 'M-family' in lines:
            intensity = lines['M-family']['height']
            for line in lines['M-family']:
                if line[0] == 'M':
                    pos = lines['M-family'][line]['position']
                    axis.plot([pos,pos], [0, intensity*lines['M-family'][line]['weight']], color=color)
                    if line in [lines['M-family']['main'], 'M5-N7', 'M4-N6']:      
                        axis.text(pos,0, key+'\n'+line, verticalalignment='top', color=color)

        index +=1
        index = index % 10
def get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd):
    background = pyTEMlib.eels_tools.power_law_background(Xsection, energy_scale, [start_bgd, end_bgd], verbose=False)
    cross_section_core = Xsection- background[0]
    cross_section_core[cross_section_core < 0] = 0.0
    cross_section_core[energy_scale < end_bgd] = 0.0
    return cross_section_core


def get_eds_line_strength(z, acceleration_voltage, max_kV=60000 ):

    keV = acceleration_voltage /1000.
    energy_scale = np.arange(10, max_kV, 1)
    edge_info = pyTEMlib.eels_tools.get_x_sections(z)
    eds_cross_sections = {'_element': {'atomic_weight': edge_info['atomic_weight'],
                                      'name': edge_info['name'],
                                      'nominal_density': edge_info['nominal_density']}}
    Xsection = pyTEMlib.eels_tools.xsec_xrpa(energy_scale, keV, z, 3000. )
    if 'K1' in edge_info:
        start_bgd = edge_info['K1']['onset'] * 0.8
        if edge_info['K1']['onset'] - start_bgd >100:
            start_bgd = edge_info['K1']['onset'] - 100
        end_bgd = edge_info['K1']['onset'] - 5
        K_eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
        eds_cross_sections['K'] = {'x-section': K_eds_xsection[int(end_bgd) : int(end_bgd)+300].sum(),
                                   'strength':  K_eds_xsection[int(end_bgd) : int(end_bgd)+300].sum()}
        if 'fluorescent_yield' in edge_info:
             eds_cross_sections['K']['fluorescent_yield'] = edge_info['fluorescent_yield']['K']
             eds_cross_sections['K']['strength'] *= edge_info['fluorescent_yield']['K']    
    if 'L3' in edge_info:
        if edge_info['L3']['onset'] > 100:           
            start_bgd = edge_info['L3']['onset'] * 0.8
            if edge_info['L3']['onset'] - start_bgd >100:
                start_bgd = edge_info['L3']['onset'] - 100
            end_bgd = edge_info['L3']['onset'] - 5
            L_eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
            eds_cross_sections['L'] = {'x-section': L_eds_xsection[int(end_bgd) : int(end_bgd)+300].sum(),
                                       'strength':  L_eds_xsection[int(end_bgd) : int(end_bgd)+300].sum()}
            if 'fluorescent_yield' in edge_info:
                if 'L' in edge_info['fluorescent_yield']:
                    eds_cross_sections['L']['fluorescent_yield'] = edge_info['fluorescent_yield']['L']
                    eds_cross_sections['L']['strength'] *= edge_info['fluorescent_yield']['L'] 
                else:
                    eds_cross_sections['L']['strength'] *= 0.
                    
    if 'M5' in edge_info:
        if(edge_info['M5']['onset']) >100:
            start_bgd = edge_info['M5']['onset'] * 0.8
            if edge_info['M5']['onset'] - start_bgd >100:
                start_bgd = edge_info['M5']['onset'] - 100
            
            end_bgd = edge_info['M5']['onset'] - 5
            M_eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
            eds_cross_sections['M'] = {'x-section': M_eds_xsection[int(end_bgd) : int(end_bgd)+300].sum(),
                                       'strength':  M_eds_xsection[int(end_bgd) : int(end_bgd)+300].sum()}
            if 'fluorescent_yield' in edge_info:
                if 'M' in edge_info['fluorescent_yield']:
                    eds_cross_sections['M']['fluorescent_yield'] = edge_info['fluorescent_yield']['M']
                    eds_cross_sections['M']['strength'] *= edge_info['fluorescent_yield']['M']
                else:
                    eds_cross_sections['M']['strength'] *= 0.
    return eds_cross_sections

get_eds_line_strength(14, 200000)

def quantify_EDS(spectrum, k_factors=None, mask=[] ):
    if k_factors is None:
        k_factors = {}
    acceleration_voltage = spectrum.metadata['experiment']['acceleration_voltage'] 
    for key in spectrum.metadata['EDS']:
        if isinstance(spectrum.metadata['EDS'][key], dict):
            if 'Z' in spectrum.metadata['EDS'][key]:
                tags = get_eds_line_strength(spectrum.metadata['EDS'][key]['Z'], acceleration_voltage, max_kV=60000 )
                spectrum.metadata['EDS'][key]['atomic_weight'] = tags['_element']['atomic_weight']
                spectrum.metadata['EDS'][key]['nominal_density'] = tags['_element']['nominal_density']
                if 'K' in tags.keys():
                    if 'K-family' in spectrum.metadata['EDS'][key]:
                        spectrum.metadata['EDS'][key]['K-family'].update(tags['K'])
                        if key in k_factors:
                            if 'Ka1' in k_factors[key].keys():
                                spectrum.metadata['EDS'][key]['K-family']['k_factor']= float(k_factors[key]['Ka1'])
                        #print(key, 'K ', tags['K'], spectrum.metadata['EDS'][key]['K-family'])
                if 'L' in tags.keys():
                    if 'L-family' in spectrum.metadata['EDS'][key]:
                        spectrum.metadata['EDS'][key]['L-family'].update(tags['L'])
                        if key in k_factors:
                            if 'La1' in k_factors[key].keys():
                                spectrum.metadata['EDS'][key]['K-family']['k_factor']= float(k_factors[key]['La1'])
                        #print(key, 'L ', tags['L'], spectrum.metadata['EDS'][key]['L-family']['areal_density'])
                if 'M' in tags.keys():
                    if 'M-family' in spectrum.metadata['EDS'][key]:
                        spectrum.metadata['EDS'][key]['M-family'].update(tags['M'])
                        if key in k_factors:
                            if 'Ma1' in k_factors[key].keys():
                                spectrum.metadata['EDS'][key]['K-family']['k_factor'] = float(k_factors[key]['Ma1'])
                        print(key, 'M ', tags['L'], spectrum.metadata['EDS'][key]['M-family']['areal_density'])

    quantification_k_factors(spectrum, mask=mask)


def quantification_k_factors(spectrum, mask=[]):
    tags = {}
    atom_sum = 0.
    weight_sum  = 0.
    for key in spectrum.metadata['EDS']:
        intensity = 0.
        k_factor = 0.
        if key in mask + ['detector', 'quantification']:
            pass
        elif isinstance(spectrum.metadata['EDS'][key], dict):
            if 'K-family' in spectrum.metadata['EDS'][key]:
                intensity = spectrum.metadata['EDS'][key]['K-family']['areal_density']
                k_factor = spectrum.metadata['EDS'][key]['K-family']['k_factor']
            elif 'L-family' in spectrum.metadata['EDS'][key]:
                intensity = spectrum.metadata['EDS'][key]['L-family']['areal_density']
                k_factor = spectrum.metadata['EDS'][key]['L-family']['k_factor']
        
            tags[key] =  {'atom%': intensity / k_factor, 'weight%': intensity / k_factor* spectrum.metadata['EDS'][key]['atomic_weight']}
            atom_sum += intensity / k_factor 
            weight_sum += intensity / k_factor * spectrum.metadata['EDS'][key]['atomic_weight']
        tags['sums'] = {'atom%': atom_sum, 'weight%': weight_sum}
    spectrum.metadata['EDS']['quantification'] = tags

    for key in spectrum.metadata['EDS']['quantification']:
        if key != 'sums':
            tags = spectrum.metadata['EDS']['quantification']
            print(f"{key:2}: {tags[key]['atom%']/tags['sums']['atom%']*100:.2f} at%  {tags[key]['weight%']/tags['sums']['weight%']*100:.2f} wt%" )
    print('excluded from quantification ', mask)


def load_k_factors(reduced=True):

    k_factors = {}
    config_path = os.path.join(os.path.expanduser('~'), '.pyTEMlib')
    for file_name in os.listdir(config_path):
        if 'k-factors' in file_name:
            with open(os.path.join(config_path, file_name), newline='') as csvfile:
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
    return k_factors