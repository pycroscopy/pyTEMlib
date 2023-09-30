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

import scipy
from scipy.interpolate import interp1d, splrep  # splev, splint
from scipy import interpolate
from scipy.signal import peak_prominences
from scipy.ndimage import gaussian_filter

import scipy.constants as const

from scipy import constants
import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# from matplotlib.widgets import SpanSelector
# import ipywidgets as widgets
# from IPython.display import display

import requests

from scipy.optimize import leastsq  # least square fitting routine fo scipy

import sidpy

import pickle  # pkg_resources
import pyTEMlib.eels_tools as eels
from pyTEMlib.xrpa_x_sections import x_sections

elements_list = eels.elements

shell_occupancy={'K1':2, 'L1':2, 'L2':2, 'L3':4, 'M1':2, 'M2':2, 'M3':4,'M4':4,'M5':6, 
                 'N1':2, 'N2':2,' N3':4,'N4':4,'N5':6, 'N6':6,'N7':8,
                 'O1':2, 'O2':2,' O3':4,'O4':4,'O5':6, 'O6':6,'O7':8, 'O8':8, 'O9': 10 }

def detector_response(detector_definition, energy_scale):
    """
    Parameters
    ----------


    Example
    -------

    tags = {}

    tags['acceleration_voltage_V'] = 30000

    tags['detector'] ={}
    tags['detector']['layers'] ={}

    ## layer thicknesses of commen materials in EDS detectors in m
    tags['detector']['layers']['alLayer'] = {}
    tags['detector']['layers']['alLayer']['thickness'] = 30 *1e-9    # in m
    tags['detector']['layers']['alLayer']['Z'] = 13

    tags['detector']['layers']['deadLayer'] = {}
    tags['detector']['layers']['deadLayer']['thickness'] =  100 *1e-9  # in m
    tags['detector']['layers']['deadLayer']['Z'] = 14

    tags['detector']['layers']['window'] = {}
    tags['detector']['layers']['window']['thickness'] =  100 *1e-9  # in m
    tags['detector']['layers']['window']['Z'] = 6

    tags['detector']['detector'] = {}
    tags['detector']['detector']['thickness'] = 45 * 1e-3  # in m
    tags['detector']['detector']['Z'] = 14
    tags['detector']['detector']['area'] = 30 * 1e-6 #in m2
    
    energy_scale = np.linspace(.1,60,1199)*1000 i eV
    detector_response(tags, energy_scale)
    """
    response = np.ones(len(energy_scale))
    x_sections = eels.get_x_sections()
    
    for key in detector_definition['layers']:
        Z = detector_definition['layers'][key]['Z']
        t = detector_definition['layers'][key]['thickness']
        photoabsorption = x_sections[str(Z)]['dat']/1e10/x_sections[str(Z)]['photoabs_to_sigma']
        lin = interp1d(x_sections[str(Z)]['ene'], photoabsorption,kind='linear') 
        mu = lin(energy_scale) * x_sections[str(Z)]['nominal_density']*100. #1/cm -> 1/m

        absorption = np.exp(-mu * t)
        response = response*absorption
    Z = detector_definition['detector']['Z']
    t = detector_definition['detector']['thickness']    
    photoabsorption = x_sections[str(Z)]['dat']/1e10/x_sections[str(Z)]['photoabs_to_sigma']
    lin = interp1d(x_sections[str(Z)]['ene']/1000., photoabsorption,kind='linear') 
    mu = lin(energy_scale) * x_sections[str(Z)]['nominal_density']*100. #1/cm -> 1/m
    response = response*(1.0 - np.exp(-mu * t))# * oo4pi;
    return(response)


def detect_peaks(dataset, minimum_number_of_peaks=30):
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('Needs an sidpy dataset')
    if not dataset.data_type.name == 'SPECTRUM':
        raise TypeError('Need a spectrum')
    resolution = 138
    if 'EDS' in dataset.metadata:
        if 'energy_resolution' in dataset.metadata['EDS']:
            resolution = dataset.metadata['EDS']['energy_resolution']
    start = np.searchsorted(dataset.energy_scale, 125)
    ## we use half the width of the resolution for smearing
    width = int(np.ceil(125/(dataset.energy_scale[1]-dataset.energy_scale[0])/2)+1)
    new_spectrum =  scipy.signal.savgol_filter(dataset[start:], width, 2) ## we use half the width of the resolution for smearing
    #new_energy_scale = dataset.energy_scale[start:]
    prominence = 10
    minor_peaks, _  = scipy.signal.find_peaks(new_spectrum, prominence=prominence)
    
    while len(minor_peaks) > minimum_number_of_peaks:
        prominence+=10
        minor_peaks, _  = scipy.signal.find_peaks(new_spectrum, prominence=prominence)
    return np.array(minor_peaks)+start

def find_elements(spectrum, minor_peaks):
    if not isinstance(spectrum, sidpy.Dataset):
        raise TypeError(' Need a sidpy dataset')
    energy_scale = spectrum.energy_scale
    elements = []
    for peak in minor_peaks:
        found = False
        for element in range(3,82):
            if 'lines' in x_sections[str(element)]:
                if 'K-L3' in x_sections[str(element)]['lines']:
                    if abs(x_sections[str(element)]['lines']['K-L3']['position']- energy_scale[peak]) <10:
                        found = True
                        if  x_sections[str(element)]['name'] not in elements:
                            elements.append( x_sections[str(element)]['name'])
                if not found:
                    if 'K-L2' in x_sections[str(element)]['lines']:
                        if abs(x_sections[str(element)]['lines']['K-L2']['position']- energy_scale[peak]) <10:
                            found = True
                            if  x_sections[str(element)]['name'] not in elements:
                                elements.append( x_sections[str(element)]['name'])
                if not found:
                    if 'L3-M5' in x_sections[str(element)]['lines']:
                        if abs(x_sections[str(element)]['lines']['L3-M5']['position']- energy_scale[peak]) <30:
                              if  x_sections[str(element)]['name'] not in elements:
                                    elements.append( x_sections[str(element)]['name'])
    return elements

def get_x_ray_lines(spectrum, elements):
    out_tags = {}
    alpha_K = 1e6
    alpha_L = 6.5e7
    alpha_M = 8*1e8#2.2e10
    # My Fit
    alpha_K = .9e6
    alpha_L = 6.e7
    alpha_M = 6*1e8#2.2e10
    # omega_K = Z**4/(alpha_K+Z**4)
    # omega_L = Z**4/(alpha_L+Z**4)
    # omega_M = Z**4/(alpha_M+Z**4)
    for element in elements:

        atomic_number = elements_list.index(element)
        out_tags[element] ={'Z': atomic_number}
        energy_scale = spectrum.energy_scale
        if 'K-L3' in x_sections[str(atomic_number)]['lines']:
            if x_sections[str(atomic_number)]['lines']['K-L3']['position'] < 1.9e4:
                height = spectrum[np.searchsorted(energy_scale, x_sections[str(atomic_number)]['lines']['K-L3']['position'] )].compute()
                out_tags[element]['K-family'] = {'height': height}
                out_tags[element]['K-family']['yield'] = atomic_number**4/(alpha_K+atomic_number**4)/4/1.4

        if 'L3-M5' in x_sections[str(atomic_number)]['lines']:
            if x_sections[str(atomic_number)]['lines']['L3-M5']['position'] < 1.9e4:
                height = spectrum[np.searchsorted(energy_scale, x_sections[str(atomic_number)]['lines']['L3-M5']['position'] )].compute()
                out_tags[element]['L-family'] = {'height': height}
                out_tags[element]['L-family']['yield'] = (atomic_number**4/(alpha_L+atomic_number**4))**2

        if 'M5-N6' in x_sections[str(atomic_number)]['lines']:
            if x_sections[str(atomic_number)]['lines']['M5-N6']['position'] < 1.9e4:
                height = spectrum[np.searchsorted(energy_scale, x_sections[str(atomic_number)]['lines']['M5-N7']['position'] )].compute()
                out_tags[element]['M-family'] = {'height': height}
                out_tags[element]['M-family']['yield'] = (atomic_number**4/(alpha_M+atomic_number**4))**2

        for key, line in x_sections[str(atomic_number)]['lines'].items():
            other = True
            if line['weight'] > 0.01 and line['position'] < 3e4:
                if 'K-family' in out_tags[element]:
                    if key[0] == 'K':
                        other = False
                        out_tags[element]['K-family'][key]=line
                if 'L-family' in out_tags[element]:
                    if key[:2] in ['L2', 'L3']:
                        other = False
                        out_tags[element]['L-family'][key]=line
                if 'M-family' in out_tags[element]:
                    if key[:2] in ['M5', 'M4']:
                        other = False
                        out_tags[element]['M-family'][key]=line
                if other:
                    if 'other' not in out_tags[element]:
                        out_tags[element]['other'] = {}
                    height = spectrum[np.searchsorted(energy_scale, x_sections[str(atomic_number)]['lines'][key]['position'] )].compute()
                    out_tags[element]['other'][key]=line
                    out_tags[element]['other'][key]['height'] = height
      
        xs = get_eds_cross_sections(atomic_number)
        if 'K' in xs and 'K-family' in out_tags[element]:
            out_tags[element]['K-family']['ionization_x_section'] = xs['K']
        if 'L' in xs and 'L-family' in out_tags[element]:
            out_tags[element]['L-family']['ionization_x_section'] = xs['L']
        if 'M' in xs and 'M-family' in out_tags[element]:
            out_tags[element]['M-family']['ionization_x_section'] = xs['M']

    """
    for key, x_lines in out_tags.items():
        if 'K-family' in x_lines:
            xs = eels.xsec_xrpa(np.arange(100)+x_sections[str(x_lines['Z'])]['K1']['onset'], 200,x_lines['Z'], 100).sum()

            x_lines['K-family']['ionization_x_section'] = xs
            
        if 'L-family' in x_lines:
            xs = eels.xsec_xrpa(np.arange(100)+x_sections[str(x_lines['Z'])]['L3']['onset'], 200,x_lines['Z'], 100).sum()
            x_lines['L-family']['ionization_x_section'] = xs
        if 'M-family' in x_lines:
            xs = eels.xsec_xrpa(np.arange(100)+x_sections[str(x_lines['Z'])]['M5']['onset'], 200,x_lines['Z'], 100).sum()
            x_lines['M-family']['ionization_x_section'] = xs
    """
    return out_tags


def getFWHM(E, E_ref, FWHM_ref):
    return np.sqrt(2.5*(E-E_ref)+FWHM_ref**2)

def gaussian(enrgy_scale, mu, FWHM):
    sig = FWHM/2/np.sqrt(2*np.log(2))
    return np.exp(-np.power(enrgy_scale - mu, 2.) / (2 * np.power(sig, 2.)))

def get_peak(E, energy_scale):
    E_ref = 5895.0
    FWHM_ref = 136 #eV
    FWHM  = getFWHM(E, E_ref, FWHM_ref)
    gaus = gaussian(energy_scale, E, FWHM)

    return gaus /gaus.sum()


def get_model(tags, spectrum):

    energy_scale = spectrum.energy_scale
    p = []
    peaks = []
    keys = []
    for element, lines in tags.items():
        if 'K-family' in lines:
            model = np.zeros(len(energy_scale))
            for line, info in lines['K-family'].items():
                if line[0] == 'K':
                    model += get_peak(info['position'], energy_scale)*info['weight']
            lines['K-family']['peaks'] = model/model.max()
            lines['K-family']['height'] /= lines['K-family']['peaks'].max()
            p.append(lines['K-family']['height'])
            peaks.append(lines['K-family']['peaks'])
            keys.append(element+':K-family')
        if 'L-family' in lines:
            model = np.zeros(len(energy_scale))
            for line, info in lines['L-family'].items():
                if line[0] == 'L':
                    model += get_peak(info['position'], energy_scale)*info['weight']
            lines['L-family']['peaks'] = model/model.max()
            lines['L-family']['height'] /= lines['L-family']['peaks'].max()
            p.append(lines['L-family']['height'])
            peaks.append(lines['L-family']['peaks'])
            keys.append(element+':L-family')
        if 'M-family' in lines:
            model = np.zeros(len(energy_scale))
            for line, info in lines['M-family'].items():
                if line[0] == 'M':
                    model += get_peak(info['position'], energy_scale)*info['weight']
            lines['M-family']['peaks'] = model/model.max()
            lines['M-family']['height'] /= lines['M-family']['peaks'].max()
            p.append(lines['M-family']['height'])
            peaks.append(lines['M-family']['peaks'])
            keys.append(element+':M-family')
            
        if 'other' in lines:
            for line, info in lines['other'].items():
                info['peak'] =  get_peak(info['position'], energy_scale)
                peaks.append(info['peak'])
                p.append(info['height'])
                keys.append(element+':other:'+line)
    return np.array(peaks), np.array(p), keys

def fit_model(spectrum, elements):
    out_tags = get_x_ray_lines(spectrum, elements)
    
    peaks, pin, keys = get_model(out_tags, spectrum)

    def residuals(pp, yy):
        model = np.zeros(len(yy))
        for i in range(len(pp)):
            model += peaks[i]*pp[i]
        err = np.abs((yy - model)[75:]) / np.sqrt(np.abs(yy[75:]))
        return err

    y = spectrum.compute()
    [p, _] = leastsq(residuals, pin, args=(y))
    update_fit_values(out_tags, p)

    if 'EDS' not in spectrum.metadata:
        spectrum.metadata['EDS'] = {}
    spectrum.metadata['EDS']['lines'] = out_tags

    return np.array(peaks), np.array(p)


def update_fit_values(out_tags, p):
    index = 0
    for element, lines in out_tags.items():
        if 'K-family' in lines:
            lines['K-family']['height'] = p[index]
            index += 1
        if 'L-family' in lines:
            lines['L-family']['height'] = p[index]
            index += 1
        if 'M-family' in lines:
            lines['M-family']['height'] =p[index]
            index += 1
        if 'other' in lines:
            for line, info in lines['other'].items():
                info['height'] = p[index]
                index += 1
                

def get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd):
    background = eels.power_law_background(Xsection, energy_scale, [start_bgd, end_bgd], verbose=False)
    cross_section_core = Xsection- background[0]
    cross_section_core[cross_section_core < 0] = 0.0
    cross_section_core[energy_scale < end_bgd] = 0.0
    return cross_section_core


def get_eds_cross_sections(z):
    energy_scale = np.arange(10, 20000)
    Xsection = eels.xsec_xrpa(energy_scale, 200, z, 400.)
    edge_info = eels.get_x_sections(z)
    eds_cross_sections = {}
    if 'K1' in edge_info:
        start_bgd = edge_info['K1']['onset'] * 0.8
        end_bgd = edge_info['K1']['onset']  - 5
        if start_bgd > end_bgd:
            start_bgd = end_bgd-100
        if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
            eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
            eds_xsection = Xsection - eds_xsection
            eds_xsection[eds_xsection<0] = 0.
            start_sum = np.searchsorted(energy_scale, edge_info['K1']['onset'])
            eds_cross_sections['K'] = eds_xsection[start_sum:start_sum+200].sum() 
    if 'L3' in edge_info:
        start_bgd = edge_info['L3']['onset'] * 0.8
        end_bgd = edge_info['L3']['onset']  - 5
        if start_bgd > end_bgd:
            start_bgd = end_bgd-100
        if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
            eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
            eds_xsection = Xsection - eds_xsection
            eds_xsection[eds_xsection<0] = 0.
            start_sum = np.searchsorted(energy_scale, edge_info['L3']['onset'])
            eds_cross_sections['L'] = eds_xsection[start_sum:start_sum+200].sum() 
    if 'M5' in edge_info:
        start_bgd = edge_info['M5']['onset'] * 0.8
        end_bgd = edge_info['M5']['onset']  - 5
        if start_bgd > end_bgd:
            start_bgd = end_bgd-100
        if start_bgd > energy_scale[0] and end_bgd< energy_scale[-1]-100:
            eds_xsection = get_eds_xsection(Xsection, energy_scale, start_bgd, end_bgd)
            eds_xsection = Xsection - eds_xsection
            eds_xsection[eds_xsection<0] = 0.
            start_sum = np.searchsorted(energy_scale, edge_info['M5']['onset'])
            eds_cross_sections['M'] = eds_xsection[start_sum:start_sum+200].sum() 
    return eds_cross_sections
