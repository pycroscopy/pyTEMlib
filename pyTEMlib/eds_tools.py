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

import pickle  # pkg_resources
import pyTEMlib.eels_tools as eels

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

