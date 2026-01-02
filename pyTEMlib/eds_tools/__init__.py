"""
eds_tools - A collection of tools to analyze EDS data
Model based quantification of energy dispersive X-ray spectroscopy data
part of pyTEMlib
Author: Gerd Duscher
collected in a directory 12/2025
"""

from .eds_xsections import quantify_cross_section, quantification_k_factors, read_csv_k_factors
from .eds_xsections import convert_k_factor_file, get_k_factor_files, write_k_factors
from .eds_xsections import load_k_factors, read_k_factors, read_esl_k_factors

from .eds_tools import detector_response, get_absorption, get_detector_response, detect_peaks
from .eds_tools import peaks_element_correlation, get_elements, add_element, get_x_ray_lines
from .eds_tools import get_fwhm, gaussian, get_peak, initial_model_parameter, get_model
from .eds_tools import fit_model, update_fit_values, get_phases, plot_phases, plot_lines
from .eds_tools import get_eds_xsection, add_k_factors, quantify_eds
from .eds_tools import get_absorption_correction, apply_absorption_correction

__all__ = ['quantify_cross_section', 'quantification_k_factors',
           'detector_response', 'get_absorption', 'get_detector_response', 'detect_peaks',
           'peaks_element_correlation', 'get_elements', 'add_element', 'get_x_ray_lines',
           'get_fwhm', 'gaussian', 'get_peak', 'initial_model_parameter', 'get_model',
           'fit_model', 'update_fit_values', 'get_phases', 'plot_phases', 'plot_lines',
           'get_eds_xsection', 'add_k_factors', 'quantify_eds', 'read_esl_k_factors',
           'get_absorption_correction', 'apply_absorption_correction', 'read_csv_k_factors',
           'convert_k_factor_file', 'get_k_factor_files', 'write_k_factors', 'read_k_factors',
           'load_k_factors']
