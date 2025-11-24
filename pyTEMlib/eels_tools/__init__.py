"""
eels_tools - A collection of tools to analyze EELS data
Model based quantification of electron energy-loss data
part of pyTEMlib
Copyright by Gerd Duscher
seperated into different files 09/2025
"""
from ..utilities import major_edges, all_edges, first_close_edges, elements
from ..utilities import get_wavelength, effective_collection_angle, set_default_metadata
from ..utilities import lorentz, gauss, get_x_sections, get_z, get_spectrum

from .zero_loss_tools import zero_loss_function, get_resolution_functions
from .zero_loss_tools import get_zero_loss_energy, shift_energy, align_zero_loss

from .low_loss_tools import drude_simulation, kroeger_core
from .low_loss_tools import get_plasmon_losses, drude, drude_lorentz
from .low_loss_tools import energy_loss_function, angle_correction, fit_plasmon
from .low_loss_tools import fit_multiple_scattering, multiple_scattering
from .low_loss_tools import inelastic_mean_free_path, model3, add_peaks

from .peak_fit_tools import model_smooth, gaussian_mixture_model, find_peaks, find_maxima
from .peak_fit_tools import sort_peaks

from .core_loss_tools import make_cross_sections, fit_edges2, power_law_background
from .core_loss_tools import list_all_edges, find_all_edges, find_associated_edges
from .core_loss_tools import find_white_lines, find_edges, assign_likely_edges
from .core_loss_tools import auto_id_edges, identify_edges, add_element_to_dataset
from .core_loss_tools import fit_dataset, auto_chemical_composition, make_edges
from .core_loss_tools import cl_model, core_loss_model, fit_edges, xsec_xrpa


__all__ = ['major_edges', 'all_edges', 'first_close_edges', 'elements', 'get_wavelength',
           'effective_collection_angle', 'set_default_metadata', 'lorentz', 'gauss',
           'get_x_sections', 'get_z', 'get_spectrum', 'zero_loss_function',
           'get_resolution_functions', 'get_zero_loss_energy', 'shift_energy', 'align_zero_loss',
           'drude_simulation', 'kroeger_core','get_plasmon_losses', 'drude', 'drude_lorentz',
           'energy_loss_function', 'angle_correction', 'fit_plasmon', 'fit_multiple_scattering', 
           'multiple_scattering', 'inelastic_mean_free_path', 'model3', 'add_peaks',
           'model_smooth', 'gaussian_mixture_model', 'find_peaks', 'find_maxima', 'sort_peaks',
           'make_cross_sections', 'fit_edges2', 'power_law_background', 'list_all_edges',
           'find_all_edges', 'find_associated_edges', 'find_white_lines', 'find_edges',
           'assign_likely_edges', 'auto_id_edges', 'identify_edges', 'add_element_to_dataset',
           'fit_dataset', 'auto_chemical_composition', 'make_edges', 'cl_model', 'core_loss_model',
           'fit_edges', 'xsec_xrpa']
