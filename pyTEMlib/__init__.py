# -*- coding: utf-8 -*-
"""
PyTEMlib
--------
A Python package for analyzing and processing transmission electron microscopy (TEM) data.
This package provides tools for data quantification through a model-based approach, 
including functionalities for imaging, spectra analysis, and data visualization.

The package is part of the pycrosccopy ecosystem and the dataformat is based on sidpy, 
the fileformat is based on pyNSDI.
Created on Sat Jan 19 10:07:35 2019
Update on Sun Jul 20 2025

@author: gduscher
"""
from .version import  __version__

from . import file_tools
from . import  image_tools
from .image import image_atoms as atom_tools
from . import graph_tools
from . import probe_tools
from . import eels_tools
from . import eds_tools
from . import crystal_tools
from . import kinematic_scattering
from . import dynamic_scattering
from .config_dir import config_path
from . import utilities
from . import xrpa_x_sections

__all__ = ['__version__', 'file_tools', 'image_tools', 'atom_tools',
           'graph_tools', 'probe_tools', 'eels_tools', 'eds_tools',
           'crystal_tools', 'kinematic_scattering', 'dynamic_scattering', 
           'config_path', 'utilities', 'xrpa_x_sections']
__author__ = 'Gerd Duscher'
