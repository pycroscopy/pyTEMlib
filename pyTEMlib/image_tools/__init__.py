"""
Image Module

Should contain
- general feature extraction
- geometry feature extraction
- atom finding
- denoising
- windowing
- transforms (e.g., radon, hough)

Submodules
----------
.. autosummary::
    :toctree: _autosummary

"""
 
from .image_window import ImageWindowing
from .image_utilities import crop_image, flatten_image, inpaint_image, warp, rebin
from .image_clean import decon_lr, clean_svd, background_correction
from .image_atoms import find_atoms, atom_refine, intensity_area, atoms_clustering
from .image_graph import find_polyhedra, breadth_first_search, breadth_first_search_flexible
from .image_graph import get_base_atoms
from .image_distortion import get_distortion_matrix, undistort, undistort_sitk
from .image_registration import complete_registration, rigid_registration, demon_registration
from .image_fft import power_spectrum, diffractogram_spots, adaptive_fourier_filter
from .image_fft import rotational_symmetry_diffractogram
from .image_main import get_atomic_pseudo_potential, convolve_probe, get_wavelength
from .image_main import fourier_transform, center_diffractogram, ImageWithLineProfile
from .image_main import LineSelector, get_profile, get_line_selection_points_interpolated
from .image_main import get_line_selection_points, get_line_profile, histogram_plot
from .image_main import calculate_ctf, calculate_scherzer, get_rotation, calibrate_image_scale 
from .image_main import align_crystal_reflections


__all__ = ['ImageWindowing', 'crop_image', 'decon_lr', 'clean_svd', 'background_correction',
           'find_atoms', 'atom_refine', 'intensity_area', 'atoms_clustering',  
           'find_polyhedra', 'breadth_first_search', 'breadth_first_search_flexible', 'get_base_atoms',
           'get_distortion_matrix', 'undistort', 'undistort_sitk',
           'complete_registration', 'demon_registration', 'rigid_registration', 
           'flatten_image', 'inpaint_image', 'warp', 'rebin',
           'power_spectrum', 'diffractogram_spots', 'adaptive_fourier_filter', 
           'rotational_symmetry_diffractogram', 'get_atomic_pseudo_potential', 'convolve_probe', 
           'get_wavelength', 'fourier_transform', 'center_diffractogram', 'ImageWithLineProfile', 
           'LineSelector', 'get_profile', 'get_line_selection_points_interpolated', 
           'get_line_selection_points', 'get_line_profile', 'histogram_plot', 'calculate_ctf', 
           'calculate_scherzer', 'get_rotation', 'calibrate_image_scale', 
           'align_crystal_reflections']