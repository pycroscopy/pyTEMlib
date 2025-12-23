import scipy
import numpy as np
import pyTEMlib

def get_atomic_pseudo_potential(fov, atoms, size=512, rotation=0):
    """Big assumption: the atoms are not near the edge of the unit cell
    # If any atoms are close to the edge (ex. [0,0]) then the potential will be clipped
    # before calling the function, shift the atoms to the center of the unit cell
    """

    pixel_size = fov / size
    max_size = int(size * np.sqrt(2) + 1)  # Maximum size to accommodate rotation

    # Create unit cell potential
    positions = atoms.get_positions()[:, :2]
    atomic_numbers = atoms.get_atomic_numbers()
    unit_cell_size = atoms.cell.cellpar()[:2]

    unit_cell_potential = np.zeros((max_size, max_size))
    for pos, atomic_number in zip(positions, atomic_numbers):
        x = pos[0] / pixel_size
        y = pos[1] / pixel_size
        atom_width = 0.5  # Angstrom
        # important for images at various fov.  Room for improvement with theory
        gauss_width = atom_width/pixel_size
        gauss = pyTEMlib.probe_tools.make_gauss(max_size, max_size,
                                                width=gauss_width,
                                                x0=x, y0=y)
        unit_cell_potential += gauss * atomic_number  # gauss is already normalized to 1

    # Create interpolation function for unit cell potential
    x_grid = np.linspace(0, fov * max_size / size, max_size)
    y_grid = np.linspace(0, fov * max_size / size, max_size)
    interpolator = scipy.interpolate.RegularGridInterpolator((x_grid, y_grid),
                                                             unit_cell_potential,
                                                             bounds_error=False,
                                                             fill_value=0)
    # Vectorized computation of the full potential map with max_size
    x_coords, y_coords = np.meshgrid(np.linspace(0, fov, max_size),
                                     np.linspace(0, fov, max_size),
                                     indexing="ij")
    xtal_x = x_coords % unit_cell_size[0]
    xtal_y = y_coords % unit_cell_size[1]
    potential_map = interpolator((xtal_x.ravel(), xtal_y.ravel())).reshape(max_size, max_size)

    # Rotate and crop the potential map
    potential_map = scipy.ndimage.rotate(potential_map, rotation, reshape=False)
    center = potential_map.shape[0] // 2
    potential_map = potential_map[center - size // 2:center + size // 2,
                                  center - size // 2:center + size // 2]
    potential_map = scipy.ndimage.gaussian_filter(potential_map,3)

    return potential_map

def convolve_probe(ab, potential):
    """ Convolve probe with potential using FFT based convolution"""
    # the pixel sizes should be the exact same as the potential
    final_sizes = potential.shape

    # Perform FFT-based convolution
    pad_height = pad_width = potential.shape[0] // 2
    potential = np.pad(potential, ((pad_height, pad_height),
                                   (pad_width, pad_width)), mode='constant')

    probe, _, _ = pyTEMlib.probe_tools.get_probe(ab, potential.shape[0],
                                                 potential.shape[1],
                                                 scale='mrad', verbose=False)
    convolved = scipy.signal.fftconvolve(potential, probe, mode='same')

    # Crop to original potential size
    start_row = pad_height
    start_col = pad_width
    end_row = start_row + final_sizes[0]
    end_col = start_col + final_sizes[1]

    image = convolved[start_row:end_row, start_col:end_col]
    return probe, image
