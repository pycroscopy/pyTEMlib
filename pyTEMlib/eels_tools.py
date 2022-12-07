"""
eels_tools
Model based quantification of electron energy-loss data
Copyright by Gerd Duscher

The University of Tennessee, Knoxville
Department of Materials Science & Engineering

Sources:
   M. Tian et al.

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

from scipy import constants
import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# from matplotlib.widgets import SpanSelector
# import ipywidgets as widgets
# from IPython.display import display

import requests

from scipy.optimize import leastsq  # least square fitting routine fo scipy

import pickle  # pkg_resources,

# ## And we use the image tool library of pyTEMlib
import pyTEMlib.file_tools as ft
from pyTEMlib.xrpa_x_sections import x_sections

from sidpy.base.num_utils import get_slope

major_edges = ['K1', 'L3', 'M5', 'N5']
all_edges = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2',
             'O3', 'O4', 'O5', 'O6', 'O7', 'P1', 'P2', 'P3']
first_close_edges = ['K1', 'L3', 'M5', 'M3', 'N5', 'N3']

elements = [' ', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
            'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V',
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
            'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
            'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']


# kroeger_core(e_data,a_data,eps_data,ee,thick, relativistic =True)
# kroeger_core2(e_data,a_data,eps_data,acceleration_voltage_kev,thickness, relativistic =True)
# get_wave_length(e0)

# plot_dispersion(plotdata, units, a_data, e_data, title, max_p, ee, ef = 4., ep= 16.8, Es = 0, IBT = [])
# drude(tags, e, ep, ew, tnm, eb)
# drude(ep, eb, gamma, e)
# drude_lorentz(epsInf,leng, ep, eb, gamma, e, Amplitude)
# zl_func( p,  x)

###
def set_previous_quantification(current_dataset):
    """Set previous quantification from a sidpy.Dataset"""

    current_channel = current_dataset.h5_dataset.parent
    found_metadata = False
    for key in current_channel:
        if 'Log' in key:
            if current_channel[key]['analysis'][()] == 'EELS_quantification':
                current_dataset.metadata.update(current_channel[key].attrs)   # ToDo: find red dictionary
                found_metadata = True
                print('found previous quantification')

    if not found_metadata:
        # setting important experimental parameter
        current_dataset.metadata['experiment'] = ft.read_dm3_info(current_dataset.original_metadata)

        if 'experiment' not in current_dataset.metadata:
            current_dataset.metadata['experiment'] = {}
        if 'convergence_angle' not in current_dataset.metadata['experiment']:
            current_dataset.metadata['experiment']['convergence_angle'] = 30
        if 'collection_angle' not in current_dataset.metadata['experiment']:
            current_dataset.metadata['experiment']['collection_angle'] = 50
        if 'acceleration_voltage' not in current_dataset.metadata['experiment']:
            current_dataset.metadata['experiment']['acceleration_voltage'] = 200000
###

# ###############################################################
# Peak Fit Functions
# ################################################################


def residuals_smooth(p, x, y, only_positive_intensity):
    """part of fit"""

    err = (y - model_smooth(x, p, only_positive_intensity))
    return err


def model_smooth(x, p, only_positive_intensity=False):
    """part of fit"""

    y = np.zeros(len(x))

    number_of_peaks = int(len(p) / 3)
    for i in range(number_of_peaks):
        if only_positive_intensity:
            p[i * 3 + 1] = abs(p[i * 3 + 1])
        p[i * 3 + 2] = abs(p[i * 3 + 2])
        if p[i * 3 + 2] > abs(p[i * 3]) * 4.29193 / 2.0:
            p[i * 3 + 2] = abs(p[i * 3]) * 4.29193 / 2.  # ## width cannot extend beyond zero, maximum is FWTM/2

        y = y + gauss(x, p[i * 3:])

    return y


def residuals_ll(p, x, y, only_positive_intensity):
    """part of fit"""

    err = (y - model_ll(x, p, only_positive_intensity)) / np.sqrt(np.abs(y))
    return err


def residuals_ll2(p, x, y, only_positive_intensity):
    """part of fit"""

    err = (y - model_ll(x, p, only_positive_intensity))
    return err


def model_ll(x, p, only_positive_intensity):
    """part of fit"""

    y = np.zeros(len(x))

    number_of_peaks = int(len(p) / 3)
    for i in range(number_of_peaks):
        if only_positive_intensity:
            p[i * 3 + 1] = abs(p[i * 3 + 1])
        p[i * 3 + 2] = abs(p[i * 3 + 2])
        if p[i * 3 + 2] > abs(p[i * 3]) * 4.29193 / 2.0:
            p[i * 3 + 2] = abs(p[i * 3]) * 4.29193 / 2.  # ## width cannot extend beyond zero, maximum is FWTM/2

        y = y + gauss(x, p[i * 3:])

    return y


def fit_peaks(spectrum, energy_scale, pin, start_fit, end_fit, only_positive_intensity=False):
    """fit peaks to spectrum

    Parameters
    ----------
    spectrum: numpy array
        spectrum to be fitted
    energy_scale: numpy array
        energy scale of spectrum
    pin: list of float
        intial guess of peaks position amplitude width
    start_fit: int
        channel where fit starts
    end_fit: int
        channel where fit starts
    only_positive_intensity: boolean
        allows only for positive amplitudes if True; default = False

    Returns
    -------
    p: list of float
        fitting parameters
    """

    # TODO: remove zero_loss_fit_width add absolute

    fit_energy = energy_scale[start_fit:end_fit]
    fit_spectrum = spectrum[start_fit:end_fit]

    pin_flat = [item for sublist in pin for item in sublist]
    [p_out, _] = leastsq(residuals_ll, np.array(pin_flat), ftol=1e-3, args=(fit_energy, fit_spectrum,
                                                                            only_positive_intensity))
    p = []
    for i in range(len(pin)):
        if only_positive_intensity:
            p_out[i * 3 + 1] = abs(p_out[i * 3 + 1])
        p.append([p_out[i * 3], p_out[i * 3 + 1], abs(p_out[i * 3 + 2])])
    return p


#################################################################
# CORE - LOSS functions
#################################################################


def get_x_sections(z=0):
    """Reads X-ray fluorescent cross-sections from a pickle file.

    Parameters
    ----------
    z: int
        atomic number if zero all cross-sections will be returned

    Returns
    -------
    dictionary
        cross-section of an element or of all elements if z = 0

    """
    # pkl_file = open(data_path + '/edges_db.pkl', 'rb')
    # x_sections = pickle.load(pkl_file)
    # pkl_file.close()
    # x_sections = pyTEMlib.config_dir.x_sections
    z = int(z)

    if z < 1:
        return x_sections
    else:
        z = str(z)
        if z in x_sections:
            return x_sections[z]
        else:
            return 0


def get_z(z):
    """Returns the atomic number independent of input as a string or number

    Parameter
    ---------
    z: int, str
        atomic number of chemical symbol (0 if not valid)
    """
    x_sections = get_x_sections()

    z_out = 0
    if str(z).isdigit():
        z_out = int(z)
    elif isinstance(z, str):
        for key in x_sections:
            if x_sections[key]['name'].lower() == z.lower():  # Well one really should know how to write elemental
                z_out = int(key)
    return z_out


def list_all_edges(z):
    """List all ionization edges of an element with atomic number z

    Parameters
    ----------
    z: int
        atomic number

    Returns
    -------
    out_string: str
        string with all major edges in energy range
    """

    element = str(z)
    x_sections = get_x_sections()
    out_string = ''
    print('Major edges')
    for key in all_edges:
        if key in x_sections[element]:
            if 'onset' in x_sections[element][key]:
                print(f" {x_sections[element]['name']}-{key}: {x_sections[element][key]['onset']:8.1f} eV ")
                out_string = out_string + f" {x_sections[element]['name']}-{key}: " \
                                          f"{x_sections[element][key]['onset']:8.1f} eV /n"
    return out_string


def find_major_edges(edge_onset, maximal_chemical_shift=5.):
    """Find all major edges within an energy range

    Parameters
    ----------
    edge_onset: float
        approximate energy of ionization edge
    maximal_chemical_shift: float
        optional, range of energy window around edge_onset to look for major edges

    Returns
    -------
    text: str
        string with all major edges in energy range

    """
    text = ''
    x_sections = get_x_sections()
    for element in x_sections:
        for key in x_sections[element]:

            # if isinstance(x_sections[element][key], dict):
            if key in major_edges:

                if abs(x_sections[element][key]['onset'] - edge_onset) < maximal_chemical_shift:
                    # print(element, x_sections[element]['name'], key, x_sections[element][key]['onset'])
                    text = text + f"\n {x_sections[element]['name']:2s}-{key}: " \
                                  f"{x_sections[element][key]['onset']:8.1f} eV "

    return text


def find_all_edges(edge_onset, maximal_chemical_shift=5):
    """Find all (major and minor) edges within an energy range

    Parameters
    ----------
    edge_onset: float
        approximate energy of ionization edge
    maximal_chemical_shift: float
        optional, range of energy window around edge_onset to look for major edges

    Returns
    -------
    text: str
        string with all edges in energy range

    """

    text = ''
    x_sections = get_x_sections()
    for element in x_sections:
        for key in x_sections[element]:

            if isinstance(x_sections[element][key], dict):
                if 'onset' in x_sections[element][key]:
                    if abs(x_sections[element][key]['onset'] - edge_onset) < maximal_chemical_shift:
                        # print(element, x_sections[element]['name'], key, x_sections[element][key]['onset'])
                        text = text + f"\n {x_sections[element]['name']:2s}-{key}: " \
                                      f"{x_sections[element][key]['onset']:8.1f} eV "
    return text


def second_derivative(dataset, sensitivity):
    """Calculates second derivative of a sidpy.dataset"""

    dim = dataset.get_spectrum_dims()
    energy_scale = np.array(dataset._axes[dim[0]])
    if dataset.data_type.name == 'SPECTRAL_IMAGE':
        spectrum = dataset.view.get_spectrum()
    else:
        spectrum = np.array(dataset)

    spec = scipy.ndimage.gaussian_filter(spectrum, 3)

    dispersion = get_slope(energy_scale)
    second_dif = np.roll(spec, -3) - 2 * spec + np.roll(spec, +3)
    second_dif[:3] = 0
    second_dif[-3:] = 0

    # find if there is a strong edge at high energy_scale
    noise_level = 2. * np.std(second_dif[3:50])
    [indices, _] = scipy.signal.find_peaks(second_dif, noise_level)
    width = 50 / dispersion
    if width < 50:
        width = 50
    start_end_noise = int(len(energy_scale) - width)
    for index in indices[::-1]:
        if index > start_end_noise:
            start_end_noise = index - 70

    noise_level_start = sensitivity * np.std(second_dif[3:50])
    noise_level_end = sensitivity * np.std(second_dif[start_end_noise: start_end_noise + 50])
    slope = (noise_level_end - noise_level_start) / (len(energy_scale) - 400)
    noise_level = noise_level_start + np.arange(len(energy_scale)) * slope
    return second_dif, noise_level


def find_edges(dataset, sensitivity=2.5):
    """find edges within a sidpy.Dataset"""

    dim = dataset.get_spectrum_dims()
    energy_scale = np.array(dataset._axes[dim[0]])

    second_dif, noise_level = second_derivative(dataset, sensitivity=sensitivity)

    [indices, peaks] = scipy.signal.find_peaks(second_dif, noise_level)

    peaks['peak_positions'] = energy_scale[indices]
    peaks['peak_indices'] = indices
    edge_energies = [energy_scale[50]]
    edge_indices = []

    [indices, _] = scipy.signal.find_peaks(-second_dif, noise_level)
    minima = energy_scale[indices]

    for peak_number in range(len(peaks['peak_positions'])):
        position = peaks['peak_positions'][peak_number]
        if position - edge_energies[-1] > 20:
            impossible = minima[minima < position]
            impossible = impossible[impossible > position - 5]
            if len(impossible) == 0:
                possible = minima[minima > position]
                possible = possible[possible < position + 5]
                if len(possible) > 0:
                    edge_energies.append((position + possible[0])/2)
                    edge_indices.append(np.searchsorted(energy_scale, (position + possible[0])/2))

    selected_edges = []
    for peak in edge_indices:
        if 525 < energy_scale[peak] < 533:
            selected_edges.append('O-K1')
        else:
            selected_edge = ''
            edges = find_major_edges(energy_scale[peak], 20)
            edges = edges.split('\n')
            minimum_dist = 100.
            for edge in edges[1:]:
                edge = edge[:-3].split(':')
                name = edge[0].strip()
                energy = float(edge[1].strip())
                if np.abs(energy - energy_scale[peak]) < minimum_dist:
                    minimum_dist = np.abs(energy - energy_scale[peak])
                    selected_edge = name

            if selected_edge != '':
                selected_edges.append(selected_edge)

    return selected_edges


def make_edges(edges_present, energy_scale, e_0, coll_angle, low_loss=None):
    """Makes the edges dictionary for quantification

    Parameters
    ----------
    edges_present: list
        list of edges
    energy_scale: numpy array
        energy scale on which to make cross-section
    e_0: float
        acceleration voltage (in V)
    coll_angle: float
        collection angle in mrad
    low_loss: numpy array with same length as energy_scale
        low_less spectrum with which to convolve the cross-section (default=None)

    Returns
    -------
    edges: dict
        dictionary with all information on cross-section
    """
    x_sections = get_x_sections()
    edges = {}
    for i, edge in enumerate(edges_present):
        element, symmetry = edge.split('-')
        z = 0
        for key in x_sections:
            if element == x_sections[key]['name']:
                z = int(key)
        edges[i] = {}
        edges[i]['z'] = z
        edges[i]['symmetry'] = symmetry
        edges[i]['element'] = element

    for key in edges:
        xsec = x_sections[str(edges[key]['z'])]
        if 'chemical_shift' not in edges[key]:
            edges[key]['chemical_shift'] = 0
        if 'symmetry' not in edges[key]:
            edges[key]['symmetry'] = 'K1'
        if 'K' in edges[key]['symmetry']:
            edges[key]['symmetry'] = 'K1'
        elif 'L' in edges[key]['symmetry']:
            edges[key]['symmetry'] = 'L3'
        elif 'M' in edges[key]['symmetry']:
            edges[key]['symmetry'] = 'M5'
        else:
            edges[key]['symmetry'] = edges[key]['symmetry'][0:2]

        edges[key]['original_onset'] = xsec[edges[key]['symmetry']]['onset']
        edges[key]['onset'] = edges[key]['original_onset'] + edges[key]['chemical_shift']
        edges[key]['start_exclude'] = edges[key]['onset'] - xsec[edges[key]['symmetry']]['excl before']
        edges[key]['end_exclude'] = edges[key]['onset'] + xsec[edges[key]['symmetry']]['excl after']

    edges = make_cross_sections(edges, energy_scale, e_0, coll_angle, low_loss)

    return edges


def make_cross_sections(edges, energy_scale, e_0, coll_angle, low_loss=None):
    """Updates the edges dictionary with collection angle-integrated X-ray photo-absorption cross-sections

    """
    for key in edges:
        if str(key).isdigit():
            edges[key]['data'] = xsec_xrpa(energy_scale, e_0 / 1000., edges[key]['z'], coll_angle,
                                           edges[key]['chemical_shift']) / 1e10  # from barnes to 1/nm^2
            if low_loss is not None:
                low_loss = np.roll(np.array(low_loss), 1024 - np.argmax(np.array(low_loss)))
                edges[key]['data'] = scipy.signal.convolve(edges[key]['data'], low_loss/low_loss.sum(), mode='same')

            edges[key]['onset'] = edges[key]['original_onset'] + edges[key]['chemical_shift']
            edges[key]['X_section_type'] = 'XRPA'
            edges[key]['X_section_source'] = 'pyTEMlib'

    return edges


def power_law(energy, a, r):
    """power law for power_law_background"""
    return a * np.power(energy, -r)


def power_law_background(spectrum, energy_scale, fit_area, verbose=False):
    """fit of power law to spectrum """

    # Determine energy window  for background fit in pixels
    startx = np.searchsorted(energy_scale, fit_area[0])
    endx = np.searchsorted(energy_scale, fit_area[1])

    x = np.array(energy_scale)[startx:endx]

    y = np.array(spectrum)[startx:endx].flatten()

    # Initial values of parameters
    p0 = np.array([1.0E+20, 3])

    # background fitting
    def bgdfit(pp, yy, xx):
        err = yy - power_law(xx, pp[0], pp[1])
        return err

    [p, _] = leastsq(bgdfit, p0, args=(y, x), maxfev=2000)

    background_difference = y - power_law(x, p[0], p[1])
    background_noise_level = std_dev = np.std(background_difference)
    if verbose:
        print(f'Power-law background with amplitude A: {p[0]:.1f} and exponent -r: {p[1]:.2f}')
        print(background_difference.max() / background_noise_level)

        print(f'Noise level in spectrum {std_dev:.3f} counts')

    # Calculate background over the whole energy scale
    background = power_law(energy_scale, p[0], p[1])
    return background, p


def cl_model(x, p, number_of_edges, xsec):
    """ core loss model for fitting"""
    y = (p[9] * np.power(x, (-p[10]))) + p[7] * x + p[8] * x * x
    for i in range(number_of_edges):
        y = y + p[i] * xsec[i, :]
    return y


def fit_edges2(spectrum, energy_scale, edges):
    """fit edges for quantification"""

    dispersion = energy_scale[1] - energy_scale[0]
    # Determine fitting ranges and masks to exclude ranges
    mask = np.ones(len(spectrum))

    background_fit_start = edges['fit_area']['fit_start']
    if edges['fit_area']['fit_end'] > energy_scale[-1]:
        edges['fit_area']['fit_end'] = energy_scale[-1]
    background_fit_end = edges['fit_area']['fit_end']

    startx = np.searchsorted(energy_scale, background_fit_start)
    endx = np.searchsorted(energy_scale, background_fit_end)
    mask[0:startx] = 0.0
    mask[endx:-1] = 0.0
    for key in edges:
        if key.isdigit():
            if edges[key]['start_exclude'] > background_fit_start + dispersion:
                if edges[key]['start_exclude'] < background_fit_end - dispersion * 2:
                    if edges[key]['end_exclude'] > background_fit_end - dispersion:
                        # we need at least one channel to fit.
                        edges[key]['end_exclude'] = background_fit_end - dispersion
                    startx = np.searchsorted(energy_scale, edges[key]['start_exclude'])
                    if startx < 2:
                        startx = 1
                    endx = np.searchsorted(energy_scale, edges[key]['end_exclude'])
                    mask[startx: endx] = 0.0

    ########################
    # Background Fit
    ########################
    bgd_fit_area = [background_fit_start, background_fit_end]
    background, [A, r] = power_law_background(spectrum, energy_scale, bgd_fit_area, verbose=False)

    #######################
    # Edge Fit
    #######################
    x = energy_scale
    blurred = gaussian_filter(spectrum, sigma=5)

    y = blurred  # now in probability
    y[np.where(y < 1e-8)] = 1e-8

    xsec = []
    number_of_edges = 0
    for key in edges:
        if key.isdigit():
            xsec.append(edges[key]['data'])
            number_of_edges += 1
    xsec = np.array(xsec)

    def model(xx, pp):
        yy = background + pp[6] + pp[7] * xx + pp[8] * xx * xx
        for i in range(number_of_edges):
            pp[i] = np.abs(pp[i])
            yy = yy + pp[i] * xsec[i, :]
        return yy

    def residuals(pp, xx, yy):
        err = np.abs((yy - model(xx, pp)) * mask)  # / np.sqrt(np.abs(y))
        return err

    scale = y[100]
    pin = np.array([scale / 5, scale / 5, scale / 5, scale / 5, scale / 5, scale / 5, -scale / 10, 1.0, 0.001])
    [p, _] = leastsq(residuals, pin, args=(x, y))

    for key in edges:
        if key.isdigit():
            edges[key]['areal_density'] = p[int(key)]

    edges['model'] = {}
    edges['model']['background'] = (background + p[6] + p[7] * x + p[8] * x * x)
    edges['model']['background-poly_0'] = p[6]
    edges['model']['background-poly_1'] = p[7]
    edges['model']['background-poly_2'] = p[8]
    edges['model']['background-A'] = A
    edges['model']['background-r'] = r
    edges['model']['spectrum'] = model(x, p)
    edges['model']['blurred'] = blurred
    edges['model']['mask'] = mask
    edges['model']['fit_parameter'] = p
    edges['model']['fit_area_start'] = edges['fit_area']['fit_start']
    edges['model']['fit_area_end'] = edges['fit_area']['fit_end']

    return edges


def fit_edges(spectrum, energy_scale, region_tags, edges):
    """fit edges for quantification"""

    # Determine fitting ranges and masks to exclude ranges
    mask = np.ones(len(spectrum))

    background_fit_end = energy_scale[-1]
    for key in region_tags:
        end = region_tags[key]['start_x'] + region_tags[key]['width_x']

        startx = np.searchsorted(energy_scale, region_tags[key]['start_x'])
        endx = np.searchsorted(energy_scale, end)

        if key == 'fit_area':
            mask[0:startx] = 0.0
            mask[endx:-1] = 0.0
        else:
            mask[startx:endx] = 0.0
            if region_tags[key]['start_x'] < background_fit_end:  # Which is the onset of the first edge?
                background_fit_end = region_tags[key]['start_x']

    ########################
    # Background Fit
    ########################
    bgd_fit_area = [region_tags['fit_area']['start_x'], background_fit_end]
    background, [A, r] = power_law_background(spectrum, energy_scale, bgd_fit_area, verbose=False)

    #######################
    # Edge Fit
    #######################
    x = energy_scale
    blurred = gaussian_filter(spectrum, sigma=5)

    y = blurred  # now in probability
    y[np.where(y < 1e-8)] = 1e-8

    xsec = []
    number_of_edges = 0
    for key in edges:
        if key.isdigit():
            xsec.append(edges[key]['data'])
            number_of_edges += 1
    xsec = np.array(xsec)

    def model(xx, pp):
        yy = background + pp[6] + pp[7] * xx + pp[8] * xx * xx
        for i in range(number_of_edges):
            pp[i] = np.abs(pp[i])
            yy = yy + pp[i] * xsec[i, :]
        return yy

    def residuals(pp, xx, yy):
        err = np.abs((yy - model(xx, pp)) * mask)  # / np.sqrt(np.abs(y))
        return err

    scale = y[100]
    pin = np.array([scale / 5, scale / 5, scale / 5, scale / 5, scale / 5, scale / 5, -scale / 10, 1.0, 0.001])
    [p, _] = leastsq(residuals, pin, args=(x, y))

    for key in edges:
        if key.isdigit():
            edges[key]['areal_density'] = p[int(key) - 1]

    edges['model'] = {}
    edges['model']['background'] = (background + p[6] + p[7] * x + p[8] * x * x)
    edges['model']['background-poly_0'] = p[6]
    edges['model']['background-poly_1'] = p[7]
    edges['model']['background-poly_2'] = p[8]
    edges['model']['background-A'] = A
    edges['model']['background-r'] = r
    edges['model']['spectrum'] = model(x, p)
    edges['model']['blurred'] = blurred
    edges['model']['mask'] = mask
    edges['model']['fit_parameter'] = p
    edges['model']['fit_area_start'] = region_tags['fit_area']['start_x']
    edges['model']['fit_area_end'] = region_tags['fit_area']['start_x'] + region_tags['fit_area']['width_x']

    return edges


def find_peaks(dataset, fit_start, fit_end, sensitivity=2):
    """find peaks in spectrum"""

    if dataset.data_type.name == 'SPECTRAL_IMAGE':
        spectrum = dataset.view.get_spectrum()
    else:
        spectrum = np.array(dataset)

    spec_dim = ft.get_dimensions_by_type('SPECTRAL', dataset)[0]
    energy_scale = np.array(spec_dim[1])

    second_dif, noise_level = second_derivative(dataset, sensitivity=sensitivity)
    [indices, _] = scipy.signal.find_peaks(-second_dif, noise_level)

    start_channel = np.searchsorted(energy_scale, fit_start)
    end_channel = np.searchsorted(energy_scale, fit_end)
    peaks = []
    for index in indices:
        if start_channel < index < end_channel:
            peaks.append(index - start_channel)

    if energy_scale[0] > 0:
        if 'edges' not in dataset.metadata:
            return
        if 'model' not in dataset.metadata['edges']:
            return
        model = dataset.metadata['edges']['model']['spectrum'][start_channel:end_channel]

    else:
        model = np.zeros(end_channel - start_channel)

    energy_scale = energy_scale[start_channel:end_channel]

    difference = np.array(spectrum)[start_channel:end_channel] - model
    fit = np.zeros(len(energy_scale))
    if len(peaks) > 0:
        p_in = np.ravel([[energy_scale[i], difference[i], .7] for i in peaks])
        [p_out, _] = scipy.optimize.leastsq(residuals_smooth, p_in, ftol=1e-3, args=(energy_scale,
                                                                                     difference,
                                                                                     False))
        fit = fit + model_smooth(energy_scale, p_out, False)

    peak_model = np.zeros(len(spec_dim[1]))
    peak_model[start_channel:end_channel] = fit

    return peak_model, p_out


def find_maxima(y, number_of_peaks):
    """ find the first most prominent peaks

    peaks are then sorted by energy

    Parameters
    ----------
    y: numpy array
        (part) of spectrum
    number_of_peaks: int

    Returns
    -------
    numpy array
        indices of peaks
    """
    blurred2 = gaussian_filter(y, sigma=2)
    peaks, _ = scipy.signal.find_peaks(blurred2)
    prominences = peak_prominences(blurred2, peaks)[0]
    prominences_sorted = np.argsort(prominences)
    peaks = peaks[prominences_sorted[-number_of_peaks:]]

    peak_indices = np.argsort(peaks)
    return peaks[peak_indices]


def gauss(x, p):  # p[0]==mean, p[1]= amplitude p[2]==fwhm,
    """Gaussian Function

        p[0]==mean, p[1]= amplitude p[2]==fwhm
        area = np.sqrt(2* np.pi)* p[1] * np.abs(p[2] / 2.3548)
        FWHM = 2 * np.sqrt(2 np.log(2)) * sigma = 2.3548 * sigma
        sigma = FWHM/3548
    """
    if p[2] == 0:
        return x * 0.
    else:
        return p[1] * np.exp(-(x - p[0]) ** 2 / (2.0 * (p[2] / 2.3548) ** 2))


def lorentz(x, p):
    """lorentzian function"""
    lorentz_peak = 0.5 * p[2] / np.pi / ((x - p[0]) ** 2 + (p[2] / 2) ** 2)
    return p[1] * lorentz_peak / lorentz_peak.max()


def zl(x, p, p_zl):
    """zero-loss function"""
    p_zl_local = p_zl.copy()
    p_zl_local[2] += p[0]
    p_zl_local[5] += p[0]
    zero_loss = zl_func(p_zl_local, x)
    return p[1] * zero_loss / zero_loss.max()


def model3(x, p, number_of_peaks, peak_shape, p_zl, pin=None, restrict_pos=0, restrict_width=0):
    """ model for fitting low-loss spectrum"""
    if pin is None:
        pin = p

    # if len([restrict_pos]) == 1:
    #    restrict_pos = [restrict_pos]*number_of_peaks
    # if len([restrict_width]) == 1:
    #    restrict_width = [restrict_width]*number_of_peaks
    y = np.zeros(len(x))

    for i in range(number_of_peaks):
        index = int(i * 3)
        if restrict_pos > 0:
            if p[index] > pin[index] * (1.0 + restrict_pos):
                p[index] = pin[index] * (1.0 + restrict_pos)
            if p[index] < pin[index] * (1.0 - restrict_pos):
                p[index] = pin[index] * (1.0 - restrict_pos)

        p[index + 1] = abs(p[index + 1])
        # print(p[index + 1])
        p[index + 2] = abs(p[index + 2])
        if restrict_width > 0:
            if p[index + 2] > pin[index + 2] * (1.0 + restrict_width):
                p[index + 2] = pin[index + 2] * (1.0 + restrict_width)

        if peak_shape[i] == 'Lorentzian':
            y = y + lorentz(x, p[index:])
        elif peak_shape[i] == 'zl':

            y = y + zl(x, p[index:], p_zl)
        else:
            y = y + gauss(x, p[index:])
    return y


def sort_peaks(p, peak_shape):
    """sort fitting parameters by peak position"""
    number_of_peaks = int(len(p) / 3)
    p3 = np.reshape(p, (number_of_peaks, 3))
    sort_pin = np.argsort(p3[:, 0])

    p = p3[sort_pin].flatten()
    peak_shape = np.array(peak_shape)[sort_pin].tolist()

    return p, peak_shape


def add_peaks(x, y, peaks, pin_in=None, peak_shape_in=None, shape='Gaussian'):
    """ add peaks to fitting parameters"""
    if pin_in is None:
        return
    if peak_shape_in is None:
        return

    pin = pin_in.copy()

    peak_shape = peak_shape_in.copy()
    if isinstance(shape, str):  # if peak_shape is only a string make a list of it.
        shape = [shape]

    if len(shape) == 1:
        shape = shape * len(peaks)
    for i, peak in enumerate(peaks):
        pin.append(x[peak])
        pin.append(y[peak])
        pin.append(.3)
        peak_shape.append(shape[i])

    return pin, peak_shape


def fit_model(x, y, pin, number_of_peaks, peak_shape, p_zl, restrict_pos=0, restrict_width=0):
    """model for fitting low-loss spectrum"""

    pin_original = pin.copy()

    def residuals3(pp, xx, yy):
        err = (yy - model3(xx, pp, number_of_peaks, peak_shape, p_zl, pin_original, restrict_pos,
                           restrict_width)) / np.sqrt(np.abs(yy))
        return err

    [p, _] = leastsq(residuals3, pin, args=(x, y))
    # p2 = p.tolist()
    # p3 = np.reshape(p2, (number_of_peaks, 3))
    # sort_pin = np.argsort(p3[:, 0])

    # p = p3[sort_pin].flatten()
    # peak_shape = np.array(peak_shape)[sort_pin].tolist()

    return p, peak_shape


def fix_energy_scale(spec, energy):
    """Shift energy scale according to zero-loss peak position"""
    # determine start and end fitting region in pixels
    start = np.searchsorted(energy, -10)
    end = np.searchsorted(energy, 10)
    startx = np.argmax(spec[start:end]) + start

    end = startx + 3
    start = startx - 3
    for i in range(10):
        if spec[startx - i] < 0.3 * spec[startx]:
            start = startx - i
        if spec[startx + i] < 0.3 * spec[startx]:
            end = startx + i
    if end - start < 3:
        end = startx + 2
        start = startx - 2

    x = np.array(energy[int(start):int(end)])
    y = np.array(spec[int(start):int(end)]).copy()

    y[np.nonzero(y <= 0)] = 1e-12

    p0 = [energy[startx], 1000.0, (energy[end] - energy[start]) / 3.]  # Initial guess is a normal distribution

    def errfunc(pp, xx, yy):
        return (gauss(xx, pp) - yy) / np.sqrt(yy)  # Distance to the target function

    [p1, _] = leastsq(errfunc, np.array(p0[:]), args=(x, y))
    fit_mu, area, fwhm = p1

    return fwhm, fit_mu


def resolution_function(energy_scale, spectrum, width, verbose=False):
    """get resolution function (zero-loss peak shape) from low-loss spectrum"""

    guess = [0.2, 1000, 0.02, 0.2, 1000, 0.2]
    p0 = np.array(guess)

    start = np.searchsorted(energy_scale, -width / 2.)
    end = np.searchsorted(energy_scale, width / 2.)
    x = energy_scale[start:end]
    y = spectrum[start:end]

    def zl2(pp, yy, xx):
        eerr = (yy - zl_func(pp, xx))  # /np.sqrt(y)
        return eerr

    def zl_restrict(pp, yy, xx):

        if pp[2] > xx[-1] * .8:
            pp[2] = xx[-1] * .8
        if pp[2] < xx[0] * .8:
            pp[2] = xx[0] * .8

        if pp[5] > xx[-1] * .8:
            pp[5] = xx[-1] * .8
        if pp[5] < x[0] * .8:
            pp[5] = xx[0] * .8

        if len(pp) > 6:
            pp[7] = abs(pp[7])
            if abs(pp[7]) > (pp[1] + pp[4]) / 10:
                pp[7] = abs(pp[1] + pp[4]) / 10
            if abs(pp[8]) > 1:
                pp[8] = pp[8] / abs(pp[8])
            pp[6] = abs(pp[6])
            pp[9] = abs(pp[9])

        pp[0] = abs(pp[0])
        pp[3] = abs(pp[3])
        if pp[0] > (xx[-1] - xx[0]) / 2.0:
            pp[0] = xx[-1] - xx[0] / 2.0
        if pp[3] > (xx[-1] - xx[0]) / 2.0:
            pp[3] = xx[-1] - xx[0] / 2.0

        yy[yy < 0] = 0.  # no negative numbers in sqrt below
        eerr = (yy - zl_func(pp, xx)) / np.sqrt(yy)

        return eerr

    [p_zl, _] = leastsq(zl2, p0, args=(y, x), maxfev=2000)
    if verbose:
        print('Fit of a Product of two Lorentzian')
        print('Positions: ', p_zl[2], p_zl[5], 'Distance: ', p_zl[2] - p_zl[5])
        print('Width: ', p_zl[0], p_zl[3])
        print('Areas: ', p_zl[1], p_zl[4])
        err = (y - zl_func(p_zl, x)) / np.sqrt(y)
        print(f'Goodness of Fit: {sum(err ** 2) / len(y) / sum(y) * 1e2:.5}%')

    z_loss = zl_func(p_zl, energy_scale)

    return z_loss, p_zl


def get_energy_shifts(spectrum_image, energy_scale, zero_loss_fit_width):
    """get shift of spectrum form zero-loss peak position"""
    shifts = np.zeros(spectrum_image.shape[0:2])
    for x in range(spectrum_image.shape[0]):
        for y in range(spectrum_image.shape[1]):
            spectrum = spectrum_image[x, y, :]
            fwhm, delta_e = fix_energy_scale(spectrum, energy_scale)
            z_loss, p_zl = resolution_function(energy_scale - delta_e, spectrum, zero_loss_fit_width)
            fwhm2, delta_e2 = fix_energy_scale(z_loss, energy_scale - delta_e)
            shifts[x, y] = delta_e + delta_e2
    return shifts


def shift_on_same_scale(spectrum_image, shift, energy_scale, master_energy_scale):
    """shift spectrum in energy"""

    new_si = np.zeros(spectrum_image.shape)
    for x in range(spectrum_image.shape[0]):
        for y in range(spectrum_image.shape[1]):
            tck = interpolate.splrep(energy_scale - shift[x, y], spectrum_image[x, y, :], k=1, s=0)
            new_si[x, y, :] = interpolate.splev(master_energy_scale, tck, der=0)
    return new_si


def get_wave_length(e0):
    """get deBroglie wavelength of electron accelerated by energy (in eV) e0"""

    ev = constants.e * e0
    return constants.h / np.sqrt(2 * constants.m_e * ev * (1 + ev / (2 * constants.m_e * constants.c ** 2)))


def drude(ep, eb, gamma, e):
    """dielectric function according to Drude theory"""

    eps = 1 - (ep ** 2 - eb * e * 1j) / (e ** 2 + 2 * e * gamma * 1j)  # Mod drude term
    return eps


def drude_lorentz(eps_inf, leng, ep, eb, gamma, e, amplitude):
    """dielectric function according to Drude-Lorentz theory"""

    eps = eps_inf
    for i in range(leng):
        eps = eps + amplitude[i] * (1 / (e + ep[i] + gamma[i] * 1j) - 1 / (e - ep[i] + gamma[i] * 1j))
    return eps


def plot_dispersion(plotdata, units, a_data, e_data, title, max_p, ee, ef=4., ep=16.8, es=0, ibt=[]):
    """Plot loss function """

    [x, y] = np.meshgrid(e_data + 1e-12, a_data[1024:2048] * 1000)

    z = plotdata
    lev = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 4.9]) * max_p / 5

    wavelength = get_wave_length(ee)
    q = a_data[1024:2048] / (wavelength * 1e9)  # in [1/nm]
    scale = np.array([0, a_data[-1], e_data[0], e_data[-1]])
    ev2hertz = constants.value('electron volt-hertz relationship')

    if units[0] == 'mrad':
        units[0] = 'scattering angle [mrad]'
        scale[1] = scale[1] * 1000.
        light_line = constants.c * a_data  # for mrad
    elif units[0] == '1/nm':
        units[0] = 'scattering vector [1/nm]'
        scale[1] = scale[1] / (wavelength * 1e9)
        light_line = 1 / (constants.c / ev2hertz) * 1e-9

    if units[1] == 'eV':
        units[1] = 'energy loss [eV]'

    if units[2] == 'ppm':
        units[2] = 'probability [ppm]'
    if units[2] == '1/eV':
        units[2] = 'probability [eV$^{-1}$ srad$^{-1}$]'

    alpha = 3. / 5. * ef / ep

    ax2 = plt.gca()
    fig2 = plt.gcf()
    im = ax2.imshow(z.T, clim=(0, max_p), origin='lower', aspect='auto', extent=scale)
    co = ax2.contour(y, x, z, levels=lev, colors='k', origin='lower')
    # ,extent=(-ang*1000.,ang*1000.,e_data[0],e_data[-1]))#, vmin = p_vol.min(), vmax = 1000)

    fig2.colorbar(im, ax=ax2, label=units[2])

    ax2.plot(a_data, light_line, c='r', label='light line')
    # ax2.plot(e_data*light_line*np.sqrt(np.real(eps_data)),e_data, color='steelblue',
    # label='$\omega = c q \sqrt{\epsilon_2}$')

    # ax2.plot(q, Ep_disp, c='r')
    ax2.plot([11.5 * light_line, 0.12], [11.5, 11.5], c='r')

    ax2.text(.05, 11.7, 'surface plasmon', color='r')
    ax2.plot([0.0, 0.12], [16.8, 16.8], c='r')
    ax2.text(.05, 17, 'volume plasmon', color='r')
    ax2.set_xlim(0, scale[1])
    ax2.set_ylim(0, 20)
    # Interband transitions
    ax2.plot([0.0, 0.25], [4.2, 4.2], c='g', label='interband transitions')
    ax2.plot([0.0, 0.25], [5.2, 5.2], c='g')
    ax2.set_ylabel(units[1])
    ax2.set_xlabel(units[0])
    ax2.legend(loc='lower right')


def zl_func(p, x):
    """zero-loss peak function"""

    p[0] = abs(p[0])

    gauss1 = np.zeros(len(x))
    gauss2 = np.zeros(len(x))
    lorentz3 = np.zeros(len(x))
    lorentz = ((0.5 * p[0] * p[1] / 3.14) / ((x - p[2]) ** 2 + ((p[0] / 2) ** 2)))
    lorentz2 = ((0.5 * p[3] * p[4] / 3.14) / ((x - (p[5])) ** 2 + ((p[3] / 2) ** 2)))
    if len(p) > 6:
        lorentz3 = (0.5 * p[6] * p[7] / 3.14) / ((x - p[8]) ** 2 + (p[6] / 2) ** 2)
        gauss2 = p[10] * np.exp(-(x - p[11]) ** 2 / (2.0 * (p[9] / 2.3548) ** 2))
        # ((0.5 *  p[9]* p[10]/3.14)/((x- (p[11]))**2+(( p[9]/2)**2)))
    y = (lorentz * lorentz2) + gauss1 + gauss2 + lorentz3

    return y


def drude2(tags, e, p):
    """dielectric function according to Drude theory for fitting"""

    return drude(e, p[0], p[1], p[2], p[3])


def xsec_xrpa(energy_scale, e0, z, beta, shift=0):
    """ Calculate momentum-integrated cross-section for EELS from X-ray photo-absorption cross-sections.

    X-ray photo-absorption cross-sections from NIST.
    Momentum-integrated cross-section for EELS according to Egerton Ultramicroscopy 50 (1993) 13-28 equation (4)

    Parameters
    ----------
    energy_scale: numpy array
        energy scale of spectrum to be analyzed
    e0: float
        acceleration voltage in keV
    z: int
        atomic number of element
    beta: float
        effective collection angle in mrad
    shift: float
        chemical shift of edge in eV
    """
    beta = beta * 0.001  # collection half angle theta [rad]
    # theta_max = self.parent.spec[0].convAngle * 0.001  # collection half angle theta [rad]
    dispersion = energy_scale[1] - energy_scale[0]

    x_sections = get_x_sections(z)
    enexs = x_sections['ene']
    datxs = x_sections['dat']

    #####
    # Cross Section according to Egerton Ultramicroscopy 50 (1993) 13-28 equation (4)
    #####

    # Relativistic correction factors
    t = 511060.0 * (1.0 - 1.0 / (1.0 + e0 / 511.06) ** 2) / 2.0
    gamma = 1 + e0 / 511.06
    a = 6.5  # e-14 *10**14
    b = beta

    theta_e = enexs / (2 * gamma * t)

    g = 2 * np.log(gamma) - np.log((b ** 2 + theta_e ** 2) / (b ** 2 + theta_e ** 2 / gamma ** 2)) - (
            gamma - 1) * b ** 2 / (b ** 2 + theta_e ** 2 / gamma ** 2)
    datxs = datxs * (a / enexs / t) * (np.log(1 + b ** 2 / theta_e ** 2) + g) / 1e8

    datxs = datxs * dispersion  # from per eV to per dispersion
    coeff = splrep(enexs, datxs, s=0)  # now in areal density atoms / m^2
    xsec = np.zeros(len(energy_scale))
    # shift = 0# int(ek -onsetXRPS)#/dispersion
    lin = interp1d(enexs, datxs, kind='linear')  # Linear instead of spline interpolation to avoid oscillations.
    if energy_scale[0] < enexs[0]:
        start = np.searchsorted(energy_scale, enexs[0])+1
    else:
        start = 0
    xsec[start:] = lin(energy_scale[start:] - shift)

    return xsec


def drude_simulation(dset, e, ep, ew, tnm, eb):
    """probabilities of dielectric function eps relative to zero-loss integral (i0 = 1)

    Gives probabilities of dielectric function eps relative to zero-loss integral (i0 = 1) per eV
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011

    # function drude(ep,ew,eb,epc,e0,beta,nn,tnm)
    # Given the plasmon energy (ep), plasmon fwhm (ew) and binding energy(eb),
    # this program generates:
    # EPS1, EPS2 from modified Eq. (3.40), ELF=Im(-1/EPS) from Eq. (3.42),
    # single scattering from Eq. (4.26) and SRFINT from Eq. (4.31)
    # The output is e, ssd into the file drude.ssd (for use in Flog etc.)
    # and e,eps1 ,eps2 into drude.eps (for use in Kroeger etc.)
    # Gives probabilities relative to zero-loss integral (i0 = 1) per eV
    # Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    # Version 10.11.26


    b.7 drude Simulation of a Low-Loss Spectrum
    The program DRUDE calculates a single-scattering plasmon-loss spectrum for
    a specimen of a given thickness tnm (in nm), recorded with electrons of a
    specified incident energy e0 by a spectrometer that accepts scattering up to a
    specified collection semi-angle beta. It is based on the extended drude model
    (Section 3.3.2), with a volume energy-loss function elf in accord with Eq. (3.64) and
    a surface-scattering energy-loss function srelf as in Eq. (4.31). Retardation effects
    and coupling between the two surface modes are not included. The surface term can
    be made negligible by entering a large specimen thickness (tnm > 1000).
    Surface intensity srfint and volume intensity volint are calculated from
    Eqs. (4.31) and (4.26), respectively. The total spectral intensity ssd is written to
    the file DRUDE.SSD, which can be used as input for KRAKRO. These intensities are
    all divided by i0, to give relative probabilities (per eV). The real and imaginary parts
    of the dielectric function are written to DRUDE.EPS and can be used for comparison
    with the results of Kramers–Kronig analysis (KRAKRO.DAT).
    Written output includes the surface-loss probability Ps, obtained by integrating
    srfint (a value that relates to two surfaces but includes the negative begrenzungs
    term), for comparison with the analytical integration represented by Eq. (3.77). The
    volume-loss probability p_v is obtained by integrating volint and is used to calculate
    the volume plasmon mean free path (lam = tnm/p_v). The latter is listed and
    compared with the MFP obtained from Eq. (3.44), which represents analytical integration
    assuming a zero-width plasmon peak. The total probability (Pt = p_v+Ps) is
    calculated and used to evaluate the thickness (lam.Pt) that would be given by the formula
    t/λ = ln(It/i0), ignoring the surface-loss probability. Note that p_v will exceed
    1 for thicker specimens (t/λ > 1), since it represents the probability of plasmon
    scattering relative to that of no inelastic scattering.
    The command-line usage is drude(ep,ew,eb,epc,beta,e0,tnm,nn), where ep is the
    plasmon energy, ew the plasmon width, eb the binding energy of the electrons (0 for
    a metal), and nn is the number of channels in the output spectrum. An example of
    the output is shown in Fig. b.1a,b.

    """
    epc = dset.energy_scale[1] - dset.energy_scale[0]  # input('ev per channel : ');
    e0 = dset.metadata['acceleration_voltage'] / 1000.  # input('incident energy e0(kev) : ');
    # tnm = input('thickness(nm) : ');

    b = dset.metadata['collection_angle']  # rad
    epc = dset.energy_scale[1] - dset.energy_scale[0]  # input('ev per channel : ');
    e0 = dset.metadata['acceleration_voltage'] / 1000.  # input('incident energy e0(kev) : ');

    t = 1000.0 * e0 * (1. + e0 / 1022.12) / (1.0 + e0 / 511.06) ** 2  # eV # equ.5.2a or Appendix E p 427
    tgt = 1000 * e0 * (1022.12 + e0) / (511.06 + e0)  # eV  Appendix E p 427
    rk0 = 2590 * (1.0 + e0 / 511.06) * np.sqrt(2.0 * t / 511060)
    os = e[0]
    ew_mod = eb
    tags = dset.metadata
    # eps = 1 - ep**2/(e**2-eb**2+2*e*ew*1j)   # eq 3.64
    # eps = 1 - ep**2/(e**2+2*e*ew*1j)   # eq 3.64
    # eps = 1 - (ep**2)/(e**2+e*ew*1j-ep**2) # Lorentz Term
    eps = 1 - (ep ** 2 - ew_mod * e * 1j) / (e ** 2 + 2 * e * ew * 1j)  # Mod drude term
    # eps1 = np.real(eps);
    # eps2 = np.imag(eps);
    # eps1 = 1. - ep.^2./(e.^2+ew.^2);  #eq 3.40
    # eps2 = ew.*ep.^2./e./(e.^2+ew.^2);#eq 3.40
    # elf = ep**2*e*ew/((e**2-ep**2)**2+(e*ew)**2);    # eq 3.40?
    eps[np.nonzero(eps == 0.0)] = 1e-19
    elf = np.imag(-1 / eps)

    the = e / tgt  # varies with energy loss! # Appendix E p 427
    # srfelf = 4..*eps2./((1+eps1).^2+eps2.^2) - elf; %equivalent
    srfelf = np.imag(-4. / (1.0 + eps)) - elf  # for 2 surfaces
    angdep = np.arctan(b / the) / the - b / (b * b + the * the)
    srfint = angdep * srfelf / (3.1416 * 0.05292 * rk0 * t)  # probability per eV
    anglog = np.log(1.0 + b * b / the / the)
    i0 = tags['spec'].sum()  # *tags['counts2e']
    # print('counts2e',1/tags['counts2e'])

    # 2 * t = m_0 v**2 !!!  a_0 = 0.05292 nm
    volint = abs(tnm / (np.pi * 0.05292 * t * 2.0) * elf * anglog)  # S equ 4.26% probability per eV
    volint = volint * i0 / epc  # S probability per channel
    ssd = volint  # + srfint;

    if os < -1.0:
        xs = int(abs(-os / epc))

        ssd[0:xs] = 0.0
        volint[0:xs] = 0.0
        srfint[0:xs] = 0.0

        # if os <0:
        p_s = np.trapz(e, srfint)  # 2 surfaces but includes negative Begrenzung contribution.
        p_v = abs(np.trapz(e, abs(volint / tags['spec'].sum())))  # integrated volume probability
        p_v = (volint / i0).sum()  # our data have he same epc and the trapez formula does not include
        lam = tnm / p_v  # does NOT depend on free-electron approximation (no damping).
        lamfe = 4.0 * 0.05292 * t / ep / np.log(1 + (b * tgt / ep) ** 2)  # Eq.(3.44) approximation

        # print('p_s(2surfaces+begrenzung terms) =', p_s, 'p_v=t/lambda(beta)= ',p_v,'\n');
        # print('Volume-plasmon MFP(nm) = ', lam,' Free-electron MFP(nm) = ',lamfe,'\n');
        # print('--------------------------------\n');

        tags['eps'] = eps
        tags['lam'] = lam
        tags['lamfe'] = lamfe
        tags['p_v'] = p_v

    return ssd  # /np.pi


def effective_collection_angle(energy_scale, alpha, beta, beam_kv):
    """Calculates the effective collection angle in mrad:

    Translate from original Fortran program
    Calculates the effective collection angle in mrad:
    Parameter
    ---------
    energy_scale: numpy array
        first and last energy loss of spectrum in eV
    alpha: float
        convergence angle in mrad
    beta: float
        collection  angle in mrad
    beamKV: float
        acceleration voltage in V

    Returns
    -------
    eff_beta: float
        effective collection angle in mrad

    # function y = effbeta(ene,  alpha, beta, beam_kv)
    #
    #       This program computes etha(alpha,beta), that is the collection
    #       efficiency associated to the following geometry :
    #
    #       alpha = half angle of illumination  (0 -> pi/2)
    #       beta  = half angle of collection    (0 -> pi/2)
    #                                           (pi/2 = 1570.795 mrad)
    #
    #           A constant angular distribution of incident electrons is assumed
    #       for any incident angle (-alpha,alpha). These electrons imping the
    #       target and a single energy-loss event occurs, with a characteristic
    #       angle theta-e (relativistic). The angular distribution of the
    #       electrons after the target is analytically derived.
    #           This program integrates this distribution from theta=0 up to
    #       theta=beta with an adjustable angular step.
    #           This program also computes beta* which is the theoretical
    #       collection angle which would give the same value of etha(alpha,beta)
    #       with a parallel incident beam.
    #
    #       subroutines and function subprograms required
    #       ---------------------------------------------
    #       none
    #
    #       comments
    #       --------
    #
    #       The following parameters are asked as input :
    #        accelerating voltage (kV), energy loss range (eV) for the study,
    #        energy loss step (eV) in this range, alpha (mrad), beta (mrad).
    #       The program returns for each energy loss step :
    #        alpha (mrad), beta (mrad), theta-e (relativistic) (mrad),
    #        energy loss (eV), etha (#), beta * (mrad)
    #
    #       author :
    #       --------
    #       Pierre TREBBIA
    #       US 41 : "Microscopie Electronique Analytique Quantitative"
    #       Laboratoire de Physique des Solides, Bat. 510
    #       Universite Paris-Sud, F91405 ORSAY Cedex
    #       Phone : (33-1) 69 41 53 68
    #
    """
    if beam_kv == 0:
        beam_kv = 100.0

    if alpha == 0:
        return beta

    if beta == 0:
        return alpha

    z1 = beam_kv  # eV
    z2 = energy_scale[0]
    z3 = energy_scale[-1]
    z4 = 100.0

    z5 = alpha * 0.001  # rad
    z6 = beta * 0.001  # rad
    z7 = 500.0  # number of integration steps to be modified at will

    #       main loop on energy loss
    #
    for zx in range(int(z2), int(z3), int(z4)):  # ! zx = current energy loss
        eta = 0.0
        x0 = float(zx) * (z1 + 511060.) / (z1 * (z1 + 1022120.))  # x0 = relativistic theta-e
        x1 = np.pi / (2. * x0)
        x2 = x0 * x0 + z5 * z5
        x3 = z5 / x0 * z5 / x0
        x4 = 0.1 * np.sqrt(x2)
        dtheta = (z6 - x4) / z7
    #
    #        calculation of the analytical expression
    #
    for zi in range(1, int(z7)):
        theta = x4 + dtheta * float(zi)
        x5 = theta * theta
        x6 = 4. * x5 * x0 * x0
        x7 = x2 - x5
        x8 = np.sqrt(x7 * x7 + x6)
        x9 = (x8 + x7) / (2. * x0 * x0)
        x10 = 2. * theta * dtheta * np.log(x9)
        eta = eta + x10

    eta = eta + x2 / 100. * np.log(1. + x3)  # addition of the central contribution
    x4 = z5 * z5 * np.log(1. + x1 * x1)  # normalisation
    eta = eta / x4
    #
    #        correction by geometrical factor (beta/alpha)**2
    #
    if z6 < z5:
        x5 = z5 / z6
        eta = eta * x5 * x5

    etha2 = eta * 100.
    #
    #        calculation of beta *
    #
    x6 = np.power((1. + x1 * x1), eta)
    x7 = x0 * np.sqrt(x6 - 1.)
    beta = x7 * 1000.  # in mrad

    return beta


def kroeger_core(e_data, a_data, eps_data, ee, thick, relativistic=True):
    """This function calculates the differential scattering probability

     .. math::
        \\frac{d^2P}{d \\Omega d_e}
    of the low-loss region for total loss and volume plasmon loss

    Args:
       e_data (array): energy scale [eV]
       a_data (array): angle or momentum range [rad]
       eps_data (array): dielectric function data
       ee (float): acceleration voltage [keV]
       thick (float): thickness in m
       relativistic: boolean include relativistic corrections

    Returns:
       P (numpy array 2d): total loss probability
       p_vol (numpy array 2d): volume loss probability
    """

    # $d^2P/(dEd\Omega) = \frac{1}{\pi^2 a_0 m_0 v^2} \Im \left[ \frac{t\mu^2}{\varepsilon \phi^2 } \right] $ \

    # ee = 200 #keV
    # thick = 32.0# nm
    thick = thick * 1e-9  # input thickness now in m
    # Define constants
    # ec = 14.4;
    m_0 = constants.value(u'electron mass')  # REST electron mass in kg
    # h = constants.Planck  # Planck's constant
    hbar = constants.hbar

    c = constants.speed_of_light  # speed of light m/s
    bohr = constants.value(u'Bohr radius')  # Bohr radius in meters
    e = constants.value(u'elementary charge')  # electron charge in Coulomb
    print('hbar =', hbar, ' [Js] =', hbar / e, '[ eV s]')

    # Calculate fixed terms of equation
    va = 1 - (511. / (511. + ee)) ** 2  # ee is incident energy in keV
    v = c * np.sqrt(va)
    beta = v / c  # non-relativistic for =1

    if relativistic:
        gamma = 1. / np.sqrt(1 - beta ** 2)
    else:
        gamma = 1  # set = 1 to correspond to E+B & Siegle

    momentum = m_0 * v * gamma  # used for xya, E&B have no gamma

    # ##### Define mapped variables

    # Define independent variables E, theta
    a_data = np.array(a_data)
    e_data = np.array(e_data)
    [energy, theta] = np.meshgrid(e_data + 1e-12, a_data)
    # Define CONJUGATE dielectric function variable eps
    [eps, _] = np.meshgrid(np.conj(eps_data), a_data)

    # ##### Calculate lambda in equation EB 2.3
    theta2 = theta ** 2 + 1e-15
    theta_e = energy * e / momentum / v
    theta_e2 = theta_e ** 2

    lambda2 = theta2 - eps * theta_e2 * beta ** 2  # Eq 2.3

    lambd = np.sqrt(lambda2)
    if (np.real(lambd) < 0).any():
        print(' error negative lambda')

    # ##### Calculate lambda0 in equation EB 2.4
    # According to Kröger real(lambda0) is defined as positive!

    phi2 = lambda2 + theta_e2  # Eq. 2.2
    lambda02 = theta2 - theta_e2 * beta ** 2  # eta=1 Eq 2.4
    lambda02[lambda02 < 0] = 0
    lambda0 = np.sqrt(lambda02)
    if not (np.real(lambda0) >= 0).any():
        print(' error negative lambda0')

    de = thick * energy * e / 2.0 / hbar / v  # Eq 2.5

    xya = lambd * de / theta_e  # used in Eqs 2.6, 2.7, 4.4

    lplus = lambda0 * eps + lambd * np.tanh(xya)  # eta=1 %Eq 2.6
    lminus = lambda0 * eps + lambd / np.tanh(xya)  # eta=1 %Eq 2.7

    mue2 = 1 - (eps * beta ** 2)  # Eq. 4.5
    phi20 = lambda02 + theta_e2  # Eq 4.6
    phi201 = theta2 + theta_e2 * (1 - (eps + 1) * beta ** 2)  # eta=1, eps-1 in E+B Eq.(4.7)

    # Eq 4.2
    a1 = phi201 ** 2 / eps
    a2 = np.sin(de) ** 2 / lplus + np.cos(de) ** 2 / lminus
    a = a1 * a2

    # Eq 4.3
    b1 = beta ** 2 * lambda0 * theta_e * phi201
    b2 = (1. / lplus - 1. / lminus) * np.sin(2. * de)
    b = b1 * b2

    # Eq 4.4
    c1 = -beta ** 4 * lambda0 * lambd * theta_e2
    c2 = np.cos(de) ** 2 * np.tanh(xya) / lplus
    c3 = np.sin(de) ** 2 / np.tanh(xya) / lminus
    c = c1 * (c2 + c3)

    # Put all the pieces together...
    p_coef = e / (bohr * np.pi ** 2 * m_0 * v ** 2)

    p_v = thick * mue2 / eps / phi2

    p_s1 = 2. * theta2 * (eps - 1) ** 2 / phi20 ** 2 / phi2 ** 2  # ASSUMES eta=1
    p_s2 = hbar / momentum
    p_s3 = a + b + c

    p_s = p_s1 * p_s2 * p_s3

    # print(p_v.min(),p_v.max(),p_s.min(),p_s.max())
    # Calculate P and p_vol (volume only)
    dtheta = a_data[1] - a_data[0]
    scale = np.sin(np.abs(theta)) * dtheta * 2 * np.pi

    p = p_coef * np.imag(p_v - p_s)  # Eq 4.1
    p_vol = p_coef * np.imag(p_v) * scale

    # lplus_min = e_data[np.argmin(np.real(lplus), axis=1)]
    # lminus_min = e_data[np.argmin(np.imag(lminus), axis=1)]

    p_simple = p_coef * np.imag(1 / eps) * thick / (
            theta2 + theta_e2) * scale  # Watch it eps is conjugated dielectric function

    return p, p * scale * 1e2, p_vol * 1e2, p_simple * 1e2  # ,lplus_min,lminus_min


def kroeger_core2(e_data, a_data, eps_data, acceleration_voltage_kev, thickness, relativistic=True):
    """This function calculates the differential scattering probability

     .. math::
        \\frac{d^2P}{d \\Omega d_e}
    of the low-loss region for total loss and volume plasmon loss

    Args:
       e_data (array): energy scale [eV]
       a_data (array): angle or momentum range [rad]
       eps_data (array) dielectric function
       acceleration_voltage_kev (float): acceleration voltage [keV]
       thickness (float): thickness in nm
       relativistic (boolean): relativistic correction

    Returns:
       P (numpy array 2d): total loss probability
       p_vol (numpy array 2d): volume loss probability

       return P, P*scale*1e2,p_vol*1e2, p_simple*1e2
    """

    # $d^2P/(dEd\Omega) = \frac{1}{\pi^2 a_0 m_0 v^2} \Im \left[ \frac{t\mu^2}{\varepsilon \phi^2 } \right]
    """
    # Internally everything is calculated in si units
    # acceleration_voltage_kev = 200 #keV
    # thick = 32.0*10-9 # m

    """
    a_data = np.array(a_data)
    e_data = np.array(e_data)
    # adjust input to si units
    wavelength = get_wave_length(acceleration_voltage_kev * 1e3)  # in m
    thickness = thickness * 1e-9  # input thickness now in m

    # Define constants
    # ec = 14.4;
    m_0 = constants.value(u'electron mass')  # REST electron mass in kg
    # h = constants.Planck  # Planck's constant
    hbar = constants.hbar

    c = constants.speed_of_light  # speed of light m/s
    bohr = constants.value(u'Bohr radius')  # Bohr radius in meters
    e = constants.value(u'elementary charge')  # electron charge in Coulomb
    # print('hbar =', hbar ,' [Js] =', hbar/e ,'[ eV s]')

    # Calculate fixed terms of equation
    va = 1 - (511. / (511. + acceleration_voltage_kev)) ** 2  # acceleration_voltage_kev is incident energy in keV
    v = c * np.sqrt(va)

    if relativistic:
        beta = v / c  # non-relativistic for =1
        gamma = 1. / np.sqrt(1 - beta ** 2)
    else:
        beta = 1
        gamma = 1  # set = 1 to correspond to E+B & Siegle

    momentum = m_0 * v * gamma  # used for xya, E&B have no gamma

    # ##### Define mapped variables

    # Define independent variables E, theta
    [energy, theta] = np.meshgrid(e_data + 1e-12, a_data)
    # Define CONJUGATE dielectric function variable eps
    [eps, _] = np.meshgrid(np.conj(eps_data), a_data)

    # ##### Calculate lambda in equation EB 2.3
    theta2 = theta ** 2 + 1e-15

    theta_e = energy * e / momentum / v  # critical angle

    lambda2 = theta2 - eps * theta_e ** 2 * beta ** 2  # Eq 2.3

    lambd = np.sqrt(lambda2)
    if (np.real(lambd) < 0).any():
        print(' error negative lambda')

    # ##### Calculate lambda0 in equation EB 2.4
    # According to Kröger real(lambda0) is defined as positive!

    phi2 = lambda2 + theta_e ** 2  # Eq. 2.2
    lambda02 = theta2 - theta_e ** 2 * beta ** 2  # eta=1 Eq 2.4
    lambda02[lambda02 < 0] = 0
    lambda0 = np.sqrt(lambda02)
    if not (np.real(lambda0) >= 0).any():
        print(' error negative lambda0')

    de = thickness * energy * e / (2.0 * hbar * v)  # Eq 2.5
    xya = lambd * de / theta_e  # used in Eqs 2.6, 2.7, 4.4

    lplus = lambda0 * eps + lambd * np.tanh(xya)  # eta=1 %Eq 2.6
    lminus = lambda0 * eps + lambd / np.tanh(xya)  # eta=1 %Eq 2.7

    mue2 = 1 - (eps * beta ** 2)  # Eq. 4.5
    phi20 = lambda02 + theta_e ** 2  # Eq 4.6
    phi201 = theta2 + theta_e ** 2 * (1 - (eps + 1) * beta ** 2)  # eta=1, eps-1 in E+b Eq.(4.7)

    # Eq 4.2
    a1 = phi201 ** 2 / eps
    a2 = np.sin(de) ** 2 / lplus + np.cos(de) ** 2 / lminus
    a = a1 * a2

    # Eq 4.3
    b1 = beta ** 2 * lambda0 * theta_e * phi201
    b2 = (1. / lplus - 1. / lminus) * np.sin(2. * de)
    b = b1 * b2

    # Eq 4.4
    c1 = -beta ** 4 * lambda0 * lambd * theta_e ** 2
    c2 = np.cos(de) ** 2 * np.tanh(xya) / lplus
    c3 = np.sin(de) ** 2 / np.tanh(xya) / lminus
    c = c1 * (c2 + c3)

    # Put all the pieces together...
    p_coef = e / (bohr * np.pi ** 2 * m_0 * v ** 2)

    p_v = thickness * mue2 / eps / phi2

    p_s1 = 2. * theta2 * (eps - 1) ** 2 / phi20 ** 2 / phi2 ** 2  # ASSUMES eta=1
    p_s2 = hbar / momentum
    p_s3 = a + b + c

    p_s = p_s1 * p_s2 * p_s3

    # print(p_v.min(),p_v.max(),p_s.min(),p_s.max())
    # Calculate P and p_vol (volume only)
    dtheta = a_data[1] - a_data[0]
    scale = np.sin(np.abs(theta)) * dtheta * 2 * np.pi

    p = p_coef * np.imag(p_v - p_s)  # Eq 4.1
    p_vol = p_coef * np.imag(p_v) * scale

    # lplus_min = e_data[np.argmin(np.real(lplus), axis=1)]
    # lminus_min = e_data[np.argmin(np.imag(lminus), axis=1)]

    p_simple = p_coef * np.imag(1 / eps) * thickness / (theta2 + theta_e ** 2) * scale
    # Watch it: eps is conjugated dielectric function

    return p, p * scale * 1e2, p_vol * 1e2, p_simple * 1e2  # ,lplus_min,lminus_min


##########################
# EELS Database
##########################


def read_msa(msa_string):
    """read msa formated file"""
    parameters = {}
    y = []
    x = []
    # Read the keywords
    data_section = False
    msa_lines = msa_string.split('\n')

    for line in msa_lines:
        if data_section is False:
            if len(line) > 0:
                if line[0] == "#":
                    try:
                        key, value = line.split(': ')
                        value = value.strip()
                    except ValueError:
                        key = line
                        value = None
                    key = key.strip('#').strip()

                    if key != 'SPECTRUM':
                        parameters[key] = value
                    else:
                        data_section = True
        else:
            # Read the data

            if len(line) > 0 and line[0] != "#" and line.strip():
                if parameters['DATATYPE'] == 'XY':
                    xy = line.replace(',', ' ').strip().split()
                    y.append(float(xy[1]))
                    x.append(float(xy[0]))
                elif parameters['DATATYPE'] == 'Y':
                    print('y')
                    data = [
                        float(i) for i in line.replace(',', ' ').strip().split()]
                    y.extend(data)
    parameters['data'] = np.array(y)
    if 'XPERCHAN' in parameters:
        parameters['XPERCHAN'] = str(parameters['XPERCHAN']).split(' ')[0]
        parameters['OFFSET'] = str(parameters['OFFSET']).split(' ')[0]
        parameters['energy_scale'] = np.arange(len(y)) * float(parameters['XPERCHAN']) + float(parameters['OFFSET'])
    return parameters


def get_spectrum_eels_db(formula=None, edge=None, title=None, element=None):
    """
    get spectra from EELS database
    chemical formula and edge is accepted.
    Could expose more of the search parameters
    """
    valid_edges = ['K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N4,5', 'O2,3', 'O4,5']
    if edge is not None and edge not in valid_edges:
        print('edge should be a in ', valid_edges)

    spectrum_type = None
    title = title
    author = None
    element = element
    min_energy = None
    max_energy = None
    resolution = None
    min_energy_compare = "gt"
    max_energy_compare = "lt",
    resolution_compare = "lt"
    max_n = -1
    monochromated = None
    order = None
    order_direction = "ASC"
    verify_certificate = True
    # Verify arguments

    if spectrum_type is not None and spectrum_type not in {'coreloss', 'lowloss', 'zeroloss', 'xrayabs'}:
        raise ValueError("spectrum_type must be one of \'coreloss\', \'lowloss\', "
                         "\'zeroloss\', \'xrayabs\'.")
    # valid_edges = ['K', 'L1', 'L2,3', 'M2,3', 'M4,5', 'N2,3', 'N4,5', 'O2,3', 'O4,5']

    params = {
        "type": spectrum_type,
        "title": title,
        "author": author,
        "edge": edge,
        "min_energy": min_energy,
        "max_energy": max_energy,
        "resolution": resolution,
        "resolution_compare": resolution_compare,
        "monochromated": monochromated,
        "formula": formula,
        'element': element,
        "min_energy_compare": min_energy_compare,
        "max_energy_compare": max_energy_compare,
        "per_page": max_n,
        "order": order,
        "order_direction": order_direction,
    }

    request = requests.get('http://api.eelsdb.eu/spectra', params=params, verify=True)
    # spectra = []
    jsons = request.json()
    if "message" in jsons:
        # Invalid query, EELSdb raises error.
        raise IOError(
            "Please report the following error to the HyperSpy developers: "
            "%s" % jsons["message"])
    reference_spectra = {}
    for json_spectrum in jsons:
        download_link = json_spectrum['download_link']
        # print(download_link)
        msa_string = requests.get(download_link, verify=verify_certificate).text
        # print(msa_string[:100])
        parameters = read_msa(msa_string)
        if 'XPERCHAN' in parameters:
            reference_spectra[parameters['TITLE']] = parameters
            print(parameters['TITLE'])
    print(f'found {len(reference_spectra.keys())} spectra in EELS database)')

    return reference_spectra
