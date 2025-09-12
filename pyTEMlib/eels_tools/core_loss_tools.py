"""
#################################################################
# Core-Loss functions
#################################################################
of eels_tools
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

Update by Austin Houston, UTK 12-2023 : Parallization of spectrum images
"""
from typing import Union
import numpy as np

import scipy
import sidpy

from ..utilities import major_edges, all_edges, elements
from ..utilities import effective_collection_angle
from ..utilities import get_z, get_x_sections, second_derivative



def list_all_edges(z: Union[str, int]=0, verbose=False)->list[str, dict]:
    """List all ionization edges of an element with atomic number z

    Parameters
    ----------
    z: int
        atomic number
    verbose: bool, optional
        more info if set to True

    Returns
    -------
    out_string: str
        string with all major edges in energy range
    """
    x_section = get_x_sections(get_z(z))
    out_string = ''
    if verbose:
        print('Major edges')
    element = x_section.get('name', None)
    edge_list = {element: {}}

    for key in all_edges:
        onset = x_section.get(key, {}).get('onset', None)
        if onset is None:
            continue
        out = f" {element}-{key}: {onset:8.1f} eV "
        if verbose:
            print(out)
        out_string = out_string + f"{out} /n"
        edge_list[element][key] = onset
    return out_string, edge_list


def find_all_edges(edge_onset: float,
                   maximal_chemical_shift: float=5.0,
                   major_edges_only: bool=False) -> str:
    """Find all (major and minor) edges within an energy range

    Parameters
    ----------
    edge_onset: float
        approximate energy of ionization edge
    maximal_chemical_shift: float, default = 5eV
        range of energy window around edge_onset to look for major edges
    major_edges_only: boolean, default = False
        only major edges are considered if True
    Returns
    -------
    text: str
        string with all edges in energy range

    """

    out_text = ''
    for z in np.arange(1, 93):
        x_section = get_x_sections(z)
        name = x_section.get('name', '')
        for key in x_section:
            if not isinstance(x_section[key], dict):
                continue
            onset = x_section[key].get('onset', 0)
            if abs(onset - edge_onset)  > maximal_chemical_shift:
                continue
            if major_edges_only:
                if key in major_edges:
                    out_text += f"\n {name:2s}-{key}: {onset:8.1f} eV "
            else:
                out_text += f"\n {name:2s}-{key}: {onset:8.1f} eV "
    return out_text


def find_associated_edges(dataset: sidpy.Dataset) -> None:
    """Find edges associated with peaks in the dataset"""
    onsets = []
    core_loss = dataset.metadata.get('core_loss', {}).get('edges', {})
    for key, edge in core_loss.items():
        if key.isdigit():
            onsets.append(edge['onset'])
            core_loss[key]['associated_peaks'] = {}
        peaks = dataset.metadata['peak_fit'].get('peaks', [])
        for key, peak in enumerate(peaks):
            distances  = (onsets-peak[0]) * -1
            distances[distances < -0.3] = 1e6
            if np.min(distances) < 50:
                index = np.argmin(distances)
                core_loss[str(index)]['associated_peaks'][key] = peak


def find_white_lines(dataset: sidpy.Dataset) -> Union[None, dict]:
    """Find white lines in the dataset"""
    white_lines_out ={'sum': {}, 'ratio': {}}
    white_lines = []
    peaks = dataset.metadata.get('peak_fit', {}).get('peaks', [])
    core_loss = dataset.metadata.get('core_loss', {})
    for index, edge in core_loss.get('edges', {}).items():
        if not index.isdigit():
            continue
        peaks = edge.get('associated_peaks', {})
        if edge['symmetry'][-2:] == 'L3' and 'L3' in edge['all_edges']:
            onset_l3 = edge['all_edges']['L3']['onset']
            onset_l2 = edge['all_edges']['L2']['onset']
            end_range1 = onset_l2 + edge['chemical_shift']
            end_range2 = onset_l2*2 - onset_l3 + edge['chemical_shift']
            white_lines = ['L3', 'L2']
        elif edge['symmetry'][-2:] == 'M5' and 'M5' in edge['all_edges']:
            onset_m5 = edge['all_edges']['M5']['onset']
            onset_m4 = edge['all_edges']['M4']['onset']
            end_range1 = onset_m4 + edge['chemical_shift']
            end_range2 = onset_m4*2 - onset_m5 + edge['chemical_shift']
            white_lines = ['M5', 'M4']
        else:
            continue
        white_line_areas = [0., 0.]
        for key, peak in peaks.items():
            if not str(key).isdigit():
                continue
            area = np.sqrt(2 * np.pi) * peak[1] * np.abs(peak[2]/np.sqrt(2 * np.log(2)))
            if peak[0] < end_range1:
                white_line_areas[0] += area
            elif peak[0] < end_range2:
                white_line_areas[1] += area

        edge['white_lines'] = {white_lines[0]: white_line_areas[0],
                               white_lines[1]: white_line_areas[1]}
        reference_counts = edge['areal_density'] * core_loss['xsections'][int(index)].sum()
        key = f"{edge['element']}-{white_lines[0]}+{white_lines[1]}"
        white_lines_out['sum'][key] = (white_line_areas[0] + white_line_areas[1])/reference_counts
        key = f"{edge['element']}-{white_lines[0]}/{white_lines[1]}"
        white_lines_out['ratio'][key] = white_line_areas[0] / white_line_areas[1]
    return white_lines_out


def find_edges(dataset: sidpy.Dataset) -> None:
    """find edges within a sidpy.Dataset"""

    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values

    second_dif, noise_level = second_derivative(dataset)

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
            edges = find_all_edges(energy_scale[peak], 20, major_edges_only=True)
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


def assign_likely_edges(edge_channels: Union[list, np.ndarray], energy: np.ndarray):
    """Assign likely edges to energy channels"""
    edges_in_list = []
    result = {}
    for channel in edge_channels:
        if channel not in edge_channels[edges_in_list]:
            shift = 5
            element_list = find_all_edges(energy[channel], maximal_chemical_shift=shift,
                                          major_edges_only=True)
            while len(element_list) < 1:
                shift += 1
                element_list = find_all_edges(energy[channel], maximal_chemical_shift=shift,
                                              major_edges_only=True)
            if len(element_list) > 1:
                while len(element_list) > 0:
                    shift-=1
                    element_list = find_all_edges(energy[channel], maximal_chemical_shift=shift,
                                                  major_edges_only=True)
                element_list = find_all_edges(energy[channel], maximal_chemical_shift=shift+1,
                                              major_edges_only=True)
            element = (element_list[:4]).strip()
            z = get_z(element)
            result[element] =[]
            _, edge_list = list_all_edges(z)

            for edge in edge_list.values():
                possible_minor_edge = np.argmin(np.abs(energy[edge_channels]-edge))
                if np.abs(energy[edge_channels[possible_minor_edge]]-edge) < 3:
                    edges_in_list.append(possible_minor_edge)
                    result[element].append(edge)
    return result


def auto_id_edges(dataset):
    """Automatically identifies edges in a dataset"""
    edge_channels = identify_edges(dataset)
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values
    found_edges = assign_likely_edges(edge_channels, energy_scale)
    return found_edges


def identify_edges(dataset: sidpy.Dataset, noise_level: float=2.0):
    """
    Using first derivative to determine edge onsets
    Any peak in first derivative higher than noise_level times standard deviation will be considered
    
    Parameters
    ----------
    dataset: sidpy.Dataset
        the spectrum
    noise_level: float
        ths number times standard deviation in first derivative decides 
        on whether an edge onset is significant
        
    Return
    ------
    edge_channel: numpy.ndarray
    
    """
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0]
    dispersion = energy_scale.slope

    spec = scipy.ndimage.gaussian_filter(dataset, 3/dispersion)  # smooth with 3eV wideGaussian

    first_derivative = spec - np.roll(spec, +2)
    first_derivative[:3] = 0
    first_derivative[-3:] = 0

    # find if there is a strong edge at high energy_scale
    noise_level = noise_level*np.std(first_derivative[3:50])
    [edge_channels, _] = scipy.signal.find_peaks(first_derivative, noise_level)
    return edge_channels


def add_element_to_dataset(dataset: sidpy.Dataset, z: Union[int, str]):
    """Adds an element to the dataset"""
    # We check whether this element is already in the
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0]

    zz = get_z(z)
    if 'edges' not in dataset.metadata:
        dataset.metadata['edges'] = {'model': {}, 'use_low_loss': False}
    index = 0
    for key, edge in dataset.metadata['edges'].items():
        if not key.isdigit():
            continue
        index += 1
        if zz == edge.get('z', ''):
            index = int(key)
            break

    major_edge = ''
    minor_edge = ''
    all_edges2 = {}
    x_section = get_x_sections(zz)
    edge_start = 10  # int(15./ft.get_slope(self.energy_scale)+0.5)
    for key in x_section:
        if len(key) == 2 and key[0] in ['K', 'L', 'M', 'N', 'O'] and key[1].isdigit():
            if energy_scale[edge_start] < x_section[key]['onset'] < energy_scale[-edge_start]:
                if key in ['K1', 'L3', 'M5', 'M3']:
                    major_edge = key
                all_edges2[key] = {'onset': x_section[key]['onset']}

    if major_edge != '':
        key = major_edge
    elif minor_edge != '':
        key = minor_edge
    else:
        print(f'Could not find no edge of {zz} in spectrum')
        return False

    edge = dataset.metadata['edges'].setdefault(str(index), {})

    start_exclude = x_section[key]['onset'] - x_section[key]['excl before']
    end_exclude = x_section[key]['onset'] + x_section[key]['excl after']

    edge.update({'z': zz, 'symmetry': key, 'element': elements[zz],
                 'onset': x_section[key]['onset'], 'end_exclude': end_exclude,
                 'start_exclude': start_exclude})
    edge['all_edges'] = all_edges2
    edge['chemical_shift'] = 0.0
    edge['areal_density'] = 0.0
    edge['original_onset'] = edge['onset']
    return True


def make_edges(edges_present: dict, energy_scale: np.ndarray, e_0:float,
               coll_angle:float, low_loss:np.ndarray=None)->dict:
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

    for key, edge in edges.items():
        xsec = x_sections[str(edge['z'])]
        if 'chemical_shift' not in edge:
            edge['chemical_shift'] = 0
        if 'symmetry' not in edge:
            edge['symmetry'] = 'K1'
        if 'K' in edge['symmetry']:
            edge['symmetry'] = 'K1'
        elif 'L' in edge['symmetry']:
            edge['symmetry'] = 'L3'
        elif 'M' in edge['symmetry']:
            edge['symmetry'] = 'M5'
        else:
            edge['symmetry'] = edge['symmetry'][0:2]

        edge['original_onset'] = xsec[edge['symmetry']]['onset']
        edge['onset'] = edge['original_onset'] + edge['chemical_shift']
        edge['start_exclude'] = edge['onset'] - xsec[edge['symmetry']]['excl before']
        edge['end_exclude'] = edge['onset'] + xsec[edge['symmetry']]['excl after']

    edges = make_cross_sections(edges, energy_scale, e_0, coll_angle, low_loss)
    return edges


def auto_chemical_composition(dataset:sidpy.Dataset)->None:
    """Automatically identifies edges in a dataset and adds them to the core_loss dictionary"""
    found_edges = auto_id_edges(dataset)
    for key in found_edges:
        add_element_to_dataset(dataset, key)
    fit_dataset(dataset)


def make_cross_sections(edges:dict, energy_scale:np.ndarray, e_0:float,
                        coll_angle:float, low_loss:np.ndarray=None)->dict:
    """
    Updates the edges dictionary with collection angle-integrated
    X-ray photo-absorption cross-sections
    """
    for key in edges:
        if str(key).isdigit():
            if edges[key]['z'] <1:
                break
            # from barnes to 1/nm^2
            edges[key]['data'] = xsec_xrpa(energy_scale, e_0 / 1000., edges[key]['z'], coll_angle,
                                           edges[key]['chemical_shift']) / 1e10
            if low_loss is not None:
                low_loss = np.roll(np.array(low_loss), 1024 - np.argmax(np.array(low_loss)))
                edges[key]['data'] = scipy.signal.convolve(edges[key]['data'],
                                                           low_loss/low_loss.sum(), mode='same')

            edges[key]['onset'] = edges[key]['original_onset'] + edges[key]['chemical_shift']
            edges[key]['X_section_type'] = 'XRPA'
            edges[key]['X_section_source'] = 'pyTEMlib'

    return edges


def power_law(energy: np.ndarray, a:float, r:float)->np.ndarray:
    """power law for power_law_background"""
    return a * np.power(energy, -r)


def power_law_background(spectrum:np.ndarray, energy_scale:np.ndarray,
                         fit_area:list, verbose:bool=False):
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

    [p, _] = scipy.optimize.leastsq(bgdfit, p0, args=(y, x), maxfev=2000)

    background_difference = y - power_law(x, p[0], p[1])
    background_noise_level = std_dev = np.std(background_difference)
    if verbose:
        print(f'Power-law background with amplitude A: {p[0]:.1f} and exponent -r: {p[1]:.2f}')
        print(background_difference.max() / background_noise_level)

        print(f'Noise level in spectrum {std_dev:.3f} counts')

    # Calculate background over the whole energy scale
    background = power_law(energy_scale, p[0], p[1])
    return background, p


def cl_model(xx, pp, number_of_edges, xsec):
    """ core loss model for fitting"""
    yy = pp[0] *  xx**pp[1] +  pp[2] + pp[3]* xx + pp[4] * xx * xx
    for i in range(number_of_edges):
        pp[i+5] = np.abs(pp[i+5])
        yy = yy + pp[i+5] * xsec[i, :]
    return yy


def get_mask(energy_scale, edges):
    """ Create a mask for the fitting area"""
    mask = np.ones(len(energy_scale))
    edges.setdefault('fit_area', {})
    background_fit_start = edges.get('fit_area', {}).get('fit_start', 0)
    background_fit_end = edges.get('fit_area', {}).setdefault('fit_end', energy_scale[-1])
    if background_fit_start == 0:
        return mask
    edges['fit_area']['fit_end'] = background_fit_end

    start_bgd = np.searchsorted(energy_scale, background_fit_start)
    end_bgd = np.searchsorted(energy_scale, background_fit_end)
    # Determine fitting ranges and masks to exclude ranges

    mask[0:start_bgd] = 0.0
    mask[end_bgd:-1] = 0.0
    for key in edges:
        if not key.isdigit():
            continue
        start_exclude = np.searchsorted(energy_scale, edges[key]['start_exclude'])
        end_exclude = np.searchsorted(energy_scale, edges[key]['end_exclude'])
        if start_bgd+1 < start_exclude < end_bgd-2 and end_exclude < end_bgd:
            start_exclude = max (start_exclude, 2)
            mask[start_exclude:end_exclude] = 0.0
    return mask

def fit_edges2(spectrum, energy_scale, edges):
    """ Fit edges in a spectrum """
    mask = get_mask(energy_scale, edges)

    ########################
    # Background Fit
    ########################
    bgd_fit_area = [edges['fit_area']['fit_start'], edges['fit_area']['fit_end']]
    _, [amplitude, r] = power_law_background(spectrum, energy_scale, bgd_fit_area, verbose=False)

    #######################
    # Edge Fit
    #######################

    blurred = scipy.ndimage.gaussian_filter(spectrum, sigma=5)
    blurred[np.where(blurred < 1e-8)] = 1e-8

    xsec = []
    number_of_edges = 0
    for key in edges:
        if key.isdigit():
            xsec.append(edges[key]['data'])
            number_of_edges += 1
    xsec = np.array(xsec)

    def model(xx, pp):
        yy = pp[0] *  xx**pp[1] +  pp[2] + pp[3] * xx + pp[4] * xx**2
        for i in range(number_of_edges):
            pp[i+5] = np.abs(pp[i+5])
            yy = yy + pp[i+5] * xsec[i, :]
        return yy

    def residuals(pp, xx, yy):
        err = np.abs((yy - model(xx, pp)) * mask)  / np.sqrt(np.abs(yy))
        return err

    scale = blurred[100]
    pin = np.array([amplitude, -r, 10., 1., 0.00] + [scale/5] * number_of_edges)
    [p, _] = scipy.optimize.leastsq(residuals, pin, args=(energy_scale, blurred))

    for key in edges:
        if key.isdigit():
            edges[key]['areal_density'] = p[int(key)+5]
    # print(p)
    background = p[0] * np.power(energy_scale, -p[1])
    background += p[2] + energy_scale**p[3] + p[4]*energy_scale**2
    edges['model'] = {'background': background,
                      'background-poly_0': p[2],
                      'background-poly_1': p[3],
                      'background-poly_2': p[4],
                      'background-A': p[0],
                      'background-r': p[1],
                      'spectrum': model(energy_scale, p),
                      'blurred': blurred,
                      'mask': mask,
                      'fit_parameter': p,
                      'fit_area_start': edges['fit_area']['fit_start'],
                      'fit_area_end': edges['fit_area']['fit_end'],
                      'xsec': xsec}
    return edges

def fit_dataset(dataset: sidpy.Dataset):
    """Fit edges in a sidpy.Dataset"""
    energy_scale = dataset.get_spectral_dims(return_axis=True)[0].values
    dataset.metadata['edges'].setdefault('fit_area', {})
    dataset.metadata['edges']['fit_area'].setdefault('fit_start', energy_scale[50])
    dataset.metadata['edges']['fit_area'].setdefault('fit_end', energy_scale[-2])
    dataset.metadata['edges'].setdefault('use_low_loss', False)

    exp = dataset.metadata.get('experiment', {})
    alpha  = exp.get('convergence_angle', None)
    if alpha is None:
        raise ValueError('need a convergence_angle in experiment of metadata dictionary ')
    beta = exp.get('collection_angle', 0)
    beam_kv = exp.get('acceleration_voltage', 0)
    eff_beta = effective_collection_angle(energy_scale, alpha, beta, beam_kv)
    edges = make_cross_sections(dataset.metadata['edges'], energy_scale, beam_kv, eff_beta)
    dataset.metadata['edges'] = fit_edges2(dataset, energy_scale, edges)
    areal_density = []
    element_list = []
    for key in edges:
        if key.isdigit():  # only edges have numbers in that dictionary
            element_list.append(edges[key]['element'])
            areal_density.append(edges[key]['areal_density'])
    areal_density = np.array(areal_density)
    out_string = '\nRelative composition: \n'
    for i, element in enumerate(element_list):
        out_string += f'{element}: {areal_density[i] / areal_density.sum() * 100:.1f}%  '
    print(out_string)


def core_loss_model(energy_scale, pp, number_of_edges, xsec):
    """ core loss model from fitting parameters"""
    xx = np.array(energy_scale)
    yy = pp[0] *  xx**pp[1] +  pp[2] + pp[3]* xx + pp[4] * xx * xx
    for i in range(number_of_edges):
        pp[i+5] = np.abs(pp[i+5])
        yy = yy + pp[i+5] * xsec[i, :]
    return yy


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
            background_fit_end = min(region_tags[key]['start_x'], background_fit_end)

    fit_area = region_tags['fit_area']
    ########################
    # Background Fit
    ########################
    bgd_fit_area = [region_tags['fit_area']['start_x'], background_fit_end]
    background, [amplitude, r] = power_law_background(spectrum, energy_scale,
                                                      bgd_fit_area, verbose=False)

    #######################
    # Edge Fit
    #######################
    x = energy_scale
    blurred = scipy.ndimage.gaussian_filter(spectrum, sigma=5)

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
    pin = np.array([scale/5, scale/5, scale/5, scale/5, scale/5, scale/5, -scale/10, 1.0, 0.001])
    [p, _] = scipy.optimize.leastsq(residuals, pin, args=(x, y))

    for key in edges:
        if key.isdigit():
            edges[key]['areal_density'] = p[int(key) - 1]
    edges['model'] = {}
    edges['model']['background'] = background + p[6] + p[7] * x + p[8] * x**2
    edges['model']['background-poly_0'] = p[6]
    edges['model']['background-poly_1'] = p[7]
    edges['model']['background-poly_2'] = p[8]
    edges['model']['background-A'] = amplitude
    edges['model']['background-r'] = r
    edges['model']['spectrum'] = model(x, p)
    edges['model']['blurred'] = blurred
    edges['model']['mask'] = mask
    edges['model']['fit_parameter'] = p
    edges['model']['fit_area_sta' \
    'rt'] = fit_area['start_x']
    edges['model']['fit_area_end'] = fit_area['start_x'] + fit_area['width_x']
    return edges


def xsec_xrpa(energy_scale, e0, z, beta, shift=0):
    """ Calculate momentum-integrated cross-section for EELS 
        from X-ray photo-absorption cross-sections.

    X-ray photo-absorption cross-sections from NIST.
    Momentum-integrated cross-section for EELS according to 
        Egerton Ultramicroscopy 50 (1993) 13-28 equation (4)

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
    # ToDo: is there and error in the (gamma-1) factor at the las should be (1-1/gamma**2)
    g = 2 * np.log(gamma) - np.log((b**2 + theta_e**2) / (b**2 + theta_e**2 / gamma**2)) - (
        (1-1/gamma**2)) * b**2 / (b**2 + theta_e**2 / gamma**2)
    datxs = datxs * (a / enexs / t) * (np.log(1 + b**2 / theta_e**2) + g) / 1e8

    datxs = datxs * dispersion  # from per eV to per dispersion
    # coeff = splrep(enexs, datxs, s=0)  # now in areal density atoms / m^2
    xsec = np.zeros(len(energy_scale))
    # shift = 0# int(ek -onsetXRPS)#/dispersion
    # Linear instead of spline interpolation to avoid oscillations.
    lin = scipy.interpolate.interp1d(enexs, datxs, kind='linear')
    if energy_scale[0] < enexs[0]:
        start = np.searchsorted(energy_scale, enexs[0])+1
    else:
        start = 0
    xsec[start:] = lin(energy_scale[start:] - shift)

    return xsec
