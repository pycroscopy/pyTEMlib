"""
kinematic_scattering
Copyright by Gerd Duscher

The University of Tennessee, Knoxville
Department of Materials Science & Engineering

Sources:
   Scattering Theory:
   Zuo and Spence, "Advanced TEM", 2017

   Spence and Zuo, Electron Microdiffraction, Plenum 1992

   Atomic Form Factor:
       Kirkland: Advanced Computing in Electron Microscopy 2nd edition
       Appendix C

Units:
    everything is in SI units, except length which is given in Angstrom.

Usage:
    See the notebooks for examples of these routines

All the input and output is done through a ase.Atoms object and the dictionary in the info attribute
"""

import itertools

import numpy as np
import sidpy
import ase

from .basic import get_all_g_vectors
from .basic import make_pretty_labels
from .basic import get_metric_tensor, get_structure_factors
from .basic import gaussian
from .basic import get_rotation_matrix, ewald_sphere_center
from .basic import get_cylinder_coordinates, get_all_miller_indices

from ..utilities import get_wavelength

def get_all_reflections(atoms, hkl_max, sg_max=None, ewald_center=None, verbose=False):
    """ get all reflections """
    hkl  = get_all_miller_indices(hkl_max)
    g = np.dot(hkl, atoms.cell.reciprocal()) # all evaluated reciprocal lattice points
    g_norm = np.linalg.norm(g, axis=1)
    indices = np.argsort(g_norm)
    hkl = hkl[indices]
    g = g[indices]

    if sg_max is None:
        return hkl, g
    # Calculate excitation errors for all reciprocal lattice points
    ## Zuo and Spence, 'Adv TEM', 2017 -- Eq 3:14
    k0_magnitude = np.linalg.norm(ewald_center)
    excitation_error = ((k0_magnitude**2- np.linalg.norm(g - ewald_center, axis=1)**2)
                        / (2*k0_magnitude))

    # Determine reciprocal lattice points
    # with excitation errors less than the maximum allowed one: Sg_max
    reflections = abs(excitation_error) < sg_max

    sg = excitation_error[reflections]
    g_hkl = g[reflections]
    hkl = hkl[reflections]
    if verbose:
        print (f'Of the {len(g)} tested reciprocal lattice points',
               f'{len(g_hkl)} have an excitation error less than {sg_max*10:.2f} 1/nm')
    return hkl, g_hkl, sg


def get_allowed_reflections(structure, verbose=False):
    """ Calculate allowed reflections of a crystal structure"""

    if isinstance(structure, ase.Atoms):
        tags = structure.info['experimental']
        atoms = structure
    elif isinstance(structure, sidpy.Dataset):
        tags = structure.metadata['experiment']
        atoms = structure.structures['Structure_000']
    else:
        raise ValueError('Input must be ase.Atoms or sidpy.Dataset object')
    tags['reciprocal_unit_cell'] = atoms.cell.reciprocal()

    hkl, g_hkl = get_all_reflections(atoms, tags.get('hkl_max', 10))

    allowed, forbidden, structure_factors = sort_bragg(atoms, g_hkl)

    if verbose:
        print(f'Of the {hkl.shape[0]} possible reflection {allowed.sum()} are allowed.')

    # information of allowed reflections
    hkl_allowed = hkl[allowed][:]
    g_allowed = np.array(g_hkl)[allowed, :]
    f_allowed = structure_factors[allowed]
    g_norm_allowed = np.linalg.norm(g_allowed, axis=1)  # length of all vectors = 1/

    ind = np.argsort(g_norm_allowed)
    hkl_sorted = hkl_allowed[ind][:]
    f_sorted = f_allowed[ind]
    g_sorted = g_allowed[ind, :]
    print('here')
    return hkl_sorted, g_sorted, f_sorted


def sort_bragg(atoms, g_hkl):
    """ Sort """
    structure_factors = get_structure_factors(atoms, g_hkl)
    allowed = np.absolute(structure_factors) > 0.000001
    forbidden = np.logical_not(allowed)
    return allowed, forbidden, structure_factors


def get_reflection_families(hkl_sorted, g_sorted, f_sorted, verbose=False):
    """ Determine reflection families and multiplicity"""

    g_norm_sorted = np.linalg.norm(g_sorted, axis=1)  # length of all vectors = 1/A
    g_norm, multiplicity = np.unique(np.around(g_norm_sorted, decimals=5),
                                     return_counts=True)
    if verbose:
        print(f'Of the {len(g_norm_sorted)} allowed reflection {len(multiplicity)} ',
              'have unique distances.')

    reflections_d = []
    reflections_m = []
    reflections_f = []
    start = 0
    for i in range(len(g_norm)):
        end = start + multiplicity[i]
        hkl_max = np.argmax(hkl_sorted[start:end].sum(axis=1))
        reflections_d.append(g_norm_sorted[start])
        reflections_m.append(hkl_sorted[start + hkl_max])
        reflections_f.append(f_sorted[start])
        start = end

    if verbose:
        print('\n\n [hkl]  \t 1/d [1/nm] \t d [nm] \t F^2 ')
        for i, spot in enumerate(g_norm):
            print(f'{reflections_m[i]} \t {spot*10:.2f} \t         '+
                  f'{1 / spot/10:.4f} \t {np.real(reflections_f[i])**2:.2f} ')
    return reflections_m, reflections_f, g_norm, multiplicity

def ring_pattern_calculation(structure, verbose=False):
    """
    Calculate the ring diffraction pattern of a crystal structure

    Parameters
    ----------
    structure: ase.Atoms or sidpy.Dataset
        crystal structure
    verbose: verbose print-outs
        set to False
    Returns
    -------
    tags: dict
        dictionary with diffraction information added
    """
    # Check sanity
    if isinstance(structure, ase.Atoms):
        tags = structure.info['experimental']
        out_tags = structure.info.setdefault('Ring_Pattern', {})
        atoms = structure
    elif isinstance(structure, sidpy.Dataset):
        tags = structure.metadata['experiment']
        atoms = structure.structures['Structure_000']
        out_tags = structure.metadata.setdefault('diffraction', {}).setdefault('Ring_Pattern', {})
    else:
        raise ValueError('Input must be ase.Atoms or sidpy.Dataset object')
    # wavelength
    tags['wave_length'] = get_wavelength(tags['acceleration_voltage'], unit='A')

    tags['metric_tensor'] = get_metric_tensor(atoms.cell.array)
    # converts hkl to g vectors and back

    ##################################
    # Calculate Structure Factors
    #################################
    hkl_sorted, g_sorted, f_sorted = get_allowed_reflections(structure, verbose)

    reflections_m, reflections_f, g_norm, multiplicity = get_reflection_families(hkl_sorted,
                                                                                 g_sorted,
                                                                                 f_sorted,
                                                                                 verbose)
    out_tags['allowed'] = {'hkl': reflections_m,
                           'g norm': g_norm,
                           'structure_factor': reflections_f,
                           'multiplicity': multiplicity}
    # print(structure.info['diffraction']['Ring_Pattern'].keys())
    out_tags['profile_x'] = np.linspace(0, g_norm.max(), 2048)
    step_size = out_tags['profile_x'][1]
    intensity = np.zeros(2048)
    x_index = [(g_norm / step_size + 0.5).astype(int)]
    intensity[x_index] = np.array(np.real(reflections_f)) * np.array(np.real(reflections_f))
    out_tags['profile_y delta'] = intensity

    if tags.get('thickness',0) > 0:
        x = np.linspace(-1024, 1023, 2048) * step_size
        p = [0.0, 1, 2 / tags['thickness']]
        gauss = gaussian(x, p)
        intensity = np.convolve(np.array(intensity), np.array(gauss), mode='same')
    out_tags['profile_y'] = intensity

    # Make pretty labels
    hkl_allowed = reflections_m
    hkl_label = make_pretty_labels(hkl_allowed)
    out_tags['allowed']['label'] = hkl_label


def get_dynamically_activated(out_tags, verbose=False):
    """ Get dynamically activated forbidden reflections"""
    zolz_allowed = out_tags['allowed']['ZOLZ']
    hkl_allowed = out_tags['allowed']['hkl'][zolz_allowed]

    zolz_forbidden = out_tags['forbidden']['ZOLZ']
    hkl_forbidden = out_tags['forbidden']['hkl'][zolz_forbidden].tolist()
    indices = range(len(hkl_allowed))
    combinations = [list(x) for x in itertools.permutations(indices, 2)]

    dynamically_activated = np.zeros(len(hkl_forbidden), dtype=bool)
    for [i, j] in combinations:
        possible = (hkl_allowed[i] + hkl_allowed[j]).tolist()
        if possible in hkl_forbidden:
            dynamically_activated[hkl_forbidden.index(possible)] = True
    out_tags['forbidden']['dynamically_activated'] = dynamically_activated
    if verbose:
        print(f"Of the {len(hkl_forbidden)} forbidden reflection in ZOLZ ",
              f"{dynamically_activated.sum()} can be dynamically activated.")


def calculate_holz(dif):
    """ Calculate HOLZ lines (of allowed reflections)"""
    intensities = dif['allowed']['intensities']
    k_0 = dif['K_0']

    # Dynamic Correction
    # Equation Spence+Zuo 3.86a
    gamma_1 = - 1./(2.*k_0) * (intensities / (2.*k_0*dif['allowed']['excitation_error'])).sum()
    # Equation Spence+Zuo 3.84
    kg = k_0 - k_0*gamma_1/(dif['allowed']['g'][:, 2]+1e-15)
    kg[dif['allowed']['ZOLZ']] = k_0

    # Calculate angle between K0 and deficient cone vector
    # For dynamic calculations K0 is replaced by kg
    kg[:] = k_0

    g_allowed = dif['allowed']['g']
    d_theta = np.abs(g_allowed[:,0]-np.arcsin(g_allowed[:,2]/g_allowed[:,3]))

    # Calculate nearest point of HOLZ and Kikuchi lines
    g_closest = dif['allowed']['g'].copy()
    g_closest[:, 0] = d_theta
    g_closest[:, 1] = g_closest[:, 1] + np.pi  # Other side: negative x, y

    # calculate and save line in Hough space coordinates (distance and theta)
    g_excess = g_closest.copy()
    g_excess[:, 0] = g_closest[:, 0] + dif['allowed']['g'][:, 0]
    holz = dif['allowed']['HOLZ']
    laue_zones = dif['allowed']['Laue_Zone'][holz]
    dif['HOLZ'] = {}
    dif['HOLZ']['g_deficient'] = g_closest[holz]
    dif['HOLZ']['g_excess'] = g_excess[holz]
    dif['HOLZ']['FOLZ'] = laue_zones == 1
    dif['HOLZ']['SOLZ'] = laue_zones == 2
    dif['HOLZ']['HOLZ_plus'] = laue_zones > 2  # even higher HOLZ
    dif['HOLZ']['Laue_zones'] = laue_zones
    dif['HOLZ']['hkl'] = dif['allowed']['hkl'][holz]
    dif['HOLZ']['intensities'] = intensities[holz]

    return dif


def get_bragg_reflections(atoms, in_tags, verbose=False):
    """ sort reflection in allowed and forbidden"""

    zone_hkl = in_tags.get('zone_hkl', None)
    if zone_hkl is None:
        raise ValueError('zone_hkl must be provided in tags')
    hkl_max = in_tags.setdefault('hkl_max', 10)

    sg_max = in_tags.setdefault('Sg_max', 0.03)  # 1/Ang  maximum allowed excitation error
    acceleration_voltage = in_tags.setdefault('acceleration_voltage', 100e3)
    mistilt_alpha = in_tags.setdefault('mistilt_alpha', 0)
    mistilt_beta = in_tags.setdefault('mistilt_beta', 0)

    # rotate in zone axis
    g_non_rot, hkl_all = get_all_g_vectors(hkl_max, atoms)

    ewald_center = ewald_sphere_center(acceleration_voltage, atoms, zone_hkl)
    k0_magnitude = np.linalg.norm(ewald_center)
    laue_circle = [mistilt_alpha, -mistilt_beta/np.pi*2.1]

    s = (k0_magnitude**2-np.linalg.norm(g_non_rot - ewald_center, axis=1)**2)/(2*k0_magnitude)
    reflections = np.abs(s)<sg_max

    g = g_non_rot[reflections]
    hkl = np.array(hkl_all[reflections])
    sg = s[reflections]
    # Weiss Zone Law
    laue_zone = np.sum(hkl * zone_hkl, axis=1)

    out_tags = {}

    # Do we have a mistilt
    if np.linalg.norm(laue_circle) > 0:
        print('mistilt')
        # Kikuchi first
        zolz = laue_zone == 0
        g_zolz = g[laue_zone == 0]
        hkl_zolz = hkl[laue_zone == 0]
        structure_factors_kikuchi = get_structure_factors(atoms, g_zolz)
        allowed = np.abs(structure_factors_kikuchi) > 0.000001
        print(allowed.sum(), len(allowed))
        g_kikuchi = get_cylinder_coordinates (zone_hkl, g_zolz, k0_magnitude)
        out_tags['Kikuchi'] = {'g': g_kikuchi[allowed],
                               'hkl': hkl_zolz[allowed],
                               'intensities': structure_factors_kikuchi[allowed],
                               'excitation_error': sg[zolz][allowed],
                               'Laue_circle': laue_circle}
        # add mistilt
        mistilt_matrix = get_rotation_matrix([mistilt_beta, mistilt_alpha,0], in_radians=True)
        g_rotated = np.dot(g_non_rot, mistilt_matrix)

        s = (k0_magnitude**2-np.linalg.norm(g_rotated - ewald_center, axis=1)**2)/(2*k0_magnitude)
        reflections = np.abs(s)<sg_max

        g = g_rotated[reflections]
        hkl = np.array(hkl_all[reflections])
        sg = s[reflections]
        # Weiss Zone Law of new hkl
        laue_zone = np.sum(hkl * zone_hkl, axis=1)

    structure_factors = get_structure_factors(atoms, g)
    allowed = np.abs(structure_factors) > 0.000001
    forbidden = np.logical_not(allowed)

    f_allowed = structure_factors[allowed]
    g_angles = get_cylinder_coordinates (zone_hkl, g, k0_magnitude)

    if verbose:
        print(f'Of the {hkl.shape[0]} possible reflection {allowed.sum()} are allowed.')

    zolz = laue_zone[allowed] == 0
    out_tags['allowed'] = {'hkl': hkl[allowed][:],
                           'g': g_angles[allowed, :],
                           'excitation_error': sg[allowed],
                           'intensities': structure_factors[allowed],
                           'Laue_Zone': laue_zone[allowed],
                           'ZOLZ': laue_zone[allowed] == 0,
                           'FOLZ': laue_zone[allowed] == 1,
                           'SOLZ': laue_zone[allowed] == 2,
                           'HOLZ': laue_zone[allowed] > 0,
                           'HOLZ_plus': laue_zone[allowed] > 2}
    out_tags['forbidden'] = {'hkl':  hkl[forbidden][:],
                             'g':  g_angles[forbidden, :],
                             'Laue_Zone': laue_zone[forbidden],
                             'ZOLZ': laue_zone[forbidden] == 0,
                             'HOLZ': laue_zone[forbidden] > 0}
    out_tags.update({'K_0': k0_magnitude,
                     'Laue_zone': laue_zone,
                     'aue_circle': laue_circle,
                     'allowed_all': allowed})
    # Calculate Intensity of beams  Reimer 7.25
    thickness = in_tags.setdefault('thickness', 0.0)
    if thickness > 0.1:
        # Calculate Extinction Distance  Reimer 7.23
        # - makes only sense for non-zero structure_factor
        xi_g = np.real(np.pi * atoms.cell.volume * k0_magnitude / f_allowed)
        s_eff = np.sqrt(sg[allowed]**2 + xi_g**-2)

        i_g = np.real(np.pi**2 / xi_g**2 * np.sin(np.pi * s_eff * thickness)**2
                      / (np.pi * s_eff)**2)
        out_tags['allowed']['Ig'] = i_g
        out_tags['thickness'] = thickness

    out_tags['parameters'] = in_tags
    if 'Kikuchi' not in out_tags:
        print('make_kikuchi')
        g_kikuchi = g_angles[allowed][zolz]
        g_kikuchi[:,0] /=2
        out_tags['Kikuchi'] = {'hkl': hkl[allowed][zolz],
                               'g': g_kikuchi,
                               'Laue_circle': laue_circle,
                               'intensities': structure_factors[allowed][zolz]}
    if verbose:
        print (f'Of those, there are {out_tags["allowed"]["ZOLZ"].sum()} in ZOLZ',
               f' and {out_tags["allowed"]["HOLZ"].sum()} in HOLZ')

    calculate_holz(out_tags)
    get_dynamically_activated(out_tags, verbose=verbose)
    return out_tags


def center_of_laue_circle(atoms, tags):
    """ Center of Laue circle in microscope coordinates"""
    k_0 = tags['incident_wave_vector']
    laue_circle = np.dot(tags['nearest_zone_axis'], tags['reciprocal_unit_cell'])
    laue_circle = np.dot(laue_circle, tags['rotation_matrix'])
    laue_circle = laue_circle / np.linalg.norm(laue_circle) * k_0
    laue_circle[2] = 0

    atoms.info.setdefault('diffraction', {})['Laue_circle'] = laue_circle
