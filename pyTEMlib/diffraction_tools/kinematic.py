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
import scipy
import sidpy

import ase

from .basic import check_sanity
from .basic import make_pretty_labels, get_all_miller_indices
from .basic import get_metric_tensor, get_structure_factors
from .basic import get_zone_rotation, gaussian, get_unit_cell
from .basic import output_verbose, form_factor
from .basic import get_rotation_matrix, ewald_sphere_center

from ..utilities import get_wavelength

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


def zone_rotation(zone, verbose=False):
    """ Get rotation matrix to zone axis"""
    #spherical coordinates of zone
    r = np.linalg.norm(zone)
    theta = np.arccos(zone[2]/r)
    if zone[0] < 0:
        theta = -theta
    if zone[0] == 0:
        phi= np.pi/2
    else: 
        phi = np.atan2(zone[1], zone[0])

    if verbose:
        print('Rotation theta ',np.degrees(theta),' phi ',np.degrees(phi))
    r = get_rotation_matrix([0, -theta, -phi], in_radians=True)
    return r.T, theta, phi


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


def get_incident_wave_vector(atoms, tags, verbose):
    """ Incident wave vector K0 in vacuum and material"""
    u0 = 0.0  # in (Ang)
    # atom form factor of zero reflection angle is the
    # inner potential in 1/A
    for atom in atoms:
        u0 += form_factor(atom.symbol, 0.0)

    e = scipy.constants.elementary_charge
    h = scipy.constants.Planck
    m0 = scipy.constants.electron_mass

    volume_unit_cell = atoms.cell.volume
    scattering_factor_to_volts = (h**2) * (1e10**2) / (2 * np.pi * m0 * e) * volume_unit_cell
    tags['inner_potential_V'] = u0 * scattering_factor_to_volts
    if verbose:
        print(f'The inner potential is {u0:.1f} V')

    # Calculating incident wave vector magnitude 'k0' in material
    wl = tags['wave_length']
    tags['incident_wave_vector_vacuum'] = 1 / wl

    tags['incident_wave_vector'] = np.sqrt(1 / wl**2 + u0/volume_unit_cell)  # 1/Ang
    return tags['incident_wave_vector']

def get_projection(atoms, zone_hkl, g_hkl, center_ewald, tags):
    """ Get projection of g_hkl points onto 2D detector plane"""
    tags['reciprocal_unit_cell'] = atoms.cell.reciprocal()
    tags['mistilt_alpha'] = 0.
    tags['mistilt_beta'] = 0.

    rotation_matrix = get_zone_rotation(tags)
    # rotation_matrix, theta, phi = zone_rotation(zone_vector)
    center_ewald = np.dot(center_ewald, rotation_matrix)
    g_hkl_rotated = np.dot(g_hkl, rotation_matrix)
    d_theta = get_d_theta(g_hkl_rotated, np.linalg.norm(center_ewald)) # as distance

    laue_distance =  np.sum(np.dot(atoms.cell.reciprocal(), rotation_matrix), axis=0)[2]

    center_ewald[:2] = 0.
    g_hkl_spherical = g_hkl_rotated + center_ewald
    r_spherical = np.linalg.norm(g_hkl_spherical, axis = 1)
    theta = np.acos(g_hkl_spherical[:, 2] / r_spherical)

    phi = np.atan2(g_hkl_spherical[:, 0], g_hkl_spherical[:, 1])
    r = np.tan(theta) * center_ewald[2]

    return np.stack((r, phi, g_hkl_rotated[:, 2], d_theta), axis=1), laue_distance


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

def calculate_holz(dif):
    """ Calculate HOLZ lines (of allowed reflections)"""
    intensities = dif['allowed']['intensities']
    k_0 = dif['K_0']
    g_norm_allowed = dif['allowed']['g'][:, 0]

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
    d_theta = np.abs(np.arcsin(g_norm_allowed/k_0/2.)
                     -np.arcsin(np.abs(g_allowed[:,2])/(g_norm_allowed+1e-15)))

    
    #calculate length of distance of deficient cone to K0 in ZOLZ plane
    
    gd_length =2*np.sin(d_theta.real/2 )*k_0
    
    # calculate length of distance of deficient cone to K0 in ZOLZ plane
    
    # Calculate nearest point of HOLZ and Kikuchi lines
    g_closest = dif['allowed']['g'].copy()
    g_closest[:, 0] = 2*np.sin(d_theta.real/2) * k_0
    g_closest[:, 1] = g_closest[:, 1] + np.pi  # Other side: negative x, y 

    # calculate and save line in Hough space coordinates (distance and theta)
    g_excess = dif['allowed']['g'].copy()
    g_excess[:, 0] = g_closest[:, 0] + dif['allowed']['g'][:, 0]
    #holz = dif['allowed']['HOLZ']
    dif['HOLZ'] = {}
    dif['HOLZ']['g_deficient'] = g_closest
    dif['HOLZ']['g_excess'] = g_excess
    dif['HOLZ']['ZOLZ'] = dif['allowed']['ZOLZ']
    dif['HOLZ']['FOLZ'] = dif['allowed']['FOLZ']
    dif['HOLZ']['SOLZ'] = dif['allowed']['SOLZ']
    dif['HOLZ']['HOLZ_plus'] = dif['allowed']['HOLZ_plus']  # even higher HOLZ
    dif['HOLZ']['hkl'] = dif['allowed']['hkl']
    dif['HOLZ']['intensities'] = intensities

    return dif

def get_d_theta(g_allowed, k_0):
    """ Calculate HOLZ lines (of allowed reflections)"""
    g_norm_allowed = np.linalg.norm(g_allowed[:, :3], axis=1)
    # Dynamic Correction
    # Equation Spence+Zuo 3.86a
    #gamma_1 = - 1./(2.*k_0) * (intensities / (2.*k_0*diff_dict['allowed']['excitation_error'])).sum()
    # Equation Spence+Zuo 3.84
    #kg = k_0 - k_0*gamma_1/(diff_dict['allowed']['g'][:, 2]+1e-15)
    #kg[diff_dict['allowed']['ZOLZ']] = k_0

    # Calculate angle between K0 and deficient cone vector
    # For dynamic calculations K0 is replaced by kg
    #kg[:] = k_0

    d_theta =(np.arcsin(g_norm_allowed/k_0/2.) 
              - np.arcsin(np.abs(g_allowed[:,2])/(g_norm_allowed+1e-15) ))
    return 2.0 * np.sin(d_theta.real/2) * k_0
     

def sort_bragg(atoms, g_hkl):
    """ Sort """
    structure_factors = get_structure_factors(atoms, g_hkl)
    allowed = np.absolute(structure_factors) > 0.000001
    forbidden = np.logical_not(allowed)
    return allowed, forbidden, structure_factors



def get_bragg_reflections(atoms, in_tags, verbose=False):
    """ sort reflection in allowed and forbidden"""

    zone_hkl = in_tags.get('zone_hkl', None)
    if zone_hkl is None:
        raise ValueError('zone_hkl must be provided in tags')
    hkl_max = in_tags.setdefault('hkl_max', 10)
    sg_max = in_tags.setdefault('Sg_max', 0.1)  # 1/Ang  maximum allowed excitation error
    acceleration_voltage = in_tags.setdefault('acceleration_voltage', 100e3)

    center_ewald = ewald_sphere_center(acceleration_voltage, atoms, zone_hkl)
    
    hkl, g_hkl, sg  = get_all_reflections(atoms, hkl_max, sg_max, center_ewald, verbose=verbose)
    
    g, laue_distance = get_projection(atoms, zone_hkl, g_hkl, center_ewald, in_tags)
    allowed, forbidden, structure_factors = sort_bragg(atoms, g_hkl)

    
    if verbose:
        print(f'Of the {hkl.shape[0]} possible reflection {allowed.sum()} are allowed.')

    laue_zone = np.round(g[:, 2] / laue_distance, 1)
    
    # thickness = tags['thickness']
    # if thickness > 0.1:
    #    i_g = np.real(np.pi ** 2 / xi_g**2 * np.sin(np.pi * thickness * sg[allowed])**2
    #                  / (np.pi * sg[allowed])**2)
    #    dif['allowed']['Ig'] = i_g

    out_tags = {'allowed': {'hkl': hkl[allowed][:],
                            'g': g[allowed, :],
                            'excitation_error': sg[allowed],
                            'intensities': structure_factors[allowed]**2,
                            'Laue_Zone': laue_zone[allowed],
                            'ZOLZ': laue_zone[allowed] == 0,
                            'FOLZ': laue_zone[allowed] == 1,
                            'SOLZ': laue_zone[allowed] == 2,
                            'HOLZ': laue_zone[allowed] > 0,
                            'HOLZ_plus': laue_zone[allowed] > 2},
                'forbidden':{'hkl':  hkl[forbidden][:],
                             'g':  g[forbidden, :],
                             'Laue_Zone': laue_zone[forbidden],
                            'ZOLZ': laue_zone[forbidden] == 0,
                            'HOLZ': laue_zone[forbidden] > 0,},
                'K_0': np.linalg.norm(center_ewald)}

    laue_zone_f = np.floor(abs(np.dot(hkl[forbidden], zone_hkl)))
    out_tags['forbidden']['Laue_Zone'] = laue_zone_f
    out_tags['forbidden']['ZOLZ'] = laue_zone_f == 0
    out_tags['forbidden']['HOLZ'] = laue_zone_f > 1

    if verbose:
        print (f'Of those, there are {out_tags["allowed"]["ZOLZ"].sum()} in ZOLZ',
               f' and {out_tags["allowed"]["HOLZ"].sum()} in HOLZ')

    calculate_holz(out_tags)
    get_dynamically_activated(out_tags, verbose=verbose)
    return out_tags

def find_sorted_bragg_reflections2(atoms, tags, verbose):
    """ Find and sort all Bragg reflections within excitation error Sg_max"""
    # #######################
    # Find all Miller indices whose reciprocal point lies near the Ewald sphere with radius k_0
    # within a maximum excitation error sg
    # #######################
    k_0 = tags['incident_wave_vector']
    sg_max = tags['Sg_max']  # 1/Ang  maximum allowed excitation error

    hkl_all = get_all_miller_indices(tags['hkl_max'])
    # all evaluated reciprocal_unit_cell points
    g_non_rot = np.dot(hkl_all, tags['reciprocal_unit_cell'])
    g_norm = np.linalg.norm(g_non_rot, axis=1)   # length of all vectors
    g = np.dot(g_non_rot, tags['rotation_matrix'])

    # #######################
    # Calculate excitation errors for all reciprocal_unit_cell points
    # #######################

    # Zuo and Spence, 'Adv TEM', 2017 -- Eq 3:14
    #excitation error sg = (k0^2 - |g - k0|^2 ) / 2k0
    excitation_error = (k_0**2-np.linalg.norm(g - tags['k0_vector'], axis=1)**2)/(2*k_0)

    # #######################
    # Determine reciprocal_unit_cell points with excitation error
    # less than the maximum allowed one: Sg_max
    # #######################

    # This is now a boolean array with True for all possible reflections
    reflections = abs(excitation_error) < sg_max

    sg = excitation_error[reflections]
    g_hkl = g[reflections]
    g_hkl_non_rot = g_non_rot[reflections]
    hkl = hkl_all[reflections]
    g_norm = g_norm[reflections]

    if verbose:
        print(f'Of the {len(g)} tested reciprocal_unit_cell points,'+
              f' {len(g_hkl)} have an excitation error less than {sg_max:.2f} 1/nm')

    # #################################
    # Calculate Structure Factors
    # ################################
    structure_factor = get_structure_factors(atoms, g_hkl_non_rot)

    # ###########################################
    # Sort reflection in allowed and forbidden #
    # ###########################################
    allowed = np.absolute(structure_factor) > 0.000001    # allowed within numerical erro
    if verbose:
        print(f'Of the {hkl.shape[0]} possible reflection {allowed.sum()} are allowed.')
    # information of allowed reflections
    f_allowed = structure_factor[allowed]

    atoms.info['diffraction'] = {}
    dif = atoms.info['diffraction']
    dif['allowed'] = {}
    dif['allowed']['sg'] = sg[allowed]
    dif['allowed']['hkl'] = hkl[allowed][:]
    dif['allowed']['g'] = g_hkl[allowed][:]
    dif['allowed']['g_non_rot'] = g_non_rot

    dif['allowed']['structure_factor'] = f_allowed

    # Calculate Extinction Distance  Reimer 7.23
    # - makes only sense for non-zero structure_factor
    xi_g = np.real(np.pi * atoms.cell.volume * k_0 / f_allowed)

    # ###########################
    # Calculate Intensities (of allowed reflections)
    # ###########################

    # Calculate Intensity of beams  Reimer 7.25
    if 'thickness' not in tags:
        tags['thickness'] = 0.
    thickness = tags['thickness']
    if thickness > 0.1:
        i_g = np.real(np.pi ** 2 / xi_g**2 * np.sin(np.pi * thickness * sg[allowed])**2
                      / (np.pi * sg[allowed])**2)
        dif['allowed']['Ig'] = i_g

    dif['allowed']['intensities'] = np.real(f_allowed)**2

    # information of forbidden reflections
    forbidden = np.logical_not(allowed)
    dif['forbidden'] = {}
    dif['forbidden']['sg'] = sg[forbidden]
    dif['forbidden']['hkl'] = hkl[forbidden]
    dif['forbidden']['g'] = g_hkl[forbidden]

def center_of_laue_circle(atoms, tags):
    """ Center of Laue circle in microscope coordinates"""
    k_0 = tags['incident_wave_vector']
    laue_circle = np.dot(tags['nearest_zone_axis'], tags['reciprocal_unit_cell'])
    laue_circle = np.dot(laue_circle, tags['rotation_matrix'])
    laue_circle = laue_circle / np.linalg.norm(laue_circle) * k_0
    laue_circle[2] = 0

    atoms.info.setdefault('diffraction', {})['Laue_circle'] = laue_circle


def calculate_laue_zones(atoms, tags, verbose):
    """ Calculate Laue Zones (of allowed reflections)

    ###########################
    Below is the expression given in most books.
    However, that would only work for orthogonal crystal systems
    Laue_Zone = abs(np.dot(hkl_allowed,tags['zone_hkl']))  
    works only for orthogonal systems

    Below expression works for all crystal systems
    Remember we have already tilted, and so the dot product is trivial
    and gives only the z-component.
    """
    dif = atoms.info['diffraction']
    length_zone_axis = np.linalg.norm(np.dot(tags['zone_hkl'], tags['unit_cell']))
    g_norm_allowed =  np.linalg.norm(dif['allowed']['g'])
    laue_zone = abs(np.dot(dif['allowed']['hkl'], tags['nearest_zone_axis']))
    dif['allowed']['Laue_Zone'] = laue_zone
    zolz_forbidden = abs(np.floor(dif['forbidden']['g'][:, 2]
                                  * length_zone_axis+0.5)) == 0
    dif['forbidden']['Laue_Zone'] = zolz_forbidden
    dif['allowed']['ZOLZ'] = laue_zone == 0
    dif['allowed']['FOLZ'] = laue_zone == 1
    dif['allowed']['SOLZ'] = laue_zone == 2
    dif['allowed']['HOLZ'] = laue_zone > 0
    dif['allowed']['HOLZ_plus'] = dif['allowed']['HHOLZ'] = laue_zone > 2

    if verbose:
        print(f' There are {(laue_zone == 0).sum()} allowed reflections in the ',
              "zero order Laue Zone")
        print(f' There are {(laue_zone == 1).sum()} allowed reflections in the ',
              "first order Laue Zone")
        print(f' There are {(laue_zone == 2).sum()} allowed reflections in the ',
              "second order Laue Zone")
        print(f' There are {(laue_zone > 2).sum()} allowed reflections in the ',
              "other higher order Laue Zones")
        print(f'Length of zone axis vector in real space {length_zone_axis:.3f} nm')

    if verbose == 2:
        print(' hkl  \t Laue zone \t Intensity (*1 and \t log) \t length \n')
        for i, hkl in enumerate(dif['allowed']['hkl']):
            print(f" {hkl} \t {laue_zone[i]} \t {dif['allowed']['intensities'][i]:.3f} \t",
                  f"  {np.log(dif['allowed']['intensities'][i]+1):.3f} ",
                  f"\t  {g_norm_allowed[i]:.3f}   ")


def get_dynamical_allowed(atoms, verbose):
    """ Determine which forbidden reflections can be dynamically activated"""
    dif = atoms.info['diffraction']
    hkl_allowed = dif['allowed']['hkl']
    hkl_forbidden = dif['forbidden']['hkl']
    zolz = dif['allowed']['ZOLZ']
    zolz_forbidden = dif['forbidden']['Laue_Zone']
    double_diffraction = (np.sum(np.array(list(itertools.combinations(
        hkl_allowed[zolz], 2))), axis=1))

    dynamical_allowed = []
    still_forbidden = []
    for i, hkl in enumerate(hkl_forbidden):
        if zolz_forbidden[i]:
            if hkl.tolist() in double_diffraction.tolist():
                dynamical_allowed.append(i)
            else:
                still_forbidden.append(i)
    dif['forbidden']['dynamically_activated'] = dynamical_allowed
    dif['forbidden']['forbidden'] = dynamical_allowed
    if verbose:
        print(f'There are {len(dynamical_allowed)} forbidden but',
              " dynamical activated diffraction spots:")
        # print(tags['forbidden']['hkl'][dynamical_allowed])


def calculate_holz2(atoms, tags):
    """ Calculate HOLZ lines (of allowed reflections)"""
    dif = atoms.info['diffraction']
    intensities = dif['allowed']['intensities']
    k_0 = tags['incident_wave_vector']
    g_norm_allowed = np.linalg.norm(dif['allowed']['g'], axis=1)

    # Dynamic Correction
    # Equation Spence+Zuo 3.86a
    gamma_1 = - 1./(2.*k_0) * (intensities / (2.*k_0*dif['allowed']['sg'])).sum()

    # Equation Spence+Zuo 3.84
    kg = k_0 - k_0*gamma_1/(dif['allowed']['g'][:, 2]+1e-15)
    kg[dif['allowed']['ZOLZ']] = k_0

    # Calculate angle between K0 and deficient cone vector
    # For dynamic calculations K0 is replaced by kg
    kg[:] = k_0
    
    d_theta = np.arcsin(g_norm_allowed/kg/2.) - (np.arcsin(np.abs(dif['allowed']['g'][:, 2])
                                                           /g_norm_allowed))

    # calculate length of distance of deficient cone to K0 in ZOLZ plane
    gd_length = 2*np.sin(d_theta/2) * k_0

    # Calculate nearest point of HOLZ and Kikuchi lines
    g_closest = dif['allowed']['g'].copy()
    g_closest = g_closest*(gd_length/np.linalg.norm(g_closest, axis=1))[:, np.newaxis]

    

    # calculate and save line in Hough space coordinates (distance and theta)
    slope = g_closest[:, 0]/(g_closest[:, 1]+1e-10)
    distance = gd_length
    theta = np.arctan2(dif['allowed']['g'][:, 0], dif['allowed']['g'][:, 1])

    dif['HOLZ'] = {}
    dif['HOLZ']['slope'] = slope
    # a line is now given by

    dif['HOLZ']['distance'] = distance
    dif['HOLZ']['theta'] = theta
    dif['HOLZ']['g_deficient'] = g_closest
    dif['HOLZ']['g_excess'] = g_closest + dif['allowed']['g']
    dif['HOLZ']['ZOLZ'] = dif['allowed']['ZOLZ']
    dif['HOLZ']['HOLZ'] = dif['allowed']['HOLZ']
    dif['HOLZ']['FOLZ'] = dif['allowed']['FOLZ']
    dif['HOLZ']['SOLZ'] = dif['allowed']['SOLZ']
    dif['HOLZ']['HHOLZ'] = dif['allowed']['HHOLZ']  # even higher HOLZ
    dif['HOLZ']['hkl'] = dif['allowed']['hkl']
    dif['HOLZ']['intensities'] = intensities

####################################
# Calculate HOLZ and Kikuchi Lines #
####################################
def calculate_kikuchi(atoms, tags, verbose):
    """ Calculate Kikuchi lines (of allowed reflections)"""
    tags_kikuchi = tags.copy()
    tags_kikuchi['mistilt_alpha'] = 0
    tags_kikuchi['mistilt_beta'] = 0
    dif = atoms.info['diffraction']
    k_0 = tags['incident_wave_vector']
    k0_vector = tags['k0_vector']

    for i in range(1):  # tags['nearest_zone_axes']['amount']):
        zone_tags = tags['nearest_zone_axes'][str(i)]
        tags_kikuchi['zone_hkl'] = zone_tags['hkl']
        if verbose:
            print('Calculating Kikuchi lines for zone: ', zone_tags['hkl'])
        tags_kikuchi['Laue_circle'] = dif['Laue_circle']
        # Rotate to nearest zone axis
        rotation_matrix = get_zone_rotation(tags_kikuchi)

        g_kikuchi_all = np.dot(dif['allowed']['g_non_rot'], rotation_matrix)

        zolz = abs(np.array(g_kikuchi_all)[:, 2]) < .1

        g_kikuchi = g_kikuchi_all[zolz]
        excitation_error = (k_0**2-np.linalg.norm(g_kikuchi - k0_vector, axis=1)**2)/(2*k_0)
        # This is now a boolean array with True for all possible reflections
        reflections = abs(excitation_error) < .01
        g_kikuchi = g_kikuchi[reflections]
        hkl_all = get_all_miller_indices(tags['hkl_max'])
        hkl_kikuchi = hkl_all[zolz][reflections]
        structure_factors = get_structure_factors(atoms, g_kikuchi)
        allowed_kikuchi = np.absolute(structure_factors) > 0.000001

        g_kikuchi = g_kikuchi[allowed_kikuchi]
        hkl_kikuchi = hkl_kikuchi[allowed_kikuchi]

        gd2 = g_kikuchi / 2.
        gd2[:, 2] = 0.

        # calculate and save line in Hough space coordinates (distance and theta)
        slope2 = gd2[:, 0] / (gd2[:, 1] + 1e-20)
        distance2 = np.sqrt(gd2[:, 0] * gd2[:, 0] + gd2[:, 1] * gd2[:, 1])
        theta2 = np.arctan(slope2)

        dif['Kikuchi'] = {}
        dif['Kikuchi']['slope'] = slope2
        dif['Kikuchi']['distance'] = distance2
        dif['Kikuchi']['theta'] = theta2
        dif['Kikuchi']['hkl'] = hkl_kikuchi
        dif['Kikuchi']['g_hkl'] = g_kikuchi
        dif['Kikuchi']['g_deficient'] = gd2
        dif['Kikuchi']['min_dist'] = gd2 + dif['Laue_circle']

def kinematic_scattering(atoms, verbose=False):
    """
        All kinematic scattering calculation

        Calculates Bragg spots, Kikuchi lines, excess, and deficient HOLZ lines

        Parameters
        ----------
        atoms: ase.Atoms
            object with crystal structure:
            and with experimental parameters in info attribute:
            'acceleration_voltage_V', 'zone_hkl', 'Sg_max', 'hkl_max'
            Optional parameters are:
            'mistilt', convergence_angle_mrad', and 'crystal_name'
            verbose = True will give extended output of the calculation
        verbose: boolean
            default is False

        Returns
        -------
        atoms:
            There are three sub_dictionaries in info attribute:
            ['allowed'], ['forbidden'], and ['HOLZ']
            ['allowed'] and ['forbidden'] dictionaries contain:
                ['sg'], ['hkl'], ['g'], ['structure_factor'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], ['HHOLZ'], ['label'], and ['Laue_zone']
            the ['HOLZ'] dictionary contains:
                ['slope'], ['distance'], ['theta'], ['g_deficient'], ['g_excess'], 
                ['hkl'], ['intensities'],
                ['ZOLZ'], ['FOLZ'], ['SOLZ'], ['HOLZ'], and  ['HHOLZ']
            Please note that the Kikuchi lines are the HOLZ lines of ZOLZ

            There are also a few parameters stored in the main dictionary:
                ['wave_length_nm'], ['reciprocal_unit_cell'], ['inner_potential_V'], 
                ['incident_wave_vector'],['volume'], ['theta'], ['phi'], 
                and ['incident_wave_vector_vacuum']
    """

    # Check sanity

    atoms.info.setdefault('output', {})
    atoms.info.setdefault('experimental', {})
    output = atoms.info['output']
    output['SpotPattern'] = True

    if not check_sanity(atoms):
        print('Input is not complete, stopping')
        print('Try \'example()\' for example input')
        return

    tags = atoms.info['experimental']

    tags['wave_length'] = get_wavelength(tags['acceleration_voltage_V'], unit='Å')

    # ###########################################
    # reciprocal_unit_cell
    # #########################################
    get_unit_cell(atoms, tags)

    # ##############################################
    # Incident wave vector k0 in vacuum and material
    # ##############################################

    k_0 = get_incident_wave_vector(atoms, tags, verbose)
    # Incident wave vector K0 in vacuum and material
    tags['convergence_angle_A-1'] = k_0*np.sin(tags['convergence_angle_mrad']/1000.)

    # ############
    # Rotate
    # ############

    # get rotation matrix to rotate zone axis onto z-axis
    get_zone_rotation(tags)

    # rotate incident wave vector
    k0_unit_vector = np.array([0, 0, 1])  # incident unit wave vector
    k0_vector = k0_unit_vector * k_0  # incident  wave vector
    tags['k0_vector'] = k0_vector
    dif = get_bragg_reflections(atoms, tags, verbose)

    # ##########################
    # Make pretty labels
    # ##########################
    # dif = atoms.info['diffraction']
    dif['allowed']['label'] = make_pretty_labels(dif['allowed']['hkl'])
    dif['forbidden']['label'] = make_pretty_labels(dif['forbidden']['hkl'])

    # Center of Laue Circle
    center_of_laue_circle(atoms, tags)

    if verbose:
        output_verbose(atoms, tags)

    # ###########################
    # Calculate Laue Zones (of allowed reflections)
    # ###########################
    calculate_laue_zones(atoms, tags, verbose)

    # ##########################
    # Dynamically Activated forbidden reflections
    # ##########################
    get_dynamical_allowed(atoms, verbose)

    # ###################################
    # Calculate HOLZ and Kikuchi Lines #
    # ###################################
    calculate_holz2(atoms, tags)

    ####################################
    # Calculate HOLZ and Kikuchi Lines #
    ####################################
    calculate_kikuchi(atoms, tags, verbose)

    if verbose:
        print('pyTEMlib\'s  \"kinematic_scattering\" finished')
