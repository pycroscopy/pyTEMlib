"""
Module for calculating electron ionization cross sections for EDS quantification.
Based on the Bote & Salvat (2008, 2009) cross sections for inner shell ionization
by electron impact.
Also includes cascade corrections for Coster-Kronig, Auger, and fluorescence
emission.
Values from xraylib package are used for fluorescence yields, radiative rates,
Coster-Kronig probabilities, and Auger yields/rates.
References:
- D. Bote and F. Salvat, "Calculations of inner-shell ionization by electron 
  impact with the distorted-wave and plane-wave Born approximations","
- Bote, David, et al. "Cross sections for ionization of K, L and M shells of 
  atoms by impact of electrons and positrons with energies up to 1 GeV: 
  Analytical formulas." 
  Atomic Data and Nuclear Data Tables 95.6 (2009): 871-909. 
- Browning, N. D., et al. "A model for calculating cross sections for electron
  impact ionization of atoms from threshold to 10 MeV." 
    Journal of Applied Physics 83.11 (1998): 5736-5744.
"""
import os
import json
import numpy as np
import scipy

from .config_dir import config_path
from .utilities import elements as elements_list
from .utilities import get_z


def get_atomic_number(z):
    """Returns the atomic number independent of input as a string or number"""
    return str(get_z(z))


def casnati_x_section(occupancy, edge_energy, beam_energy):
    """(Casnati's equation) was found to fit cross-section data to
    typically better than +-10% over the range 1<=Uk<=20 and 6<=Z<=79."
    Note: This result is for K shell. L & M are much less wellcharacterized.
    C. Powell indicated in conversation with Richie that he believed that Casnati's
    expression was the best available for L & M also.
    """
    rest_energy = 5.10998918e5  # electron rest energy in eV
    res = 0.0
    ee = edge_energy
    u = beam_energy / ee
    if u > 1.0:
        phi = 10.57 * np.exp((-1.736 / u) + (0.317 / u**2))
        psi = np.power(ee / scipy.constants.R, -0.0318 + (0.3160 / u) + (-0.1135 / u**2))
        i = ee / rest_energy
        t = beam_energy / rest_energy
        f = ((2.0 + i) / (2.0 + t)) * np.square((1.0 + t) / (1.0 + i))
        f *= np.power(((i + t) * (2.0 + t) * np.square(1.0 + i))
                       / ((t * (2.0 + t) * np.square(1.0 + i))
                          + (i * (2.0 + i))), 1.5)
        res = ((occupancy * np.sqrt((scipy.constants.value('Bohr radius')
                                     * scipy.constants.Rydberg) / ee)
               * f * psi * phi * np.log(u)) / u)
    return res


def browning_x_section(z, acceleration_energy):
    """
    Browning, N. D., et al. "A model for calculating cross sections for electron
    impact ionization of atoms from threshold to 10 MeV."
    Journal of Applied Physics 83.11 (1998): 5736-5744.

    Computes the total ionization cross section for energetic electrons.
    Parameters
    ----------
    z : int
        The atomic number z in the range 1:99
    acceleration_energy : float
        The kinetic energy of the incident electron in eV

    Returns
    -------
    float
        The ionization cross section in square meters
    """
    e = acceleration_energy
    x_section = (3.0e-22 * z**1.7) / (e + (0.005 * z**1.7 * np.sqrt(e))
                              + ((0.0007 * z**2) / np.sqrt(e)))
    return x_section

def bote_salvat_xsection(acceleration_voltage, ):
    """
    Using Tabulated Values for electrons of: 
    - D. Bote and F. Salvat, "Calculations of inner-shell ionization by electron 
      impact with the distorted-wave and plane-wave Born approximations", 
      Phys. Rev. A77, 042701 (2008).
    
    - Bote, David, et al. "Cross sections for ionization of K, L and M shells of 
      atoms by impact of electrons and positrons with energies up to 1 GeV: 
      Analytical formulas." 
      Atomic Data and Nuclear Data Tables 95.6 (2009): 871-909.

    computed by emtables: https://github.com/adriente/emtables
    with line=True option and stored in a json file in ~/.pyTEMlib directory
    Parameters
    ----------
    z : int
        The atomic number z in the range 1:99   
    subshell : str
        The atomic sub-shell being ionized K1, L₁, L₂, ..., M₅
    acceleration_energy : float
        The kinetic energy of the incident electron in eV
    

    Returns
    -------
    dict
        The ionization cross section in barns per line
    """
    filename = os.path.join(config_path, f'xrays_X_section_{int(acceleration_voltage/1000)}kV.json')

    return json.load(open(filename, 'r', encoding='utf-8'))


def get_bote_salvat_dict(acceleration_voltage, z=0):
    """ Get Bote and Salvat X-ray cross section dictionary."""
    filename = os.path.join(config_path, f'xrays_X_section_{int(acceleration_voltage/1000)}kV.json')
    # print('Loading cross sections from ', filename)
    with open(filename, 'r', encoding='utf-8') as file:
        x_sections = json.load(file)
    if z > 0:
        return x_sections['table'][str(z)]
    return x_sections


def get_families(spectrum):
    """Get the line families for all elements in the spectrum."""
    spectrum.metadata['EDS'].setdefault('GUI', {})
    for key in spectrum.metadata['EDS']:
        if key in ['detector', 'quantification']:
            pass
        elif isinstance(spectrum.metadata['EDS'][key], dict) and key in elements_list:
            family = spectrum.metadata['EDS'][key].get('GUI', {}).get('symmetry', None)
            if family is None:
                if 'K-family' in spectrum.metadata['EDS'][key]:
                    family = 'K-family'
                elif 'L-family' in spectrum.metadata['EDS'][key]:
                    family = 'L-family'
                elif 'M-family' in spectrum.metadata['EDS'][key]:
                    family = 'M-family'
            spectrum.metadata['EDS']['GUI'][key] = {'symmetry': family}
    return spectrum.metadata['EDS']['GUI']

def quantify_cross_section(spectrum, mask=None):
    """Calculate quantification for EDS spectrum with cross sections."""
    spectrum.metadata['EDS'].setdefault('GUI', {})
    acceleration_voltage = spectrum.metadata.get('experiment', {}).get('acceleration_voltage',
                                                                       200000)
    families = get_families(spectrum)

    total = 0
    total_amu = 0
    for key, family in families.items():
        if key in mask:
            continue
        amu = spectrum.metadata['EDS'][key]['atomic_weight']
        intensity  = spectrum.metadata['EDS'][key][family['symmetry']].get('areal_density', 0)
        z = get_atomic_number(key)
        x_sect = family_ionization(z, family['symmetry'][0], acceleration_voltage)*1e23

        spectrum.metadata['EDS']['GUI'][key]['cross_section'] = x_sect
        spectrum.metadata['EDS']['GUI'][key]['composition_counts'] = intensity/x_sect
        total += intensity / x_sect
        total_amu += intensity / x_sect * amu

    for key, family in families.items():
        intensity  = spectrum.metadata['EDS'][key][family['symmetry']].get('areal_density', 0)
        if key in mask:
            spectrum.metadata['EDS']['GUI'][key] = {'atom%': 0,
                                                    'weight%': 0,
                                                    'excluded': True,
                                                    'intensity': intensity,
                                                    'symmetry': family['symmetry']}
            continue
        amu = spectrum.metadata['EDS'][key]['atomic_weight']
        x_sect = spectrum.metadata['EDS']['GUI'][key]['cross_section']
        spectrum.metadata['EDS']['GUI'][key] = {'atom%': intensity/x_sect/total*100,
                                                'weight%': intensity/x_sect*amu/total_amu*100,
                                                'excluded': False,
                                                'intensity': intensity,
                                                'symmetry': family['symmetry']}
        element = spectrum.metadata['EDS']['GUI'][key]
        out_text = f"{key:2}: {element['atom%']:.2f} at% {element['weight%']:.2f} wt%"
        print(out_text)


def quantification_k_factors(spectrum, mask=None):
    """Calculate quantification for EDS spectrum with k-factors."""
    tags = {}
    if not isinstance(mask, list) or mask is None:
        mask = []
    atom_sum = 0.
    weight_sum  = 0.
    spectrum.metadata['EDS'].setdefault('GUI', {})
    for key in spectrum.metadata['EDS']:
        intensity = 0.
        k_factor = 0.
        if key in ['detector', 'quantification']:
            pass
        elif isinstance(spectrum.metadata['EDS'][key], dict) and key in elements_list:
            family = spectrum.metadata['EDS'].get('GUI', {}).get(key, {}).get('symmetry', None)
            if family is None:
                if 'K-family' in spectrum.metadata['EDS'][key]:
                    family = 'K-family'
                elif 'L-family' in spectrum.metadata['EDS'][key]:
                    family = 'L-family'
                elif 'M-family' in spectrum.metadata['EDS'][key]:
                    family = 'M-family'
            spectrum.metadata['EDS']['GUI'][key] = {'symmetry': family}
            intensity = spectrum.metadata['EDS'][key][family].get('areal_density', 0)
            k_factor = spectrum.metadata['EDS'][key][family].get('k_factor', 0)
            atomic_weight = spectrum.metadata['EDS'][key]['atomic_weight']
            if key in mask:
                spectrum.metadata['EDS']['GUI'][key] = {'atom%': 0,
                                                        'weight%': 0,
                                                        'excluded': True,
                                                        'symmetry': family,
                                                        'k_factor': k_factor,
                                                        'intensity': intensity}
                continue

            tags[key] =  {'atom%': intensity*k_factor/ atomic_weight,
                          'weight%': (intensity*k_factor) ,
                          'k_factor': k_factor,
                          'intensity': intensity,
                          'family': family,
                          'excluded': False}
            atom_sum += intensity*k_factor/ atomic_weight
            weight_sum += intensity*k_factor
        tags['sums'] = {'atom%': atom_sum, 'weight%': weight_sum}

    spectrum.metadata['EDS']['quantification'] = tags
    eds_dict = spectrum.metadata['EDS']
    for key in eds_dict['quantification']:
        if key != 'sums':
            tags = eds_dict['quantification']
            out_string = f"{key:2}: {tags[key]['atom%']/tags['sums']['atom%']*100:.2f} at%"
            out_string += f" {tags[key]['weight%']/tags['sums']['weight%']*100:.2f} wt%"
            if key in eds_dict['GUI']:
                eds_dict['GUI'][key]['atom%'] = tags[key]['atom%']/tags['sums']['atom%']*100
                eds_dict['GUI'][key]['weight%'] = tags[key]['weight%']/tags['sums']['weight%']*100
                print(out_string)
                eds_dict['GUI'][key]['excluded'] = tags[key]['excluded']
                eds_dict['GUI'][key]['k_factor'] = tags[key]['k_factor']
                eds_dict['GUI'][key]['intensity'] = tags[key]['intensity']
    print('excluded from quantification ', mask)

    return tags

def family_ionization(element_z, family='K', acceleration_voltage=200000):
    """Calculate ionization cross sections for all subshells in a line family."""
    element_z = get_z(element_z)
    x_sections = get_bote_salvat_dict(acceleration_voltage, element_z)
    family_x_sections = {}
    for key, value in x_sections.items():
        if len(key) > 2:  # must be a  line
            family_x_sections[key[0]+'-family'] = family_x_sections.get(key[0]+'-family', 0)
            family_x_sections[key[0]+'-max'] = family_x_sections.get(key[0]+'-max', 0)
            family_x_sections[key[0]+'-family'] += value['cs']
            family_x_sections[key[0]+'-max'] = max(family_x_sections[key[0]+'-max'], value['cs'])
    x_sections.update(family_x_sections)
    if f"{family}-family" in family_x_sections:
        return family_x_sections[f"{family}-family"]
    return family_x_sections
