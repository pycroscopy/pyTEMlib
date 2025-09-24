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

import xraylib as xr

from .config_dir import config_path
from .utilities import elements as elements_list
from .utilities import get_z


def get_atomic_number(z):
    """Returns the atomic number independent of input as a string or number"""
    return get_z(z)

line_list = []
for _key in xr.__dict__:
    if '_LINE' in _key:
        line_list.append(_key.split('_')[0])

shell_list = []
for _key  in xr.__dict__:
    if '_SHELL' in _key:
        shell_list.append(_key.split('_')[0])

auger_list = []
for _key in xr.__dict__:
    if '_AUGER' in _key:
        auger_list.append(_key.split('_')[:-1])

def get_bote_salvat_dict(z=0):
    """Load the Bote & Salvat cross section data from JSON file."""
    filename = os.path.join(config_path, 'Bote_Salvat.json')
    x_sections = json.load(open(filename, 'r', encoding='utf-8'))
    if z > 0:
        return x_sections[str(z)]
    return x_sections

def bote_salvat_xsection(x_section, subshell, acceleration_energy):
    """
    Using Tabulated Values for electrons of: 
    - D. Bote and F. Salvat, "Calculations of inner-shell ionization by electron 
      impact with the distorted-wave and plane-wave Born approximations", 
      Phys. rest_energy. A77, 042701 (2008).
    
    - Bote, David, et al. "Cross sections for ionization of K, L and M shells of 
      atoms by impact of electrons and positrons with energies up to 1 GeV: 
      Analytical formulas." 
      Atomic Data and Nuclear Data Tables 95.6 (2009): 871-909.

    Computes the inner sub-shell ionization cross section for energetic electrons. 
    `subshell` is 1->K, 2->L₁, 3->L₂, ..., 9->M₅
    Parameters
    ----------
    z : int
        The atomic number z in the range 1:99   
    subshell : int
        The atomic sub-shell being ionized 1->K, 2->L₁, 3->L₂, ..., 9->M₅
    acceleration_energy : float
        The kinetic energy of the incident electron in eV
    edge_energy : float
        The edge energy of the sub-shell in eV

    Returns
    -------
    float
        The ionization cross section in barns
    """
    edge_energy = x_section['edge'][subshell-1]
    over_voltage = acceleration_energy / edge_energy
    if over_voltage < 1.0:
        return
    if over_voltage <= 16:
        a = np.array(x_section['A'])[subshell, :]
        opu = 1.0 / (1.0 + over_voltage)
        ffitlo = a[0] + a[1] * over_voltage + opu*(a[2] + opu**2*(a[3] + opu**2*a[4]))
        x_ion_e = (over_voltage - 1.0) * (ffitlo / over_voltage)**2
    else:
        rest_ene = 5.10998918e5  # electron rest energy in eV
        e_0 = acceleration_energy
        beta2 = (e_0 * (e_0 + 2.0 * rest_ene)) / ((e_0 + rest_ene)**2)
        x = np.sqrt(e_0 * (e_0 + 2.0 * rest_ene)) / rest_ene
        g = x_section['G'][subshell-1]
        ffitup = ((((2.0 * np.log(x)) - beta2) * (1.0 + g[0] / x)) + g[1] + g[2]
                  * np.sqrt(rest_ene / (e_0 + rest_ene)) + g[3] / x)
        factr = x_section['Anlj'][subshell-1] / beta2
        x_ion_e = ((factr * over_voltage) / (over_voltage + x_section['Be'][subshell-1])) * ffitup
    return 4.0 * np.pi * 5.291772108e-9**2 * x_ion_e  # in barns


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


def get_transition_dict(shell):
    """Get the transition dictionary for a given shell."""
    transition_dict = {}
    for index, line in enumerate(line_list):
        if line[:-2] == shell:
            # lines have negative !!!!  indices in xraylib
            transition_dict[line] = -index - 1
    return transition_dict


def radiation_ionization(element, shell):
    """Radiative transitions from higher shells to the given shell"""
    end_family = shell[0]
    ionization = 0.0
    for index, line in enumerate(shell_list):
        start_family = line[0]
        if end_family > start_family:
            tr_dict =  get_transition_dict(line)
            try:
                fy = xr.FluorYield(element, index)
                rr = xr.RadRate(element, tr_dict[line + shell])
                ionization +=  fy * rr
            except ValueError:
                pass
    return ionization


def get_all_subshells(line_family):
    """Get all subshells for a given line family."""
    subshells = []
    for subshell in shell_list:
        if line_family[0] == subshell[0]:
            subshells.append(subshell)
    return subshells


def full_auger_ionization (element, shell) :
    """ Auger transitions from lower shells to the given shell"""
    ionization = 0
    for i in range(shell_list.index(shell)) :
        ionization += auger_ionization(element, shell, shell_list[i])
    return ionization


def get_auger_dict(shell, lower_shell) :
    """Get the Auger transition dictionary for a given shell and lower shell."""
    auger_dict = {}
    for key, item in xr.__dict__.items():
        if '_AUGER' in key:
            if key[:2] == lower_shell and key[3:5] == shell:
                auger_dict[key[:7]] = item
    return auger_dict


def get_trans_dictionary(shell):
    """Get the transition dictionary for a given shell."""
    trans = {}
    shell_group = shell[0]
    if shell[0] == 'K':
        return trans
    for i in range(1, int(shell[1])):
        trans[f"{shell_group}{i}{shell[1]}"] = xr.__dict__[f"F{shell_group}{i}{shell[1]}_TRANS"]
    return trans


def coster_kronic_ionization (tags, shell) :
    """ Coster Kronig transitions 
        For the L1, there is not coster kronig
        transitions from higher to lower shells are added
    """
    element = int(tags['atomic_number'])

    ionization = 0
    if shell[0] == 'K':
        return ionization
    for trans, item in get_trans_dictionary(shell).items():
        try:
            ionization += tags[trans[:2]] * xr.CosKronTransProb(element, item)
        except ValueError:
            pass
    return ionization


def auger_ionization (element, shell, lower_shell):
    """ Auger transitions from lower shells to the given shell"""
    if shell == "K":
        return 0
    else:
        auger_dict = get_auger_dict(shell, lower_shell)
        try:
            auger_yield = xr.AugerYield(element, shell_list.index(lower_shell))
            auger_rate = 0
            for trans in auger_dict.values():
                try:
                    auger_rate += xr.AugerRate(element, trans)
                except ValueError:
                    pass
            return auger_rate * auger_yield
        except ValueError:
            return 0


def radiation_emission(element, shell):
    """Radiative transitions from the given shell to higher shells
    Does not work"""

    start_family = shell[0]
    if start_family > 'M':
        return 0.0
    tr_dict =  get_transition_dict(shell)
    radiation_propability  = 0.0
    for index, line in enumerate(shell_list):
        end_family = line[0]
        if end_family > start_family and end_family < 'N':
            try:
                fluorescent_yield = xr.FluorYield(element, index)
                rate = xr.RadRate(element, tr_dict[shell + line])
                radiation_propability +=  fluorescent_yield * rate
            except ValueError:
                pass
    return radiation_propability


def family_ionization(tags, line_family, acceleration_energy=200000):
    """Calculate ionization cross sections for all subshells in a line family."""
    element_z = tags['atomic_number']
    x_sections = get_bote_salvat_dict(z=element_z)
    # direct
    for subshell in get_all_subshells(line_family):
        index = shell_list.index(subshell)
        tags[subshell] =  bote_salvat_xsection(x_sections, index, acceleration_energy)
    # cascade corrections
    for subshell in get_all_subshells(line_family):
        index = shell_list.index(subshell)
        coster_kronig = coster_kronic_ionization (tags, subshell)
        cascade_ionization = 1 + radiation_ionization(element_z, subshell)
        cascade_ionization += full_auger_ionization(element_z, subshell)
        tags[subshell] *= cascade_ionization
        tags[subshell] += coster_kronig
    return tags


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

def quantify_cross_section(spectrum, x_section=None, mask=None):
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
        peaks = spectrum.metadata['EDS'][key][family['symmetry']].get('peaks', np.array([1]))
        f_yield = spectrum.metadata['EDS'][key][family['symmetry']].get('fluorescent_yield',0)
        tags = {'atomic_number':z}
        tags = family_ionization(tags, family['symmetry'], acceleration_voltage)
        x_sect = tags['K']*1e14 / f_yield / peaks.max()
        # print(x_sect)
        # x_sect = get_cross_section(x_section[str(z)], family['symmetry'],
        #                            acceleration_voltage)*1e14/ f_yield/peaks.max()
        spectrum.metadata['EDS']['GUI'][key]['cross_section'] = x_sect
        spectrum.metadata['EDS']['GUI'][key]['composition_counts'] = intensity*x_sect
        # print(key, ' - ', family['symmetry'], intensity,  x_sect,  f_yield, peaks.max())

        total += intensity * x_sect
        total_amu += intensity * x_sect * amu

    for key, family in families.items():
        if key in mask:
            continue
        amu = spectrum.metadata['EDS'][key]['atomic_weight']
        intensity  = spectrum.metadata['EDS'][key][family['symmetry']].get('areal_density', 0)
        x_sect = spectrum.metadata['EDS']['GUI'][key]['cross_section']
        spectrum.metadata['EDS']['GUI'][key] = {'atom%': intensity*x_sect/total,
                                                'weight%': intensity*x_sect*amu/total}
        out_text = f"{key:2}: {intensity*x_sect/total*100:.2f} at% "
        out_text += f"{intensity*x_sect*amu/total_amu*100:.2f} wt%"
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
        if key in mask + ['detector', 'quantification']:
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

            intensity = spectrum.metadata['EDS'][key][family]['areal_density']  # /peaks_max
            k_factor = spectrum.metadata['EDS'][key][family]['k_factor']
            atomic_weight = spectrum.metadata['EDS'][key]['atomic_weight']
            tags[key] =  {'atom%': intensity*k_factor/ atomic_weight,
                          'weight%': (intensity*k_factor) ,
                          'k_factor': k_factor,
                          'intensity': intensity}
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
    print('excluded from quantification ', mask)

    return tags


def get_emission (element, line_family, accelerating_energy=200000):
    """Calculate the total emission for a given element and line family."""
    element_z = get_z(element)
    tags = {element:{'atomic_number': element_z}}
    family_ionization(tags[element], line_family, accelerating_energy)

    tags[element][f'{line_family}_emission'] = 0.
    for line in shell_list:
        if line_family in line:
            print(line, radiation_emission(element_z, line))
            print('emission', tags[element][line] * radiation_emission(element_z, line))
            tags[element][f'{line_family}_emission'] += (tags[element][line]
                                                         * radiation_emission(element_z, line))
    # tags[element][f'{line_family}_emission']

    return tags
