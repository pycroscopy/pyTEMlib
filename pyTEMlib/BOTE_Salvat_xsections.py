import os
import json
from typing import final
import numpy as np
import scipy

from .config_dir import config_path

def get_bote_salvat_dict(z=0):
    filename = os.path.join(config_path, 'Bote_Salvat.json')
    x_sections = json.load(open(filename))
    if z > 0:
        return x_sections[str(z)]
    return x_sections

def bote_salvat_xsection(x_section, subshell, acceleration_energy):
    """
    Using Tabulated Values for electrons of: 
    - D. Bote and F. Salvat, "Calculations of inner-shell ionization by electron 
      impact with the distorted-wave and plane-wave Born approximations", 
      Phys. Rev. A77, 042701 (2008).
    
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
        a = x_section['A'][subshell, :]
        opu = 1.0 / (1.0 + over_voltage)
        ffitlo = a[0] + a[1] * over_voltage + opu*(a[2] + opu**2*(a[3] + opu**2*a[4]))
        x_ion_e = (over_voltage - 1.0) * (ffitlo / over_voltage)**2
    else:
        REV = 5.10998918e5  # electron rest energy in eV
        e_0 = acceleration_energy
        beta2 = (e_0 * (e_0 + 2.0 * REV)) / ((e_0 + REV)**2)
        x = np.sqrt(e_0 * (e_0 + 2.0 * REV)) / REV
        g = x_section['G'][subshell-1]
        ffitup = (((2.0 * np.log(x)) - beta2) * (1.0 + g[0] / x)) + g[1] + g[2] * np.sqrt(REV / (e_0 + REV)) + g[3] / x
        factr = x_section['Anlj'][subshell-1] / beta2
        x_ion_e = ((factr * over_voltage) / (over_voltage + x_section['Be'][subshell-1])) * ffitup
    return 4.0 * np.pi * 5.291772108e-9**2 * x_ion_e  # in barns


def casnati_x_section(subshell, occupancy, edge_energy, beamE):
    """(Casnati's equation) was found to fit cross-section data to
    typically better than +-10% over the range 1<=Uk<=20 and 6<=Z<=79."
    Note: This result is for K shell. L & M are much less wellcharacterized.
    C. Powell indicated in conversation with Richie that he believed that Casnati's
    expression was the best available for L & M also.
    """
    REV = 5.10998918e5  # electron rest energy in eV
    res = 0.0;
    ee = edge_energy
    u = beamE / ee
    if u > 1.0:
        phi = 10.57 * np.exp((-1.736 / u) + (0.317 / u**2))
        psi = np.power(ee / scipy.constants.R, -0.0318 + (0.3160 / u) + (-0.1135 / u**2))
        i = ee / REV
        t = beamE / REV
        f = ((2.0 + i) / (2.0 + t)) * np.square((1.0 + t) / (1.0 + i))
        f *= np.power(((i + t) * (2.0 + t) * np.square(1.0 + i))
                       / ((t * (2.0 + t) * np.square(1.0 + i))
                          + (i * (2.0 + i))), 1.5)
        res = ((occupancy * np.sqrt((scipy.constants.BohrRadius *scipy.constants.R) / ee)
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
    x = (3.0e-22 * z**1.7) / (e + (0.005 * z**1.7 * np.sqrt(e))
                              + ((0.0007 * z**2) / np.sqrt(e)))
    return x

x_section_dict = get_bote_salvat_dict(22)
x = bote_salvat_xsection(x_section_dict, 3, 100000)
print(x)
