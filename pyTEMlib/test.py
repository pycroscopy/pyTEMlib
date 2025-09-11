import sys

from matplotlib.pyplot import plot
sys.path.insert(0, './')
import pyTEMlib
print(pyTEMlib.__version__)

filename = "C:\\Users\\gduscher\\Desktop\\ESL-506-15kV.esl"
filename = "C:\\Users\\gduscher\\.pyTEMlib\\k-factors-Spectra300UTK200keV.csv"
# pyTEMlib.eds_tools.read_esl_k_factors("C:\\Users\\gduscher\\Desktop\\ESL-506-15kV.esl")

# print(pyTEMlib.eds_tools.read_csv_k_factors(filename))
# pp =pyTEMlib.eds_tools.convert_k_factor_file(filename)
# print('read bruker k-factors')
#
# print(pyTEMlib.eds_tools.read_k_factors('k_factors_Thermo_200keV.json')) 

# print(pyTEMlib.eds_tools.get_k_factor_files())
import os
file = os.path.join(pyTEMlib.config_dir.config_path, 'Dirac_GOS.gosh')    
import h5py

"""
element = 'Si'
edge = 'L3'
with h5py.File(file, 'r') as gos_file:
    gos = gos_file[element][edge]['data'][:].squeeze().T
    free_energies = gos_file[element][edge]['free_energies'][:][ :]  # two dimensional array
    q_axis = gos_file[element][edge]['q'][:] # in  [1/m]
    ionization_energy = gos_file[element][edge]['metadata'].attrs['ionization_energy']
print (ionization_energy)

import matplotlib.pylab as plt
import scipy
import numpy as np

x = free_energies
y = q_axis
y = y/min(y)
z = gos
print(x.shape, y.shape, z.shape)
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
print(min(y), max(y))
grid_x, grid_y = np.meshgrid(X, Y)

interp = scipy.interpolate.griddata(zip(x, y), z, (grid_x, grid_y), method='linear', fill_value=0)
print(interp.shape, print())

plt.imshow(interp[:,:,0].T, extent=(min(x), max(x), min(y), max(y)), origin='lower')

plt.colorbar()

plt.show()


plt.figure()
plt.pcolormesh( q_axis/1e10,free_energies+ionization_energy, gos)
plt.xlabel('q [1/A]')
plt.ylabel('Energy above ionization [eV]')
plt.colorbar()
plt.title('GOS for ' + element + ' ' + edge)
plt.show()
"""
import matplotlib.pylab as plt

import scipy
import numpy as np






def getinterpolatedgos(E, q, E_axis, q_axis, GOSmatrix):
    """
    Gets the interpolated value of the GOS from the E and q value.
    """
    index_q = np.searchsorted(q_axis, q, side='left')
    index_E = np.searchsorted(E_axis, E, side='left')

    if index_E == 0:
        return 0
    if index_E == E_axis.size:
        return 0.
    if index_q == 0:
        return GOSmatrix[index_E, 0]
    if index_q == q_axis.size:
        return 0.0

    dE = E_axis[index_E] - E_axis[index_E - 1]
    dq = q_axis[index_q] - q_axis[index_q - 1]

    distE = E - E_axis[index_E - 1]
    distq = q - q_axis[index_q - 1]

    r = GOSmatrix[index_E - 1, index_q - 1] * (1 / (dE * dq)) * (dE - distE) * (dq - distq)
    r += GOSmatrix[index_E - 1, index_q] * (1 / (dE * dq)) * (dE - distE) * (distq)
    r += GOSmatrix[index_E, index_q - 1] * (1 / (dE * dq)) * (distE) * (dq - distq)
    r += GOSmatrix[index_E, index_q] * (1 / (dE * dq)) * (distE) * (distq)
    return r


def gaussian(E, integral, x0, sigma):
    # A = integral / (np.sqrt(2 * np.pi) * sigma)
    g = np.exp(-0.5 * (E - x0)**2 / sigma**2)
    g = integral * g / g.sum()
    return g

def getinterpolatedq(q, GOSarray, q_axis):
    """
    Gets the interpolated value of the GOS array as a function of q.
    Usefull for the bounded states

    Parameters
    ----------
    q: float
        The q from the GOS should be interpolated
    GOSarray:
        dddd
    q_axis: numpy array
        The q axis on which the GOS is calculated


    Returns
    -------
    interpolated GOS matrix

    """
    index_q = np.searchsorted(q_axis, q, side='left')

    if index_q == 0:
        return GOSarray[0]
    if index_q == q_axis.size:
        return 0.0

    dq = q_axis[index_q] - q_axis[index_q - 1]

    distq = q - q_axis[index_q - 1]

    r0 = GOSarray[index_q - 1] * (1/dq) * (dq-distq)
    r1 = GOSarray[index_q] * (1/dq) * (distq)

    return r0 + r1

def correction_factor_kohl(alpha, beta, theta, min_alpha=1e-6):
    """
    STILL NEEDS TO BE VALIDATED
    Calculates the correction factor when using a convergent
    probe. For probes having is convergence angle smaller than
    min_alpha no correction is applied.
    Ultramicroscopy 16 (1985) 265-268:
    https://doi.org/10.1016/0304-3991(85)90081-6
     Parameters
     ----------
     alpha : float
         Convergence angle in radians
     beta : float
         Collection angle in radians
     theta : float
         The angle for which the correction factor should be calculated
     min_alpha : float
        Minimum convergence angle for which the correction is applied

     Returns
     -------
     corr_factor : float
        correction factor used in the integration
    """
    if alpha < min_alpha:
        corr_factor = 1.
    elif theta <= np.abs(alpha - beta):
        min_thetasq = min(alpha**2, beta**2)
        corr_factor = min_thetasq / alpha**2
    else:
        x = (alpha**2 + theta**2 - beta**2) / (2. * alpha * theta)
        y = (beta**2 + theta**2 - alpha**2) / (2. * beta * theta)
        wortel = np.sqrt(4 * alpha**2 * beta**2 - (alpha**2 + beta**2 - theta**2)**2)
        corr_factor = (1 / np.pi) * (np.arccos(x) + (beta**2 / alpha**2 * np.arccos(y)) - (
                                      1 / (2 * alpha**2) * wortel))
    return corr_factor

def get_dirac_X_section(element = 'Si', edge = 'L3', file = file,
                        q_steps=100, E0=200000, beta=0.100, alpha=0):
    """ Calculates the cross section from the GOS matrix"""

    
    with h5py.File(file, 'r') as gos_file:
        GOSmatrix = gos_file[element][edge]['data'][:].squeeze().T
        free_energies = gos_file[element][edge]['free_energies'][:][ :]  # two dimensional array
        q_axis = gos_file[element][edge]['q'][:] # in  [1/m]
        ek = gos_file[element][edge]['metadata'].attrs['ionization_energy']

    energy_axis = np.linspace(50, 850, int(800/5))
    shell_occupancy = 1
    pref = 1e28 * shell_occupancy
    
    e = scipy.constants.e
    c = scipy.constants.c
    m = scipy.constants.electron_mass
    a_0 = scipy.constants.physical_constants['Bohr radius'][0]
    gamma = 1 + e * E0 / (m * c ** 2)

    effective_incident_energy = E0 * (1 + gamma) / (2 * gamma**2)
    T = effective_incident_energy
    R = scipy.constants.Rydberg

    bool0 = free_energies < 0
    Ebound = free_energies[bool0] + ek

    dsigma_dE = np.zeros(energy_axis.shape)
    dsigma_dE_bound = np.zeros(energy_axis.shape)
    sigma = 2*(energy_axis[1] - energy_axis[0])
    rel_energy_axis = free_energies + ek

    for i in range(Ebound.size):
        E = Ebound[i]
        integral = 0
        # the bounded states are differently interpolated
        qa0sq_min = E ** 2 / (4 * R * T) + (E ** 3) / (8 * gamma ** 3 * R * T ** 2)
        qa0sq_max = qa0sq_min + 4 * gamma ** 2 * (T / R) * (np.sin((beta + alpha) / 2)) ** 2
        logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max),
                                    q_steps)
        lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
        print(i, logqa0sq_axis, lnqa0sqstep)
        for j in range(logqa0sq_axis.size):
            q = np.sqrt(np.exp(logqa0sq_axis[j])) / scipy.constants.physical_constants['Bohr radius'][0]
            theta = 2. * np.sqrt(np.abs( R * (np.exp(logqa0sq_axis[j]) - qa0sq_min) / 
                                        (4. * gamma**2 * T)))
            GOSarray = GOSmatrix[i, :]
            df_dE = getinterpolatedq(q, GOSarray, q_axis)

            # integral+= df_dE*lnqa0sqstep
            integral += df_dE * lnqa0sqstep * correction_factor_kohl(alpha, beta, theta)

        sig = 4 * np.pi * a_0 ** 2 * (R / E) * (R / T) * integral
        dsigma_dE_bound += gaussian(energy_axis, sig, E, sigma)

    
    # the for loop over the bound states
    for i in range(energy_axis.size):
        E = energy_axis[i]
        integral = 0
        if (E > ek) & (E <= rel_energy_axis[-1]):
            qa0sq_min = E**2 / (4 * R * T) + (E**3) / (8 * gamma**3 * R * T**2)
            qa0sq_max = qa0sq_min + 4 * gamma**2 * (T / R) * (np.sin((beta + alpha) / 2))**2
            logqa0sq_axis = np.linspace(np.log(qa0sq_min), np.log(qa0sq_max), q_steps)
            lnqa0sqstep = (logqa0sq_axis[1] - logqa0sq_axis[0])
            for j in range(logqa0sq_axis.size):
                q = np.sqrt(np.exp(logqa0sq_axis[j])) / a_0
                theta = 2. * np.sqrt(np.abs(
                    R * (np.exp(logqa0sq_axis[j]) - qa0sq_min) / (
                            4. * gamma ** 2 * T)))
                df_dE = getinterpolatedgos(E, q, rel_energy_axis, q_axis, GOSmatrix)
                # integral+= df_dE*lnqa0sqstep
                integral += df_dE * lnqa0sqstep * correction_factor_kohl(alpha, beta, theta)
            # dsigma_dE[i] = 4*np.pi*pc.a0()**2*(R/E)*(R/T)*integral*dispersion
            dsigma_dE[i] = 4 * np.pi * a_0 ** 2 * (R / E) * (R / T) * integral
        else:
            dsigma_dE[i] = 0

    cross_section =  dsigma_dE + dsigma_dE_bound * pref

    return cross_section

def energy2wavelength(e0: float) -> float:
    """get deBroglie wavelength of electron accelerated by energy (in eV) e0"""
    ev = scipy.constants.e * e0
    m_e = scipy.constants.m_e
    c = scipy.constants.c
    h = scipy.constants.h
    return h / np.sqrt(2 * m_e * ev * (1 + ev / (2 * m_e * c**2)))*1e10


def ddscs_dE_dOmega(free_energies, ek, E0, q_axis, GOSmatrix):
    """scattering cross section as a function of energy loss and solid angle

    Args:
        free_energies: 1d numpy array
            The energy axis on which the GOS table is calculated without the onset
            energy [eV]
        ek: float
            The onset energy of the calculated edge [eV]
        E0: float
            The acceleration voltage of the incoming electrons [V]
        q_axis: 1d numpy array
            The momentum on which the GOS table are calculated. [kg m /s]?
        GOSmatrix: 2d numpy array
            The GOS

    Returns:
        np.array: scattering cross section as a function of energy loss and solid angle
    """
    R = scipy.constants.Rydberg
    e = scipy.constants.e
    e = scipy.constants.e
    c = scipy.constants.c
    m = scipy.constants.electron_mass
    a_0 = scipy.constants.physical_constants['Bohr radius'][0]
    gamma = 1 + e * E0 / (m * c ** 2)
    energy_losses = free_energies + ek
    
    k0 = 2 * np.pi / energy2wavelength(E0)

    scs_list = []
    for idx, epsilon in enumerate(free_energies):
        kn = 2 * np.pi / energy2wavelength(E0-energy_losses[idx])
        scs = (
            4
            * gamma ** 2
            / q_axis**2
            * kn
            / k0
            * GOSmatrix[idx]
            / energy_losses[idx]
            * R
        )
        scs_list.append(scs)
    scs_list = np.array(scs_list).squeeze()

    return scs_list

def plot_ddscs(element = 'Si', edge = 'L3', file = file,
                        q_steps=100, E0=200000, beta=0.100, alpha=0):
    with h5py.File(file, 'r') as gos_file:
        GOSmatrix = gos_file[element][edge]['data'][:].squeeze().T
        free_energies = gos_file[element][edge]['free_energies'][:][ :]  # two dimensional array
        q_axis = gos_file[element][edge]['q'][:] # in  [1/m]
        ek = gos_file[element][edge]['metadata'].attrs['ionization_energy']
        for k in gos_file[element][edge]['metadata'].attrs.keys():
            print(f"{k} => {gos_file[element][edge]['metadata'].attrs[k]}")
        occupancy = gos_file[element][edge]['metadata'].attrs['occupancy_ratio']
    k0 = 2 * np.pi / energy2wavelength(E0)
    ddscs = ddscs_dE_dOmega(free_energies, ek, E0, q_axis/1e10, GOSmatrix)
    plt.subplots(1, 2, figsize=(12, 5))
    plt.subplot(121)
    plt.pcolormesh(q_axis/1e10, free_energies, ddscs)
    plt.ylabel('Energy loss [eV]')
    plt.xlabel('Scattering vector [1/A]')
    plt.colorbar()
    plt.tight_layout()
    plt.title('Double differential scattering cross section')
    plt.subplot(122)
    theta = np.arctan(q_axis/1e10 /k0)*1e3 # this is approximation
    plt.pcolormesh(theta, free_energies, ddscs) 
    plt.xlim([0, 50])
    plt.xlabel('Scattering angle [mrad]')
    plt.ylabel('Energy loss [eV]')
    plt.title('Double differential scattering cross section')
    plt.colorbar()  
    plt.tight_layout()
    max_q = np.searchsorted(theta, 30)
    max_e = np.searchsorted(free_energies, 1000)
    max_q_value = np.tan(30e-3)*k0*1e10
    print(f"min_q: {q_axis[0]}, end_q: {q_axis[-1]}, max_q_value: {max_q_value}")
    print(q_axis[max_q])
    y_sparse = theta[:max_q]
    x_sparse = free_energies[:max_e]
    z_sparse = ddscs[:max_e, :max_q]
    xnew = np.linspace(0, 1000, 1000)
    ynew = np.linspace(0, 30, 100)

    Xnew, Ynew = np.meshgrid(xnew, ynew)

    # Flatten the input data for griddata
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse, indexing='ij')
    points = np.column_stack([X_sparse.ravel(), Y_sparse.ravel()])
    values = z_sparse.ravel()
    znew_reggrid = scipy.interpolate.griddata(points, values, (Xnew, Ynew), method='linear', fill_value=0)
    plt.figure()
    plt.imshow(znew_reggrid, extent=(0, 400, 0, 30), origin='lower', aspect='auto')

    plt.colorbar()

    print(ddscs.shape, q_axis.shape, free_energies.shape, znew_reggrid.shape)

    plt.figure()
    #plt.plot(free_energies, ddscs[:, :max_q].sum(axis=1),)
    plt.plot(xnew, znew_reggrid.sum(axis=0),)
    plt.show()

def plot_ddscs(element = 'Si', edge = 'L3', file = file, 
               energy_scale = np.linspace(50, 850, int(800*3)),
               q_steps=100, E0=200000, beta=0.100, alpha=0):
    with h5py.File(file, 'r') as gos_file:
        GOSmatrix = gos_file[element][edge]['data'][:].squeeze().T
        free_energies = gos_file[element][edge]['free_energies'][:][ :]  # two dimensional array
        q_axis = gos_file[element][edge]['q'][:] # in  [1/m]
        ek = gos_file[element][edge]['metadata'].attrs['ionization_energy']
        for k in gos_file[element][edge]['metadata'].attrs.keys():
            print(f"{k} => {gos_file[element][edge]['metadata'].attrs[k]}")
        occupancy = gos_file[element][edge]['metadata'].attrs['occupancy_ratio']
    
    
    k0 = 2 * np.pi / energy2wavelength(E0)
    ddscs = ddscs_dE_dOmega(free_energies, ek, E0, q_axis/1e10, GOSmatrix)
    theta = np.arctan(q_axis/1e10 /k0)*1e3 # this is an approximation
    
    max_q = np.searchsorted(theta, beta*1e3)+1
    max_e = np.searchsorted(free_energies+ek, energy_scale[-1])+1
    max_q_value = np.tan(beta)*k0
    print(f"min_q: {q_axis[0]/1e10}, end_q: {q_axis[-1]/1e10}, max_q_value: {max_q_value}")
    print(q_axis[max_q])
    y_sparse = q_axis[:max_q]/1e10
    x_sparse = free_energies[:max_e]+ek
    z_sparse = ddscs[:max_e, :max_q]
    xnew = energy_scale
    ynew = np.linspace(0, max_q_value, q_steps)

    Xnew, Ynew = np.meshgrid(xnew, ynew)

    # Flatten the input data for griddata
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse, indexing='ij')
    points = np.column_stack([X_sparse.ravel(), Y_sparse.ravel()])
    values = z_sparse.ravel()
    znew_reggrid = scipy.interpolate.griddata(points, values, (Xnew, Ynew), method='linear', fill_value=0)
    plt.figure()
    plt.imshow(znew_reggrid, extent=(xnew[0], xnew[-1], 0, ynew[-1]), origin='lower', aspect='auto')

    plt.colorbar()
    delta_q = ynew[1]-ynew[0]

    print(ddscs.shape, q_axis.shape, free_energies.shape, znew_reggrid.shape)

    plt.figure()
    #plt.plot(free_energies, ddscs[:, :max_q].sum(axis=1),)
    plt.plot(xnew, znew_reggrid.sum(axis=0)*delta_q,)
    plt.show()

plot_ddscs(element='Si', edge='L3', beta=0.030, alpha=0.0)
plt.show()
#xsec = get_dirac_X_section(element='Si', edge='M2', beta=0.0001, alpha=0.0)
#print('calculated cross section')
#plt.figure()
#plt.plot(np.linspace(50, 850, int(800/5)), xsec)
#plt.xlabel('Energy loss [eV]')
#plt.show()
