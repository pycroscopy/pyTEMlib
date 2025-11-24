""" Part of eels_tools for pyTEMlib"""
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import scipy

import sidpy
from sidpy.proc.fitter import SidFitter

from ..utilities import get_wavelength, effective_collection_angle
from ..utilities import gauss, lorentz
from .zero_loss_tools import zl


def drude(energy_scale, peak_position, peak_width, gamma):
    """dielectric function according to Drude theory"""

    eps = (1 - (peak_position ** 2 - peak_width * energy_scale * 1j) /
           (energy_scale ** 2 + 2 * energy_scale * gamma * 1j))  # Mod drude term
    return eps


def drude_lorentz(eps_inf, leng, ep, eb, gamma, e, amplitude):
    """dielectric function according to Drude-Lorentz theory"""
    eps = eps_inf
    for i in range(leng):
        eps = eps + amplitude[i] * (1 / (e + ep[i] + gamma[i]*1j) -
                                    1 / (e - ep[i] + gamma[i]*1j))
    return eps


def energy_loss_function(energy: np.ndarray, p: np.ndarray, anglog=1) -> np.ndarray:
    """Energy loss function based on dielectric function."""
    eps = 1 - p[0]**2/(energy**2+p[1]**2) + 1j * p[1] * p[0]**2/energy/(energy**2+p[1]**2)
    elf = (-1/eps).imag
    return elf*p[2]*anglog


def get_plasmon_losses(energy, params):
    """ Volume plasmons for spectrum images"""
    dset = np.zeros((params.shape[0], params.shape[1], energy.shape[0]))
    for x in range(params.shape[0]):
        for y in range(params.shape[1]):
            dset[x, y] +=  energy_loss_function(energy, params[x, y])
    return dset


def fit_plasmon(dataset: Union[sidpy.Dataset, np.ndarray],
                start_fit_energy: float, end_fit_energy: float,
                number_workers: int = 4, number_threads: int = 8
                ) -> Union[sidpy.Dataset, np.ndarray]:
    """
    Fit plasmon peak positions and widths in a TEM dataset using a Drude model.

    This function applies the Drude model to fit plasmon peaks in a dataset obtained 
    from transmission electron microscopy (TEM). It processes the dataset to determine 
    peak positions, widths, and amplitudes within a specified energy range. The function 
    can handle datasets with different dimensions and offers parallel processing capabilities.

    Parameters:
        dataset: sidpy.Dataset or numpy.ndarray
            The dataset containing TEM spectral data.
        start_fit_energy: float
            The start energy of the fitting window.
        end_fit_energy: float
            The end energy of the fitting window.
        plot_result: bool, optional
            If True, plots the fitting results (default is False).
        number_workers: int, optional
            The number of workers for parallel processing (default is 4).
        number_threads: int, optional
            The number of threads for parallel processing (default is 8).

    Returns:
        fitted_dataset: sidpy.Dataset or numpy.ndarray
            The dataset with fitted plasmon peak parameters. The dimensions and 
            format depend on the input dataset.

    Raises:
        ValueError: If the input dataset does not have the expected dimensions or format.

    Notes:
        - The function uses the Drude model to fit plasmon peaks.
        - The fitting parameters are peak position (e_p), peak width (e_w), and amplitude (A).
        - If `plot_result` is True, the function plots e_p, e_w, and A as separate subplots.
    """
    # define Drude function for plasmon fitting

    anglog, _, _ = angle_correction(dataset)

    def energy_loss_function2(e: np.ndarray, e_p: float,e_w: float,
                              amplitude: float) -> np.ndarray:
        eps = 1 - e_p**2/(e**2+e_w**2) + 1j * e_w * e_p**2/e/(e**2+e_w**2)
        elf = (-1/eps).imag
        return amplitude*elf

    # define window for fitting
    energy = dataset.get_spectral_dims(return_axis=True)[0].values
    start_fit_pixel = np.searchsorted(energy, start_fit_energy)
    end_fit_pixel = np.searchsorted(energy, end_fit_energy)

    # rechunk dataset
    if dataset.ndim == 3:
        dataset = dataset.rechunk(chunks=(1, 1, -1))
        fit_dset = dataset[:, :, start_fit_pixel:end_fit_pixel]
    elif dataset.ndim == 2:
        dataset = dataset.rechunk(chunks=(1, -1))
        fit_dset = dataset[:, start_fit_pixel:end_fit_pixel]
    else:
        fit_dset = np.array(dataset[start_fit_pixel:end_fit_pixel]/ anglog[start_fit_pixel:end_fit_pixel])
        guess_pos = np.argmax(fit_dset)
        guess_amplitude = fit_dset[guess_pos]
        guess_width =(end_fit_energy-start_fit_energy)/4
        guess_pos = energy[start_fit_pixel+guess_pos]

        if guess_width >8:
            guess_width=8
        try:
            popt, _ =  scipy.optimize.curve_fit(energy_loss_function2, energy[start_fit_pixel:end_fit_pixel], fit_dset,
                                p0=[guess_pos, guess_width, guess_amplitude])
        except:
            end_fit_pixel = np.searchsorted(energy, 30)
            fit_dset = np.array(dataset[start_fit_pixel:end_fit_pixel]/ anglog[start_fit_pixel:end_fit_pixel])
            try:
                popt, _ =  scipy.optimize.curve_fit(energy_loss_function,
                                                    energy[start_fit_pixel:end_fit_pixel], fit_dset,
                                                    p0=[guess_pos, guess_width, guess_amplitude])
            except:
                popt=[0,0,0]

        plasmon = dataset.like_data(energy_loss_function2(energy, popt[0], popt[1], popt[2]))
        plasmon *= anglog
        start_plasmon = np.searchsorted(energy, 0)+1
        plasmon[:start_plasmon] = 0.

        epsilon = drude(energy, popt[0], popt[1], 1) * popt[2]
        epsilon[:start_plasmon] = 0.

        plasmon.metadata['plasmon'] = {'parameter': popt, 'epsilon':epsilon}
        return plasmon

    # if it can be parallelized:
    fitter = SidFitter(fit_dset, energy_loss_function, num_workers=number_workers,
                       threads=number_threads, return_cov=False, return_fit=False, return_std=False,
                       km_guess=False, num_fit_parms=3)
    [fit_parameter] = fitter.do_fit()

    plasmon_dset = dataset * 0.0
    fit_parameter = np.array(fit_parameter)
    plasmon_dset += get_plasmon_losses(np.array(energy), fit_parameter)
    if 'plasmon' not in plasmon_dset.metadata:
        plasmon_dset.metadata['plasmon'] = {}
    plasmon_dset.metadata['plasmon'].update({'start_fit_energy': start_fit_energy,
                                              'end_fit_energy': end_fit_energy,
                                              'fit_parameter': fit_parameter,
                                              'original_low_loss': dataset.title})

    return plasmon_dset


def angle_correction(spectrum):
    """ angle correction per energy loss"""
    acceleration_voltage = spectrum.metadata['experiment']['acceleration_voltage']
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0]
    # eff_beta = effective_collection_angle(energy_scale,
    #                                       spectrum.metadata['experiment']['convergence_angle'],
    #                                       spectrum.metadata['experiment']['collection_angle'],
    #                                       acceleration_voltage)

    epc = energy_scale.slope  # input('ev per channel : ')

    alpha = spectrum.metadata['experiment']['convergence_angle']  # input('Alpha (mrad) : ')
    beta = spectrum.metadata['experiment']['collection_angle']  # input('Beta (mrad) : ')
    e = energy_scale.values
    e0 = acceleration_voltage/1000 # input('E0 (keV) : ')

    t = 1000.0*e0*(1.+e0/1022.12)/(1.0+e0/511.06)**2  # %eV # equ.5.2a or Appendix E p 427

    tgt=e0*(1.+e0/1022.)/(1+e0/511.)
    thetae=(e+1e-6)/tgt # % avoid NaN for e=0
    # %     A2,B2,T2 ARE SQUARES OF ANGLES IN RADIANS**2
    a2=alpha*alpha*1e-6 + 1e-7  # % avoid inf for alpha=0
    b2=beta*beta*1e-6
    t2=thetae*thetae*1e-6
    eta1=np.sqrt((a2+b2+t2)**2-4*a2*b2)-a2-b2-t2
    eta2=2.*b2*np.log(0.5/t2*(np.sqrt((a2+t2-b2)**2+4.*b2*t2)+a2+t2-b2))
    eta3=2.*a2*np.log(0.5/t2*(np.sqrt((b2+t2-a2)**2+4.*a2*t2)+b2+t2-a2))
    # eta=(eta1+eta2+eta3)/a2/np.log(4./t2)
    f1=(eta1+eta2+eta3)/2./a2/np.log(1.+b2/t2)
    f2=f1
    if alpha/beta > 1:
        f2=f1*a2/b2

    bstar=thetae*np.sqrt(np.exp(f2*np.log(1.+b2/t2))-1.)
    anglog = f2
    """
    b = eff_beta/1000.0 # %rad
    e0 = acceleration_voltage/1000.0 # %keV
    t = 1000.0*e0*(1.+e0/1022.12)/(1.0+e0/511.06)**2  # %eV # equ.5.2a or Appendix E p 427 
    tgt = 1000*e0*(1022.12 + e0)/(511.06 + e0)  # %eV  Appendix E p 427 

    the = energy_scale/tgt # varies with energy loss! # Appendix E p 427 
    anglog = np.log(1.0+ b*b/the/the)
    # 2 * t = m_0 v**2 !!!  a_0 = 0.05292 nm  epc is for sum over I0
    """
    return anglog,   (np.pi*0.05292* t / 2.0)/epc, bstar[0]


def inelastic_mean_free_path(e_p, spectrum):
    """ inelastc mean free path"""
    acceleration_voltage = spectrum.metadata['experiment']['acceleration_voltage']
    energy_scale = spectrum.get_spectral_dims(return_axis=True)[0].values

    e0 = acceleration_voltage/1000.0  # %keV

    eff_beta = effective_collection_angle(energy_scale,
                                          spectrum.metadata['experiment']['convergence_angle'],
                                          spectrum.metadata['experiment']['collection_angle'],
                                          acceleration_voltage)
    beta = eff_beta/1000.0 # %rad

    # %eV # equ.5.2a or Appendix E p 427
    t = 1000.0*e0*(1.+e0/1022.12)/(1.0+e0/511.06)**2
    #  Appendix E p 427
    tgt = 1000*e0*(1022.12 + e0)/(511.06 + e0)  # %eV
    # Appendix E p 427
    theta_e = e_p/tgt # varies with energy loss!

    # 2 * T = m_0 v**2 !!!
    a_0 = 0.05292 # nm
    imfp = 4 * t* a_0 / e_p / np.log(1 + beta**2/ theta_e**2)
    return imfp, theta_e


def multiple_scattering(energy_scale: np.ndarray, p: list, core_loss=False)-> np.ndarray:
    """Multiple scattering calculation based on plasmon peak fitting parameters."""
    p = np.abs(p)
    tmfp = p[3]
    if core_loss:
        dif = 1
    else:
        dif = 16
    ll_energie = np.linspace(1, 2048-1,2048)/dif

    ssd = energy_loss_function(ll_energie, p)
    ssd  = np.fft.fft(ssd)
    ssd2 = ssd.copy()

    ### sum contribution from each order of scattering:
    psd = np.zeros(len(ll_energie))
    for order in range(15):
        # This order convoluted spectrum
        # convoluted ssd is SSD2
        ssd2 = np.fft.ifft(ssd).real

        # scale right (could be done better? GERD)
        # And add this order to final spectrum
        #using equation 4.1 of Egerton ed2
        psd += ssd2*abs(sum(ssd)/sum(ssd2)) / scipy.special.factorial(order+1)*np.power(tmfp, (order+1))*np.exp(-tmfp)

        # next order convolution
        ssd = ssd * ssd2

    psd /=tmfp*np.exp(-tmfp)
    bgd_coef = scipy.interpolate.splrep(ll_energie, psd, s=0)
    msd = scipy.interpolate.splev(energy_scale, bgd_coef)
    start_plasmon = np.searchsorted(energy_scale, 0)+1
    msd[:start_plasmon] = 0.0
    return msd

def fit_multiple_scattering(dataset: Union[sidpy.Dataset, np.ndarray],
                            start_fit_energy: float, end_fit_energy: float, pin=None,
                            number_workers: int = 4, number_threads: int = 8
                            ) -> Union[sidpy.Dataset, np.ndarray]:
    """
    Fit multiple scattering of plasmon peak in a TEM dataset.

    Parameters:
        dataset: sidpy.Dataset or numpy.ndarray
            The dataset containing TEM spectral data.
        start_fit_energy: float
            The start energy of the fitting window.
        end_fit_energy: float
            The end energy of the fitting window.
        number_workers: int, optional
            The number of workers for parallel processing (default is 4).
        number_threads: int, optional
            The number of threads for parallel processing (default is 8).

    Returns:
        fitted_dataset: sidpy.Dataset or numpy.ndarray
            The dataset with fitted plasmon peak parameters. The dimensions and 
            format depend on the input dataset.

    Raises:
        ValueError: If the input dataset does not have the expected dimensions or format.

    Notes:
        - The function uses the Drude model to fit plasmon peaks.
        - The fitting parameters are peak position (e_p), peak width (e_w), and amplitude (A).
        - If `plot_result` is True, the function plots e_p, e_w, and A as separate subplots.
    """
    # define window for fitting
    energy = dataset.get_spectral_dims(return_axis=True)[0].values
    start_fit_pixel = np.searchsorted(energy, start_fit_energy)
    end_fit_pixel = np.searchsorted(energy, end_fit_energy)

    def errf_multi(p, y, x):
        elf = multiple_scattering(x, p)
        err = y - elf
        return np.abs(err)  # /np.sqrt(y)

    if pin is None:
        pin = np.array([9,1,.7, 0.3])
    fit_dset = np.array(dataset[start_fit_pixel:end_fit_pixel])
    popt, _ = scipy.optimize.leastsq(errf_multi, pin, args=(fit_dset,
                                               energy[start_fit_pixel:end_fit_pixel]), maxfev=2000)
    multi = dataset.like_data(multiple_scattering(energy, popt))
    multi.metadata['multiple_scattering'] = {'parameter': popt}
    return multi


def drude_simulation(dset, e, ep, ew, tnm, eb):
    """probabilities of dielectric function eps relative to zero-loss integral (i0 = 1)

    Gives probabilities of dielectric function eps relative to zero-loss integral (i0 = 1) per eV
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011

    # Given the plasmon energy (ep), plasmon fwhm (ew) and binding energy(eb),
    # this program generates:
    # EPS1, EPS2 from modified Eq. (3.40), ELF=Im(-1/EPS) from Eq. (3.42),
    # single scattering from Eq. (4.26) and SRFINT from Eq. (4.31)
    # The output is e, ssd into the file drude.ssd (for use in Flog etc.)
    # and e,eps1 ,eps2 into drude.eps (for use in Kroeger etc.)
    # Gives probabilities relative to zero-loss integral (i0 = 1) per eV
    # Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011
    # Version 10.11.26

    """
    energy_scale = dset.get_spectral_dims(return_axis=True)[0].values

    epc = energy_scale[1] - energy_scale[0]  # input('ev per channel : ')

    b = dset.metadata['collection_angle'] / 1000.  # rad
    epc = dset.energy_scale[1] - dset.energy_scale[0]  # input('ev per channel : ');
    e0 = dset.metadata['acceleration_voltage'] / 1000.  # input('incident energy e0(kev) : ');

    # effective kinetic energy: t = m_o v^2/2,
    # eV # equ.5.2a or Appendix E p 427
    t = 1000.0 * e0 * (1. + e0 / 1022.12) / (1.0 + e0 / 511.06)**2

    # 2 gamma t
    tgt = 1000 * e0 * (1022.12 + e0) / (511.06 + e0)  # eV  Appendix E p 427

    rk0 = 2590 * (1.0 + e0 / 511.06) * np.sqrt(2.0 * t / 511060)

    # os = e[0]
    ew_mod = eb
    tags = dset.metadata

    eps = 1 - (ep ** 2 - ew_mod * e * 1j) / (e ** 2 + 2 * e * ew * 1j)  # Mod drude term

    eps[np.nonzero(eps == 0.0)] = 1e-19
    elf = np.imag(-1 / eps)

    the = e / tgt  # varies with energy loss! # Appendix E p 427
    # srfelf = 4..*eps2./((1+eps1).^2+eps2.^2) - elf; %equivalent
    srfelf = np.imag(-4. / (1.0 + eps)) - elf  # for 2 surfaces
    angdep = np.arctan(b / the) / the - b / (b * b + the * the)
    srfint = angdep * srfelf / (3.1416 * 0.05292 * rk0 * t)  # probability per eV
    anglog = np.log(1.0 + b * b / the / the)
    i0 = dset.sum()  # *tags['counts2e']

    # 2 * t = m_0 v**2 !!!  a_0 = 0.05292 nm
    volint = abs(tnm / (np.pi * 0.05292 * t * 2.0) * elf * anglog)  # S equ 4.26% probability per eV
    volint = volint * i0 / epc  # S probability per channel
    ssd = volint  # + srfint

    if e[0] < -1.0:
        xs = int(abs(-e[0] / epc))

        ssd[0:xs] = 0.0
        volint[0:xs] = 0.0
        srfint[0:xs] = 0.0

        # if os <0:
        # 2 surfaces but includes negative Begrenzung contribution
        # p_s = np.trapezoid(e, srfint)

        # integrated volume probability
        p_v = abs(np.trapezoid(e, abs(volint / tags['spec'].sum())))
        # our data have he same epc and the trapez formula does not include
        p_v = (volint / i0).sum()
        # does NOT depend on free-electron approximation (no damping).
        lam = tnm / p_v
        # Eq.(3.44) approximation
        lamfe = 4.0 * 0.05292 * t / ep / np.log(1 + (b * tgt / ep) ** 2)

        tags['eps'] = eps
        tags['lam'] = lam
        tags['lamfe'] = lamfe
        tags['p_v'] = p_v

    return ssd  # /np.pi


def kroeger_core(e_data, a_data, eps_data, acceleration_voltage_kev, thickness, relativistic=True):
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
    

    $d^2P/(dEd\\Omega) = \\frac{1}{\\pi^2 a_0 m_0 v^2} \\Im \left[ \\frac{t\\mu^2}{\\varepsilon \\phi^2 } \right]
    
    # Internally everything is calculated in SI units
    # acceleration_voltage_kev = 200 #keV
    # thick = 32.0*10-9 # m
    """
    a_data = np.array(a_data)
    e_data = np.array(e_data)
    # adjust input to si units
    # wavelength = get_wavelength(acceleration_voltage_kev * 1e3)  # in m
    thickness = thickness * 1e-9  # input thickness now in m

    # Define constants
    m_0 = scipy.constants.electron_mass  # REST electron mass in kg
    hbar = scipy.constants.hbar

    c = scipy.constants.speed_of_light  # speed of light m/s
    bohr = scipy.constants.physical_constants['Bohr radius'][0]  # Bohr radius in meters
    e = scipy.constants.e  # electron charge in Coulomb

    # Calculate fixed terms of equation
    # acceleration_voltage_kev is incident energy in keV
    va = 1 - (511. / (511. + acceleration_voltage_kev))**2
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


def plot_dispersion(plotdata, units, a_data, e_data, max_p, title, ee):
    """Plot loss function """

    # [x, y] = np.meshgrid(e_data + 1e-12, a_data[1024:2048] * 1000)

    z = plotdata
    # lev = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 4.9]) * max_p / 5

    wavelength = get_wavelength(ee)
    # q = a_data[1024:2048] / (wavelength * 1e9)  # in [1/nm]
    scale = np.array([0, a_data[-1], e_data[0], e_data[-1]])
    ev2hertz = scipy.constants.value('electron volt-hertz relationship')

    if units[0] == 'mrad':
        units[0] = 'scattering angle [mrad]'
        scale[1] = scale[1] * 1000.
        light_line = scipy.constants.c * a_data  # for mrad
    elif units[0] == '1/nm':
        units[0] = 'scattering vector [1/nm]'
        scale[1] = scale[1] / (wavelength * 1e9)
        light_line = 1 / (scipy.constants.c / ev2hertz) * 1e-9

    if units[1] == 'eV':
        units[1] = 'energy loss [eV]'

    if units[2] == 'ppm':
        units[2] = 'probability [ppm]'
    if units[2] == '1/eV':
        units[2] = 'probability [eV$^{-1}$ srad$^{-1}$]'

    # alpha = 3. / 5. * ef / ep

    ax2 = plt.gca()
    fig2 = plt.gcf()
    fig2.suptitle(title)
    im = ax2.imshow(z.t, clim=(0, max_p), origin='lower', aspect='auto', extent=scale)
    # co = ax2.contour(y, x, z, levels=lev, colors='k', origin='lower')
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


def add_peaks(x, y, peaks, pin_in=None, peak_shape_in=None, shape='Gaussian'):
    """ add peaks to fitting parameters"""
    if pin_in is None:
        return [], []
    if peak_shape_in is None:
        return [], []
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


def model3(x, p, number_of_peaks, peak_shape, p_zl, pin=None, restrict_pos=0, restrict_width=0):
    """ model for fitting low-loss spectrum"""
    if pin is None:
        pin = p
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
            p[index + 2] = pin[index + 2]
            if p[index + 2] > pin[index + 2] * (1.0 + restrict_width):
                p[index + 2] = pin[index + 2] * (1.0 + restrict_width)
        if peak_shape[i] == 'Lorentzian':
            y = y + lorentz(x, p[index:])
        elif peak_shape[i] == 'zl':
            y = y + zl(x, p[index:], p_zl)
        else:
            y = y + gauss(x, p[index:])
    return y
