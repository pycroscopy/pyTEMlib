import scipy
import numpy as np
from ..utilities import get_wavelength
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
    wavelength = get_wavelength(acceleration_voltage_kev * 1e3)  # in m
    thickness = thickness * 1e-9  # input thickness now in m

    # Define constants
    # ec = 14.4;
    m_0 = scipy.constants.value(u'electron mass')  # REST electron mass in kg
    # h = scipy.constants.Planck  # Planck's constant
    hbar = scipy.constants.hbar

    c = scipy.constants.speed_of_light  # speed of light m/s
    bohr = scipy.constants.value(u'Bohr radius')  # Bohr radius in meters
    e = scipy.constants.value(u'elementary charge')  # electron charge in Coulomb
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
