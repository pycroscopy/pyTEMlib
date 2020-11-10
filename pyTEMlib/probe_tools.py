import numpy as np


def make_gauss(size_x, size_y, width=1.0, x0=0.0, y0=0.0, intensity=1.0):
    size_x = size_x/2
    size_y = size_y/2
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = np.exp(-((x-x0)**2 + (y-y0)**2) / 2.0 / width**2)
    probe = g / g.sum() * intensity

    return probe


def make_lorentz(size_x, size_y, gamma=1.0, x0=0., y0=0., intensity=1.):
    size_x = np.floor(size_x / 2)
    size_y = np.floor(size_y / 2)
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = gamma / (2*np.pi) / np.power(((x-x0)**2 + (y-y0)**2 + gamma**2), 1.5)
    probe = g / g.sum() * intensity
    return probe


def zero_loss_peak_weight():
    # US100 zero_loss peak for Cc of aberrations
    x = np.linspace(-0.5, 0.9, 29)
    y = [0.0143, 0.0193, 0.0281, 0.0440, 0.0768, 0.1447, 0.2785, 0.4955, 0.7442, 0.9380, 1.0000, 0.9483, 0.8596,
         0.7620, 0.6539, 0.5515, 0.4478, 0.3500, 0.2683, 0.1979, 0.1410, 0.1021, 0.0752, 0.0545, 0.0401, 0.0300,
         0.0229, 0.0176, 0.0139]
    return x, y


# ## Aberration Function for Probe calculations
def make_chi1(phi, theta, wl, ab, c1_include):
    """
    # ##
    # Aberration function chi without defocus
    # ##
    """
    t0 = np.power(theta, 1) / 1 * (float(ab['C01a']) * np.cos(1 * phi) + float(ab['C01b']) * np.sin(1 * phi))

    if c1_include == 1:  # First and second terms
        t1 = np.power(theta, 2) / 2 * (ab['C10'] + ab['C12a'] * np.cos(2 * phi) + ab['C12b'] * np.sin(2 * phi))
    elif c1_include == 2:  # Second terms only
        t1 = np.power(theta, 2) / 2 * (ab['C12a'] * np.cos(2 * phi) + ab['C12b'] * np.sin(2 * phi))
    else:  # none for zero
        t1 = t0 * 0.
    t2 = np.power(theta, 3) / 3 * (ab['C21a'] * np.cos(1 * phi) + ab['C21b'] * np.sin(1 * phi)
                                   + ab['C23a'] * np.cos(3 * phi) + ab['C23b'] * np.sin(3 * phi))

    t3 = np.power(theta, 4) / 4 * (ab['C30']
                                   + ab['C32a'] * np.cos(2 * phi)
                                   + ab['C32b'] * np.sin(2 * phi)
                                   + ab['C34a'] * np.cos(4 * phi)
                                   + ab['C34b'] * np.sin(4 * phi))

    t4 = np.power(theta, 5) / 5 * (ab['C41a'] * np.cos(1 * phi)
                                   + ab['C41b'] * np.sin(1 * phi)
                                   + ab['C43a'] * np.cos(3 * phi)
                                   + ab['C43b'] * np.sin(3 * phi)
                                   + ab['C45a'] * np.cos(5 * phi)
                                   + ab['C45b'] * np.sin(5 * phi))

    t5 = np.power(theta, 6) / 6 * (ab['C50']
                                   + ab['C52a'] * np.cos(2 * phi)
                                   + ab['C52b'] * np.sin(2 * phi)
                                   + ab['C54a'] * np.cos(4 * phi)
                                   + ab['C54b'] * np.sin(4 * phi)
                                   + ab['C56a'] * np.cos(6 * phi)
                                   + ab['C56b'] * np.sin(6 * phi))

    chi = t0 + t1 + t2 + t3 + t4 + t5
    if 'C70' in ab:
        chi += np.power(theta, 8) / 8 * (ab['C70'])
    return chi * 2 * np.pi / wl  # np.power(theta,6)/6*(  ab['C50'] )


def probe2(ab, size_x, size_y, tags, verbose=False):
    """

    * This function creates a incident STEM probe
    * at position (0,0)
    * with parameters given in ab dictionary
    *
    * The following Aberration functions are being used:
    * 1) ddf = Cc*de/E  but not  + Cc2*(de/E)^2,
    *    Cc, Cc2 = chrom. Aber. (1st, 2nd order) [1]
    * 2) chi(qx,qy) = (2*pi/lambda)*{0.5*C1*(qx^2+qy^2)+
    *                 0.5*C12a*(qx^2-qy^2)+
    *                 C12b*qx*qy+
    *                 C21a/3*qx*(qx^2+qy^2)+
    *                 ...
    *                 +0.5*C3*(qx^2+qy^2)^2
    *                 +0.125*C5*(qx^2+qy^2)^3
    *                 ... (need to finish)
    *
    *
    *    qx = acos(kx/K), qy = acos(ky/K)
    *
    * References:
    * [1] J. Zach, M. Haider,
    *    "Correction of spherical and Chromatic Aberration
    *     in a low Voltage SEM", Optik 98 (3), 112-118 (1995)
    * [2] O.L. Krivanek, N. Delby, A.R. Lupini,
    *    "Towards sub-Angstroem Electron Beams",
    *    Ultramicroscopy 78, 1-11 (1999)
    *


    # Internally reciprocal lattice vectors in 1/nm or rad.
    # All calculations of chi in angles.
    # All aberration coefficients in nm
    """

    if 'fov' not in ab:
        if 'fov' not in tags:
            print(' need field of view in tags ')
        else:
            ab['fov'] = tags['fov']

    if 'convAngle' not in ab:
        ab['convAngle'] = 30  # in mrad

    ap_angle = ab['convAngle'] / 1000.0  # in rad

    e0 = ab['EHT'] = float(ab['EHT'])  # acceleration voltage in ev

    # defocus = ab['C10']

    if 'C01a' not in ab:
        ab['C01a'] = 0.
    if 'C01b' not in ab:
        ab['C01b'] = 0.

    if 'C50' not in ab:
        ab['C50'] = 0.
    if 'C70' not in ab:
        ab['C70'] = 0.

    if 'Cc' not in ab:
        ab['Cc'] = 1.3e6  # Cc in  nm

    def get_wl():
        h = 6.626 * 10 ** -34
        m0 = 9.109 * 10 ** -31
        ev = 1.602 * 10 ** -19 * e0
        c = 2.998 * 10 ** 8
        return h / np.sqrt(2 * m0 * ev * (1 + ev / (2 * m0 * c ** 2))) * 10 ** 9

    wl = get_wl()
    if verbose:
        print('Acceleration voltage {0:}kV => wavelength {1:.2f}pm'.format(int(e0 / 1000), wl * 1000))
    ab['wavelength'] = wl

    # Reciprocal plane in 1/nm
    dk = 1 / ab['fov']
    kx = np.array(dk * (-size_x / 2. + np.arange(size_x)))
    ky = np.array(dk * (-size_y / 2. + np.arange(size_y)))
    t_xv, t_yv = np.meshgrid(kx, ky)

    # define reciprocal plane in angles
    phi = np.arctan2(t_xv, t_yv)
    theta = np.arctan2(np.sqrt(t_xv ** 2 + t_yv ** 2), 1 / wl)

    # calculate chi but omit defocus
    chi = np.fft.ifftshift(make_chi1(phi, theta, wl, ab, 2))
    probe = np.zeros((size_x, size_y))

    # Aperture function
    mask = theta >= ap_angle

    # Calculate probe with Cc

    for i in range(len(ab['zeroLoss'])):
        df = ab['C10'] + ab['Cc'] * ab['zeroEnergy'][i] / e0
        if verbose:
            print('defocus due to Cc: {0:.2f} nm with weight {1:.2f}'.format(df, ab['zeroLoss'][i]))
        # Add defocus
        chi2 = chi + np.power(theta, 2) / 2 * df
        # Calculate exponent of - i * chi
        chi_t = np.fft.ifftshift(np.vectorize(complex)(np.cos(chi2), -np.sin(chi2)))
        # Apply aperture function
        chi_t[mask] = 0.
        # inverse fft of aberration function
        i2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(chi_t)))
        # add intensities
        probe = probe + np.real(i2 * np.conjugate(i2)).T * ab['zeroLoss'][i]

    ab0 = {}
    for key in ab:
        ab0[key] = 0.
    # chiIA = np.fft.fftshift(make_chi1(phi, theta, wl, ab0, 0))  # np.ones(chi2.shape)*2*np.pi/wl
    chi_i = np.ones((size_y, size_x))
    chi_i[mask] = 0.
    i2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(chi_i)))
    ideal = np.real(i2 * np.conjugate(i2))

    probe_f = np.fft.fft2(probe, probe.shape) + 1e-12
    ideal_f = np.fft.fft2(ideal, probe.shape)
    fourier_space_division = ideal_f / probe_f
    probe_r = (np.fft.ifft2(fourier_space_division, probe.shape))

    return probe / sum(ab['zeroLoss']), np.real(probe_r)
