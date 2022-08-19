"""Functions to calculate electron probe"""
import numpy as np
import pyTEMlib.image_tools
import scipy.ndimage as ndimage


def make_gauss(size_x, size_y, width=1.0, x0=0.0, y0=0.0, intensity=1.0):
    """Make a Gaussian shaped probe """
    size_x = size_x / 2
    size_y = size_y / 2
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / width ** 2)
    probe = g / g.sum() * intensity

    return probe


def make_lorentz(size_x, size_y, gamma=1.0, x0=0., y0=0., intensity=1.):
    """Make a Lorentzian shaped probe """

    size_x = np.floor(size_x / 2)
    size_y = np.floor(size_y / 2)
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = gamma / (2 * np.pi) / np.power(((x - x0) ** 2 + (y - y0) ** 2 + gamma ** 2), 1.5)
    probe = g / g.sum() * intensity
    return probe


def zero_loss_peak_weight():
    # US100 zero_loss peak for Cc of aberrations
    x = np.linspace(-0.5, 0.9, 29)
    y = [0.0143, 0.0193, 0.0281, 0.0440, 0.0768, 0.1447, 0.2785, 0.4955, 0.7442, 0.9380, 1.0000, 0.9483, 0.8596,
         0.7620, 0.6539, 0.5515, 0.4478, 0.3500, 0.2683, 0.1979, 0.1410, 0.1021, 0.0752, 0.0545, 0.0401, 0.0300,
         0.0229, 0.0176, 0.0139]
    return x, y


def make_chi(phi, theta, aberrations):
    maximum_aberration_order = 5
    chi = np.zeros(theta.shape)
    for n in range(maximum_aberration_order + 1):  # First Sum up to fifth order
        term_first_sum = np.power(theta, n + 1) / (n + 1)  # term in first sum

        second_sum = np.zeros(theta.shape)  # second Sum initialized with zeros
        for m in range((n + 1) % 2, n + 2, 2):
            if m > 0:
                if f'C{n}{m}a' not in aberrations:  # Set non existent aberrations coefficient to zero
                    aberrations[f'C{n}{m}a'] = 0.
                if f'C{n}{m}b' not in aberrations:
                    aberrations[f'C{n}{m}b'] = 0.

                # term in second sum
                second_sum = second_sum + aberrations[f'C{n}{m}a'] * np.cos(m * phi) + aberrations[
                    f'C{n}{m}b'] * np.sin(m * phi)
            else:
                if f'C{n}{m}' not in aberrations:  # Set non existent aberrations coefficient to zero
                    aberrations[f'C{n}{m}'] = 0.

                # term in second sum
                second_sum = second_sum + aberrations[f'C{n}{m}']
        chi = chi + term_first_sum * second_sum * 2 * np.pi / aberrations['wavelength']

    return chi


def get_chi(ab, size_x, size_y, verbose=False):
    """  Get aberration function chi without defocus spread

    # Internally reciprocal lattice vectors in 1/nm or rad.
    # All calculations of chi in angles.
    # All aberration coefficients in nm
    """
    aperture_angle = ab['convergence_angle'] / 1000.0  # in rad

    wavelength = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])
    if verbose:
        print(f"Acceleration voltage {ab['acceleration_voltage'] / 1000:}kV => wavelength {wavelength * 1000.:.2f}pm")

    ab['wavelength'] = wavelength

    # Reciprocal plane in 1/nm
    dk = 1 / ab['FOV']
    k_x = np.array(dk * (-size_x / 2. + np.arange(size_x)))
    k_y = np.array(dk * (-size_y / 2. + np.arange(size_y)))
    t_x_v, t_y_v = np.meshgrid(k_x, k_y)

    # define reciprocal plane in angles
    phi = np.arctan2(t_x_v, t_y_v)
    theta = np.arctan2(np.sqrt(t_x_v ** 2 + t_y_v ** 2), 1 / wavelength)

    # calculate chi
    chi = make_chi(phi, theta, ab)

    # Aperture function
    mask = theta >= aperture_angle

    aperture = np.ones((size_x, size_y), dtype=float)
    aperture[mask] = 0.

    return chi, aperture


def print_aberrations(ab):
    from IPython.display import HTML, display
    output = '<html><body>'
    output += f"Aberrations [nm] for acceleration voltage: {ab['acceleration_voltage'] / 1e3:.0f} kV"
    output += '<table>'
    output += f"<tr><td> C10 </td><td> {ab['C10']:.1f} </tr>"
    output += f"<tr><td> C12a </td><td> {ab['C12a']:20.1f} <td> C12b </td><td> {ab['C12b']:20.1f} </tr>"
    output += f"<tr><td> C21a </td><td> {ab['C21a']:.1f} <td> C21b </td><td> {ab['C21b']:.1f} "
    output += f"    <td> C23a </td><td> {ab['C23a']:.1f} <td> C23b </td><td> {ab['C23b']:.1f} </tr>"
    output += f"<tr><td> C30 </td><td> {ab['C30']:.1f} </tr>"
    output += f"<tr><td> C32a </td><td> {ab['C32a']:20.1f} <td> C32b </td><td> {ab['C32b']:20.1f} "
    output += f"<td> C34a </td><td> {ab['C34a']:20.1f} <td> C34b </td><td> {ab['C34b']:20.1f} </tr>"
    output += f"<tr><td> C41a </td><td> {ab['C41a']:.3g} <td> C41b </td><td> {ab['C41b']:.3g} "
    output += f"    <td> C43a </td><td> {ab['C43a']:.3g} <td> C43b </td><td> {ab['C41b']:.3g} "
    output += f"    <td> C45a </td><td> {ab['C45a']:.3g} <td> C45b </td><td> {ab['C45b']:.3g} </tr>"
    output += f"<tr><td> C50 </td><td> {ab['C50']:.3g} </tr>"
    output += f"<tr><td> C52a </td><td> {ab['C52a']:20.1f} <td> C52b </td><td> {ab['C52b']:20.1f} "
    output += f"<td> C54a </td><td> {ab['C54a']:20.1f} <td> C54b </td><td> {ab['C54b']:20.1f} "
    output += f"<td> C56a </td><td> {ab['C56a']:20.1f} <td> C56b </td><td> {ab['C56b']:20.1f} </tr>"
    output += f"<tr><td> Cc </td><td> {ab['Cc']:.3g} </tr>"

    output += '</table></body></html>'

    display(HTML(output))


def get_ronchigram(size, ab, scale='mrad'):
    """ Get Ronchigram

    """
    size_x = size_y = size
    chi, A_k = get_chi(ab, size_x, size_y)

    v_noise = np.random.rand(size_x, size_y)
    smoothing = 5
    phi_r = ndimage.gaussian_filter(v_noise, sigma=(smoothing, smoothing), order=0)

    sigma = 6  # 6 for carbon and thin

    q_r = np.exp(-1j * sigma * phi_r)
    # q_r = 1-phi_r * sigma

    T_k = A_k * (np.exp(-1j * chi))
    t_r = (np.fft.ifft2(np.fft.fftshift(T_k)))

    psi_k = np.fft.fftshift(np.fft.fft2(q_r * t_r))

    ronchigram = np.absolute(psi_k * np.conjugate(psi_k))

    fov_reciprocal = 1 / ab['FOV'] * size_x / 2
    if scale == '1/nm':
        extent = [-fov_reciprocal, fov_reciprocal, -fov_reciprocal, fov_reciprocal]
        ylabel = 'reciprocal distance [1/nm]'
    else:
        fov_mrad = fov_reciprocal * ab['wavelength'] * 1000
        extent = [-fov_mrad, fov_mrad, -fov_mrad, fov_mrad]
        ylabel = 'reciprocal distance [mrad]'

    ab['ronchi_extent'] = extent
    ab['ronchi_label'] = ylabel
    return ronchigram


def get_chi_2(ab, u, v):
    chi1 = ab['C10'] * (u ** 2 + v ** 2) / 2 \
           + ab['C12a'] * (u ** 2 - v ** 2) / 2 \
           - ab['C12b'] * u * v

    chi2 = ab['C21a'] * (u ** 3 + u * v ** 2) / 3 \
           - ab['C21b'] * (u ** 2 * v + v ** 3) / 3 \
           + ab['C23a'] * (u ** 3 - 3 * u * v ** 2) / 3 \
           - ab['C23b'] * (3 * u ** 2 * v - v ** 3) / 3

    chi3 = ab['C30'] * (u ** 4 + 2 * u ** 2 * v ** 2 + v ** 4) / 4 \
           + ab['C32a'] * (u ** 4 - v ** 4) / 4 \
           - ab['C32b'] * (u ** 3 * v + u * v ** 3) / 2 \
           + ab['C34a'] * (u ** 4 - 6 * u ** 2 * v ** 2 + v ** 4) / 4 \
           - ab['C34b'] * (4 * u ** 3 * v - 4 * u * v ** 3) / 4

    chi4 = ab['C41a'] * (u ** 5 + 2 * u ** 3 * v ** 2 + u * v ** 4) / 5 \
           - ab['C41b'] * (u ** 4 * v + 2 * u ** 2 * v ** 3 + v ** 5) / 5 \
           + ab['C43a'] * (u ** 5 - 2 * u ** 3 * v ** 2 - 3 * u * v ** 4) / 5 \
           - ab['C43b'] * (3 * u ** 4 * v + 2 * u ** 2 * v ** 3 - v ** 5) / 5 \
           + ab['C45a'] * (u ** 5 - 10 * u ** 3 * v ** 2 + 5 * u * v ** 4) / 5 \
           - ab['C45b'] * (5 * u ** 4 * v - 10 * u ** 2 * v ** 3 + v ** 5) / 5

    chi5 = ab['C50'] * (u ** 6 + 3 * u ** 4 * v ** 2 + 3 * u ** 2 * v ** 4 + v ** 6) / 6 \
           + ab['C52a'] * (u ** 6 + u ** 4 * v ** 2 - u ** 2 * v ** 4 - v ** 6) / 6 \
           - ab['C52b'] * (2 * u ** 5 * v + 4 * u ** 3 * v ** 3 + 2 * u * v ** 5) / 6 \
           + ab['C54a'] * (u ** 6 - 5 * u ** 4 * v ** 2 - 5 * u ** 2 * v ** 4 + v ** 6) / 6 \
           - ab['C54b'] * (4 * u ** 5 * v - 4 * u * v ** 5) / 6 \
           + ab['C56a'] * (u ** 6 - 15 * u ** 4 * v ** 2 + 15 * u ** 2 * v ** 4 - v ** 6) / 6 \
           - ab['C56b'] * (6 * u ** 5 * v - 20 * u ** 3 * v ** 3 + 6 * u * v ** 5) / 6

    chi = chi1 + chi2 + chi3 + chi4 + chi5
    return chi * 2 * np.pi / ab['wavelength']


def get_d2chidu2(ab, u, v):
    d2chi1du2 = ab['C10'] + ab['C12a']

    d2chi2du2 = ab['C21a'] * 2 * u \
                - ab['C21b'] * 2 / 3 * v \
                + ab['C23a'] * 2 * u \
                - ab['C23b'] * 2 * v

    d2chi3du2 = ab['C30'] * (3 * u ** 2 + v ** 2) \
                + ab['C32a'] * 3 * u ** 2 \
                - ab['C32b'] * 3 * u * v \
                + ab['C34a'] * (3 * u ** 2 - 3 * v ** 2) \
                - ab['C34b'] * 6 * u * v

    d2chi4du2 = ab['C41a'] * 4 / 5 * (5 * u ** 3 + 3 * u * v ** 2) \
                - ab['C41b'] * 4 / 5 * (3 * u ** 2 * v + v ** 3) \
                + ab['C43a'] * 4 / 5 * (5 * u ** 3 - 3 * u * v ** 2) \
                - ab['C43b'] * 4 / 5 * (9 * u ** 2 * v + v ** 3) \
                + ab['C45a'] * 4 * (u ** 3 - 3 * u * v ** 2) \
                - ab['C45b'] * 4 * (3 * u ** 2 * v - v ** 3)

    d2chi5du2 = ab['C50'] * (5 * u ** 4 + 6 * u ** 2 * v ** 2 + v ** 4) \
                + ab['C52a'] * (15 * u ** 4 + 6 * u ** 2 * v ** 2 - v ** 4) / 3 \
                - ab['C52b'] * (20 * u ** 3 * v + 12 * u * v ** 3) / 3 \
                + ab['C54a'] * 5 / 3 * (3 * u ** 4 - 6 * u ** 2 * v ** 2 - v ** 4) \
                - ab['C54b'] * 5 / 3 * (8 * u ** 3 * v) \
                + ab['C56a'] * 5 * (u ** 4 - 6 * u ** 2 * v ** 2 + v ** 4) \
                - ab['C56b'] * 20 * (u ** 3 * v - u * v ** 3)

    d2chidu2 = d2chi1du2 + d2chi2du2 + d2chi3du2 + d2chi4du2 + d2chi5du2
    return d2chidu2


def get_d2chidudv(ab, u, v):
    d2chi1dudv = -ab['C12b']

    d2chi2dudv = ab['C21a'] * 2 / 3 * v \
                 - ab['C21b'] * 2 / 3 * u \
                 - ab['C23a'] * 2 * v \
                 - ab['C23b'] * 2 * u

    d2chi3dudv = ab['C30'] * 2 * u * v \
                 + ab['C32a'] * 0 \
                 - ab['C32b'] * 3 / 2 * (u ** 2 + v ** 2) \
                 - ab['C34a'] * 6 * u * v \
                 - ab['C34b'] * 3 * (u ** 2 - v ** 2)

    d2chi4dudv = ab['C41a'] * 4 / 5 * (3 * u ** 2 * v + v ** 3) \
                 - ab['C41b'] * 4 / 5 * (u ** 3 + 3 * u * v ** 2) \
                 - ab['C43a'] * 12 / 5 * (u ** 2 * v + v ** 3) \
                 - ab['C43b'] * 12 / 5 * (u ** 3 + u * v ** 2) \
                 - ab['C45a'] * 4 * (3 * u ** 2 * v - v ** 3) \
                 - ab['C45b'] * 4 * (u ** 3 - 3 * u * v ** 2)

    d2chi5dudv = ab['C50'] * 4 * u * v * (u ** 2 + v ** 2) \
                 + ab['C52a'] * 4 / 3 * (u ** 3 * v - u * v ** 3) \
                 - ab['C52b'] * (5 * u ** 4 + 18 * u ** 2 * v ** 2 + 5 * v ** 4) / 3 \
                 - ab['C54a'] * 20 / 3 * (u ** 3 * v + u * v ** 3) \
                 - ab['C54b'] * 10 / 3 * (u ** 4 - v ** 4) \
                 - ab['C56a'] * 20 * (u ** 3 * v - u * v ** 3) \
                 - ab['C56b'] * 5 * (u ** 4 - 6 * u ** 2 * v ** 2 + v ** 4)

    d2chidudv = d2chi1dudv + d2chi2dudv + d2chi3dudv + d2chi4dudv + d2chi5dudv
    return d2chidudv


def get_d2chidv2(ab, u, v):
    d2chi1dv2 = ab['C10'] - ab['C12a']

    d2chi2dv2 = ab['C21a'] * 2 / 3 * u \
                - ab['C21b'] * 2 * v \
                - ab['C23a'] * 2 * u \
                + ab['C23b'] * 2 * v

    d2chi3dv2 = ab['C30'] * (u ** 2 + 3 * v ** 2) \
                - ab['C32a'] * 3 * v ** 2 \
                - ab['C32b'] * 3 * v * u \
                - ab['C34a'] * 3 * (u ** 2 - v ** 2) \
                + ab['C34b'] * 6 * u * v

    d2chi4dv2 = ab['C41a'] * 4 / 5 * (u ** 3 + 3 * u * v ** 2) \
                - ab['C41b'] * 4 / 5 * (3 * u ** 2 * v + 5 * v ** 3) \
                - ab['C43a'] * 4 / 5 * (u ** 3 + 9 * u * v ** 2) \
                - ab['C43b'] * 4 / 5 * (3 * u ** 2 * v - 5 * v ** 3) \
                - ab['C45a'] * 4 * (u ** 3 - 3 * u * v ** 2) \
                + ab['C45b'] * 4 * (3 * u ** 2 * v - v ** 3)

    d2chi5dv2 = ab['C50'] * (u ** 4 + 6 * u ** 2 * v ** 2 + 5 * v ** 4) \
                + ab['C52a'] * (u ** 4 - 6 * u ** 2 * v ** 2 - 15 * v ** 4) / 3 \
                - ab['C52b'] * (12 * u ** 3 * v + 20 * u * v ** 3) / 3 \
                - ab['C54a'] * 5 / 3 * (u ** 4 + 6 * u ** 2 * v ** 2 - 3 * v ** 4) \
                + ab['C54b'] * 40 / 3 * u * v ** 3 \
                - ab['C56a'] * 5 * (u ** 4 - 6 * u ** 2 * v ** 2 + v ** 4) \
                + ab['C56b'] * 20 * (u ** 3 * v - u * v ** 3)

    d2chidv2 = d2chi1dv2 + d2chi2dv2 + d2chi3dv2 + d2chi4dv2 + d2chi5dv2
    return d2chidv2


def get_source_energy_spread():
    x = np.linspace(-0.5, .9, 29)
    y = [0.0143, 0.0193, 0.0281, 0.0440, 0.0768, 0.1447, 0.2785, 0.4955, 0.7442, 0.9380, 1.0000, 0.9483, 0.8596, 0.7620,
         0.6539, 0.5515, 0.4478, 0.3500, 0.2683, 0.1979, 0.1410, 0.1021, 0.0752, 0.0545, 0.0401, 0.0300, 0.0229, 0.0176,
         0.0139]

    return x, y


def get_target_aberrations(TEM_name, acceleration_voltage):
    ab = {}
    if TEM_name == 'NionUS200':
        if int(acceleration_voltage) == 200000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV for Aberrations of {TEM_name}****')
            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': -335., 'C21b': 283., 'C23a': -34., 'C23b': 220.,
                  'C30': -8080.,
                  'C32a': 18800., 'C32b': -2260., 'C34a': 949., 'C34b': 949., 'C41a': 54883., 'C41b': -464102.,
                  'C43a': 77240.5,
                  'C43b': -540842., 'C45a': -79844.4, 'C45b': -76980.8, 'C50': 9546970., 'C52a': -2494290.,
                  'C52b': 2999910.,
                  'C54a': -2020140., 'C54b': -2019630., 'C56a': -535079., 'C56b': 1851850.}
            ab['source_size'] = 0.051
            ab['acceleration_voltage'] = acceleration_voltage
            ab['convergence_angle'] = 30

            ab['Cc'] = 1.3e6  # // Cc in  nm

        if int(acceleration_voltage) == 100000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV for Aberrations of {TEM_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 157., 'C21b': 169, 'C23a': -173., 'C23b': 48.7, 'C30': 201.,
                  'C32a': 1090., 'C32b': 6840., 'C34a': 1010., 'C34b': 79.9, 'C41a': -210696., 'C41b': -262313.,
                  'C43a': 348450., 'C43b': -9.7888e4, 'C45a': 6.80247e4, 'C45b': -3.14637e1, 'C50': -193896.,
                  'C52a': -1178950, 'C52b': -7414340, 'C54a': -1753890, 'C54b': -1753890, 'C56a': -631786,
                  'C56b': -165705}
            ab['source_size'] = 0.051
            ab['acceleration_voltage'] = acceleration_voltage
            ab['convergence_angle'] = 30
            ab['Cc'] = 1.3e6

        if int(acceleration_voltage) == 60000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV for Aberrations of {TEM_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 11.5, 'C21b': 113, 'C23a': -136., 'C23b': 18.2, 'C30': 134.,
                  'C32a': 1080., 'C32b': 773., 'C34a': 1190., 'C34b': -593., 'C41a': -179174., 'C41b': -350378.,
                  'C43a': 528598, 'C43b': -257349., 'C45a': 63853.4, 'C45b': 1367.98, 'C50': 239021., 'C52a': 1569280.,
                  'C52b': -6229310., 'C54a': -3167620., 'C54b': -449198., 'C56a': -907315., 'C56b': -16281.9}
            ab['source_size'] = 0.081
            ab['acceleration_voltage'] = acceleration_voltage
            ab['convergence_angle'] = 30
            ab['Cc'] = 1.3e6  # // Cc in  nm

        ab['origin'] = 'target aberrations'
        ab['TEM_name'] = TEM_name
        ab['wavelength'] = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])

    if TEM_name == 'NionUS100':
        if int(acceleration_voltage) == 100000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV for Aberrations of {TEM_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 157., 'C21b': 169, 'C23a': -173., 'C23b': 48.7, 'C30': 201.,
                  'C32a': 1090., 'C32b': 6840., 'C34a': 1010., 'C34b': 79.9, 'C41a': -210696., 'C41b': -262313.,
                  'C43a': 348450., 'C43b': -9.7888e4, 'C45a': 6.80247e4, 'C45b': -3.14637e1, 'C50': -193896.,
                  'C52a': -1178950, 'C52b': -7414340, 'C54a': -1753890, 'C54b': -1753890, 'C56a': -631786,
                  'C56b': -165705}
            ab['source_size'] = 0.051
            ab['acceleration_voltage'] = acceleration_voltage
            ab['convergence_angle'] = 30
            ab['Cc'] = 1.3e6  # // Cc in  nm

        if int(acceleration_voltage) == 60000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV for Aberrations of {TEM_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 11.5, 'C21b': 113, 'C23a': -136., 'C23b': 18.2, 'C30': 134.,
                  'C32a': 1080., 'C32b': 773., 'C34a': 1190., 'C34b': -593., 'C41a': -179174., 'C41b': -350378.,
                  'C43a': 528598, 'C43b': -257349., 'C45a': 63853.4, 'C45b': 1367.98, 'C50': 239021., 'C52a': 1569280.,
                  'C52b': -6229310., 'C54a': -3167620., 'C54b': -449198., 'C56a': -907315., 'C56b': -16281.9}
            ab['source_size'] = 0.081
            ab['acceleration_voltage'] = acceleration_voltage
            ab['convergence_angle'] = 30
            ab['Cc'] = 1.3e6  # // Cc in  nm

        ab['origin'] = 'target aberrations'
        ab['TEM_name'] = TEM_name
        ab['wavelength'] = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])

    if TEM_name == 'ZeissMC200':
        ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 0, 'C21b': 0, 'C23a': 0, 'C23b': 0, 'C30': 0.,
              'C32a': 0., 'C32b': -0., 'C34a': 0., 'C34b': 0., 'C41a': 0., 'C41b': -0., 'C43a': 0.,
              'C43b': -0., 'C45a': -0., 'C45b': -0., 'C50': 0., 'C52a': -0., 'C52b': 0.,
              'C54a': -0., 'C54b': -0., 'C56a': -0., 'C56b': 0.}
        ab['C30'] = 2.2 * 1e6

        ab['Cc'] = 2.0 * 1e6

        ab['source_size'] = 0.2
        ab['acceleration_voltage'] = acceleration_voltage
        ab['convergence_angle'] = 10

        ab['origin'] = 'target aberrations'
        ab['TEM_name'] = TEM_name

        ab['wavelength'] = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])
    return ab


def get_ronchigram_2(size, ab, scale='mrad', threshold=3):
    aperture_angle = ab['convergence_angle'] / 1000.0  # in rad

    wavelength = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])
    # if verbose:
    #    print(f"Acceleration voltage {ab['acceleration_voltage']/1000:}kV => wavelength {wavelength*1000.:.2f}pm")

    ab['wavelength'] = wavelength

    size_x = size_y = size

    # Reciprocal plane in 1/nm
    dk = ab['reciprocal_FOV'] / size
    k_x = np.array(dk * (-size_x / 2. + np.arange(size_x)))
    k_y = np.array(dk * (-size_y / 2. + np.arange(size_y)))
    t_x_v, t_y_v = np.meshgrid(k_x, k_y)

    chi = get_chi_2(ab, t_x_v, t_y_v)  # , verbose= True)
    # define reciprocal plane in angles
    phi = np.arctan2(t_x_v, t_y_v)
    theta = np.arctan2(np.sqrt(t_x_v ** 2 + t_y_v ** 2), 1 / wavelength)

    # Aperture function
    mask = theta >= aperture_angle

    aperture = np.ones((size_x, size_y), dtype=float)
    aperture[mask] = 0.

    v_noise = np.random.rand(size_x, size_y)
    smoothing = 5
    phi_r = ndimage.gaussian_filter(v_noise, sigma=(smoothing, smoothing), order=0)

    sigma = 6  # 6 for carbon and thin

    q_r = np.exp(-1j * sigma * phi_r)
    # q_r = 1-phi_r * sigma

    T_k = aperture * (np.exp(-1j * chi))
    t_r = np.fft.ifft2(np.fft.fftshift(T_k))

    Psi_k = np.fft.fftshift(np.fft.fft2(q_r * t_r))

    ronchigram = I_k = np.absolute(Psi_k * np.conjugate(Psi_k))

    fov_reciprocal = ab['reciprocal_FOV']
    if scale == '1/nm':
        extent = [-fov_reciprocal, fov_reciprocal, -fov_reciprocal, fov_reciprocal]
        ylabel = 'reciprocal distance [1/nm]'
    else:
        fov_mrad = fov_reciprocal * ab['wavelength'] * 1000
        extent = [-fov_mrad, fov_mrad, -fov_mrad, fov_mrad]
        ylabel = 'reciprocal distance [mrad]'

    ab['ronchi_extent'] = extent
    ab['ronchi_label'] = ylabel

    h = np.zeros([chi.shape[0], chi.shape[1], 2, 2])
    h[:, :, 0, 0] = get_d2chidu2(ab, t_x_v, t_y_v)
    h[:, :, 0, 1] = get_d2chidudv(ab, t_x_v, t_y_v)
    h[:, :, 1, 0] = get_d2chidudv(ab, t_x_v, t_y_v)
    h[:, :, 1, 1] = get_d2chidv2(ab, t_x_v, t_y_v)

    # get Eigenvalues
    _, s, _ = np.linalg.svd(h)

    # get smallest Eigenvalue per pixel
    infinite_magnification = np.min(s, axis=2)

    # set all values below a threshold value to one, otherwise 0
    infinite_magnification[infinite_magnification <= threshold] = 1
    infinite_magnification[infinite_magnification > threshold] = 0

    return ronchigram, infinite_magnification


# ## Aberration Function for Probe calculations
def make_chi1(phi, theta, wavelength, ab, c1_include):
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

    return chi * 2 * np.pi / wavelength  # np.power(theta,6)/6*(  ab['C50'] )


def probe2(ab, size_x, size_y, tags, verbose=False):
    """

    * This function creates an incident STEM probe
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
    *    qx = acos(k_x/K), qy = acos(k_y/K)
    *
    * References:
    * [1] J. Zach, M. Haider,
    *    "Correction of spherical and Chromatic Aberration
    *     in a low Voltage SEM", Optik 98 (3), 112-118 (1995)
    * [2] O.L. Krivanek, N. Delby, A.R. Lupini,
    *    "Towards sub-Angstrom Electron Beams",
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

    wavelength = get_wl()
    if verbose:
        print('Acceleration voltage {0:}kV => wavelength {1:.2f}pm'.format(int(e0 / 1000), wavelength * 1000))
    ab['wavelength'] = wavelength

    # Reciprocal plane in 1/nm
    dk = 1 / ab['fov']
    k_x = np.array(dk * (-size_x / 2. + np.arange(size_x)))
    k_y = np.array(dk * (-size_y / 2. + np.arange(size_y)))
    t_xv, t_yv = np.meshgrid(k_x, k_y)

    # define reciprocal plane in angles
    phi = np.arctan2(t_xv, t_yv)
    theta = np.arctan2(np.sqrt(t_xv ** 2 + t_yv ** 2), 1 / wavelength)

    # calculate chi but omit defocus
    chi = np.fft.ifftshift(make_chi1(phi, theta, wavelength, ab, 2))
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
    # chiIA = np.fft.fftshift(make_chi1(phi, theta, wavelength, ab0, 0))  # np.ones(chi2.shape)*2*np.pi/wavelength
    chi_i = np.ones((size_y, size_x))
    chi_i[mask] = 0.
    i2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(chi_i)))
    ideal = np.real(i2 * np.conjugate(i2))

    probe_f = np.fft.fft2(probe, probe.shape) + 1e-12
    ideal_f = np.fft.fft2(ideal, probe.shape)
    fourier_space_division = ideal_f / probe_f
    probe_r = (np.fft.ifft2(fourier_space_division, probe.shape))

    return probe / sum(ab['zeroLoss']), np.real(probe_r)
