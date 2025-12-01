"""Functions to calculate electron probe"""
import numpy as np
import scipy
import skimage

import pyTEMlib

def make_gauss(size_x, size_y, width=1.0, x0=0.0, y0=0.0, intensity=1.0):
    """Make a Gaussian shaped probe """
    size_x = size_x / 2
    size_y = size_y / 2
    x, y = np.mgrid[-size_x:size_x, -size_y:size_y]
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / width ** 2)
    probe = g / (g.sum()+ 0.00001) * intensity

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
    """ US100 zero_loss peak for Cc of aberrations"""
    x = np.linspace(-0.5, 0.9, 29)
    y = [0.0143, 0.0193, 0.0281, 0.0440, 0.0768, 0.1447, 0.2785, 0.4955,
         0.7442, 0.9380, 1.0000, 0.9483, 0.8596, 0.7620, 0.6539, 0.5515,
         0.4478, 0.3500, 0.2683, 0.1979, 0.1410, 0.1021, 0.0752, 0.0545,
         0.0401, 0.0300, 0.0229, 0.0176, 0.0139]
    return x, y


def make_chi(phi, theta, aberrations):
    """ Make aberration function chi"""
    maximum_aberration_order = 5
    chi = np.zeros(theta.shape)
    for n in range(maximum_aberration_order + 1):  # First Sum up to fifth order
        term_first_sum = np.power(theta, n + 1) / (n + 1)  # term in first sum

        second_sum = np.zeros(theta.shape)  # second Sum initialized with zeros
        for m in range((n + 1) % 2, n + 2, 2):
            if m > 0:
                aberrations.setdefault(f'C{n}{m}a', 0.)
                aberrations.setdefault(f'C{n}{m}b', 0.)
                # term in second sum
                second_sum = (second_sum + aberrations[f'C{n}{m}a'] * np.cos(m * phi)
                              + aberrations[f'C{n}{m}b'] * np.sin(m * phi))
            else:
                aberrations.setdefault(f'C{n}{m}', 0.)

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
        print(f"Acceleration voltage {ab['acceleration_voltage'] / 1000:}kV",
              f" => wavelength {wavelength * 1000.:.2f}pm")

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
    print(aperture_angle)
    mask = theta >= aperture_angle

    aperture = np.ones((size_x, size_y), dtype=float)
    aperture[mask] = 0.

    return chi, aperture


def print_aberrations(ab):
    """ Print aberrations in cartesian format """
    from IPython.display import HTML, display
    output = '<html><body>'
    output += "Aberrations [nm] for acceleration voltage: "
    output += f"{ab['acceleration_voltage'] / 1e3:.0f} kV"
    output += '<table>'
    output += f"<tr><td> C10 </td><td> {ab['C10']:.1f} </tr>"
    output += f"<tr><td> C12a (A1) </td><td> {ab['C12a']:20.1f}"
    output += f" <td> C12b (A1) </td><td> {ab['C12b']:20.1f} </tr>"
    output += f"<tr><td> C21a (B2) </td><td> {ab['C21a']:.1f}"
    output += f" <td> C21b (B2)</td><td> {ab['C21b']:.1f} "
    output += f"    <td> C23a (A2) </td><td> {ab['C23a']:.1f}"
    output += f" <td> C23b (A2) </td><td> {ab['C23b']:.1f} </tr>"
    output += f"<tr><td> C30 </td><td> {ab['C30']:.1f} </tr>"
    output += f"<tr><td> C32a (S3) </td><td> {ab['C32a']:20.1f}"
    output += f" <td> C32b (S3)</td><td> {ab['C32b']:20.1f} "
    output += f"<td> C34a (A3) </td><td> {ab['C34a']:20.1f}"
    output += f" <td> C34b (A3) </td><td> {ab['C34b']:20.1f} </tr>"
    output += f"<tr><td> C41a (B4) </td><td> {ab['C41a']:.3g}"
    output += f" <td> C41b (B4) </td><td> {ab['C41b']:.3g} "
    output += f"    <td> C43a (D4) </td><td> {ab['C43a']:.3g}"
    output += f" <td> C43b (D4) </td><td> {ab['C41b']:.3g} "
    output += f"    <td> C45a (A4) </td><td> {ab['C45a']:.3g}"
    output += f" <td> C45b (A4)</td><td> {ab['C45b']:.3g} </tr>"
    output += f"<tr><td> C50 </td><td> {ab['C50']:.3g} </tr>"
    output += f"<tr><td> C52a </td><td> {ab['C52a']:20.1f}"
    output += f" <td> C52b </td><td> {ab['C52b']:20.1f} "
    output += f"<td> C54a </td><td> {ab['C54a']:20.1f}"
    output += f" <td> C54b </td><td> {ab['C54b']:20.1f} "
    output += f"<td> C56a </td><td> {ab['C56a']:20.1f}"
    output += f" <td> C56b </td><td> {ab['C56b']:20.1f} </tr>"
    output += f"<tr><td> Cc </td><td> {ab['Cc']:.3g} </tr>"

    output += '</table></body></html>'

    display(HTML(output))



def print_aberrations_polar(ab):
    """ Print aberrations in polar format """
    from IPython.display import HTML, display

    ab['C12_r'], ab['C12_phi'] = cart2pol(ab['C12a'], ab['C12b'])
    ab['C21_r'], ab['C21_phi'] = cart2pol(ab['C21a'], ab['C21b'])
    ab['C23_r'], ab['C23_phi'] = cart2pol(ab['C23a'], ab['C23b'])
    ab['C32_r'], ab['C32_phi'] = cart2pol(ab['C32a'], ab['C32b'])
    ab['C34_r'], ab['C34_phi'] = cart2pol(ab['C34a'], ab['C34b'])
    ab['C41_r'], ab['C41_phi'] = cart2pol(ab['C41a'], ab['C41b'])
    ab['C43_r'], ab['C43_phi'] = cart2pol(ab['C43a'], ab['C43b'])
    ab['C45_r'], ab['C45_phi'] = cart2pol(ab['C45a'], ab['C45b'])
    ab['C52_r'], ab['C52_phi'] = cart2pol(ab['C52a'], ab['C52b'])
    ab['C54_r'], ab['C54_phi'] = cart2pol(ab['C54a'], ab['C54b'])
    ab['C56_r'], ab['C56_phi'] = cart2pol(ab['C56a'], ab['C56b'])

    output = '<html><body>'
    output += "Aberrations [nm] for acceleration voltage: "
    output += f"{ab['acceleration_voltage'] / 1e3:.0f} kV"
    output += '<table>'
    output += f"<tr><td> C10 </td><td> {ab['C10']:.1f} </tr>"
    output += f"<tr><td> C12(A1): r </td><td> {ab['C12_r']:20.1f}"
    output += f"<td> φ  </td><td> {ab['C12_phi']:20.1f} </tr>"
    output += f"<tr><td> C21a (B2): r</td><td> {ab['C21_r']:20.1f}"
    output += f" <td> φ </td><td> {ab['C21_phi']:20.1f} "
    output += f"    <td> C23a (A2) </td><td> {ab['C23_r']:20.1f} "
    output += f"<td> φ  </td><td> {ab['C23_phi']:20.1f} </tr>"
    output += f"<tr><td> C30 </td><td> {ab['C30']:.1f} </tr>"
    output += f"<tr><td> C32 (S3) </td><td> {ab['C32_r']:20.1f} "
    output += f"<td> φ </td><td> {ab['C32_phi']:20.1f} "
    output += f"<td> C34a (A3) </td><td> {ab['C34a']:20.1f}"
    output += f" <td> φ  </td><td> {ab['C34_phi']:20.1f} </tr>"
    output += f"<tr><td> C41 (B4) </td><td> {ab['C41_r']:.3g}"
    output += f" <td> φ  </td><td> {ab['C41_phi']:20.1f} "
    output += f"    <td> C43 (D4) </td><td> {ab['C43_r']:.3g}"
    output += f" <td> φ (D4) </td><td> {ab['C43_phi']:20.1f} "
    output += f"    <td> C45 (A4) </td><td> {ab['C45_r']:.3g}"
    output += f" <td> φ (A4)</td><td> {ab['C45_phi']:20.1f} </tr>"
    output += f"<tr><td> C50 </td><td> {ab['C50']:.3g} </tr>"
    output += f"<tr><td> C52 </td><td> {ab['C52a']:.3g}"
    output += f" <td> φ </td><td> {ab['C52_phi']:20.1f} "
    output += f"<td> C54 </td><td> {ab['C54_r']:.3g}"
    output += f" <td> C54 φ </td><td> {ab['C54_phi']:.1f} "
    output += f"<td> C56 </td><td> {ab['C56_r']:.3g}"
    output += f" <td> C56  </td><td> {ab['C56_phi']:.1f} </tr>"
    output += f"<tr><td> Cc </td><td> {ab['Cc']:.3g} </tr>"

    output += '</table></body></html>'

    display(HTML(output))

def ceos_to_nion(ab):
    """ Convert aberrations from CEOS polar format to Nion format"""
    aberrations = {'C10': 0, 'C12a': 0, 'C12b': 0,
                   'C21a': 0, 'C21b': 0, 'C23a': 0, 'C23b': 0,
                   'C30': 0., 'C32a': 0., 'C32b': -0., 'C34a': 0., 'C34b': 0.,
                   'C41a': 0., 'C41b': -0., 'C43a': 0., 'C43b': -0., 'C45a': -0., 'C45b': -0.,
                   'C50': 0., 'C52a': -0., 'C52b': 0., 'C54a': -0., 'C54b': -0.,
                   'C56a': -0., 'C56b': 0.,
                   'C70': 0.}
    aberrations['acceleration_voltage'] = 200000
    aberrations['C10'] = ab.get('C1', 0)
    aberrations['C30'] = ab.get('C3', 0)
    aberrations['C50'] = ab.get('C5', 0)
    for key in ab.keys():
        if key == 'A1-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['A1-a'], ab['A1-p'])
            aberrations['C12a'] = x
            aberrations['C12b'] = y
        elif key == 'B2-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['B2-a'], ab['B2-p'])
            aberrations['C21a'] = 3 * x
            aberrations['C21b'] = 3 * y
        elif key == 'A2-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['A2-a'], ab['A2-p'])
            aberrations['C23a'] = x
            aberrations['C23b'] = y
        elif key == 'S3-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['S3-a'], ab['S3-p'])
            aberrations['C32a'] = 4 * x
            aberrations['C32b'] = 4 * y
        elif key == 'A3-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['A3-a'], ab['A3-p'])
            aberrations['C34a'] = x
            aberrations['C34b'] = y
        elif key == 'B4-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['B4-a'], ab['B4-p'])
            aberrations['C41a'] = 4 * x
            aberrations['C41b'] = 4 * y
        elif key == 'D4-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['D4-a'], ab['D4-p'])
            aberrations['C43a'] = 4 * x
            aberrations['C43b'] = 4 * y
        elif key == 'A4-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['A4-a'], ab['A4-p'])
            aberrations['C45a'] = x
            aberrations['C45b'] = y
        elif key == 'A5-a':
            x, y = pyTEMlib.image_tools.pol2cart(ab['A5-a'], ab['A5-p'])
            aberrations['C56a'] = x
            aberrations['C56b'] = y
    return aberrations

def ceos_carth_to_nion(ab):
    """ Convert aberrations from CEOS cartesian format to Nion format"""
    aberrations = {'C10': 0, 'C12a': 0, 'C12b': 0,
                   'C21a': 0, 'C21b': 0, 'C23a': 0, 'C23b': 0,
                   'C30': 0., 'C32a': 0., 'C32b': -0., 'C34a': 0., 'C34b': 0.,
                   'C41a': 0., 'C41b': -0., 'C43a': 0., 'C43b': -0., 'C45a': -0., 'C45b': -0.,
                   'C50': 0., 'C52a': -0., 'C52b': 0., 'C54a': -0., 'C54b': -0.,
                   'C56a': -0., 'C56b': 0.,
                   'C70': 0.}
    aberrations['acceleration_voltage'] = 200000
    for key in ab.keys():
        if key == 'C1':
            aberrations['C10'] = ab['C1'][0]*1e9
        elif key == 'A1':
            aberrations['C12a'] = ab['A1'][0]*1e9
            aberrations['C12b'] = ab['A1'][1]*1e9
        elif key == 'B2':
            print('B2', ab['B2'])
            aberrations['C21a'] = 3 * ab['B2'][0]*1e9
            aberrations['C21b'] = 3 * ab['B2'][1]*1e9
        elif key == 'A2':
            aberrations['C23a'] = ab['A2'][0]*1e9
            aberrations['C23b'] = ab['A2'][1]*1e9
        elif key == 'C3':
            aberrations['C30'] = ab['C3'][0]*1e9
        elif key == 'S3':
            aberrations['C32a'] = 4 * ab['S3'][0]*1e9
            aberrations['C32b'] = 4 * ab['S3'][1]*1e9
        elif key == 'A3':
            aberrations['C34a'] = ab['A3'][0]*1e9
            aberrations['C34b'] = ab['A3'][1]*1e9
        elif key == 'B4':
            aberrations['C41a'] = 4 * ab['B4'][0]*1e9
            aberrations['C41b'] = 4 * ab['B4'][1]*1e9
        elif key == 'D4':
            aberrations['C43a'] = 4 * ab['D4'][0]*1e9
            aberrations['C43b'] = 4 * ab['D4'][1]*1e9
        elif key == 'A4':
            aberrations['C45a'] = ab['A4'][0]*1e9
            aberrations['C45b'] = ab['A4'][1]*1e9
        elif key == 'C5':
            aberrations['C50'] = ab['C5'][0]*1e9
        elif key == 'A5':
            aberrations['C56a'] = ab['A5'][0]*1e9
            aberrations['C56b'] = ab['A5'][1]*1e9
    return aberrations

def cart2pol(x, y):
    """ Convert cartesian to polar coordinates"""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def nion_to_ceos(ab):
    """ Convert aberrations from Nion format to CEOS format"""
    aberrations = {'C1': 0, 'A1-a': 0, 'A1-b': 0,
                   'B2-a': 0, 'B2-p': 0, 'A2-a': 0, 'A2-p': 0,
                   'C3': 0.,'S3-a': 0., 'S3-p': -0., 'A3-a': 0., 'A3-p': 0.,
                   'B4-a': 0., 'B4-p': -0., 'D4-a': 0., 'D4-p': -0., 'A4-s': -0., 'A4-p': -0.,
                   'C5': 0., 'A5-a': -0., 'A5-p': 0.}
    aberrations['acceleration_voltage'] = 200000
    for key in ab.keys():
        if key == 'C10':
            aberrations['C1'] = ab['C10']
        elif key == 'C12a':
            r, p = cart2pol(ab['C12a'], ab['C12b'])
            aberrations['A1-a'] = r
            aberrations['A1-p'] = p
        elif key == 'C21a':
            r, p = cart2pol(ab['C21a'], ab['C21b'])
            aberrations['B2-a'] = r/3
            aberrations['B2-p'] = p
        elif key == 'C23a':
            r, p = cart2pol(ab['C23a'], ab['C23b'])
            aberrations['A2-a'] = r
            aberrations['A2-p'] = p
        elif key == 'C30':
            aberrations['C3'] = ab['C30']
        elif key == 'C32a':
            r, p = cart2pol(ab['C32a'], ab['C32b'])
            aberrations['S3-a'] = r/4
            aberrations['S3-p'] = p
        elif key == 'C34a':
            r, p = cart2pol(ab['C34a'], ab['C34b'])
            aberrations['A3-a'] = r
            aberrations['A3-p'] = p
        elif key == 'C41a':
            r, p = cart2pol(ab['C41a'], ab['C41b'])
            aberrations['B4-a'] = r/4
            aberrations['B4-p'] = p
        elif key == 'C43a':
            r, p = cart2pol(ab['C43a'], ab['C43b'])
            aberrations['D4-a'] = r/4
            aberrations['D4-p'] = p
        elif key == 'C31a':
            r, p = cart2pol(ab['C41a'], ab['C41b'])
            aberrations['A4-a'] = r
            aberrations['A4-p'] = p
        elif key == 'C50':
            aberrations['C5'] = ab['C50']
        elif key == 'C56a':
            r, p = cart2pol(ab['C56a'], ab['C56b'])
            aberrations['A5-a'] = r
            aberrations['A5-p'] = p

    return aberrations

def get_ronchigram(size, ab, scale='mrad'):
    """ Get Ronchigram"""
    size_x = size_y = size
    chi, a_k = get_chi(ab, size_x, size_y)

    v_noise = np.random.rand(size_x, size_y)
    smoothing = 5
    phi_r = scipy.ndimage.gaussian_filter(v_noise, sigma=(smoothing, smoothing), order=0)

    sigma = 6  # 6 for carbon and thin

    q_r = np.exp(-1j * sigma * phi_r)
    # q_r = 1-phi_r * sigma

    t_k = a_k * (np.exp(-1j * chi))
    t_r = (np.fft.ifft2(np.fft.fftshift(t_k)))

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

    ab['chi'] = chi
    ab['ronchi_extent'] = extent
    ab['ronchi_label'] = ylabel
    return ronchigram


def make_probe (chi, aperture):
    """ Make electron probe from aberration function chi and aperture function"""
    chi2 = np.fft.ifftshift(chi)
    chi_t = np.fft.ifftshift (np.vectorize(complex)(np.cos(chi2), -np.sin(chi2)) )
    ## Aply aperture function
    chi_t = chi_t*aperture
    ## inverse fft of aberration function
    i2  = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift (chi_t)))
    ## intensity
    probe = np.real(i2 * np.conjugate(i2))

    return probe


def get_probe( ab, size_x, size_y, verbose= True):
    """ Get electron probe from aberrations """
    chi, a_k  = get_chi( ab, size_x, size_y, verbose)
    probe = make_probe (chi, a_k)
    return probe, a_k, chi


def get_probe_large(ab):
    """ Get a large probe for convolution purposes """
    ab['FOV'] = 20
    size_x = 512*2
    probe, _, _  = pyTEMlib.probe_tools.get_probe(ab, size_x, size_x, verbose= True)
    res = np.zeros((512, 512))
    res[256-32:256+32, 256-32:256+32 ] = skimage.transform.resize(probe, (64, 64))
    return res


def get_chi_2(ab, u, v):
    """ Aberration function chi up to fifth order"""
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
    """ Second derivative of chi respect to u"""
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
    """ Second derivative of chi respect to u and v"""
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
    """ Second derivative of chi respect to v"""
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
    """ Get source energy spread distribution from Nion"""
    x = np.linspace(-0.5, .9, 29)
    y = [0.0143, 0.0193, 0.0281, 0.0440, 0.0768, 0.1447, 0.2785, 0.4955, 0.7442, 0.9380, 1.0000,
         0.9483, 0.8596, 0.7620, 0.6539, 0.5515, 0.4478, 0.3500, 0.2683, 0.1979, 0.1410, 0.1021,
         0.0752, 0.0545, 0.0401, 0.0300, 0.0229, 0.0176, 0.0139]
    return x, y


def get_target_aberrations(tem_name, acceleration_voltage):
    """ Get target aberrations for specific TEMs"""
    ab = {}
    if tem_name == 'NionUS200':
        if int(acceleration_voltage) == 200000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV',
                  'f for Aberrations of {tem_name}****')
            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': -335., 'C21b': 283.,
                  'C23a': -34., 'C23b': 220.,
                  'C30': -8080.,
                  'C32a': 18800., 'C32b': -2260., 'C34a': 949., 'C34b': 949.,
                  'C41a': 54883., 'C41b': -464102., 'C43a': 77240.5, 'C43b': -540842.,
                  'C45a': -79844.4, 'C45b': -76980.8,
                  'C50': 9546970.,
                  'C52a': -2494290., 'C52b': 2999910.,
                  'C54a': -2020140., 'C54b': -2019630., 'C56a': -535079., 'C56b': 1851850.,
                  'source_size': 0.051,
                  'acceleration_voltage': acceleration_voltage,
                  'convergence_angle': 30,
                  'Cc': 1.3e6}  # // Cc in  nm
        if int(acceleration_voltage) == 100000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV',
                  f' for Aberrations of {tem_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 157., 'C21b': 169,
                  'C23a': -173., 'C23b': 48.7,
                  'C30': 201.,
                  'C32a': 1090., 'C32b': 6840., 'C34a': 1010., 'C34b': 79.9,
                  'C41a': -210696., 'C41b': -262313.,
                  'C43a': 348450., 'C43b': -9.7888e4, 'C45a': 6.80247e4, 'C45b': -3.14637e1,
                  'C50': -193896., 'C52a': -1178950, 'C52b': -7414340,
                  'C54a': -1753890, 'C54b': -1753890, 'C56a': -631786, 'C56b': -165705,
                  'source_size': 0.051,
                  'acceleration_voltage': acceleration_voltage,
                  'convergence_angle': 30,
                  'Cc': 1.3e6}

        if int(acceleration_voltage) == 60000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV',
                  f' for Aberrations of {tem_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 11.5, 'C21b': 113,
                  'C23a': -136., 'C23b': 18.2,
                  'C30': 134.,
                  'C32a': 1080., 'C32b': 773., 'C34a': 1190., 'C34b': -593.,
                  'C41a': -179174., 'C41b': -350378.,
                  'C43a': 528598, 'C43b': -257349., 'C45a': 63853.4, 'C45b': 1367.98,
                  'C50': 239021., 'C52a': 1569280., 'C52b': -6229310.,
                  'C54a': -3167620., 'C54b': -449198., 'C56a': -907315., 'C56b': -16281.9,
                  'source_size': 0.081,
                  'acceleration_voltage': acceleration_voltage,
                  'convergence_angle': 30,
                  'Cc': 1.3e6}  # // Cc in  nm

        ab['origin'] = 'target aberrations'
        ab['tem_name'] = tem_name
        ab['wavelength'] = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])

    if tem_name == 'NionUS100':
        if int(acceleration_voltage) == 100000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV',
                  f' for Aberrations of {tem_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 157., 'C21b': 169,
                  'C23a': -173., 'C23b': 48.7,
                  'C30': 201.,
                  'C32a': 1090., 'C32b': 6840., 'C34a': 1010., 'C34b': 79.9,
                  'C41a': -210696., 'C41b': -262313.,
                  'C43a': 348450., 'C43b': -9.7888e4, 'C45a': 6.80247e4, 'C45b': -3.14637e1,
                  'C50': -193896.,
                  'C52a': -1178950, 'C52b': -7414340, 'C54a': -1753890, 'C54b': -1753890,
                  'C56a': -631786, 'C56b': -165705,
                  'source_size': 0.051,
                  'acceleration_voltage': acceleration_voltage,
                  'convergence_angle': 30,
                  'Cc': 1.3e6}  # // Cc in  nm

        if int(acceleration_voltage) == 60000:
            print(f' **** Using Target Values at {acceleration_voltage / 1000}kV ',
                  f'for Aberrations of {tem_name}****')

            ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 11.5, 'C21b': 113,
                  'C23a': -136., 'C23b': 18.2, 'C30': 134.,
                  'C32a': 1080., 'C32b': 773., 'C34a': 1190., 'C34b': -593.,
                  'C41a': -179174., 'C41b': -350378.,
                  'C43a': 528598, 'C43b': -257349., 'C45a': 63853.4, 'C45b': 1367.98, 
                  'C50': 239021., 'C52a': 1569280., 'C52b': -6229310., 
                  'C54a': -3167620., 'C54b': -449198.,
                  'C56a': -907315., 'C56b': -16281.9,
                  'source_size': 0.081,
                  'acceleration_voltage': acceleration_voltage,
                  'convergence_angle': 30,
                  'Cc': 1.3e6}  # // Cc in  nm

        ab['origin'] = 'target aberrations'
        ab['tem_name'] = tem_name
        ab['wavelength'] = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])

    if tem_name == 'ZeissMC200':
        ab = {'C10': 0, 'C12a': 0, 'C12b': 0, 'C21a': 0, 'C21b': 0, 'C23a': 0, 'C23b': 0,
              'C32a': 0., 'C32b': -0., 'C34a': 0., 'C34b': 0., 'C41a': 0., 'C41b': -0., 'C43a': 0.,
              'C43b': -0., 'C45a': -0., 'C45b': -0., 'C50': 0., 'C52a': -0., 'C52b': 0.,
              'C54a': -0., 'C54b': -0., 'C56a': -0., 'C56b': 0.,
              'C30': 2.2 * 1e6,
              'Cc': 2.0 * 1e6,
              'source_size': 0.2,
              'acceleration_voltage': acceleration_voltage,
              'convergence_angle': 10}

        ab['origin'] = 'target aberrations'
        ab['tem_name'] = tem_name
        ab['wavelength'] = pyTEMlib.image_tools.get_wavelength(ab['acceleration_voltage'])

    if tem_name == 'Spectra300':
        ab = {'C10': 0, 'C12a': 0, 'C12b': 0.38448128113770325,
              'C21a': -68.45251255685642, 'C21b': 64.85359774641199, 
              'C23a': 11.667578600494137, 'C23b': -29.775627778458194, 
              'C30': 123,
              'C32a': 95.3047364258614, 'C32b': -189.72105710231244,
              'C34a': -47.45099594807912, 'C34b': -94.67424667529909,
              'C41a': -905.31842572806, 'C41b': 981.316128853203,
              'C43a': 4021.8433526960034, 'C43b': 131.72716642732158,
              'C45a': -4702.390968272048,  'C45b': -208.25028574642903,
              'C50': 552000., 'C52a': -0., 'C52b': 0.,
              'C54a': -0., 'C54b': -0., 'C56a': -36663.643489934424, 'C56b': 21356.079837905396,
              'acceleration_voltage': 200000,
              'FOV': 34.241659495148205,
              'Cc': 1* 1e6,
              'convergence_angle': 30,
              'wavelength': 0.0025079340450548005}
    return ab


def get_ronchigram_2(size, ab, scale='mrad', threshold=3):
    """ Get Ronchigram and map of infinite magnification"""
    aperture_angle = ab['convergence_angle'] / 1000.0  # in rad

    wavelength = pyTEMlib.utilities.get_wavelength(ab['acceleration_voltage'])*1e9  # in nm
    ab['wavelength'] = wavelength

    size_x = size_y = size

    # Reciprocal plane in 1/nm
    dk = ab['reciprocal_FOV'] / size
    k_x = np.array(dk * (-size_x / 2. + np.arange(size_x)))
    k_y = np.array(dk * (-size_y / 2. + np.arange(size_y)))
    t_x_v, t_y_v = np.meshgrid(k_x, k_y)

    chi = get_chi_2(ab, t_x_v, t_y_v)  # , verbose= True)
    # define reciprocal plane in angles
    _ = np.arctan2(t_x_v, t_y_v)
    theta = np.arctan2(np.sqrt(t_x_v ** 2 + t_y_v ** 2), 1 / wavelength)

    # Aperture function
    mask = theta >= aperture_angle

    aperture = np.ones((size_x, size_y), dtype=float)
    aperture[mask] = 0.

    v_noise = np.random.rand(size_x, size_y)
    smoothing = 10
    phi_r = scipy.ndimage.gaussian_filter(v_noise, sigma=(smoothing, smoothing), order=0)

    sigma = 6  # 6 for carbon and thin

    q_r = np.exp(-1j * sigma * phi_r)
    # q_r = 1-phi_r * sigma

    t_k = aperture * (np.exp(-1j * chi))
    t_r = np.fft.ifft2(np.fft.fftshift(t_k))

    psi_k = np.fft.fftshift(np.fft.fft2(q_r * t_r))

    ronchigram = np.absolute(psi_k * np.conjugate(psi_k))

    fov_reciprocal = ab['reciprocal_FOV']
    if scale == '1/nm':
        extent = [-fov_reciprocal, fov_reciprocal, -fov_reciprocal, fov_reciprocal]
        ylabel = 'reciprocal distance [1/nm]'
    else:
        fov_mrad = fov_reciprocal * ab['wavelength'] * 1000
        extent = [-fov_mrad, fov_mrad, -fov_mrad, fov_mrad]
        ylabel = 'reciprocal distance [mrad]'

    ab['chi'] = chi
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
    t0 = np.power(theta, 1) / 1 * (float(ab['C01a']) * np.cos(1 * phi)
                                   + float(ab['C01b']) * np.sin(1 * phi))

    if c1_include == 1:  # First and second terms
        t1 = np.power(theta, 2) / 2 * (ab['C10'] + ab['C12a'] * np.cos(2 * phi)
                                       + ab['C12b'] * np.sin(2 * phi))
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
    ab['fov'].setdefault(tags.get('fov', None))
    if 'fov' not in ab:
        print(' need field of view in tags ')

    ab['convAngle'].setdefault(30.)  # in mrad
    ap_angle = ab['convAngle'] / 1000.0  # in rad

    e0 = ab['EHT'] = float(ab['EHT'])  # acceleration voltage in eV

    ab['C01a'].setdefault(0.)
    ab['C01b'].setdefault(0.)

    ab['C50'].setdefault(0.)
    ab['C70'].setdefault(0.)

    ab['Cc'].setdefault(1.3e6)  # Cc in  nm
    wavelength = pyTEMlib.utilities.get_wavelength(e0) *1e9  # in nm
    if verbose:
        print(f'Acceleration voltage {int(e0 / 1000)}kV => wavelength {wavelength * 1000:.2f}pm')
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
            print(f"defocus due to Cc: {df:.2f} nm with weight {ab['zeroLoss'][i]:.2f}")
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
    chi_i = np.ones((size_y, size_x))
    chi_i[mask] = 0.
    i2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(chi_i)))
    ideal = np.real(i2 * np.conjugate(i2))

    probe_f = np.fft.fft2(probe, probe.shape) + 1e-12
    ideal_f = np.fft.fft2(ideal, probe.shape)
    fourier_space_division = ideal_f / probe_f
    probe_r = np.fft.ifft2(fourier_space_division, probe.shape)

    return probe / sum(ab['zeroLoss']), np.real(probe_r)
