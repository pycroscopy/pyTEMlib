"""Figures and Animations for TEM in jupyter notebooks
part of MSE 672 course at UTK

Author: Gerd Duscher
revision: 01/11/2021
 """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def geometric_ray_diagram(focal_length=1., magnification=False):
    """ Sketch of geometric ray diagram od one lens

    Parameters
    ----------
    focal_length: float
        focal length of lens
    magnification: boolean
        draw magnification on the side

    Returns
    -------
    matplotlib figure
    """

    f = focal_length

    u = 1.5
    v = 1 / (1 / f - 1 / u)
    m = v / u
    if magnification:
        line_strong = .5
    else:
        line_strong = 2

    x = 0.4

    fig, ax = plt.subplots()

    # add an ellipse
    ellipse = patches.Ellipse((0.0, 0.0), 3.4, 0.3, alpha=0.3, color='blue')
    ax.add_patch(ellipse)
    ax.plot([1.5, -1.5], [0, 0], '--', color='black')
    ax.plot([0, 0], [u, -v], '--', color='black')
    single_prop = dict(arrowstyle="->", shrinkA=0, shrinkB=0)
    double_prop = dict(arrowstyle="<->", shrinkA=0, shrinkB=0)

    if magnification:
        ax.annotate("", xy=(-x, u), xytext=(x, u), arrowprops=single_prop)
        ax.annotate("", xy=(x * m, -v), xytext=(-x * m, -v), arrowprops=single_prop)

    else:
        ax.annotate("", xy=(-x, u), xytext=(0, u), arrowprops=single_prop)
        ax.annotate("", xy=(x * m, -v), xytext=(0, -v), arrowprops=single_prop)

    ax.text(x + 0.1, u, 'object plane', va='center')
    ax.plot([1, -1], [-f, -f], '--', color='black')
    ax.text(1.1, -f, 'back focal\n plane', va='center')
    ax.text(x * m + 0.1, -v, 'image plane', va='center')

    ax.annotate("", xy=(-.9, 0), xytext=(-.9, -f), arrowprops=double_prop)
    ax.text(-1, -f / 2, 'f')
    if magnification:
        ax.annotate("", xy=(-1.8, 0), xytext=(-1.8, -v), arrowprops=double_prop)
        ax.text(-1.7, -v / 2, 'v')
        ax.annotate("", xy=(-1.8, 0), xytext=(-1.8, u), arrowprops=double_prop)
        ax.text(-1.7, u / 2, 'u')

    ax.plot([-x, x * m], [u, -v], color='black', linewidth=line_strong)
    ax.plot([-x, -x], [u, 0], color='black', linewidth=line_strong)
    ax.plot([-x, x * m], [0, -v], color='black', linewidth=line_strong)

    ax.plot([-x, -2 * x], [u, 0], color='black', linewidth=0.5)
    ax.plot([-2 * x, x * m], [0, -v], color='black', linewidth=0.5)
    if magnification:
        ax.plot([x, -x * m], [u, -v], color='black', linewidth=0.5)
        ax.plot([x, x], [u, 0], color='black', linewidth=0.5)
        ax.plot([x, -x * m], [0, -v], color='black', linewidth=0.5)

        ax.plot([x, 2 * x], [u, 0], color='black', linewidth=0.5)
        ax.plot([2 * x, -x * m], [0, -v], color='black', linewidth=0.5)
    else:
        ax.plot([-x, x * m], [u, 0], color='black', linewidth=0.5)
        ax.plot([x * m, x * m], [0, -v], color='black', linewidth=0.5)

    ax.set_xlim(-2, 3)
    ax.set_ylim(-3.5, 2)
    ax.set_aspect('equal')


# ----------------------------------------------------------------
# Modified from Michael Fairchild :simply draws a thin-lens at the provided location parameters:
#     - z:    location along the optical axis (in mm)
#     - f:    focal length (in mm, can be negative if div. lens)
#     - diam: lens diameter in mm
#     - lens_labels:  label to identify the lens on the drawing
# ----------------------------------------------------------------
def add_lens(z, f, diam, lens_labels):
    """add lens to propagate beam plot"""
    ww, tw, rad = diam / 10.0, diam / 3.0, diam / 2.0
    plt.plot([z, z], [-rad, rad], 'k', linewidth=2)
    plt.plot([z, z + tw], [-rad, -rad + np.sign(f) * ww], 'k', linewidth=2)
    plt.plot([z, z - tw], [-rad, -rad + np.sign(f) * ww], 'k', linewidth=2)
    plt.plot([z, z + tw], [rad, rad - np.sign(f) * ww], 'k', linewidth=2)
    plt.plot([z, z - tw], [rad, rad - np.sign(f) * ww], 'k', linewidth=2)
    plt.plot([z + f, z + f], [-ww, ww], 'k', linewidth=2)
    plt.plot([z - f, z - f], [-ww, ww], 'k', linewidth=2)
    plt.text(z, rad + 5.0, lens_labels, fontsize=12)
    plt.text(z, rad + 2.0, 'f=' + str(int(f)), fontsize=10)


def add_aperture(z, diam, radius, lens_labels):
    """add aperture to propagate beam plot"""

    ww, tw, rad = diam / 10.0, diam / 3.0, diam / 2.0
    radius = radius / 2
    plt.plot([z, z], [-rad, -radius], 'k', linewidth=2)
    plt.plot([z, z], [rad, radius], 'k', linewidth=2)
    plt.text(z, -rad - 2.0, lens_labels, fontsize=12)


def propagate_beam(source_position, numerical_aperture, number_of_rays, lens_positions, focal_lengths,
                   lens_labels='', color='b'):
    """geometrical propagation of light rays from given source

    Parameters
    ----------
    source_position:  list
        location of the source (z0, x0) along and off axis (in mm)
    numerical_aperture:  float
        numerical aperture of the beam (in degrees)
    number_of_rays:  int
        number of rays to trace
    lens_positions:  numpy array
        array with the location of the lenses
    focal_lengths:  numpy array
        array with the focal length of lenses
    lens_labels: list of string
        label for the nature of lenses
    color: str
        color of the rays on plot
    """

    plt.figure()
    z_max = 1600.

    # aperture (maximum angle) in radians
    apa = numerical_aperture * np.pi / 180.0

    for i in range(np.size(lens_positions)):
        add_lens(lens_positions[i], focal_lengths[i], 25, lens_labels[i])

    add_aperture(840, 25, 7, 'CA')

    # position of source is z0,x0
    z0 = source_position[0]
    if np.size(source_position) == 2:
        x0 = source_position[1]
    else:
        x0 = 0.0

    # list of lens positions
    zl1, ff1 = lens_positions[(z0 < lens_positions)], focal_lengths[(z0 < lens_positions)]
    nl = np.size(zl1)  # number of lenses

    zz, xx, tani = np.zeros(nl + 2), np.zeros(nl + 2), np.zeros(nl + 2)
    tan0 = np.tan(apa / 2.0) - np.tan(apa) * np.arange(number_of_rays) / (number_of_rays - 1)

    for i in range(number_of_rays):
        tani[0] = tan0[i]  # initial incidence angle
        zz[0], xx[0] = z0, x0
        for j in range(nl):
            zz[j + 1] = zl1[j]
            xx[j + 1] = xx[j] + (zz[j + 1] - zz[j]) * tani[j]
            tani[j + 1] = tani[j] - xx[j + 1] / ff1[j]

        zz[nl + 1] = z_max
        xx[nl + 1] = xx[nl] + (zz[nl + 1] - zz[nl]) * tani[nl]
        plt.plot(zz, xx, color)
        plt.axis([-20, z_max, -20, 20])
