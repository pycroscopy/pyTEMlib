"""Figures and Animations for TEM in jupyter notebooks
part of MSE 672 course at UTK

Author: Gerd Duscher
revision: 01/11/2021
03/17/2021 added Aberration Animation
 """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ipywidgets import widgets
from IPython.display import display

import pyTEMlib.kinematic_scattering as ks


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


def deficient_holz_line(exact_bragg=False, shift=False, laue_zone=1, color='black'):
    """
    Ewald sphere construction to explain Laue Circle and deficient HOLZ lines

    Parameters:
    exact_bragg: boolean
        whether to tilt into exact Bragg condition or along zone axis
    shift: boolean
        whether to shift exact Bragg-condition onto zone axis origin
    laue_zone: int
        first or second Laue zone only
    color: string
        color of wave vectors and Ewald sphere
    """

    k_0 = [0, 1 / ks.get_wavelength(600)]

    d = 5.  # lattice parameter in nm

    if laue_zone == 0:
        s_g = 1 / d + 0.06
    else:
        s_g = .1

    g = np.linspace(-5, 6, 12) * 1 / d
    g_d = np.array([5. / d + laue_zone * 1 / d / 2, laue_zone * 1 / d])
    g_sg = g_d.copy()
    g_sg[1] = g_d[1] + s_g  # point on Ewald sphere

    # reciprocal lattice
    plt.scatter(g[:-1], [0] * 11, color='red')
    plt.scatter(g - 1 / d / 2, [1 / d] * 12, color='blue')

    shift_x = shift_y = 0.
    d_theta = d_theta1 = d_theta2 = 0

    if exact_bragg:

        d_theta1 = np.arctan((1 / d * laue_zone + s_g) / g_d[0])
        d_theta2 = np.arctan((1 / d * laue_zone) / g_d[0])
        d_theta = -(d_theta1 - d_theta2)
        s_g = 0
        s = np.sin(d_theta)
        c = np.cos(d_theta)
        k_0 = [-s * k_0[1], c * k_0[1]]
        if shift:
            shift_x = -k_0[0]
            shift_y = np.linalg.norm(k_0) - k_0[1]
        d_theta = np.degrees(d_theta)

    k_0[0] += shift_x
    k_0[1] += shift_y

    # Ewald Sphere
    ewald_sphere = patches.Circle((k_0[0], k_0[1]), radius=np.linalg.norm(k_0), clip_on=False, zorder=10, linewidth=1,
                                  edgecolor=color, fill=False)
    plt.gca().add_artist(ewald_sphere)

    plt.gca().arrow(g[-1] + .1 / d / 4, 1 / d / 2, 0, 1 / d / 2, head_width=0.03, head_length=0.04, fc='k', ec='k',
                    length_includes_head=True)
    plt.gca().arrow(g[-1] + .1 / d / 4, 1 / d / 2, 0, -1 / d / 2, head_width=0.03, head_length=0.04, fc='k', ec='k',
                    length_includes_head=True)
    plt.gca().annotate("$|g_{HOLZ}|$", xytext=(g[-1] + .1 / d / 3, 1 / d / 3), xy=(g[-1] + 1 / d / 3, 1 / d / 3))

    # k_0
    plt.scatter(k_0[0], k_0[1])
    plt.gca().arrow(k_0[0], k_0[1], -k_0[0] + shift_x, -k_0[1] + shift_y, head_width=0.03, head_length=0.04, fc=color,
                    ec=color, length_includes_head=True)
    plt.gca().annotate("K$_0$", xytext=(k_0[0] / 2, k_0[1] / 3), xy=(k_0[0] / 2, k_0[1] / 2))

    # K_d Bragg of HOLZ reflection
    plt.gca().arrow(k_0[0], k_0[1], -k_0[0] + g_d[0] + shift_x, -k_0[1] + g_d[1] + s_g + shift_y, head_width=0.03,
                    head_length=0.04, fc=color,
                    ec=color, length_includes_head=True)
    plt.gca().annotate("K$_d$", xytext=(k_0[0] + (g_d[0] - k_0[0]) / 2, k_0[1] / 2), xy=(6.5 / d / 2, k_0[1] / 2))

    # s_g excitation Error of HOLZ reflection
    if s_g > 0:
        plt.gca().arrow(g_d[0], g_d[1], 0, s_g, head_width=0.03, head_length=0.04, fc='k',
                        ec='k', length_includes_head=True)
        plt.gca().annotate("s$_g$", xytext=(g_d[0] * 1.01, g_d[1] + s_g / 3), xy=(g_d[0] * 1.01, g_d[1] + s_g / 3))

    # Bragg angle
    g_sg = g_d
    g_sg[1] = g_d[1] + s_g
    plt.plot([0 + shift_x, g_sg[0] + shift_x], [0 + shift_y, g_d[1] + shift_y], color=color, linewidth=1, alpha=0.5,
             linestyle='--')
    plt.plot([k_0[0], g_sg[0] / 2 + shift_x], [k_0[1], g_sg[1] / 2 + shift_y], color=color, linewidth=1, alpha=0.5,
             linestyle='--')
    # d_theta = np.degrees(np.arctan(k_0[0]/k_0[1]))
    bragg_angle = patches.Arc((k_0[0], k_0[1]), width=k_0[1], height=k_0[1], theta1=-90 + d_theta,
                              theta2=-90 + d_theta + np.degrees(np.arcsin(np.linalg.norm(g_sg / 2) / k_0[1])), fc=color,
                              ec=color)

    plt.gca().annotate(r"$\theta $", xytext=(k_0[0] / 1.3, k_0[1] / 1.5), xy=(k_0[0] / 2 + g_d[0] / 4, k_0[1] / 2))
    plt.gca().add_patch(bragg_angle)

    # deviation/tilt angle
    if np.abs(d_theta) > 0:
        if shift:
            deviation_angle = patches.Arc((k_0[0], k_0[1]), width=k_0[1] * 1.5, height=k_0[1] * 1.5,
                                          theta1=-90 + d_theta,
                                          theta2=-90,
                                          fc=color, ec=color, linewidth=3)
            plt.gca().annotate(r"$d \theta $", xytext=(k_0[0] - .13, k_0[1] / 3.7),
                               xy=(k_0[0] + g_d[0] / 4, k_0[1] / 2))
            plt.gca().arrow(shift_x, -.2, 0, .2, head_width=0.05, head_length=0.06, fc=color, ec='black',
                            length_includes_head=True, linewidth=3)
            plt.gca().annotate("deficient line", xytext=(shift_x * 2, -.2), xy=(shift_x, 0))
        else:
            deviation_angle = patches.Arc((0, 0), width=k_0[1], height=k_0[1],
                                          theta1=np.degrees(d_theta2),
                                          theta2=np.degrees(d_theta1),
                                          fc=color, ec=color, linewidth=3)
            plt.gca().annotate(r"$d \theta $", xytext=(g_d[0] * .8, 1 / d / 3), xy=(g_d[0], 1 / d))

        plt.gca().add_patch(deviation_angle)
    plt.gca().set_aspect('equal')
    plt.gca().set_ylim(-.5, 2.2)
    plt.gca().set_xlim(-1.1, 1.6)


def deficient_kikuchi_line(s_g=0., color_b='black'):
    k_len = 1 / ks.get_wavelength(20)
    d = 2  # lattice parameter in nm

    g = np.linspace(-2, 2, 5) * 1 / d
    g_d = np.array([1 / d, 0])

    # reciprocal lattice
    plt.scatter(g, [0] * 5, color='blue')

    alpha = -np.arctan(s_g / g_d[0])
    theta = -np.arcsin(g_d[0] / 2 / k_len)

    k_0 = np.array([-np.sin(theta - alpha) * k_len, np.cos(theta - alpha) * k_len])
    k_d = np.array([-np.sin(-theta - alpha) * k_len, np.cos(-theta - alpha) * k_len])
    k_i = np.array([-np.sin(theta - alpha) * 1., np.cos(theta - alpha) * 1.])
    k_i_t = np.array([-np.sin(-alpha), np.cos(-alpha)])

    kk_e = np.array([-np.sin(-theta) * k_len, np.cos(-theta) * k_len])
    kk_d = np.array([-np.sin(theta) * k_len, np.cos(theta) * k_len])

    # Ewald Sphere
    ewald_sphere = patches.Circle((k_0[0], k_0[1]), radius=np.linalg.norm(k_0), clip_on=False, zorder=10, linewidth=1,
                                  edgecolor=color_b, fill=False)
    plt.gca().add_artist(ewald_sphere)

    # K_0
    plt.plot([k_0[0], k_0[0]], [k_0[1], k_0[1] + .4], color='gray', linestyle='-', alpha=0.3)

    plt.gca().arrow(k_0[0] + k_i[0], k_0[1] + k_i[1], -k_i[0], -k_i[1], head_width=0.01, head_length=0.015, fc=color_b,
                    ec=color_b, length_includes_head=True)
    plt.plot([k_0[0] + k_i_t[0], k_0[0] - k_i_t[0]], [k_0[1] + k_i_t[1], k_0[1] - k_i_t[1]], color='black',
             linestyle='--', alpha=0.5)
    plt.scatter(k_0[0], k_0[1], color='black')
    plt.gca().arrow(k_0[0], k_0[1], -k_0[0], -k_0[1], head_width=0.01, head_length=0.015, fc=color_b,
                    ec=color_b, length_includes_head=True)
    plt.gca().annotate("K$_0$", xytext=(-k_0[0] / 2, 0), xy=(k_0[0] / 2, 0))

    plt.gca().arrow(k_0[0], k_0[1], -k_d[0], -k_d[1], head_width=0.01, head_length=0.015, fc=color_b,
                    ec=color_b, length_includes_head=True)
    # K_e excess line
    plt.gca().arrow(k_0[0], k_0[1], -kk_e[0], -kk_e[1], head_width=0.01, head_length=0.015, fc='red',
                    ec='red', length_includes_head=True)
    plt.gca().annotate("excess", xytext=(k_0[0] - kk_e[0], -1), xy=(-kk_e[0] + k_0[0], 0))
    plt.plot([k_0[0] - kk_e[0], k_0[0] - kk_e[0]], [-.1, .1], color='red')

    # k_d deficient line
    plt.gca().arrow(k_0[0], k_0[1], -kk_d[0], -kk_d[1], head_width=0.01, head_length=0.015, fc='blue',
                    ec='blue', length_includes_head=True)
    plt.plot([k_0[0] - kk_d[0], k_0[0] - kk_d[0]], [-.1, .1], color='blue')
    plt.gca().annotate("deficient", xytext=(k_0[0] - kk_d[0], -1), xy=(k_0[0] - kk_d[0], 0))

    # s_g excitation Error of HOLZ reflection
    plt.gca().arrow(g_d[0], g_d[1], 0, s_g, head_width=0.01, head_length=0.015, fc='k',
                    ec='k', length_includes_head=True)
    plt.gca().annotate("s$_g$", xytext=(g_d[0] * 1.01, g_d[1] + s_g / 3), xy=(g_d[0] * 1.01, g_d[1] + s_g / 3))

    theta = np.degrees(theta)
    alpha = np.degrees(alpha)

    bragg_angle = patches.Arc((k_0[0], k_0[1]), width=.55, height=.55,
                              theta1=90 + theta - alpha, theta2=90 - alpha, fc='black', ec='black')
    if alpha > 0:
        deviation_angle = patches.Arc((k_0[0], k_0[1]), width=.6, height=.6,
                                      theta1=90 - alpha, theta2=90, fc='black', ec='red')
    else:
        deviation_angle = patches.Arc((k_0[0], k_0[1]), width=.6, height=.6,
                                      theta1=90, theta2=90 - alpha, fc='black', ec='red')

    plt.gca().annotate(r"$\theta$", xytext=(k_0[0] + k_i_t[0] / 20, k_0[1] + .2), xy=(k_0[0] + k_i_t[0], k_0[1] + .2))
    plt.gca().annotate(r"$\alpha$", xytext=(k_0[0] + k_i_t[0] / 10, k_0[1] + .3), xy=(k_0[0] + k_i_t[0], k_0[1] + .3),
                       color='red')
    plt.gca().add_patch(bragg_angle)
    plt.gca().add_patch(deviation_angle)

    plt.gca().set_aspect('equal')
    plt.gca().set_xlabel('angle (1/$\AA$)')
    plt.gca().set_ylim(-.1, k_0[1] * 2.2)
    plt.gca().set_xlim(-.2, 1.03)


class InteractiveAberration(object):
    """
    ### Interactive explanation of aberrations

    """

    def __init__(self, horizontal=True):

        box_layout = widgets.Layout(display='flex',
                                    flex_flow='row',
                                    align_items='stretch',
                                    width='100%')

        self.words = ['ideal rays', 'aberrated rays', 'aberrated wavefront', 'aberration function']

        self.buttons = [widgets.ToggleButton(value=False, description=word, disabled=False) for word in self.words]
        box = widgets.Box(children=self.buttons, layout=box_layout)
        display(box)

        # Button(description='edge_quantification')
        for button in self.buttons:
            button.observe(self.on_button_clicked, 'value')  # on_click(self.on_button_clicked)

        self.figure = plt.figure()
        self.ax = plt.gca()
        self.horizontal = horizontal
        self.ax.set_aspect('equal')
        self.analysis = []
        self.update()
        # self.cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)

    def on_button_clicked(self, b):
        # print(b['owner'].description)
        selection = b['owner'].description
        if selection in self.analysis:
            self.analysis.remove(selection)
        else:
            self.analysis.append(selection)
        self.update()

    def update(self):
        ax = self.ax
        ax.clear()
        selection = self.analysis
        ax.plot([0, 15], [0, 0], color='black')
        ax.plot([9, 9], [-.3, .3], color='black')
        lens = patches.Ellipse((2, 0),
                               width=.4,
                               height=7,
                               facecolor='gray')
        ax.add_patch(lens)
        ax.set_ylim(-6.5, 6.5)
        ax.set_aspect('equal')

        if self.words[0] in selection:
            color = 'gray'
            ax.plot([0, 2], [1, 1], color=color)
            ax.plot([0, 2], [-1, -1], color=color)
            ax.plot([2, 9], [1, 0], color=color)
            ax.plot([2, 9], [-1, 0], color=color)

            gauss = patches.Ellipse((9, 0),
                                    width=12,
                                    height=12,
                                    fill=False)
            ax.add_patch(gauss)

        if self.words[1] in selection:
            color = 'blue'
            ax.plot([0, 2], [2, 2], color=color)
            ax.plot([0, 2], [-2, -2], color=color)
            ax.plot([2, 7], [2, 0], color=color)
            ax.plot([2, 7], [-2, 0], color=color)
            gauss2 = patches.Ellipse((7, 0),
                                     width=8,
                                     height=8,
                                     fill=False,
                                     color=color, linestyle='--')
            plt.gca().add_patch(gauss2)

        if self.words[2] in selection:
            color = 'red'
            ax.plot([0, 2], [2, 2], color=color)
            ax.plot([0, 2], [-2, -2], color=color)
            ax.plot([2, 7], [2, 0], color=color)
            ax.plot([2, 7], [-2, 0], color=color)
            ax.plot([0, 2], [1, 1], color=color)
            ax.plot([0, 2], [-1, -1], color=color)
            ax.plot([2, 9], [1, 0], color=color)
            ax.plot([2, 9], [-1, 0], color=color)
            gauss3 = patches.Ellipse((9, 0),
                                     width=12,
                                     height=9.7,
                                     fill=False,
                                     color=color)
            plt.gca().add_patch(gauss3)

        if self.words[3] in selection:
            color = 'green'
            x = np.arange(100) / 100 - 6
            x2 = np.arange(100) / 100 * 1.5 - 6
            b = 4.8
            a = 6
            y = np.sqrt(a ** 2 - x ** 2)
            y2 = b / a * np.sqrt(a ** 2 - x2 ** 2)

            x = np.append(x[::-1], x[1:])
            y = np.append(y[::-1], -y[1:])
            x2 = np.append(x2[::-1], x2[1:])
            y2 = np.append(y2[::-1], -y2[1:])

            dif = y2 - y

            x = np.append(x[::-1], x2)
            y = np.append(y[::-1], y2)
            aberration = patches.Polygon(np.array([x + 9, y]).T,
                                         fill=True,
                                         color=color, alpha=.5)

            aberration2 = patches.Polygon(np.array(
                [np.append(np.abs(dif), [0, 0]) * 2 + 2.5, np.append(np.linspace(-3.3, 3.3, len(dif)), [3.3, -3.3])]).T,
                                          fill=True,
                                          color=color, alpha=.9)

            plt.gca().add_patch(aberration)
            plt.gca().add_patch(aberration2)


class InteractiveRonchigramMagnification(object):
    """    
    ### Interactive explanation of magnification 

    """

    def __init__(self, horizontal=True):

        box_layout = widgets.Layout(display='flex',
                                    flex_flow='row',
                                    align_items='stretch',
                                    width='100%')

        self.words = ['ideal rays', 'radial circle rays', 'axial circle rays', 'over-focused rays']

        self.buttons = [widgets.ToggleButton(value=False, description=word, disabled=False) for word in self.words]
        box = widgets.Box(children=self.buttons, layout=box_layout)
        display(box)

        # Button(description='edge_quantification')
        for button in self.buttons:
            button.observe(self.on_button_clicked, 'value')  # on_click(self.on_button_clicked)

        self.figure = plt.figure()
        self.ax = plt.gca()
        self.horizontal = horizontal
        self.ax.set_aspect('equal')
        self.analysis = []
        self.update()
        # self.cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)

    def on_button_clicked(self, b):
        # print(b['owner'].description)
        selection = b['owner'].description
        if selection in self.analysis:
            self.analysis.remove(selection)
        else:
            self.analysis.append(selection)
        self.update()

    def update(self):
        ax = self.ax
        ax.clear()
        selection = self.analysis
        ax.plot([0, 24], [0, 0], color='black')
        ax.plot([14, 14], [-.3, .3], color='black')
        ax.text(14, 1, s='f')
        lens = patches.Ellipse((4, 0),
                               width=.8,
                               height=14,
                               facecolor='gray')
        ax.add_patch(lens)
        ax.text(4, 8, s='lens')
        sample = patches.Rectangle((10, -2),
                                   width=.2,
                                   height=4,
                                   facecolor='gray')

        ax.add_patch(sample)
        ax.text(9, 3, s='sample')
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')

        if self.words[0] in selection:
            color = 'gray'
            ax.plot([0, 4], [1, 1], color=color)
            ax.plot([0, 4], [-1, -1], color=color)
            ax.plot([4, 24], [1, -1], color=color)
            ax.plot([4, 24], [-1, 1], color=color)

            circle1 = patches.Ellipse((24, 0), width=.2, height=2, fill=False, color=color)
            ax.add_patch(circle1)

        if self.words[1] in selection:
            color = 'red'
            ax.plot([0, 4], [3, 3], color=color)
            ax.plot([0, 4], [-3, -3], color=color)
            ax.plot([4, 24], [3, -4], color=color)
            ax.plot([4, 24], [-3, 4], color=color)
            ax.plot([0, 4], [2.5, 2.5], color=color)
            ax.plot([0, 4], [-2.50, -2.5], color=color)
            ax.plot([4, 24], [2.5, -2.8], color=color)
            ax.plot([4, 24], [-2.5, 2.8], color=color)

            circle2 = patches.Ellipse((24, 0), width=.9, height=8, fill=False, color=color)
            ax.add_patch(circle2)
            circle3 = patches.Ellipse((24, 0), width=.6, height=5.6, fill=False, color=color)
            ax.add_patch(circle3)
            circle3 = patches.Ellipse((24, 0), width=.7, height=7.3, fill=False, color=color, linewidth=5, alpha=.5)
            ax.add_patch(circle3)

        if self.words[2] in selection:
            color = 'orange'
            ax.plot([0, 4], [4, 4], color=color)
            ax.plot([0, 4], [-4, -4], color=color)
            ax.plot([4, 24], [4, -9.25], color=color)
            ax.plot([4, 24], [-4, 9.25], color=color)

            circle4 = patches.Ellipse((24, 0), width=2, height=18.5, fill=False, color=color)
            plt.gca().add_patch(circle4)

        if self.words[3] in selection:
            color = 'green'
            ax.plot([0, 4], [5, 5], color=color, linestyle='--')
            ax.plot([0, 4], [-5, -5], color=color, linestyle='--')
            ax.plot([4, 24], [5, -13], color=color, linestyle='--')
            ax.plot([4, 24], [-5, 13], color=color, linestyle='--')

            circle6 = patches.Ellipse((24, 0), width=4, height=26, fill=False, color=color, linestyle='--')
            plt.gca().add_patch(circle6)
