import matplotlib.pyplot as plt

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Circle  # , Ellipse, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates, geometric_transform

import ase
import numpy as np
import sidpy

# ##################################
# Plot Reciprocal Unit Cell in 2D #
# ##################################


def plot_reciprocal_unit_cell_2D(atoms):
    """Plot # unit cell in reciprocal space in 2D"""

    reciprocal_unit_cell = atoms.get_reciprocal_cell()

    # ignore y direction

    x = [reciprocal_unit_cell[0, 0], reciprocal_unit_cell[0, 0], reciprocal_unit_cell[1, 0], reciprocal_unit_cell[1, 0]]
    z = [reciprocal_unit_cell[0, 2], reciprocal_unit_cell[2, 2], reciprocal_unit_cell[2, 2], reciprocal_unit_cell[0, 2]]

    # Plot 2D
    fig = plt.figure()
    ax = plt.gca()  # current axis

    ax.scatter(x, z, c='red', s=80)
    ax.add_patch(
        patches.Rectangle(
            (0, 0),  # (x,y)
            reciprocal_unit_cell[0, 0],  # width
            reciprocal_unit_cell[2, 2],  # height
            fill=False  # remove background
        )
    )
    ax.add_patch(
        patches.FancyArrow(0, 0, reciprocal_unit_cell[0, 0], 0, width=0.02,
                           color='black',
                           head_width=0.08,  # Default: 3 * width
                           head_length=0.1,  # Default: 1.5 * head_width
                           length_includes_head=True  # Default: False
                           )
    )
    ax.add_patch(
        patches.FancyArrow(0, 0, 0, reciprocal_unit_cell[2, 2], width=0.02,
                           color='black',
                           head_width=0.08,  # Default: 3 * width
                           head_length=0.1,  # Default: 1.5 * head_width
                           length_includes_head=True  # Default: False
                           )
    )

    plt.xlabel('x 1/nm')
    plt.ylabel('z 1/nm')
    ax.axis('equal')
    # plt.title('Unit Cell in Reciprocal Space of {0}'.format(tags['crystal']) )
    # texfig.savefig("recip_unit_cell")
    # fig.savefig('recip_unit_cell.jpg', dpi=90, bbox_inches='tight')
    plt.show()
    return fig


# ####################
# Plot SAED Pattern #
# ####################
def plotSAED_parameter(gray=False):

    tags = {'convergence_angle_A-1': 0,
            'background': 'white',  # 'white'  'grey'
            'color_map': 'plasma',  # ,'cubehelix' #'Greys'#'plasma'
            'color_reflections': 'intensity'}

    if gray:
        tags['color_map'] = 'gray'
        tags['background'] = '#303030'  # 'darkgray'
        tags['color_reflections'] = 'intensity'
    tags['plot_HOLZ'] = 0
    tags['plot_HOLZ_excess'] = 0
    tags['plot_Kikuchi'] = 1
    tags['plot_reflections'] = 1

    tags['color_Kikuchi'] = 'green'

    tags['linewidth_HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    tags['linewidth_Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2

    tags['label_HOLZ'] = 0
    tags['label_Kikuchi'] = 0
    tags['label_reflections'] = 0

    tags['label_color'] = 'white'
    tags['label_size'] = 10

    tags['color_Laue_Zones'] = ['red', 'blue', 'green', 'blue', 'green']  # , 'green', 'red'] #for OLZ give a sequence
    tags['color_zero'] = 'red'  # 'None' #'white'
    tags['color_ring_zero'] = 'None'  # 'Red' #'white' #, 'None'
    tags['width_ring_zero'] = .2

    # plotDiffPattern(tags,True)
    tags['plot_rotation'] = 0.  # degree
    tags['plot_shift_x'] = -0.0
    tags['plot_shift_y'] = 0.0
    
    return tags


########################
# Plot Kikuchi Pattern #
########################
def plotKikuchi(grey=False):
    tags = {'background': 'black',  # 'white'  'grey'
            'color_map': 'plasma',  # ,'cubehelix'#'Greys'#'plasma'
            'color_reflections': 'intensity',
            'plot_HOLZ': 0,
            'plot_HOLZ_excess': 0,
            'plot_Kikuchi': 1,
            'plot_reflections': 1,
            'label_HOLZ': 0,
            'label_Kikuchi': 0,
            'label_reflections': 0,
            'label_color': 'white',
            'label_size': 10,
            'color_Kikuchi': 'green',
            'linewidth_HOLZ': -1,  # -1: linewidth according to intensity (structure factor F^2
            'linewidth_Kikuchi': -1,  # -1: linewidth according to intensity (structure factor F^2
            'color_Laue_Zones': ['red', 'blue', 'green', 'blue', 'green'],  # , 'green', 'red'] #for OLZ give a sequence
            'color_zero': 'white',  # 'None' #'white'
            'color_ring_zero': 'None',  # 'Red' #'white' #, 'None'
            'width_ring_zero': 2}

    if grey:
        tags['color_map'] = 'gray'
        tags['background'] = '#303030'  # 'darkgray'
        tags['color_reflections'] = 'intensity'
    
    return tags

    # plotDiffPattern(tags,True)


########################
# Plot HOLZ Pattern #
########################

def plotHOLZ_parameter(grey=False):
    tags = {'background': 'gray', 'color_map': 'plasma', 'color_reflections': 'intensity', 'plot_HOLZ': 1,
            'plot_HOLZ_excess': 1, 'plot_Kikuchi': 1, 'plot_reflections': 1, 'label_HOLZ': 0, 'label_Kikuchi': 0,
            'label_reflections': 0, 'label_color': 'white', 'label_size': 12, 'color_Kikuchi': 'green',
            'linewidth_HOLZ': 1, 'linewidth_Kikuchi': -1,
            'color_Laue_Zones': ['red', 'blue', 'lightblue', 'green', 'red'], 'color_zero': 'None',
            'color_ring_zero': 'Red', 'width_ring_zero': 2, 'plot_rotation': 0., 'plot_shift_x': -0.0,
            'plot_shift_y': 0.0}  # 'white'  'grey'

    # plotDiffPattern(holz,True)
    return tags


########################
# Plot CBED Pattern #
########################

def plotCBED_parameter():
    tags = {'background': 'black', 'color_map': 'plasma', 'color_reflections': 'intensity', 'plot_HOLZ': 1,
            'plot_HOLZ_excess': 1, 'plot_Kikuchi': 1, 'plot_reflections': 1, 'label_HOLZ': 0, 'label_Kikuchi': 0,
            'label_reflections': 0, 'label_color': 'white', 'label_size': 10, 'color_Kikuchi': 'green',
            'linewidth_HOLZ': -1, 'linewidth_Kikuchi': -1, 'color_Laue_Zones': ['red', 'blue', 'green'],
            'color_zero': 'white', 'color_ring_zero': 'Red', 'width_ring_zero': 2}  # 'white'  'grey'

    # plotDiffPattern(tags,True)
    return tags

########################
# Plot HOLZ Pattern #
########################


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles.
    Similar to plt.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_, picker=True)
               for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def cartesian2polar(x, y, grid, r, t, order=3):
    R, T = np.meshgrid(r, t)

    new_x = R * np.cos(T)
    new_y = R * np.sin(T)

    ix = interp1d(x, np.arange(len(x)))
    iy = interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    return map_coordinates(grid, np.array([new_ix, new_iy]),
                           order=order).reshape(new_x.shape)


def warp(diff, center):
    """
    Define original polar grid

    Parameter:
    ----------
    diff: sidpy object or numpy ndarray of
        diffraction pattern
    center: list or numpy array of length 2
        coordinates of center in pixel

    Return:
    ------
    numpy array of diffraction pattern in polar coordinates

    """
    nx = diff.shape[0]
    ny = diff.shape[1]

    x = np.linspace(1, nx, nx, endpoint=True) - center[0]
    y = np.linspace(1, ny, ny, endpoint=True) - center[1]
    z = diff

    # Define new polar grid
    nr = int(min([center[0], center[1], diff.shape[0] - center[0], diff.shape[1] - center[1]]) - 1)
    print(nr)
    nt = 360 * 3

    r = np.linspace(1, nr, nr)
    t = np.linspace(0., np.pi, nt, endpoint=False)
    return cartesian2polar(x, y, z, r, t, order=3).T


def topolar(img, order=1):
    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order.
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal
    # from a corner to the mid-point of img.
    max_radius = 0.5 * np.linalg.norm(img.shape)

    def transform(coords):
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2 * np.pi * coords[1] / (img.shape[1] - 1.)

        # Then map it to the interval [0, max_radius].
        # radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5 * img.shape[0] - radius * np.sin(theta)
        j = radius * np.cos(theta) + 0.5 * img.shape[1]
        return i, j

    polar = geometric_transform(img, transform, order=order)

    rads = max_radius * np.linspace(0, 1, img.shape[0])
    angs = np.linspace(0, 2 * np.pi, img.shape[1])

    return polar, (rads, angs)


def plot_ring_pattern(atoms, diffraction_pattern=None, grey=False):
    """
    Plot of ring diffraction pattern with matplotlib
    
    Parameters
    ----------
    atoms: dictionary or sidpy.Dataset
        information stored as dictionary either directly or in metadata attribute of sidpy.Dataset
    grey: bool
        plotting in greyscale if True
        
    Returns
    -------
    fig: matplotlib figure
        reference to matplotlib figure
    """
    
    if isinstance(atoms, dict):
        tags = atoms
    elif isinstance(atoms, ase.Atoms):
        if 'diffraction' in atoms.info:
            tags = atoms.info['diffraction']
            plot_diffraction_pattern = True
        else:
            raise TypeError('Diffraction information must be in metadata')
    else:
        raise TypeError('Diffraction info must be in sidpy Dataset or dictionary form')
    if diffraction_pattern is not None:
        if not(diffraction_pattern, sidpy.Dataset):
            print('diffraction_pattern must be a sidpy.Dataset \n -> Ignoring this variable')
            diffraction_pattern = None
    d = tags['Ring_Pattern']['allowed']['g norm']
    label = tags['Ring_Pattern']['allowed']['label']
    if 'label_color' not in tags:
        tags['label_color'] = 'navy'
    if 'profile color' not in tags:
        tags['profile color'] = 'navy'
    if 'ring color' not in tags:
        tags['ring color'] = 'red'
    if 'label_size' not in tags:
        tags['label_size'] = 10
    if 'profile height' not in tags:
        tags['profile height'] = 5
    if 'plot_scalebar' not in tags:
        tags['plot_scalebar'] = False
        
    fg, ax = plt.subplots(1, 1)

    # ###
    # plot arcs of the rings
    # ###
    for i in range(len(d)):
        pac = patches.Arc((0, 0), d[i] * 2, d[i] * 2, angle=0, theta1=45, theta2=360, color=tags['ring color'])
        ax.add_patch(pac)

    ####
    # show image in background
    ####
    if plot_diffraction_pattern is not None:
        plt.imshow(diffraction_pattern, extent=diffraction_pattern.get_extent(), cmap='gray')

    ax.set_aspect("equal")

    # fg.canvas.draw()

    if tags['plot_scalebar']:
        def f(axis):
            l = axis.get_majorticklocs()
            return len(l) > 1 and (l[1] - l[0])

        sizex = f(ax.xaxis)
        labelx = str(sizex) + ' 1/nm'
        scalebar = AnchoredSizeBar(ax.transData, sizex, labelx, loc=3,
                                   pad=0.5, color='white', frameon=False)
        # size_vertical=.2, fill_bar = True) # will be implemented in matplotlib 2.1

        ax.add_artist(scalebar)
        ax.axis('off')

    # ####
    # plot profile
    # ####

    y = tags['Ring_Pattern']['profile_y']
    y = y / y.max() * tags['profile height']
    x = tags['Ring_Pattern']['profile_x']
    ax.plot(x, y, c=tags['profile color'])

    ax.plot([0, x[-1]], [0, 0], c=tags['profile color'])

    if 'experimental profile_y' in tags:
        yy = tags['experimental profile_y']
        yy = yy / yy.max() * tags['profile height']
        xx = tags['experimental profile_x']
        ax.plot(xx, yy, c=tags['experimental profile color'])

    if 'plot_image_FOV' in tags:
        max_d = tags['plot_image_FOV'] / 2 + tags['plot_shift_x']
    else:
        max_d = d.max()
    for i in range(len(d)):
        if d[i] < max_d:
            plt.text(d[i] - .2, -.5, label[i], fontsize=tags['label_size'], color=tags['label_color'], rotation=90)

    if 'plot_FOV' in tags:
        l = -tags['plot_FOV'] / 2
        r = tags['plot_FOV'] / 2
        t = -tags['plot_FOV'] / 2
        b = tags['plot_FOV'] / 2
        plt.xlim(l, r)
        plt.ylim(t, b)

    fg.show()
    return fg


def plot_diffraction_pattern(atoms, diffraction_pattern=None, grey=False):
    """
    Plot of spot diffraction pattern with matplotlib
    Plot of spot diffraction pattern with matplotlib

    Parameters
    ----------
    atoms: dictionary or ase.Atoms object
        information stored as dictionary either directly or in info attribute of ase.Atoms object
    diffraction_pattern: None or sidpy.Dataset
        diffraction pattern in background
    grey: bool
        plotting in greyscale if True
        
    Returns
    -------
    fig: matplotlib figure
        reference to matplotlib figure
    """
    
    if isinstance(atoms, dict):
        tags_out = atoms
        
    elif isinstance(atoms, ase.Atoms):
        if 'diffraction' in atoms.info:
            tags_out = atoms.info['diffraction']
            plot_diffraction_pattern = True
        else:
            raise TypeError('Diffraction information must be in info dictionary of ase.Atoms object')
    else:
        raise TypeError('Diffraction info must be in ase.Atoms object or dictionary form')
        
    if 'output' not in atoms.info:
        return
    
    # Get information from dictionary
    HOLZ = tags_out['HOLZ']
    ZOLZ = tags_out['allowed']['ZOLZ']
    # Kikuchi = tags_out['Kikuchi']

    Laue_Zone = tags_out['allowed']['Laue_Zone']

    label = tags_out['allowed']['label']
    hkl_label = tags_out['allowed']['hkl']

    angle = np.radians(atoms.info['output']['plot_rotation'])  # mrad
    c = np.cos(angle)
    s = np.sin(angle)
    r_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # HOLZ and Kikuchi lines coordinates in Hough space
    LC = tags_out['Laue_circle']
    gd = np.dot(tags_out['HOLZ']['g_deficient'] + LC, r_mat)
    ge = np.dot(tags_out['HOLZ']['g_excess'], r_mat)
    points = np.dot(tags_out['allowed']['g'] + LC, r_mat)

    theta = tags_out['HOLZ']['theta'] + angle

    if 'thickness' not in tags_out:
        tags_out['thickness'] = 0.
    if tags_out['thickness'] > 0.1:
        intensity = np.real(tags_out['allowed']['Ig'])
    else:
        intensity = tags_out['allowed']['intensities']

    radius = atoms.info['experimental']['convergence_angle_A-1']

    if radius < 0.1:
        radiusI = 2
    else:
        radiusI = radius
    # Beginning and ends of HOLZ lines
    max_length = radiusI * 1.3
    h_xp = gd[:, 0] + max_length * np.cos(np.pi - theta)
    h_yp = gd[:, 1] + max_length * np.sin(np.pi - theta)
    h_xm = gd[:, 0] - max_length * np.cos(np.pi - theta)
    h_ym = gd[:, 1] - max_length * np.sin(np.pi - theta)

    # Beginning and ends of excess HOLZ lines
    max_length = radiusI * .8
    e_xp = ge[:, 0] + max_length * np.cos(np.pi - theta)
    e_yp = ge[:, 1] + max_length * np.sin(np.pi - theta)
    e_xm = ge[:, 0] - max_length * np.cos(np.pi - theta)
    e_ym = ge[:, 1] - max_length * np.sin(np.pi - theta)

    # Beginning and ends of Kikuchi lines
    if 'max_length' not in tags_out['Kikuchi']:
        tags_out['Kikuchi']['max_length'] = 20
    max_length = tags_out['Kikuchi']['max_length']

    gd = tags_out['Kikuchi']['g_deficient']
    theta = tags_out['Kikuchi']['theta']
    k_xp = gd[:, 0] + max_length * np.cos(np.pi - theta)
    k_yp = gd[:, 1] + max_length * np.sin(np.pi - theta)
    k_xm = gd[:, 0] - max_length * np.cos(np.pi - theta)
    k_ym = gd[:, 1] - max_length * np.sin(np.pi - theta)

    if atoms.info['output']['linewidth_Kikuchi'] < 0:
        if len(intensity[ZOLZ]) > 0:
            intensity_kikuchi = intensity * 4. / intensity[ZOLZ].max()
        else:
            intensity_kikuchi = intensity
    else:
        intensity_kikuchi = np.ones(len(intensity)) * atoms.info['output']['linewidth_Kikuchi']

    if atoms.info['output']['linewidth_HOLZ'] < 0:
        intensity_holz = np.log(intensity + 1)

        if tags_out['HOLZ']['HOLZ'].any():
            pass  # intensity_holz = intensity/intensity[tags_out['HOLZ']['HOLZ']].max()*4.
    else:
        intensity_holz = np.ones(len(intensity)) * atoms.info['output']['linewidth_HOLZ']

    # #######
    # Plot #
    # #######
    # cms = mpl.cm
    # cm = cms.plasma#jet#, cms.gray, cms.autumn]
    cm = plt.get_cmap(atoms.info['output']['color_map'])

    # fig = plt.figure()
    fig = plt.figure()

    ax = plt.gca()
    if 'background' not in atoms.info['output']:
        atoms.info['output']['background'] = None
    if atoms.info['output']['background'] is not None:
        ax.set_facecolor(atoms.info['output']['background'])

    if diffraction_pattern is not None:
        plt.imshow(diffraction_pattern, extent=diffraction_pattern.get_extent([0, 1]), cmap='gray')

    ix = np.argsort((points ** 2).sum(axis=1))
    p = points[ix]
    inten = intensity[ix]
    reflection = hkl_label[ix]
    laue_color = []

    labelP = ''
    lineLabel = []

    def onpick(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            ind = ax.lines.index(thisline)
            print(ind, len(points), ind - len(points))
            # ind = ind- len(points)
            h, k, l = lineLabel[ind]

            if Laue_Zone[ind] > 0:
                labelP = 'Laue Zone %1d; HOLZ line: [%1d,%1d,%1d]' % (Laue_Zone[ind], h, k, l)
            else:
                labelP = 'Kikuchi line: [%1d,%1d,%1d]' % (h, k, l)
            # print(labelP)

        elif isinstance(event.artist, Circle):
            print('Circle')

        else:
            ind = event.ind[0]
            h, k, l = reflection[ind]

            print('Reflection: [%1d,%1d,%1d]' % (h, k, l))

    for i in range(int(Laue_Zone.max()) + 1):
        if i < len(atoms.info['output']['color_Laue_Zones']):
            laue_color.append(atoms.info['output']['color_Laue_Zones'][i])
        else:
            laue_color.append(atoms.info['output']['color_Laue_Zones'][-1])

    if 'plot_labels' not in atoms.info['output']:
        atoms.info['output']['plot_labels'] = True
    if atoms.info['output']['plot_reflections']:
        if radius < 0.01:
            if atoms.info['output']['color_reflections'] == 'intensity':
                for i in range(len(points)):
                    ax.scatter(points[i, 0], points[i, 1], c=np.log(intensity[i] + 1), cmap=cm, s=100)

                    if atoms.info['output']['plot_labels']:
                        plt.text(points[i, 0], points[i, 1], label[i], fontsize=10)
            else:
                for i in range(len(Laue_Zone)):
                    color = laue_color[int(Laue_Zone[i])]
                    ax.scatter(points[i, 0], points[i, 1], c=color, cmap=cm, s=100)
                    if atoms.info['output']['plot_labels']:
                        plt.text(points[i, 0], points[i, 1], label[i], fontsize=8)

            ax.scatter(LC[0], LC[1], c=atoms.info['output']['color_zero'], s=100)
            radius = .2
        else:
            ix = np.argsort((points ** 2).sum(axis=1))
            p = points[ix]
            inten = intensity[ix]
            if atoms.info['output']['color_reflections'] == 'intensity':
                circles(p[:, 0], p[:, 1], s=radius, c=np.log(inten + 1), cmap=cm, alpha=0.9, edgecolor=None, picker=5)
            else:
                for i in range(len(Laue_Zone)):
                    color = laue_color[int(Laue_Zone[i])]
                    circles(p[i, 0], p[i, 1], s=radius, c=color, cmap=cm, alpha=0.9, edgecolor='', picker=5)  #
                    plt.text(points[i, 0], points[i, 1], label[i], fontsize=8)

    if 'plot_dynamically_allowed' not in atoms.info['output']:
        atoms.info['output']['plot_dynamically_allowed'] = False
    if 'plot_forbidden' not in atoms.info['output']:
        atoms.info['output']['plot_forbidden'] = False

    if atoms.info['output']['plot_dynamically_allowed']:
        if 'dynamically_allowed' not in atoms.info['diffraction']['forbidden']:
            print('To plot dynamically allowed reflections you must run the get_dynamically_allowed function of '
                  'kinematic_scattering library first!')
        else:
            points = atoms.info['diffraction']['forbidden']['g']
            dynamically_allowed = atoms.info['diffraction']['forbidden']['dynamically_allowed']
            dyn_allowed = atoms.info['diffraction']['forbidden']['g'][dynamically_allowed, :]
            dyn_label = atoms.info['diffraction']['forbidden']['hkl'][dynamically_allowed, :]

            color = laue_color[0]
            ax.scatter(dyn_allowed[:, 0], dyn_allowed[:, 1], c='blue', alpha=0.4, s=70)
            if atoms.info['output']['plot_labels']:
                for i in range(len(dyn_allowed)):
                    plt.text(dyn_allowed[i, 0], dyn_allowed[i, 1], dyn_label[i], fontsize=8)
            if atoms.info['output']['plot_forbidden']:
                forbidden_g = atoms.info['diffraction']['forbidden']['g'][np.logical_not(dynamically_allowed), :]
                forbidden_hkl = atoms.info['diffraction']['forbidden']['hkl'][np.logical_not(dynamically_allowed), :]
                ax.scatter(forbidden_g[:, 0], forbidden_g[:, 1], c='orange', alpha=0.4, s=70)
                if atoms.info['output']['plot_labels']:
                    for i in range(len(forbidden_g)):
                        plt.text(forbidden_g[i, 0], forbidden_g[i, 1], forbidden_hkl[i], fontsize=8)
    elif atoms.info['output']['plot_forbidden']:
        forbidden_g = atoms.info['diffraction']['forbidden']['g']
        forbidden_hkl = atoms.info['diffraction']['forbidden']['hkl']
        ax.scatter(forbidden_g[:, 0], forbidden_g[:, 1], c='orange', alpha=0.4, s=70)
        if atoms.info['output']['plot_labels']:
            for i in range(len(forbidden_g)):
                plt.text(forbidden_g[i, 0], forbidden_g[i, 1], forbidden_hkl[i], fontsize=8)

    k = 0
    if atoms.info['output']['plot_HOLZ']:
        for i in range(len(h_xp)):
            if tags_out['HOLZ']['HOLZ'][i]:
                color = laue_color[int(Laue_Zone[i])]
                if atoms.info['output']['plot_HOLZ']:
                    # plot HOLZ lines
                    line, = plt.plot((h_xp[i], h_xm[i]), (h_yp[i], h_ym[i]), c=color, linewidth=intensity_holz[i],
                                     picker=5)
                    if atoms.info['output']['label_HOLZ']:  # Add indices
                        plt.text(h_xp[i], h_yp[i], label[i], fontsize=8)
                    lineLabel.append(hkl_label[i])
                    # print(i, hkl_label[i], intensity_holz[i])

                if atoms.info['output']['plot_HOLZ_excess']:
                    line, = plt.plot((e_xp[i], e_xm[i]), (e_yp[i], e_ym[i]), c=color, linewidth=intensity_holz[i])
                    lineLabel.append(hkl_label[i])

                    if atoms.info['output']['label_HOLZ']:  # Add indices
                        plt.text(e_xp[i], e_yp[i], label[i], fontsize=8)

                    elif atoms.info['output']['label_Kikuchi']:  # Add indices
                        if ZOLZ[i]:
                            plt.text(k_xp[i], k_yp[i], label[i], fontsize=atoms.info['output']['label_size'],
                                     color=atoms.info['output']['label_color'])
                    lineLabel.append(hkl_label[i])
    if atoms.info['output']['plot_Kikuchi']:
        # Beginning and ends of Kikuchi lines
        if atoms.info['output']['label_Kikuchi']:
            label_kikuchi = []
            for i in range(len(label)):
                if ZOLZ[i]:
                    label_kikuchi.append(label[i])
        for i in range(len(k_xp)):
            line, = plt.plot((k_xp[i], k_xm[i]), (k_yp[i], k_ym[i]), c=atoms.info['output']['color_Kikuchi'],
                             linewidth=2)
            if atoms.info['output']['label_Kikuchi']:  # Add indices
                plt.text(k_xp[i], k_yp[i], label[i], fontsize=atoms.info['output']['label_size'],
                         color=atoms.info['output']['label_color'])

    def format_coord(x, y):
        return labelP + 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord

    if atoms.info['output']['color_ring_zero'] != 'None':
        ring = plt.Circle(LC, radius, color=atoms.info['output']['color_ring_zero'], fill=False, linewidth=2)
        ax.add_artist(ring)
        # print(ring)
    if atoms.info['output']['color_zero'] != 'None':
        circle = plt.Circle(LC, radius, color=atoms.info['output']['color_zero'], linewidth=2)
        ax.add_artist(circle)

    plt.axis('equal')
    if 'plot_FOV' in tags_out:
        l = -tags_out['plot_FOV'] / 2
        r = tags_out['plot_FOV'] / 2
        t = -tags_out['plot_FOV'] / 2
        b = tags_out['plot_FOV'] / 2
        plt.xlim(l, r)
        plt.ylim(t, b)

    fig.canvas.mpl_connect('pick_event', onpick)
    # texfig.savefig("HOLZ")

    # plt.title( tags_out['crystal'])
    plt.show()
