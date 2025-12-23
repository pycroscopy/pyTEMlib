"""
Diffraction pattern plotting tools
part of pyTEMlib's diffraction tools subpackage
author: Gerd Duscher, UTK
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import ase
import numpy as np
import scipy
import sidpy
import skimage



def scattering_profiles(diff_pattern, center):
    """Determine scattering profiles from diffraction pattern
    
    Parameters
    ----------
    diff_pattern :  Dataset
        2D diffraction pattern
    center : tuple
        center of the diffraction pattern (x,y) in pixels
    Returns
    -------
    out_tags : dict
        dictionary with the following entries:
        'center' : center of the diffraction pattern (x,y) in pixels
        'polar_projection' : 2D array with the polar projection (r, theta)
        'radial_average' : 1D array with the radial average
    """
    polar_projection = skimage.transform.warp_polar(diff_pattern, center=center).T
    polar_projection[polar_projection<0.] = 0.

    out_tags={'center': center,
              'polar_projection': polar_projection,
              'radial_average': polar_projection.sum(axis=1)}
    diff_pattern.metadata.setdefault('diffraction', {}).update(out_tags)
    return out_tags

# ##################################
# Plot Reciprocal Unit Cell in 2D #
# ##################################
def plot_reciprocal_unit_cell_2d(atoms):
    """Plot # unit cell in reciprocal space in 2D"""

    reciprocal_unit_cell = atoms.get_reciprocal_cell()

    # ignore y direction

    x = [reciprocal_unit_cell[0, 0], reciprocal_unit_cell[0, 0],
         reciprocal_unit_cell[1, 0], reciprocal_unit_cell[1, 0]]
    z = [reciprocal_unit_cell[0, 2], reciprocal_unit_cell[2, 2],
         reciprocal_unit_cell[2, 2], reciprocal_unit_cell[0, 2]]

    # Plot 2D
    fig = plt.figure()
    ax = plt.gca()  # current axis

    ax.scatter(x, z, c='red', s=80)
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (0, 0),  # (x,y)
            reciprocal_unit_cell[0, 0],  # width
            reciprocal_unit_cell[2, 2],  # height
            fill=False  # remove background
        ))
    ax.add_patch(
        matplotlib.patches.FancyArrow(0, 0, reciprocal_unit_cell[0, 0], 0, width=0.02,
                           color='black',
                           head_width=0.08,  # Default: 3 * width
                           head_length=0.1,  # Default: 1.5 * head_width
                           length_includes_head=True  # Default: False
                           )
    )
    ax.add_patch(
        matplotlib.patches.FancyArrow(0, 0, 0, reciprocal_unit_cell[2, 2], width=0.02,
                                      color='black',
                                      head_width=0.08,  # Default: 3 * width
                                      head_length=0.1,  # Default: 1.5 * head_width
                                      length_includes_head=True))  # Default: False
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
def plot_saed_parameter(gray=False):
    """ Plot SAED pattern parameters"""
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

    tags['color_Laue_Zones'] = ['red', 'blue', 'green', 'blue', 'green']  # for OLZ give a sequence
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
def plot_kikuchi(grey=False):
    """ Plot Kikuchi pattern parameters"""
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
            #for OLZ give a sequence
            'color_Laue_Zones': ['red', 'blue', 'green', 'blue', 'green'],  # , 'green', 'red']
            'color_zero': 'white',  # 'None' #'white'
            'color_ring_zero': 'None',  # 'Red' #'white' #, 'None'
            'width_ring_zero': 2}
    if grey:
        tags['color_map'] = 'gray'
        tags['background'] = '#303030'  # 'darkgray'
        tags['color_reflections'] = 'intensity'
    return tags


########################
# Plot HOLZ Pattern #
########################

def plot_holz_parameter():
    """ Plot HOLZ pattern parameters"""
    tags = {'background': 'gray', 'color_map': 'plasma', 'color_reflections': 'intensity',
            'plot_HOLZ': 1, 'plot_HOLZ_excess': 1, 'plot_Kikuchi': 1, 'plot_reflections': 1,
            'label_HOLZ': 0, 'label_Kikuchi': 0, 'label_reflections': 0, 'label_color': 'white',
            'label_size': 12, 'color_Kikuchi': 'green', 'linewidth_HOLZ': 1, 
            'linewidth_Kikuchi': -1, 'color_ring_zero': 'Red', 'width_ring_zero': 2,
            'color_Laue_Zones': ['red', 'blue', 'lightblue', 'green', 'red'], 'color_zero': 'None',
            'plot_rotation': 0., 'plot_shift_x': -0.0, 'plot_shift_y': 0.0}  # 'white'  'grey'

    # plotDiffPattern(holz,True)
    return tags


########################
# Plot CBED Pattern #
########################

def plot_cbed_parameter():
    """ Plot CBED pattern parameters"""
    tags = {'background': 'black', 'color_map': 'plasma', 'color_reflections': 'intensity',
            'plot_HOLZ': 1, 'plot_HOLZ_excess': 1, 'plot_Kikuchi': 1, 'plot_reflections': 1,
            'label_HOLZ': 0, 'label_Kikuchi': 0, 'label_reflections': 0, 'label_color': 'white',
            'label_size': 10, 'color_Kikuchi': 'green', 'linewidth_HOLZ': -1, 
            'linewidth_Kikuchi': -1, 'color_Laue_Zones': ['red', 'blue', 'green'], 
            'color_zero': 'white', 'color_ring_zero': 'Red', 'width_ring_zero': 2}
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
        kwargs.setdefault('facecolor', kwargs.pop('fc', None))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec', None))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls', None))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw', None))

    zipped = np.broadcast(x, y, s)
    patches = [matplotlib.patches.Circle((x_, y_), s_, picker=True)
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
    """ Transform cartesian grid to polar grid"""
    rr, tt = np.meshgrid(r, t)

    new_x = rr * np.cos(tt)
    new_y = rr * np.sin(tt)

    ix = scipy.interpolate.interp1d(x, np.arange(len(x)))
    iy = scipy.interpolate.interp1d(y, np.arange(len(y)))

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    return scipy.ndimage.map_coordinates(grid, np.array([new_ix, new_iy]),
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

    polar = scipy.ndimage.geometric_transform(img, transform, order=order)

    rads = max_radius * np.linspace(0, 1, img.shape[0])
    angs = np.linspace(0, 2 * np.pi, img.shape[1])

    return polar, (rads, angs)

def set_center(main_dataset, center, scale=None):
    """ Set the u and v axes of a diffraction pattern dataset to center on origin"""
    if scale is None:
        axes= main_dataset.get_image_dims(return_axis=True)
        scale = (axes[0].slope, axes[1].slope)
    if isinstance(scale, float):
        scale = (scale, scale)

    x_axis = np.linspace(0, main_dataset.shape[0]-1, main_dataset.shape[0])-center[1]
    x_axis *= scale[0]
    y_axis = np.linspace(0, main_dataset.shape[1]-1, main_dataset.shape[1])-center[0]
    y_axis *= -scale[1]
    # x = sidpy.Dimension(name='u', values=x_axis)

    main_dataset.set_dimension(0, sidpy.Dimension(name='u', values=x_axis, units='1/nm',
                                                  dimension_type='spatial',
                                                  quantity='reciprocal distance'))
    main_dataset.set_dimension(1, sidpy.Dimension(name='v', values=y_axis, units='1/nm',
                                                  dimension_type='spatial',
                                                  quantity='reciprocal distance'))


def plot_ring_pattern(atoms, diffraction_pattern=None):
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
        if 'Ring_Pattern' in atoms.info:
            tags = atoms.info   
        else:
           raise TypeError('Ring_Pattern information must be in info')
 
    elif hasattr(diffraction_pattern, 'metadata'):
        if diffraction_pattern.metadata.get('Ring_Pattern', {}):
            tags = diffraction_pattern.metadata['Ring_Pattern']
    else:
        raise TypeError('Ring_Pattern info must be in sidpy Dataset or dictionary form')
    if diffraction_pattern is not None:
        if not(diffraction_pattern, sidpy.Dataset):
            print('diffraction_pattern must be a sidpy.Dataset \n -> Ignoring this variable')
            diffraction_pattern = None
    unique = tags['Ring_Pattern']['allowed']['g norm'] *10 # now in 1/nm
    family = tags['Ring_Pattern']['allowed']['hkl']
    intensity = np.array(tags['Ring_Pattern']['allowed']['structure_factor'])
    # label = tags['Ring_Pattern']['allowed']['label']
    
    if diffraction_pattern is not None:
        scale = diffraction_pattern.get_image_dims(return_axis=True)[0].slope
        extent=diffraction_pattern.get_extent([0,1])
        profile = tags.setdefault('radial_average', None)
        profile *=  extent[1]*0.5 /profile.max()
        profile_x = np.linspace(1,len(profile),len(profile))*scale
    else:
        scale = 0.01  # 1/nm per pixel
        extent = (-256*scale, 256*scale, -256*scale, 256*scale)
        profile = None  
    intensity *= extent[1]*0.25 /intensity.max()

    tags.setdefault('label_color', 'navy')
    tags.setdefault('profile color', 'navy')
    tags.setdefault('ring color', 'red')
    tags.setdefault('label_size', 10)
    tags.setdefault('profile height', 5)
    tags.setdefault('plot_scalebar', False)

    fig = plt.figure()
    # ###
    # show image in background
    # ###

    if diffraction_pattern is not None:
        im = plt.imshow(np.log2(1+diffraction_pattern), extent=extent, cmap='gray')
        plt.colorbar(im)
    ax = plt.gca()  # current axis
    ax.set_aspect("equal")

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

    # ###
    # plot arcs of the rings
    # ###
    if profile is not None:
        max_radius = len(profile)
    else:
        max_radius = 512

    for j in range(len(unique)-1):
        if unique[j] < max_radius*scale:
            # plot lines
            plt.plot([unique[j],unique[j]], [0, intensity[j]],c='r')
            arc = matplotlib.patches.Arc((0,0), unique[j]*2, unique[j]*2,
                            angle=90.0,
                            theta1=0.0, theta2=270.0,
                            color='r', fill= False, alpha = 0.5)
            ax.add_artist(arc)

    # ####
    # plot profile
    # ####
    if profile is not None:
        ax.plot(profile_x, profile, c=tags['profile color'])
        ax.plot([0, profile_x[-1]], [0, 0], c=tags['profile color'])
    ax.scatter(0,0)
    ax.set_xlim(extent[0], extent[1])
    for i in range(6):
        # pretty index string
        index = '{'+f'{family[i][0]:.0f} {family[i][1]:.0f} {family[i][2]:.0f}'+'}'
        ax.text(unique[i],-0.05, index, horizontalalignment='center',
                verticalalignment='top', rotation = 'vertical', fontsize=8, color = 'white')
    return fig


def plotting_coordinates(g, rotation=0, feature='spot'):
    """ Convert g-vectors to plotting coordinates"""
    if feature == 'line':
        # Note: d_theta in g{: 3] is negative so we need to rotate phi by 180 degree
        x = g[:, 3] * np.cos(g[:, 1]+np.pi+rotation)*10
        y = g[:, 3] * np.sin(g[:, 1]+np.pi+rotation)*10
        return np.stack((x, y, np.tan(g[:, 1]+rotation-np.pi/2)), axis= 1)

    x = g[:, 0] * np.cos(g[:, 1]+rotation)*10
    y = g[:, 0] * np.sin(g[:, 1]+rotation)*10
    return np.stack((x, y), axis= 1)


def plot_lines(lines, color, alpha, linewidth, label, indices=None):
    """ Plot lines in matplotlib plot"""
    if isinstance(alpha, float):
        alpha = [alpha]* len(lines)
    line = lines[0]
    plt.axline( (line[0], line[1]), slope=line[2], color=color, alpha=alpha[0],
               label=label, linewidth=linewidth[0])
    for i, line in enumerate(lines):
        if i > 0:
            plt.axline( (line[0], line[1]), slope=line[2], color=color,
                       alpha=alpha[i], linewidth=linewidth[i])
            if indices is not None:
                plt.text(line[0], line[1], indices[i], fontsize=8)


def plot_diffraction_pattern(atoms, diffraction_pattern=None):
    """
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
            # plot_diffraction_pattern = True
        else:
            raise TypeError('Diffraction information must be in info dictionary',
                            ' of ase.Atoms object')
    else:
        raise TypeError('Diffraction info must be in ase.Atoms object or dictionary form')

    # Get information from dictionary
    zolz = tags_out['allowed']['ZOLZ']
    folz = tags_out['allowed']['FOLZ']
    solz = tags_out['allowed']['SOLZ']
    # holz = tags_out['allowed']['HOLZ']
    hholz = tags_out['allowed']['HOLZ_plus']

    laue_zone = tags_out['allowed']['Laue_Zone']
    laue_zones = [zolz, folz, solz, hholz]
    laue_circle = tags_out.get('Laue_circle', [0,0, 0, 0])
    hkl_label = tags_out['allowed']['hkl']
    label = tags_out['allowed'].get('label', hkl_label)

    rotation = np.radians(tags_out.setdefault('output', {}).get('plot_rotation', 0))  # rad
    g_vectors = tags_out['allowed']['g'] + laue_circle + [0, rotation, 0 , 0]
    points = plotting_coordinates(g_vectors, feature='spot')
    lines = plotting_coordinates(g_vectors, feature='line')

    tags_out.setdefault('thickness', 0)
    if tags_out['thickness'] > 0.1:
        intensity = np.real(tags_out['allowed']['Ig'])
    else:
        intensity = tags_out['allowed']['intensities']
    radius = tags_out.setdefault('experimental', {}).setdefault('convergence_angle', 0)
    if radius < 0.1:
        radius = 2

    if tags_out['output'].setdefault('linewidth_Kikuchi', 1) < 0:
        if len(intensity[zolz]) > 0:
            intensity_kikuchi = intensity * 4. / intensity[zolz].max()
        else:
            intensity_kikuchi = intensity
    else:
        intensity_kikuchi = np.ones(len(intensity)) * tags_out['output']['linewidth_Kikuchi']

    if tags_out.setdefault('output', {}).setdefault('linewidth_HOLZ', 1)  < 0:
        intensity_holz = np.log(intensity + 1)
    else:
        intensity_holz = np.ones(len(intensity)) * tags_out['output']['linewidth_HOLZ']

    # #######
    # Plot #
    # #######
    cm = plt.get_cmap(tags_out['output'].setdefault('color_map', 'gnuplot'))

    fig = plt.figure()

    ax = plt.gca()
    if tags_out['output'].setdefault('background', None) is not None:
        ax.set_facecolor(tags_out['output']['background'])

    if diffraction_pattern is not None:
        plt.imshow(diffraction_pattern, extent=diffraction_pattern.get_extent([0, 1]), cmap='gray')

    def onpick(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            ind = ax.lines.index(thisline)
            print(ind, len(points), ind - len(points))
            # ind = ind- len(points)
            # h, k, l = line_label[ind]

            # if laue_zone[ind] > 0:
            #     label_p = f'Laue Zone {laue_zone[ind]:d}; HOLZ line: [{h:d},{k:d},{l:d}]'
            # else:
            #    label_p = f'Kikuchi line: [{h:d},{k:d},{l:d}]'
            # print(label_p)

        elif isinstance(event.artist, matplotlib.patches.Circle):
            print('Circle')

        else:
            ind = event.ind[0]
            h, k, l = g_vectors[ind]

            print(f'Reflection: [{h:d},{k:d},{l:d}]')

    laue_color = []
    for i in range(len(laue_zones)):
        if i < len(tags_out['output'].setdefault('color_Laue_Zones',
                                                 ['blue', 'red', 'green', 'orange'])):
            laue_color.append(tags_out['output']['color_Laue_Zones'][i])
        else:
            laue_color.append(tags_out['output']['color_Laue_Zones'][-1])
    tags_out['output'].setdefault('plot_labels', False)

    if tags_out['output'].setdefault('plot_reflections', True):
        if radius < 0.01:
            if tags_out['output'].setdefault('color_reflections', None) == 'intensity':
                ax.scatter(points[:, 0], points[:, 1],
                               c=np.log(intensity[i] + 1), cmap=cm, s=100)

                if tags_out['output']['plot_labels']:
                    plt.text(points[i, 0], points[i, 1], label[i], fontsize=10)
            else:
                for i, zone in enumerate(laue_zones):
                    color = laue_color[i]
                    ax.scatter(points[zone, 0], points[zone, 1], c=color, s=100)
                    if tags_out['output']['plot_labels']:
                        plt.text(points[zone, 0], points[zone, 1], label[i], fontsize=8)
            # TODO in right coordinates
            ax.scatter(laue_circle[0], laue_circle[1],
                       c=tags_out['output'].setdefault('color_zero', 'blue'), s=100)
            radius = .2
        else:
            if tags_out['output'].setdefault('color_reflections', None) == 'intensity':
                circles(points[:, 0], points[:, 1], s=radius, c=np.log(intensity[:] + 1),
                        cmap=cm, alpha=0.9, edgecolor=None, picker=5)
            else:
                for i, zone in enumerate(laue_zones):
                    color = laue_color[i]
                    circles(points[zone, 0], points[zone, 1], s=radius, c=color, cmap=cm,
                            alpha=0.9, edgecolor=None, picker=5)  #
                    #plt.text(points[i, 0], points[i, 1], label[i], fontsize=8)

    tags_out['output'].setdefault('plot_forbidden', False)

    points_forbidden = plotting_coordinates(tags_out['forbidden']['g'])
    if tags_out['output'].setdefault('plot_dynamically_allowed', False):
        if 'dynamically_allowed' not in atoms.info['diffraction']['forbidden']:
            print('To plot dynamically allowed reflections you must run the get_dynamically_allowed'
                  'function of kinematic_scattering library first!')
        else:
            dynamically_allowed = tags_out['forbidden']['dynamically_activated']
            dyn_allowed = points_forbidden[dynamically_allowed]
            color = laue_color[0]
            ax.scatter(dyn_allowed[:, 0], dyn_allowed[:, 1], c='blue', alpha=0.4, s=70)
            if tags_out['output']['plot_labels']:
                for i in range(len(dyn_allowed)):
                    dyn_label = tags_out['forbidden']['hkl'][dynamically_allowed, :]
                    plt.text(dyn_allowed[i, 0], dyn_allowed[i, 1], dyn_label[i], fontsize=8)
            if tags_out['output'].setdefault('plot_forbidden', False):
                forbidden_g = points_forbidden[np.logical_not(dynamically_allowed), :]
                forbidden_hkl = tags_out['forbidden']['hkl'][np.logical_not(dynamically_allowed), :]
                ax.scatter(forbidden_g[:, 0], forbidden_g[:, 1], c='orange', alpha=0.4, s=70)
                if tags_out['output']['plot_labels']:
                    for i in range(len(forbidden_g)):
                        plt.text(forbidden_g[i, 0], forbidden_g[i, 1], forbidden_hkl[i], fontsize=8)
    elif tags_out['output'].setdefault('plot_forbidden', False):
        forbidden_hkl = tags_out['forbidden']['hkl']
        ax.scatter(points_forbidden[:, 0], points_forbidden[:, 1], c='orange', alpha=0.4, s=70)
        if atoms.info['output']['plot_labels']:
            for i, g in enumerate(points_forbidden):
                plt.text(g[0], g[1], forbidden_hkl[i], fontsize=8)

    for i, zone in enumerate(laue_zones):
        if zone.sum() ==0:
            continue
        if i == 0:
            if tags_out['output'].setdefault('plot_Kikuchi',
                                             tags_out['output'].setdefault('plot_HOLZ', False)):
                if tags_out['output'].setdefault('label_HOLZ', False):
                    label = (hkl_label[zone])[i]
                else:
                    label = None
                plot_lines(lines[zone], laue_color[i], 0.5, intensity_kikuchi, 'Kikuchi', label )
        else:
            if tags_out['output'].setdefault('plot_HOLZ', False):
                zone_names= ['Kiku', 'FOLZ', 'SOLZ', 'higher HOLZ']
                if tags_out['output'].setdefault('label_HOLZ', False):
                    label = (hkl_label[zone])[i]
                else:
                    label = None
                plot_lines(lines[zone], laue_color[i], 0.6-i*0.1, intensity_holz[zone],
                           zone_names[i], label)

            if tags_out['output'].setdefault('plot_HOLZ_excess', False):
                excess_s = tags_out['allowed']['g']
                excess_s[:, 3] = tags_out['allowed']['g'][:, 1] - tags_out['allowed']['g'][:, 3]
                excess_s[:, 1] += np.pi
                lines_excess = plotting_coordinates(excess_s, feature='line')
                plot_lines(lines_excess[zone], laue_color[i], 0.6-i*0.1,
                           intensity_holz[zone],
                           zone_names[i])


    """    if atoms.info['output']['plot_Kikuchi']:
            # Beginning and ends of Kikuchi lines
            if atoms.info['output']['label_Kikuchi']:
                label_kikuchi = []
                for i, text in enumerate(label):
                    if zolz[i]:
                        label_kikuchi.append(text)
            for i, k_x in enumerate(k_xp):
                _, = plt.plot((k_x, k_xm[i]), (k_yp[i], k_ym[i]),
                              c=atoms.info['output']['color_Kikuchi'], linewidth=2)
                if atoms.info['output']['label_Kikuchi']:  # Add indices
                    plt.text(k_xp[i], k_yp[i], label[i],
                             fontsize=atoms.info['output']['label_size'],
                             color=atoms.info['output']['label_color'])
    """
    def format_coord(x, y):
        return  f'x={x:.4f}, y={y:.4f}' # label_p + f'x={x:.4f}, y={y:.4f}'

    ax.format_coord = format_coord

    if tags_out['output'].setdefault('color_ring_zero', None) is not None:
        ring = plt.Circle(laue_circle, radius, color=tags_out['output']['color_ring_zero'],
                          fill=False, linewidth=2)
        ax.add_artist(ring)
    if tags_out['output'].setdefault('color_zero', None) is not None:
        circle = plt.Circle(laue_circle, radius,
                            color=tags_out['output']['color_zero'],
                            linewidth=2)
        ax.add_artist(circle)

    plt.axis('equal')
    if tags_out['output'].setdefault('plot_FOV', None):
        l = -tags_out['output']['plot_FOV'] / 2
        r = tags_out['output']['plot_FOV'] / 2
        t = -tags_out['output']['plot_FOV'] / 2
        b = tags_out['output']['plot_FOV'] / 2
        plt.xlim(l, r)
        plt.ylim(t, b)

    fig.canvas.mpl_connect('pick_event', onpick)
    # texfig.savefig("HOLZ")
    plt.legend()
    # plt.title( tags_out['crystal'])
    return fig
