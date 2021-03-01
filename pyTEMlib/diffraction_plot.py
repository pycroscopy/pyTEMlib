import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Circle  # , Ellipse, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from scipy.ndimage.interpolation import geometric_transform

import numpy as np

###################################
# Plot Reciprocal Unit Cell in 2D #
###################################


def plot_reciprocal_unit_cell_2D(tags):
    # Plot # unit cell in reciprocal space in 2D
    reciprocal_unit_cell = tags['reciprocal_unit_cell']

    # ignore y direction

    x = [reciprocal_unit_cell[0, 0], reciprocal_unit_cell[0, 0], reciprocal_unit_cell[1, 0], reciprocal_unit_cell[1, 0]]
    z = [reciprocal_unit_cell[0, 2], reciprocal_unit_cell[2, 2], reciprocal_unit_cell[2, 2], reciprocal_unit_cell[0, 2]]

    print(x, z)
    print(reciprocal_unit_cell)

    # Plot 2D
    fig = plt.figure()
    # ax = fig.add_subplot(111)
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


#####################
# Plot SAED Pattern #
#####################
def plotSAED_parameter(tags, gray=False):
    tags['convergence_angle_nm-1'] = 0

    tags['background'] = 'white'  # 'white'  'grey'
    tags['color map'] = 'plasma'  # ,'cubehelix'#'Greys'#'plasma'
    tags['color reflections'] = 'intensity'

    if gray:
        tags['color map'] = 'gray'
        tags['background'] = '#303030'  # 'darkgray'
        tags['color reflections'] = 'intensity'
    tags['plot HOLZ'] = 0
    tags['plot HOLZ excess'] = 0
    tags['plot Kikuchi'] = 1
    tags['plot reflections'] = 1

    tags['color Kikuchi'] = 'green'

    tags['linewidth HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    tags['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2

    tags['label HOLZ'] = 0
    tags['label Kikuchi'] = 0
    tags['label reflections'] = 0

    tags['label color'] = 'white'
    tags['label size'] = 10

    tags['color Laue Zones'] = ['red', 'blue', 'green', 'blue', 'green']  # , 'green', 'red'] #for OLZ give a sequence
    tags['color zero'] = 'red'  # 'None' #'white'
    tags['color ring zero'] = 'None'  # 'Red' #'white' #, 'None'
    tags['width ring zero'] = 2

    # plotDiffPattern(tags,True)
    tags['plot rotation'] = 0.  # degree
    tags['plot shift x'] = -0.0
    tags['plot shift y'] = 0.0


########################
# Plot Kikuchi Pattern #
########################
def plotKikuchi(tags, grey=False):
    tags['background'] = 'black'  # 'white'  'grey'
    tags['color map'] = 'plasma'  # ,'cubehelix'#'Greys'#'plasma'
    tags['color reflections'] = 'intensity'

    tags['plot HOLZ'] = 0
    tags['plot HOLZ excess'] = 0
    tags['plot Kikuchi'] = 1
    tags['plot reflections'] = 1

    tags['label HOLZ'] = 0
    tags['label Kikuchi'] = 0
    tags['label reflections'] = 0

    tags['label color'] = 'white'
    tags['label size'] = 10

    tags['color Kikuchi'] = 'green'
    tags['linewidth HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    tags['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2

    tags['color Laue Zones'] = ['red', 'blue', 'green', 'blue', 'green']  # , 'green', 'red'] #for OLZ give a sequence
    tags['color zero'] = 'white'  # 'None' #'white'
    tags['color ring zero'] = 'None'  # 'Red' #'white' #, 'None'
    tags['width ring zero'] = 2

    # plotDiffPattern(tags,True)


########################
# Plot HOLZ Pattern #
########################

def plotHOLZ_parameter(tags, grey=False):
    tags['background'] = 'gray'  # 'white'  'grey'
    tags['color map'] = 'plasma'  # ,'cubehelix'#'Greys'#'plasma'
    tags['color reflections'] = 'intensity'

    tags['plot HOLZ'] = 1
    tags['plot HOLZ excess'] = 1
    tags['plot Kikuchi'] = 1
    tags['plot reflections'] = 1

    tags['label HOLZ'] = 0
    tags['label Kikuchi'] = 0
    tags['label reflections'] = 0

    tags['label color'] = 'white'
    tags['label size'] = 12

    tags['color Kikuchi'] = 'green'
    tags['linewidth HOLZ'] = 1  # -1: linewidth according to intensity (structure factor F^2
    tags['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2

    tags['color Laue Zones'] = ['red', 'blue', 'lightblue', 'green', 'red']  # for OLZ give a sequence
    tags['color zero'] = 'None'  # 'white'
    tags['color ring zero'] = 'Red'  # 'Red' #'white' #, 'None'
    tags['width ring zero'] = 2

    tags['plot rotation'] = 0.  # degree
    tags['plot shift x'] = -0.0
    tags['plot shift y'] = 0.0

    # plotDiffPattern(holz,True)


########################
# Plot CBED Pattern #
########################

def plotCBED_parameter(tags):
    tags['background'] = 'black'  # 'white'  'grey'
    tags['color map'] = 'plasma'  # ,'cubehelix'#'Greys'#'plasma'
    tags['color reflections'] = 'intensity'

    tags['plot HOLZ'] = 1
    tags['plot HOLZ excess'] = 1
    tags['plot Kikuchi'] = 1
    tags['plot reflections'] = 1

    tags['label HOLZ'] = 0
    tags['label Kikuchi'] = 0
    tags['label reflections'] = 0

    tags['label color'] = 'white'
    tags['label size'] = 10

    tags['color Kikuchi'] = 'green'
    tags['linewidth HOLZ'] = -1  # -1: linewidth according to intensity (structure factor F^2
    tags['linewidth Kikuchi'] = -1  # -1: linewidth according to intensity (structure factor F^2

    tags['color reflections'] = 'intensity'

    tags['color Laue Zones'] = ['red', 'blue', 'green']  # , 'green', 'red'] #for OLZ give a sequence
    tags['color zero'] = 'white'  # 'None' #'white'
    tags['color ring zero'] = 'Red'  # 'Red' #'white' #, 'None'
    tags['width ring zero'] = 2

    # plotDiffPattern(tags,True)


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


def plotRingPattern(tags, grey=False):
    d = tags['Ring_Pattern']['allowed']['g norm']
    label = tags['Ring_Pattern']['allowed']['label']
    if 'label color' not in tags:
        tags['label color'] = 'navy'
    if 'profile color' not in tags:
        tags['profile color'] = 'navy'
    if 'ring color' not in tags:
        tags['ring color'] = 'red'
    if 'label size' not in tags:
        tags['label size'] = 10
    if 'profile height' not in tags:
        tags['profile height'] = 5
    if 'plot scalebar' not in tags:
        tags['plot scalebar'] = False
    fg, ax = plt.subplots(1, 1)

    ####
    # plot arcs of the rings
    ####
    for i in range(len(d)):
        pac = patches.Arc((0, 0), d[i] * 2, d[i] * 2, angle=0, theta1=45, theta2=360, color=tags['ring color'])
        ax.add_patch(pac)

    ####
    # show image in background
    ####
    if 'plot image' in tags:
        l = -tags['plot image FOV'] / 2 + tags['plot shift x']
        r = tags['plot image FOV'] / 2 + tags['plot shift x']
        t = -tags['plot image FOV'] / 2 + tags['plot shift y']
        b = tags['plot image FOV'] / 2 + tags['plot shift y']
        if 'plot image FOV Y' in tags:
            t = -tags['plot image FOV Y'] / 2 + tags['plot shift y']
            b = tags['plot image FOV Y'] / 2 + tags['plot shift y']

        plt.imshow(tags['plot image'], extent=(l, r, t, b), cmap='gray')

    ax.set_aspect("equal")

    # fg.canvas.draw()

    if tags['plot scalebar']:
        def f(axis):
            l = axis.get_majorticklocs()
            return len(l) > 1 and (l[1] - l[0])

        sizex = f(ax.xaxis)
        labelx = str(sizex) + ' 1/nm'
        scalebar = AnchoredSizeBar(ax.transData, sizex, labelx, loc=3,
                                   pad=0.5, color='white', frameon=False)
        # size_vertical=.2, fill_bar = True) # will be implented in matplotlib 2.1

        ax.add_artist(scalebar)
        ax.axis('off')

    #####
    # plot profile
    #####

    y = tags['Ring_Pattern']['profile_y']
    y = y / y.max() * tags['profile height']
    x = tags['Ring_Pattern']['profile_x']
    ax.plot(x, y, c=tags['profile color'])

    ax.plot([0, x[-1]], [0, 0], c=tags['profile color'])

    if 'experimental profile y' in tags:
        yy = tags['experimental profile y']
        yy = yy / yy.max() * tags['profile height']
        xx = tags['experimental profile x']
        ax.plot(xx, yy, c=tags['experimental profile color'])

    if 'plot image FOV' in tags:
        max_d = tags['plot image FOV'] / 2 + tags['plot shift x']
    else:
        max_d = d.max()
    for i in range(len(d)):
        if d[i] < max_d:
            plt.text(d[i] - .2, -.5, label[i], fontsize=tags['label size'], color=tags['label color'], rotation=90)

    if 'plot FOV' in tags:
        l = -tags['plot FOV'] / 2
        r = tags['plot FOV'] / 2
        t = -tags['plot FOV'] / 2
        b = tags['plot FOV'] / 2
        plt.xlim(l, r)
        plt.ylim(t, b)

    fg.show()


def plot_diffraction_pattern(tagsD, grey=False):
    # Get information from dictionary
    HOLZ = tagsD['HOLZ']
    ZOLZ = tagsD['allowed']['ZOLZ']
    Kikuchi = tagsD['Kikuchi']

    Laue_Zone = tagsD['allowed']['Laue_Zone']

    label = tagsD['allowed']['label']
    hkl_label = tagsD['allowed']['hkl']

    angle = np.radians(tagsD['plot rotation'])  # mrad
    c = np.cos(angle)
    s = np.sin(angle)
    r_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # HOLZ and Kikuchi lines coordinates in Hough space
    LC = tagsD['Laue_circle']
    gd = np.dot(tagsD['HOLZ']['g deficient'] + LC, r_mat)
    ge = np.dot(tagsD['HOLZ']['g excess'] + LC, r_mat)
    points = np.dot(tagsD['allowed']['g'] + LC, r_mat)

    theta = tagsD['HOLZ']['theta'] + angle

    if 'thickness' not in tagsD:
        tagsD['thickness'] = 0.
    if tagsD['thickness'] > 0.1:
        intensity = np.real(tagsD['allowed']['Ig'])
    else:
        intensity = tagsD['allowed']['intensities']

    radius = tagsD['convergence_angle_nm-1']

    if radius < 0.1:
        radiusI = 2
    else:
        radiusI = radius
    # Beginning and ends of HOLZ lines
    maxlength = radiusI * 1.3
    Hxp = gd[:, 0] + maxlength * np.cos(np.pi - theta)
    Hyp = gd[:, 1] + maxlength * np.sin(np.pi - theta)
    Hxm = gd[:, 0] - maxlength * np.cos(np.pi - theta)
    Hym = gd[:, 1] - maxlength * np.sin(np.pi - theta)

    # Beginning and ends of excess HOLZ lines
    maxlength = radiusI * .8
    Exp = ge[:, 0] + maxlength * np.cos(np.pi - theta)
    Eyp = ge[:, 1] + maxlength * np.sin(np.pi - theta)
    Exm = ge[:, 0] - maxlength * np.cos(np.pi - theta)
    Eym = ge[:, 1] - maxlength * np.sin(np.pi - theta)

    # Beginning and ends of HOLZ lines
    maxlength = 20
    Kxp = gd[:, 0] + maxlength * np.cos(np.pi - theta)
    Kyp = gd[:, 1] + maxlength * np.sin(np.pi - theta)
    Kxm = gd[:, 0] - maxlength * np.cos(np.pi - theta)
    Kym = gd[:, 1] - maxlength * np.sin(np.pi - theta)

    if tagsD['linewidth Kikuchi'] < 0:
        if len(intensity[ZOLZ]) > 0:
            intensity_Kikuchi = intensity * 4. / intensity[ZOLZ].max()
        else:
            intensity_Kikuchi = intensity
    else:
        intensity_Kikuchi = np.ones(len(intensity)) * tagsD['linewidth Kikuchi']

    if tagsD['linewidth HOLZ'] < 0:
        intensity_HOLZ = np.log(intensity + 1)

        if tagsD['HOLZ']['HOLZ'].any():
            pass  # intensity_HOLZ = intensity/intensity[tagsD['HOLZ']['HOLZ']].max()*4.
    else:
        intensity_HOLZ = np.ones(len(intensity)) * tagsD['linewidth HOLZ']

    ########
    # Plot #
    ########
    cms = mpl.cm
    # cm = cms.plasma#jet#, cms.gray, cms.autumn]
    cm = plt.get_cmap(tagsD['color map'])

    # fig = plt.figure()
    fig = plt.figure()

    ax = plt.gca()
    if 'background' not in tagsD:
        tagsD['background'] = None
    if tagsD['background'] is not None:
        ax.set_facecolor(tagsD['background'])

    if 'plot image' in tagsD:
        l = -tagsD['plot image FOV'] / 2 + tagsD['plot shift x']
        r = tagsD['plot image FOV'] / 2 + tagsD['plot shift x']
        t = -tagsD['plot image FOV'] / 2 + tagsD['plot shift y']
        b = tagsD['plot image FOV'] / 2 + tagsD['plot shift y']
        if 'plot image FOV Y' in tagsD:
            t = -tagsD['plot image FOV Y'] / 2 + tagsD['plot shift y']
            b = tagsD['plot image FOV Y'] / 2 + tagsD['plot shift y']

        plt.imshow(tagsD['plot image'], extent=(l, r, t, b))

    ix = np.argsort((points ** 2).sum(axis=1))
    p = points[ix]
    inten = intensity[ix]
    reflection = hkl_label[ix]
    Lauecolor = []

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
        if i < len(tagsD['color Laue Zones']):
            Lauecolor.append(tagsD['color Laue Zones'][i])
        else:
            Lauecolor.append(tagsD['color Laue Zones'][-1])

    if 'plot_labels' not in tagsD:
        tagsD['plot_labels'] = True
    if tagsD['plot reflections']:
        print(radius)
        if radius < 0.1:
            if tagsD['color reflections'] == 'intensity':
                for i in range(len(points)):
                    ax.scatter(points[i, 0], points[i, 1], c=np.log(intensity[i] + 1), cmap=cm, s=100)

                    if tagsD['plot_labels']:
                            plt.text(points[i, 0], points[i, 1], label[i], fontsize=10)
                    #print(label[i])

            else:
                for i in range(len(Laue_Zone)):
                    color = Lauecolor[int(Laue_Zone[i])]
                    ax.scatter(points[i, 0], points[i, 1], c=color, cmap=cm, s=100)
                    if tagsD['plot_labels']:
                        plt.text(points[i, 0], points[i, 1], label[i], fontsize=8)
                    #print(label[i])

            ax.scatter(0, 0, c=tagsD['color zero'], s=100)
            radius = 2
        else:
            ix = np.argsort((points ** 2).sum(axis=1))
            p = points[ix]
            inten = intensity[ix]
            if tagsD['color reflections'] == 'intensity':
                circles(p[:, 0], p[:, 1], s=radius, c=np.log(inten + 1), cmap=cm, alpha=0.9, edgecolor='', picker=5)  #
            else:
                for i in range(len(Laue_Zone)):
                    color = Lauecolor[int(Laue_Zone[i])]
                    circles(p[i, 0], p[i, 1], s=radius, c=color, cmap=cm, alpha=0.9, edgecolor='', picker=5)  #
                    plt.text(points[i, 0], points[i, 1], label[i], fontsize=8)

    k = 0
    if tagsD['plot HOLZ']:
        for i in range(len(Hxp)):

            if tagsD['HOLZ']['HOLZ'][i]:
                color = Lauecolor[int(Laue_Zone[i])]
                if tagsD['plot HOLZ']:
                    # plot HOLZ lines
                    line, = plt.plot((Hxp[i], Hxm[i]), (Hyp[i], Hym[i]), c=color, linewidth=intensity_HOLZ[i], picker=5)
                    if tagsD['label HOLZ']:  ## Add indices
                        plt.text(Hxp[i], Hyp[i], label[i], fontsize=8)
                    lineLabel.append(hkl_label[i])
                    # print(i, hkl_label[i], intensity_HOLZ[i])

                if tagsD['plot HOLZ excess']:
                    line, = plt.plot((Exp[i], Exm[i]), (Eyp[i], Eym[i]), c=color, linewidth=intensity_HOLZ[i])
                    lineLabel.append(hkl_label[i])

                    if tagsD['label HOLZ']:  ## Add indices
                        plt.text(Exp[i], Eyp[i], label[i], fontsize=8)


            else:
                # Plot Kikuchi lines
                if tagsD['plot Kikuchi']:
                    line, = plt.plot((Kxp[i], Kxm[i]), (Kyp[i], Kym[i]), c=tagsD['color Kikuchi'],
                                     linewidth=intensity_Kikuchi[i] / 10, picker=5)
                    if tagsD['label Kikuchi']:  # Add indices

                        plt.text(Kxp[i], Kyp[i], label[i], fontsize=tagsD['label size'], color=tagsD['label color'])
                    lineLabel.append(hkl_label[i])
    elif tagsD['plot Kikuchi']:
        # Beginning and ends of Kikuchi lines
        maxlength = tagsD['plot FOV']
        gd = tagsD['Kikuchi']['min dist']
        theta = tagsD['Kikuchi']['theta']
        Kxp = gd[:, 0] + maxlength * np.cos(np.pi - theta)
        Kyp = gd[:, 1] + maxlength * np.sin(np.pi - theta)
        Kxm = gd[:, 0] - maxlength * np.cos(np.pi - theta)
        Kym = gd[:, 1] - maxlength * np.sin(np.pi - theta)
        for i in range(len(Kxp)):
            line, = plt.plot((Kxp[i], Kxm[i]), (Kyp[i], Kym[i]), c=tagsD['color Kikuchi'], linewidth=2)

    def format_coord(x, y):
        return labelP + 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord

    if not (tagsD['color ring zero'] == 'None'):
        ring = plt.Circle((0, 0), radius, color=tagsD['color ring zero'], fill=False, linewidth=2)
        ax.add_artist(ring)
        # print(ring)
    if not tagsD['color zero'] == 'None':
        circle = plt.Circle((0, 0), radius, color=tagsD['color zero'], linewidth=2)
        ax.add_artist(circle)

    plt.axis('equal')
    if 'plot FOV' in tagsD:
        l = -tagsD['plot FOV'] / 2
        r = tagsD['plot FOV'] / 2
        t = -tagsD['plot FOV'] / 2
        b = tagsD['plot FOV'] / 2
        plt.xlim(l, r)
        plt.ylim(t, b)

    fig.canvas.mpl_connect('pick_event', onpick)
    # texfig.savefig("HOLZ")

    # plt.title( tagsD['crystal'])
    plt.show()
