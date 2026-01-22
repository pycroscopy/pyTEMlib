import matplotlib
import matplotlib.pyplot as plt
import pyTEMlib

class EdgesAtCursor(object):
    """ Enables interactive identification of edges at cursor position in a matplotlib axis.
    - Left mouse click shows  the most likely edge within maximal_chemical_shift eV.
    - Right mouse click shows all possible edges.
    """
    def __init__(self, ax, energy, data, maximal_chemical_shift=5):
        self.ax = ax
        self.maximal_chemical_shift = maximal_chemical_shift
        self.energy = energy
        self.label = None
        self.line = None
        self.marker, = ax.plot(energy[0], data[0], marker="o", color="crimson", zorder=3)
        self.cursor = matplotlib.widgets.Cursor(ax, useblit=True, color='blue',
                                                linewidth=2, horizOn=False, alpha=.3)
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.edges_on_click)

    def edges_on_click(self, event):
        """ get edges at cursor position """
        if not event.inaxes:
            return
        x= event.xdata
        if self.label is not None:
            self.label.remove()
        if self.line is not None:
            self.line.remove()
        if event.button ==  1:
            self.label = plt.text(x, plt.gca().get_ylim()[1],
                                  pyTEMlib.eels_tools.find_all_edges(x, self.maximal_chemical_shift, True),
                                  verticalalignment='top')
        else:
            self.label = plt.text(x, plt.gca().get_ylim()[1],
                                  pyTEMlib.eels_tools.find_all_edges(x, self.maximal_chemical_shift),
                                  verticalalignment='top')
        self.line = plt.axvline(x=x, color='gray')

class RegionSelector(object):
    """
        Selects fitting region and the regions that are excluded for each edge.

        Select a region with a spanSelector and then type 'a' for all of the 
        fitting region or a number for the edge
        you want to define the region excluded from the fit (solid state effects).

        see Chapter4 'CH4-Working_with_X-Sections,ipynb' notebook
    """
    def __init__(self, ax):
        self.ax = ax
        self.regions = {}
        self.rect = None
        self.xmin = 0
        self.width = 0

        self.span = matplotlib.widgets.SpanSelector(ax, self.onselect1, 'horizontal', useblit=True,
                                 props=dict(alpha=0.5, facecolor='red'), interactive=True)# , span_stays=True)
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self.click)
        self.draw = ax.figure.canvas.mpl_connect('draw_event', self.onresize)

    def onselect1(self, xmin, xmax):
        self.xmin = xmin
        self.width = xmax-xmin

    def onresize(self, event):
        self.update()

    def delete_region(self, key):
        if key in self.regions:
            if 'Rect' in self.regions[key]:
                self.regions[key]['Rect'].remove()
                self.regions[key]['Text'].remove()
            del(self.regions[key])

    def update(self):

        y_min, y_max = self.ax.get_ylim()
        for key in self.regions:
            if 'Rect' in self.regions[key]:
                self.regions[key]['Rect'].remove()
                self.regions[key]['Text'].remove()

            xmin = self.regions[key]['xmin']
            width = self.regions[key]['width']
            height = y_max-y_min
            alpha = self.regions[key]['alpha']
            color = self.regions[key]['color']
            self.regions[key]['Rect'] = matplotlib.patches.Rectangle((xmin, y_min), width, height,
                                                          edgecolor=color, alpha=alpha, facecolor=color)
            self.ax.add_patch(self.regions[key]['Rect'])

            self.regions[key]['Text'] = self.ax.text(xmin, y_max, self.regions[key]['text'], verticalalignment='top')

    def click(self, event):
        if str(event.key) in ['1', '2', '3', '4', '5', '6']:
            key = str(event.key)
            text = 'exclude \nedge ' + key
            alpha = 0.5
            color = 'red'
        elif str(event.key) in ['a', 'A', 'b', 'B', 'f', 'F']:
            key = '0'
            color = 'blue'
            alpha = 0.2
            text = 'fit region'
        else:
            return

        if key not in self.regions:
            self.regions[key] = {}

        self.regions[key]['xmin'] = self.xmin
        self.regions[key]['width'] = self.width
        self.regions[key]['color'] = color
        self.regions[key]['alpha'] = alpha
        self.regions[key]['text'] = text

        self.update()

    def set_regions(self, region, start_x, width):
        """ set regions from dictionary """ 
        if 'fit' in str(region):
            key = '0'
        elif int(region) in [0, 1, 2, 3, 4, 5, 6]:
            key = str(region)
        else:
            return

        self.regions.setdefault(key, {})
        if int(key) > 0:
            self.regions[key]['text'] = 'exclude \nedge ' + key
            self.regions[key]['alpha'] = 0.5
            self.regions[key]['color'] = 'red'
        elif key == '0':
            self.regions[key]['text'] = 'fit region'
            self.regions[key]['alpha'] = 0.2
            self.regions[key]['color'] = 'blue'

        self.regions[key]['xmin'] = start_x
        self.regions[key]['width'] = width

        self.update()

    def get_regions(self):
        tags = {}
        for key, region in self.regions.items():
            if key == '0':
                area = 'fit_area'
            else:
                area = key
            tags[area] = {}
            tags[area]['start_x'] = region['xmin']
            tags[area]['width_x'] = region['width']

        return tags

    def disconnect(self):
        for region in self.regions.values():
            if 'Rect' in region:
                region['Rect'].remove()
                region['Text'].remove()
        del self.span
        self.ax.figure.canvas.mpl_disconnect(self.cid)
