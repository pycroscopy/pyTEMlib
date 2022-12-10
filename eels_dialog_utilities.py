""" Interactive routines for EELS analysis

this file provides additional dialogs for EELS quantification

Author: Gerd Duscher
"""

import numpy as np
Qt_available = True
try:
    from PyQt5 import QtCore, QtGui, QtWidgets

except:
    Qt_available = False
    print('Qt dialogs are not available')

import sidpy
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector, SpanSelector

import h5py  # TODO: needs to go
import matplotlib.pyplot as plt

from IPython.display import display
import ipywidgets as widgets

from pyTEMlib import eels_tools as eels
from pyTEMlib import file_tools as ft

major_edges = ['K1', 'L3', 'M5', 'N5']
all_edges = ['K1', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2',
             'O3', 'O4', 'O5', 'O6', 'O7', 'P1', 'P2', 'P3']
first_close_edges = ['K1', 'L3', 'M5', 'M3', 'N5', 'N3']

if Qt_available:

    class PeriodicTableDialog(QtWidgets.QDialog):
        """ Modal dialog to get a selection of elements.

        Elements that are not having a valid cross-sections are disabled.

        Parameters
        ----------
        initial_elements: list of str
            the elements that are already selected
        energy_scale: list or numpy array
            energy-scale of spectrum/spectra to determine likely edges

        Returns
        -------
        list of strings: elements.

        Example
        -------
        >> PT_dialog =  periodic_table_dialog(None, ['Mn', 'O'])
        >> if PT_dialog.exec_() == periodic_table_dialog.Accepted:
        >>     selected_elements = PT_dialog.get_output()
        >> print(selected_elements)
        """

        signal_selected = QtCore.pyqtSignal(list)

        def __init__(self, initial_elements=None, energy_scale=None, parent=None):
            super(PeriodicTableDialog, self).__init__(None, QtCore.Qt.WindowStaysOnTopHint)

            if initial_elements is None:
                initial_elements = [' ']
            if energy_scale is None:
                energy_scale = [100., 150., 200.]
            self.parent = parent
            self._output = []
            self.elements_selected = initial_elements
            self.energy_scale = np.array(energy_scale)

            self.setWindowTitle("Periodic Table")
            likely_edges = get_likely_edges(self.energy_scale)
            self.likely_edges = likely_edges

            # GD:font = wx.Font(10, wx.MODERN, wx.NORMAL, wx.BOLD)
            self.buttons1 = []
            self.button = []
            self.pt_info = get_periodic_table_info()
            self.init_ui()

            for button in self.button:
                if button.text() in initial_elements:
                    button.toggle()
                pass

        def on_close(self):
            self.get_output()
            self.signal_selected[list].emit(self._output)
            self.accept()

        def get_output(self):
            self._output = []
            for btn in self.button:
                if btn.isChecked():
                    self._output.append(btn.text())

        def exec_(self):
            super(PeriodicTableDialog, self).exec_()
            return self._output

        def init_ui(self):

            v_sizer = QtWidgets.QVBoxLayout()
            g_sizer = QtWidgets.QGridLayout()

            main_group = QtWidgets.QWidget()

            color1 = "background-color: lightblue;\n"
            color1l = "background-color: dodgerblue;\n"
            color2 = "background-color: coral;\n"

            for symbol, parameter in self.pt_info.items():
                self.button.append(QtWidgets.QPushButton(symbol))
                if parameter['PT_row'] > 7:
                    self.button[-1].setStyleSheet(color2)
                elif '*' in symbol:
                    self.button[-1].setStyleSheet(color2)
                else:
                    if symbol in self.likely_edges:
                        self.button[-1].setStyleSheet(color1l)
                    else:
                        self.button[-1].setStyleSheet(color1)
                if parameter['Z'] == 0:
                    self.button[-1].setEnabled(False)
                self.button[-1].setFixedWidth(50)
                self.button[-1].setCheckable(True)
                g_sizer.addWidget(self.button[-1], parameter['PT_row'], parameter['PT_col'])
            main_group.setLayout(g_sizer)

            v_sizer.addWidget(main_group)
            self.setLayout(v_sizer)

            ok_button = QtWidgets.QPushButton('OK')
            ok_button.clicked.connect(self.on_close)

            v_sizer.addWidget(ok_button)
            self.setLayout(v_sizer)


    class EnergySelector(QtWidgets.QDialog):
        """Dialog and cursor to set energy scale"""

        signal_selected = QtCore.pyqtSignal(bool)

        def __init__(self, dset=None):
            super(EnergySelector, self).__init__(None, QtCore.Qt.WindowStaysOnTopHint)

            if not isinstance(dset, sidpy.Dataset):
                return
            if dset is None:
                return
            if dset.view is None:
                return
            self.dataset = dset

            if hasattr(dset.view, 'axis'):
                self.axis = dset.view.axis
                # self.setWindowTitle('p')
            elif hasattr(dset.view, 'axes'):
                self.axis = dset.view.axes[1]
            else:
                return

            self.spec_dim = -1
            for dim, axis in self.dataset._axes.items():
                if axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                    self.spec_dim = dim
            if self.spec_dim < 0:
                raise TypeError('We need at least one SPECTRAL dimension')

            self.energy_scale = self.dataset._axes[self.spec_dim].values
            self.dispersion = self.energy_scale[1] - self.energy_scale[0]
            self.offset = self.energy_scale[0]
            self.spectrum = np.zeros(2)

            self.change = 0

            self.x_min = self.energy_scale[int(len(self.energy_scale)/4)]
            self.x_max = self.energy_scale[int(len(self.energy_scale) / 4*3)]
            self.setWindowTitle("Select Energy")

            valid_float = QtGui.QDoubleValidator()

            layout = QtWidgets.QGridLayout()
            layout.setVerticalSpacing(2)
            self.label1 = QtWidgets.QLabel('Start:')
            self.edit1 = QtWidgets.QLineEdit('0')
            self.edit1.setValidator(valid_float)
            self.unit1 = QtWidgets.QLabel('eV')

            self.label2 = QtWidgets.QLabel('End:')
            self.edit2 = QtWidgets.QLineEdit('0')
            self.edit2.setValidator(valid_float)
            self.unit2 = QtWidgets.QLabel('eV')

            self.label3 = QtWidgets.QLabel('Dispersion:')
            self.edit3 = QtWidgets.QLineEdit('0')
            self.edit3.setValidator(valid_float)
            self.unit3 = QtWidgets.QLabel('eV')

            self.edit1.editingFinished.connect(self.on_enter)
            self.edit2.editingFinished.connect(self.on_enter)
            self.edit3.editingFinished.connect(self.on_enter)

            layout.addWidget(self.label1, 0, 0)
            layout.addWidget(self.edit1, 0, 1)
            layout.addWidget(self.unit1, 0, 2)

            layout.addWidget(self.label2, 1, 0)
            layout.addWidget(self.edit2, 1, 1)
            layout.addWidget(self.unit2, 1, 2)

            layout.addWidget(self.label3, 2, 0)
            layout.addWidget(self.edit3, 2, 1)
            layout.addWidget(self.unit3, 2, 2)

            self.ok_button = QtWidgets.QPushButton('OK')
            self.ok_button.clicked.connect(self.on_close)
            self.cancel_button = QtWidgets.QPushButton('Cancel')
            self.cancel_button.clicked.connect(self.on_close)

            layout.addWidget(self.ok_button, 3, 0)
            layout.addWidget(self.cancel_button, 3, 2)

            self.setLayout(layout)
            self.edit1.setFocus()
            self.plot()

            self.selector = SpanSelector(self.axis, self.line_select_callback,
                                         direction="horizontal",
                                         span_stays=True,
                                         props=dict(facecolor='blue', alpha=0.2))
            self.edit1.setText(f'{self.x_min:.3f}')
            self.edit2.setText(f'{self.x_max:.3f}')
            self.edit3.setText(f'{self.dispersion:.4f}')
            self.update()

        def line_select_callback(self, eclick, erelease):
            y_min, y_max = self.axis.get_ylim()
            self.x_min = self.selector.extents[0]
            self.x_max = self.selector.extents[1]
            # self.selector.extents = (self.x_min, self.x_max, y_min, y_max)

            self.edit1.setText(f'{self.x_min:.3f}')
            self.edit2.setText(f'{self.x_max:.3f}')

        def on_enter(self):
            sender = self.sender()

            if sender == self.edit1:
                value = float(str(sender.displayText()).strip())
                if value == self.x_min:
                    return
                self.change = value - self.x_min
                self.x_min += self.change
                self.x_max += self.change
                self.offset += self.change

                self.edit1.setText(f"{self.x_min:.2f}")
                self.edit2.setText(f"{self.x_max:.2f}")

                self.energy_scale = np.arange(len(self.energy_scale)) * self.dispersion + self.offset

                self.update()
                # self.axis.draw()
                # self.setWindowTitle(f'shift, {self.change}, {self.x_min}')

            elif sender == self.edit2:
                value = float(str(sender.displayText()).strip())
                if value == self.x_max:
                    return
                start_channel = np.searchsorted(self.energy_scale, self.x_min)
                end_channel = np.searchsorted(self.energy_scale, self.x_max)

                self.x_max = value

                if end_channel - start_channel != 0:
                    self.dispersion = (self.x_max - self.x_min) / (end_channel - start_channel)
                self.offset = self.x_min - start_channel * self.dispersion
                self.edit2.setText(f"{self.x_max:.3f}")
                self.edit3.setText(f"{self.dispersion:.4f}")
                self.energy_scale = np.arange(len(self.energy_scale)) * self.dispersion + self.offset

                self.update()
                # self.axis.draw()
                # self.setWindowTitle(f'range, {self.change}, {self.dispersion}')

            elif sender == self.edit3:
                value = float(str(sender.displayText()).strip())
                if self.dispersion == value:
                    return

                start_channel = np.searchsorted(self.energy_scale, self.x_min)
                end_channel = np.searchsorted(self.energy_scale, self.x_max)
                self.dispersion = value
                self.energy_scale = np.arange(len(self.energy_scale)) * self.dispersion + self.offset
                self.x_min = self.energy_scale[start_channel]
                self.x_max = self.energy_scale[end_channel]
                self.update()
                # self.axis.draw()
                self.edit3.setText(f"{self.dispersion:.3f}")
                self.change = 0

        def on_close(self):
            sender = self.sender()
            if sender == self.ok_button:
                pass
                self.dataset.set_dimension(self.spec_dim, sidpy.Dimension(self.energy_scale, name='energy_scale',
                                                                          units='eV', quantity='energy loss',
                                                                          dimension_type='spectral'))
            else:
                pass
            self.selector.set_visible(False)
            self.signal_selected[bool].emit(True)
            self.accept()

        def plot(self):
            if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                self.spectrum = self.dataset.view.get_spectrum()
            else:
                self.spectrum = np.array(self.dataset)
            x_limit = self.axis.get_xlim()
            y_limit = self.axis.get_ylim()

            self.axis.clear()
            self.cplot = self.axis.plot(self.energy_scale, self.spectrum, label='spectrum')
            self.axis.set_xlim(x_limit)
            self.axis.set_ylim(y_limit)

            self.axis.figure.canvas.draw()

        def update(self):
            x_limit = self.axis.get_xlim()
            y_limit = self.axis.get_ylim()
            self.selector.extents = (self.x_min, self.x_max)

            x_limit = np.array(x_limit) + self.change

            self.cplot[0].set_data(self.energy_scale, self.spectrum)
            self.axis.set_xlim(x_limit)
            self.axis.set_ylim(y_limit)
            self.axis.figure.canvas.draw()


class RegionSelector(object):
    """Selects fitting region and the regions that are excluded for each edge.

        Select a region with a spanSelector and then type 'a' for all the fitting region or a number for the edge
        you want to define the region excluded from the fit (solid state effects).

        see Chapter4 'CH4-Working_with_X-Sections,ipynb' notebook

    """

    def __init__(self, ax):
        self.ax = ax
        self.regions = {}
        self.rect = None
        self.xmin = 0
        self.width = 0

        self.span = SpanSelector(ax, self.on_select1, 'horizontal', useblit=True,
                                 rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)
        self.cid = ax.figure.canvas.mpl_connect('key_press_event', self.click)
        self.draw = ax.figure.canvas.mpl_connect('draw_event', self.onresize)

    def on_select1(self, xmin, xmax):
        self.xmin = xmin
        self.width = xmax - xmin

    def onresize(self, event):
        self.update()

    def delete_region(self, key):
        if key in self.regions:
            if 'Rect' in self.regions[key]:
                self.regions[key]['Rect'].remove()
                self.regions[key]['Text'].remove()
            del (self.regions[key])

    def update(self):

        y_min, y_max = self.ax.get_ylim()
        for key in self.regions:
            if 'Rect' in self.regions[key]:
                self.regions[key]['Rect'].remove()
                self.regions[key]['Text'].remove()

            xmin = self.regions[key]['xmin']
            width = self.regions[key]['width']
            height = y_max - y_min
            alpha = self.regions[key]['alpha']
            color = self.regions[key]['color']
            self.regions[key]['Rect'] = patches.Rectangle((xmin, y_min), width, height,
                                                          edgecolor=color, alpha=alpha, facecolor=color)
            self.ax.add_patch(self.regions[key]['Rect'])

            self.regions[key]['Text'] = self.ax.text(xmin, y_max, self.regions[key]['text'], verticalalignment='top')

    def click(self, event):
        if str(event.key) in ['1', '2', '3', '4', '5', '6']:
            key = str(event.key)
            text = 'exclude \nedge ' + key
            alpha = 0.5
            color = 'red'
        elif str(event.key) in ['a', 'A', 'B', 'b', 'f', 'F']:
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
        key = ''
        if 'fit' in str(region):
            key = '0'
        if region in ['0', '1', '2', '3', '4', '5', '6']:
            key = region
        if region in [0, 1, 2, 3, 4, 5, 6]:
            key = str(region)

        if key not in self.regions:
            self.regions[key] = {}
            if key in ['1', '2', '3', '4', '5', '6']:
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
        for key in self.regions:
            if key == '0':
                area = 'fit_area'
            else:
                area = key
            tags[area] = {}
            tags[area]['start_x'] = self.regions[key]['xmin']
            tags[area]['width_x'] = self.regions[key]['width']

        return tags

    def disconnect(self):
        for key in self.regions:
            if 'Rect' in self.regions[key]:
                self.regions[key]['Rect'].remove()
                self.regions[key]['Text'].remove()
        del self.span
        self.ax.figure.canvas.mpl_disconnect(self.cid)
        # self.ax.figure.canvas.mpl_disconnect(self.draw)
        pass


class RangeSelector(RectangleSelector):
    """Select ranges of edge fitting interactively"""
    def __init__(self, ax, on_select):
        drawtype = 'box'
        spancoords = 'data'
        rectprops = dict(facecolor="blue", edgecolor="black", alpha=0.2, fill=True)

        super().__init__(ax, on_select, drawtype=drawtype,
                         minspanx=0, minspany=0, useblit=False,
                         lineprops=None, rectprops=rectprops, spancoords=spancoords,
                         button=None, maxdist=10, marker_props=None,
                         interactive=True, state_modifier_keys=None)

        self.artists = [self.to_draw, self._center_handle.artist,
                        self._edge_handles.artist]

    def draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        # ymin, ymax = sorted([y0, y1])
        xlim = sorted(self.ax.get_xlim())
        ylim = sorted(self.ax.get_ylim())

        xmin = max(xlim[0], xmin)
        ymin = ylim[0]
        xmax = min(xmax, xlim[1])
        ymax = ylim[1]

        self.to_draw.set_x(xmin)
        self.to_draw.set_y(ymin)
        self.to_draw.set_width(xmax - xmin)
        self.to_draw.set_height(ymax - ymin)


def get_likely_edges(energy_scale):
    """get likely ionization edges within energy_scale"""
    x_sections = eels.get_x_sections()
    # print(energy_scale)
    energy_origin = energy_scale[0]
    energy_window = energy_scale[-1] - energy_origin
    selected_edges_unsorted = {}
    likely_edges = []
    selected_elements = []
    for element in range(1, 83):
        # print(element)
        element_z = str(eels.get_z(element))

        for key in x_sections[element_z]:
            if key in all_edges:
                onset = x_sections[element_z][key]['onset']
                if onset > energy_origin:
                    if onset - energy_origin < energy_window:
                        if element not in selected_edges_unsorted:
                            selected_edges_unsorted[element] = {}
                        # print(element, x_sections[element]['name'], key, x_sections[element][key]['onset'])
                        # text = f"\n {x_sections[element_z]['name']:2s}-{key}: " \
                        #       f"{x_sections[element_z][key]['onset']:8.1f} eV "
                        # print(text)

                        selected_edges_unsorted[element][key] = {}
                        selected_edges_unsorted[element][key]['onset'] = x_sections[element_z][key]['onset']

                        if key in major_edges:
                            selected_edges_unsorted[element][key]['intensity'] = 'major'
                            selected_elements.append(x_sections[element_z]['name'])
                        else:
                            selected_edges_unsorted[element][key]['intensity'] = 'minor'

        if element in selected_edges_unsorted:
            for key in selected_edges_unsorted[element]:
                if selected_edges_unsorted[element][key]['intensity'] == 'major':
                    likely_edges.append(x_sections[str(element)]['name'])  # = {'z':element, 'symmetry': key}

    return likely_edges


def get_periodic_table_info():
    """Info for periodic table dialog"""
    pt_info = \
        {'H': {'PT_row': 0, 'PT_col': 0, 'Z': 0},
         'He': {'PT_row': 0, 'PT_col': 17, 'Z': 2}, 'Li': {'PT_row': 1, 'PT_col': 0, 'Z': 3},
         'Be': {'PT_row': 1, 'PT_col': 1, 'Z': 4}, 'B': {'PT_row': 1, 'PT_col': 12, 'Z': 5},
         'C': {'PT_row': 1, 'PT_col': 13, 'Z': 6}, 'N': {'PT_row': 1, 'PT_col': 14, 'Z': 7},
         'O': {'PT_row': 1, 'PT_col': 15, 'Z': 8}, 'F': {'PT_row': 1, 'PT_col': 16, 'Z': 9},
         'Ne': {'PT_row': 1, 'PT_col': 17, 'Z': 10}, 'Na': {'PT_row': 2, 'PT_col': 0, 'Z': 11},
         'Mg': {'PT_row': 2, 'PT_col': 1, 'Z': 12}, 'Al': {'PT_row': 2, 'PT_col': 12, 'Z': 13},
         'Si': {'PT_row': 2, 'PT_col': 13, 'Z': 14}, 'P': {'PT_row': 2, 'PT_col': 14, 'Z': 15},
         'S': {'PT_row': 2, 'PT_col': 15, 'Z': 16}, 'Cl': {'PT_row': 2, 'PT_col': 16, 'Z': 17},
         'Ar': {'PT_row': 2, 'PT_col': 17, 'Z': 18}, 'K': {'PT_row': 3, 'PT_col': 0, 'Z': 19},
         'Ca': {'PT_row': 3, 'PT_col': 1, 'Z': 20}, 'Sc': {'PT_row': 3, 'PT_col': 2, 'Z': 21},
         'Ti': {'PT_row': 3, 'PT_col': 3, 'Z': 22}, 'V ': {'PT_row': 3, 'PT_col': 4, 'Z': 23},
         'Cr': {'PT_row': 3, 'PT_col': 5, 'Z': 24}, 'Mn': {'PT_row': 3, 'PT_col': 6, 'Z': 25},
         'Fe': {'PT_row': 3, 'PT_col': 7, 'Z': 26}, 'Co': {'PT_row': 3, 'PT_col': 8, 'Z': 27},
         'Ni': {'PT_row': 3, 'PT_col': 9, 'Z': 28}, 'Cu': {'PT_row': 3, 'PT_col': 10, 'Z': 29},
         'Zn': {'PT_row': 3, 'PT_col': 11, 'Z': 30}, 'Ga': {'PT_row': 3, 'PT_col': 12, 'Z': 31},
         'Ge': {'PT_row': 3, 'PT_col': 13, 'Z': 32}, 'As': {'PT_row': 3, 'PT_col': 14, 'Z': 33},
         'Se': {'PT_row': 3, 'PT_col': 15, 'Z': 34}, 'Br': {'PT_row': 3, 'PT_col': 16, 'Z': 35},
         'Kr': {'PT_row': 3, 'PT_col': 17, 'Z': 36}, 'Rb': {'PT_row': 4, 'PT_col': 0, 'Z': 37},
         'Sr': {'PT_row': 4, 'PT_col': 1, 'Z': 38}, 'Y': {'PT_row': 4, 'PT_col': 2, 'Z': 39},
         'Zr': {'PT_row': 4, 'PT_col': 3, 'Z': 40}, 'Nb': {'PT_row': 4, 'PT_col': 4, 'Z': 41},
         'Mo': {'PT_row': 4, 'PT_col': 5, 'Z': 42}, 'Tc': {'PT_row': 4, 'PT_col': 6, 'Z': 43},
         'Ru': {'PT_row': 4, 'PT_col': 7, 'Z': 44}, 'Rh': {'PT_row': 4, 'PT_col': 8, 'Z': 45},
         'Pd': {'PT_row': 4, 'PT_col': 9, 'Z': 46}, 'Ag': {'PT_row': 4, 'PT_col': 10, 'Z': 47},
         'Cd': {'PT_row': 4, 'PT_col': 11, 'Z': 48}, 'In': {'PT_row': 4, 'PT_col': 12, 'Z': 49},
         'Sn': {'PT_row': 4, 'PT_col': 13, 'Z': 50}, 'Sb': {'PT_row': 4, 'PT_col': 14, 'Z': 51},
         'Te': {'PT_row': 4, 'PT_col': 15, 'Z': 52}, 'I': {'PT_row': 4, 'PT_col': 16, 'Z': 53},
         'Xe': {'PT_row': 4, 'PT_col': 17, 'Z': 54}, 'Cs': {'PT_row': 5, 'PT_col': 0, 'Z': 55},
         'Ba': {'PT_row': 5, 'PT_col': 1, 'Z': 56}, 'Hf': {'PT_row': 5, 'PT_col': 3, 'Z': 72},
         'Ta': {'PT_row': 5, 'PT_col': 4, 'Z': 73}, 'W': {'PT_row': 5, 'PT_col': 5, 'Z': 74},
         'Re': {'PT_row': 5, 'PT_col': 6, 'Z': 75}, 'Os': {'PT_row': 5, 'PT_col': 7, 'Z': 76},
         'Ir': {'PT_row': 5, 'PT_col': 8, 'Z': 77}, 'Pt': {'PT_row': 5, 'PT_col': 9, 'Z': 78},
         'Au': {'PT_row': 5, 'PT_col': 10, 'Z': 79}, 'Hg': {'PT_row': 5, 'PT_col': 11, 'Z': 80},
         'Pb': {'PT_row': 5, 'PT_col': 13, 'Z': 82}, 'Bi': {'PT_row': 5, 'PT_col': 14, 'Z': 0},
         'Po': {'PT_row': 5, 'PT_col': 15, 'Z': 0}, 'At': {'PT_row': 5, 'PT_col': 16, 'Z': 0},
         'Rn': {'PT_row': 5, 'PT_col': 17, 'Z': 0}, 'Fr': {'PT_row': 6, 'PT_col': 0, 'Z': 0},
         'Ra': {'PT_row': 6, 'PT_col': 1, 'Z': 0}, 'Rf': {'PT_row': 6, 'PT_col': 3, 'Z': 0},
         'Db': {'PT_row': 6, 'PT_col': 4, 'Z': 0}, 'Sg': {'PT_row': 6, 'PT_col': 5, 'Z': 0},
         'Bh': {'PT_row': 6, 'PT_col': 6, 'Z': 0}, 'Hs': {'PT_row': 6, 'PT_col': 7, 'Z': 0},
         'Mt': {'PT_row': 6, 'PT_col': 8, 'Z': 0}, 'Ds': {'PT_row': 6, 'PT_col': 9, 'Z': 0},
         'Rg': {'PT_row': 6, 'PT_col': 10, 'Z': 0}, 'La': {'PT_row': 8, 'PT_col': 3, 'Z': 57},
         'Ce': {'PT_row': 8, 'PT_col': 4, 'Z': 58}, 'Pr': {'PT_row': 8, 'PT_col': 5, 'Z': 59},
         'Nd': {'PT_row': 8, 'PT_col': 6, 'Z': 60}, 'Pm': {'PT_row': 8, 'PT_col': 7, 'Z': 61},
         'Sm': {'PT_row': 8, 'PT_col': 8, 'Z': 62}, 'Eu': {'PT_row': 8, 'PT_col': 9, 'Z': 63},
         'Gd': {'PT_row': 8, 'PT_col': 10, 'Z': 64}, 'Tb': {'PT_row': 8, 'PT_col': 11, 'Z': 65},
         'Dy': {'PT_row': 8, 'PT_col': 12, 'Z': 66}, 'Ho': {'PT_row': 8, 'PT_col': 13, 'Z': 67},
         'Er': {'PT_row': 8, 'PT_col': 14, 'Z': 68}, 'Tm': {'PT_row': 8, 'PT_col': 15, 'Z': 69},
         'Yb': {'PT_row': 8, 'PT_col': 16, 'Z': 70}, 'Lu': {'PT_row': 8, 'PT_col': 17, 'Z': 71},
         'Ac': {'PT_row': 9, 'PT_col': 3, 'Z': 0}, 'Th': {'PT_row': 9, 'PT_col': 4, 'Z': 0},
         'Pa': {'PT_row': 9, 'PT_col': 5, 'Z': 0}, 'U': {'PT_row': 9, 'PT_col': 6, 'Z': 0},
         'Np': {'PT_row': 9, 'PT_col': 7, 'Z': 0}, 'Pu': {'PT_row': 9, 'PT_col': 8, 'Z': 0},
         'Am': {'PT_row': 9, 'PT_col': 9, 'Z': 0}, 'Cm': {'PT_row': 9, 'PT_col': 10, 'Z': 0},
         'Bk': {'PT_row': 9, 'PT_col': 11, 'Z': 0}, 'Cf': {'PT_row': 9, 'PT_col': 12, 'Z': 0},
         'Es': {'PT_row': 9, 'PT_col': 13, 'Z': 0}, 'Fm': {'PT_row': 9, 'PT_col': 14, 'Z': 0},
         'Md': {'PT_row': 9, 'PT_col': 15, 'Z': 0}, 'No': {'PT_row': 9, 'PT_col': 16, 'Z': 0},
         'Lr': {'PT_row': 9, 'PT_col': 17, 'Z': 0},
         '*': {'PT_row': 5, 'PT_col': 2, 'PT_col2': 8, 'PT_row2': 2, 'Z': 0},
         '**': {'PT_row': 6, 'PT_col': 2, 'PT_col2': 9, 'PT_row2': 2, 'Z': 0}}

    return pt_info


class InteractiveSpectrumImage(object):
    """Interactive spectrum imaging plot

    Attributes:
    -----------
    dictionary with a minimum of the following keys:
        ['image']: displayed image
        ['data']:  data cube
        ['intensity_scale_ppm']: intensity scale
        ['ylabel']: intensity label
        ['spectra'] dictionary which contains dictionaries for each spectrum style ['1-2']:
            ['spectrum'] = tags['cube'][y,x,:]
            ['spectra'][f'{x}-{y}']['energy_scale'] = tags['energy_scale']
            ['intensity_scale'] = 1/tags['cube'][y,x,:].sum()*1e6

    Please note the possibility to load any image for the selection of the spectrum
    Also there is the possibility to display the survey image.

    For analysis, we have the following options:
        'fix_energy': set zero-loss peak maximum to zero !! Low loss spectra only!!
        'fit_zero_loss': fit zero-loss peak with model function !! Low loss spectra only!!
        'fit_low_loss': fit low-loss spectrum with model peaks !! Low loss spectra only!!


        'fit_composition': fit core-loss spectrum with background and cross-sections!! Core loss spectra only!!
        'fit_ELNES': fit core-loss edge with model peaks  !! Core loss spectra only!!
    """

    def __init__(self, data_source, horizontal=True):

        box_layout = widgets.Layout(display='flex',
                                    flex_flow='row',
                                    align_items='stretch',
                                    width='100%')

        words = ['fix_energy', 'fit_zero_loss', 'fit_low_loss', 'fit_composition', 'fit_ELNES']

        self.buttons = [widgets.ToggleButton(value=False, description=word, disabled=False) for word in words]
        box = widgets.Box(children=self.buttons, layout=box_layout)
        display(box)

        # MAKE Dictionary

        if isinstance(data_source, dict):
            self.tags = data_source
        elif isinstance(data_source, h5py.Group):
            self.tags = self.set_tags(data_source)
        else:
            print('Data source must be a dictionary or channel')
            return

        # Button(description='edge_quantification')
        for button in self.buttons:
            button.observe(self.on_button_clicked, 'value')  # on_click(self.on_button_clicked)

        self.figure = plt.figure()
        self.horizontal = horizontal
        self.x = 0
        self.y = 0

        self.extent = [0, self.tags['cube'].shape[1], self.tags['cube'].shape[0], 0]
        self.rectangle = [0, self.tags['cube'].shape[1], 0, self.tags['cube'].shape[0]]
        self.scaleX = 1.0
        self.scaleY = 1.0
        self.analysis = []
        self.plot_legend = False
        if 'ylabel' not in self.tags:
            self.tags['ylabel'] = 'intensity [a.u.]'
        self.SI = False

        if horizontal:
            self.ax1 = plt.subplot(1, 2, 1)
            self.ax2 = plt.subplot(1, 2, 2)
        else:
            self.ax1 = plt.subplot(2, 1, 1)
            self.ax2 = plt.subplot(2, 1, 2)

        self.cube = self.tags['cube']
        self.image = self.tags['cube'].sum(axis=2)

        self.ax1.imshow(self.image, extent=self.extent)
        if horizontal:
            self.ax1.set_xlabel('distance [pixels]')
        else:
            self.ax1.set_ylabel('distance [pixels]')
        self.ax1.set_aspect('equal')

        self.rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='red', alpha=0.2)
        self.ax1.add_patch(self.rect)
        self.intensity_scale = self.tags['spectra'][f'{self.x}-{self.y}']['intensity_scale']
        self.spectrum = self.tags['spectra'][f'{self.x}-{self.y}']['spectrum'] * self.intensity_scale
        self.energy_scale = self.tags['spectra'][f'{self.x}-{self.y}']['energy_scale']

        self.ax2.plot(self.energy_scale, self.spectrum)
        self.ax2.set_title(f' spectrum {self.x},{self.y} ')
        self.ax2.set_xlabel('energy loss [eV]')
        self.ax2.set_ylabel(self.tags['ylabel'])
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)

        plt.tight_layout()

    def on_button_clicked(self, b):
        # print(b['owner'].description)
        selection = b['owner'].description
        if b['new']:
            if selection == 'fit_composition':
                if 'region_tags' in self.tags and 'edges_present' in self.tags \
                        and 'acceleration_voltage' in self.tags \
                        and 'collection_angle' in self.tags:
                    pass
                else:
                    self.buttons[3].value = False
                    return
            elif selection in ['fix_energy', 'fit_zero_loss']:
                if self.energy_scale[0] > 0:
                    button_index = ['fix_energy', 'fit_zero_loss'].index(selection)
                    self.buttons[button_index].value = False
                    return
            self.analysis.append(selection)
            self.update()
        else:

            if selection in self.analysis:
                self.analysis.remove(selection)

    def do_all(self, selection=None, verbose=True):
        x = self.x
        y = self.y
        if selection is None:
            selection = self.analysis
        for self.x in range(self.cube.shape[1]):
            if verbose:
                print(f' row: {self.x}')
            for self.y in range(self.cube.shape[0]):

                if 'fit_zero_loss' in selection:
                    title = self.fit_zero_loss(plot_this=False)

                elif 'fix_energy' in selection:
                    self.ax2.set_title('bn')
                    title = self.fix_energy()

                elif 'fit_composition' in selection:
                    title = self.fit_quantification(plot_this=False)

        self.x = x
        self.y = y

    def onclick(self, event):
        x = int(event.xdata)
        y = int(event.ydata)

        # print(x,y)
        if self.rectangle[0] <= x < self.rectangle[0] + self.rectangle[1]:
            if self.rectangle[2] <= y < self.rectangle[2] + self.rectangle[3]:
                self.x = int((x - self.rectangle[0]) / self.rectangle[1] * self.cube.shape[1])
                self.y = int((y - self.rectangle[2]) / self.rectangle[3] * self.cube.shape[0])
            else:
                return
        else:
            return

        if event.inaxes in [self.ax1]:
            x = (self.x * self.rectangle[1] / self.cube.shape[1] + self.rectangle[0])
            y = (self.y * self.rectangle[3] / self.cube.shape[0] + self.rectangle[2])

            self.rect.set_xy([x, y])
            self.update()

    def update(self):
        xlim = self.ax2.get_xlim()
        ylim = self.ax2.get_ylim()
        self.ax2.clear()
        self.intensity_scale = self.tags['spectra'][f'{self.x}-{self.y}']['intensity_scale']
        self.spectrum = self.tags['spectra'][f'{self.x}-{self.y}']['spectrum'] * self.intensity_scale
        self.energy_scale = self.tags['spectra'][f'{self.x}-{self.y}']['energy_scale']

        if 'fit_zero_loss' in self.analysis:
            title = self.fit_zero_loss()
            self.ax2.set_title(title)
        elif 'fix_energy' in self.analysis:
            self.ax2.set_title('bn')
            title = self.fix_energy()
            self.ax2.set_title(title)

        elif 'fit_composition' in self.analysis:
            title = self.fit_quantification()
            self.ax2.set_title(title)

        else:
            self.ax2.set_title(f' spectrum {self.x},{self.y} ')
        self.ax2.plot(self.energy_scale, self.spectrum, color='#1f77b4', label='experiment')

        if self.plot_legend:
            self.ax2.legend(shadow=True)
        self.ax2.set_xlim(xlim)
        self.ax2.set_ylim(ylim)
        self.ax2.set_xlabel('energy loss [eV]')
        self.ax2.set_ylabel(self.tags['ylabel'])
        self.ax2.set_xlim(xlim)

        # self.ax2.draw()

    def set_tags(self, channel):
        # TODO: change to sidpy dataset tags = ft.h5_get_dictionary(channel)
        tags = {}
        if tags['data_type'] == 'spectrum_image':
            tags['image'] = tags['data']
            tags['data'] = tags['cube'][0, 0, :]
            if 'intensity_scale_ppm' not in channel:
                channel['intensity_scale_ppm'] = 1

            tags['ylabel'] = 'intensity [a.u.]'
            tags['spectra'] = {}
            for x in range(tags['spatial_size_y']):
                for y in range(tags['spatial_size_x']):
                    tags['spectra'][f'{x}-{y}'] = {}
                    tags['spectra'][f'{x}-{y}']['spectrum'] = tags['cube'][y, x, :]
                    tags['spectra'][f'{x}-{y}']['energy_scale'] = tags['energy_scale']
                    tags['spectra'][f'{x}-{y}']['intensity_scale'] = 1 / tags['cube'][y, x, :].sum() * 1e6
            tags['ylabel'] = 'inel. scat. int.  [ppm]'

        return tags

    def fix_energy(self):

        energy_scale = self.tags['spectra'][f'{self.x}-{self.y}']['energy_scale']
        spectrum = self.tags['spectra'][f'{self.x}-{self.y}']['spectrum'] * self.intensity_scale
        fwhm, delta_e = eels.fix_energy_scale(spectrum, energy_scale)
        self.tags['spectra'][f'{self.x}-{self.y}']['delta_e'] = delta_e
        self.tags['spectra'][f'{self.x}-{self.y}']['fwhm'] = fwhm
        self.energy_scale = energy_scale - delta_e
        title = f'spectrum {self.x},{self.y} fwhm: {fwhm:.2f}, dE: {delta_e:.3f}'
        return title

    def fit_zero_loss(self, plot_this=True):

        energy_scale = self.tags['spectra'][f'{self.x}-{self.y}']['energy_scale']
        spectrum = self.tags['spectra'][f'{self.x}-{self.y}']['spectrum'] * self.intensity_scale
        if 'zero_loss_fit_width' not in self.tags:
            self.tags['zero_loss_fit_width'] = .5
        if self.tags['zero_loss_fit_width'] / (energy_scale[1] - energy_scale[0]) < 6:
            self.tags['zero_loss_fit_width'] = (energy_scale[1] - energy_scale[0]) * 6
        fwhm, delta_e = eels.fix_energy_scale(spectrum, energy_scale)
        energy_scale = energy_scale - delta_e
        z_oss, p_zl = eels.resolution_function(energy_scale, spectrum, self.tags['zero_loss_fit_width'])
        fwhm2, delta_e2 = eels.fix_energy_scale(z_oss, energy_scale)

        self.tags['spectra'][f'{self.x}-{self.y}']['resolution_function'] = z_oss
        self.tags['spectra'][f'{self.x}-{self.y}']['p_zl'] = p_zl
        self.tags['spectra'][f'{self.x}-{self.y}']['delta_e'] = delta_e
        self.tags['spectra'][f'{self.x}-{self.y}']['fwhm_resolution'] = fwhm2
        self.tags['spectra'][f'{self.x}-{self.y}']['fwhm'] = fwhm

        if plot_this:
            self.ax2.plot(energy_scale, z_oss, label='resolution function', color='black')
            self.ax2.plot(energy_scale, self.spectrum - z_oss, label='difference', color='orange')
            self.ax2.axhline(linewidth=0.5, color='black')
        self.energy_scale = energy_scale
        title = f'spectrum {self.x},{self.y} fwhm: {fwhm:.2f}'  # ', dE: {delta_e2:.5e}'
        return title

    def fit_quantification(self, plot_this=True):
        energy_scale = self.tags['spectra'][f'{self.x}-{self.y}']['energy_scale']
        spectrum = self.tags['spectra'][f'{self.x}-{self.y}']['spectrum'] * self.intensity_scale
        edges = eels.make_edges(self.tags['edges_present'], energy_scale, self.tags['acceleration_voltage'],
                                self.tags['collection_angle'])
        edges = eels.fit_edges(spectrum, self.tags['spectra'][f'{self.x}-{self.y}']['energy_scale'],
                               self.tags['region_tags'], edges)
        self.tags['spectra'][f'{self.x}-{self.y}']['edges'] = edges.copy()
        if plot_this:
            self.ax2.plot(energy_scale, edges['model']['spectrum'], label='model')
            self.ax2.plot(energy_scale, self.spectrum - edges['model']['spectrum'], label='difference')
            self.ax2.axhline(linewidth=0.5, color='black')
        else:
            self.tags['spectra'][f'{self.x}-{self.y}']['do_all'] = 'done'
        title = f'spectrum {self.x},{self.y} '

        for key in edges:
            if key.isdigit():
                title = title + f"{edges[key]['element']}:  {edges[key]['areal_density']:.2e};  "

        return title

    def set_legend(self, set_legend):
        self.plot_legend = set_legend

    def get_xy(self):
        return [self.x, self.y]

    def get_current_spectrum(self):
        return self.cube[self.y, self.x, :]

    def set_z_contrast_image(self, z_channel=None):
        if z_channel is not None:
            self.tags['Z_contrast_channel'] = z_channel
        if 'Z_contrast_channel' not in self.tags:
            print('add Z contrast channel group to dictionary first!')
            return

        z_tags = {}   # TODO change to sidpy dataset ft.h5_get_dictionary(z_channel)
        extent = [self.rectangle[0], self.rectangle[0] + self.rectangle[1],
                  self.rectangle[2] + self.rectangle[3], self.rectangle[2]]
        self.ax1.imshow(z_tags['data'], extent=extent, cmap='gray')

    def overlay_z_contrast_image(self, z_channel=None):

        if self.SI:
            if z_channel is not None:
                self.tags['Z_contrast_channel'] = z_channel
            if 'Z_contrast_channel' not in self.tags:
                print('add survey channel group to dictionary first!')
                return

            z_tags = {}  # TODO: change to sidpy ft.h5_get_dictionary(self.tags['Z_contrast_channel'])

            xlim = self.ax1.get_xlim()
            ylim = self.ax1.get_ylim()
            extent = [self.rectangle[0], self.rectangle[0] + self.rectangle[1],
                      self.rectangle[2] + self.rectangle[3], self.rectangle[2]]
            self.ax1.imshow(z_tags['data'], extent=extent, cmap='viridis', alpha=0.5)
            self.ax1.set_ylim(ylim)
            self.ax1.set_xlim(xlim)

    def overlay_data(self, data=None):

        if self.SI:
            if data is None:
                data = self.cube.sum(axis=2)

            xlim = self.ax1.get_xlim()
            ylim = self.ax1.get_ylim()
            extent = [self.rectangle[0], self.rectangle[0] + self.rectangle[1],
                      self.rectangle[2] + self.rectangle[3], self.rectangle[2]]
            self.ax1.imshow(data, extent=extent, alpha=0.7, cmap='viridis')
            self.ax1.set_ylim(ylim)
            self.ax1.set_xlim(xlim)

    def set_survey_image(self, si_channel=None):

        if si_channel is not None:
            self.tags['survey_channel'] = si_channel
        if 'survey_channel' not in self.tags:
            print('add survey channel group to dictionary first!')
            return
        si_channel = self.tags['survey_channel']
        si_tags = {}  # TODO: change to sidpy  ft.h5_get_dictionary(si_channel)
        tags2 = dict(si_channel.attrs)

        self.ax1.set_aspect('equal')
        self.scaleX = si_channel['spatial_scale_x'][()]
        self.scaleY = si_channel['spatial_scale_y'][()]

        self.ax1.imshow(si_tags['data'], extent=si_tags['extent'], cmap='gray')
        if self.horizontal:
            self.ax1.set_xlabel('distance [nm]')
        else:
            self.ax1.set_ylabel('distance [nm]')

        annotation_done = []
        for key in tags2:
            if 'annotations' in key:
                annotation_number = key[12]
                if annotation_number not in annotation_done:
                    annotation_done.append(annotation_number)

                    if tags2['annotations_' + annotation_number + '_type'] == 'text':
                        x = tags2['annotations_' + annotation_number + '_x']
                        y = tags2['annotations_' + annotation_number + '_y']
                        text = tags2['annotations_' + annotation_number + '_text']
                        self.ax1.text(x, y, text, color='r')

                    elif tags2['annotations_' + annotation_number + '_type'] == 'circle':
                        radius = 20 * self.scaleX  # tags['annotations'][key]['radius']
                        xy = tags2['annotations_' + annotation_number + '_position']
                        circle = patches.Circle(xy, radius, color='r', fill=False)
                        self.ax1.add_artist(circle)

                    elif tags2['annotations_' + annotation_number + '_type'] == 'spectrum image':
                        width = tags2['annotations_' + annotation_number + '_width']
                        height = tags2['annotations_' + annotation_number + '_height']
                        position = tags2['annotations_' + annotation_number + '_position']
                        rectangle = patches.Rectangle(position, width, height, color='r', fill=False)
                        self.rectangle = [position[0], width, position[1], height]
                        self.ax1.add_artist(rectangle)
                        self.ax1.text(position[0], position[1], 'Spectrum Image', color='r')
                        self.rect.set_width(width / self.cube.shape[1])
                        self.rect.set_height(height / self.cube.shape[0])
        self.SI = True


class ElementalEdges(object):
    """ Adds ionization edges of element z to plot with axis ax

        There is an optional parameter maximum_chemical_shift which allows to change
        the energy range in which the edges are searched.

        available functions:
        - update(): updates the drawing of ionization edges
        - set_edge(Z) : changes atomic number and updates everything accordingly
        - disconnect: makes everything invisible and stops drawing
        - reconnect: undo of disconnect

        usage:
        >> fig, ax = plt.subplots()
        >> ax.plot(energy_scale, spectrum)
        >> Z= 42
        >> cursor = ElementalEdges(ax, Z)


        see Chapter4 'CH4-Working_with_X-Sections' notebook
    """

    def __init__(self, ax, z):
        self.ax = ax
        self.labels = None
        self.lines = None
        self.Z = eels.get_z(z)
        self.color = 'black'
        self.x_sections = eels.get_x_sections()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self.onresize)
        # self.update() is not necessary because of a drawing event is issued

    def set_edge(self, z):
        self.Z = eels.get_z(z)
        if self.cid is None:
            self.cid = self.ax.figure.canvas.mpl_connect('draw_event', self.onresize)
        self.update()

    def onresize(self, event):
        self.update()

    def update(self):
        if self.labels is not None:
            for label in self.labels:
                label.remove()
        if self.lines is not None:
            for line in self.lines:
                line.remove()
        self.labels = []
        self.lines = []
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        element = str(self.Z)
        x_sections = self.x_sections
        for key in all_edges:
            if key in x_sections[element] and 'onset' in x_sections[element][key]:
                x = x_sections[element][key]['onset']
                if x_min < x < x_max:
                    if key in first_close_edges:
                        label2 = self.ax.text(x, y_max, f"{x_sections[element]['name']}-{key}",
                                              verticalalignment='top', rotation=0, color=self.color)
                    else:
                        label2 = self.ax.text(x, y_max, f"\n{x_sections[element]['name']}-{key}",
                                              verticalalignment='top', color=self.color)
                    line2 = self.ax.axvline(x, ymin=0, ymax=1, color=self.color)

                    self.labels.append(label2)
                    self.lines.append(line2)

    def reconnect(self):
        self.cid = self.ax.figure.canvas.mpl_connect('draw_event', self.onresize)
        self.update()

    def disconnect(self):
        if self.labels is not None:
            for label in self.labels:
                label.remove()
        if self.lines is not None:
            for line in self.lines:
                line.remove()
        self.labels = None
        self.lines = None
        self.ax.figure.canvas.mpl_disconnect(self.cid)


class EdgesAtCursor(object):
    """
        Adds a Cursor to a plot, which plots all major (possible) ionization edges at
        the cursor location if left (right) mouse button is clicked.

        Attributes
        ----------
        ax: matplotlib axis
        x: numpy array
            energy_scale of spectrum
        y: numpy array
            intensities of spectrum
        maximal_chemical_shift: float
            optional parameter maximum_chemical_shift which allows to change the energy range in which the edges
            are searched.

        Example
        -------
        fig, ax = plt.subplots()
        ax.plot(energy_scale, spectrum)
        cursor = EdgesAtCursor(ax, energy_scale, spectrum)

        see Chapter4 'CH4-Working_with_X-Sections' notebook

    """

    def __init__(self, ax, x, y, maximal_chemical_shift=5):
        self.ax = ax
        self.ly = ax.axvline(x[0], color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot(x[0], y[0], marker="o", color="crimson", zorder=3)
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '', verticalalignment='bottom')
        self.select = 0
        self.label = None
        self.line = None
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.click)
        self.mouse_cid = ax.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.maximal_chemical_shift = maximal_chemical_shift

    def click(self, event):

        # print('click', event)
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata

        index = np.searchsorted(self.x, [x])[0]
        x = self.x[index]
        y = self.y[index]
        self.select = x

        y_min, y_max = self.ax.get_ylim()

        if self.label is not None:
            self.label.remove()
            self.line.remove()
        if event.button == 1:
            self.label = self.ax.text(x, y_max, eels.find_major_edges(event.xdata, self.maximal_chemical_shift),
                                      verticalalignment='top')
            self.line, = self.ax.plot([x, x], [y_min, y_max], color='black')
        if event.button == 3:
            self.line, = self.ax.plot([x, x], [y_min, y_max], color='black')
            self.label = self.ax.text(x, y_max, eels.find_all_edges(event.xdata, self.maximal_chemical_shift),
                                      verticalalignment='top')
        self.ax.set_ylim(y_min, y_max)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        index = np.searchsorted(self.x, [x])[0]
        x = self.x[index]
        y = self.y[index]
        self.select = x
        self.ly.set_xdata(x)
        self.marker.set_data([x], [y])
        self.txt.set_text(f'\n x={x:1.2f}, y={y:1.2g}\n')

        # self.ax.text(x, y*2,find_major_edges(x))
        self.txt.set_position((x, y))
        self.ax.figure.canvas.draw_idle()

    def del_edges(self):
        if self.label is not None:
            self.label.remove()
            self.line.remove()
            self.label = None

    def disconnect(self):
        self.ly.remove()
        self.marker.remove()
        self.txt.remove()

        self.ax.figure.canvas.mpl_disconnect(self.cid)
        self.ax.figure.canvas.mpl_disconnect(self.mouse_cid)


def make_box_layout():
    return widgets.Layout(border='solid 1px black', margin='0px 10px 10px 0px', padding='5px 5px 5px 5px')
    

class plot_EELS(widgets.HBox):
    def __init__(self, dataset):
        super().__init__()
        output = widgets.Output()
        self.dataset = dataset
        self.spec_dim = 0
        initial_color = '#FF00DD'
 
        with output:
            self.fig, self.axis = plt.subplots(constrained_layout=True, figsize=(5, 3.5))
        
        self.axis.set_title(dataset.title.split('/')[-1])
        self.line, = self.axis.plot(dataset.dim_0.values, dataset, lw=2, label='spectrum')
        legend = self.axis.legend(fancybox=True, shadow=True)

        lines = [self.line]
        self.line_dictionary = {}  # Will map legend lines to original lines.
        for legend_line, original_line in zip(legend.get_lines(), lines):
            legend_line.set_picker(True)  # Enable picking on the legend line.
            self.line_dictionary[legend_line] = original_line
        self.ax = self.axis
        self.fig.canvas.toolbar_position = 'bottom'
        self.fig.canvas.mpl_connect('pick_event', self.on_legend_pick)
        
        # define widgets
        int_slider = widgets.IntSlider(
            value=1, 
            min=0, 
            max=10, 
            step=1, 
            description='freq'
        )
        self.offset = widgets.Text(
            value='0', 
            width=5,
            description='offset', 
            continuous_update=False
        )
        self.dispersion = widgets.Text(
            value='0', 
            width=5,
            description='dispersion', 
            continuous_update=False
        )

        self.exposure = widgets.Text(
            value='0', 
            width=5,
            description='exposure', 
            continuous_update=False
        )
        
        button_energy_scale = widgets.Button(description='Cursor')
        button_elements_at_cursor = widgets.Button(description='Elements Cursor')
        button_main_elements = widgets.Button(description='Main Elements')
        
        controls = widgets.VBox([
            widgets.HBox([self.offset, widgets.Label('eV')]),
            widgets.HBox([self.dispersion, widgets.Label('eV/channel')]),
            widgets.HBox([self.exposure, widgets.Label('s')]),
            button_energy_scale,
            widgets.HBox([button_elements_at_cursor, button_main_elements])
        ])
            
        controls.layout = make_box_layout()
         
        out_box = widgets.Box([output])
        output.layout = make_box_layout()
 
        # observe stuff
        int_slider.observe(self.update, 'value')
        
        self.offset.value = f'{self.dataset.dim_0.values[0]}'
        self.offset.observe(self.set_dimension, 'value')
        self.offset.value = f'{self.dataset.dim_0.values[0]}'
        
        self.dispersion.observe(self.set_dimension, 'value')
        self.dispersion.value = f'{self.dataset.dim_0.values[1] - self.dataset.dim_0.values[0]}'
        self.dispersion.value = '0'
        self.exposure.observe(self.update_exposure, 'value')
        self.exposure.value = '0'
 
        # add to children
        self.children = [controls, output]
        
    def update(self):
        """Draw line in plot"""
        self.line.set_ydata(self.dataset)
        self.line.set_xdata(self.dataset.dim_0.values)
        # self.axis.plot(self.dataset.energy_loss, self.dataset)
        self.fig.canvas.draw()
        
    def line_color(self, change):
        self.line.set_color(change.new)
        
    def update_exposure(self):
        pass
  
    def update_ylabel(self, change):
        self.ax.set_ylabel(change.new)
    
    def set_dimension(self, change):
        self.spec_dim = ft.get_dimensions_by_type('SPECTRAL', self.dataset)
        self.spec_dim = self.spec_dim[0]
        old_energy_scale = self.spec_dim[1]
        energy_scale = np.arange(len(self.dataset.dim_0.values))*float(self.dispersion.value)+float(self.offset.value)
        self.dataset.set_dimension(self.spec_dim[0], sidpy.Dimension(energy_scale,
                                                                     name=old_energy_scale.name,
                                                                     dimension_type='SPECTRAL',
                                                                     units='eV',
                                                                     quantity='energy loss'))
        self.update()

    def on_legend_pick(self, event):
        legend_line = event.artist
        original_line = self.line_dictionary[legend_line]
        visible = not original_line.get_visible()
        original_line.set_visible(visible)
        legend_line.set_alpha(1.0 if visible else 0.2)
        self.fig.canvas.draw()
