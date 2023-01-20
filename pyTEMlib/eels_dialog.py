"""
QT dialog window for EELS compositional analysis

Author: Gerd Duscher
"""
Qt_available = True
try:
    from PyQt5 import QtCore, QtWidgets
except:
    Qt_available = False
    print('Qt dialogs are not available')
from pyTEMlib import eels_dlg

import numpy as np

from pyTEMlib import eels_tools as eels

import matplotlib.pylab as plt
import matplotlib.patches as patches

from pyTEMlib import file_tools as ft

import sidpy

_version = 000

if Qt_available:
    from pyTEMlib import eels_dialog_utilities

    class EELSDialog(QtWidgets.QDialog):
        """
        EELS Input Dialog for Chemical Analysis
        """

        def __init__(self, dataset=None):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            # Create an instance of the GUI
            if dataset is None:
                # make a dummy dataset
                dataset = ft.make_dummy_dataset(sidpy.DataType.SPECTRUM)
            if not isinstance(dataset, sidpy.Dataset):
                raise TypeError('dataset has to be a sidpy dataset')
            self.spec_dim = ft.get_dimensions_by_type('spectral', dataset)
            if len(self.spec_dim) != 1:
                raise TypeError('We need exactly one SPECTRAL dimension')
            self.spec_dim = self.spec_dim[0]

            self.ui = eels_dlg.UiDialog(self)
            # Run the .setup_ui() method to show the GUI
            # self.ui.setup_ui(self)

            self.set_action()

            self.dataset = dataset
            self.energy_scale = np.array([])
            self.model = np.array([])
            self.y_scale = 1.0
            self.change_y_scale = 1.0
            
            self.edges = {}

            self.show_regions = False
            self.show()

            self.set_dataset(dataset)
            # TODO: set elements does not work correctly for periodic table
            # selected_edges = eels.find_edges(dataset, sensitivity=3)
            selected_edges = []
            initial_elements = []

            for edge in selected_edges:
                initial_elements.append(edge.split('-')[0])
            if len(initial_elements) > 0:
                self.set_elements(initial_elements)

            self.pt_dialog = eels_dialog_utilities.PeriodicTableDialog(energy_scale=self.energy_scale,
                                                                       initial_elements=initial_elements)
            self.pt_dialog.signal_selected[list].connect(self.set_elements)

            self.dataset.plot()

            if hasattr(self.dataset.view, 'axes'):
                self.axis = self.dataset.view.axes[-1]
            elif hasattr(self.dataset.view, 'axis'):
                self.axis = self.dataset.view.axis

            self.figure = self.axis.figure
            self.updY = 0
            self.figure.canvas.mpl_connect('button_press_event', self.plot)

            self.plot()
            self.ui.edit4.setFocus()

        def set_dataset(self, dataset):

            self.dataset = dataset
            if 'edges' not in self.dataset.metadata or self.dataset.metadata['edges'] == {}:
                self.dataset.metadata['edges'] = {'0': {}, 'model': {}, 'use_low_loss': False}
            self.edges = self.dataset.metadata['edges']

            spec_dim = ft.get_dimensions_by_type('spectral', dataset)[0]

            if len(spec_dim) == 0:
                raise TypeError('We need at least one SPECTRAL dimension')

            self.spec_dim = spec_dim[0]
            self.energy_scale = dataset._axes[self.spec_dim].values
            self.ui.edit2.setText(f"{self.energy_scale[-2]:.3f}")

            if 'fit_area' not in self.edges:
                self.edges['fit_area'] = {}
            if 'fit_start' not in self.edges['fit_area']:
                self.ui.edit1.setText(f"{self.energy_scale[50]:.3f}")
                self.edges['fit_area']['fit_start'] = float(self.ui.edit1.displayText())
            else:
                self.ui.edit1.setText(f"{self.edges['fit_area']['fit_start']:.3f}")
            if 'fit_end' not in self.edges['fit_area']:
                self.ui.edit2.setText(f"{self.energy_scale[-2]:.3f}")
                self.edges['fit_area']['fit_end'] = float(self.ui.edit2.displayText())
            else:
                self.ui.edit2.setText(f"{self.edges['fit_area']['fit_end']:.3f}")

            if self.dataset.data_type.name == 'SPECTRAL_IMAGE':
                if 'SI_bin_x' not in self.dataset.metadata['experiment']:
                    self.dataset.metadata['experiment']['SI_bin_x'] = 1
                    self.dataset.metadata['experiment']['SI_bin_y'] = 1

                bin_x = self.dataset.metadata['experiment']['SI_bin_x']
                bin_y = self.dataset.metadata['experiment']['SI_bin_y']
                self.dataset.view.set_bin([bin_x, bin_y])
            self.update()

        def update(self):
            index = self.ui.list3.currentIndex()  # which edge
            edge = self.edges[str(index)]

            if 'z' in edge:
                self.ui.list5.setCurrentIndex(self.ui.edge_sym.index(edge['symmetry']))
                self.ui.edit4.setText(str(edge['z']))
                self.ui.unit4.setText(edge['element'])
                self.ui.edit6.setText(f"{edge['onset']:.2f}")
                self.ui.edit7.setText(f"{edge['start_exclude']:.2f}")
                self.ui.edit8.setText(f"{edge['end_exclude']:.2f}")
                if self.y_scale == 1.0:
                    self.ui.edit9.setText(f"{edge['areal_density']:.2e}")
                    self.ui.unit9.setText('a.u.')
                else:
                    dispersion = self.energy_scale[1]-self.energy_scale[0]
                    self.ui.edit9.setText(f"{edge['areal_density']*self.y_scale*1e-6/dispersion:.2f}")
                    self.ui.unit9.setText(r'atoms/nm$^2$')
            else:
                self.ui.list3.setCurrentIndex(0)
                self.ui.edit4.setText(str(0))
                self.ui.unit4.setText(' ')
                self.ui.edit6.setText(f"{0:.2f}")
                self.ui.edit7.setText(f"{0:.2f}")
                self.ui.edit8.setText(f"{0:.2f}")
                self.ui.edit9.setText(f"{0:.2e}")

        def update_element(self, z):
            # We check whether this element is already in the
            zz = eels.get_z(z)
            for key, edge in self.edges.items():
                if key.isdigit():
                    if 'z' in edge:
                        if zz == edge['z']:
                            return False

            major_edge = ''
            minor_edge = ''
            all_edges = {}
            x_section = eels.get_x_sections(zz)
            edge_start = 10  # int(15./ft.get_slope(self.energy_scale)+0.5)
            for key in x_section:
                if len(key) == 2 and key[0] in ['K', 'L', 'M', 'N', 'O'] and key[1].isdigit():
                    if self.energy_scale[edge_start] < x_section[key]['onset'] < self.energy_scale[-edge_start]:
                        if key in ['K1', 'L3', 'M5']:
                            major_edge = key
                        elif key in self.ui.edge_sym:
                            if minor_edge == '':
                                minor_edge = key
                            if int(key[-1]) % 2 > 0:
                                if int(minor_edge[-1]) % 2 == 0 or key[-1] > minor_edge[-1]:
                                    minor_edge = key

                        all_edges[key] = {'onset': x_section[key]['onset']}

            if major_edge != '':
                key = major_edge
            elif minor_edge != '':
                key = minor_edge
            else:
                print(f'Could not find no edge of {zz} in spectrum')
                return False

            index = self.ui.list3.currentIndex()

            if str(index) not in self.edges:
                self.edges[str(index)] = {}

            start_exclude = x_section[key]['onset'] - x_section[key]['excl before']
            end_exclude = x_section[key]['onset'] + x_section[key]['excl after']

            self.edges[str(index)] = {'z': zz, 'symmetry': key, 'element': eels.elements[zz],
                                      'onset': x_section[key]['onset'], 'end_exclude': end_exclude,
                                      'start_exclude': start_exclude}
            self.edges[str(index)]['all_edges'] = all_edges
            self.edges[str(index)]['chemical_shift'] = 0.0
            self.edges[str(index)]['areal_density'] = 0.0
            self.edges[str(index)]['original_onset'] = self.edges[str(index)]['onset']
            return True

        def on_enter(self):
            sender = self.sender()
            edge_list = self.ui.list3

            if sender.objectName() == 'fit_start_edit':
                value = float(str(sender.displayText()).strip())
                if value < self.energy_scale[0]:
                    value = self.energy_scale[0]
                if value > self.energy_scale[-5]:
                    value = self.energy_scale[-5]
                self.edges['fit_area']['fit_start'] = value
                sender.setText(str(self.edges['fit_area']['fit_start']))
            elif sender.objectName() == 'fit_end_edit':
                value = float(str(sender.displayText()).strip())
                if value < self.energy_scale[5]:
                    value = self.energy_scale[5]
                if value > self.energy_scale[-1]:
                    value = self.energy_scale[-1]
                self.edges['fit_area']['fit_end'] = value
                sender.setText(str(self.edges['fit_area']['fit_end']))
            elif sender.objectName() == 'element_edit':
                if str(sender.displayText()).strip() == '0':
                    sender.setText('PT')
                    self.pt_dialog.energy_scale = self.energy_scale
                    self.pt_dialog.show()
                else:
                    self.update_element(str(sender.displayText()).strip())
                self.update()
            elif sender.objectName() in ['onset_edit', 'excl_start_edit', 'excl_end_edit']:
                self.check_area_consistency()

            elif sender.objectName() == 'multiplier_edit':
                index = edge_list.currentIndex()
                self.edges[str(index)]['areal_density'] = float(self.ui.edit9.displayText())
                if 'background' not in self.edges['model']:
                    print(' no background')
                    return
                self.model = self.edges['model']['background']
                for key in self.edges:
                    if key.isdigit():
                        self.model = self.model + self.edges[key]['areal_density'] * self.edges[key]['data']
                self.plot()
            else:
                return
            if self.show_regions:
                self.plot()

        def sort_elements(self):
            onsets = []
            for index, edge in self.edges.items():
                if index.isdigit():
                    onsets.append(float(edge['onset']))

            arg_sorted = np.argsort(onsets)
            edges = self.edges.copy()
            for index, i_sorted in enumerate(arg_sorted):
                self.edges[str(index)] = edges[str(i_sorted)].copy()

            index = 0
            edge = self.edges['0']
            dispersion = self.energy_scale[1]-self.energy_scale[0]

            while str(index + 1) in self.edges:
                next_edge = self.edges[str(index + 1)]
                if edge['end_exclude'] > next_edge['start_exclude'] - 5 * dispersion:
                    edge['end_exclude'] = next_edge['start_exclude'] - 5 * dispersion
                edge = next_edge
                index += 1

            if edge['end_exclude'] > self.energy_scale[-3]:
                edge['end_exclude'] = self.energy_scale[-3]

        def set_elements(self, selected_elements):

            edge_list = self.ui.list3
            index = 0  # edge_list.currentIndex()

            for elem in selected_elements:
                if self.update_element(elem):
                    index = edge_list.currentIndex()
                    edge_list.setCurrentIndex(index + 1)
            self.sort_elements()
            edge_list.setCurrentIndex(index)
            self.update()

        def plot(self, event=None):
            self.energy_scale = self.dataset._axes[self.spec_dim].values
            if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                spectrum = self.dataset.view.get_spectrum()
                self.axis = self.dataset.view.axes[1]
            else:
                spectrum = np.array(self.dataset)
                self.axis = self.dataset.view.axis

            if self.ui.select10.isChecked():
                if 'experiment' in self.dataset.metadata:
                    exp = self.dataset.metadata['experiment']
                    if 'convergence_angle' not in exp:
                        raise ValueError('need a convergence_angle in experiment of metadata dictionary ')
                    alpha = exp['convergence_angle']
                    beta = exp['collection_angle']
                    beam_kv = exp['acceleration_voltage']

                    eff_beta = eels.effective_collection_angle(self.energy_scale, alpha, beta, beam_kv)
                    edges = eels.make_cross_sections(self.edges, np.array(self.energy_scale), beam_kv, eff_beta)
                    self.edges = eels.fit_edges2(spectrum, self.energy_scale, edges)
                    areal_density = []
                    elements = []
                    for key in edges:
                        if key.isdigit():  # only edges have numbers in that dictionary
                            elements.append(edges[key]['element'])
                            areal_density.append(edges[key]['areal_density'])
                    areal_density = np.array(areal_density)
                    out_string = '\nRelative composition: \n'
                    for i, element in enumerate(elements):
                        out_string += f'{element}: {areal_density[i] / areal_density.sum() * 100:.1f}%  '

                    self.model = self.edges['model']['spectrum']
                    self.update()

            x_limit = self.axis.get_xlim()
            y_limit = np.array(self.axis.get_ylim())*self.change_y_scale
            self.change_y_scale = 1.0
            
            self.axis.clear()

            line1, = self.axis.plot(self.energy_scale, spectrum*self.y_scale, label='spectrum')
            lines = [line1]

            def onpick(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg_line = event.artist
                orig_line = lined[legline]
                vis = not origline.get_visible()
                orig_line.set_visible(vis)
                # Change the alpha on the line in the legend, so we can see what lines
                # have been toggled
                if vis:
                    leg_line.set_alpha(1.0)
                else:
                    leg_line.set_alpha(0.2)
                self.figure.canvas.draw()

            if len(self.model) > 1:
                line2, = self.axis.plot(self.energy_scale, self.model*self.y_scale, label='model')
                line3, = self.axis.plot(self.energy_scale, (spectrum - self.model)*self.y_scale, label='difference')
                line4, = self.axis.plot(self.energy_scale, (spectrum - self.model) / np.sqrt(spectrum)*self.y_scale, label='Poisson')
                lines = [line1, line2, line3, line4]
                lined = dict()

                legend = self.axis.legend(loc='upper right', fancybox=True, shadow=True)

                legend.get_frame().set_alpha(0.4)
                for legline, origline in zip(legend.get_lines(), lines):
                    legline.set_picker(5)  # 5 pts tolerance
                    lined[legline] = origline
                self.figure.canvas.mpl_connect('pick_event', onpick)
            self.axis.set_xlim(x_limit)
            self.axis.set_ylim(y_limit)
            
            if self.y_scale != 1.:
                self.axis.set_ylabel('scattering intensity (ppm)')
            else:
                self.axis.set_ylabel('intensity (counts)')
            self.axis.set_xlabel('energy_loss (eV)')
                

            if self.ui.show_edges.isChecked():
                self.show_edges()
            if self.show_regions:
                self.plot_regions()
            self.figure.canvas.draw_idle()

        def plot_regions(self):
            y_min, y_max = self.axis.get_ylim()
            height = y_max - y_min

            rect = []
            if 'fit_area' in self.edges:
                color = 'blue'
                alpha = 0.2
                x_min = self.edges['fit_area']['fit_start']
                width = self.edges['fit_area']['fit_end'] - x_min
                rect.append(patches.Rectangle((x_min, y_min), width, height,
                                              edgecolor=color, alpha=alpha, facecolor=color))
                self.axis.add_patch(rect[0])
                self.axis.text(x_min, y_max, 'fit region', verticalalignment='top')
            color = 'red'
            alpha = 0.5
            for key in self.edges:
                if key.isdigit():
                    x_min = self.edges[key]['start_exclude']
                    width = self.edges[key]['end_exclude']-x_min
                    rect.append(patches.Rectangle((x_min, y_min), width, height,
                                                  edgecolor=color, alpha=alpha, facecolor=color))
                    self.axis.add_patch(rect[-1])
                    self.axis.text(x_min, y_max, f"exclude\n edge {int(key)+1}", verticalalignment='top')

        def show_edges(self):
            x_min, x_max = self.axis.get_xlim()
            y_min, y_max = self.axis.get_ylim()

            for key, edge in self.edges.items():
                i = 0
                if key.isdigit():
                    element = edge['element']
                    for sym in edge['all_edges']:
                        x = edge['all_edges'][sym]['onset'] + edge['chemical_shift']
                        if x_min < x < x_max:
                            self.axis.text(x, y_max, '\n' * i + f"{element}-{sym}",
                                           verticalalignment='top', color='black')
                            self.axis.axvline(x, ymin=0, ymax=1, color='gray')
                            i += 1

        def check_area_consistency(self):
            if self.dataset is None:
                return
            onset = float(self.ui.edit6.displayText())
            excl_start = float(self.ui.edit7.displayText())
            excl_end = float(self.ui.edit8.displayText())
            if onset < self.energy_scale[2]:
                onset = self.energy_scale[2]
                excl_start = self.energy_scale[2]
            if onset > self.energy_scale[-2]:
                onset = self.energy_scale[-2]
                excl_end = self.energy_scale[-2]
            if excl_start > onset:
                excl_start = onset
            if excl_end < onset:
                excl_end = onset

            index = self.ui.list3.currentIndex()
            self.edges[str(index)]['chemical_shift'] = onset - self.edges[str(index)]['original_onset']
            self.edges[str(index)]['onset'] = onset
            self.edges[str(index)]['end_exclude'] = excl_end
            self.edges[str(index)]['start_exclude'] = excl_start

            self.update()

        def on_list_enter(self):
            sender = self.sender()

            if sender.objectName() == 'edge_list':
                index = self.ui.list3.currentIndex()

                number_of_edges = 0
                for key in self.edges:
                    if key.isdigit():
                        if int(key) > number_of_edges:
                            number_of_edges = int(key)
                number_of_edges += 1
                if index > number_of_edges:
                    index = number_of_edges
                self.ui.list3.setCurrentIndex(index)
                if str(index) not in self.edges:
                    self.edges[str(index)] = {'z': 0, 'symmetry': 'K1', 'element': 'H', 'onset': 0, 'end_exclude': 0,
                                              'start_exclude': 0, 'areal_density': 0}

                self.update()
            elif sender.objectName() == 'symmetry_list':
                sym = self.ui.list5.currentText()
                index = self.ui.list3.currentIndex()
                zz = self.edges[str(index)]['z']
                if zz > 1:
                    x_section = eels.get_x_sections(zz)
                    if sym in x_section:
                        start_exclude = x_section[sym]['onset'] - x_section[sym]['excl before']
                        end_exclude = x_section[sym]['onset'] + x_section[sym]['excl after']
                        self.edges[str(index)].update({'symmetry': sym, 'onset': x_section[sym]['onset'],
                                                       'end_exclude': end_exclude, 'start_exclude': start_exclude})
                        self.edges[str(index)]['chemical_shift'] = 0.0
                        self.edges[str(index)]['areal_density'] = 0.0
                        self.edges[str(index)]['original_onset'] = self.edges[index]['onset']
                        self.update()
            elif sender.objectName() == 'symmetry_method':
                self.ui.select5.setCurrentIndex(0)

        def on_check(self):
            sender = self.sender()

            if sender.objectName() == 'edge_check':
                self.show_regions = sender.isChecked()
            elif sender.objectName() == 'conv_ll':
                self.edges['use_low_loss'] = self.ui.check10.isChecked()
                if self.ui.check10.isChecked():
                    self.low_loss()
            elif sender.objectName() == 'probability':
                dispersion = self.energy_scale[1]-self.energy_scale[0]
                if sender.isChecked():
                    self.y_scale = 1/self.dataset.metadata['experiment']['flux_ppm']*dispersion
                    self.change_y_scale = 1/self.dataset.metadata['experiment']['flux_ppm']*dispersion
                else:
                    self.y_scale = 1.0
                    self.change_y_scale = self.dataset.metadata['experiment']['flux_ppm']/dispersion
                self.update()
            self.plot()

        def low_loss(self):
            self.edges['use_low_loss'] = self.ui.check10.isChecked()

            if 'low_loss' not in self.edges:
                self.edges['low_loss'] = {}
            if 'spectrum' not in self.edges['low_loss']:
                spectrum_ll = ft.open_file(write_hdf_file=False)

                self.edges['low_loss']['spectrum'] = np.array(spectrum_ll)
            self.spectrum_ll = self.edges['low_loss']['spectrum']

        def do_all_button_click(self):

            if self.dataset.data_type.name != 'SPECTRAL_IMAGE':
                self.do_fit_button_click()
                return

            if 'experiment' in self.dataset.metadata:
                exp = self.dataset.metadata['experiment']
                if 'convergence_angle' not in exp:
                    raise ValueError('need a convergence_angle in experiment of metadata dictionary ')
                alpha = exp['convergence_angle']
                beta = exp['collection_angle']
                beam_kv = exp['acceleration_voltage']
            else:
                raise ValueError('need a experiment parameter in metadata dictionary')

            self.energy_scale = self.dataset._axes[self.spec_dim].values
            eff_beta = eels.effective_collection_angle(self.energy_scale, alpha, beta, beam_kv)
            if self.edges['use_low_loss']:
                low_loss = self.spectrum_ll/self.spectrum_ll.sum()
            else:
                low_loss = None

            edges = eels.make_cross_sections(self.edges, np.array(self.energy_scale), beam_kv, eff_beta,
                                             low_loss=low_loss)

            view = self.dataset.view
            bin_x = view.bin_x
            bin_y = view.bin_y

            start_x = view.x
            start_y = view.y

            number_of_edges = 0
            for key in self.edges:
                if key.isdigit():
                    number_of_edges += 1

            results = np.zeros([int(self.dataset.shape[0]/bin_x), int(self.dataset.shape[1]/bin_y), number_of_edges])
            total_spec = int(self.dataset.shape[0]/bin_x)*int(self.dataset.shape[1]/bin_y)
            self.ui.progress.setMaximum(total_spec)
            self.ui.progress.setValue(0)
            ind = 0
            for x in range(int(self.dataset.shape[0]/bin_x)):

                for y in range(int(self.dataset.shape[1]/bin_y)):
                    ind += 1
                    self.ui.progress.setValue(ind)
                    view.x = x*bin_x
                    view.y = y*bin_y
                    spectrum = view.get_spectrum()

                    edges = eels.fit_edges2(spectrum, self.energy_scale, edges)
                    for key, edge in edges.items():
                        if key.isdigit():
                            # element.append(edge['element'])
                            results[x, y, int(key)] = edge['areal_density']
            edges['spectrum_image_quantification'] = results
            self.ui.progress.setValue(total_spec)
            view.x = start_x
            view.y = start_y

        def do_fit_button_click(self):
            if 'experiment' in self.dataset.metadata:
                exp = self.dataset.metadata['experiment']
                if 'convergence_angle' not in exp:
                    raise ValueError('need a convergence_angle in experiment of metadata dictionary ')
                alpha = exp['convergence_angle']
                beta = exp['collection_angle']
                beam_kv = exp['acceleration_voltage']

            else:
                raise ValueError('need a experiment parameter in metadata dictionary')
            self.energy_scale = self.dataset._axes[self.spec_dim].values
            eff_beta = eels.effective_collection_angle(self.energy_scale, alpha, beta, beam_kv)

            if self.edges['use_low_loss']:
                low_loss = self.spectrum_ll / self.spectrum_ll.sum()
            else:
                low_loss = None
            edges = eels.make_cross_sections(self.edges, np.array(self.energy_scale), beam_kv, eff_beta, low_loss)

            if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                spectrum = self.dataset.view.get_spectrum()
            else:
                spectrum = self.dataset
            self.edges = eels.fit_edges2(spectrum, self.energy_scale, edges)
            areal_density = []
            elements = []
            for key in edges:
                if key.isdigit():  # only edges have numbers in that dictionary
                    elements.append(edges[key]['element'])
                    areal_density.append(edges[key]['areal_density'])
            areal_density = np.array(areal_density)
            out_string = '\nRelative composition: \n'
            for i, element in enumerate(elements):
                out_string += f'{element}: {areal_density[i] / areal_density.sum() * 100:.1f}%  '

            self.model = self.edges['model']['spectrum']
            self.update()
            self.plot()

        def set_action(self):
            self.ui.edit1.editingFinished.connect(self.on_enter)
            self.ui.edit2.editingFinished.connect(self.on_enter)
            self.ui.list3.activated[str].connect(self.on_list_enter)
            self.ui.check3.clicked.connect(self.on_check)
            self.ui.edit4.editingFinished.connect(self.on_enter)
            self.ui.list5.activated[str].connect(self.on_list_enter)
            self.ui.select5.activated[str].connect(self.on_list_enter)

            self.ui.edit6.editingFinished.connect(self.on_enter)
            self.ui.edit7.editingFinished.connect(self.on_enter)
            self.ui.edit8.editingFinished.connect(self.on_enter)
            self.ui.edit9.editingFinished.connect(self.on_enter)

            self.ui.check10.clicked.connect(self.on_check)
            self.ui.select10.clicked.connect(self.on_check)
            self.ui.show_edges.clicked.connect(self.on_check)
            self.ui.check_probability.clicked.connect(self.on_check)
            
            self.ui.do_all_button.clicked.connect(self.do_all_button_click)
            self.ui.do_fit_button.clicked.connect(self.do_fit_button_click)


    class CurveVisualizer(object):
        """Plots a sidpy.Dataset with spectral dimension-type

        """
        def __init__(self, dset, spectrum_number=None, axis=None, leg=None, **kwargs):
            if not isinstance(dset, sidpy.Dataset):
                raise TypeError('dset should be a sidpy.Dataset object')
            if axis is None:
                self.fig = plt.figure()
                self.axis = self.fig.add_subplot(1, 1, 1)
            else:
                self.axis = axis
                self.fig = axis.figure

            self.dset = dset
            self.selection = []
            [self.spec_dim, self.energy_scale] = ft.get_dimensions_by_type('spectral', self.dset)[0]

            self.lined = dict()
            self.plot(**kwargs)

        def plot(self, **kwargs):
            if self.dset.data_type.name == 'IMAGE_STACK':
                line1, = self.axis.plot(self.energy_scale.values, self.dset[0, 0], label='spectrum', **kwargs)
            else:
                line1, = self.axis.plot(self.energy_scale.values, self.dset, label='spectrum', **kwargs)
            lines = [line1]
            if 'add2plot' in self.dset.metadata:
                data = self.dset.metadata['add2plot']
                for key, line in data.items():
                    line_add, = self.axis.plot(self.energy_scale.values,  line['data'], label=line['legend'])
                    lines.append(line_add)

                legend = self.axis.legend(loc='upper right', fancybox=True, shadow=True)
                legend.get_frame().set_alpha(0.4)

                for legline, origline in zip(legend.get_lines(), lines):
                    legline.set_picker(True)
                    legline.set_pickradius(5)  # 5 pts tolerance
                    self.lined[legline] = origline
                self.fig.canvas.mpl_connect('pick_event', self.onpick)

            self.axis.axhline(0, color='gray', alpha=0.6)
            self.axis.set_xlabel(self.dset.labels[0])
            self.axis.set_ylabel(self.dset.data_descriptor)
            self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
            self.fig.canvas.draw_idle()

        def update(self, **kwargs):
            x_limit = self.axis.get_xlim()
            y_limit = self.axis.get_ylim()
            self.axis.clear()
            self.plot(**kwargs)
            self.axis.set_xlim(x_limit)
            self.axis.set_ylim(y_limit)

        def onpick(self, event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = self.lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend, so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            self.fig.canvas.draw()
