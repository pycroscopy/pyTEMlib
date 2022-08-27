"""
    EELS Input Dialog for ELNES Analysis
"""
from os import error
Qt_available = True
try:
    from PyQt5 import QtCore,  QtWidgets
except:
    Qt_available = False
    print('Qt dialogs are not available')

import numpy as np
import scipy
import scipy.optimize
import scipy.signal

import sidpy
import pyTEMlib.file_tools as ft
import pyTEMlib.eels_tools as eels
import pyTEMlib.peak_dlg as peak_dlg

advanced_present = True
try:
    import advanced_eels_tools
    print('advanced EELS features enabled')
except ModuleNotFoundError:
    advanced_present = False

_version = .001

if Qt_available:
    class PeakFitDialog(QtWidgets.QDialog):
        """
        EELS Input Dialog for ELNES Analysis
        """

        def __init__(self, dataset=None):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            # Create an instance of the GUI
            self.ui = peak_dlg.UiDialog(self)

            self.set_action()

            self.dataset = dataset
            self.energy_scale = np.array([])
            self.peak_out_list = []
            self.p_out = []
            self.axis = None
            self.show_regions = False
            self.show()

            if dataset is None:
                # make a dummy dataset
                dataset = ft.make_dummy_dataset('spectrum')

            if not isinstance(dataset, sidpy.Dataset):
                raise TypeError('dataset has to be a sidpy dataset')
            self.dataset = dataset
            self.spec_dim = ft.get_dimensions_by_type('spectral', dataset)
            if len(self.spec_dim) != 1:
                raise TypeError('We need exactly one SPECTRAL dimension')
            self.spec_dim = self.spec_dim[0]
            self.energy_scale = self.spec_dim[1].values.copy()

            if 'peak_fit' not in self.dataset.metadata:
                self.dataset.metadata['peak_fit'] = {}
                if 'edges' in self.dataset.metadata:
                    if 'fit_area' in self.dataset.metadata['edges']:
                        self.dataset.metadata['peak_fit']['fit_start'] = \
                            self.dataset.metadata['edges']['fit_area']['fit_start']
                        self.dataset.metadata['peak_fit']['fit_end'] = self.dataset.metadata['edges']['fit_area']['fit_end']
                    self.dataset.metadata['peak_fit']['peaks'] = {'0': {'position': self.energy_scale[1],
                                                                        'amplitude': 1000.0, 'width': 1.0,
                                                                        'type': 'Gauss', 'asymmetry': 0}}

            self.peaks = self.dataset.metadata['peak_fit']
            if 'fit_start' not in self.peaks:
                self.peaks['fit_start'] = self.energy_scale[1]
                self.peaks['fit_end'] = self.energy_scale[-2]

            if 'peak_model' in self.peaks:
                self.peak_model = self.peaks['peak_model']
                self.model = self.peak_model
                if 'edge_model' in self.peaks:
                    self.model = self.model + self.peaks['edge_model']
            else:
                self.model = np.array([])
                self.peak_model = np.array([])
            if 'peak_out_list' in self.peaks:
                self.peak_out_list = self.peaks['peak_out_list']
            self.set_peak_list()

            # check whether a core loss analysis has been done previously
            if not hasattr(self, 'core_loss') and 'edges' in self.dataset.metadata:
                self.core_loss = True
            else:
                self.core_loss = False

            self.update()
            self.dataset.plot()

            if self.dataset.data_type.name == 'SPECTRAL_IMAGE':
                if 'SI_bin_x' not in self.dataset.metadata['experiment']:
                    self.dataset.metadata['experiment']['SI_bin_x'] = 1
                    self.dataset.metadata['experiment']['SI_bin_y'] = 1
                bin_x = self.dataset.metadata['experiment']['SI_bin_x']
                bin_y = self.dataset.metadata['experiment']['SI_bin_y']

                self.dataset.view.set_bin([bin_x, bin_y])

            if hasattr(self.dataset.view, 'axes'):
                self.axis = self.dataset.view.axes[-1]
            elif hasattr(self.dataset.view, 'axis'):
                self.axis = self.dataset.view.axis
            self.figure = self.axis.figure

            if not advanced_present:
                self.ui.iteration_list = ['0']
                self.ui.smooth_list.clear()
                self.ui.smooth_list.addItems(self.ui.iteration_list)
                self.ui.smooth_list.setCurrentIndex(0)

            self.figure.canvas.mpl_connect('button_press_event', self.plot)
            self.plot()

        def update(self):
            # self.setWindowTitle('update')
            self.ui.edit1.setText(f"{self.peaks['fit_start']:.2f}")
            self.ui.edit2.setText(f"{self.peaks['fit_end']:.2f}")

            peak_index = self.ui.list3.currentIndex()
            if str(peak_index) not in self.peaks['peaks']:
                self.peaks['peaks'][str(peak_index)] = {'position': self.energy_scale[1], 'amplitude': 1000.0,
                                                        'width': 1.0, 'type': 'Gauss', 'asymmetry': 0}
                self.ui.list4.setCurrentText(self.peaks['peaks'][str(peak_index)]['type'])
            if 'associated_edge' in self.peaks['peaks'][str(peak_index)]:
                self.ui.unit3.setText(self.peaks['peaks'][str(peak_index)]['associated_edge'])
            else:
                self.ui.unit3.setText('')
            self.ui.edit5.setText(f"{self.peaks['peaks'][str(peak_index)]['position']:.2f}")
            self.ui.edit6.setText(f"{self.peaks['peaks'][str(peak_index)]['amplitude']:.2f}")
            self.ui.edit7.setText(f"{self.peaks['peaks'][str(peak_index)]['width']:.2f}")
            if 'asymmetry' not in self.peaks['peaks'][str(peak_index)]:
                self.peaks['peaks'][str(peak_index)]['asymmetry'] = 0.
            self.ui.edit8.setText(f"{self.peaks['peaks'][str(peak_index)]['asymmetry']:.2f}")

        def plot(self):
            spec_dim = ft.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL, self.dataset)
            spec_dim = spec_dim[0]
            self.energy_scale = spec_dim[1].values
            if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                spectrum = self.dataset.view.get_spectrum()
                self.axis = self.dataset.view.axes[1]
            else:
                spectrum = np.array(self.dataset)
                self.axis = self.dataset.view.axis

            x_limit = self.axis.get_xlim()
            y_limit = self.axis.get_ylim()
            self.axis.clear()

            self.axis.plot(self.energy_scale, spectrum, label='spectrum')
            if len(self.model) > 1:
                self.axis.plot(self.energy_scale, self.model, label='model')
                self.axis.plot(self.energy_scale, spectrum - self.model, label='difference')
                self.axis.plot(self.energy_scale, (spectrum - self.model) / np.sqrt(spectrum), label='Poisson')
                self.axis.legend()
            self.axis.set_xlim(x_limit)
            self.axis.set_ylim(y_limit)
            self.axis.figure.canvas.draw_idle()

            for index, peak in self.peaks['peaks'].items():
                p = [peak['position'], peak['amplitude'], peak['width']]
                self.axis.plot(self.energy_scale, eels.gauss(self.energy_scale, p))

        def fit_peaks(self):
            """Fit spectrum with peaks given in peaks dictionary"""
            print('Fitting peaks...')
            p_in = []
            for key, peak in self.peaks['peaks'].items():
                if key.isdigit():
                    p_in.append(peak['position'])
                    p_in.append(peak['amplitude'])
                    p_in.append(peak['width'])

            # check whether we have a spectral image or just a single spectrum
            if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                spectrum = self.dataset.view.get_spectrum()
            else:
                spectrum = np.array(self.dataset)

            # set the energy scale and fit start and end points
            energy_scale = np.array(self.energy_scale)
            start_channel = np.searchsorted(energy_scale, self.peaks['fit_start'])
            end_channel = np.searchsorted(energy_scale, self.peaks['fit_end'])

            energy_scale = self.energy_scale[start_channel:end_channel]
            # select the core loss model if it exists. Otherwise, we will fit to the full spectrum.
            if self.core_loss:
                print('Core loss model found. Fitting on top of the model.')
                model = self.dataset.metadata['edges']['model']['spectrum'][start_channel:end_channel]
            else:
                print('No core loss model found. Fitting to the full spectrum.')
                model = np.zeros(end_channel - start_channel)

            # if we have a core loss model we will only fit the difference between the model and the data.
            difference = np.array(spectrum[start_channel:end_channel] - model)

            # find the optimum fitting parameters
            [self.p_out, _] = scipy.optimize.leastsq(eels.residuals_smooth, np.array(p_in), ftol=1e-3,
                                                     args=(energy_scale, difference, False))

            # construct the fit data from the optimized parameters
            self.peak_model = np.zeros(len(self.energy_scale))
            self.model = np.zeros(len(self.energy_scale))
            self.model[start_channel:end_channel] = model
            fit = eels.model_smooth(energy_scale, self.p_out, False)
            self.peak_model[start_channel:end_channel] = fit
            self.dataset.metadata['peak_fit']['edge_model'] = self.model
            self.model = self.model + self.peak_model
            self.dataset.metadata['peak_fit']['peak_model'] = self.peak_model

            for key, peak in self.peaks['peaks'].items():
                if key.isdigit():
                    p_index = int(key)*3
                    self.peaks['peaks'][key] = {'position': self.p_out[p_index],
                                                'amplitude': self.p_out[p_index+1],
                                                'width': self.p_out[p_index+2],
                                                'associated_edge': ''}

            self.find_associated_edges()
            self.find_white_lines()
            self.update()
            self.plot()

        def smooth(self):
            """Fit lots of Gaussian to spectrum and let the program sort it out

            We sort the peaks by area under the Gaussians, assuming that small areas mean noise.

            """
            iterations = int(self.ui.smooth_list.currentIndex())

            self.peak_model, self.peak_out_list, number_of_peaks = smooth(self.dataset, iterations, advanced_present)

            spec_dim = ft.get_dimensions_by_type('SPECTRAL', self.dataset)[0]
            if spec_dim[1][0] > 0:
                self.model = self.dataset.metadata['edges']['model']['spectrum']
            else:
                self.model = np.zeros(len(spec_dim[1]))

            self.ui.find_edit.setText(str(number_of_peaks))

            self.dataset.metadata['peak_fit']['edge_model'] = self.model
            self.model = self.model + self.peak_model
            self.dataset.metadata['peak_fit']['peak_model'] = self.peak_model
            self.dataset.metadata['peak_fit']['peak_out_list'] = self.peak_out_list

            self.update()
            self.plot()

        def find_associated_edges(self):
            onsets = []
            edges = []
            if 'edges' in self.dataset.metadata:
                for key, edge in self.dataset.metadata['edges'].items():
                    if key.isdigit():
                        element = edge['element']
                        for sym in edge['all_edges']:  # TODO: Could be replaced with exclude
                            onsets.append(edge['all_edges'][sym]['onset'] + edge['chemical_shift'])
                            # if 'sym' == edge['symmetry']:
                            edges.append([key, f"{element}-{sym}", onsets[-1]])
            for key, peak in self.peaks['peaks'].items():
                if key.isdigit():
                    distance = self.energy_scale[-1]
                    index = -1
                    for ii, onset in enumerate(onsets):
                        if onset < peak['position'] < onset+50:
                            if distance > np.abs(peak['position'] - onset):
                                distance = np.abs(peak['position'] - onset)  # TODO: check whether absolute is good
                                distance_onset = peak['position'] - onset
                                index = ii
                    if index >= 0:
                        peak['associated_edge'] = edges[index][1]  # check if more info is necessary
                        peak['distance_to_onset'] = distance_onset

        def find_white_lines(self):
            white_lines = {}
            for index, peak in self.peaks['peaks'].items():
                if index.isdigit():
                    if 'associated_edge' in peak:
                        if peak['associated_edge'][-2:] in ['L3', 'L2', 'M5', 'M4']:
                            if peak['distance_to_onset'] < 10:
                                area = np.sqrt(2 * np.pi) * peak['amplitude'] * np.abs(peak['width']/np.sqrt(2 * np.log(2)))
                                if peak['associated_edge'] not in white_lines:
                                    white_lines[peak['associated_edge']] = 0.
                                if area > 0:
                                    white_lines[peak['associated_edge']] += area  # TODO: only positive ones?
            white_line_ratios = {}
            white_line_sum = {}
            for sym, area in white_lines.items():
                if sym[-2:] in ['L2', 'M4', 'M2']:
                    if area > 0 and f"{sym[:-1]}{int(sym[-1]) + 1}" in white_lines:
                        if white_lines[f"{sym[:-1]}{int(sym[-1]) + 1}"] > 0:
                            white_line_ratios[f"{sym}/{sym[-2]}{int(sym[-1]) + 1}"] = area / white_lines[
                                f"{sym[:-1]}{int(sym[-1]) + 1}"]
                            white_line_sum[f"{sym}+{sym[-2]}{int(sym[-1]) + 1}"] = (
                                        area + white_lines[f"{sym[:-1]}{int(sym[-1]) + 1}"])

                            areal_density = 1.
                            if 'edges' in self.dataset.metadata:
                                for key, edge in self.dataset.metadata['edges'].items():
                                    if key.isdigit():
                                        if edge['element'] == sym.split('-')[0]:
                                            areal_density = edge['areal_density']
                                            break
                            white_line_sum[f"{sym}+{sym[-2]}{int(sym[-1]) + 1}"] /= areal_density

            self.peaks['white_lines'] = white_lines
            self.peaks['white_line_ratios'] = white_line_ratios
            self.peaks['white_line_sums'] = white_line_sum
            self.ui.wl_list = []
            self.ui.wls_list = []
            if len(self.peaks['white_line_ratios']) > 0:
                for key in self.peaks['white_line_ratios']:
                    self.ui.wl_list.append(key)
                for key in self.peaks['white_line_sums']:
                    self.ui.wls_list.append(key)

                self.ui.listwl.clear()
                self.ui.listwl.addItems(self.ui.wl_list)
                self.ui.listwl.setCurrentIndex(0)
                self.ui.unitswl.setText(f"{self.peaks['white_line_ratios'][self.ui.wl_list[0]]:.2f}")

                self.ui.listwls.clear()
                self.ui.listwls.addItems(self.ui.wls_list)
                self.ui.listwls.setCurrentIndex(0)
                self.ui.unitswls.setText(f"{self.peaks['white_line_sums'][self.ui.wls_list[0]]*1e6:.4f} ppm")
            else:
                self.ui.wl_list.append('Ratio')
                self.ui.wls_list.append('Sum')

                self.ui.listwl.clear()
                self.ui.listwl.addItems(self.ui.wl_list)
                self.ui.listwl.setCurrentIndex(0)
                self.ui.unitswl.setText('')

                self.ui.listwls.clear()
                self.ui.listwls.addItems(self.ui.wls_list)
                self.ui.listwls.setCurrentIndex(0)
                self.ui.unitswls.setText('')

        def find_peaks(self):
            number_of_peaks = int(str(self.ui.find_edit.displayText()).strip())

            # is now sorted in smooth function
            # flat_list = [item for sublist in self.peak_out_list for item in sublist]
            # new_list = np.reshape(flat_list, [len(flat_list) // 3, 3])
            # arg_list = np.argsort(np.abs(new_list[:, 1]))

            self.ui.peak_list = []
            self.peaks['peaks'] = {}
            for i in range(number_of_peaks):
                self.ui.peak_list.append(f'Peak {i+1}')
                p = self.peak_out_list[i]
                self.peaks['peaks'][str(i)] = {'position': p[0], 'amplitude': p[1], 'width': p[2], 'type': 'Gauss',
                                               'asymmetry': 0}

            self.ui.peak_list.append(f'add peak')
            self.ui.list3.clear()
            self.ui.list3.addItems(self.ui.peak_list)
            self.ui.list3.setCurrentIndex(0)
            self.find_associated_edges()
            self.find_white_lines()

            self.update()
            self.plot()

        def set_peak_list(self):
            self.ui.peak_list = []
            if 'peaks' not in self.peaks:
                self.peaks['peaks'] = {}
            key = 0
            for key in self.peaks['peaks']:
                if key.isdigit():
                    self.ui.peak_list.append(f'Peak {int(key) + 1}')
            self.ui.find_edit.setText(str(int(key) + 1))
            self.ui.peak_list.append(f'add peak')
            self.ui.list3.clear()
            self.ui.list3.addItems(self.ui.peak_list)
            self.ui.list3.setCurrentIndex(0)

        def on_enter(self):
            if self.sender() == self.ui.edit1:
                value = float(str(self.ui.edit1.displayText()).strip())
                if value < self.energy_scale[0]:
                    value = self.energy_scale[0]
                if value > self.energy_scale[-5]:
                    value = self.energy_scale[-5]
                self.peaks['fit_start'] = value
                self.ui.edit1.setText(str(self.peaks['fit_start']))
            elif self.sender() == self.ui.edit2:
                value = float(str(self.ui.edit2.displayText()).strip())
                if value < self.energy_scale[5]:
                    value = self.energy_scale[5]
                if value > self.energy_scale[-1]:
                    value = self.energy_scale[-1]
                self.peaks['fit_end'] = value
                self.ui.edit2.setText(str(self.peaks['fit_end']))
            elif self.sender() == self.ui.edit5:
                value = float(str(self.ui.edit5.displayText()).strip())
                peak_index = self.ui.list3.currentIndex()
                self.peaks['peaks'][str(peak_index)]['position'] = value
            elif self.sender() == self.ui.edit6:
                value = float(str(self.ui.edit6.displayText()).strip())
                peak_index = self.ui.list3.currentIndex()
                self.peaks['peaks'][str(peak_index)]['amplitude'] = value
            elif self.sender() == self.ui.edit7:
                value = float(str(self.ui.edit7.displayText()).strip())
                peak_index = self.ui.list3.currentIndex()
                self.peaks['peaks'][str(peak_index)]['width'] = value

        def on_list_enter(self):
            # self.setWindowTitle('list')
            if self.sender() == self.ui.list3:
                if self.ui.list3.currentText().lower() == 'add peak':
                    peak_index = self.ui.list3.currentIndex()
                    self.ui.list3.insertItem(peak_index, f'Peak {peak_index+1}')
                    self.peaks['peaks'][str(peak_index+1)] = {'position': self.energy_scale[1],
                                                              'amplitude': 1000.0, 'width': 1.0,
                                                              'type': 'Gauss', 'asymmetry': 0}
                    self.ui.list3.setCurrentIndex(peak_index)
                self.update()

            elif self.sender() == self.ui.listwls or self.sender() == self.ui.listwl:
                wl_index = self.sender().currentIndex()

                self.ui.listwl.setCurrentIndex(wl_index)
                self.ui.unitswl.setText(f"{self.peaks['white_line_ratios'][self.ui.wl_list[wl_index]]:.2f}")
                self.ui.listwls.setCurrentIndex(wl_index)
                self.ui.unitswls.setText(f"{self.peaks['white_line_sums'][self.ui.wls_list[wl_index]] * 1e6:.4f} ppm")

        def set_action(self):
            pass
            self.ui.edit1.editingFinished.connect(self.on_enter)
            self.ui.edit2.editingFinished.connect(self.on_enter)
            self.ui.edit5.editingFinished.connect(self.on_enter)
            self.ui.edit6.editingFinished.connect(self.on_enter)
            self.ui.edit7.editingFinished.connect(self.on_enter)
            self.ui.edit8.editingFinished.connect(self.on_enter)
            self.ui.list3.activated[str].connect(self.on_list_enter)
            self.ui.find_button.clicked.connect(self.find_peaks)
            self.ui.smooth_button.clicked.connect(self.smooth)
            self.ui.fit_button.clicked.connect(self.fit_peaks)
            self.ui.listwls.activated[str].connect(self.on_list_enter)
            self.ui.listwl.activated[str].connect(self.on_list_enter)


    def smooth(dataset, iterations, advanced_present):
        """Gaussian mixture model (non-Bayesian)

        Fit lots of Gaussian to spectrum and let the program sort it out
        We sort the peaks by area under the Gaussians, assuming that small areas mean noise.

        """

        # TODO: add sensitivity to dialog and the two functions below
        peaks = dataset.metadata['peak_fit']

        if advanced_present and iterations > 1:
            peak_model, peak_out_list = advanced_eels_tools.smooth(dataset, peaks['fit_start'],
                                                                   peaks['fit_end'], iterations=iterations)
        else:
            peak_model, peak_out_list = eels.find_peaks(dataset, peaks['fit_start'], peaks['fit_end'])
            peak_out_list = [peak_out_list]

        flat_list = [item for sublist in peak_out_list for item in sublist]
        new_list = np.reshape(flat_list, [len(flat_list) // 3, 3])
        area = np.sqrt(2 * np.pi) * np.abs(new_list[:, 1]) * np.abs(new_list[:, 2] / np.sqrt(2 * np.log(2)))
        arg_list = np.argsort(area)[::-1]
        area = area[arg_list]
        peak_out_list = new_list[arg_list]

        number_of_peaks = np.searchsorted(area * -1, -np.average(area))

        return peak_model, peak_out_list, number_of_peaks
