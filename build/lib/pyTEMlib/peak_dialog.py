from PyQt5 import QtCore,  QtWidgets

import numpy as np
import scipy
import scipy.optimize
import scipy.signal
import eels_tools as eels

import peak_dlg
import sidpy
import file_tools_nsid as ft

_version = 000


class PeakFitDialog(QtWidgets.QDialog):
    """
    EELS Input Dialog for Chemical Analysis
    """

    def __init__(self, dataset=None):
        super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
        # Create an instance of the GUI
        self.ui = peak_dlg.UiDialog(self)

        self.set_action()

        self.dataset = dataset
        self.energy_scale = np.array([])
        self.model = np.array([])
        self.peak_model = np.array([])
        self.peak_out_list = []
        self.edges = {}
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
                    self.dataset.metadata['peak_fit']['fit_start'] = self.dataset.metadata['edges']['fit_area']['fit_start']
                    self.dataset.metadata['peak_fit']['fit_end'] = self.dataset.metadata['edges']['fit_area']['fit_end']

                self.dataset.metadata['peak_fit']['peaks'] = {'0': {'position': self.energy_scale[1], 'amplitude': 1000.0,
                                         'width': 1.0, 'type': 'Gauss', 'asymmetry': 0}}

        self.peaks = self.dataset.metadata['peak_fit']
        if 'fit_start' not in self.peaks:
            self.peaks['fit_start'] = self.energy_scale[1]
            self.peaks['fit_end'] = self.energy_scale[-2]

        self.update()
        self.dataset.plot()
        self.figure = dataset.view.axis.figure
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
        self.energy_scale = self.dataset.energy_scale
        if self.dataset.data_type == sidpy.DataTypes.SPECTRAL_IMAGE:
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
            self.axis.plot(self.energy_scale, eels.gauss(self.dataset.energy_scale, p))


    def fit_peaks(self):
        p_in = []
        for key, peak in self.peaks['peaks'].items():
            if key.isdigit():
                p_in.append(peak['position'])
                p_in.append(peak['amplitude'])
                p_in.append(peak['width'])

        energy_scale = self.dataset.energy_scale
        start_channel = np.searchsorted(energy_scale, self.peaks['fit_start'])
        end_channel = np.searchsorted(energy_scale, self.peaks['fit_end'])

        energy_scale = self.dataset.energy_scale[start_channel:end_channel]
        if self.dataset.energy_scale[0] > 0:
            if 'edges' not in self.dataset.metadata:
                return
            if 'model' not in self.dataset.metadata['edges']:
                return
            model = self.dataset.metadata['edges']['model']['spectrum'][start_channel:end_channel]

        else:
            model = np.zeros(end_channel - start_channel)

        difference = np.array(self.dataset[start_channel:end_channel] - model)

        self.p_out, cov = scipy.optimize.leastsq(eels.residuals_smooth, p_in, ftol=1e-3, args=(energy_scale, difference,
                                                                                          False))
        self.peak_model = np.zeros(len(self.dataset.energy_scale))
        fit = eels.model_smooth(energy_scale, p_out, False)
        self.peak_model[start_channel:end_channel] = fit
        if self.dataset.energy_scale[0] > 0:
            self.model = self.dataset.metadata['edges']['model']['spectrum']
        else:
            self.model = np.zeros(len(self.dataset.energy_scale))
        self.model = self.model + self.peak_model

        self.plot()

    def smooth(self):
        iterations = int(self.ui.smooth_list.currentIndex())+1
        energy_scale = self.dataset.energy_scale
        start_channel = np.searchsorted(energy_scale, self.peaks['fit_start'])
        end_channel = np.searchsorted(energy_scale, self.peaks['fit_end'])

        energy_scale = self.dataset.energy_scale[start_channel:end_channel]
        if self.dataset.energy_scale[0] > 0:
            if 'edges' not in self.dataset.metadata:
                return
            if 'model' not in self.dataset.metadata['edges']:
                return
            model = self.dataset.metadata['edges']['model']['spectrum'][start_channel:end_channel]

        else:
            model = np.zeros(end_channel-start_channel)

        original_difference = np.array(self.dataset[start_channel:end_channel] - model)
        n_pks = 30

        self.peak_out_list = []
        fit = np.zeros(len(energy_scale))
        difference = np.array(original_difference)
        for i in range(iterations):
            i_pk = scipy.signal.find_peaks_cwt(np.abs(difference), widths=range(3, len(energy_scale) // n_pks))
            p_in = np.ravel([[energy_scale[i], difference[i], 1.0] for i in i_pk])  # starting guess for fit

            p_out, cov = scipy.optimize.leastsq(eels.residuals_smooth, p_in, ftol=1e-3, args=(energy_scale, difference,
                                                                                              False))
            self.peak_out_list.append(p_out)
            fit = fit + eels.model_smooth(energy_scale, p_out, False)
            difference = np.array(original_difference - fit)

        self.peak_model = np.zeros(len(self.dataset.energy_scale))
        self.peak_model[start_channel:end_channel] = fit



        if self.dataset.energy_scale[0] > 0:
            self.model = self.dataset.metadata['edges']['model']['spectrum']
        else:
            self.model = np.zeros(len(self.dataset.energy_scale))
        self.model = self.model+self.peak_model

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
                    if onset - 3 < peak['position'] < onset + 100:
                        if distance > np.abs(peak['position'] - onset):
                            distance = np.abs(peak['position'] - onset)  # TODO check whether absolute is good
                            index = ii
                if index > 0:
                    peak['associated_edge'] = edges[index][1]  # check if more info is necessary

    def find_white_lines(self):
        white_lines = {}
        for index, peak in self.peaks['peaks'].items():
            if index.isdigit():
                if 'associated_edge' in peak:
                    if peak['associated_edge'][-2:] in ['L3', 'L2', 'M5', 'M4']:
                        area = np.sqrt(2 * np.pi) * peak['amplitude'] * np.abs(peak['width'] / np.sqrt(2 * np.log(2)))
                        if peak['associated_edge'] not in white_lines:
                            white_lines[peak['associated_edge']] = 0.
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

        flat_list = [item for sublist in self.peak_out_list for item in sublist]
        new_list = np.reshape(flat_list, [len(flat_list) // 3, 3])
        arg_list = np.argsort(np.abs(new_list[:, 1]))

        self.ui.peak_list = []
        self.peaks['peaks'] = {}
        for i in range(number_of_peaks):
            self.ui.peak_list.append(f'Peak {i+1}')
            p = new_list[arg_list[-i-1]]
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

    def on_list_enter(self):
        # self.setWindowTitle('list')
        self.update()

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

