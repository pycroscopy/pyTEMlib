"""
    EELS Input Dialog for ELNES Analysis
"""
from os import error
Qt_available = True
try:
    from PyQt5 import QtCore,  QtWidgets
except:
    Qt_available = False
    # print('Qt dialogs are not available')

import numpy as np
import scipy
import scipy.optimize
import scipy.signal

import ipywidgets
from IPython.display import display
import matplotlib
import matplotlib.pylab as plt
import matplotlib.patches as patches

import sidpy
import pyTEMlib.file_tools as ft
from pyTEMlib import  eels_tools
from pyTEMlib import peak_dlg
from pyTEMlib import eels_dialog_utilities

advanced_present = True
try:
    import advanced_eels_tools
    print('advanced EELS features enabled')
except ModuleNotFoundError:
    advanced_present = False

_version = .001

def get_sidebar():
    side_bar = ipywidgets.GridspecLayout(16, 3, width='auto', grid_gap="0px")
    row = 0
    side_bar[row, :3] = ipywidgets.Button(description='Fit Area',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5,description='Fit Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Fit End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Peak Finding',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))

    row += 1
    
    
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4)],
            value=0,
            description='Peaks:',
            disabled=False,
            layout=ipywidgets.Layout(width='200px'))
    
    side_bar[row, 2] = ipywidgets.Button(
                                    description='Smooth',
                                    disabled=False,
                                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltip='Do Gaussian Mixing',
                                    layout=ipywidgets.Layout(width='100px'))
 
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Number:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.Button(
                                    description='Find',
                                    disabled=False,
                                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                    tooltip='Find first peaks from Gaussian mixture',
                                    layout=ipywidgets.Layout(width='100px'))
    
    row += 1
    
    side_bar[row, :3] = ipywidgets.Button(description='Peaks',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[('Peak 1', 0), ('add peak', -1)],
            value=0,
            description='Peaks:',
            disabled=False,
            layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[ 'Gauss', 'Lorentzian', 'Drude', 'Zero-Loss'],
            value='Gauss',
            description='Symmetry:',
            disabled=False,
            layout=ipywidgets.Layout(width='200px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Position:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Amplitude:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Width FWHM:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Asymmetry:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="a.u.", layout=ipywidgets.Layout(width='100px'))
    row += 1
    
    side_bar[row, :3] = ipywidgets.Button(description='White-Line',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Ratio:',
            disabled=False,
            layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value=" ", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description= 'Sum:',
            disabled=False,
            layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value=" ", layout=ipywidgets.Layout(width='100px'))
    return side_bar

class PeakFitWidget(object):
    def __init__(self, datasets, key):
        self.datasets = datasets
        if not isinstance(datasets, dict):
            raise TypeError('need dictioary of sidpy datasets')
            
        self.sidebar = get_sidebar()
        self.key = key
        self.dataset = datasets[self.key]
        if not isinstance(self.dataset, sidpy.Dataset):
            raise TypeError('dataset or first item inhas to be a sidpy dataset')
        
        self.model = np.array([])
        self.y_scale = 1.0
        self.change_y_scale = 1.0
        self.spectrum_ll = None
        self.low_loss_key = None

        self.peaks = {}

        self.show_regions = False
            
        self.set_dataset()
                                      
        self.app_layout = ipywidgets.AppLayout(
            left_sidebar=self.sidebar,
            center=self.view.panel,
            footer=None,#message_bar,
            pane_heights=[0, 10, 0],
            pane_widths=[4, 10, 0],
        )
        display(self.app_layout)
        self.set_action()
        
    def line_select_callback(self, x_min, x_max):
            self.start_cursor.value = np.round(x_min,3)
            self.end_cursor.value = np.round(x_max, 3)
            self.start_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.start_cursor.value)
            self.end_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.end_cursor.value)
            
            
    def set_peak_list(self):
        self.peak_list = []
        if 'peaks' not in self.peaks:
            self.peaks['peaks'] = {}
        key = 0
        for key in self.peaks['peaks']:
            if key.isdigit():
                self.peak_list.append((f'Peak {int(key) + 1}', int(key)))
        self.peak_list.append(('add peak', -1))
        #self.sidebar[7, 0].options = self.peak_list
        #self.sidebar[7, 0].value = 0


    def plot(self, scale=True):
        
        self.view.change_y_scale = self.change_y_scale
        self.view.y_scale = self.y_scale
        self.energy_scale = self.dataset.energy_loss.values
        
        if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
            spectrum = self.dataset.view.get_spectrum()
        else:
            spectrum = self.dataset
        #if 'features' in self.peaks:
        if 'resolution_function' in self.datasets.keys():
            
            zl = self.datasets['resolution_function']  # self.peaks['features']]
        additional_spectra = {}
        if len(self.model) > 1:
            additional_spectra = {'model': self.model,
                                  'difference': spectrum-self.model,
                                  'zero_loss': self.datasets['resolution_function']}   
        else:
            additional_spectra = {}
        if 'peaks' in self.peaks:
            if len(self.peaks)>0:
                for index, peak in self.peaks['peaks'].items(): # ll
                    p = [peak['position'], peak['amplitude'], peak['width']]
                    additional_spectra[f'peak {index}']= gauss(np.array(self.energy_scale), p)
        self.view.plot(scale=True, additional_spectra=additional_spectra )
        self.change_y_scale = 1.
    
        self.view.figure.canvas.draw_idle()
      
        
    def set_dataset(self, index=0):    
        self.spec_dim = ft.get_dimensions_by_type('spectral', self.dataset)
        if len(self.spec_dim) != 1:
            raise TypeError('We need exactly one SPECTRAL dimension')
        self.spec_dim = self.spec_dim[0]        
        self.energy_scale = self.spec_dim[1]
        
        self.y_scale = 1.0
        self.change_y_scale = 1.0
        
        if 'peak_fit' not in self.dataset.metadata:
            self.dataset.metadata['peak_fit'] = {}
            if 'edges' in self.dataset.metadata:
                if 'fit_area' in self.dataset.metadata['edges']:
                    self.dataset.metadata['peak_fit']['fit_start'] = self.dataset.metadata['edges']['fit_area']['fit_start']
                    self.dataset.metadata['peak_fit']['fit_end'] = self.dataset.metadata['edges']['fit_area']['fit_end']
                self.dataset.metadata['peak_fit']['peaks'] = {'0': {'position': self.energy_scale[1],
                                                                    'amplitude': 1000.0, 'width': 1.0,
                                                                    'type': 'Gauss', 'asymmetry': 0}}

        self.peaks = self.dataset.metadata['peak_fit']
        if 'fit_start' not in self.peaks:
            self.peaks['fit_start'] = self.energy_scale[1]
        if 'fit_end' not in self.peaks:
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
        if self.dataset.data_type.name =='SPECTRAL_IMAGE':
            self.view = eels_dialog_utilities.SIPlot(self.dataset)
        else:
            self.view = eels_dialog_utilities.SpectrumPlot(self.dataset)
        self.dataset.view = self.view
        #self.view.legend(loc='Upper Right')
        self.y_scale = 1.0
        self.change_y_scale = 1.0
                
    def set_fit_area(self, value):
       
        self.peaks['fit_start'] = self.sidebar[1, 0].value 
        self.peaks['fit_end'] = self.sidebar[2, 0].value 
        
        self.plot()
        
    def set_y_scale(self, value):  
        self.change_y_scale = 1/self.y_scale
        if self.sidebar[12, 0].value:
            dispersion = self.energy_scale[1] - self.energy_scale[0]
            self.y_scale = 1/self.dataset.metadata['experiment']['flux_ppm'] * dispersion
        else:
            self.y_scale = 1.0
            
        self.change_y_scale *= self.y_scale
        self.update()
        self.plot()
        
    def update(self, index=0):
       
        # self.setWindowTitle('update')
        self.sidebar[1, 0].value = self.peaks['fit_start']
        self.sidebar[2, 0].value = self.peaks['fit_end']

        peak_index = self.sidebar[7, 0].value
        self.peak_index = self.sidebar[7, 0].value
        if str(peak_index) not in self.peaks['peaks']:
            self.peaks['peaks'][str(peak_index)] = {'position': self.energy_scale[1], 'amplitude': 1000.0,
                                                    'width': 1.0, 'type': 'Gauss', 'asymmetry': 0}
        self.sidebar[8, 0].value = self.peaks['peaks'][str(peak_index)]['type']
        if 'associated_edge' in self.peaks['peaks'][str(peak_index)]:
            self.sidebar[7, 2].value = (self.peaks['peaks'][str(peak_index)]['associated_edge'])
        else:
            self.sidebar[7, 2].value = ''
        self.sidebar[9, 0].value = self.peaks['peaks'][str(peak_index)]['position']
        self.sidebar[10, 0].value = self.peaks['peaks'][str(peak_index)]['amplitude']
        self.sidebar[11, 0].value = self.peaks['peaks'][str(peak_index)]['width']
        if 'asymmetry' not in self.peaks['peaks'][str(peak_index)]:
            self.peaks['peaks'][str(peak_index)]['asymmetry'] = 0.
        self.sidebar[12, 0].value = self.peaks['peaks'][str(peak_index)]['asymmetry']

    
    def get_input(self):
        p_in = []
        for key, peak in self.peaks['peaks'].items():
            if key.isdigit():
                p_in.append(peak['position'])
                p_in.append(peak['amplitude'])
                p_in.append(peak['width'])
        return p_in

       
    def fit_peaks(self, value=0):
        """Fit spectrum with peaks given in peaks dictionary"""
        # print('Fitting peaks...')
        
        if self.dataset.data_type.name == 'SPECTRUM':
            spectrum = np.array(self.dataset)
        else:
            spectrum = self.dataset.view.get_spectrum()
        spectrum -= spectrum.min() - 1
        # set the energy scale and fit start and end points
        energy_scale = np.array(self.energy_scale)
        start_channel = np.searchsorted(energy_scale, self.peaks['fit_start'])
        end_channel = np.searchsorted(energy_scale, self.peaks['fit_end'])

        energy_scale = self.energy_scale[start_channel:end_channel]
        # select the core loss model if it exists. Otherwise, we will fit to the full spectrum.
        if 'model' in self.dataset.metadata:
            model = self.dataset.metadata['model'][start_channel:end_channel]
        elif self.core_loss:
            # print('Core loss model found. Fitting on top of the model.')
            model = self.dataset.metadata['edges']['model']['spectrum'][start_channel:end_channel]
        else:
            
            # print('No core loss model found. Fitting to the full spectrum.')
            model = np.zeros(end_channel - start_channel)

        # if we have a core loss model we will only fit the difference between the model and the data.
        difference = np.array(spectrum[start_channel:end_channel] - model)
        p_in = self.get_input()
        # find the optimum fitting parameters
        #[self.p_out, _] = scipy.optimize.leastsq(eels_tools.residuals_smooth, np.array(p_in), ftol=1e-3,
        #                                            args=(energy_scale, difference, False))

        [self.p_out, _] = scipy.optimize.leastsq(eels_tools.residuals3, np.array(p_in, dtype=np.float64),
                                                    args=(energy_scale, difference)  ) # , False))
        # construct the fit data from the optimized parameters
        #self.peak_model = np.zeros(len(self.energy_scale))
        #self.model = np.zeros(len(self.energy_scale))
        #self.model[start_channel:end_channel] = model
        #fit = eels_tools.model_smooth(energy_scale, self.p_out, False)
        fit = eels_tools.gmm(energy_scale, self.p_out)  # , False)
        self.peak_model = fit
        
        #self.peak_model[start_channel:end_channel] = fit
        #self.dataset.metadata['peak_fit']['edge_model'] = self.model
        #self.model = self.model + self.peak_model
        #self.dataset.metadata['peak_fit']['peak_model'] = self.peak_model

        for key, peak in self.peaks['peaks'].items():
            if key.isdigit():
                p_index = int(key)*3
                self.peaks['peaks'][key] = {'position': self.p_out[p_index],
                                            'amplitude': self.p_out[p_index+1],
                                            'width': self.p_out[p_index+2],
                                            'type': 'Gauss',
                                            'associated_edge': ''}

        eels_tools.find_associated_edges(self.dataset)
        self.find_white_lines()
        self.update()
        self.plot()
    
   

    def find_white_lines(self):
        eels_tools.find_white_lines(self.dataset)
        self.wl_list = []
        self.wls_list = []
        if 'white_line_ratios' in self.dataset.metadata['peak_fit']:
            if len(self.dataset.metadata['peak_fit']['white_line_ratios']) > 0:
                for key in self.dataset.metadata['peak_fit']['white_line_ratios']:
                    self.wl_list.append(key)
                for key in self.dataset.metadata['peak_fit']['white_line_sums']:
                    self.wls_list.append(key)

                self.sidebar[14, 0].options = self.wl_list
                self.sidebar[14, 0].value = self.wl_list[0]
                self.sidebar[14, 2].value = f"{self.dataset.metadata['peak_fit']['white_line_ratios'][self.wl_list[0]]:.2f}"
                
                self.sidebar[15, 0].options = self.wls_list
                self.sidebar[15, 0].value = self.wls_list[0]
                self.sidebar[15, 2].value = f"{self.dataset.metadata['peak_fit']['white_line_sums'][self.wls_list[0]]*1e6:.4f} ppm"

            else:
                self.wl_list.append('Ratio')
                self.wls_list.append('Sum')

                self.sidebar[14, 0].options = ['None']
                self.sidebar[14, 0].value = 'None'
                self.sidebar[14, 2].value = ' '
                
                self.sidebar[15, 0].options = ['None']
                self.sidebar[15, 0].value = 'None'
                self.sidebar[15, 2].value = ' '

    def find_peaks(self, value=0):
        number_of_peaks = int(self.sidebar[5, 0].value)
        if number_of_peaks > len(self.peak_out_list):
            number_of_peaks = len(self.peak_out_list)
            self.sidebar[5, 0].value = str(len(self.peak_out_list))
        self.peak_list = []
        self.peaks['peaks'] = {}
        new_number_of_peaks = 0

        peaks, prop = scipy.signal.find_peaks(self.peak_model, width=5)
        print(len(peaks), number_of_peaks, len(peaks)>= number_of_peaks)
        if len(peaks) >= number_of_peaks:
            if self.dataset.data_type.name == 'SPECTRUM':
                spectrum = np.array(self.dataset)
            else:
                spectrum = self.dataset.view.get_spectrum()
            for i in range(number_of_peaks):
                self.peak_list.append((f'Peak {i+1}', i))
                p = [self.energy_scale[peaks[i]], np.float32(spectrum[peaks[i]]), np.sqrt(prop['widths'][i])]
                if p[1]>0:
                    self.peaks['peaks'][str(new_number_of_peaks)] = {'position': p[0], 'amplitude': p[1], 'width': p[2], 'type': 'Gauss',
                                            'asymmetry': 0}
                    new_number_of_peaks += 1
        else:
            for i in range(number_of_peaks):
                self.peak_list.append((f'Peak {i+1}', i))
                p = self.peak_out_list[i]
                if p[1]>0:
                    self.peaks['peaks'][str(new_number_of_peaks)] = {'position': p[0], 'amplitude': p[1], 'width': p[2], 'type': 'Gauss',
                                            'asymmetry': 0}
                    new_number_of_peaks += 1
        self.sidebar[5, 0].value = str(new_number_of_peaks)
        self.peak_list.append((f'add peak', -1))
        
        self.sidebar[7, 0].options = self.peak_list
        self.sidebar[7, 0].value = 0

        #eels_tools.find_associated_edges(self.dataset)
        #self.find_white_lines()

        self.update()
        self.plot()

    def smooth(self, value=0):
        """Fit lots of Gaussian to spectrum and let the program sort it out

        We sort the peaks by area under the Gaussians, assuming that small areas mean noise.

        """
        iterations = self.sidebar[4, 0].value
        self.sidebar[5, 0].value =  0
        
        if self.key == self.datasets['_relationship']['low_loss']:
           if 'resolution_function' in self.datasets['_relationship'].keys():
               self.model = np.array(self.datasets['resolution_function'])


        self.peak_model, self.peak_out_list, number_of_peaks = smooth(self.dataset-self.model, iterations, advanced_present)

        spec_dim = ft.get_dimensions_by_type('SPECTRAL', self.dataset)[0]
        if spec_dim[1][0] > 0:
            self.model = self.dataset.metadata['edges']['model']['spectrum']
        elif 'model' in self.dataset.metadata:
            self.model = self.dataset.metadata['model']
        else:
            self.model = np.zeros(len(spec_dim[1]))

        self.dataset.metadata['peak_fit']['edge_model'] = self.model
        self.model = self.model + self.peak_model
        self.dataset.metadata['peak_fit']['peak_model'] = self.peak_model
        self.dataset.metadata['peak_fit']['peak_out_list'] = self.peak_out_list
        
        peaks, prop = scipy.signal.find_peaks(self.peak_model, width=5)

        self.sidebar[5, 0].value = str(len(peaks))
        self.update()
        self.plot()
        
    def make_model(self):
        p_peaks = []
        for key, peak in self.peaks['peaks'].items():
            if key.isdigit():
                p_peaks.append(peak['position'])
                p_peaks.append(peak['amplitude'])
                p_peaks.append(peak['width'])

        
        # set the energy scale and fit start and end points
        energy_scale = np.array(self.energy_scale)
        start_channel = np.searchsorted(energy_scale, self.peaks['fit_start'])
        end_channel = np.searchsorted(energy_scale, self.peaks['fit_end'])
        energy_scale = self.energy_scale # [start_channel:end_channel]
        # select the core loss model if it exists. Otherwise, we will fit to the full spectrum.
        
        p_peaks = np.array(p_peaks, dtype=np.float64)
        
        fit = eels_tools.gmm(energy_scale, p_peaks)  # , False)
        self.peak_model = fit
        #self.peak_model[start_channel:end_channel] = fit
        """if 'edge_model' in self.dataset.metadata['peak_fit']:
            self.model = self.dataset.metadata['peak_fit']['edge_model'] + self.peak_model
        else:
            self.model = np.zeros(self.dataset.shape)
        """
        self.model = fit

    def modify_peak_position(self, value=-1):
        peak_index = self.sidebar[7, 0].value
        self.peaks['peaks'][str(peak_index)]['position'] = self.sidebar[9,0].value
        self.make_model()
        self.plot()

    def modify_peak_amplitude(self, value=-1):
        peak_index = self.sidebar[7, 0].value
        self.peaks['peaks'][str(peak_index)]['amplitude'] = self.sidebar[10,0].value
        self.make_model()
        self.plot()
    
    def modify_peak_width(self, value=-1):
        peak_index = self.sidebar[7, 0].value
        self.peaks['peaks'][str(peak_index)]['width'] = self.sidebar[11,0].value
        self.make_model()
        self.plot()

    def peak_selection(self, change=None):
        options = list(self.sidebar[7,0].options)
            
        if self.sidebar[7, 0].value < 0:
            options.insert(-1, (f'Peak {len(options)}', len(options)-1))
            self.sidebar[7, 0].value = 0
            self.sidebar[7,0].options = options 
            self.sidebar[7, 0].value = int(len(options)-2)
            
        self.update()
    
    def set_action(self):
        self.sidebar[1, 0].observe(self.set_fit_area, names='value')
        self.sidebar[2, 0].observe(self.set_fit_area, names='value')
        
        self.sidebar[4, 2].on_click(self.smooth)
        self.sidebar[7,0].observe(self.peak_selection)
        self.sidebar[5,2].on_click(self.find_peaks)
        
        self.sidebar[6, 0].on_click(self.fit_peaks)
        self.sidebar[9, 0].observe(self.modify_peak_position, names='value')
        self.sidebar[10, 0].observe(self.modify_peak_amplitude, names='value')
        self.sidebar[11, 0].observe(self.modify_peak_width, names='value')
        



if Qt_available:
    class PeakFitDialog(QtWidgets.QDialog):
        """
        EELS Input Dialog for ELNES Analysis
        """

        def __init__(self, datasets=None):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            
            if datasets is None:
                # make a dummy dataset
                datasets = ft.make_dummy_dataset('spectrum')
            if not isinstance(datasets, dict):
                datasets= {'Channel_000': datasets}

            self.dataset = datasets[list(datasets.keys())[0]]
            self.datasets = datasets
            # Create an instance of the GUI
            if 'low_loss' in self.dataset.metadata:
                mode = 'low_loss'
            else:
                mode = 'core_loss'

            self.ui = peak_dlg.UiDialog(self, mode=mode)

            self.set_action()

            self.energy_scale = np.array([])
            self.peak_out_list = []
            self.p_out = []
            self.axis = None
            self.show_regions = False
            self.show()

            

            if not isinstance(self.dataset, sidpy.Dataset):
                raise TypeError('dataset has to be a sidpy dataset')
            self.spec_dim = ft.get_dimensions_by_type('spectral', self.dataset)
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

            if 'low_loss' in self.dataset.metadata:
                self.ui.iteration_list = ['0']

            
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
                name = 's'
                if 'zero_loss' in self.dataset.metadata:
                    x = self.dataset.view.x
                    y = self.dataset.view.y
                    self.energy_scale -= self.dataset.metadata['zero_loss']['shifts'][x, y]
                    name = f"shift { self.dataset.metadata['zero_loss']['shifts'][x, y]:.3f}"
                    self.setWindowTitle(f'plot {x}')
            else:
                spectrum = np.array(self.dataset)
                self.axis = self.dataset.view.axis

            x_limit = self.axis.get_xlim()
            y_limit = self.axis.get_ylim()
            self.axis.clear()

            self.axis.plot(self.energy_scale, spectrum, label='spectrum')
            #if 'features' in self.peaks:
            zl = self.datasets[self.peaks['features']]
            self.axis.plot(self.energy_scale, zl, label='zero_loss')

            if len(self.model) > 1:
                self.axis.plot(self.energy_scale, self.model, label='model')
                self.axis.plot(self.energy_scale, spectrum - self.model, label='difference')
                #self.axis.plot(self.energy_scale, (spectrum - self.model) / np.sqrt(spectrum), label='Poisson')
            
            self.axis.set_xlim(x_limit)
            self.axis.set_ylim(y_limit)
            
            for index, peak in self.peaks['peaks'].items():
                p = [peak['position'], peak['amplitude'], peak['width']]
                self.axis.plot(self.energy_scale, eels_tools.gauss(self.energy_scale, p))
            self.axis.legend(loc="upper right")
            self.axis.figure.canvas.draw_idle()
            
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
            spectrum -= spectrum.min()-1
            # set the energy scale and fit start and end points
            energy_scale = np.array(self.energy_scale)
            """start_channel = np.searchsorted(energy_scale, self.peaks['fit_start'])
            end_channel = np.searchsorted(energy_scale, self.peaks['fit_end'])

            energy_scale = self.energy_scale[start_channel:end_channel]
            # select the core loss model if it exists. Otherwise, we will fit to the full spectrum.
            if 'model' in self.dataset.metadata:
                model = self.dataset.metadata['model'][start_channel:end_channel]
            elif self.core_loss:
                print('Core loss model found. Fitting on top of the model.')
                model = self.dataset.metadata['edges']['model']['spectrum'][start_channel:end_channel]
            else:
                print('No core loss model found. Fitting to the full spectrum.')
                model = np.zeros(end_channel - start_channel)

            # if we have a core loss model we will only fit the difference between the model and the data.
            
            
            difference = np.array(spectrum[start_channel:end_channel] - model)
            """
            difference = spectrum
            if self.key == self.datasets['_relationships']['low_loss']:
                if 'resolution_function' in self.datasets['_relationships'].keys():
                    difference -= np.array(self.datasets['_relationships']['resolution_function'])
                    self.peaks['peaks']['features'] = 'resolution_function'
                    self.model = np.array(self.datasets['_relationships']['resolution_function'])

            # find the optimum fitting parameters
            [self.p_out, _] = scipy.optimize.leastsq(eels_tools.residuals3, np.array(p_in), ftol=1e-3,
                                                     args=(energy_scale, difference, False))

            # construct the fit data from the optimized parameters
            #self.peak_model = np.zeros(len(self.energy_scale))
            #self.model = np.zeros(len(self.energy_scale))
            #self.model[start_channel:end_channel] = model
            fit = eels_tools.gmm(energy_scale, self.p_out, False)
            self.peak_model = fit
            #self.peak_model[start_channel:end_channel] = fit
            #self.dataset.metadata['peak_fit']['edge_model'] = self.model
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
            if 'edges' in self.dataset.metadata:
                if 'model' in  self.dataset.metadata['edges']:
                    self.dataset.metadata['model'] = self.dataset.metadata['edges']['model']
            if 'resolution_function' in self.datasets:
                self.dataset.metadata['model'] = np.array(self.datasets['resolution_function'])
            iterations = int(self.ui.smooth_list.currentIndex())
            
            if self.key == self.datasets['_relationships']['low_loss']:
                if 'resolution_function' in self.datasets['_relationships'].keys():
                    self.model = np.array(self.datasets['_relationships']['resolution_function'])


            self.peak_model, self.peak_out_list, number_of_peaks = smooth(self.dataset-self.model, iterations, advanced_present)

            spec_dim = ft.get_dimensions_by_type('SPECTRAL', self.dataset)[0]
            if spec_dim[1][0] > 0:
                self.model = self.dataset.metadata['edges']['model']['spectrum']
            elif 'model' in self.dataset.metadata:
                self.model = self.dataset.metadata['model']
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
            eels_tools.find_white_lines(self.dataset)

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
            self.setWindowTitle(f'list {self.sender}, {self.ui.list_model}')
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
            elif self.sender() == self.ui.list_model:
                self.setWindowTitle('list 1')
                if self.sender().currentIndex() == 1:
                    if 'resolution_function' in self.datasets:
                        self.setWindowTitle('list 2')
                        self.dataset.metadata['model'] = np.array(self.datasets['resolution_function'])
                    else:
                        self.ui.list_model.setCurrentIndex(0)
                else:
                    self.ui.list_model.setCurrentIndex(0)
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
            if hasattr(self.ui, 'listwls'):
                self.ui.listwls.activated[str].connect(self.on_list_enter)
                self.ui.listwl.activated[str].connect(self.on_list_enter)
            else:
                self.ui.zl_button.clicked.connect(self.fit_zero_loss)
                self.ui.drude_button.clicked.connect(self.smooth)
                self.ui.list_model.activated[str].connect(self.on_list_enter)

        def fit_zero_loss(self):
            """get shift of spectrum form zero-loss peak position"""
            zero_loss_fit_width=0.3

            energy_scale = self.dataset.energy_loss
            zl_dataset = self.dataset.copy()
            zl_dataset.title = 'resolution_function'
            shifts = np.zeros(self.dataset.shape[0:2])
            zero_p = np.zeros([self.dataset.shape[0],self.dataset.shape[1],6])
            fwhm_p = np.zeros(self.dataset.shape[0:2])
            bin_x = bin_y = 1
            total_spec = int(self.dataset.shape[0]/bin_x)*int(self.dataset.shape[1]/bin_y)
            self.ui.progress.setMaximum(total_spec)
            self.ui.progress.setValue(0)
            zero_loss_fit_width=0.3
            ind = 0
            for x in range(self.dataset.shape[0]):
                for y in range(self.dataset.shape[1]):
                    ind += 1
                    self.ui.progress.setValue(ind)
                    spectrum = self.dataset[x, y, :]
                    fwhm, delta_e = eels_tools.fix_energy_scale(spectrum, energy_scale)
                    z_loss, p_zl = eels_tools.resolution_function(energy_scale - delta_e, spectrum, zero_loss_fit_width)
                    fwhm2, delta_e2 = eels_tools.fix_energy_scale(z_loss, energy_scale - delta_e)
                    shifts[x, y] = delta_e + delta_e2
                    zero_p[x,y,:] = p_zl
                    zl_dataset[x,y] = z_loss
                    fwhm_p[x,y] = fwhm2
            
            zl_dataset.metadata['zero_loss'] = {'parameter': zero_p,
                                                'shifts': shifts,
                                                'fwhm': fwhm_p}
            self.dataset.metadata['zero_loss'] = {'parameter': zero_p,
                                                'shifts': shifts,
                                                'fwhm': fwhm_p}
            
            self.datasets['resolution_function'] = zl_dataset
            self.update()
            self.plot()
                    
            

def smooth(dataset, iterations, advanced_present):
    from pyTEMlib import advanced_eels_tools

    """Gaussian mixture model (non-Bayesian)

    Fit lots of Gaussian to spectrum and let the program sort it out
    We sort the peaks by area under the Gaussians, assuming that small areas mean noise.

    """

    # TODO: add sensitivity to dialog and the two functions below
    #peaks = dataset.metadata['peak_fit']
    
    #peak_model, peak_out_list = eels_tools.find_peaks(dataset, peaks['fit_start'], peaks['fit_end'])
    peak_model, peak_out_list = eels_tools.gaussian_mixture_model(dataset, p_in=None)

    # 
    # if advanced_present and iterations > 1:
    # peak_model, peak_out_list = advanced_eels_tools.smooth(dataset, peaks['fit_start'],
    #                                                       peaks['fit_end'], iterations=iterations)
    # else:
    #    peak_model, peak_out_list = eels_tools.find_peaks(dataset, peaks['fit_start'], peaks['fit_end'])
    #    peak_out_list = [peak_out_list]

    new_list = np.reshape(peak_out_list, [len(peak_out_list) // 3, 3])
    area = np.sqrt(2 * np.pi) * np.abs(new_list[:, 1]) * np.abs(new_list[:, 2] / np.sqrt(2 * np.log(2)))
    arg_list = np.argsort(area)[::-1]
    area = area[arg_list]
    peak_out_list = new_list[arg_list]

    number_of_peaks = np.searchsorted(area * -1, -np.average(area))

    return peak_model, peak_out_list, number_of_peaks


def gauss(x, p):  # p[0]==mean, p[1]= amplitude p[2]==fwhm,
    """Gaussian Function

        p[0]==mean, p[1]= amplitude p[2]==fwhm
        area = np.sqrt(2* np.pi)* p[1] * np.abs(p[2] / 2.3548)
        FWHM = 2 * np.sqrt(2 np.log(2)) * sigma = 2.3548 * sigma
        sigma = FWHM/3548
    """
    if p[2] == 0:
        return x * 0.
    else:
        return p[1] * np.exp(-(x - p[0]) ** 2 / (2.0 * (p[2] / 2.3548) ** 2))
    
