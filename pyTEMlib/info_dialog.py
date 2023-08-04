"""
Input Dialog for EELS Analysis

Author: Gerd Duscher

"""
import numpy as np
import sidpy

Qt_available = True
try:
    from PyQt5 import QtCore,  QtWidgets
except:
    Qt_available = False
    # print('Qt dialogs are not available')


import pyTEMlib.eels_dialog_utilities as ieels
from pyTEMlib.microscope import microscope
import ipywidgets
import matplotlib.pylab as plt
import matplotlib
from IPython.display import display
from pyTEMlib import file_tools as ft
_version = 000


if Qt_available:
    from pyTEMlib import info_dlg
    class InfoDialog(QtWidgets.QDialog):
        """
        Input Dialog for EELS Analysis

        Opens a PyQt5 GUi Dialog that allows to set the experimental parameter necessary for a Quantification.


        The dialog operates on a sidpy dataset
        """

        def __init__(self, datasets=None, key=None):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            # Create an instance of the GUI
            self.ui = info_dlg.UiDialog(self)
            self.set_action()
            self.datasets = datasets

            self.spec_dim = []
            self.energy_scale = np.array([])
            self.experiment = {}
            self.energy_dlg = None
            self.axis = None
            
            self.y_scale = 1.0
            self.change_y_scale = 1.0
            self.show()

            if self.datasets is None:
                # make a dummy dataset for testing
                key = 'Channel_000'
                self.datasets={key: ft.make_dummy_dataset(sidpy.DataType.SPECTRUM)}
            if key is None:
                key = list(self.datasets.keys())[0]
            self.dataset = self.datasets[key]
            self.key = key
            if not isinstance(self.dataset, sidpy.Dataset):
                raise TypeError('dataset has to be a sidpy dataset')

            self.set_dataset(self.dataset)

            view = self.dataset.plot()
            if hasattr(self.dataset.view, 'axes'):
                self.axis = self.dataset.view.axes[-1]
            elif hasattr(self.dataset.view, 'axis'):
                self.axis = self.dataset.view.axis
            self.figure = self.axis.figure
            self.plot()
            self.update()

        def set_dataset(self, dataset):
            self.dataset = dataset
            if not hasattr(self.dataset, '_axes'):
                self.dataset._axes = self.dataset.axes
            if not hasattr(self.dataset, 'meta_data'):
                self.dataset.meta_data = {}

            spec_dim = dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
            if len(spec_dim) != 1:
                raise TypeError('We need exactly one SPECTRAL dimension')
            self.spec_dim = self.dataset._axes[spec_dim[0]]
            self.energy_scale = self.spec_dim.values.copy()

            minimum_info = {'offset': self.energy_scale[0],
                            'dispersion': self.energy_scale[1] - self.energy_scale[0],
                            'exposure_time': 0.0,
                            'convergence_angle': 0.0, 'collection_angle': 0.0,
                            'acceleration_voltage': 100.0, 'binning': 1, 'conversion': 1.0,
                            'flux_ppm': -1.0, 'flux_unit': 'counts', 'current': 1.0, 'SI_bin_x': 1, 'SI_bin_y': 1}
            if 'experiment' not in self.dataset.metadata:
                self.dataset.metadata['experiment'] = minimum_info
            self.experiment = self.dataset.metadata['experiment']

            for key, item in minimum_info.items():
                if key not in self.experiment:
                    self.experiment[key] = item
            self.set_flux_list()

        def set_dimension(self):
            spec_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
            self.spec_dim = self.dataset._axes[spec_dim[0]]
            old_energy_scale = self.spec_dim
            self.dataset.set_dimension(spec_dim[0], sidpy.Dimension(np.array(self.energy_scale),
                                                                    name=old_energy_scale.name,
                                                                    dimension_type=sidpy.DimensionType.SPECTRAL,
                                                                    units='eV',
                                                                    quantity='energy loss'))

        def update(self):

            self.ui.offsetEdit.setText(f"{self.experiment['offset']:.3f}")
            self.ui.dispersionEdit.setText(f"{self.experiment['dispersion']:.3f}")
            self.ui.timeEdit.setText(f"{self.experiment['exposure_time']:.6f}")

            self.ui.convEdit.setText(f"{self.experiment['convergence_angle']:.2f}")
            self.ui.collEdit.setText(f"{self.experiment['collection_angle']:.2f}")
            self.ui.E0Edit.setText(f"{self.experiment['acceleration_voltage']/1000.:.2f}")

            self.ui.binningEdit.setText(f"{self.experiment['binning']}")
            self.ui.conversionEdit.setText(f"{self.experiment['conversion']:.2f}")
            self.ui.fluxEdit.setText(f"{self.experiment['flux_ppm']:.2f}")
            self.ui.fluxUnit.setText(f"{self.experiment['flux_unit']}")
            self.ui.VOAEdit.setText(f"{self.experiment['current']:.2f}")
            self.ui.statusBar.showMessage('Message in statusbar.')

        def on_enter(self):
            sender = self.sender()

            if sender == self.ui.offsetEdit:
                value = float(str(sender.displayText()).strip())
                self.experiment['offset'] = value
                sender.setText(f"{value:.2f}")
                self.energy_scale = self.energy_scale - self.energy_scale[0] + value
                self.set_dimension()
                self.plot()
            elif sender == self.ui.dispersionEdit:
                value = float(str(sender.displayText()).strip())
                self.experiment['dispersion'] = value
                self.energy_scale = np.arange(len(self.energy_scale)) * value + self.energy_scale[0]
                self.set_dimension()
                self.plot()
                sender.setText(f"{value:.3f}")
            elif sender == self.ui.timeEdit:
                value = float(str(sender.displayText()).strip())
                self.experiment['exposure_time'] = value
                sender.setText(f"{value:.2f}")
            elif sender == self.ui.convEdit:
                value = float(str(sender.displayText()).strip())
                self.experiment['convergence_angle'] = value
                sender.setText(f"{value:.2f}")
            elif sender == self.ui.collEdit:
                value = float(str(sender.displayText()).strip())
                self.experiment['collection_angle'] = value
                sender.setText(f"{value:.2f}")
            elif sender == self.ui.E0Edit:
                value = float(str(sender.displayText()).strip())
                self.experiment['acceleration_voltage'] = value*1000.0
                sender.setText(f"{value:.2f}")
            elif sender == self.ui.fluxEdit:
                value = float(str(sender.displayText()).strip())
                if value == 0:
                    self.set_flux()
                else:
                    self.experiment['flux_ppm'] = value
                    sender.setText(f"{value:.2f}")
            elif sender == self.ui.binXEdit or sender == self.ui.binYEdit:
                if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                    bin_x = int(self.ui.binXEdit.displayText())
                    bin_y = int(self.ui.binYEdit.displayText())
                    self.experiment['SI_bin_x'] = bin_x
                    self.experiment['SI_bin_y'] = bin_y
                    self.dataset.view.set_bin([bin_x, bin_y])
                    self.ui.binXEdit.setText(str(self.dataset.view.bin_x))
                    self.ui.binYEdit.setText(str(self.dataset.view.bin_y))
            else:
                print('not supported yet')

        def plot(self):
            if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                spectrum = self.dataset.view.get_spectrum()
                self.axis = self.dataset.view.axes[1]
            else:
                spectrum = np.array(self.dataset)
                self.axis = self.dataset.view.axis

            spectrum *= self.y_scale

            x_limit = self.axis.get_xlim()
            y_limit = np.array(self.axis.get_ylim())
            self.axis.clear()
            

            self.axis.plot(self.energy_scale, spectrum, label='spectrum')
            self.axis.set_xlim(x_limit)
            if self.change_y_scale !=1.0:
                y_limit *= self.change_y_scale
                self.change_y_scale = 1.0
            self.axis.set_ylim(y_limit)
            
            if self.y_scale != 1.:
                self.axis.set_ylabel('scattering intensity (ppm)')
                
            self.axis.set_xlabel('energy_loss (eV)')

            self.figure.canvas.draw_idle()

        def on_list_enter(self):
            sender = self.sender()
            if sender == self.ui.TEMList:
                microscope.set_microscope(self.ui.TEMList.currentText())
                self.experiment['microscope'] = microscope.name
                self.experiment['convergence_angle'] = microscope.alpha
                self.experiment['collection_angle'] = microscope.beta
                self.experiment['acceleration_voltage'] = microscope.E0
                self.update()

        def set_energy_scale(self):
            self.energy_dlg = ieels.EnergySelector(self.dataset)

            self.energy_dlg.signal_selected[bool].connect(self.set_energy)
            self.energy_dlg.show()

        def set_energy(self, k):
            spec_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
            self.spec_dim = self.dataset._axes[spec_dim[0]]

            self.energy_scale = self.spec_dim.values
            self.experiment['offset'] = self.energy_scale[0]
            self.experiment['dispersion'] = self.energy_scale[1] - self.energy_scale[0]
            self.update()

        def set_flux(self, key):
            self.ui.statusBar.showMessage('on_set_flux')
            new_flux = 1.0
            title = key
            metadata = {}
            if key in self.datasets.keys():
                flux_dataset = self.datasets[key]
                if isinstance(flux_dataset, sidpy.Dataset):
                    exposure_time = -1.0
                    flux_dataset = self.datasets[key]
                    if flux_dataset.data_type.name == 'IMAGE' or 'SPECTRUM' in flux_dataset.data_type.name:
                        if 'exposure_time' in flux_dataset.metadata['experiment']:
                            if 'number_of_frames' in flux_dataset.metadata['experiment']:
                                exposure_time = flux_dataset.metadata['experiment']['single_exposure_time'] * flux_dataset.metadata['experiment']['number_of_frames'] 
                            else:
                                exposure_time = flux_dataset.metadata['experiment']['exposure_time']
                        else:
                            exposure_time = -1.0
                            flux_dataset.metadata['experiment']['exposure_time'] = -1
                            print('Did not find exposure time assume 1s')
                        if exposure_time > 0:
                            new_flux  = np.sum(np.array(flux_dataset*1e-6))/exposure_time*self.dataset.metadata['experiment']['exposure_time']
                            title = flux_dataset.title
                            metadata = flux_dataset.metadata
                            self.experiment['flux_ppm'] = new_flux
                            self.experiment['flux_units'] = 'Mcounts '
                            self.experiment['flux_source'] = title
                            self.experiment['flux_metadata'] = metadata
    
                            self.update()

        def on_check(self):
            sender = self.sender()
        
            if sender.objectName() == 'probability':
                dispersion = self.energy_scale[1]-self.energy_scale[0]
                if sender.isChecked():
                    self.y_scale = 1/self.experiment['flux_ppm']*dispersion
                    self.change_y_scale = 1/self.experiment['flux_ppm']*dispersion
                else:
                    self.y_scale = 1.
                    self.change_y_scale = self.experiment['flux_ppm']/dispersion
                self.plot()

        def set_flux_list(self):
            length_list = self.ui.select_flux.count()+1
            for i in range(2, length_list):
                self.ui.select_flux.removeItem(i)
            for key in self.datasets.keys():
                if isinstance(self.datasets[key], sidpy.Dataset):
                    if self.datasets[key].title != self.dataset.title:
                        self.ui.select_flux.addItem(key+': '+self.datasets[key].title)

        def on_list_enter(self):
            self.ui.statusBar.showMessage('on_list')
            sender = self.sender()
            if sender.objectName() == 'select_flux_list':
                self.ui.statusBar.showMessage('list')
                index = self.ui.select_flux.currentIndex()
                self.ui.statusBar.showMessage('list'+str(index))
                if index == 1:
                    ft.add_dataset_from_file(self.datasets, key_name='Reference')
                    self.set_flux_list()
                else:
                    key = str(self.ui.select_flux.currentText()).split(':')[0]
                    self.set_flux(key)
                    
                self.update()

        def set_action(self):
            self.ui.statusBar.showMessage('action')
            self.ui.offsetEdit.editingFinished.connect(self.on_enter)
            self.ui.dispersionEdit.editingFinished.connect(self.on_enter)
            self.ui.timeEdit.editingFinished.connect(self.on_enter)

            self.ui.TEMList.activated[str].connect(self.on_list_enter)

            self.ui.convEdit.editingFinished.connect(self.on_enter)
            self.ui.collEdit.editingFinished.connect(self.on_enter)
            self.ui.E0Edit.editingFinished.connect(self.on_enter)
            self.ui.binningEdit.editingFinished.connect(self.on_enter)
            self.ui.conversionEdit.editingFinished.connect(self.on_enter)
            self.ui.fluxEdit.editingFinished.connect(self.on_enter)
            self.ui.VOAEdit.editingFinished.connect(self.on_enter)
            self.ui.energy_button.clicked.connect(self.set_energy_scale)
            self.ui.select_flux.activated[str].connect(self.on_list_enter)

            self.ui.check_probability.clicked.connect(self.on_check)
            
            self.ui.binXEdit.editingFinished.connect(self.on_enter)
            self.ui.binYEdit.editingFinished.connect(self.on_enter)


def get_sidebar():
    side_bar = ipywidgets.GridspecLayout(17, 3,width='auto', grid_gap="0px")

    side_bar[0, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Main Dataset:',
            disabled=False)
    
    row = 1
    side_bar[row, :3] = ipywidgets.Button(description='Energy Scale',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5,description='Offset:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Dispersion:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Microscope',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5,description='Conv.Angle:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="mrad", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Coll.Angle:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="mrad", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Acc Voltage:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="keV", layout=ipywidgets.Layout(width='100px'))
    row += 1

    side_bar[row, :3] = ipywidgets.Button(description='Quantification',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row+=1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Reference:',
            disabled=False)
    side_bar[row,2] = ipywidgets.ToggleButton(
            description='Probability',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Changes y-axis to probability if flux is given',
                     layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Exp_Time:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="s", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5,description='Flux:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="Mcounts", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Conversion:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value=r"e$^-$/counts", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Current:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="pA", layout=ipywidgets.Layout(width='100px') )
    
    row += 1

    side_bar[row, :3] = ipywidgets.Button(description='Spectrum Image',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    
    row += 1
    side_bar[row, :2] = ipywidgets.IntText(value=1, description='bin X:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    row += 1
    side_bar[row, :2] = ipywidgets.IntText(value=1, description='bin X:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    
    for i in range(14, 17):
        side_bar[i, 0].layout.display = "none"
    return side_bar


class SpectrumPlot(sidpy.viz.dataset_viz.CurveVisualizer):
    def __init__(self, dset, spectrum_number=0, figure=None, **kwargs):
        with plt.ioff():
            self.figure = plt.figure()
        self.figure.canvas.toolbar_position = 'right'
        self.figure.canvas.toolbar_visible = True

        super().__init__(dset, spectrum_number=spectrum_number, figure=self.figure, **kwargs)
        
        self.start_cursor = ipywidgets.FloatText(value=0, description='Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
        self.end_cursor = ipywidgets.FloatText(value=0, description='End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
        self.panel = ipywidgets.VBox([ipywidgets.HBox([ipywidgets.Label('',layout=ipywidgets.Layout(width='100px')), ipywidgets.Label('Cursor:'),
                                                       self.start_cursor,ipywidgets.Label('eV'), 
                                                       self.end_cursor, ipywidgets.Label('eV')]),
                                      self.figure.canvas])
        
        self.selector = matplotlib.widgets.SpanSelector(self.axis, self.line_select_callback,
                                         direction="horizontal",
                                         interactive=True,
                                         props=dict(facecolor='blue', alpha=0.2))
        #self.axis.legend()
        display(self.panel)

    def line_select_callback(self, x_min, x_max):
        self.start_cursor.value = np.round(x_min, 3)
        self.end_cursor.value = np.round(x_max, 3)
        self.start_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.start_cursor.value)
        self.end_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.end_cursor.value)

    
class SIPlot(sidpy.viz.dataset_viz.SpectralImageVisualizer):
    def __init__(self, dset, figure=None, horizontal=True, **kwargs):
        if figure is None:
            with plt.ioff():
                self.figure = plt.figure()
        else:
            self.figure = figure
        self.figure.canvas.toolbar_position = 'right'
        self.figure.canvas.toolbar_visible = True

        super().__init__(dset, figure= self.figure, horizontal=horizontal, **kwargs)
        
        self.start_cursor = ipywidgets.FloatText(value=0, description='Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
        self.end_cursor = ipywidgets.FloatText(value=0, description='End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
        self.panel = ipywidgets.VBox([ipywidgets.HBox([ipywidgets.Label('',layout=ipywidgets.Layout(width='100px')), ipywidgets.Label('Cursor:'),
                                                       self.start_cursor,ipywidgets.Label('eV'), 
                                                       self.end_cursor, ipywidgets.Label('eV')]),
                                      self.figure.canvas])
        self.axis = self.axes[-1]
        self.selector = matplotlib.widgets.SpanSelector(self.axis, self.line_select_callback,
                                         direction="horizontal",
                                         interactive=True,
                                         props=dict(facecolor='blue', alpha=0.2))
       
    def line_select_callback(self, x_min, x_max):
        self.start_cursor.value = np.round(x_min, 3)
        self.end_cursor.value = np.round(x_max, 3)
        self.start_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.start_cursor.value)
        self.end_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.end_cursor.value)

    def _update(self, ev=None):

        xlim = self.axes[1].get_xlim()
        ylim = self.axes[1].get_ylim()
        self.axes[1].clear()
        self.get_spectrum()
        if len(self.energy_scale)!=self.spectrum.shape[0]:
            self.spectrum = self.spectrum.T
        self.axes[1].plot(self.energy_scale, self.spectrum.compute(), label='experiment')

        if self.set_title:
            self.axes[1].set_title('spectrum {}, {}'.format(self.x, self.y))
        self.fig.tight_layout()
        self.selector = matplotlib.widgets.SpanSelector(self.axes[1], self.line_select_callback,
                                         direction="horizontal",
                                         interactive=True,
                                         props=dict(facecolor='blue', alpha=0.2))
        
        self.axes[1].set_xlim(xlim)
        self.axes[1].set_ylim(ylim)
        self.axes[1].set_xlabel(self.xlabel)
        self.axes[1].set_ylabel(self.ylabel)

        self.fig.canvas.draw_idle()

class InfoWidget(object):
    def __init__(self, datasets=None):
        self.datasets = datasets
        self.dataset = None

        self.sidebar = get_sidebar()
        
        self.set_dataset()
        self.set_action()
        
                                      
        self.app_layout = ipywidgets.AppLayout(
            left_sidebar=self.sidebar,
            center=self.view.panel,
            footer=None,#message_bar,
            pane_heights=[0, 10, 0],
            pane_widths=[4, 10, 0],
        )
        
        display(self.app_layout)
        
            
    def get_spectrum(self):
        if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
            spectrum = self.dataset.view.get_spectrum()
            self.axis = self.dataset.view.axes[1]
        else:
            spectrum = np.array(self.dataset)
            self.axis = self.dataset.view.axis

        spectrum *= self.y_scale
        return spectrum

    def plot(self, scale=True):
        spectrum = self.get_spectrum()
        self.energy_scale = self.dataset.energy_loss.values
        x_limit = self.axis.get_xlim()
        y_limit = np.array(self.axis.get_ylim())
        """
        self.axis.clear()

        self.axis.plot(self.energy_scale, spectrum, label='spectrum')
                
        
        self.axis.set_xlabel(self.datasets[self.key].labels[0])
        self.axis.set_ylabel(self.datasets[self.key].data_descriptor)
        self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
        if scale:
            self.axis.set_ylim(np.array(y_limit)*self.change_y_scale)
        self.change_y_scale = 1.0
        if self.y_scale != 1.:
                self.axis.set_ylabel('scattering probability (ppm/eV)')
        self.selector = matplotlib.widgets.SpanSelector(self.axis, self.line_select_callback,
                                         direction="horizontal",
                                         interactive=True,
                                         props=dict(facecolor='blue', alpha=0.2))
        self.axis.legend()
        self.figure.canvas.draw_idle()
        """

    def set_dataset(self, index=0):
       
        spectrum_list = []
        reference_list =[('None', -1)]
        dataset_index = self.sidebar[0, 0].value
        for index, key in enumerate(self.datasets.keys()):
            if 'Reference' not in key:
                if 'SPECTR' in self.datasets[key].data_type.name:
                    spectrum_list.append((f'{key}: {self.datasets[key].title}', index)) 
            reference_list.append((f'{key}: {self.datasets[key].title}', index))
       
        self.sidebar[0,0].options = spectrum_list
        self.sidebar[9,0].options = reference_list
        self.key = list(self.datasets)[dataset_index]
        self.dataset = self.datasets[self.key]
        if 'SPECTRUM' in self.dataset.data_type.name:
           for i in range(14, 17):
                self.sidebar[i, 0].layout.display = "none"
        else:
            for i in range(14, 17):
                self.sidebar[i, 0].layout.display = "flex"
        #self.sidebar[0,0].value = dataset_index #f'{self.key}: {self.datasets[self.key].title}'
        self.sidebar[2,0].value = np.round(self.datasets[self.key].energy_loss[0], 3)  
        self.sidebar[3,0].value = np.round(self.datasets[self.key].energy_loss[1] - self.datasets[self.key].energy_loss[0], 4)  
        self.sidebar[5,0].value = np.round(self.datasets[self.key].metadata['experiment']['convergence_angle'], 1)  
        self.sidebar[6,0].value = np.round(self.datasets[self.key].metadata['experiment']['collection_angle'], 1)
        self.sidebar[7,0].value = np.round(self.datasets[self.key].metadata['experiment']['acceleration_voltage']/1000, 1)
        self.sidebar[10,0].value = np.round(self.datasets[self.key].metadata['experiment']['exposure_time'], 4)
        if 'flux_ppm' not in self.datasets[self.key].metadata['experiment']:
            self.datasets[self.key].metadata['experiment']['flux_ppm'] = 0
        self.sidebar[11,0].value = self.datasets[self.key].metadata['experiment']['flux_ppm']
        if 'count_conversion' not in self.datasets[self.key].metadata['experiment']:
            self.datasets[self.key].metadata['experiment']['count_conversion'] = 1
        self.sidebar[12,0].value = self.datasets[self.key].metadata['experiment']['count_conversion']
        if 'beam_current' not in self.datasets[self.key].metadata['experiment']:
            self.datasets[self.key].metadata['experiment']['beam_current'] = 0
        self.sidebar[13,0].value = self.datasets[self.key].metadata['experiment']['beam_current']
        
        self.view = SIPlot(self.dataset)
        
        self.y_scale = 1.0
        self.change_y_scale = 1.0
        
        
    def cursor2energy_scale(self, value):
       
        dispersion = (self.end_cursor.value - self.start_cursor.value) / (self.end_channel - self.start_channel)
        self.datasets[self.key].energy_loss *= (self.sidebar[3, 0].value/dispersion)
        self.sidebar[3, 0].value = dispersion
        offset = self.start_cursor.value - self.start_channel * dispersion
        self.datasets[self.key].energy_loss += (self.sidebar[2, 0].value-self.datasets[self.key].energy_loss[0])
        self.sidebar[2, 0].value = offset
        self.plot()
        
    def set_energy_scale(self, value):
        dispersion = self.datasets[self.key].energy_loss[1] - self.datasets[self.key].energy_loss[0]
        self.datasets[self.key].energy_loss *= (self.sidebar[3, 0].value/dispersion)
        self.datasets[self.key].energy_loss += (self.sidebar[2, 0].value-self.datasets[self.key].energy_loss[0])
        self.plot()
        
    def set_y_scale(self, value):  
        self.change_y_scale = 1/self.y_scale
        if self.sidebar[9,2].value:
            dispersion = self.datasets[self.key].energy_loss[1] - self.datasets[self.key].energy_loss[0]
            self.y_scale = 1/self.datasets[self.key].metadata['experiment']['flux_ppm'] * dispersion
        else:
            self.y_scale = 1.0
            
        self.change_y_scale *= self.y_scale
        self.plot()
        
    
    def set_flux(self, value):  
        self.datasets[self.key].metadata['experiment']['exposure_time'] = self.sidebar[10,0].value
        if self.sidebar[9,0].value < 0:
            self.datasets[self.key].metadata['experiment']['flux_ppm'] = 0.
        else:
            key = list(self.datasets.keys())[self.sidebar[9,0].value]
            self.datasets[self.key].metadata['experiment']['flux_ppm'] = (np.array(self.datasets[key])*1e-6).sum() / self.datasets[key].metadata['experiment']['exposure_time']
            self.datasets[self.key].metadata['experiment']['flux_ppm'] *= self.datasets[self.key].metadata['experiment']['exposure_time']
        self.sidebar[11,0].value = np.round(self.datasets[self.key].metadata['experiment']['flux_ppm'], 2)
        
    def set_microscope_parameter(self, value):
        self.datasets[self.key].metadata['experiment']['convergence_angle'] = self.sidebar[5,0].value
        self.datasets[self.key].metadata['experiment']['collection_angle'] = self.sidebar[6,0].value
        self.datasets[self.key].metadata['experiment']['acceleration_voltage'] = self.sidebar[7,0].value*1000
    
    def set_binning(self, value):
        if 'SPECTRAL' in self.dataset.data_type.name:
            bin_x = self.sidebar[15,0].value
            bin_y = self.sidebar[16,0].value
            self.dataset.view.set_bin([bin_x, bin_y])
            self.datasets[self.key].metadata['experiment']['SI_bin_x'] = bin_x
            self.datasets[self.key].metadata['experiment']['SI_bin_y'] = bin_y

    def set_action(self):
        self.sidebar[0,0].observe(self.set_dataset)
        self.sidebar[1,0].on_click(self.cursor2energy_scale)
        self.sidebar[2,0].observe(self.set_energy_scale, names='value')
        self.sidebar[3,0].observe(self.set_energy_scale, names='value')
        self.sidebar[5,0].observe(self.set_microscope_parameter)
        self.sidebar[6,0].observe(self.set_microscope_parameter)
        self.sidebar[7,0].observe(self.set_microscope_parameter)
        self.sidebar[9,0].observe(self.set_flux)
        self.sidebar[9,2].observe(self.set_y_scale)
        self.sidebar[10,0].observe(self.set_flux)
        self.sidebar[15,0].observe(self.set_binning)
        self.sidebar[16,0].observe(self.set_binning)
        