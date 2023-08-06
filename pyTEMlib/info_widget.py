import numpy as np
import sidpy


import pyTEMlib.eels_dialog_utilities as ieels
import pyTEMlib.file_tools as ft
from pyTEMlib.microscope import microscope
import ipywidgets
import matplotlib.pylab as plt
import matplotlib

from IPython.display import display


from pyTEMlib import file_tools
from pyTEMlib import eels_tools

def get_info_sidebar():
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

from sidpy.io.interface_utils import open_file_dialog
class EELSWidget(object):
    def __init__(self, datasets, sidebar, tab_title = None):
        
        self.datasets = datasets
        self.dataset = None
        
        if not isinstance(sidebar, list):
            tab = ipywidgets.Tab()
            tab.children = [ft.FileWidget(), sidebar]
            tab.titles = ['Load', 'Info']
        else:
            tab = sidebar

        self.sidebar = sidebar
        with plt.ioff():
            self.figure = plt.figure()
        
        self.figure.canvas.toolbar_position = 'right'
        self.figure.canvas.toolbar_visible = True

        self.start_cursor = ipywidgets.FloatText(value=0, description='Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
        self.end_cursor = ipywidgets.FloatText(value=0, description='End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
        self.panel = ipywidgets.VBox([ipywidgets.HBox([ipywidgets.Label('',layout=ipywidgets.Layout(width='100px')), ipywidgets.Label('Cursor:'),
                                                       self.start_cursor,ipywidgets.Label('eV'), 
                                                       self.end_cursor, ipywidgets.Label('eV')]),
                                      self.figure.canvas])
        
                                      
        self.app_layout = ipywidgets.AppLayout(
            left_sidebar=tab,
            center=self.panel,
            footer=None,#message_bar,
            pane_heights=[0, 10, 0],
            pane_widths=[4, 10, 0],
        )
        self.set_dataset()

        display(self.app_layout)

    def plot(self, scale=True):
        self.figure.clear()
        self.energy_scale = self.dataset.energy_loss.values

        if self.dataset.data_type.name == 'SPECTRUM':
            self.axis = self.figure.subplots(ncols=1)
        else:
            self.plot_spectrum_image() 
            self.axis = self.axes[-1]
        self.spectrum = self.get_spectrum()
        
        self.plot_spectrum()
    
    def plot_spectrum(self):    
        self.axis.plot(self.energy_scale, self.spectrum, label='spectrum')
        x_limit = self.axis.get_xlim()
        y_limit = np.array(self.axis.get_ylim())
        self.xlabel = self.datasets[self.key].labels[0]
        self.ylabel = self.datasets[self.key].data_descriptor
        self.axis.set_xlabel(self.datasets[self.key].labels[0])
        self.axis.set_ylabel(self.datasets[self.key].data_descriptor)
        self.axis.ticklabel_format(style='sci', scilimits=(-2, 3))
        #if scale:
        #    self.axis.set_ylim(np.array(y_limit)*self.change_y_scale)
        self.change_y_scale = 1.0
        if self.y_scale != 1.:
                self.axis.set_ylabel('scattering probability (ppm/eV)')
        self.selector = matplotlib.widgets.SpanSelector(self.axis, self.line_select_callback,
                                         direction="horizontal",
                                         interactive=True,
                                         props=dict(facecolor='blue', alpha=0.2))
        self.axis.legend()
        if self.dataset.data_type.name == 'SPECTRUM':
             self.axis.set_title(self.dataset.title)
        else:
            self.axis.set_title(f'spectrum {self.x}, {self.y}')
        self.figure.canvas.draw_idle()

    def _update(self, ev=None):

        xlim = np.array(self.axes[1].get_xlim())
        ylim = np.array(self.axes[1].get_ylim())
        self.axes[1].clear()
        self.get_spectrum()
        if len(self.energy_scale)!=self.spectrum.shape[0]:
            self.spectrum = self.spectrum.T
        self.axes[1].plot(self.energy_scale, self.spectrum.compute(), label='experiment')

        self.axes[1].set_title(f'spectrum {self.x}, {self.y}')
        self.figure.tight_layout()
        self.selector = matplotlib.widgets.SpanSelector(self.axis, self.line_select_callback,
                                         direction="horizontal",
                                         interactive=True,
                                         props=dict(facecolor='blue', alpha=0.2))

        self.axes[1].set_xlim(xlim)
        self.axes[1].set_ylim(ylim*self.change_y_scale)
        self.axes[1].set_xlabel(self.xlabel)
        self.axes[1].set_ylabel(self.ylabel)
        self.change_y_scale = 1.0
        self.figure.canvas.draw_idle()

    def _onclick(self, event):
        self.event = event
        if event.inaxes in [self.axes[0]]:
            x = int(event.xdata)
            y = int(event.ydata)

            x = int(x - self.rectangle[0])
            y = int(y - self.rectangle[2])

            if x >= 0 and y >= 0:
                if x <= self.rectangle[1] and y <= self.rectangle[3]:
                    self.x = int(x / (self.rect.get_width() / self.bin_x))
                    self.y = int(y / (self.rect.get_height() / self.bin_y))
                    image_dims = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
            
                    if self.x + self.bin_x > self.dataset.shape[image_dims[0]]:
                        self.x = self.dataset.shape[image_dims[0]] - self.bin_x
                    if self.y + self.bin_y > self.dataset.shape[image_dims[1]]:
                        self.y = self.dataset.shape[image_dims[1]] - self.bin_y
            
                    self.rect.set_xy([self.x * self.rect.get_width() / self.bin_x + self.rectangle[0],
                                      self.y * self.rect.get_height() / self.bin_y + self.rectangle[2]])
                # self.get_spectrum()
                self._update()
        else:
            if event.dblclick:
                bottom = float(self.spectrum.min())
                if bottom < 0:
                    bottom *= 1.02
                else:
                    bottom *= 0.98
                top = float(self.spectrum.max())
                if top > 0:
                    top *= 1.02
                else:
                    top *= 0.98
                self.axis.set_ylim(bottom=bottom, top=top)

    def get_spectrum(self):
        if self.dataset.data_type == sidpy.DataType.SPECTRUM:
            self.spectrum = self.dataset.copy()
        else:
            image_dims = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
            if self.x > self.dataset.shape[image_dims[0]] - self.bin_x:
                self.x = self.dataset.shape[image_dims[0]] - self.bin_x
            if self.y > self.dataset.shape[image_dims[1]] - self.bin_y:
                self.y = self.dataset.shape[image_dims[1]] - self.bin_y
            selection = []
            self.axis.clear()
            for dim, axis in self.dataset._axes.items():
                # print(dim, axis.dimension_type)
                if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                    if dim == image_dims[0]:
                        selection.append(slice(self.x, self.x + self.bin_x))
                    else:
                        selection.append(slice(self.y, self.y + self.bin_y))

                elif axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                    selection.append(slice(None))
                elif axis.dimension_type == sidpy.DimensionType.CHANNEL:
                    selection.append(slice(None))
                else:
                    selection.append(slice(0, 1))
            
            self.spectrum = self.dataset[tuple(selection)].mean(axis=tuple(image_dims))
            
        self.spectrum *= self.y_scale
        
        return self.spectrum.squeeze()

    def plot_spectrum_image(self): 
        self.axes = self.figure.subplots(ncols=2)
        self.axis = self.axes[-1]

        spec_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
        if len(spec_dim) != 1:
            raise ValueError('Only one spectral dimension')
            
        channel_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.CHANNEL)
        channel_dim =[]
        if len(channel_dim) > 1:
            raise ValueError('Maximal one channel dimension')
                
        if len(channel_dim) > 0:
            self.image = self.dataset.mean(axis=(spec_dim[0] ,channel_dim[0]))
        else:
            self.image = self.dataset.mean(axis=(spec_dim[0]))

        self.rect = matplotlib.patches.Rectangle((0, 0), self.bin_x, self.bin_y, linewidth=1, edgecolor='r',
                                      facecolor='red', alpha=0.2)
        size_x = self.image.shape[0]
        size_y = self.image.shape[1]
        self.extent = [0, size_x, size_y, 0]
        self.rectangle = [0, size_x, 0, size_y]
        self.axes[0].imshow(self.image.T, extent=self.extent)
        self.axes[0].set_aspect('equal')
        self.axes[0].add_patch(self.rect)
        self.cid = self.axes[0].figure.canvas.mpl_connect('button_press_event', self._onclick)


    def line_select_callback(self, x_min, x_max):
        self.start_cursor.value = np.round(x_min, 3)
        self.end_cursor.value = np.round(x_max, 3)
        self.start_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.start_cursor.value)
        self.end_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.end_cursor.value)

    def set_dataset(self, index=0):

        if len(self.datasets) == 0:
            data_set = sidpy.Dataset.from_array([0, 1], name='generic')
            data_set.set_dimension(0, sidpy.Dimension([0,1], 'energy_loss',  units='channel', quantity='generic',    
                                                      dimension_type='spectral'))
            data_set.data_type = 'spectrum'
            data_set.metadata= {'experiment':{'convergence_angle': 0,
                                             'collection_angle': 0,
                                             'acceleration_voltage':0,
                                             'exposure_time':0}}
            self.datasets={'Channel_000': data_set}
            index = 0
        
        dataset_index = index
        
        self.key = list(self.datasets)[dataset_index]
        self.dataset = self.datasets[self.key]

        self._udpate_sidbar()
        self.y_scale = 1.0
        self.change_y_scale = 1.0
        self.x = 0
        self.y = 0
        self.bin_x = 1
        self.bin_y = 1
        self.count = 0

        self.plot()
        
    def _udpate_sidbar(self):
        pass    
        
    
    def set_energy_scale(self, value):
        dispersion = self.datasets[self.key].energy_loss[1] - self.datasets[self.key].energy_loss[0]
        self.datasets[self.key].energy_loss *= (self.sidebar[3, 0].value/dispersion)
        self.datasets[self.key].energy_loss += (self.sidebar[2, 0].value-self.datasets[self.key].energy_loss[0])
        self.plot()
        
    def set_y_scale(self, value):  
        self.count += 1
        self.change_y_scale = 1.0/self.y_scale
        if self.sidebar[9,2].value:
            dispersion = self.datasets[self.key].energy_loss[1] - self.datasets[self.key].energy_loss[0]
            self.y_scale = 1/self.datasets[self.key].metadata['experiment']['flux_ppm'] * dispersion
            self.ylabel='scattering probability (ppm)'
        else:
            self.y_scale = 1.0
            self.ylabel='intensity (counts)'
        self.change_y_scale *= self.y_scale
        self._update()
        
class InfoWidget(EELSWidget):
    def __init__(self, datasets):
        
        sidebar = get_info_sidebar()
        super().__init__(datasets, sidebar)
        self.set_action()

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
    
    def cursor2energy_scale(self, value):
        dispersion = (self.end_cursor.value - self.start_cursor.value) / (self.end_channel - self.start_channel)
        self.datasets[self.key].energy_loss *= (self.sidebar[3, 0].value/dispersion)
        self.sidebar[3, 0].value = dispersion
        offset = self.start_cursor.value - self.start_channel * dispersion
        self.datasets[self.key].energy_loss += (self.sidebar[2, 0].value-self.datasets[self.key].energy_loss[0])
        self.sidebar[2, 0].value = offset
        self.plot()

    def set_binning(self, value):
        if 'SPECTRAL' in self.dataset.data_type.name:
            bin_x = self.sidebar[15,0].value
            bin_y = self.sidebar[16,0].value
            self.dataset.view.set_bin([bin_x, bin_y])
            self.datasets[self.key].metadata['experiment']['SI_bin_x'] = bin_x
            self.datasets[self.key].metadata['experiment']['SI_bin_y'] = bin_y

    def _udpate_sidbar(self):
        spectrum_list = []
        reference_list =[('None', -1)]
        for index, key in enumerate(self.datasets.keys()):
            if 'Reference' not in key:
                if 'SPECTR' in self.datasets[key].data_type.name:
                    spectrum_list.append((f'{key}: {self.datasets[key].title}', index)) 
            reference_list.append((f'{key}: {self.datasets[key].title}', index))
       
        self.sidebar[0,0].options = spectrum_list
        self.sidebar[9,0].options = reference_list
        
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
    
    def update_dataset(self):
        dataset_index = self.sidebar[0, 0].value
        self.set_dataset(dataset_index)

    def set_action(self):
        self.sidebar[0,0].observe(self.update_dataset)
        self.sidebar[1,0].on_click(self.cursor2energy_scale)
        self.sidebar[2,0].observe(self.set_energy_scale, names='value')
        self.sidebar[3,0].observe(self.set_energy_scale, names='value')
        self.sidebar[5,0].observe(self.set_microscope_parameter)
        self.sidebar[6,0].observe(self.set_microscope_parameter)
        self.sidebar[7,0].observe(self.set_microscope_parameter)
        self.sidebar[9,0].observe(self.set_flux)
        self.sidebar[9,2].observe(self.set_y_scale, names='value')
        self.sidebar[10,0].observe(self.set_flux)
        self.sidebar[15,0].observe(self.set_binning)
        self.sidebar[16,0].observe(self.set_binning)
        
def get_low_loss_sidebar():
    side_bar = ipywidgets.GridspecLayout(17, 3,width='auto', grid_gap="0px")

    side_bar[0, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Main Dataset:',
            disabled=False)
    
    row = 1
    side_bar[row, :3] = ipywidgets.Button(description='Fix Energy Scale',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5,description='Offset:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Dispersion:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Resolution_function',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.3, description='Fit Window:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row,:2] = ipywidgets.ToggleButton(
            description='Show Resolution Function',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Changes y-axis to probability if flux is given',
                     layout=ipywidgets.Layout(width='100px'))
    side_bar[row,2] = ipywidgets.ToggleButton(
            description='Probability',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Changes y-axis to probability if flux is given',
                     layout=ipywidgets.Layout(width='100px'))
    row += 2

    side_bar[row, :3] = ipywidgets.Button(description='Drude Fit',
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
        pass
        # side_bar[i, 0].layout.display = "none"
    return side_bar

class LowLossWidget(EELSWidget):
    def __init__(self, datasets):
        sidebar = get_low_loss_sidebar()
        super().__init__(datasets, sidebar)
        self.sidebar[3,0].value = self.energy_scale[0]
        self.sidebar[4,0].value = self.energy_scale[1] - self.energy_scale[0]

        self.set_action()

    def _udpate_sidbar(self):
        spectrum_list = []
        reference_list =[('None', -1)]
        for index, key in enumerate(self.datasets.keys()):
            if 'Reference' not in key:
                if 'SPECTR' in self.datasets[key].data_type.name:
                    spectrum_list.append((f'{key}: {self.datasets[key].title}', index)) 
            reference_list.append((f'{key}: {self.datasets[key].title}', index))
       
        self.sidebar[0,0].options = spectrum_list
        self.sidebar[9,0].options = reference_list
        
        if 'SPECTRUM' in self.dataset.data_type.name:
           for i in range(14, 17):
                self.sidebar[i, 0].layout.display = "none"
        else:
            for i in range(14, 17):
                self.sidebar[i, 0].layout.display = "flex"

    def get_resolution_function(self, value):
        self.datasets['resolution_functions'] = eels_tools.get_resolution_functions(self.dataset, zero_loss_fit_width=self.sidebar[5,0].value)
        if 'low_loss' not in self.dataset.metadata:
            self.dataset.metadata['low_loss'] = {}
        self.dataset.metadata['low_loss'].update(self.datasets['resolution_functions'].metadata['low_loss'])
        self.sidebar[6,0].value = True

    def update_dataset(self):
        dataset_index = self.sidebar[0, 0].value
        self.set_dataset(dataset_index)

    def set_action(self):
        self.sidebar[0,0].observe(self.update_dataset)
        self.sidebar[1,0].on_click(self.fix_energy_scale)
        self.sidebar[2,0].observe(self.set_energy_scale, names='value')
        self.sidebar[3,0].observe(self.set_energy_scale, names='value')
        self.sidebar[4,0].on_click(self.get_resolution_function)
        self.sidebar[6,2].observe(self.set_y_scale, names='value')
        self.sidebar[6,0].observe(self._update, names='value')
        
    def fix_energy_scale(self, value=0):
        self.dataset = eels_tools.shift_on_same_scale(self.dataset)
        self.datasets[self.key] = self.dataset
        if 'resolution_functions' in self.datasets:
            self.datasets['resolution_functions'] = eels_tools.shift_on_same_scale(self.datasets['resolution_functions'])
        self._update()
        

    def set_y_scale(self, value):  
        self.change_y_scale = 1.0/self.y_scale
        if self.sidebar[6,2].value:
            dispersion = self.dataset.energy_loss[1] - self.dataset.energy_loss[0]
            if self.dataset.data_type.name == 'SPECTRUM':
                sum = self.dataset.sum()
            else:
                image_dims = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
                sum = np.average(self.dataset, axis=image_dims).sum()

            self.y_scale = 1/sum * dispersion * 1e6
            # self.datasets[self.key].metadata['experiment']['flux_ppm'] * dispersion
            self.ylabel='scattering probability (ppm)'
        else:
            self.y_scale = 1.0
            self.ylabel='intensity (counts)'
        self.change_y_scale *= self.y_scale
        self._update()

    def _update(self, ev=0):
        super()._update(ev)
        if self.sidebar[6,0].value:
            if 'resolution_functions' in self.datasets:
                resolution_function = self.get_additional_spectrum('resolution_functions')
                self.axis.plot(self.energy_scale, resolution_function, label='resolution_function')
                self.axis.legend()

    def get_additional_spectrum(self, key):
        if key not in self.datasets.keys():
            return
        
        if self.datasets[key].data_type == sidpy.DataType.SPECTRUM:
            self.spectrum = self.datasets[key].copy()
        else:
            image_dims = self.datasets[key].get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
            selection = []
            for dim, axis in self.datasets[key]._axes.items():
                # print(dim, axis.dimension_type)
                if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                    if dim == image_dims[0]:
                        selection.append(slice(self.x, self.x + self.bin_x))
                    else:
                        selection.append(slice(self.y, self.y + self.bin_y))

                elif axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                    selection.append(slice(None))
                elif axis.dimension_type == sidpy.DimensionType.CHANNEL:
                    selection.append(slice(None))
                else:
                    selection.append(slice(0, 1))
            
            self.spectrum = self.datasets[key][tuple(selection)].mean(axis=tuple(image_dims))
            
        self.spectrum *= self.y_scale
        
        return self.spectrum.squeeze()
    
    def set_binning(self, value):
        if 'SPECTRAL' in self.dataset.data_type.name:
            bin_x = self.sidebar[15,0].value
            bin_y = self.sidebar[16,0].value
            self.dataset.view.set_bin([bin_x, bin_y])
            self.datasets[self.key].metadata['experiment']['SI_bin_x'] = bin_x
            self.datasets[self.key].metadata['experiment']['SI_bin_y'] = bin_y

