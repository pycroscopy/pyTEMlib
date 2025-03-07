from typing import Any

import numpy as np
import os
import sys
import ipywidgets
from IPython.display import display
import plotly.graph_objects as go

import sidpy
# from pyTEMlib.microscope import microscope
from pyTEMlib import file_tools
from pyTEMlib import eels_tools
from pyTEMlib.core_loss_widget import get_core_loss_sidebar, CoreLoss
from pyTEMlib.low_loss_widget import get_low_loss_sidebar, LowLoss

def get_image_sidebar() -> Any:
    side_bar = ipywidgets.GridspecLayout(14, 3, width='auto', grid_gap="0px")

    side_bar[0, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Main Dataset:',
            disabled=False)
    row = 1
    side_bar[row, :3] = ipywidgets.Button(description='Image Scale',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5, description='x dim:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="nm", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5, description='y dim:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="nm", layout=ipywidgets.Layout(width='20px'))
    row += 1

    side_bar[row, :3] = ipywidgets.Button(description='Microscope',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=-1, description='Conv.Angle:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="mrad", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=-0.1, description='Coll.Angle:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="mrad", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=.1, description='Acc Voltage:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="keV", layout=ipywidgets.Layout(width='100px'))
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Calibration',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Pixel_Time:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="μs", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5, description='Screen Curr:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="pA", layout=ipywidgets.Layout(width='50px'))
    row += 1

    side_bar[row, 0] = ipywidgets.Button(description='FFT', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 1] = ipywidgets.Button(description='LR-Decon', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 2] = ipywidgets.Button(description='Find atoms', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Image Stack',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, 0] = ipywidgets.Button(description='Rig Reg.', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 1] = ipywidgets.Button(description='Demon', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 2] = ipywidgets.Button(description='Sum', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
   
    side_bar[-2, 0].layout.display = "none"
    for i in range(3):
        side_bar[-1, i].layout.display = "none"
    return side_bar
    

def get_info_sidebar() -> Any:
    side_bar = ipywidgets.GridspecLayout(18, 3, width='auto', grid_gap="0px")

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
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5, description='Offset:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Dispersion:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Microscope',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=-1, description='Conv.Angle:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="mrad", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=-0.1, description='Coll.Angle:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="mrad", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Acc Voltage:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="keV", layout=ipywidgets.Layout(width='100px'))
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Calibration',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Reference:',
            disabled=False)
    side_bar[row, 2] = ipywidgets.ToggleButton(description='Probability',
                                               disabled=False,
                                               button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                               tooltip='Changes y-axis to probability if flux is given',
                                               layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Exp_Time:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="s", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5, description='Flux:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="Mcounts", layout=ipywidgets.Layout(width='50px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Conversion:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="e⁻/counts", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Current:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="pA", layout=ipywidgets.Layout(width='100px'))
    
    row += 1
    side_bar[row, 0] = ipywidgets.Button(description='Get Shift', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 1] = ipywidgets.Button(description='Shift Spec', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    """ 
    side_bar[row, 2] = ipywidgets.Button(description='Res.Fct.', disabled=True,
                                         layout=ipywidgets.Layout(width='auto'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    """
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Spectrum Image',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    
    row += 1
    side_bar[row, :2] = ipywidgets.IntText(value=1, description='bin X:', disabled=False, color='black',
                                           layout=ipywidgets.Layout(width='200px'))
    row += 1
    side_bar[row, :2] = ipywidgets.IntText(value=1, description='bin Y:', disabled=False, color='black',
                                           layout=ipywidgets.Layout(width='200px'))
    
    for i in range(15, 18):
        side_bar[i, 0].layout.display = "none"
    return side_bar



def get_file_widget_ui():
    side_bar = ipywidgets.GridspecLayout(7, 3, height='500px', width='auto', grid_gap="0px")
    row = 0
    side_bar[row, :3] = ipywidgets.Dropdown(options=['None'], value='None', description='directory:', disabled=False,
                                            button_style='', layout=ipywidgets.Layout(width='auto', grid_area='header'))
    row += 1
    side_bar[row, :3] = ipywidgets.Select(options=['.'], value='.', description='Select file:', disabled=False,
                                          rows=10, layout=ipywidgets.Layout(width='auto'))
    row += 1
    side_bar[row, 0] = ipywidgets.Button(description='Select Main',
                                         layout=ipywidgets.Layout(width='100px'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 1] = ipywidgets.Button(description='Add',
                                         layout=ipywidgets.Layout(width='50px'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 2] = ipywidgets.Dropdown(options=['None'], value='None', description='loaded:', disabled=False,
                                           button_style='')

    row += 1
    side_bar[row, :3] = ipywidgets.Select(options=['None'], value='None', description='Spectral:',
                                          disabled=False, rows=3, layout=ipywidgets.Layout(width='auto'))
    row += 1
    side_bar[row, :3] = ipywidgets.Select(options=['Sum'], value='Sum', description='Images:',
                                          disabled=False, rows=3, layout=ipywidgets.Layout(width='auto'))
    row += 1
    side_bar[row, :3] = ipywidgets.Select(options=['None'], value='None', description='Survey:',
                                          disabled=False, rows=2, layout=ipywidgets.Layout(width='auto'))
    for i in range(3, 6):
        side_bar[i, 0].layout.display = "none"

    row += 1
    side_bar[row, 0] = ipywidgets.Button(description='Save',
                                         layout=ipywidgets.Layout(width='100px'),
                                         style=ipywidgets.ButtonStyle(button_color='lightblue'))
    side_bar[row, 1] = ipywidgets.Text(
                                        value='Test.hf5',
                                        placeholder='Type something',
                                        description='File:',
                                        disabled=False   
                                    )
    return side_bar  

out = ipywidgets.Output()

class EELSBaseWidget(object):

    def __init__(self, datasets=None, sidebar=None, tab_title=None):
        
        self.datasets = datasets
        self.dataset = None
        self.save_path = False
        self.dir_dictionary = {}
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']
        self.dataset_list = ['None']
        self.image_list = ['Sum']
        self.added_spectra = {}
        self.dir_name = file_tools.get_last_path()

        self.key = None
        self.new_info = False
        self.image = 'Sum'
        if 'google.colab' in sys.modules:
            self.google = True
        else:
            self.google = False
        # self.google = True

        self.save_path = True

        if not os.path.isdir(self.dir_name):
            self.dir_name = '.'

        self.get_directory(self.dir_name)
        self.dir_list = ['.']
        self.extensions = '*'
        self.file_name = ''
        self.datasets = {}
        self.dataset = None
        self.sd0 = 0
        self.sds = 0

        self.bin_x = 0
        self.bin_y = 0

        self.start_channel = -1
        self.end_channel = -2

        self.file_bar = get_file_widget_ui()
        children = [self.file_bar]
        titles = ['File']
        if isinstance(sidebar, dict):
            for sidebar_key, sidebar_gui in sidebar.items():
                children.append(sidebar_gui)
                titles.append(sidebar_key)
        elif not isinstance(sidebar, list):
            children = [self.file_bar, sidebar]
            titles = ['File', 'Info']
        if sidebar is None:
            children = [self.file_bar]
            titles = ['File']

        if self.google:
            self.buttons = []
            for i in range(len(children)):
                self.buttons.append(ipywidgets.Button(description=titles[i],
                                    disabled=False,
                                    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                    layout=ipywidgets.Layout(width='800px')))


            self.tab_buttons = ipywidgets.ToggleButtons(options=titles, description='', disabled=False,
                                                        layout=ipywidgets.Layout(width='auto'),
                                                        style={"button_width": "auto"})
            tab = ipywidgets.VBox([self.tab_buttons, self.file_bar])
            self.children = children

        else:
            tab = ipywidgets.Tab()
            tab.children = children
            tab.titles = titles

        

        self.spectrum_plot = go.FigureWidget()
        self.spectrum_plot.update_xaxes(showgrid=True, zeroline=True, showticklabels=True,
                        showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid')
        self.spectrum_plot['layout'].update(height=500)
        self.image_plot = go.FigureWidget()
        self.image_plot['layout'].update(height=500, 
                       width=500, 
                       autosize=True, 
                       xaxis_showgrid=False, 
                       yaxis_showgrid=False,
                       yaxis = dict(scaleanchor = 'x', autorange = "reversed"),
                       plot_bgcolor="white")
        
        self.tab =tab
        self.canvas_plot = ipywidgets.HBox([self.spectrum_plot])
        self.canvas = ipywidgets.VBox([self.canvas_plot, out])

        self.start_cursor = ipywidgets.FloatText(value=0, description='Start:', disabled=False, color='black',
                                                 layout=ipywidgets.Layout(width='200px'))
        self.end_cursor = ipywidgets.FloatText(value=0, description='End:', disabled=False, color='black',
                                               layout=ipywidgets.Layout(width='200px'))
        self.statusbar = ipywidgets.Text(value='Starting',
                                    placeholder='Type something',
                                    description='Status:',
                                    disabled=True,
                                    layout=ipywidgets.Layout(width='100%'))
        self.panel = ipywidgets.VBox([ipywidgets.HBox([ipywidgets.Label('', layout=ipywidgets.Layout(width='100px')),
                                                       ipywidgets.Label('Cursor:'),
                                                       self.start_cursor, ipywidgets.Label('eV'),
                                                       self.end_cursor, ipywidgets.Label('eV')]),
                                     self.canvas,
                                     self.statusbar])
        
        self.app_layout = ipywidgets.AppLayout(
            left_sidebar=tab,
            center=self.panel,
            footer=None,  # message_bar,
            pane_heights=[0, 10, 0],
            pane_widths=[4, 10, 0],
        )
        # self.set_dataset()
        self.change_y_scale = 1.0
        self.x = 0
        self.y = 0
        self.bin_x = 1
        self.bin_y = 1
        self.count = 0
        display(self.app_layout)

        self.select_files = self.file_bar[1, 0]
        self.path_choice = self.file_bar[0, 0]
        self.set_file_options()
        select_button = self.file_bar[2, 0]
        add_button = self.file_bar[2, 1]
        self.loaded_datasets = self.file_bar[2, 2]
        self.select_files.observe(self.get_file_name, names='value')
        self.path_choice.observe(self.set_dir, names='value')

        select_button.on_click(self.select_main)
        add_button.on_click(self.add_dataset)
        self.loaded_datasets.observe(self.select_dataset, names='value')
        self.file_bar[4, 0].observe(self.set_image, names='value')
        self.file_bar[5, 0].observe(self.set_survey_image, names='value')

        self.file_bar[6, 0].observe(self.save_datasets)
    
    def save_datasets(self, value=0):
        import warnings
        file_name = self.file_bar[6, 1].value
        path = self.path_choice.options[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        h5_group = file_tools.save_dataset(self.datasets, os.path.join(path, file_name))
        h5_group.file.close()
        self.status_message(' File saved')

    def status_message(self, out: str):
        self.statusbar.value = out 

    def set_survey_image(self, key=None):
        
        self.datasets['_relationship']['survey_image'] = self.file_bar[5, 0].value
        # ToDo: Find boundaries of scan. 
        
    def get_image(self):
        if self.file_bar[4, 0].value == 'Sum':
            spec_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
            if len(spec_dim) != 1:
                raise ValueError('Only one spectral dimension')

            channel_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.CHANNEL)

            if len(channel_dim) > 1:
                raise ValueError('Maximal one channel dimension')

            if len(channel_dim) > 0:
                self.image = self.dataset.mean(axis=(spec_dim[0], channel_dim[0]))
            else:
                self.image = self.dataset.mean(axis=(spec_dim[0]))
            image_key = 'Sum'
        else:
            image_key = self.file_bar[4, 0].value.split(':')[0]
            self.image = self.datasets[image_key]
        self.datasets['_relationship']['image'] = image_key
        
    def set_image(self, key=None):
        self.get_image()
        self.plot()

    def plot(self, scale=True):
        
        spec_dims = self.dataset.get_spectral_dims(return_axis=True)
        if len(spec_dims)>0:
            self.energy_scale = spec_dims[0]
            if self.dataset.data_type.name == 'SPECTRUM':
                self.canvas_plot.children = [self.spectrum_plot]
            else:
                self.get_image()
                self.canvas_plot.children = [self.image_plot, self.spectrum_plot]
                self.plot_spectrum_image() 
                # self.axis = self.axes[-1]
            self.spectrum = self.get_spectrum()
            self.plot_spectrum()
    
        else:
            self.canvas_plot.children = [self.image_plot]
            self.image = self.dataset
            self.plot_image()
        
    def plot_image(self, add_annotations=True): 
        image_dims = self.dataset.get_image_dims(return_axis=True)
        
        if len(self.image_plot.data) == 0:
                    self.image_plot.add_trace(go.Heatmap(z=self.image.T))
        else:
            self.image_plot.data[0].z=np.array(self.image).T
        self.image_plot.data[0].x = image_dims[0].values
        self.image_plot.data[0].y = image_dims[1].values
        
        self.image_plot.update_layout(xaxis_title = f"{image_dims[0].quantity} ({image_dims[0].units})", 
                                      yaxis_title = f"{image_dims[1].quantity} ({image_dims[1].units})")
        
        if 'annotations' in self.dataset.metadata and add_annotations:                      
            annotations = self.dataset.metadata['annotations']
            for key in annotations:
                if annotations[key]['type'] == 'spectral_image':
                    pos, w, h = annotations[key]['position'], annotations[key]['width'], annotations[key]['height']
                    self.image_plot.add_trace(go.Scatter(x= [pos[0], pos[0], pos[0]+w, pos[0]+w, pos[0]],  y = [pos[1], pos[1]+h, pos[1]+h, pos[1], pos[1]], mode='lines'))
                    self.image_plot.add_trace(go.Scatter(x= [pos[0]],  y = [pos[1]], mode='text', text=['spectrum image'], 
                                                         textfont=dict(color="red"),
                                                         textposition="top right"))

                elif annotations[key]['type'] == 'text':
                    self.image_plot.add_trace(go.Scatter(y=[annotations[key]['position'][0]], x=[annotations[key]['position'][1]], 
                                                         text=[annotations[key]['label']]))
                elif annotations[key]['type'] == 'circle':
                    [x, y] = annotations[key]['position']
                    r = annotations[key]['radius']
                    self.image_plot.add_shape(type="circle", xref="x", yref="y", x0=x-r, y0 = y-r, x1=x+r, y1=y+r,
                                                line_color="LightSeaGreen")
                    
        
    def plot_spectrum(self):    
        if len(self.spectrum_plot.data) == 0:
            self.spectrum_plot.add_trace(go.Scatter(x=self.energy_scale, y=self.spectrum, mode='markers+lines', marker_size=.1, name=self.dataset.title))
        self.spectrum_plot.data = [self.spectrum_plot.data[0]]

        self.xlabel = self.spectrum.labels[0]
        self.ylabel = self.datasets[self.key].data_descriptor
        # self.change_y_scale = 1.0
        if self.y_scale != 1.:
            self.ylabel = 'scattering probability (ppm/eV)'
        
        self.spectrum_plot.update_layout(xaxis_title=self.xlabel, yaxis_title=self.ylabel)
        self.spectrum_plot.data[0].y=self.spectrum
        self.spectrum_plot.data[0].x=self.energy_scale
        self.spectrum_plot.data[0].on_selection(self.selection_cursor)
       
        if self.dataset.data_type.name != 'SPECTRUM':
             self.spectrum_plot.data[0].name = f'spectrum {self.x}, {self.y}'

        for key in self.added_spectra:
            self.added_spectra[key]
            spectrum = self.get_additional_spectrum(key)
            
            self.spectrum_plot.add_trace(go.Scatter(x=self.energy_scale, y=spectrum, mode='markers+lines', marker_size=.1, name=self.added_spectra[key]))
            if key == 'zero_loss':
                self.low_loss.low_loss_tab[14, 1].value = np.round(np.log(self.spectrum.sum()/spectrum.sum()), 4)

                
    def _update(self, ev=None):
        self.get_spectrum()
        self.plot_spectrum()

    def update_tab_spectra(self):
        pass

    def get_spectrum(self):
        if self.dataset.data_type.name == 'SPECTRUM':
            self.spectrum = self.dataset.copy()
        else:
            image_dims = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
            if self.x > self.dataset.shape[image_dims[0]] - self.bin_x:
                self.x = self.dataset.shape[image_dims[0]] - self.bin_x
            if self.y > self.dataset.shape[image_dims[1]] - self.bin_y:
                self.y = self.dataset.shape[image_dims[1]] - self.bin_y
            selection = []
            for dim, axis in self.dataset._axes.items():
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
        self.spectrum.squeeze()
        self.spectrum.data_type = 'spectrum'
        return self.spectrum

    @out.capture(clear_output=True)
    def click_callback(self, trace, points, selector):
        if selector.shift:
            self.spectrum_plot.add_trace(go.Scatter(x=self.energy_scale, 
                                                    y=self.dataset[points.point_inds[0][1], points.point_inds[0][0]], 
                                                    mode='lines',
                                                    name='spectrum'+str(points.point_inds[0])))
        else: 
            if selector.ctrl:
                self.spectrum_plot.data =[self.spectrum_plot.data[0]]
            
        
            self.image_plot.data[1].x = [points.point_inds[0][1]]
            self.image_plot.data[1].y = [points.point_inds[0][0]]
            self.x = points.point_inds[0][1]
            self.y = points.point_inds[0][0]

            self._update()

    @out.capture(clear_output=True)
    def selection_fn(self, trace,points,selector):
        if selector.type == 'box':
            xr = np.array(selector.xrange)
            if xr[0]<0:
                xr[0] = 0
            yr = np.array(selector.yrange)
            if yr[0]<1:
                yr[0] = 0   
            size_sel = (int(xr[1])-int(xr[0]))*(int(yr[1])-int(yr[0]))
            
            self.spectrum_plot.data[0].y= self.dataset[int(xr[0]):int(xr[1]), int(yr[0]):int(yr[1]),:].sum(axis=[0,1]).compute()/ size_sel
            self.spectrum_plot.data[0].name = str(size_sel)+ ' spectra'
        else:
            print(selector)

    def plot_spectrum_image(self): 
        if len(self.image_plot.data) == 0:
                    self.image_plot.add_trace(go.Heatmap(z=self.image.T))
        else:
            self.image_plot.data = [self.image_plot.data[0]]
            self.image_plot.data[0].z=np.array(self.image).T
        self.plot_spectrum()
        self.image_plot.data = [self.image_plot.data[0]]
        self.image_plot.add_trace(
                    go.Scatter(mode="markers", x=[0], y=[0], marker_symbol=[101],
                   marker_color="darkgray", 
                   marker_line_width=1, marker_size=11, hovertemplate= 'x: %{x}<br>y: %{y}'))

        self.image_plot.data[0].on_selection(self.selection_fn)
        self.image_plot.data[0].on_click(self.click_callback)
    
    @out.capture(clear_output=True)
    def selection_cursor(self, trace, points, selector):
        if selector.type == 'box':
            self.start_cursor.value = np.round(selector.xrange[0], 3)
            self.end_cursor.value = np.round(selector.xrange[1], 3)

            energy_scale = self.dataset.get_spectral_dims(return_axis=True)[0]
            self.start_channel = np.searchsorted(energy_scale, self.start_cursor.value)
            self.end_channel = np.searchsorted(energy_scale, self.end_cursor.value)

    def set_dataset(self, key=None):
         
        if len(self.datasets) == 0:
            data_set = sidpy.Dataset.from_array([0, 1], name='generic')
            data_set.set_dimension(0, sidpy.Dimension([0, 1], 'energy_loss',  units='channel', quantity='generic',
                                                      dimension_type='spectral'))
            data_set.data_type = 'spectrum'
            data_set.metadata = {'experiment': {'convergence_angle': 0,
                                                'collection_angle': 0,
                                                'acceleration_voltage': 0,
                                                'exposure_time': 0}}
            self.datasets = {'Nothing': data_set}
            key = 'Nothing'
        
        dataset_key = key
        
        self.dataset_list = []
        dataset_keys = []
        for key in self.datasets.keys():
            if isinstance(self.datasets[key], sidpy.Dataset):
                self.dataset_list.append(f'{key}: {self.datasets[key].title}')
                dataset_keys.append(key)
        if dataset_key not in dataset_keys:
            dataset_key = dataset_keys[0]
        self.key = dataset_key

        self.dataset = self.datasets[self.key]
        
        spectral_dims = self.dataset.get_spectral_dims(return_axis=True)
        if len(spectral_dims) >0:
            self.energy_scale = spectral_dims[0]
        self.y_scale = 1.0
        self.change_y_scale = 1.0
        self.x = 0
        self.y = 0
        self.bin_x = 1
        self.bin_y = 1
        self.count = 0

        self.update_sidebars()
        #self.update_sidebar()
        self.plot()
        
    def update_sidebars(self):
        pass

    def select_main(self, value=0):
        self.sds +=1
        self.datasets = {}
        # self.loaded_datasets.options = self.dataset_list
        
        self.datasets = file_tools.open_file(self.file_name)
        
        file_tools.save_path(self.file_name)
        self.dataset_list = []
        self.image_list = ['Sum']
        self.survey_list = ['None']
        self.spectral_list = ['None']
        for key in self.datasets.keys():
            if isinstance(self.datasets[key], sidpy.Dataset):
                self.dataset_list.append(f'{key}: {self.datasets[key].title}')
                if 'SPECTR' in self.datasets[key].data_type.name:
                    self.spectral_list.append(f'{key}: {self.datasets[key].title}')
                if 'IMAGE' == self.datasets[key].data_type.name:
                    if 'survey' in self.datasets[key].title.lower():
                        self.survey_list.append(f'{key}: {self.datasets[key].title}')
                    else:
                        self.image_list.append(f'{key}: {self.datasets[key].title}')

        
        self.key = self.dataset_list[0].split(':')[0]
        self.dataset = self.datasets[self.key]
        if len(self.image_plot.data)>0:
            self.image_plot.data = [self.image_plot.data[0]]
        self.spectrum_plot.data = [self.spectrum_plot.data[0]] 
        self.new_info = True

        self.selected_dataset = self.dataset
        if len(self.image_list) > 0:
            self.file_bar[4, 0].options = self.image_list
            self.file_bar[5, 0].options = self.survey_list
            self.file_bar[4, 0].layout.display = "flex"
            self.file_bar[4, 0].value = self.image_list[0]
            self.file_bar[5, 0].layout.display = "flex"
            self.file_bar[5, 0].value = self.survey_list[0]

        self.file_bar[3, 0].options = self.dataset_list
        self.loaded_datasets.options = self.dataset_list
        self.loaded_datasets.value = self.dataset_list[0]
        path, filename = os.path.split(self.file_name)
        name, extension = os.path.splitext(filename)
        self.file_bar[6, 1].value = name+'.hf5'
        self.status_message(' New file loaded')

    def add_dataset(self, value=0):
        key = file_tools.add_dataset_from_file(self.datasets, self.file_name, 'Channel')
        self.dataset_list.append(f'{key}: {self.datasets[key].title}')
        self.loaded_datasets.options = self.dataset_list
        self.loaded_datasets.value = self.dataset_list[-1]
        
        if 'SPECTR' in self.datasets[key].data_type.name:
            self.spectral_list.append(f'{key}: {self.datasets[key].title}')
            energy = self.datasets[key].get_spectral_dims(return_axis=True)[0]
            self.spectrum_plot.add_trace(go.Scatter(x=energy, y=self.datasets[key], mode='markers+lines', marker_size=.1, name=key))
        if 'IMAGE' == self.datasets[key].data_type.name:
            if 'survey' in self.datasets[key].title.lower():
                self.survey_list.append(f'{key}: {self.datasets[key].title}')
                self.file_bar[5, 0].options = self.survey_list
            else:
                self.image_list.append(f'{key}: {self.datasets[key].title}')
                self.file_bar[4, 0].options = self.image_list
                self.status_message(' image list file loaded')

    def get_directory(self, directory='.'):
        self.dir_name = directory
        self.dir_dictionary = {}
        self.dir_list = []
        self.dir_list = ['.', '..'] + os.listdir(directory)

    def set_dir(self, value=0):
        self.dir_name = self.path_choice.value
        self.select_files.index = 0
        self.set_file_options()

    def select_dataset(self, value=0):
        key = self.loaded_datasets.value.split(':')[0]
        if key != 'None':
            self.selected_dataset = self.datasets[key]
            self.selected_key = key
            self.key = key
        self.datasets['_relationship'] = {'main_dataset': self.key}
        
        self.set_dataset()

    def set_file_options(self):
        self.dir_name = os.path.abspath(os.path.join(self.dir_name, self.dir_list[self.select_files.index]))
        dir_list = os.listdir(self.dir_name)
        file_dict = file_tools.update_directory_list(self.dir_name)

        sort = np.argsort(file_dict['directory_list'])
        self.dir_list = ['.', '..']
        self.display_list = ['.', '..']
        for j in sort:
            self.display_list.append(f" * {file_dict['directory_list'][j]}")
            self.dir_list.append(file_dict['directory_list'][j])

        sort = np.argsort(file_dict['display_file_list'])

        for i, j in enumerate(sort):
            if '--' in dir_list[j]:
                self.display_list.append(f" {i:3} {file_dict['display_file_list'][j]}")
            else:
                self.display_list.append(f" {i:3}   {file_dict['display_file_list'][j]}")
            self.dir_list.append(file_dict['file_list'][j])

        self.dir_label = os.path.split(self.dir_name)[-1] + ':'
        self.select_files.options = self.display_list
        
        path = self.dir_name
        old_path = ' '
        path_list = []
        while path != old_path:
            path_list.append(path)
            old_path = path
            path = os.path.split(path)[0]
        self.path_choice.options = path_list
        self.path_choice.value = path_list[0]

    def get_file_name(self, b):

        if os.path.isdir(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.set_file_options()

        elif os.path.isfile(os.path.join(self.dir_name, self.dir_list[self.select_files.index])):
            self.file_name = os.path.join(self.dir_name, self.dir_list[self.select_files.index])

class EELSWidget(EELSBaseWidget):
    def __init__(self, datasets=None):
        self.lowloss_key = 'None'
        self.coreloss_key = 'None'
        self.info_key = 'None'
        self.tabval = 0

        sidebar = {'Spec.': get_info_sidebar(),
                   'LowLoss': get_low_loss_sidebar(),
                   'CoreLoss': get_core_loss_sidebar()}
        super().__init__(datasets, sidebar)
        self.info_tab = sidebar['Spec.']
        self.core_loss_tab = sidebar['CoreLoss']
        self.low_loss_tab = sidebar['LowLoss']
        super().set_dataset()
        self.info = Info(self.info_tab, self)
        self.low_loss = LowLoss(self.low_loss_tab, self)
        self.core_loss = CoreLoss(self.core_loss_tab, self)
        
        self.set_action()
    
    def set_action(self):
        if self.google:
            self.tab_buttons.observe(self.tab_activated)
        self.tab.observe(self.tab_activated)

    def tab_activated(self, val=0):
        if self.google:
            self.tab.children = [self.tab_buttons, self.children[self.tab_buttons.index]]  # update sidebar gui
            self.tabval = self.tab_buttons.index
        else:
            if isinstance(val.new, int):
                self.tabval = val.new
            # self.update_sidebars()
        if self.tabval == 1:
            self.info.update_dataset()
        elif self.tabval == 2:
            self.low_loss.update_ll_sidebar()
        elif self.tabval == 3:
            self.core_loss.update_cl_sidebar()
    
    def update_tab_spectra(self):
        if self.tabval == 2:
            self.low_loss._update()

    def update_sidebars(self):
        if hasattr(self, 'info'):
            self.info.update_sidebar()
        #if hasattr(self, 'low_loss'):
        #    self.low_loss.update_ll_sidebar()
        #if hasattr(self, 'core_loss'):
        #    self.core_loss.update_cl_sidebar()


    def get_additional_spectrum(self, key):
        if key not in self.datasets.keys():
            return    
        if isinstance(self.datasets[key], np.ndarray):
            return self.datasets[key]*self.y_scale
        
        if isinstance(self.datasets[key], sidpy.Dataset):
            if self.datasets[key].data_type == sidpy.DataType.SPECTRUM:
                spectrum = self.datasets[key].copy()
            else:
                image_dims = self.datasets[key].get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
                selection = []
                x = self.x
                y = self.y
                bin_x = self.bin_x
                bin_y = self.bin_y
                for dim, axis in self.datasets[key]._axes.items():
                    # print(dim, axis.dimension_type)
                    if axis.dimension_type == sidpy.DimensionType.SPATIAL:
                        if dim == image_dims[0]:
                            selection.append(slice(x, x + bin_x))
                        else:
                            selection.append(slice(y, y + bin_y))

                    elif axis.dimension_type == sidpy.DimensionType.SPECTRAL:
                        selection.append(slice(None))
                    elif axis.dimension_type == sidpy.DimensionType.CHANNEL:
                        selection.append(slice(None))
                    else:
                        selection.append(slice(0, 1))
                
                spectrum = self.datasets[key][tuple(selection)].mean(axis=tuple(image_dims))
                spectrum.data_type = 'spectrum'
            
        spectrum *= self.y_scale
        return spectrum.squeeze()

    
class Info(object):
    def __init__(self, sidebar=None, parent=None):
        self.parent = parent
        self.info_tab = sidebar
        self.key = self.parent.info_key
        self.update_sidebar()        
        self.set_action()
        self.count =0

    def set_energy_scale(self, value):
        self.ens = 1
        self.energy_scale = self.parent.datasets[self.key].get_spectral_dims(return_axis=True)[0]
        dispersion = self.parent.datasets[self.key].get_dimension_slope(self.energy_scale)
        self.ens = dispersion
        self.energy_scale *= (self.info_tab[3, 0].value / dispersion)
        self.energy_scale += (self.info_tab[2, 0].value - self.energy_scale[0])
        self.parent.plot()

    def set_y_scale(self, value):
        self.count += 1
        self.parent.change_y_scale = 1.0 / self.parent.y_scale
        if self.parent.datasets[self.parent.key].metadata['experiment']['flux_ppm'] > 1e-12:
            if self.info_tab[9, 2].value:
                dispersion = self.parent.datasets[self.parent.key].get_dimension_slope(self.parent.energy_scale)
                self.parent.y_scale = 1 / self.parent.datasets[self.parent.key].metadata['experiment']['flux_ppm'] * dispersion
                self.parent.ylabel = 'scattering probability (ppm)'
            else:
                self.parent.y_scale = 1.0
                self.parent.ylabel = 'intensity (counts)'
            self.parent.change_y_scale *= self.parent.y_scale
            self.parent._update()

    def set_flux(self, value):
        # self.parent.datasets[self.key].metadata['experiment']['exposure_time'] = self.info_tab[10, 0].value
        if self.info_tab[9, 0].value == 'None':
            self.parent.datasets[self.parent.key].metadata['experiment']['flux_ppm'] = 0.
        else:
            ll_key = self.info_tab[9, 0].value.split(':')[0]
            self.parent.datasets['_relationship']['low_loss'] = ll_key
            self.parent.lowloss_key = ll_key
            spectrum_dimensions = self.parent.dataset.get_spectral_dims()

            number_of_pixels = 1
            for index, dimension in enumerate(self.parent.dataset.shape):
                if index not in spectrum_dimensions:
                    number_of_pixels *= dimension
            if self.parent.datasets[ll_key].metadata['experiment']['exposure_time'] == 0.0:
                if self.parent.datasets[ll_key].metadata['experiment']['single_exposure_time'] == 0.0:
                    return
                else:
                    self.parent.datasets[ll_key].metadata['experiment']['exposure_time'] = (self.parent.datasets[ll_key].metadata['experiment']['single_exposure_time'] *
                                                                                     self.parent.datasets[ll_key].metadata['experiment']['number_of_frames'])

            self.parent.datasets[self.parent.key].metadata['experiment']['flux_ppm'] = ((np.array(self.parent.datasets[ll_key])*1e-6).sum() /
                                                                          self.parent.datasets[ll_key].metadata['experiment']['exposure_time'] /
                                                                          number_of_pixels)
            self.parent.datasets[self.parent.key].metadata['experiment']['flux_ppm'] *= self.parent.datasets[self.parent.key].metadata['experiment']['exposure_time']
            if 'SPECT' in self.parent.datasets[ll_key].data_type.name:
                self.info_tab[14, 0].disabled = False
        self.info_tab[11, 0].value = np.round(self.parent.datasets[self.parent.key].metadata['experiment']['flux_ppm'], 2)

    def set_microscope_parameter(self, value):
        if not self.parent.new_info:
            self.parent.datasets[self.key].metadata['experiment']['convergence_angle'] = self.info_tab[5, 0].value
            self.parent.datasets[self.key].metadata['experiment']['collection_angle'] = self.info_tab[6, 0].value
            self.parent.datasets[self.key].metadata['experiment']['acceleration_voltage'] = self.info_tab[7, 0].value*1000
    
    def cursor2energy_scale(self, value):
        self.energy_scale = self.parent.datasets[self.key].get_spectral_dims(return_axis=True)[0]
        dispersion = (self.parent.end_cursor.value - self.parent.start_cursor.value) / (self.parent.end_channel - self.parent.start_channel)

        self.energy_scale *= (self.info_tab[3, 0].value/dispersion)
        
        offset = self.parent.start_cursor.value - self.parent.start_channel * dispersion
        self.parent.energy_scale += (self.info_tab[2, 0].value-self.parent.energy_scale[0])
        self.info_tab[2, 0].value = np.round(offset,4)
        self.info_tab[3, 0].value = np.round(dispersion,4)
        self.parent.plot()

    def set_binning(self, value):
        if 'SPECTRAL' in self.parent.dataset.data_type.name:
            image_dims = self.parent.dataset.get_image_dims()

            self.bin_x = int(self.info_tab[16, 0].value)
            self.bin_y = int(self.info_tab[17, 0].value)
            if self.bin_x < 1:
                self.bin_x = 1
                self.info_tab[16, 0].value = self.bin_x
            if self.bin_y < 1:
                self.bin_y = 1
                self.info_tab[17, 0].value = self.bin_y
            if self.bin_x > self.parent.dataset.shape[image_dims[0]]:
                self.bin_x = self.parent.dataset.shape[image_dims[0]]
                self.info_tab[16, 0].value = self.bin_x
            if self.bin_y > self.parent.dataset.shape[image_dims[1]]:
                self.bin_y = self.parent.dataset.shape[image_dims[1]]
                self.info_tab[17, 0].value = self.bin_y
            self.parent.bin_x = self.bin_x
            self.parent.bin_y = self.bin_y

            self.parent.datasets[self.key].metadata['experiment']['SI_bin_x'] = self.bin_x
            self.parent.datasets[self.key].metadata['experiment']['SI_bin_y'] = self.bin_y
            self.parent.plot()

    

    def update_sidebar(self):
        spectrum_list = ['None']
        reference_list = ['None']
        data_list = []
        
        self.key = self.info_key = self.parent.info_key
        
        spectrum_data = False
        info_index= 0
        for key in self.parent.datasets.keys():
            if isinstance(self.parent.datasets[key], sidpy.Dataset):
                if key[0] != '_' :
                    data_list.append(f'{key}: {self.parent.datasets[key].title}')
                    if 'SPECTR' in self.parent.datasets[key].data_type.name:
                        spectrum_data = True
                        spectrum_list.append(f'{key}: {self.parent.datasets[key].title}')
                        if self.info_key == key:
                            info_index = len(spectrum_list)-1
                reference_list.append(f'{key}: {self.parent.datasets[key].title}')
        self.info_tab[0, 0].options = spectrum_list
        self.info_tab[9, 0].options = reference_list
        self.info_tab[0, 0].value = spectrum_list[info_index]
            
        if 'SPECTRUM' in self.parent.dataset.data_type.name:
            for i in range(15, 18):
                self.info_tab[i, 0].layout.display = "none"
        else:
            for i in range(15, 18):
                self.info_tab[i, 0].layout.display = "flex"

        if 'None' not in self.key:
            self.parent.new_info = True
            energy_scale = self.parent.datasets[self.key].get_spectral_dims(return_axis=True)
            if len(energy_scale) == 0:
                return
            energy_scale = energy_scale[0]
            offset = energy_scale[0]
            # dispersion = self.parent.datasets[self.key].get_dimension_slope(energy_scale)
            dispersion = energy_scale[1] - offset
            
            # self.info_tab[0,0].value = dataset_index #f'{self.key}: {self.parent.datasets[self.key].title}'
            self.info_tab[2, 0].unobserve_all()
            self.info_tab[2, 0].value = np.round(offset, 3)
            self.info_tab[3, 0].value = np.round(dispersion, 4)
            self.info_tab[5, 0].value = np.round(self.parent.datasets[self.key].metadata['experiment']['convergence_angle'], 1)
            self.info_tab[6, 0].value = np.round(self.parent.datasets[self.key].metadata['experiment']['collection_angle'], 1)
            self.info_tab[7, 0].value = np.round(self.parent.datasets[self.key].metadata['experiment']['acceleration_voltage']/1000, 1)
            self.info_tab[2, 0].observe(self.set_energy_scale, names='value')
            # print(self.parent.datasets[self.key].metadata['experiment']['acceleration_voltage'])
            self.info_tab[10, 0].value = np.round(self.parent.datasets[self.key].metadata['experiment']['exposure_time'], 4)
            if 'flux_ppm' not in self.parent.datasets[self.key].metadata['experiment']:
                self.parent.datasets[self.key].metadata['experiment']['flux_ppm'] = 0
            self.info_tab[11, 0].value = self.parent.datasets[self.key].metadata['experiment']['flux_ppm']
            if 'count_conversion' not in self.parent.datasets[self.key].metadata['experiment']:
                self.parent.datasets[self.key].metadata['experiment']['count_conversion'] = 1
            self.info_tab[12, 0].value = self.parent.datasets[self.key].metadata['experiment']['count_conversion']
            if 'beam_current' not in self.parent.datasets[self.key].metadata['experiment']:
                self.parent.datasets[self.key].metadata['experiment']['beam_current'] = 0
            self.info_tab[13, 0].value = self.parent.datasets[self.key].metadata['experiment']['beam_current']
            ll_key = 'None'
            if '_relationship' in self.parent.datasets:
                if 'low_loss' in self.parent.datasets['_relationship']:
                    ll_key = self.parent.datasets['_relationship']['low_loss']  
                    ll_key = f'{ll_key}: {self.parent.datasets[ll_key].title}'
                    self.lowloss_key = ll_key
            self.info_tab[9, 0].value = ll_key
            self.parent.new_info = False

    def update_dataset(self, value=0):
        self.key = self.info_tab[0, 0].value.split(':')[0]
        
        self.info_key = self.key
        self.parent.info_key = self.key
        
        if self.info_key != 'None':
            self.parent.set_dataset(self.info_key)
        self.parent.status_message(self.key+'  ,  '+ self.parent.info_key)
        if '_relationship' in self.parent.datasets.keys():
            self.parent.datasets['_relationship']['spectrum'] = self.info_key
        self.update_sidebar()
        self.parent._update(0)
        
    def get_shift(self,  value=0):
        if 'low_loss' in self.parent.datasets['_relationship']:
            low_loss = self.parent.datasets[self.parent.datasets['_relationship']['low_loss']]

            self.parent.datasets['shifted_low_loss']  = eels_tools.align_zero_loss(low_loss)
            self.parent.datasets['shifted_low_loss'].title = self.parent.dataset.title + '_shifted'
            self.parent.datasets['_relationship']['low_loss'] = 'shifted_low_loss'
            self.update_sidebar()
           
        if 'low_loss' in self.parent.datasets['_relationship']:
            if 'zero_loss' in self.parent.datasets[self.parent.datasets['_relationship']['low_loss']].metadata:
                if 'shifted' in self.parent.datasets[self.parent.datasets['_relationship']['low_loss']].metadata['zero_loss'].keys():
                    self.info_tab[14, 1].disabled = False
                    

    def shift_spectrum(self,  value=0):
        shifts = self.parent.dataset.shape
        if 'low_loss' in self.parent.datasets['_relationship']:
            if 'zero_loss' in self.parent.datasets[self.parent.datasets['_relationship']['low_loss']].metadata:
                if 'shifted' in self.parent.datasets[self.parent.datasets['_relationship']['low_loss']].metadata['zero_loss'].keys():
                    shifts = self.parent.datasets[self.parent.datasets['_relationship']['low_loss']].metadata['zero_loss']['shifted']
                    shifts_new = shifts.copy()
                    if 'zero_loss' in self.parent.dataset.metadata:
                        if 'shifted' in self.parent.dataset.metadata['zero_loss'].keys():
                            shifts_new = shifts-self.parent.dataset.metadata['zero_loss']['shifted']
                    else:
                        self.parent.dataset.metadata['zero_loss'] = {}
                    

                    self.parent.dataset = eels_tools.shift_energy(self.parent.dataset, shifts_new)
                    self.parent.dataset.metadata['zero_loss']['shifted'] = shifts
                    self.parent.plot()


    def set_action(self):
        self.info_tab[0, 0].observe(self.update_dataset, names='value')
        self.info_tab[1, 0].on_click(self.cursor2energy_scale)
        self.info_tab[2, 0].observe(self.set_energy_scale, names='value')
        self.info_tab[3, 0].observe(self.set_energy_scale, names='value')
        self.info_tab[5, 0].observe(self.set_microscope_parameter)
        self.info_tab[6, 0].observe(self.set_microscope_parameter)
        self.info_tab[7, 0].observe(self.set_microscope_parameter)
        self.info_tab[9, 0].observe(self.set_flux, names='value')
        self.info_tab[9, 2].observe(self.set_y_scale, names='value')
        self.info_tab[10, 0].observe(self.set_flux)
        self.info_tab[14, 0].on_click(self.get_shift)
        self.info_tab[14, 1].on_click(self.shift_spectrum)
        self.info_tab[14, 1].on_click(self.shift_spectrum)
        
        self.info_tab[16, 0].observe(self.set_binning)
        self.info_tab[17, 0].observe(self.set_binning)
