from typing import Any

import numpy as np
import os
import ipywidgets
import matplotlib.pylab as plt
import matplotlib
from IPython.display import display

import sidpy
# from pyTEMlib.microscope import microscope
from pyTEMlib import file_tools
from pyTEMlib import eels_tools


def get_low_loss_sidebar() -> Any:
    side_bar = ipywidgets.GridspecLayout(9, 3, width='auto', grid_gap="0px")

    side_bar[0, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Main Dataset:',
            disabled=False)
    
    row = 1
    
    side_bar[row, :3] = ipywidgets.Button(description='Resolution Function',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5, description='fit width:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row +=1
    side_bar[row, 0] = ipywidgets.widgets.Label(value="thickness", layout=ipywidgets.Layout(width='100px'))
    side_bar[row, 1] = ipywidgets.widgets.Label(value="", layout=ipywidgets.Layout(width='100px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="* iMFP", layout=ipywidgets.Layout(width='100px'))
    row +=1
    side_bar[row, 0] = ipywidgets.ToggleButton(description='Plot Res.Fct.',
                                               disabled=False,
                                               button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                               tooltip='Plots resolution function on right',
                                               layout=ipywidgets.Layout(width='100px'))
    
    side_bar[row, 2] = ipywidgets.ToggleButton(description='Probability',
                                               disabled=False,
                                               button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                               tooltip='Changes y-axis to probability if flux is given',
                                               layout=ipywidgets.Layout(width='100px'))
    
    
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Drude',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Start Fit:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5, description='End Fit:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='50px'))
    row +=1
    side_bar[row, 0] = ipywidgets.ToggleButton(description='Plot Drude',
                                               disabled=False,
                                               button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                               tooltip='Plots resolution function on right',
                                               layout=ipywidgets.Layout(width='100px'))
    
    side_bar[row, 2] = ipywidgets.ToggleButton(description='Plot Diel.Fct.',
                                               disabled=False,
                                               button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                               tooltip='Changes y-axis to probability if flux is given',
                                               layout=ipywidgets.Layout(width='100px'))
    
    
    return side_bar

class LowLoss(object):
    def __init__(self, sidebar=None, parent=None):
        self.parent = parent
        self.dataset = parent.dataset
        self.low_loss_tab = sidebar
        self.set_ll_action()
        self.update_ll_sidebar()

    def update_ll_sidebar(self):
        spectrum_list = ['None']
        for index, key in enumerate(self.parent.datasets.keys()):
            if isinstance(self.parent.datasets[key], sidpy.Dataset):
                if 'SPECTR' in self.parent.datasets[key].data_type.name:
                    energy_offset = self.parent.datasets[key].get_spectral_dims(return_axis=True)[0][0]
                    if energy_offset < 0:
                        spectrum_list.append(f'{key}: {self.parent.datasets[key].title}') 
        
        self.low_loss_tab[0, 0].options = spectrum_list

    def get_resolution_function(self, value):
        self.low_loss_tab[4, 0].value = False
        zero_loss_fit_width=self.low_loss_tab[2, 0].value
        self.parent.datasets['resolution_functions'] = eels_tools.get_resolution_functions(self.parent.dataset,
                                                                                    startFitEnergy=-zero_loss_fit_width, 
                                                                                    endFitEnergy=zero_loss_fit_width)
        if 'low_loss' not in self.dataset.metadata:
            self.dataset.metadata['zero_loss'] = {}
        self.dataset.metadata['zero_loss'].update(self.parent.datasets['resolution_functions'].metadata['zero_loss'])
        self.low_loss_tab[4, 0].value = True
        self.low_loss_tab[3, 1].value = f"{np.log(self.parent.dataset.sum()/self.parent.datasets['resolution_functions'].sum())}"

    

    def set_ll_action(self):
        self.low_loss_tab[0, 0].observe(self.update_ll_dataset)
        #self.low_loss_tab[1, 0].on_click(self.fix_energy_scale)
        #self.low_loss_tab[2, 0].observe(self.set_energy_scale, names='value')
        #self.low_loss_tab[3, 0].observe(self.set_energy_scale, names='value')
        self.low_loss_tab[1, 0].on_click(self.get_resolution_function)
        self.low_loss_tab[4, 2].observe(self.parent.info.set_y_scale, names='value')
        self.low_loss_tab[4, 0].observe(self._update, names='value')
        
    def _update(self, ev=0):
        self.parent._update(ev)
        
        if self.low_loss_tab[4, 0].value:
            if 'resolution_functions' in self.parent.datasets:
                resolution_function = self.get_additional_spectrum('resolution_functions')
                self.parent.axis.plot(self.parent.energy_scale, resolution_function, label='resolution_function')
                self.parent.axis.plot(self.parent.energy_scale, 
                                      self.parent.spectrum -resolution_function, label='difference')

                self.parent.axis.legend()

    def get_additional_spectrum(self, key):
        if key not in self.parent.datasets.keys():
            return
        
        if self.parent.datasets[key].data_type == sidpy.DataType.SPECTRUM:
            self.spectrum = self.parent.datasets[key].copy()
        else:
            image_dims = self.parent.datasets[key].get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
            selection = []
            for dim, axis in self.parent.datasets[key]._axes.items():
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
            
            self.spectrum = self.parent.datasets[key][tuple(selection)].mean(axis=tuple(image_dims))
            
        self.spectrum *= self.parent.y_scale
        
        return self.spectrum.squeeze()
    
    def update_ll_dataset(self, value=0):
        self.ll_key = self.low_loss_tab[0, 0].value.split(':')[0]
        self.parent.set_dataset(self.ll_key)
        self.dataset = self.parent.dataset

    
    def set_binning(self, value):
        if 'SPECTRAL' in self.dataset.data_type.name:
            bin_x = self.info_tab[15, 0].value
            bin_y = self.info_tab[16, 0].value
            self.dataset.view.set_bin([bin_x, bin_y])
            self.datasets[self.key].metadata['experiment']['SI_bin_x'] = bin_x
            self.datasets[self.key].metadata['experiment']['SI_bin_y'] = bin_y
