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
    side_bar = ipywidgets.GridspecLayout(16, 3, width='auto', grid_gap="0px")

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
    side_bar[row, :2] = ipywidgets.FloatText(value=5, description='Start Fit:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=25, description='End Fit:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='50px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=5, description='Energy:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=25, description='Width:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='50px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=25, description='Amplitude:', disabled=False, color='black',
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
    row += 1
    side_bar[row, :3] = ipywidgets.Button(description='Multiple Scattering',
                                          layout=ipywidgets.Layout(width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=5, description='Start Fit:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=-1, description='End Fit:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='50px'))
    row +=1
    side_bar[row, :2] = ipywidgets.FloatText(value=25, description='thickness:', disabled=False, color='black',
                                             layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="* iMFP", layout=ipywidgets.Layout(width='50px'))
    
    row +=1
    side_bar[row, 0] = ipywidgets.ToggleButton(description='Plot LowLoss',
                                               disabled=False,
                                               button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                               tooltip='Plots resolution function on right',
                                               layout=ipywidgets.Layout(width='100px'))
    
    side_bar[row, 2] = ipywidgets.ToggleButton(description='Nix',
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
        self.ll_key = ''
        self.update_ll_sidebar()

    def update_ll_sidebar(self):
        spectrum_list = ['None']
        ll_index = 0
        self.ll_key = self.parent.lowloss_key 
        for index, key in enumerate(self.parent.datasets.keys()):
            if isinstance(self.parent.datasets[key], sidpy.Dataset):
                if 'SPECTR' in self.parent.datasets[key].data_type.name:
                    energy_offset = self.parent.datasets[key].get_spectral_dims(return_axis=True)[0][0]
                    if energy_offset < 0:
                        spectrum_list.append(f'{key}: {self.parent.datasets[key].title}') 
                if key == self.ll_key:
                    ll_index = index+1
        self.low_loss_tab[0, 0].options = spectrum_list
        self.low_loss_tab[0, 0].value = spectrum_list[ll_index]
        
        self.update_ll_dataset()
        
    def update_ll_dataset(self, value=0):
        self.ll_key = self.low_loss_tab[0, 0].value.split(':')[0]
        self.parent.lowloss_key = self.ll_key
        if 'None' in self.ll_key:
            return
        self.parent.set_dataset(self.ll_key)
        self.dataset = self.parent.dataset
        if self.low_loss_tab[13, 0].value < 0:
            energy_scale = self.dataset.get_spectral_dims(return_axis=True)[0]
            self.low_loss_tab[13, 0].value = np.round(self.dataset.get_spectral_dims(return_axis=True)[0][-2], 3)

        
    def get_resolution_function(self, value=0):
        self.low_loss_tab[4, 0].value = False
        zero_loss_fit_width=self.low_loss_tab[2, 0].value
        spectrum = self.parent.spectrum
        self.parent.datasets['resolution_function'] = eels_tools.get_resolution_functions(spectrum,
                                                                                           startFitEnergy=-zero_loss_fit_width, 
                                                                                           endFitEnergy=zero_loss_fit_width)
        self.parent.datasets['_relationship']['resolution_function'] = 'resolution_function'
        if 'low_loss' not in self.dataset.metadata:
            self.dataset.metadata['zero_loss'] = {}
        self.dataset.metadata['zero_loss'].update(self.parent.datasets['resolution_function'].metadata['zero_loss'])
        self.low_loss_tab[3, 0].value = True
        self.low_loss_tab[14, 1].value = np.round(np.log(self.parent.dataset.sum()/self.parent.datasets['resolution_function'].sum()), 4)
        self.parent.status_message('Fitted zero-loss peak')
        
    def get_drude(self, value=0):
        self.low_loss_tab[8, 0].value = False
        fit_start = self.low_loss_tab[5, 0].value
        fit_end = self.low_loss_tab[6, 0].value
        
        plasmon = eels_tools.fit_plasmon(self.parent.spectrum, fit_start, fit_end)
        

        self.parent.datasets['plasmon'] = plasmon
        self.parent.datasets['_relationship']['plasmon'] = 'plasmon'
        
        #self.dataset.metadata['plasmon'].update(self.parent.datasets['plasmon'].metadata['zero_loss'])
        self.low_loss_tab[10, 0].value = True
        p = plasmon.metadata['plasmon']['parameter']
        self.low_loss_tab[7, 0].value = np.round(p[0],3)
        self.low_loss_tab[8, 0].value = np.round(p[1],3)
        self.low_loss_tab[9, 0].value = np.round(p[2],1)

        _, dsdo, _ = eels_tools.angle_correction(self.parent.spectrum)


        I0 = self.parent.datasets['resolution_function'].sum() + p[2] 
        # I0 = self.parent.spectrum.sum()
        # print(I0)
        # T = m_0 v**2 !!!  a_0 = 0.05292 nm p[2] = S(E)/elf
        t_nm  = p[2]/I0*dsdo  #Egerton equ 4.26% probability per eV
        relative_thickness = self.low_loss_tab[14, 0].value
        imfp, _ = eels_tools.inelatic_mean_free_path(p[0], self.parent.spectrum)
        t_nm = float(relative_thickness * imfp)
        # print(t_nm, relative_thickness, imfp)
        self.parent.status_message(f'Fitted plasmon peak: thickness :{t_nm:.1f} nm and IMFP: {t_nm/relative_thickness:.1f} nm in free electron approximation')

        plasmon.metadata['plasmon']['thickness'] = t_nm
        plasmon.metadata['plasmon']['relative_thickness'] = relative_thickness
        plasmon.metadata['plasmon']['IMFP'] = t_nm/relative_thickness

        self.parent.spectrum.metadata['plasmon'] = plasmon.metadata['plasmon']
        

    def get_multiple_scattering(self, value=0):
        self.low_loss_tab[15, 0].value = False
        fit_start = self.low_loss_tab[12, 0].value
        fit_end = self.low_loss_tab[13, 0].value
        
        p = [self.low_loss_tab[7, 0].value, self.low_loss_tab[8, 0].value, self.low_loss_tab[9, 0].value, self.low_loss_tab[14, 0].value]
        low_loss = eels_tools.fit_multiple_scattering(self.parent.spectrum, fit_start, fit_end, pin=p)
        

        self.parent.datasets['multiple_scattering'] = low_loss
        self.parent.datasets['_relationship']['multiple_scattering'] = 'multiple_scattering'
        self.low_loss_tab[10, 0].value = False
        self.low_loss_tab[15, 0].value = True
        p = low_loss.metadata['multiple_scattering']['parameter']
        self.low_loss_tab[14, 0].value = np.round(p[3],3)
        
        self.parent.status_message('Fitted multiple scattering')
        

        return low_loss

    def set_ll_action(self):
        self.low_loss_tab[0, 0].observe(self.update_ll_dataset)
        #self.low_loss_tab[1, 0].on_click(self.fix_energy_scale)
        #self.low_loss_tab[2, 0].observe(self.set_energy_scale, names='value')
        #self.low_loss_tab[3, 0].observe(self.set_energy_scale, names='value')
        self.low_loss_tab[1, 0].on_click(self.get_resolution_function)
        self.low_loss_tab[3, 2].observe(self.parent.info.set_y_scale, names='value')
        self.low_loss_tab[3, 0].observe(self._update, names='value')
        self.low_loss_tab[4, 0].on_click(self.get_drude)
        self.low_loss_tab[10, 0].observe(self._update, names='value')
        self.low_loss_tab[10, 2].observe(self._update, names='value')
        self.low_loss_tab[11, 0].on_click(self.get_multiple_scattering)
        self.low_loss_tab[15, 0].observe(self._update, names='value')
        
        
    def _update(self, ev=0):
        
        self.parent._update(ev)
        spectrum = self.parent.spectrum
        anglog, _, _ = eels_tools.angle_correction(spectrum)
        resolution_function = None
        if self .low_loss_tab[3, 0].value:
            if 'resolution_function' in self.parent.datasets:
                resolution_function = self.get_additional_spectrum('resolution_function')
                self.parent.axis.plot(self.parent.energy_scale, resolution_function, label='resolution function')
        if self.low_loss_tab[10, 0].value:
            p = [self.low_loss_tab[7, 0].value, self.low_loss_tab[8, 0].value, self.low_loss_tab[9, 0].value]
            self.parent.datasets['plasmon'] = self.parent.datasets['plasmon'].like_data(eels_tools.energy_loss_function(spectrum.energy_loss, p))*anglog
            plasmon = self.get_additional_spectrum('plasmon')
            self.parent.axis.plot(self.parent.energy_scale, plasmon, label='plasmon')
        else: 
            plasmon = None
        if self.low_loss_tab[15, 0].value:
            p = [self.low_loss_tab[7, 0].value, self.low_loss_tab[8, 0].value, self.low_loss_tab[9, 0].value, self.low_loss_tab[14, 0].value]
            low_loss = eels_tools.multiple_scattering(self.parent.energy_scale, p) * anglog
            self.parent.axis.plot(self.parent.energy_scale, low_loss*self.parent.y_scale, label='multiple scattering')        
        else:
            low_loss = None
        
        difference = spectrum
        if resolution_function is not None:
            difference -= resolution_function
        if low_loss is not None:
            difference -= low_loss *self.parent.y_scale
        else:
            if plasmon is not None:
                difference -= plasmon
        if self.low_loss_tab[3, 0].value + self.low_loss_tab[10, 0].value + self.low_loss_tab[15, 0].value > 0:
            self.parent.axis.plot(self.parent.energy_scale, difference, label='difference')               
            self.parent.axis.legend()

    def get_additional_spectrum(self, key):
        if key not in self.parent.datasets.keys():
            return
        
        if self.parent.datasets[key].data_type == sidpy.DataType.SPECTRUM:
            spectrum = self.parent.datasets[key].copy()
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
            
            spectrum = self.parent.datasets[key][tuple(selection)].mean(axis=tuple(image_dims))
            
        spectrum *= self.parent.y_scale
        
        return spectrum.squeeze()
    
    

    
    def set_binning(self, value):
        if 'SPECTRAL' in self.dataset.data_type.name:
            bin_x = self.info_tab[15, 0].value
            bin_y = self.info_tab[16, 0].value
            self.dataset.view.set_bin([bin_x, bin_y])
            self.datasets[self.key].metadata['experiment']['SI_bin_x'] = bin_x
            self.datasets[self.key].metadata['experiment']['SI_bin_y'] = bin_y
