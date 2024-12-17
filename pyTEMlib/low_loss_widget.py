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
    side_bar = ipywidgets.GridspecLayout(17, 3, width='auto', grid_gap="0px")

    side_bar[0, :2] = ipywidgets.Dropdown(
            options=[('None', 0)],
            value=0,
            description='Low-Loss:',
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
    side_bar[row, 1] = ipywidgets.ToggleButton(description='Do All',
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
    
    
    side_bar[row, 1:3] = ipywidgets.IntProgress(value=0, min=0, max=10, description=' ', bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                              style={'bar_color': 'maroon'}, orientation='horizontal')
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
                    ll_index = index-1

        if ll_index >len(spectrum_list) - 1:
            ll_index = len(spectrum_list) - 1

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
        
        zero_loss_fit_width=self.low_loss_tab[2, 0].value
        spectrum = self.parent.spectrum
        if 'zero_loss' not in self.parent.datasets.keys():
            self.parent.datasets['zero_loss'] = self.parent.dataset.copy()*0
        # if 'zero_loss' not in self.parent.datasets['zero_loss'].metadata.keys():
        self.parent.datasets['zero_loss'].metadata['zero_loss']={}
        self.parent.datasets['zero_loss'].metadata['zero_loss']['parameter'] = np.zeros([self.dataset.shape[0], self.dataset.shape[1], 6])
          
           
        res  = eels_tools.get_resolution_functions(spectrum, startFitEnergy=-zero_loss_fit_width, endFitEnergy=zero_loss_fit_width)
        if len(self.parent.datasets['zero_loss'].shape) > 2:
            self.parent.datasets['zero_loss'][self.parent.x, self.parent.y] = np.array(res)
            self.parent.datasets['zero_loss'].metadata['zero_loss'][self.parent.x, self.parent.y] = res.metadata['zero_loss']['fit_parameter']
        else:
            self.parent.datasets['zero_loss'] = res
            self.parent.datasets['zero_loss'].metadata['zero_loss'].update(res.metadata['zero_loss'])

        self.parent.datasets['_relationship']['resolution_function'] = 'zero_loss'
        
        self.parent.dataset.metadata['zero_loss'].update(self.parent.datasets['zero_loss'].metadata['zero_loss'])
        
        if self.low_loss_tab[3, 0].value: 
            self.parent._update()
        else:
            self.low_loss_tab[3, 0].value = True
        self.low_loss_tab[14, 1].value = np.round(np.log(self.parent.spectrum.sum()/res.sum()), 4)
        self.parent.status_message('Fitted zero-loss peak')
        
    def get_drude(self, value=0):
        self.low_loss_tab[8, 0].value = False
        fit_start = self.low_loss_tab[5, 0].value
        fit_end = self.low_loss_tab[6, 0].value
        if 'plasmon' not in self.parent.datasets.keys():
            self.parent.datasets['plasmon'] = self.parent.dataset.copy()*0
        if 'plasmon' not in self.parent.datasets['plasmon'].metadata.keys():
            self.parent.datasets['plasmon'].metadata['plasmon'] = {}
        if 'fit_parameter' not in self.parent.datasets['plasmon'].metadata['plasmon'].keys():
            if len(self.dataset.shape) > 2:
                self.parent.datasets['plasmon'].metadata['plasmon']['fit_parameter'] = np.zeros([self.dataset.shape[0], self.dataset.shape[1], 4])
                self.parent.datasets['plasmon'].metadata['plasmon']['IMFP'] = np.zeros([self.dataset.shape[0], self.dataset.shape[1]])
            
        if 'low_loss_model' not in self.parent.datasets.keys():
            self.parent.datasets['low_loss_model'] = self.parent.dataset.copy()*0
        self.parent.status_message(str(self.parent.x))
        plasmon = eels_tools.fit_plasmon(self.parent.spectrum, fit_start, fit_end)
        
        p = plasmon.metadata['plasmon']['parameter']
        p = list(np.abs(p))
        p.append(self.low_loss_tab[14, 0].value)

        
        anglog, _, _ = eels_tools.angle_correction(self.parent.spectrum)
        
        low_loss = eels_tools.multiple_scattering(self.parent.energy_scale, p) * anglog
            

        if len(self.parent.datasets['plasmon'].shape) > 2:
            self.parent.datasets['plasmon'][self.parent.x, self.parent.y] = np.array(plasmon)
            self.parent.datasets['low_loss_model'][self.parent.x, self.parent.y] = np.array(low_loss)
            self.parent.datasets['plasmon'].metadata['plasmon']['fit_parameter'][self.parent.x, self.parent.y] = p
        
            if 'zero_loss' in self.parent.datasets:
                res = self.parent.datasets['zero_loss'][self.parent.x, self.parent.y]
           
        else:
            self.parent.datasets['plasmon'] = plasmon
            self.parent.datasets['low_loss_model'] = low_loss
            if 'zero_loss' in self.parent.datasets:
                res = self.parent.datasets['zero_loss']
        self.parent.datasets['_relationship']['plasmon'] = 'plasmon'
        self.parent.datasets['_relationship']['low_loss_model'] = 'low_loss_model'
        
        #self.dataset.metadata['plasmon'].update(self.parent.datasets['plasmon'].metadata['zero_loss'])
        if self.low_loss_tab[10, 0].value:
            self.parent._update() 
            self._update()
        else:
            self.low_loss_tab[10, 0].value = True
        
        self.low_loss_tab[7, 0].value = np.round(np.abs(p[0]),3)
        self.low_loss_tab[8, 0].value = np.round(p[1],3)
        self.low_loss_tab[9, 0].value = np.round(p[2],1)

        _, dsdo, _ = eels_tools.angle_correction(self.parent.spectrum)

        if 'zero_loss' in self.parent.datasets:
            I0 = res.sum() + p[2] 
        else:
            I0 = self.parent.spectrum.sum()
        # I0 = self.parent.spectrum.sum()
        # print(I0)
        # T = m_0 v**2 !!!  a_0 = 0.05292 nm p[2] = S(E)/elf
        t_nm  = p[2]/I0*dsdo  #Egerton equ 4.26% probability per eV
        relative_thickness = self.low_loss_tab[14, 0].value
        imfp, _ = eels_tools.inelatic_mean_free_path(p[0], self.parent.spectrum)
        t_nm = float(relative_thickness * imfp)
        # print(t_nm, relative_thickness, imfp)
        self.parent.status_message(f'Fitted plasmon peak: thickness :{t_nm:.1f} nm and IMFP: {t_nm/relative_thickness:.1f} nm in free electron approximation')

        if self.dataset.ndim>1:
            # self.parent.datasets['plasmon'].metadata['plasmon'][self.parent.x, self.parent.y]['thickness'] = t_nm
            # self.parent.datasets['plasmon'].metadata['plasmon'][self.parent.x, self.parent.y]['relative_thickness'] = relative_thickness
            self.parent.datasets['plasmon'].metadata['plasmon']['IMFP'][self.parent.x, self.parent.y] = t_nm/relative_thickness

        else:
            self.parent.datasets['plasmon'].metadata['plasmon']['thickness'] = t_nm
            self.parent.datasets['plasmon'].metadata['plasmon']['relative_thickness'] = relative_thickness
            self.parent.datasets['plasmon'].metadata['plasmon']['IMFP'] = t_nm/relative_thickness


    def multiple_scattering(self, value=0):
        if self.dataset.ndim >1:
            anglog, dsdo, _ = eels_tools.angle_correction(self.parent.spectrum)
            par = np.array(self.parent.datasets['plasmon'].metadata['plasmon']['fit_parameter'])
            for x in range(self.parent.dataset.shape[0]):
                for y in range(self.parent.dataset.shape[1]):
                    self.parent.datasets['low_loss_model'][x, y] = eels_tools.multiple_scattering(self.parent.energy_scale, par[x, y]) * anglog
        

    def do_all(self, value=0):
        if len(self.parent.dataset.shape) < 3:
            return
            
        zero_loss_fit_width=self.low_loss_tab[2, 0].value
        fit_start = self.low_loss_tab[5, 0].value
        fit_end = self.low_loss_tab[6, 0].value
        
        
        if 'low_loss_model' not in self.parent.datasets.keys():
            self.parent.datasets['low_loss_model'] = self.parent.dataset.copy()*0
            self.parent.datasets['low_loss_model'].title = self.parent.dataset.title + ' low_loss_model'
        
        self.low_loss_tab[15,1].max = self.parent.dataset.shape[0]*self.parent.dataset.shape[1]
        
        self.parent.datasets['zero_loss']  = eels_tools.get_resolution_functions(self.dataset, startFitEnergy=-zero_loss_fit_width, endFitEnergy=zero_loss_fit_width)
        self.parent.datasets['zero_loss'].title = self.parent.dataset.title + ' zero_loss'
        self.parent.status_message('Fitted zero-loss peak')

        self.parent.datasets['plasmon'] = eels_tools.fit_plasmon(self.dataset, fit_start, fit_end)
        self.parent.datasets['plasmon'].title = self.parent.dataset.title + ' plasmon'
        
        self.parent.status_message('Fitted zero-loss + plasmon peak')

        
        """
        anglog, _, _ = eels_tools.angle_correction(self.parent.spectrum)
        i = 0
        for x in range(self.parent.dataset.shape[0]):   
            for y in range(self.parent.dataset.shape[1]):
                self.low_loss_tab[15,1].value = i
                i+= 1

                spectrum = self.parent.dataset[x, y]
                
                plasmon = eels_tools.fit_plasmon(spectrum, fit_start, fit_end)
                p =np.abs(plasmon.metadata['plasmon']['parameter'])
                p = list(np.abs(p))
                
                p.append(np.log(spectrum.sum()/self.parent.datasets['zero_loss'][x,y].sum()))
                if p[-1] is np.nan:
                    p[-1] = 0
                low_loss = eels_tools.multiple_scattering(self.parent.energy_scale, p) * anglog
                self.parent.datasets['plasmon'][x, y] = np.array(plasmon.compute())
                self.parent.datasets['low_loss_model'][x, y] = np.array(low_loss)
                drude_p[x, y, :] = np.array(p)

       

        self.parent.datasets['plasmon'].metadata['plasmon'].update({'parameter': drude_p})
        self.parent.datasets['low_loss_model'].metadata['low_loss'] = ({'parameter': drude_p})
        """

        imfp = np.log(self.parent.dataset.sum(axis=2)/self.parent.datasets['zero_loss'].sum(axis=2)) 
        self.parent.datasets['plasmon'].metadata['plasmon']['fit_parameter'] = np.append(self.parent.datasets['plasmon'].metadata['plasmon']['fit_parameter'], imfp[..., np.newaxis], axis=2)
        E_p = self.parent.datasets['plasmon'].metadata['plasmon']['fit_parameter'][:,:,0]    
        self.parent.datasets['plasmon'].metadata['plasmon']['IMFP'], _ = eels_tools.inelatic_mean_free_path(E_p, self.parent.spectrum)
        self.parent.datasets['_relationship']['zero_loss'] = 'zero_loss'
        self.parent.datasets['_relationship']['plasmon'] = 'plasmon'
        self.multiple_scattering()        
        self.parent.datasets['_relationship']['low_loss_model'] = 'low_loss_model'
        
        self.low_loss_tab[10, 1].value = False
            
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
        self.low_loss_tab[10, 1].observe(self.do_all, names='value')
        self.low_loss_tab[10, 2].observe(self._update, names='value')
        self.low_loss_tab[11, 0].on_click(self.get_multiple_scattering)
        self.low_loss_tab[15, 0].observe(self._update, names='value')
        
    
    def _update(self, ev=0):
        low_loss = None
        plasmon = None
        resolution_function = None
        if 'zero_loss' in self.parent.added_spectra.keys():
            del self.parent.added_spectra['zero_loss']
        if 'plasmon' in self.parent.added_spectra.keys():
            del self.parent.added_spectra['plasmon']
        if 'low_loss_model' in self.parent.added_spectra.keys():
            del self.parent.added_spectra['low_loss_model']
        
        if self .low_loss_tab[3, 0].value:
            if 'zero_loss' in self.parent.datasets.keys():
                resolution_function = np.array(self.parent.get_additional_spectrum('zero_loss'))
                self.parent.added_spectra.update({'zero_loss': 'resolution'})
        if self.low_loss_tab[10, 0].value:
            if 'plasmon' in self.parent.datasets.keys():
                plasmon = self.parent.get_additional_spectrum('plasmon')
                if len(self.dataset.shape) > 1:
                    p = np.round(plasmon.metadata['plasmon']['fit_parameter'][self.parent.x, self.parent.y], 3)
                    imfp = np.array(plasmon.metadata['plasmon']['IMFP'][self.parent.x, self.parent.y])
                else:
                    p = np.round(plasmon.metadata['plasmon']['fit_parameter'], 3)
                    imfp = plasmon.metadata['plasmon']['IMFP']
                
                self.parent.added_spectra.update({'plasmon': 'plasmon'})
                self.low_loss_tab[7, 1].value =p[0]
                self.low_loss_tab[8, 1].value = p[1]
                self.low_loss_tab[8, 1].value = p[2]
                
                self.low_loss_tab[14, 1].value =p[-1]
                t_nm = float(p[-1] * imfp)
                # print(t_nm, p[-1], imfp)
                self.parent.status_message(f'Fitted plasmon peak: thickness :{t_nm:.1f} nm and IMFP: {imfp:.1f} nm in free electron approximation')

        if self.low_loss_tab[15, 0].value:
            low_loss = np.array(self.parent.get_additional_spectrum('low_loss_model'))
            self.parent.added_spectra.update({'low_loss': 'low_loss'})
        
        if self.low_loss_tab[3, 0].value + self.low_loss_tab[10, 0].value + self.low_loss_tab[15, 0].value > 0:
            self.parent.datasets['_difference'] = np.array(self.parent.spectrum)
            if resolution_function is not None:
                self.parent.datasets['_difference'] -= resolution_function
            if low_loss is not None:
                self.parent.datasets['_difference'] -= low_loss
            else:
                if plasmon is not None:
                    self.parent.datasets['_difference'] -= np.array(plasmon)
            self.parent.added_spectra.update({'_difference': 'difference'})
        else:
            if '_difference' in self.parent.datasets.keys():
                del self.parent.datasets['_difference']
        self.parent._update()

    def get_additional_spectrum(self, key):
        if key not in self.parent.datasets.keys():
            return
        
        if self.parent.datasets[key].data_type == sidpy.DataType.SPECTRUM:
            spectrum = self.parent.datasets[key].copy()
        else:
            image_dims = self.parent.datasets[key].get_dimensions_by_type(sidpy.DimensionType.SPATIAL)
            selection = []
            x = self.parent.x
            y = self.parent.y
            bin_x = self.parent.bin_x
            bin_y = self.parent.bin_y
            for dim, axis in self.parent.datasets[key]._axes.items():
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
