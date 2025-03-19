"""
Author: Gerd Duscher
"""


import numpy as np
import warnings

import ipywidgets
import IPython.display
# from IPython.display import display
import matplotlib
import matplotlib.pylab as plt
import matplotlib.patches as patches

from pyTEMlib import file_tools as ft
from pyTEMlib import eels_tools as eels
from pyTEMlib import eels_dialog_utilities

import sidpy


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

def get_core_loss_sidebar():
    side_bar = ipywidgets.GridspecLayout(14, 3,width='auto', grid_gap="0px")

    
    row = 0
    side_bar[row, :3] = ipywidgets.ToggleButton(description='Fit Area',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     tooltip='Shows fit regions and regions excluded from fit', 
                     button_style='info') #ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=7.5,description='Fit Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Fit End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='20px'))
    
    row += 1
    
    side_bar[row, :3] = ipywidgets.Button(description='Elements',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=[('Edge 1', 0), ('Edge 2', 1), ('Edge 3', 2), ('Edge 4', 3),('Add Edge', -1)],
            value=0,
            description='Edges:',
            disabled=False,
            layout=ipywidgets.Layout(width='200px'))
    """side_bar[row,2] = ipywidgets.ToggleButton(
            description='Regions',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Shows fit regions and regions excluded from fit', 
            layout=ipywidgets.Layout(width='100px')
        )
    """
    row += 1
    side_bar[row, :2] = ipywidgets.IntText(value=7.5,description='Z:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
            options=['K1','L3', 'M5', 'M3', 'M1', 'N7', 'N5', 'N3', 'N1'],
            value='K1',
            description='Symmetry:',
            disabled=False,
            layout=ipywidgets.Layout(width='200px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Onset:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Excl.Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Excl.End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(value=0.1, description='Mutliplier:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(value="a.u.", layout=ipywidgets.Layout(width='100px'))
    row += 1
    
    side_bar[row, :3] = ipywidgets.Button(description='Quantification',
                     layout=ipywidgets.Layout(width='auto', grid_area='header'),
                     style=ipywidgets.ButtonStyle(button_color='lightblue'))
    
    row += 1
    side_bar[row,0] = ipywidgets.ToggleButton(
            description='Probabiity',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Changes y-axis to probability of flux is given', 
            layout=ipywidgets.Layout(width='100px')
        )
    side_bar[row,1] = ipywidgets.ToggleButton(
            description='Conv.LL',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Changes y-axis to probability of flux is given', 
            layout=ipywidgets.Layout(width='100px')
        )
    side_bar[row,2] = ipywidgets.ToggleButton(
            description='Show Edges',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Changes y-axis to probability of flux is given', 
            layout=ipywidgets.Layout(width='100px')
        )
    
    row += 1
    side_bar[row,0] = ipywidgets.ToggleButton(
            description='Do All',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Fits all spectra of spectrum image', 
            layout=ipywidgets.Layout(width='100px')
            )

    side_bar[row,1] = ipywidgets.IntProgress(value=0, min=0, max=10, description=' ', bar_style='', # 'success', 'info', 'warning', 'danger' or ''
                                             style={'bar_color': 'maroon'}, orientation='horizontal')
    return side_bar



class CompositionWidget(object):
    def __init__(self, datasets=None):
        
        if not isinstance(datasets, dict):
            raise TypeError('dataset or first item has to be a sidpy dataset')
        self.datasets = datasets

        
        self.model = []
        self.sidebar = get_core_loss_sidebar()
        
        self.set_dataset()
        
        self.periodic_table = eels_dialog_utilities.PeriodicTableWidget(self.energy_scale)
        self.elements_cancel_button = ipywidgets.Button(description='Cancel')
        self.elements_select_button = ipywidgets.Button(description='Select')
        self.elements_auto_button = ipywidgets.Button(description='Auto ID')
       
        self.periodic_table_panel = ipywidgets.VBox([self.periodic_table.periodic_table,
                                                     ipywidgets.HBox([self.elements_cancel_button, self.elements_auto_button, self.elements_select_button])])
        
                                      
        self.app_layout = ipywidgets.AppLayout(
            left_sidebar=self.sidebar,
            center=self.view.panel,
            footer=None,#message_bar,
            pane_heights=[0, 10, 0],
            pane_widths=[4, 10, 0],
        )
        self.set_action()
        IPython.display.display(self.app_layout)

        
    def line_select_callback(self, x_min, x_max):
            self.start_cursor.value = np.round(x_min,3)
            self.end_cursor.value = np.round(x_max, 3)

            self.start_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.start_cursor.value)
            self.end_channel = np.searchsorted(self.datasets[self.key].energy_loss, self.end_cursor.value)
       
            
    def plot(self, scale=True):
        self.view.change_y_scale = self.change_y_scale
        self.view.y_scale = self.y_scale
        self.energy_scale = self.dataset.energy_loss.values
        
        if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
            spectrum = self.view.get_spectrum()
        else:
            spectrum = self.dataset
        if len(self.model) > 1:
            additional_spectra = {'model': self.model,
                                  'difference': spectrum-self.model}   
        else:
            additional_spectra = None
        self.view.plot(scale=True, additional_spectra=additional_spectra )
        self.change_y_scale = 1.
    
        if self.sidebar[12, 2].value:
            self.show_edges()
        if self.sidebar[0, 0].value:
            self.plot_regions()
        self.view.figure.canvas.draw_idle()
        
        
    def plot_regions(self):
        axis = self.view.figure.gca()
        y_min, y_max = axis.get_ylim()
        height = y_max - y_min

        rect = []
        if 'fit_area' in self.edges:
            color = 'blue'
            alpha = 0.2
            x_min = self.edges['fit_area']['fit_start']
            width = self.edges['fit_area']['fit_end'] - x_min
            rect.append(patches.Rectangle((x_min, y_min), width, height,
                                          edgecolor=color, alpha=alpha, facecolor=color))
            axis.add_patch(rect[0])
            axis.text(x_min, y_max, 'fit region', verticalalignment='top')
        color = 'red'
        alpha = 0.5
        
        for key in self.edges:
            if key.isdigit():
                x_min = self.edges[key]['start_exclude']
                width = self.edges[key]['end_exclude']-x_min
                rect.append(patches.Rectangle((x_min, y_min), width, height,
                                              edgecolor=color, alpha=alpha, facecolor=color))
                axis.add_patch(rect[-1])
                axis.text(x_min, y_max, f"exclude\n edge {int(key)+1}", verticalalignment='top')

    def show_edges(self):
        axis = self.view.figure.gca()
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()
        
        for key, edge in self.edges.items():
            i = 0
            if key.isdigit():
                element = edge['element']
                for sym in edge['all_edges']:
                    x = edge['all_edges'][sym]['onset'] + edge['chemical_shift']
                    if x_min < x < x_max:
                        axis.text(x, y_max, '\n' * i + f"{element}-{sym}",
                                       verticalalignment='top', color='black')
                        axis.axvline(x, ymin=0, ymax=1, color='gray')
                        i += 1

        
    
        
    def set_dataset(self, set_key=None):
        spectrum_list = []
        self.spectrum_keys_list = []
        reference_list =[('None', -1)]
        
        for index, key in enumerate(self.datasets.keys()):
            if '_rel' not in key:
                if 'Reference' not in key :
                    if 'SPECTR' in self.datasets[key].data_type.name:
                        spectrum_list.append((f'{key}: {self.datasets[key].title}', index)) 
                        self.spectrum_keys_list.append(key)
                        if key == self.parent.coreloss_key:
                            self.key = key
                            self.coreloss_key = self.key
                            dataset_index = len(spectrum_list)-1

                reference_list.append((f'{key}: {self.datasets[key].title}', index))  
        self.sidebar[0, 0].options = spectrum_list
        self.sidebar[0, 0].value = dataset_index

        if self.coreloss_key is None:
            return
        self.dataset = self.datasets[self.coreloss_key]
        
        self.spec_dim = self.dataset.get_spectral_dims(return_axis=True)[0]

        self.energy_scale = self.spec_dim.values
        self.dd = (self.energy_scale[0], self.energy_scale[1])

        self.dataset.metadata['experiment']['offset'] = self.energy_scale[0]
        self.dataset.metadata['experiment']['dispersion'] = self.spec_dim.slope
        if 'edges' not in self.dataset.metadata or self.dataset.metadata['edges'] == {}:
            self.dataset.metadata['edges'] = {'0': {}, 'model': {}, 'use_low_loss': False}
       
        self.edges = self.dataset.metadata['edges']
        if '0' not in self.edges:
            self.edges['0'] = {}
        
        if 'fit_area' not in self.edges:
            self.edges['fit_area'] = {}
        if 'fit_start' not in self.edges['fit_area']:
            self.sidebar[1,0].value = np.round(self.energy_scale[50], 3)
            self.edges['fit_area']['fit_start'] = self.sidebar[1,0].value 
        else:
            self.sidebar[1,0].value = np.round(self.edges['fit_area']['fit_start'],3)
        if 'fit_end' not in self.edges['fit_area']:
            self.sidebar[2,0].value = np.round(self.energy_scale[-2], 3)
            self.edges['fit_area']['fit_end'] = self.sidebar[2,0].value 
        else:
            self.sidebar[2,0].value = np.round(self.edges['fit_area']['fit_end'],3)
        
        if self.dataset.data_type.name == 'SPECTRAL_IMAGE':
            if 'SI_bin_x' not in self.dataset.metadata['experiment']:
                self.dataset.metadata['experiment']['SI_bin_x'] = 1
                self.dataset.metadata['experiment']['SI_bin_y'] = 1

            bin_x = self.dataset.metadata['experiment']['SI_bin_x']
            bin_y = self.dataset.metadata['experiment']['SI_bin_y']
            # self.dataset.view.set_bin([bin_x, bin_y])
        if self.dataset.data_type.name =='SPECTRAL_IMAGE':
            self.view = eels_dialog_utilities.SIPlot(self.dataset)
        else:
            self.view = eels_dialog_utilities.SpectrumPlot(self.dataset)    
        self.y_scale = 1.0
        self.change_y_scale = 1.0
        
        self.update()
        
    def update_element(self, z=0, index=-1):
        # We check whether this element is already in the
        if z == 0:
            z = self.sidebar[5,0].value
    
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
                    elif key in self.sidebar[6,0].options:
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
        if index == -1:
            index = self.sidebar[4, 0].value
        # self.ui.dialog.setWindowTitle(f'{index}, {zz}')

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

    def set_elements(self, value=0):
        selected_elements = self.periodic_table.get_output()
        edges = self.edges.copy()
        to_delete = []
        old_elements = []
        if len(selected_elements) > 0:
            for key in self.edges:
                if key.isdigit():
                    to_delete.append(key)
                    old_elements.append(self.edges[key]['element'])

        for key in to_delete:
            edges[key] = self.edges[key]
            del self.edges[key]
    
        for index, elem in enumerate(selected_elements):
            if elem  in old_elements:
                self.edges[str(index)] = edges[str(old_elements.index(elem))]   
            else:
                self.update_element(elem, index=index)
        self.sort_elements()
        self.update()
        self.set_figure_pane()

    def set_element(self, elem):
        self.update_element(self.sidebar[5, 0].value)
        # self.sort_elements()
        self.update()
       
    def cursor2energy_scale(self, value):
        dispersion = (self.end_cursor.value - self.start_cursor.value) / (self.end_channel - self.start_channel)
        self.datasets[self.key].energy_loss *= (self.sidebar[3, 0].value/dispersion)
        self.sidebar[3, 0].value = dispersion
        offset = self.start_cursor.value - self.start_channel * dispersion
        self.datasets[self.key].energy_loss += (self.sidebar[2, 0].value-self.datasets[self.key].energy_loss[0])
        self.sidebar[2, 0].value = offset
        self.plot()
        
    def set_fit_area(self, value):
        if self.sidebar[1,0].value > self.sidebar[2,0].value:
            self.sidebar[1,0].value = self.sidebar[2,0].value -1
        if self.sidebar[1,0].value < self.energy_scale[0]:
            self.sidebar[1,0].value = self.energy_scale[0]
        if self.sidebar[2,0].value > self.energy_scale[-1]:
            self.sidebar[2,0].value = self.energy_scale[-1]
        self.edges['fit_area']['fit_start'] = self.sidebar[1,0].value 
        self.edges['fit_area']['fit_end'] = self.sidebar[2,0].value 
        
        self.plot()
        
    def set_y_scale(self, value):  
        self.change_y_scale = 1/self.y_scale
        self.y_scale = 1.0
        if self.dataset.metadata['experiment']['flux_ppm'] > 0:
            if self.sidebar[12, 0].value:
                dispersion = self.energy_scale[1] - self.energy_scale[0]
                self.y_scale = 1/self.dataset.metadata['experiment']['flux_ppm'] * dispersion
            
        self.change_y_scale *= self.y_scale
        self.update()
        self.plot()

    def auto_id(self, value=0):
        found_edges = eels.auto_id_edges(self.dataset)
        if len(found_edges) > 0:
            self.periodic_table.elements_selected = found_edges
            self.periodic_table.update()
        
    def find_elements(self, value=0):
        
        if '0' not in self.edges:
            self.edges['0'] = {}
        # found_edges = eels.auto_id_edges(self.dataset)
        found_edges = {}

        selected_elements = []
        elements = self.edges.copy()

        for key in self.edges:
            if key.isdigit():
                if 'element' in self.edges[key]:
                    selected_elements.append(self.edges[key]['element'])
        self.periodic_table.elements_selected = selected_elements
        self.periodic_table.update()
        self.app_layout.center = self.periodic_table_panel # self.periodic_table.periodic_table

    def set_figure_pane(self, value=0):
        
        self.app_layout.center = self.view.panel
    
    def update(self, index=0):
        
        index = self.sidebar[4,0].value  # which edge
        if index < 0:
            options  = list(self.sidebar[4,0].options)
            options.insert(-1, (f'Edge {len(self.sidebar[4,0].options)}', len(self.sidebar[4,0].options)-1))
            self.sidebar[4,0].options= options
            self.sidebar[4,0].value = len(self.sidebar[4,0].options)-2
        if str(index) not in self.edges:
            self.edges[str(index)] = {'z': 0,  'element': 'x', 'symmetry': 'K1', 'onset': 0, 'start_exclude': 0, 'end_exclude':0,
                                     'areal_density': 0, 'chemical_shift':0}
        if 'z' not in self.edges[str(index)]:
             self.edges[str(index)] = {'z': 0,  'element': 'x', 'symmetry': 'K1', 'onset': 0, 'start_exclude': 0, 'end_exclude':0,
                                      'areal_density': 0, 'chemical_shift':0}
        edge = self.edges[str(index)]
            
        self.sidebar[5,0].value = edge['z']
        self.sidebar[5,2].value = edge['element']
        self.sidebar[6,0].value = edge['symmetry']
        self.sidebar[7,0].value = edge['onset']
        self.sidebar[8,0].value = edge['start_exclude']
        self.sidebar[9,0].value = edge['end_exclude']
        if self.y_scale == 1.0:
            self.sidebar[10, 0].value = edge['areal_density']
            self.sidebar[10, 2].value =  'a.u.'
        else:
            dispersion = self.energy_scale[1]-self.energy_scale[0]
            self.sidebar[10, 0].value = np.round(edge['areal_density']/self.dataset.metadata['experiment']['flux_ppm']*1e-6, 2)
            self.sidebar[10, 2].value = 'atoms/nm²'
        
    
    def do_fit(self, value=0):
        if 'experiment' in self.dataset.metadata:
            exp = self.dataset.metadata['experiment']
            if 'convergence_angle' not in exp:
                raise ValueError('need a convergence_angle in experiment of metadata dictionary ')
            alpha = exp['convergence_angle']
            beta = exp['collection_angle']
            beam_kv = exp['acceleration_voltage']

        else:
            raise ValueError('need a experiment parameter in metadata dictionary')
        
        eff_beta = eels.effective_collection_angle(self.energy_scale, alpha, beta, beam_kv)

        self.low_loss = None
        if self.sidebar[12, 1].value:
            for key in self.datasets.keys():
                if key != self.key:
                    if isinstance(self.datasets[key], sidpy.Dataset):
                        if self.datasets[key].data_type.name == 'SPECTRUM':
                            if self.datasets[key].energy_loss[0] < 0:
                                self.low_loss = self.datasets[key]/self.datasets[key].sum()

        edges = eels.make_cross_sections(self.edges, np.array(self.energy_scale), beam_kv, eff_beta, self.low_loss)

        if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
            spectrum = self.view.get_spectrum()
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

    def do_all_button_click(self, value=0):
            if self.sidebar[13,0].value==False:
                return
            
            if self.dataset.data_type.name != 'SPECTRAL_IMAGE':
                self.do_fit()
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

            eff_beta = eels.effective_collection_angle(self.energy_scale, alpha, beta, beam_kv)
            eff_beta = beta
            self.low_loss = None
            if self.sidebar[12, 1].value:
                for key in self.datasets.keys():
                    if key != self.key:
                        if isinstance(self.datasets[key], sidpy.Dataset):
                            if 'SPECTR' in self.datasets[key].data_type.name:
                                if self.datasets[key].energy_loss[0] < 0:
                                    self.low_loss = self.datasets[key]/self.datasets[key].sum()

            edges = eels.make_cross_sections(self.edges, np.array(self.energy_scale), beam_kv, eff_beta, self.low_loss)
    
            view = self.view
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
            self.sidebar[13,1].max = total_spec
            #self.ui.progress.setMaximum(total_spec)
            #self.ui.progress.setValue(0)
            ind = 0
            for x in range(int(self.dataset.shape[0]/bin_x)):
                for y in range(int(self.dataset.shape[1]/bin_y)):
                    ind += 1
                    self.sidebar[13,1].value = ind
                    view.x = x*bin_x
                    view.y = y*bin_y
                    spectrum = view.get_spectrum()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        edges = eels.fit_edges2(spectrum, self.energy_scale, edges)
                    for key, edge in edges.items():
                        if key.isdigit():
                            # element.append(edge['element'])
                            results[x, y, int(key)] = edge['areal_density']
            edges['spectrum_image_quantification'] = results
            self.sidebar[13,1].value = total_spec
            view.x = start_x
            view.y = start_y
            self.sidebar[13,0].value = False
            
    
    def modify_onset(self, value=-1):
        edge_index = self.sidebar[4, 0].value
        edge = self.edges[str(edge_index)]
        edge['onset'] = self.sidebar[7,0].value
        if 'original_onset' not in edge:
            edge['original_onset'] = edge['onset']
        edge['chemical_shift'] = edge['onset'] -  edge['original_onset']
        self.update()
        
            
    def modify_start_exclude(self, value=-1):
        edge_index = self.sidebar[4, 0].value
        edge = self.edges[str(edge_index)]
        edge['start_exclude'] = self.sidebar[8,0].value
        self.plot()
        
    def modify_end_exclude(self, value=-1):
        edge_index = self.sidebar[4, 0].value
        edge = self.edges[str(edge_index)]
        edge['end_exclude'] = self.sidebar[9,0].value
        self.plot()
    
    def modify_areal_density(self, value=-1):
        edge_index = self.sidebar[4, 0].value
        edge = self.edges[str(edge_index)]
        
        edge['areal_density'] = self.sidebar[10, 0].value
        if self.y_scale != 1.0:
            dispersion = self.energy_scale[1]-self.energy_scale[0]
            edge['areal_density'] = self.sidebar[10, 0].value *self.dataset.metadata['experiment']['flux_ppm']/1e-6

        self.model = self.edges['model']['background']
        for key in self.edges:
            if key.isdigit():
                if 'data' in self.edges[key]:

                    self.model = self.model + self.edges[key]['areal_density'] * self.edges[key]['data']
        self.plot()

    def set_action(self):
        self.sidebar[1, 0].observe(self.set_fit_area, names='value')
        self.sidebar[2, 0].observe(self.set_fit_area, names='value')
        
        self.sidebar[3, 0].on_click(self.find_elements)
        self.sidebar[4, 0].observe(self.update, names='value')
        self.sidebar[5, 0].observe(self.set_element, names='value')

        self.sidebar[7, 0].observe(self.modify_onset, names='value')
        self.sidebar[8, 0].observe(self.modify_start_exclude, names='value')
        self.sidebar[9, 0].observe(self.modify_end_exclude, names='value')
        self.sidebar[10, 0].observe(self.modify_areal_density, names='value')
        
        self.sidebar[11, 0].on_click(self.do_fit)
        self.sidebar[12, 2].observe(self.plot, names='value')
        self.sidebar[0, 0].observe(self.set_dataset, names='value')
        self.sidebar[12,0].observe(self.set_y_scale, names='value')
        self.sidebar[13,0].observe(self.do_all_button_click, names='value')

        self.elements_cancel_button.on_click(self.set_figure_pane)
        self.elements_auto_button.on_click(self.auto_id)
        self.elements_select_button.on_click(self.set_elements)
