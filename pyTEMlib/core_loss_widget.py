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


def get_core_loss_sidebar():
    side_bar = ipywidgets.GridspecLayout(15, 3, width='auto', grid_gap="0px")

    side_bar[0, :2] = ipywidgets.Dropdown(
        options=[('None', 0)],
        value=0,
        description='Main Dataset:',
        disabled=False)

    row = 1
    side_bar[row, :3] = ipywidgets.ToggleButton(description='Fit Area',
                                                layout=ipywidgets.Layout(
                                                    width='auto', grid_area='header'),
                                                tooltip='Shows fit regions and regions excluded from fit',
                                                button_style='info')  # ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(
        value=7.5, description='Fit Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(
        value="eV", layout=ipywidgets.Layout(width='20px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(
        value=0.1, description='Fit End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(
        value="eV", layout=ipywidgets.Layout(width='20px'))

    row += 1

    side_bar[row, :3] = ipywidgets.Button(description='Elements',
                                          layout=ipywidgets.Layout(
                                              width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
        options=[('Edge 1', 0), ('Edge 2', 1), ('Edge 3', 2),
                 ('Edge 4', 3), ('Add Edge', -1)],
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
    side_bar[row, :2] = ipywidgets.IntText(
        value=7.5, description='Z:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(
        value="", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.Dropdown(
        options=['K1', 'L3', 'M5', 'M3', 'M1', 'N7', 'N5', 'N3', 'N1'],
        value='K1',
        description='Symmetry:',
        disabled=False,
        layout=ipywidgets.Layout(width='200px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(
        value=0.1, description='Onset:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(
        value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(
        value=0.1, description='Excl.Start:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(
        value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(
        value=0.1, description='Excl.End:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(
        value="eV", layout=ipywidgets.Layout(width='100px'))
    row += 1
    side_bar[row, :2] = ipywidgets.FloatText(
        value=0.1, description='Mutliplier:', disabled=False, color='black', layout=ipywidgets.Layout(width='200px'))
    side_bar[row, 2] = ipywidgets.widgets.Label(
        value="a.u.", layout=ipywidgets.Layout(width='100px'))
    row += 1

    side_bar[row, :3] = ipywidgets.Button(description='Quantification',
                                          layout=ipywidgets.Layout(
                                              width='auto', grid_area='header'),
                                          style=ipywidgets.ButtonStyle(button_color='lightblue'))

    row += 1
    side_bar[row, 0] = ipywidgets.ToggleButton(
        description='Probabiity',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Changes y-axis to probability of flux is given',
        layout=ipywidgets.Layout(width='100px')
    )
    side_bar[row, 1] = ipywidgets.ToggleButton(
        description='Conv.LL',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Changes y-axis to probability of flux is given',
        layout=ipywidgets.Layout(width='100px')
    )
    side_bar[row, 2] = ipywidgets.ToggleButton(
        description='Show Edges',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Changes y-axis to probability of flux is given',
        layout=ipywidgets.Layout(width='100px')
    )

    row += 1
    side_bar[row, 0] = ipywidgets.ToggleButton(
        description='Do All',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Fits all spectra of spectrum image',
        layout=ipywidgets.Layout(width='100px')
    )

    side_bar[row, 1:3] = ipywidgets.IntProgress(value=0, min=0, max=10, description=' ', bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                              style={'bar_color': 'maroon'}, orientation='horizontal')
    return side_bar


class CoreLoss(object):
    def __init__(self, sidebar=None, parent=None):
        self.parent = parent
        self.dataset = parent.dataset
        self.core_loss_tab = sidebar

        self.model = []
        self.edges = {}
        self.count = 0
        self.cl_key = 'None'

        self.periodic_table = eels_dialog_utilities.PeriodicTableWidget(
            self.parent.energy_scale)
        self.elements_cancel_button = ipywidgets.Button(description='Cancel')
        self.elements_select_button = ipywidgets.Button(description='Select')
        self.elements_auto_button = ipywidgets.Button(description='Auto ID')

        self.periodic_table_panel = ipywidgets.VBox([self.periodic_table.periodic_table,
                                                     ipywidgets.HBox([self.elements_cancel_button, self.elements_auto_button, self.elements_select_button])])

        # self.update_cl_sidebar()
        self.set_cl_action()

    def update_cl_key(self, value=0):
        self.cl_key = self.core_loss_tab[0, 0].value.split(':')[0]
        self.parent.coreloss_key = self.cl_key
        if 'None' in self.cl_key:
            return
        self.parent.set_dataset(self.cl_key)

        self.dataset = self.parent.datasets[self.cl_key]
    

    def update_cl_sidebar(self):
        spectrum_list = ['None']
        cl_index = 0
        self.cl_key = self.parent.coreloss_key
        for index, key in enumerate(self.parent.datasets.keys()):
            if isinstance(self.parent.datasets[key], sidpy.Dataset):
                if 'SPECTR' in self.parent.datasets[key].data_type.name:
                    spectrum_list.append(
                        f'{key}: {self.parent.datasets[key].title}')
                if key == self.cl_key:
                    cl_index = index+1
        self.core_loss_tab[0, 0].options = spectrum_list
        self.core_loss_tab[0, 0].value = spectrum_list[cl_index]
        if '_relationship' in self.parent.datasets.keys():
            self.update_cl_dataset()
            self.set_fit_start()
            self.parent.plot()
        
    def update_cl_dataset(self, value=0):
        self.cl_key = self.core_loss_tab[0, 0].value.split(':')[0]
        self.parent.coreloss_key = self.cl_key
        if '_relationship' in self.parent.datasets.keys():
            self.parent.datasets['_relationship']['core_loss'] = self.cl_key
        
        if 'None' in self.cl_key:
            return
        self.parent.set_dataset(self.cl_key)
        self.dataset = self.parent.dataset

    def line_select_callback(self, x_min, x_max):
        self.start_cursor.value = np.round(x_min, 3)
        self.end_cursor.value = np.round(x_max, 3)

        self.start_channel = np.searchsorted(
            self.datasets[self.cl_key].energy_loss, self.start_cursor.value)
        self.end_channel = np.searchsorted(
            self.datasets[self.cl_key].energy_loss, self.end_cursor.value)

    def plot(self, scale=True):
        self.parent.dataset.metadata['edges'] = self.edges
        self.parent.plot(scale=scale)
        y_scale = self.parent.y_scale
        spectrum = self.parent.spectrum
        if len(self.model) > 1:
            self.model = self.edges['model']['spectrum'].copy()
            # self.parent.axis.plot(self.parent.energy_scale, (self.edges['model']['spectrum'])*y_scale, label='difference')
            self.parent.axis.plot(self.parent.energy_scale,
                                  self.model*y_scale, label='model')
            self.parent.axis.plot(self.parent.energy_scale,
                                  spectrum-self.model*y_scale, label='difference')
            self.parent.axis.legend()
            pass
        if self.core_loss_tab[13, 2].value:
            self.show_edges()
        if self.core_loss_tab[1, 0].value:
            self.plot_regions()
        self.parent.figure.canvas.draw_idle()

    def plot_regions(self):
        axis = self.parent.figure.gca()
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
                axis.text(
                    x_min, y_max, f"exclude\n edge {int(key)+1}", verticalalignment='top')

    def show_edges(self):
        axis = self.parent.figure.gca()
        x_min, x_max = axis.get_xlim()
        y_min, y_max = axis.get_ylim()

        for key, edge in self.edges.items():
            i = 0
            if key.isdigit():
                element = edge['element']
                for sym in edge['all_edges']:
                    x = edge['all_edges'][sym]['onset'] + \
                        edge['chemical_shift']
                    if x_min < x < x_max:
                        axis.text(x, y_max, '\n' * i + f"{element}-{sym}",
                                  verticalalignment='top', color='black')
                        axis.axvline(x, ymin=0, ymax=1, color='gray')
                        i += 1

    def update_element(self, z=0, index=-1):
        # We check whether this element is already in the
        if z == 0:
            z = self.core_loss_tab[6, 0].value

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
                if self.parent.energy_scale[edge_start] < x_section[key]['onset'] < self.parent.energy_scale[-edge_start]:
                    if key in ['K1', 'L3', 'M5']:
                        major_edge = key
                    elif key in self.core_loss_tab[7, 0].options:
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
            index = self.core_loss_tab[5, 0].value
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
        self.edges[str(index)]['original_onset'] = self.edges[str(
            index)]['onset']
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
        dispersion = self.parent.energy_scale[1]-self.parent.energy_scale[0]

        while str(index + 1) in self.edges:
            next_edge = self.edges[str(index + 1)]
            if edge['end_exclude'] > next_edge['start_exclude'] - 5 * dispersion:
                edge['end_exclude'] = next_edge['start_exclude'] - 5 * dispersion
            edge = next_edge
            index += 1

        if edge['end_exclude'] > self.parent.energy_scale[-3]:
            edge['end_exclude'] = self.parent.energy_scale[-3]

    def set_elements(self, value=0):
        selected_elements = self.periodic_table.get_output()
        edges = self.edges.copy()
        to_delete = []
        old_elements = []
        if len(selected_elements) > 0:
            for key in self.edges:
                if key.isdigit():
                    if 'element' in self.edges[key]:
                        to_delete.append(key)
                        old_elements.append(self.edges[key]['element'])

        for key in to_delete:
            edges[key] = self.edges[key]
            del self.edges[key]

        for index, elem in enumerate(selected_elements):
            if elem in old_elements:
                self.edges[str(index)] = edges[str(old_elements.index(elem))]
            else:
                self.update_element(elem, index=index)
        self.sort_elements()
        self.update()
        self.set_figure_pane()

    def set_element(self, elem):
        self.update_element(self.core_loss_tab[6, 0].value)
        # self.sort_elements()
        self.update()

    def set_fit_start(self, value=0):
        if 'edges' not in self.dataset.metadata:
            self.edges = self.dataset.metadata['edges'] = {}
        if 'fit_area' not in self.edges:
            self.edges['fit_area'] = {'fit_start': self.parent.energy_scale[10],
                                      'fit_end': self.parent.energy_scale[-10]}
            self.core_loss_tab[3, 0].value = str(
                self.edges['fit_area']['fit_end'])
            self.core_loss_tab[2, 0].value = str(
                self.edges['fit_area']['fit_start'])
        if self.core_loss_tab[2, 0].value < self.parent.energy_scale[0]:
            self.core_loss_tab[2, 0].value = self.parent.energy_scale[10]
        self.edges['fit_area']['fit_start'] = float(
            self.core_loss_tab[2, 0].value)
        self.parent.plot()

    def set_fit_end(self, value=0):
        if 'edges' not in self.dataset.metadata:
            self.edges = self.dataset.metadata['edges'] = {}
        if 'fit_area' not in self.edges:
            self.edges['fit_area'] = {'fit_start': self.parent.energy_scale[10],
                                      'fit_end': self.parent.energy_scale[-10]}
            self.core_loss_tab[3, 0].value = str(
                self.edges['fit_area']['fit_end'])
            self.core_loss_tab[2, 0].value = str(
                self.edges['fit_area']['fit_start'])
        if self.core_loss_tab[3, 0].value > self.parent.energy_scale[-1]:
            self.core_loss_tab[3, 0].value = self.parent.energy_scale[-10]
        self.edges['fit_area']['fit_end'] = self.core_loss_tab[3, 0].value
        self.parent.plot()

    def set_fit_area(self, value=1):
        if 'fit_area' not in self.edges:
            self.edges['fit_area'] = {'fit_start': self.parent.energy_scale[10],
                                      'fit_end': self.parent.energy_scale[-10]}

        fit_end = str(self.edges['fit_area']['fit_end'])
        fit_start = str(self.edges['fit_area']['fit_start'])

        if fit_end > fit_start:
            fit_start = self.parent.energy_scale[10]
            fit_end = self.parent.energy_scale[-10]
        self.core_loss_tab[2, 0].value = fit_start
        self.core_loss_tab[3, 0].value = fit_end
        self.edges['fit_area']['fit_start'] = self.core_loss_tab[2, 0].value
        self.edges['fit_area']['fit_end'] = self.core_loss_tab[3, 0].value

        self.parent.plot()

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
        # self.periodic_table.periodic_table
        self.parent.app_layout.center = self.periodic_table_panel

    def set_figure_pane(self, value=0):
        self.parent.app_layout.center = self.parent.panel

    def update(self, index=0):
        self.dataset = self.parent.dataset
        index = self.core_loss_tab[5, 0].value  # which edge
        if index < 0:
            options = list(self.core_loss_tab[5, 0].options)
            options.insert(-1, (f'Edge {len(self.core_loss_tab[5, 0].options)}', len(
                self.sidebar[4, 0].options)-1))
            self.core_loss_tab[5, 0].options = options
            self.core_loss_tab[5, 0].value = len(
                self.core_loss_tab[5, 0].options)-2
        if str(index) not in self.edges:
            self.edges[str(index)] = {'z': 0,  'element': 'x', 'symmetry': 'K1', 'onset': 0, 'start_exclude': 0, 'end_exclude': 0,
                                      'areal_density': 0, 'chemical_shift': 0}
        if 'z' not in self.edges[str(index)]:
            self.edges[str(index)] = {'z': 0,  'element': 'x', 'symmetry': 'K1', 'onset': 0, 'start_exclude': 0, 'end_exclude': 0,
                                      'areal_density': 0, 'chemical_shift': 0}
        edge = self.edges[str(index)]

        self.core_loss_tab[6, 0].value = edge['z']
        self.core_loss_tab[6, 2].value = edge['element']
        self.core_loss_tab[7, 0].value = edge['symmetry']
        self.core_loss_tab[8, 0].value = edge['onset']
        self.core_loss_tab[9, 0].value = edge['start_exclude']
        self.core_loss_tab[10, 0].value = edge['end_exclude']
        self.core_loss_tab[13, 0].value = self.parent.info_tab[9, 2].value
        if self.parent.y_scale == 1.0:
            self.core_loss_tab[11, 0].value = edge['areal_density']
            self.core_loss_tab[11, 2].value = 'a.u.'
        else:
            dispersion = self.parent.energy_scale.slope
            self.core_loss_tab[11, 0].value = np.round(
                edge['areal_density']/self.dataset.metadata['experiment']['flux_ppm']*1e-6, 2)
            self.core_loss_tab[11, 2].value = 'atoms/nm²'

    def do_fit(self, value=0):
        if 'experiment' in self.dataset.metadata:
            exp = self.dataset.metadata['experiment']
            if 'convergence_angle' not in exp:
                self.parent.status_message('Aborted Quantification: need a convergence_angle in experiment of metadata dictionary')
                return
    
            alpha = exp['convergence_angle']
            beta = exp['collection_angle']
            beam_kv = exp['acceleration_voltage']
            if beam_kv < 20:
                self.parent.status_message('Aborted Quantification: no acceleration voltage')
                return
        else:
            raise ValueError(
                'need a experiment parameter in metadata dictionary')

        self.parent.status_message('Fitting cross-sections ')
        eff_beta = eels.effective_collection_angle(
            self.parent.energy_scale, alpha, beta, beam_kv)
        self.dataset.metadata['experiment']['eff_beta'] = eff_beta
        self.low_loss = None
        if self.core_loss_tab[13, 1].value:
            if 'low_loss' in self.parent.datasets['_relationship'].keys():
                ll_key = self.parent.datasets['_relationship']['low_loss']
                self.low_loss = np.array(self.parent.datasets[ll_key] / \
                                    self.parent.datasets[ll_key].sum())
                    

        edges = eels.make_cross_sections(self.edges, np.array(
            self.parent.energy_scale), beam_kv, eff_beta, self.low_loss)
        if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
            spectrum = self.parent.get_spectrum()
        else:
            spectrum = self.dataset
        self.edges = eels.fit_edges2(spectrum, self.parent.energy_scale, edges)
        self.model = self.edges['model']['spectrum'].copy()
        print('set_model',  self.edges['model']['spectrum'][0], self.model[0])

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

        self.update()
        self.plot()
        self.parent.status_message('Fitting cross-sections -- success')

    def do_all_button_click(self, value=0):
        if self.sidebar[13, 0].value == False:
            return

        if self.dataset.data_type.name != 'SPECTRAL_IMAGE':
            self.do_fit()
            return

        if 'experiment' in self.dataset.metadata:
            exp = self.dataset.metadata['experiment']
            if 'convergence_angle' not in exp:
                raise ValueError(
                    'need a convergence_angle in experiment of metadata dictionary ')
            alpha = exp['convergence_angle']
            beta = exp['collection_angle']
            beam_kv = exp['acceleration_voltage']
        else:
            raise ValueError(
                'need a experiment parameter in metadata dictionary')

        eff_beta = eels.effective_collection_angle(
            self.energy_scale, alpha, beta, beam_kv)
        eff_beta = beta
        self.low_loss = None
        if self.sidebar[12, 1].value:
            for key in self.datasets.keys():
                if key != self.parent.lowloss_key:
                    if isinstance(self.datasets[key], sidpy.Dataset):
                        if 'SPECTR' in self.datasets[key].data_type.name:
                            if self.datasets[key].energy_loss[0] < 0:
                                self.low_loss = self.datasets[key] / \
                                    self.datasets[key].sum()

        edges = eels.make_cross_sections(self.edges, np.array(
            self.energy_scale), beam_kv, eff_beta, self.low_loss)

        view = self.parent
        bin_x = view.bin_x
        bin_y = view.bin_y

        start_x = view.x
        start_y = view.y

        number_of_edges = 0
        for key in self.edges:
            if key.isdigit():
                number_of_edges += 1

        results = np.zeros([int(self.dataset.shape[0]/bin_x),
                           int(self.dataset.shape[1]/bin_y), number_of_edges])
        total_spec = int(
            self.dataset.shape[0]/bin_x)*int(self.dataset.shape[1]/bin_y)
        self.sidebar[13, 1].max = total_spec
        # self.ui.progress.setMaximum(total_spec)
        # self.ui.progress.setValue(0)
        ind = 0
        for x in range(int(self.dataset.shape[0]/bin_x)):
            for y in range(int(self.dataset.shape[1]/bin_y)):
                ind += 1
                self.sidebar[13, 1].value = ind
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
        self.sidebar[13, 1].value = total_spec
        view.x = start_x
        view.y = start_y
        self.sidebar[13, 0].value = False

    def modify_onset(self, value=-1):
        edge_index = self.core_loss_tab[5, 0].value
        edge = self.edges[str(edge_index)]
        edge['onset'] = self.core_loss_tab[8, 0].value
        if 'original_onset' not in edge:
            edge['original_onset'] = edge['onset']
        edge['chemical_shift'] = edge['onset'] - edge['original_onset']
        self.update()

    def modify_start_exclude(self, value=-1):
        edge_index = self.core_loss_tab[5, 0].value
        edge = self.edges[str(edge_index)]
        edge['start_exclude'] = self.core_loss_tab[9, 0].value
        self.plot()

    def modify_end_exclude(self, value=-1):
        edge_index = self.core_loss_tab[5, 0].value
        edge = self.edges[str(edge_index)]
        edge['end_exclude'] = self.core_loss_tab[10, 0].value
        self.plot()

    def modify_areal_density(self, value=-1):
        edge_index = self.core_loss_tab[5, 0].value
        edge = self.edges[str(edge_index)]

        edge['areal_density'] = self.core_loss_tab[11, 0].value
        if self.parent.y_scale != 1.0:
            dispersion = self.parent.energy_scale.slope
            edge['areal_density'] = self.core_loss_tab[11, 0].value * \
                self.dataset.metadata['experiment']['flux_ppm']/1e-6
        if 'model' in self.edges:
            self.model = self.edges['model']['background']
            for key in self.edges:
                if key.isdigit():
                    if 'data' in self.edges[key]:
                        self.model = self.model + \
                            self.edges[key]['areal_density'] * \
                            self.edges[key]['data']
            self.model = self.edges['model']['background']
            for key in self.edges:
                if key.isdigit():
                    if 'data' in self.edges[key]:
                        self.model = self.model + \
                            self.edges[key]['areal_density'] * \
                            self.edges[key]['data']
        self.plot()

    def set_y_scale(self, value):
        self.parent.info_tab[9, 2].value = self.core_loss_tab[13, 0].value
        self.update()
        
    def set_convolution(self, value=0):
        self.do_fit()
    
    
    def set_cl_action(self):

        self.core_loss_tab[0, 0].observe(self.update_cl_dataset, names='value')
        self.core_loss_tab[2, 0].observe(self.set_fit_start, names='value')
        self.core_loss_tab[3, 0].observe(self.set_fit_end, names='value')

        self.core_loss_tab[4, 0].on_click(self.find_elements)
        self.core_loss_tab[5, 0].observe(self.update, names='value')
        self.core_loss_tab[6, 0].observe(self.set_element, names='value')

        self.core_loss_tab[8, 0].observe(self.modify_onset, names='value')
        self.core_loss_tab[9, 0].observe(
            self.modify_start_exclude, names='value')
        self.core_loss_tab[10, 0].observe(
            self.modify_end_exclude, names='value')
        self.core_loss_tab[11, 0].observe(
            self.modify_areal_density, names='value')

        self.core_loss_tab[12, 0].on_click(self.do_fit)
        self.core_loss_tab[13, 2].observe(self.plot, names='value')
        self.core_loss_tab[1, 0].observe(self.plot, names='value')
        self.core_loss_tab[13, 0].observe(self.set_y_scale, names='value')
        self.core_loss_tab[13, 1].observe(self.set_convolution, names='value')
        self.core_loss_tab[14, 0].observe(
            self.do_all_button_click, names='value')

        self.elements_cancel_button.on_click(self.set_figure_pane)
        self.elements_auto_button.on_click(self.auto_id)
        self.elements_select_button.on_click(self.set_elements)
