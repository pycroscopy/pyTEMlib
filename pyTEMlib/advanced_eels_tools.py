""" Advanced EELS tools
Tools not yet publicly available with pyTEMlib.

Do not distribute!!
Do not use if you did not get this file directly from Gerd Duscher

Copyright exclusively by Gerd Duscher, UTK
"""
import numpy as np
import scipy
from scipy import optimize  

import plotly.graph_objects as go
from ipywidgets import widgets

import sys
import pyTEMlib.eels_tools as eels
import pyTEMlib.file_tools as ft


def smooth(dataset, fit_start, fit_end, peaks=None, iterations=2, sensitivity=2.):
    """Using Gaussian mixture model (non-Bayesian) to fit spectrum

    Set up to fit lots of Gaussian to spectrum

    Parameter
    ---------
    dataset: sidpy dataset
    fit_start: float
        start of energy window of fitting
    fit_end: float
        start of energy window of fitting
    peaks: numpy array float
    iterations: int
    sensitivity: float
    """

    if dataset.data_type.name == 'SPECTRAL_IMAGE':
        spectrum = dataset.view.get_spectrum()
    else:
        spectrum = np.array(dataset)

    spec_dim = ft.get_dimensions_by_type('SPECTRAL', dataset)[0]
    energy_scale = np.array(spec_dim[1])
    start_channel = np.searchsorted(energy_scale, fit_start)
    end_channel = np.searchsorted(energy_scale, fit_end)

    if peaks is None:
        second_dif, noise_level = eels.second_derivative(dataset, sensitivity=sensitivity)
        [indices, _] = scipy.signal.find_peaks(-second_dif, noise_level)

        peaks = []
        for index in indices:
            if start_channel < index < end_channel:
                peaks.append(index - start_channel)
    else:
        peaks = peaks[::3]

    if energy_scale[0] > 0:
        if 'edges' not in dataset.metadata:
            return
        if 'model' not in dataset.metadata['edges']:
            return
        model = dataset.metadata['edges']['model']['spectrum'][start_channel:end_channel]

    else:
        model = np.zeros(end_channel - start_channel)

    energy_scale = energy_scale[start_channel:end_channel]

    difference = np.array(spectrum)[start_channel:end_channel] - model

    peak_model, peak_out_list = gaussian_mixing(difference, model, energy_scale, iterations=iterations, n_pks=30, peaks=peaks)
    peak_model2 = np.zeros(len(spec_dim[1]))
    peak_model2[start_channel:end_channel] = peak_model

    return peak_model2, peak_out_list
def Drude(E,Ep,Ew):
    eps = 1 - Ep**2/(E**2+Ew**2) +1j* Ew* Ep**2/E/(E**2+Ew**2)
    elf = (-1/eps).imag
    return eps,elf

def errfDrude(p, y, x):
    eps,elf = Drude(x,p[0],p[1])
    err = y - p[2]*elf
    #print (p,sum(np.abs(err)))
    return np.abs(err)#/np.sqrt(y)

def  fit_drude(dataset):
    pin2 = np.array([15,1,.7])
    E = energy_scale = dataset.energy_loss
    startFit =np.argmin(abs(energy_scale-13))
    endFit = np.argmin(abs(energy_scale-18))
        
    p2, lsq = scipy.optimize.leastsq(errfDrude, pin2, args=(dataset[startFit:endFit], energy_scale[startFit:endFit]), maxfev=2000)

    eps, elf =Drude(energy_scale,p2[0],p2[1])
    drude_dataset = dataset.like_data(p2[2]* elf)
    drude_dataset.title = 'drude_function'
    drude_dataset.metadata={'fit': {'energy': p2[0],'width': p2[1], 'amplitude': p2[2]}}
    return  drude_dataset



def resolution_function(dataset, width = .4):
    guess = [0.2, 1000, 0.02, 0.2, 1000, 0.2]
    p0 = np.array(guess)

    start = np.searchsorted(dataset.energy_loss, -width / 2.)
    end = np.searchsorted(dataset.energy_loss, width / 2.)
    x = dataset.energy_loss[start:end]
    y = np.array(dataset)[start:end]
    def zl2(pp, yy, xx):
        eerr = (yy - eels.zl_func(pp, xx))  # /np.sqrt(y)
        return eerr
    
    [p_zl, _] = scipy.optimize.leastsq(zl2, p0, args=(y, x), maxfev=2000)

    z_loss = eels.zl_func(p_zl, dataset.energy_loss)
    z_loss = dataset.like_data(z_loss)
    z_loss.title = 'resolution_function'
    Izl = np.array(z_loss).sum()
    Itotal = np.array(dataset).sum()
    tmfp = np.log(Itotal/Izl)
    z_loss.metadata['fit'] = {'zero_loss_parameter': p_zl,
                              'thickness_imfp':tmfp}

    return z_loss


def residuals_smooth(p, x, y, only_positive_intensity):
    """part of fit"""

    err = (y - model_smooth(x, p, only_positive_intensity=only_positive_intensity))
    return err

def model_smooth(x, p, only_positive_intensity=False):
    """part of fit"""

    x = x.reshape(-1, 1)

    amplitudes = np.array(p)[1::3].reshape(1, -1)
    if only_positive_intensity:
        amplitudes = np.abs(amplitudes)

    positions = np.array(p)[0::3].reshape(1, -1)
    
    widths = np.array(p)[2::3].reshape(1, -1)/ 2.3548  # from FWHM to sigma
    widths[widths == 0] = 1e-12
    
    return np.sum(amplitudes * np.exp(-(x-positions)**2 / (2.0*widths**2)), axis=1)


def gaussian_mixing(dataset, model, energy_scale, iterations=2, n_pks=30, peaks=None, resolution=0.3, only_positive_intensity=False):
    """Gaussian mixture model (non-Bayesian) """
    original_difference = np.array(dataset-model)
    # energy_scale = dataset.energy_loss
    peak_out_list = []
    fit = np.zeros(len(dataset))
    
    if peaks is not None:
        if len(peaks) > 0:
            p_in = np.ravel([[energy_scale[i], original_difference[i], resolution] for i in peaks])
            p_out, cov = scipy.optimize.leastsq(residuals_smooth, p_in, ftol=1e-3, args=(energy_scale,
                                                                                              original_difference,
                                                                                              only_positive_intensity))
            peak_out_list.extend(p_out)
            fit = fit + model_smooth(energy_scale, p_out, only_positive_intensity)
    
    difference = np.array(original_difference - fit)

    for i in range(iterations):
        i_pk = scipy.signal.find_peaks_cwt(np.abs(difference), widths=range(3, len(energy_scale) // n_pks))
        p_in = np.ravel([[energy_scale[i], difference[i], .5] for i in i_pk])  # starting guess for fit

        p_out, cov = scipy.optimize.leastsq(residuals_smooth, p_in, ftol=1e-3, args=(energy_scale, difference, only_positive_intensity))        
        peak_out_list.extend(p_out)
        
        fit = fit + model_smooth(energy_scale, p_out, only_positive_intensity)

        difference = np.array(original_difference - fit)

    fit_dataset = model+fit
    # fit_dataset.title = 'peak_fit'

    # fit_dataset.metadata = {'fit': {'peaks': peak_out_list,
    #                                'iterations': iterations}}
    return fit_dataset, peak_out_list


def get_gaussian_mixing(dataset, model, energy_scale, iterations=2, resolution=0.3, only_positive_intensity=False ):

    difference = np.array(dataset - model)

    peak_model, peak_out_list = gaussian_mixing(dataset, model, energy_scale, iterations=iterations, n_pks=30, peaks=None, resolution=resolution, only_positive_intensity=only_positive_intensity)


    p_out, p_model = scipy.optimize.leastsq(residuals_smooth, np.array(peak_out_list), ftol=1e-3,
                                            args=(energy_scale, difference, only_positive_intensity))
    peak_fit = dataset.like_data(model_smooth(energy_scale, p_out, only_positive_intensity))+model
    
    peak_fit.title = 'peak_fit'
    peak_fit.metadata = {'fit': {'peaks':  sort_peaks(p_out)}}

    return peak_fit


def sort_peaks(peak_out_list):
    new_list = np.reshape(peak_out_list, [len(peak_out_list) // 3, 3])
    area = np.sqrt(2 * np.pi) * np.abs(new_list[:, 1]) * np.abs(new_list[:, 2] / np.sqrt(2 * np.log(2)))
    arg_list = np.argsort(area)[::-1]
    area = area[arg_list]
    peak_out_list = new_list[arg_list]
    return peak_out_list.flatten()


def peak_fit(dataset, peak_fit, model, number_of_peaks=10):
    
    difference =  np.array(dataset - model)
    peak_out_list = peak_fit.metadata['fit']['peaks'][:number_of_peaks*3]
   
    p_out, p_model = scipy.optimize.leastsq(residuals_smooth, np.array(peak_out_list), ftol=1e-3,
                                            args=(dataset.energy_loss, difference, False))
   
    peak_fit = dataset.like_data(model_smooth(dataset.energy_loss, p_out, False))+model
    peak_fit.title = 'peak_model'
    peak_fit.metadata = {'fit': {'peaks':  p_out}}
    return peak_fit


def get_peaks(dataset):
    peak_out_list = dataset.metadata['fit']['peaks']
    peaks = dataset.like_data(model_smooth(dataset.energy_loss, peak_out_list, False))
    peaks.title = 'peaks'
    return peaks

def smooth2(dataset, iterations, advanced_present):
    """Gaussian mixture model (non-Bayesian)

    Fit lots of Gaussian to spectrum and let the program sort it out
    We sort the peaks by area under the Gaussians, assuming that small areas mean noise.

    """

    # TODO: add sensitivity to dialog and the two functions below
    peaks = dataset.metadata['peak_fit']

    if advanced_present and iterations > 1:
        peak_model, peak_out_list = advanced_eels_tools.smooth(dataset, peaks['fit_start'],
                                                               peaks['fit_end'], iterations=iterations)
    else:
        peak_model, peak_out_list = eels.find_peaks(dataset, peaks['fit_start'], peaks['fit_end'])
        peak_out_list = [peak_out_list]

    flat_list = [item for sublist in peak_out_list for item in sublist]
    new_list = np.reshape(flat_list, [len(flat_list) // 3, 3])
    area = np.sqrt(2 * np.pi) * np.abs(new_list[:, 1]) * np.abs(new_list[:, 2] / np.sqrt(2 * np.log(2)))
    arg_list = np.argsort(area)[::-1]
    area = area[arg_list]
    peak_out_list = new_list[arg_list]

    number_of_peaks = np.searchsorted(area * -1, -np.average(area))

    return peak_model, peak_out_list, number_of_peaks


def plot_spectrum_datasets(datasets, **args):

    first_spectrum = datasets[list(datasets)[0]]
    energy_dim = first_spectrum.get_spectrum_dims()
    energy_dim = first_spectrum.get_dimension_by_number(energy_dim[0])[0]

    if 'plot_parameter' not in first_spectrum.metadata:
        first_spectrum.metadata['plot_parameter'] = {}
    plot_dic = first_spectrum.metadata['plot_parameter']
    
    if 'title' not in args:
         plot_dic['title'] = ''
    else:
        plot_dic['title'] = args['title']

    if 'theme' not in args:
        theme="plotly_white"
    else:
        theme = args['theme']
    
    if 'y_scale' not in args:
        if 'incident_current_in_counts' in first_spectrum.metadata['experiment']:
            plot_dic['y_scale'] = 1e6/first_spectrum.metadata['experiment']['incident_current_in_counts']
        else:
            plot_dic['y_scale'] = 1e6/first_spectrum.sum()
    else:
        plot_dic['y_scale'] = args['y_scale']
    if 'y_axis_label' not in args:
        plot_dic['y_axis_label'] = f'{first_spectrum.quantity} ({first_spectrum.units})'
    else:
        plot_dic['y_axis_label'] = args['y_axis_label']
    if 'height'  in args:
        plot_dic['height'] = args['height']
    else:
        plot_dic['height'] = 500

    fig = go.Figure()

    for key, dat in datasets.items():
        if dat.data_type == first_spectrum.data_type:
            energy_dim = dat.get_spectrum_dims()
            energy_dim = dat.get_dimension_by_number(energy_dim[0])[0]

            fig.add_trace(
                go.Scatter(x=energy_dim.values, y=np.array(dat), name=dat.title, mode="lines+markers",
                        marker=dict(size=2)))
            

    plot_dic['x_axis_label'] = f'{energy_dim.name} ({energy_dim.units})'

    fig.update_layout(
        selectdirection='h',
        showlegend = False,
        dragmode='select',
        title_text=plot_dic['title'],
        yaxis_title_text=plot_dic['y_axis_label'],
        xaxis_title_text=plot_dic['x_axis_label'],
        height=plot_dic['height'],
        template=theme
    )
    fig.update_layout(hovermode='x unified')
    config = {'displayModeBar': True}

    #fig.show(config=config)
    return fig



def image_plot_widget(dataset):
    image_dims = dataset.get_image_dims()
    if len(image_dims) != 2:
            raise TypeError('We need two dimensions with dimension_type SPATIAL: to plot an image')
    energy_dim = dataset.get_spectrum_dims()
    if len(energy_dim) == 1:
        image = dataset.sum(energy_dim[0])
    else:
         image = dataset
    image_widget = go.FigureWidget()

    image = go.Heatmap(z=image.T)
    
    image_widget.add_trace(image)
    image_widget['layout'].update(width=500, height=500, 
                                  autosize=False,
                                  yaxis_title_text=f'{dataset.y.name} ({dataset.y.units})',
                                  xaxis_title_text=f'{dataset.x.name} ({dataset.x.units})',
                                  modebar_add=['drawopenpath', 'eraseshape'])
    
    return image_widget



class SpectralImageVisualizer(object):
    def __init__(self, dataset, spectrum_number=0, figure=None, **kwargs):

        energy_dim = dataset.get_spectrum_dims()
        if len(energy_dim) >1 :
            raise TypeError('This spectrum image has more than one spectral axis')
        self.energy_scale = dataset.get_dimension_by_number(energy_dim[0])[0]
        self.spectrum_selection = [0, 0]
        self.dataset = dataset
        
        def click_callback(trace, points, selector):
            self.spectrum_selection = points.point_inds[0]
            self.spectrum_plot_update()
        
        self.image_widget = image_plot_widget(dataset)
        self.image_widget.data[0].on_click(click_callback)

        self.get_spectrum()
        fig = plot_spectrum_datasets({'1': self.spectrum})
        self.spectrum_widget = go.FigureWidget(fig)

        def selection_fn(trace,points,selector):
            self.energy_selection = points
        self.spectrum_widget.data[0].on_selection(selection_fn)
        self.plot = display(widgets.HBox([self.image_widget, self.spectrum_widget]))

    def set_spectrum(self, x=0, y=0):
         self.spectrum_selection = [x,y]
         self.spectrum_plot_update()
             
    def get_spectrum(self):
        import sidpy
        energy_dim = self.dataset.get_spectrum_dims()
        energy_dim = self.dataset.get_dimension_by_number(energy_dim[0])[0]

        selection = []
        [y, x] = self.spectrum_selection
        image_dims = self.dataset.get_image_dims()
        for i in range(3):
                if i in image_dims:
                    if self.dataset.get_dimension_by_number(image_dims[i])[0].name == 'x':
                        selection.append(slice(x, x+1))
                    else:
                        selection.append(slice(y, y+1))
                else:
                    selection.append(slice(None))
    
        spec = sidpy.Dataset.from_array(self.dataset[tuple(selection)].mean(axis=tuple(image_dims)).squeeze())
        spec.set_dimension(0, energy_dim)
        spec.data_type = 'spectrum'
        spec.quantity = self.dataset.quantity
        spec.units = self.dataset.units
        spec.metadata['experiment'] = self.dataset.metadata['experiment'].copy()
        
        spec.metadata['provenance'] = {'original_dataset': self.dataset.title}
        spec.metadata['spectral_image'] = {'x': y, 'y': x}

        spec.title = f'spectrum {self.spectrum_selection[0]}, {self.spectrum_selection[1]}'

        self.spectrum = spec
        
    def spectrum_plot_update(self):
        self.get_spectrum()

        self.spectrum_widget.data[0].y = self.spectrum
        self.spectrum_widget.update_layout(title=f'spectrum {self.spectrum_selection[0]}, {self.spectrum_selection[1]}')
