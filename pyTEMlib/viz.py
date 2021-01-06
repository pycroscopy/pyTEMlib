import numpy as np
import sidpy

from bokeh.layouts import column
from bokeh.plotting import figure  # , show, output_notebook
from bokeh.models import CustomJS, Slider, Span
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, BoxSelectTool
from bokeh.palettes import Spectral11


def plot(dataset, palette='Viridis256'):
    if dataset.data_type.name == 'IMAGE_STACK':
        p = plot_stack(dataset, palette=palette)
    elif dataset.data_type.name == 'IMAGE':
        p = plot_image(dataset, palette=palette)
    elif dataset.data_type.name == 'SPECTRUM':
        p = plot_spectrum(dataset, palette=palette)
    else:
        p = None
    return p


def plot_stack(dataset, palette="Viridis256"):
    """Plotting a stack of images

    Plotting a stack of images contained in a sidpy.Dataset.
    The images can be scrolled through with a slider widget.

    Parameters
    ----------
    dataset: sidpy.Dataset
        sidpy dataset with data_type 'IMAGE_STACK'
    palette: bokeh palette
        palette is optional

    Returns
    -------
    p: bokeh plot

    Example
    -------
    >> import pyTEMlib
    >> from bokeh.plotting import figure, show, output_notebook
    >> output_notebook()
    >> p = pyTEMlib.viz(dataset)
    >> p.show(p)
    """

    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('Need a sidpy dataset for plotting')
    if dataset.data_type.name != 'IMAGE_STACK':
        raise TypeError('Need an IMAGE_STACK for plotting a stack')

    stack = np.array(dataset-dataset.min())
    stack = stack/stack.max()*256
    stack = np.array(stack, dtype=int)

    color_mapper = LinearColorMapper(palette=palette, low=0, high=256)

    p = figure(match_aspect=True, plot_width=600, plot_height=600)
    im_plot = p.image(image=[stack[0]], x=[0], y=[0], dw=[dataset.x[-1]], dh=[dataset.y[-1]], color_mapper=color_mapper)
    p.x_range.range_padding = 0
    p.y_range.range_padding = 0
    p.xaxis.axis_label = 'distance (nm)'
    p.yaxis.axis_label = 'distance (nm)'

    slider = Slider(start=0, end=stack.shape[0]-1, value=0, step=1, title="frame")

    update_curve = CustomJS(args=dict(source=im_plot, slider=slider, stack=stack),
                            code="""var f = slider.value;
                                    source.data_source.data['image'] = [stack[f]];
                                    // necessary because we mutated source.data in-place
                                    source.data_source.change.emit(); """)
    slider.js_on_change('value', update_curve)

    return column(slider, p)


def plot_image(dataset, palette="Viridis256"):
    """Plotting an image

        Plotting an image contained in a sidpy.Dataset.

        Parameters
        ----------
        dataset: sidpy.Dataset
            sidpy dataset with data_type 'IMAGE_STACK'
        palette: bokeh palette
            palette is optional

        Returns
        -------
        p: bokeh plot

        Example
        -------
        >> import pyTEMlib
        >> from bokeh.plotting import figure, show, output_notebook
        >> output_notebook()
        >> p = pyTEMlib.viz(dataset)
        >> p.show(p)


        """
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('Need a sidpy dataset for plotting')

    if dataset.data_type.name not in ['IMAGE', 'IMAGE_STACK']:
        raise TypeError('Need an IMAGE or IMAGE_STACK for plotting an image')

    if dataset.data_type.name == 'IMAGE_STACK':
        image = dataset.sum(axis=0)
        image = sidpy.Dataset.from_array(image)
        image.data_type = 'image'
        image.title = dataset.title
        image.set_dimension(0, dataset.dim_1)
        image.set_dimension(1, dataset.dim_2)
    else:
        image = dataset

    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")], match_aspect=True,
               plot_width=675, plot_height=600, )
    color_mapper = LinearColorMapper(palette=palette, low=float(image.min()), high=float(image.max()))

    # must give a vector of image data for image parameter
    p.image(image=[np.array(image)], x=0, y=0, dw=image.x[-1], dh=image.y[-1], color_mapper=color_mapper,
            level="image")
    p.x_range.range_padding = 0
    p.y_range.range_padding = 0

    p.grid.grid_line_width = 0.
    p.xaxis.axis_label = 'distance (nm)'
    p.yaxis.axis_label = 'distance (nm)'

    color_bar = ColorBar(color_mapper=color_mapper, major_label_text_font_size="7pt",
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')
    return p



def plot_spectrum(dataset, selected_range, palette=Spectral11):
    if not isinstance(dataset, sidpy.Dataset):
        raise TypeError('Need a sidpy dataset for plotting')

    if dataset.data_type.name not in ['SPECTRUM']:
        raise TypeError('Need an sidpy.Dataset of data_type SPECTRUM for plotting a spectrum ')

    p = figure(x_axis_type="linear", plot_width=800, plot_height=400,
               tooltips=[("index", "$index"),("(x,y)", "($x, $y)")],
               tools="pan,wheel_zoom,box_zoom,reset, hover, lasso_select")
    p.add_tools(BoxSelectTool(dimensions="width"))

    # first line is dataset
    spectrum = ColumnDataSource(data=dict(x=dataset.dim_0, y=np.array(dataset)))
    p.scatter('x', 'y', color='blue', size=1, alpha=0., source=spectrum,
              selection_color="firebrick", selection_alpha=0.)
    p.line(x='x', y='y', source=spectrum, legend_label=dataset.title, color=palette[0], line_width=2)
    # add other lines if available
    if 'add2plot' in dataset.metadata:
        data = dataset.metadata['add2plot']
        for key, line in data.items():
            p.line(dataset.dim_0.values, line['data'], legend_label=line['legend'], color=palette[key], line_width=2)
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = dataset.labels[0]
    p.yaxis.axis_label = dataset.data_descriptor
    p.title.text = dataset.title

    my_span = Span(location=0, dimension='width', line_color='gray', line_width=1)
    p.add_layout(my_span)

    callback = CustomJS(args=dict(s1=spectrum), code="""
        var inds = s1.selected.indices;
        if (inds.length == 0)
            return;
        var kernel = IPython.notebook.kernel;
        kernel.execute("selected_range = " + [inds[0], inds[inds.length-1]]);""")

    spectrum.selected.js_on_change('indices', callback)
    return p, selected_range

class CurveVisualizer(object):
    def __init__(self, dataset, spectrum_number=None, fig=None, **kwargs):

        if not isinstance(dataset, sidpy.Dataset):
            raise TypeError('dset should be a sidpy.Dataset object')
        if not isinstance(dataset, sidpy.Dataset):
            raise TypeError('Need a sidpy dataset for plotting')

        if dataset.data_type.name not in ['SPECTRUM']:
            raise TypeError('Need an sidpy.Dataset of data_type SPECTRUM for plotting a spectrum ')


        fig_args = dict()

        if fig is None:
            self.fig = figure(x_axis_type="linear", plot_width=800, plot_height=400,
                              tooltips=[("index", "$index"),("(x,y)", "($x, $y)")],
                              tools="pan,wheel_zoom,box_zoom, reset, hover")
            self.fig.add_tools(BoxSelectTool(dimensions="width"))
        else:
            self.fig = fig

        self.dataset = dataset
        self.selection = []
        self.spectral_dims = []

        for dim, axis in dataset._axes.items():
            if axis.dimension_type.name == 'SPECTRAL':
                self.selection.append(slice(None))
                self.spectral_dims.append(dim)
            else:
                if spectrum_number <= dataset.shape[dim]:
                    self.selection.append(slice(spectrum_number, spectrum_number + 1))
                else:
                    self.spectrum_number = 0
                    self.selection.append(slice(0, 1))


        self.dim = self.dataset._axes[self.spectral_dims[0]]
        self.palette = Spectral11
        # first line is dataset
        self.spectrum = ColumnDataSource(data=dict(x=dataset.dim_0, y=np.array(dataset)))
        self.fig.scatter('x', 'y', color='blue', size=1, alpha=0., source=self.spectrum,
                  selection_color="firebrick", selection_alpha=0.)
        self.fig.line(x='x', y='y', source=self.spectrum, legend_label=dataset.title, color=self.palette[0], line_width=2)
        # add other lines if available
        if 'add2plot' in dataset.metadata:
            data = dataset.metadata['add2plot']
            for key, line in data.items():
                self.fig.line(dataset.dim_0.values, line['data'], legend_label=line['legend'], color=self.palette[key],
                       line_width=2)
        self.fig.legend.click_policy = "hide"
        self.fig.xaxis.axis_label = dataset.labels[0]
        self.fig.yaxis.axis_label = dataset.data_descriptor
        self.fig.title.text = dataset.title

        my_span = Span(location=0, dimension='width', line_color='gray', line_width=1)
        self.fig.add_layout(my_span)
        self.dataset.selected_range = []

        callback = CustomJS(args=dict(s1=self.spectrum), code="""
                var inds = s1.selected.indices;
                if (inds.length == 0)
                    return;
                var kernel = IPython.notebook.kernel;
                kernel.execute("dataset.selected_range = " + [inds[0], inds[inds.length-1]]);""")



        self.spectrum.selected.js_on_change('indices', callback)


