"""
Input Dialog for Image Analysis

Author: Gerd Duscher

"""
# -*- coding: utf-8 -*-

import numpy as np
import sidpy

Qt_available = True
try:
    from PyQt5 import QtCore,  QtWidgets
except:
    Qt_available = False
    print('Qt dialogs are not available')

from matplotlib.widgets import SpanSelector
from skimage import exposure

from pyTEMlib.image_dlg import *
from pyTEMlib.microscope import microscope

_version = 000

if Qt_available:
    class ImageDialog(QtWidgets.QDialog):
        """
            Input Dialog for Image Analysis

            Opens a PyQt5 GUi Dialog that allows to set the experimental parameter necessary for a Quantification.


            The dialog operates on a sidpy dataset
            """

        def __init__(self, dataset, parent=None):
            super(ImageDialog, self).__init__(parent)
            if not isinstance(dataset, sidpy.Dataset):
                raise TypeError("we need a sidpy.Dataset")
            self.parent = parent
            self.debug = 0
            self.dataset = dataset
            self.image = np.array(self.dataset)
            self.v_min = np.array(dataset).min()
            self.v_max = np.array(dataset).max()

            self.ui = UiDialog(self)
            self.setWindowTitle('Image Info')

            self.dataset.plot()
            self.histogram()
            self.cid = self.ui.histogram.axes.figure.canvas.mpl_connect('button_press_event', self.onclick)

            self.span = SpanSelector(self.ui.histogram.axes, self.on_select, 'horizontal', useblit=False,
                                     button=1, minspan=5,
                                     rectprops=dict(alpha=0.3, facecolor='blue'))
            minimum_info = {'size': self.dataset.shape,
                            'exposure_time': 0.0,
                            'convergence_angle': 0.0,
                            'acceleration_voltage': 100.0,
                            'binning': 1, 'conversion': 1.0,
                            'flux': 1.0, 'current': 1.0}

            if 'experiment' not in self.dataset.metadata:
                self.dataset.metadata['experiment'] = {}
            for key, value in minimum_info.items():
                if key not in self.dataset.metadata['experiment']:
                    self.dataset.metadata['experiment'][key] = value
            self.experiment = self.dataset.metadata['experiment']
            self.set_action()
            self.update()

        def histogram(self, bins=256):
            ax_hist = self.ui.histogram.axes
            ax_hist.clear()

            hist, bin_edges = np.histogram(np.array(self.image), range=[self.v_min, self.v_max], bins=bins, density=True)
            ax_hist.plot(np.array(bin_edges)[:-1], np.array(hist))

            image = self.image * 1.0
            image[image < self.v_min] = self.v_min
            image[image > self.v_max] = self.v_max

            img_cdf, bins = exposure.cumulative_distribution(np.array(image), bins)
            ax_hist.plot(bins, img_cdf * hist.max(), 'r')
            ax_hist.figure.canvas.draw()
            self.span = SpanSelector(self.ui.histogram.axes, self.on_select, 'horizontal', useblit=False,
                                     button=1, minspan=5,
                                     rectprops=dict(alpha=0.3, facecolor='blue'))
            self.plot()

        def onclick(self, event):
            if event.dblclick:
                self.v_min = np.array(self.dataset).min()
                self.v_max = np.array(self.dataset).max()
                self.histogram()

        def on_select(self, v_min, v_max):
            self.v_min = v_min
            self.v_max = v_max
            self.histogram()

        def plot(self):
            ax = self.dataset.view.axis
            img = self.dataset.view.img
            img.set_data(self.image)
            img.set_clim(vmin=self.v_min, vmax=self.v_max)
            ax.figure.canvas.draw()

        def on_enter(self):
            sender = self.sender()
            if sender == self.ui.timeEdit:
                value = float(str(sender.displayText()).strip())
                self.experiment['exposure_time'] = value
                sender.setText(f"{value:.2f}")
            elif sender == self.ui.convEdit:
                value = float(str(sender.displayText()).strip())
                self.experiment['convergence_angle'] = value
                sender.setText(f"{value:.2f}")
            elif sender == self.ui.E0Edit:
                value = float(str(sender.displayText()).strip())
                self.experiment['acceleration_voltage'] = value * 1000.0
                sender.setText(f"{value:.2f}")

        def on_list_enter(self):
            sender = self.sender()
            if sender == self.ui.TEMList:
                microscope.set_microscope(self.ui.TEMList.currentText())
                self.setWindowTitle(microscope.name)

                self.experiment['microscope'] = microscope.name
                self.experiment['convergence_angle'] = microscope.alpha
                self.experiment['acceleration_voltage'] = microscope.E0
                self.update()

        def update(self):
            self.ui.convEdit.setText(f"{self.experiment['convergence_angle']:.2f}")
            self.ui.E0Edit.setText(f"{self.experiment['acceleration_voltage']/1000.:.2f}")

            self.ui.timeEdit.setText(f"{self.experiment['exposure_time']:.6f}")
            size_text = f'{self.dataset.shape[0]}'
            for size in self.dataset.shape[1:]:
                size_text = size_text + f' x {size}'
            self.ui.sizeEdit.setText(size_text)

            # self.ui.binningEdit.setText(f"{self.experiment['binning']}")
            # self.ui.conversionEdit.setText(f"{self.experiment['conversion']:.2f}")
            # self.ui.fluxEdit.setText(f"{self.experiment['flux']:.2f}")
            # self.ui.VOAEdit.setText(f"{self.experiment['current']:.2f}")

        def set_action(self):
            self.ui.timeEdit.editingFinished.connect(self.on_enter)
            self.ui.TEMList.activated[str].connect(self.on_list_enter)

            self.ui.convEdit.editingFinished.connect(self.on_enter)
            self.ui.E0Edit.editingFinished.connect(self.on_enter)
