"""
Input Dialog for EELS Analysis

Author: Gerd Duscher

"""
import numpy as np
import sidpy

from PyQt5 import QtCore,  QtWidgets

from pyTEMlib import info_dlg
import pyTEMlib.interactive_eels as ieels
from pyTEMlib.microscope import microscope

from pyTEMlib import file_tools as ft
_version = 000


class InfoDialog(QtWidgets.QDialog):
    """
    Input Dialog for EELS Analysis

    Opens a PyQt5 GUi Dialog that allows to set the experimental parameter necessary for a Quantification.


    The dialog operates on a sidpy dataset
    """

    def __init__(self, dataset=None):
        super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
        # Create an instance of the GUI
        self.ui = info_dlg.UiDialog(self)
        self.set_action()

        self.dataset = dataset
        self.spec_dim = []
        self.energy_scale = np.array([])
        self.experiment = {}
        self.energy_dlg = None
        self.axis = None
        self.show()

        if dataset is None:
            # make a dummy dataset for testing
            dataset = ft.make_dummy_dataset(sidpy.DataType.SPECTRUM)
        if not isinstance(dataset, sidpy.Dataset):
            raise TypeError('dataset has to be a sidpy dataset')

        self.set_dataset(dataset)

        self.dataset.plot()
        if hasattr(self.dataset.view, 'axes'):
            self.axis = self.dataset.view.axes[-1]
        elif hasattr(self.dataset.view, 'axis'):
            self.axis = self.dataset.view.axis
        self.figure = self.axis.figure
        self.plot()
        self.update()

    def set_dataset(self, dataset):
        self.dataset = dataset
        if not hasattr(self.dataset, '_axes'):
            self.dataset._axes = self.dataset.axes
        if not hasattr(self.dataset, 'meta_data'):
            self.dataset.meta_data = {}

        spec_dim = dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
        if len(spec_dim) != 1:
            raise TypeError('We need exactly one SPECTRAL dimension')
        self.spec_dim = self.dataset._axes[spec_dim[0]]
        self.energy_scale = self.spec_dim.values.copy()

        minimum_info = {'offset': self.energy_scale[0],
                        'dispersion': self.energy_scale[1] - self.energy_scale[0],
                        'exposure_time': 0.0,
                        'convergence_angle': 0.0, 'collection_angle': 0.0,
                        'acceleration_voltage': 100.0, 'binning': 1, 'conversion': 1.0,
                        'flux': 1.0, 'current': 1.0, 'SI_bin_x': 1, 'SI_bin_y': 1}
        if 'experiment' not in self.dataset.metadata:
            self.dataset.metadata['experiment'] = minimum_info
        self.experiment = self.dataset.metadata['experiment']

        for key, item in minimum_info.items():
            if key not in self.experiment:
                self.experiment[key] = item

    def set_dimension(self):
        spec_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
        self.spec_dim = self.dataset._axes[spec_dim[0]]
        old_energy_scale = self.spec_dim
        self.dataset.set_dimension(spec_dim[0], sidpy.Dimension(np.array(self.energy_scale),
                                                                     name=old_energy_scale.name,
                                                                     dimension_type=sidpy.DimensionType.SPECTRAL,
                                                                     units='eV',
                                                                     quantity='energy loss'))

    def update(self):

        self.ui.offsetEdit.setText(f"{self.experiment['offset']:.3f}")
        self.ui.dispersionEdit.setText(f"{self.experiment['dispersion']:.3f}")
        self.ui.timeEdit.setText(f"{self.experiment['exposure_time']:.6f}")

        self.ui.convEdit.setText(f"{self.experiment['convergence_angle']:.2f}")
        self.ui.collEdit.setText(f"{self.experiment['collection_angle']:.2f}")
        self.ui.E0Edit.setText(f"{self.experiment['acceleration_voltage']/1000.:.2f}")

        self.ui.binningEdit.setText(f"{self.experiment['binning']}")
        self.ui.conversionEdit.setText(f"{self.experiment['conversion']:.2f}")
        self.ui.fluxEdit.setText(f"{self.experiment['flux']:.2f}")
        self.ui.VOAEdit.setText(f"{self.experiment['current']:.2f}")

    def on_enter(self):
        sender = self.sender()

        if sender == self.ui.offsetEdit:
            value = float(str(sender.displayText()).strip())
            self.experiment['offset'] = value
            sender.setText(f"{value:.2f}")
            self.energy_scale = self.energy_scale - self.energy_scale[0] + value
            self.set_dimension()
            self.plot()
        elif sender == self.ui.dispersionEdit:
            value = float(str(sender.displayText()).strip())
            self.experiment['dispersion'] = value
            self.energy_scale = np.arange(len(self.energy_scale)) * value + self.energy_scale[0]
            self.set_dimension()
            self.plot()
            sender.setText(f"{value:.3f}")
        elif sender == self.ui.timeEdit:
            value = float(str(sender.displayText()).strip())
            self.experiment['exposure_time'] = value
            sender.setText(f"{value:.2f}")
        elif sender == self.ui.convEdit:
            value = float(str(sender.displayText()).strip())
            self.experiment['convergence_angle'] = value
            sender.setText(f"{value:.2f}")
        elif sender == self.ui.collEdit:
            value = float(str(sender.displayText()).strip())
            self.experiment['collection_angle'] = value
            sender.setText(f"{value:.2f}")
        elif sender == self.ui.E0Edit:
            value = float(str(sender.displayText()).strip())
            self.experiment['acceleration_voltage'] = value*1000.0
            sender.setText(f"{value:.2f}")
        elif sender == self.ui.E0Edit:
            value = float(str(sender.displayText()).strip())
            self.experiment['acceleration_voltage'] = value
            sender.setText(f"{value:.2f}")
        elif sender == self.ui.binXEdit or sender == self.ui.binYEdit:
            if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
                bin_x = int(self.ui.binXEdit.displayText())
                bin_y = int(self.ui.binYEdit.displayText())
                self.experiment['SI_bin_x'] = bin_x
                self.experiment['SI_bin_y'] = bin_y
                self.dataset.view.set_bin([bin_x, bin_y])
                self.ui.binXEdit.setText(str(self.dataset.view.bin_x))
                self.ui.binYEdit.setText(str(self.dataset.view.bin_y))
        else:
            print('not supported yet')

    def plot(self):
        if self.dataset.data_type == sidpy.DataType.SPECTRAL_IMAGE:
            spectrum = self.dataset.view.get_spectrum()
            self.axis = self.dataset.view.axes[1]
        else:
            spectrum = np.array(self.dataset)
            self.axis = self.dataset.view.axis

        x_limit = self.axis.get_xlim()
        y_limit = self.axis.get_ylim()
        self.axis.clear()

        self.axis.plot(self.energy_scale, spectrum, label='spectrum')
        self.axis.set_xlim(x_limit)
        self.axis.set_ylim(y_limit)
        self.figure.canvas.draw_idle()

    def on_list_enter(self):
        sender = self.sender()
        if sender == self.ui.TEMList:
            microscope.set_microscope(self.ui.TEMList.currentText())
            self.experiment['microscope'] = microscope.name
            self.experiment['convergence_angle'] = microscope.alpha
            self.experiment['collection_angle'] = microscope.beta
            self.experiment['acceleration_voltage'] = microscope.E0
            self.update()

    def set_energy_scale(self):
        self.energy_dlg = ieels.EnergySelector(self.dataset)

        self.energy_dlg.signal_selected[bool].connect(self.set_energy)
        self.energy_dlg.show()

    def set_energy(self, k):
        spec_dim = self.dataset.get_dimensions_by_type(sidpy.DimensionType.SPECTRAL)
        self.spec_dim = self.dataset._axes[spec_dim[0]]

        self.energy_scale = self.spec_dim.values
        self.experiment['offset'] = self.energy_scale[0]
        self.experiment['dispersion'] = self.energy_scale[1] - self.energy_scale[0]
        self.update()

    def on_check(self):
        pass

    def set_action(self):
        self.ui.offsetEdit.editingFinished.connect(self.on_enter)
        self.ui.dispersionEdit.editingFinished.connect(self.on_enter)
        self.ui.timeEdit.editingFinished.connect(self.on_enter)

        self.ui.TEMList.activated[str].connect(self.on_list_enter)

        self.ui.convEdit.editingFinished.connect(self.on_enter)
        self.ui.collEdit.editingFinished.connect(self.on_enter)
        self.ui.E0Edit.editingFinished.connect(self.on_enter)
        self.ui.binningEdit.editingFinished.connect(self.on_enter)
        self.ui.conversionEdit.editingFinished.connect(self.on_enter)
        self.ui.fluxEdit.editingFinished.connect(self.on_enter)
        self.ui.VOAEdit.editingFinished.connect(self.on_enter)
        self.ui.energy_button.clicked.connect(self.set_energy_scale)
        self.ui.binXEdit.editingFinished.connect(self.on_enter)
        self.ui.binYEdit.editingFinished.connect(self.on_enter)
