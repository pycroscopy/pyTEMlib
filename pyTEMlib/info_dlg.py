"""
Gui for info_dialog

Author: Gerd Duscher

"""
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

from pyTEMlib.microscope import microscope

Qt_available = True
try:
    from PyQt5 import QtCore, QtGui, QtWidgets

except:
    Qt_available = False
    print('Qt dialogs are not available')


if Qt_available:
    class UiDialog(object):
        def __init__(self, dialog):
            dialog.setObjectName('Spectrum Info')
            dialog.resize(371, 184)

            valid_float = QtGui.QDoubleValidator()
            valid_int = QtGui.QIntValidator()

            self.TEM = []
            self.TEM = microscope.get_available_microscope_names()

            self.layout = QtWidgets.QGridLayout()
            self.layout.setVerticalSpacing(2)

            self.separator1 = QtWidgets.QLabel(dialog)
            self.separator1.setAutoFillBackground(True)
            palette = self.separator1.palette()
            palette.setColor(self.separator1.backgroundRole(), QtCore.Qt.blue)
            palette.setColor(self.separator1.foregroundRole(), QtCore.Qt.white)
            self.separator1.setAlignment(QtCore.Qt.AlignCenter)
            self.separator1.setMaximumHeight(50)

            self.separator1.setPalette(palette)
            ######################################################################
            self.separator1.setText("Energy Scale")

            row = 0
            self.layout.addWidget(self.separator1, row, 0, 1, 4)
            ######################################################################

            row += 1
            self.offsetLabel = QtWidgets.QLabel("Offset")
            self.offsetEdit = QtWidgets.QLineEdit(" 1.00")
            self.offsetEdit.setValidator(valid_float)
            self.offsetUnit = QtWidgets.QLabel("eV")

            self.layout.addWidget(self.offsetLabel, row, 0)
            self.layout.addWidget(self.offsetEdit, row, 1)
            self.layout.addWidget(self.offsetUnit, row, 2)

            row += 1
            self.dispersionLabel = QtWidgets.QLabel("Dispersion")
            self.dispersionEdit = QtWidgets.QLineEdit(" 1.00")
            self.dispersionEdit.setValidator(valid_float)
            self.dispersionUnit = QtWidgets.QLabel("eV/channel")

            self.layout.addWidget(self.dispersionLabel, row, 0)
            self.layout.addWidget(self.dispersionEdit, row, 1)
            self.layout.addWidget(self.dispersionUnit, row, 2)

            row += 1
            self.energy_button = QtWidgets.QPushButton('Set Energy Scale', dialog)
            self.energy_button.setCheckable(True)

            self.layout.addWidget(self.energy_button, row, 1)

            row += 1
            self.timeLabel = QtWidgets.QLabel("Exp. Time")
            self.timeEdit = QtWidgets.QLineEdit(" 100.0")
            self.timeEdit.setValidator(valid_float)
            self.timeUnit = QtWidgets.QLabel("s")

            self.layout.addWidget(self.timeLabel, row, 0)
            self.layout.addWidget(self.timeEdit, row, 1)
            self.layout.addWidget(self.timeUnit, row, 2)

            self.separator2 = QtWidgets.QLabel(dialog)
            self.separator2.setAutoFillBackground(True)
            self.separator2.setAlignment(QtCore.Qt.AlignCenter)
            self.separator2.setMaximumHeight(50)
            self.separator2.setPalette(palette)

            row += 1
            ######################################################################
            self.separator2.setText("Microscope")
            self.layout.addWidget(self.separator2, row, 0, 1, 4)
            ######################################################################

            row += 1
            self.TEMList = QtWidgets.QComboBox()
            self.TEMList.setEditable(False)
            self.TEMList.addItems(self.TEM)

            self.layout.addWidget(self.TEMList, row, 1)

            row += 1
            self.convLabel = QtWidgets.QLabel("Conv. Angle")
            self.convEdit = QtWidgets.QLineEdit(" 100.0")
            self.convEdit.setValidator(valid_float)
            self.convUnit = QtWidgets.QLabel("mrad")
            self.layout.addWidget(self.convLabel, row, 0)
            self.layout.addWidget(self.convEdit, row, 1)
            self.layout.addWidget(self.convUnit, row, 2)

            row += 1
            self.collLabel = QtWidgets.QLabel("Coll. Angle")
            self.collEdit = QtWidgets.QLineEdit(" 10.0")
            self.collEdit.setValidator(valid_float)
            self.collUnit = QtWidgets.QLabel("mrad")
            self.layout.addWidget(self.collLabel, row, 0)
            self.layout.addWidget(self.collEdit, row, 1)
            self.layout.addWidget(self.collUnit, row, 2)

            row += 1
            self.E0Label = QtWidgets.QLabel("Acc. Voltage")
            self.E0Edit = QtWidgets.QLineEdit(" 100.0")
            self.E0Edit.setValidator(valid_float)
            self.E0Unit = QtWidgets.QLabel("kV")
            self.layout.addWidget(self.E0Label, row, 0)
            self.layout.addWidget(self.E0Edit, row, 1)
            self.layout.addWidget(self.E0Unit, row, 2)

            self.separator3 = QtWidgets.QLabel(dialog)
            self.separator3.setAutoFillBackground(True)
            self.separator3.setAlignment(QtCore.Qt.AlignCenter)
            self.separator3.setMaximumHeight(50)
            self.separator3.setPalette(palette)

            row += 1
            ######################################################################
            self.separator3.setText("Quantification")
            self.layout.addWidget(self.separator3, row, 0, 1, 4)
            ######################################################################

            row += 1
            self.binningLabel = QtWidgets.QLabel("Binning")
            self.binningEdit = QtWidgets.QLineEdit(" 1")
            self.binningEdit.setValidator(valid_int)
            self.layout.addWidget(self.binningLabel, row, 0)
            self.layout.addWidget(self.binningEdit, row, 1)

            
            
            row += 1
            self.check_probability = QtWidgets.QCheckBox("Probab.")
            self.check_probability.setObjectName('probability')
            
            self.probabilityLabel = QtWidgets.QLabel("Flux")
            self.get_flux_button = QtWidgets.QPushButton('', )
            self.flux_choice = ['None', 'Open File']
            self.select_flux = QtWidgets.QComboBox()
            self.select_flux.setEditable(False)
            self.select_flux.addItems(self.flux_choice)
            self.layout.addWidget(self.probabilityLabel, row, 0)
            self.layout.addWidget(self.select_flux, row, 1)
            self.layout.addWidget(self.check_probability, row, 2)
            self.select_flux.setObjectName('select_flux_list')
            
            row += 1
            self.conversionLabel = QtWidgets.QLabel("Conversion")
            self.conversionEdit = QtWidgets.QLineEdit(" 25.0")
            self.conversionEdit.setValidator(valid_float)
            self.conversionUnit = QtWidgets.QLabel("e<sup>-</sup>/counts")
            self.layout.addWidget(self.conversionLabel, row, 0)
            self.layout.addWidget(self.conversionEdit, row, 1)
            self.layout.addWidget(self.conversionUnit, row, 2)
            row += 1
            self.fluxLabel = QtWidgets.QLabel("Flux")
            self.fluxEdit = QtWidgets.QLineEdit(" 100.0")
            self.fluxEdit.setValidator(valid_float)
            self.fluxUnit = QtWidgets.QLabel("e<sup>-</sup>/s")
            self.layout.addWidget(self.fluxLabel, row, 0)
            self.layout.addWidget(self.fluxEdit, row, 1)
            self.layout.addWidget(self.fluxUnit, row, 2)
            row += 1
            self.VOALabel = QtWidgets.QLabel("Measurement")
            self.VOAEdit = QtWidgets.QLineEdit(" 10.0")
            self.VOAEdit.setValidator(valid_float)
            self.VOAUnit = QtWidgets.QLabel("pA")
            self.layout.addWidget(self.VOALabel, row, 0)
            self.layout.addWidget(self.VOAEdit, row, 1)
            self.layout.addWidget(self.VOAUnit, row, 2)

            self.separator4 = QtWidgets.QLabel(dialog)
            self.separator4.setAutoFillBackground(True)
            self.separator4.setAlignment(QtCore.Qt.AlignCenter)
            self.separator4.setMaximumHeight(50)
            self.separator4.setPalette(palette)
            ######################################################################
            self.separator4.setText("Spectrum Image")

            row += 1
            self.layout.addWidget(self.separator4, row, 0, 1, 4)
            ######################################################################

            self.binXLabel = QtWidgets.QLabel('Bin X:')
            self.binXEdit = QtWidgets.QLineEdit('1')
            self.binXEdit.setValidator(valid_int)
            self.binXUnit = QtWidgets.QLabel('pixel')

            row += 1
            self.layout.addWidget(self.binXLabel, row, 0)
            self.layout.addWidget(self.binXEdit, row, 1)
            self.layout.addWidget(self.binXUnit, row, 2)

            self.binYLabel = QtWidgets.QLabel('Bin Y:')
            self.binYEdit = QtWidgets.QLineEdit('1')
            self.binYEdit.setValidator(valid_int)
            self.binYUnit = QtWidgets.QLabel('pixel')

            row += 1
            self.layout.addWidget(self.binYLabel, row, 0)
            self.layout.addWidget(self.binYEdit, row, 1)
            self.layout.addWidget(self.binYUnit, row, 2)

            self.statusBar = QtWidgets.QStatusBar()
            self.layout.addWidget(self.statusBar, row+1, 0, 1, 4)

            dialog.setLayout(self.layout)

            dialog.setWindowTitle("Spectrum Info")

            QtCore.QMetaObject.connectSlotsByName(dialog)
