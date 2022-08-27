"""GUI definitions for peak_fit_dialog"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

Qt_available = True
try:
    from PyQt5 import QtCore, QtGui, QtWidgets

except:
    Qt_available = False
    print('Qt dialogs are not available')


if Qt_available:
    class UiDialog(object):
        """GUI definitions for peak_fit_dialog"""

        def __init__(self, dialog):
            dialog.setObjectName('Fit Peaks')
            dialog.resize(371, 184)

            valid_float = QtGui.QDoubleValidator()
            valid_int = QtGui.QIntValidator()
            valid_int.setBottom(0)

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
            row = 0
            ######################################################################
            self.separator1.setText("Fit Region")
            self.layout.addWidget(self.separator1, row, 0, 1, 4)
            ######################################################################

            row += 1
            name = 'fit_start'
            self.label1 = QtWidgets.QLabel('Fit Start:')
            self.edit1 = QtWidgets.QLineEdit('0')
            self.edit1.setValidator(valid_float)
            self.unit1 = QtWidgets.QLabel('eV')
            self.edit1.setObjectName(name+'_edit')
            self.unit1.setObjectName(name+'_units')

            self.layout.addWidget(self.label1, row, 0)
            self.layout.addWidget(self.edit1, row, 1)
            self.layout.addWidget(self.unit1, row, 2)

            row += 1
            name = 'fit_end'
            self.label2 = QtWidgets.QLabel('Fit End:')
            self.edit2 = QtWidgets.QLineEdit('0')
            self.edit2.setValidator(valid_float)
            self.unit2 = QtWidgets.QLabel('eV')
            self.edit2.setObjectName(name + '_edit')
            self.unit2.setObjectName(name + '_units')

            self.layout.addWidget(self.label2, row, 0)
            self.layout.addWidget(self.edit2, row, 1)
            self.layout.addWidget(self.unit2, row, 2)

            self.separator2 = QtWidgets.QLabel(dialog)
            self.separator2.setAutoFillBackground(True)
            self.separator2.setAlignment(QtCore.Qt.AlignCenter)
            self.separator2.setMaximumHeight(50)
            self.separator2.setPalette(palette)

            row += 1
            ######################################################################
            self.separator2.setText("Processing")
            self.layout.addWidget(self.separator2, row, 0, 1, 4)
            ######################################################################
            row += 1
            # self.progress = QtWidgets.QProgressBar()
            # self.progress.setGeometry(0, 0, 300, 25)
            # self.progress.setMaximum(100)

            self.smooth_label = QtWidgets.QLabel('Iterations:')
            self.iteration_list = ['0', '1', '2', '3', '4']
            self.smooth_list = QtWidgets.QComboBox()
            self.smooth_list.setEditable(False)
            self.smooth_list.addItems(self.iteration_list)
            self.smooth_list.setCurrentIndex(1)

            self.smooth_button = QtWidgets.QPushButton('Smooth', dialog)
            self.smooth_button.setCheckable(False)

            self.layout.addWidget(self.smooth_label, row, 0)
            self.layout.addWidget(self.smooth_list, row, 1)
            self.layout.addWidget(self.smooth_button, row, 2)

            row += 1
            self.find_label = QtWidgets.QLabel('Nr. of Peaks:')
            self.find_edit = QtWidgets.QLineEdit('1')
            self.find_edit.setValidator(valid_int)
            self.find_button = QtWidgets.QPushButton('Find Peaks', dialog)
            self.find_button.setCheckable(False)

            self.layout.addWidget(self.find_label, row, 0)
            self.layout.addWidget(self.find_edit, row, 1)
            self.layout.addWidget(self.find_button, row, 2)

            self.separator2 = QtWidgets.QLabel(dialog)
            self.separator2.setAutoFillBackground(True)
            self.separator2.setAlignment(QtCore.Qt.AlignCenter)
            self.separator2.setMaximumHeight(50)
            self.separator2.setPalette(palette)

            row += 1
            ######################################################################
            self.separator2.setText("Peaks")
            self.layout.addWidget(self.separator2, row, 0, 1, 4)
            ######################################################################

            row += 1
            name = 'peak'
            self.label3 = QtWidgets.QLabel('Peaks:')
            self.peak_list = ['Peak 1', 'Add Peak']
            self.list3 = QtWidgets.QComboBox()
            self.list3.setEditable(False)
            self.list3.addItems(self.peak_list)
            self.unit3 = QtWidgets.QLabel('')

            self.layout.addWidget(self.label3, row, 0)
            self.layout.addWidget(self.list3, row, 1)
            self.layout.addWidget(self.unit3, row, 2)

            self.list3.setObjectName(name + '_list')
            # self.check3.setObjectName(name + '_check')

            name = 'symmetry'
            self.edge_sym = ['Gauss', 'Lorentzian', 'Drude', 'ZL']

            self.label4 = QtWidgets.QLabel('Symmetry:')
            self.list4 = QtWidgets.QComboBox()
            self.list4.setEditable(False)
            self.list4.addItems(self.edge_sym)

            row += 1
            self.layout.addWidget(self.label4, row, 0)
            self.layout.addWidget(self.list4, row, 1)

            self.list4.setObjectName(name + '_list')

            name = 'position'
            self.label5 = QtWidgets.QLabel('Position:')
            self.edit5 = QtWidgets.QLineEdit('1')
            self.unit5 = QtWidgets.QLabel('eV')
            self.edit5.setObjectName(name + '_edit')

            row += 1
            self.layout.addWidget(self.label5, row, 0)
            self.layout.addWidget(self.edit5, row, 1)
            self.layout.addWidget(self.unit5, row, 2)

            row += 1
            name = 'amplitude'
            self.label6 = QtWidgets.QLabel('Amplitude:')
            self.edit6 = QtWidgets.QLineEdit('0')
            self.edit6.setValidator(valid_float)
            self.unit6 = QtWidgets.QLabel('counts')
            self.edit6.setObjectName(name + '_edit')
            self.unit6.setObjectName(name + '_units')

            self.layout.addWidget(self.label6, row, 0)
            self.layout.addWidget(self.edit6, row, 1)
            self.layout.addWidget(self.unit6, row, 2)

            name = 'width'
            self.label7 = QtWidgets.QLabel('Width (FWHM):')
            self.edit7 = QtWidgets.QLineEdit('0')
            self.edit7.setValidator(valid_float)
            self.unit7 = QtWidgets.QLabel('eV')
            self.edit7.setObjectName(name + '_edit')
            self.unit7.setObjectName(name + '_units')

            row += 1
            self.layout.addWidget(self.label7, row, 0)
            self.layout.addWidget(self.edit7, row, 1)
            self.layout.addWidget(self.unit7, row, 2)

            row += 1
            name = 'asymmetry'
            self.label8 = QtWidgets.QLabel('Asymmetry:')
            self.edit8 = QtWidgets.QLineEdit('0')
            self.edit8.setValidator(valid_float)
            self.unit8 = QtWidgets.QLabel('%')
            self.label8.setObjectName(name + '_name')
            self.edit8.setObjectName(name + '_edit')
            self.unit8.setObjectName(name + '_units')

            self.layout.addWidget(self.label8, row, 0)
            self.layout.addWidget(self.edit8, row, 1)
            self.layout.addWidget(self.unit8, row, 2)

            row += 1
            self.fit_button = QtWidgets.QPushButton('Fit Peaks', dialog)
            self.fit_button.setCheckable(False)

            self.layout.addWidget(self.fit_button, row, 2)

            row += 1
            self.separator3 = QtWidgets.QLabel(dialog)
            self.separator3.setAutoFillBackground(True)
            self.separator3.setAlignment(QtCore.Qt.AlignCenter)
            self.separator3.setMaximumHeight(50)
            self.separator3.setPalette(palette)

            ######################################################################
            self.separator3.setText("Analysis")
            self.layout.addWidget(self.separator3, row, 0, 1, 4)
            ######################################################################

            row += 1
            self.labelwl = QtWidgets.QLabel('White-Line Ratio:')
            self.wl_list = ['Ratio']
            self.listwl = QtWidgets.QComboBox()
            self.listwl.setEditable(False)
            self.listwl.addItems(self.wl_list)
            self.unitswl = QtWidgets.QLabel('')

            self.layout.addWidget(self.labelwl, row, 0)
            self.layout.addWidget(self.listwl, row, 1)
            self.layout.addWidget(self.unitswl, row, 2)

            row += 1
            self.labelwls = QtWidgets.QLabel('White-Line Sum:')
            self.wls_list = ['Sum']
            self.listwls = QtWidgets.QComboBox()
            self.listwls.setEditable(False)
            self.listwls.addItems(self.wls_list)
            self.unitswls = QtWidgets.QLabel('')

            self.layout.addWidget(self.labelwls, row, 0)
            self.layout.addWidget(self.listwls, row, 1)
            self.layout.addWidget(self.unitswls, row, 2)

            dialog.setLayout(self.layout)
            dialog.setWindowTitle('Peak Fit Input')

            # self.buttonBox.accepted.connect(Dialog.accept)
            # self.buttonBox.rejected.connect(Dialog.reject)
            QtCore.QMetaObject.connectSlotsByName(dialog)
