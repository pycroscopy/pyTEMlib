"""
GUI definitions for EEELS_dialog
"""

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
        """ GUI definitions"""
        def __init__(self, dialog):
            dialog.setObjectName('Fit Composition Input')
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

            name = 'fit_start'

            self.label1 = QtWidgets.QLabel('Fit Start:')
            self.edit1 = QtWidgets.QLineEdit('0')
            self.edit1.setValidator(valid_float)
            self.unit1 = QtWidgets.QLabel('eV')
            self.edit1.setObjectName(name+'_edit')
            self.unit1.setObjectName(name+'_units')

            row += 1
            self.layout.addWidget(self.label1, row, 0)
            self.layout.addWidget(self.edit1, row, 1)
            self.layout.addWidget(self.unit1, row, 2)

            name = 'fit_end'

            self.label2 = QtWidgets.QLabel('Fit End:')
            self.edit2 = QtWidgets.QLineEdit('0')
            self.edit2.setValidator(valid_float)
            self.unit2 = QtWidgets.QLabel('eV')
            self.edit2.setObjectName(name + '_edit')
            self.unit2.setObjectName(name + '_units')

            row += 1
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
            self.separator2.setText("Elements")
            self.layout.addWidget(self.separator2, row, 0, 1, 4)
            ######################################################################

            name = 'edge'
            self.label3 = QtWidgets.QLabel('Edge:')
            self.edge_list = ['Edge 1', 'Edge 2', 'Edge 3', 'Edge 4', 'Edge 5', 'Edge 6']
            self.list3 = QtWidgets.QComboBox()
            self.list3.setEditable(False)
            self.list3.addItems(self.edge_list)
            self.check3 = QtWidgets.QCheckBox("Regions")

            row += 1
            self.layout.addWidget(self.label3, row, 0)
            self.layout.addWidget(self.list3, row, 1)
            self.layout.addWidget(self.check3, row, 2)

            self.list3.setObjectName(name + '_list')
            self.check3.setObjectName(name + '_check')

            name = 'element'

            self.label4 = QtWidgets.QLabel('Z:')
            self.edit4 = QtWidgets.QLineEdit('1')
            self.unit4 = QtWidgets.QLabel('H')
            self.edit4.setObjectName(name + '_edit')
            self.unit4.setObjectName(name + '_name')

            row += 1
            self.layout.addWidget(self.label4, row, 0)
            self.layout.addWidget(self.edit4, row, 1)
            self.layout.addWidget(self.unit4, row, 2)

            name = 'symmetry'
            self.edge_sym = ['K1', 'L3', 'M5', 'M3', 'M1', 'N1', 'N3', 'N5', 'N7', 'O1', 'O3', 'O5']

            self.label5 = QtWidgets.QLabel('Symmetry:')
            self.list5 = QtWidgets.QComboBox()
            self.list5.setEditable(False)
            self.list5.addItems(self.edge_sym)

            self.x_sec_meth = ['XRPA', 'Hydrog.', 'H-S GOS']
            self.select5 = QtWidgets.QComboBox()
            self.select5.setEditable(False)
            self.select5.addItems(self.x_sec_meth)

            row += 1
            self.layout.addWidget(self.label5, row, 0)
            self.layout.addWidget(self.list5, row, 1)
            self.layout.addWidget(self.select5, row, 2)

            self.list5.setObjectName(name + '_list')
            self.select5.setObjectName(name + '_method')

            name = 'onset'

            self.label6 = QtWidgets.QLabel('Onset:')
            self.edit6 = QtWidgets.QLineEdit('0')
            self.edit6.setValidator(valid_float)
            self.unit6 = QtWidgets.QLabel('eV')
            self.edit6.setObjectName(name + '_edit')
            self.unit6.setObjectName(name + '_units')

            row += 1
            self.layout.addWidget(self.label6, row, 0)
            self.layout.addWidget(self.edit6, row, 1)
            self.layout.addWidget(self.unit6, row, 2)

            name = 'excl_start'

            self.label7 = QtWidgets.QLabel('Excl. Start:')
            self.edit7 = QtWidgets.QLineEdit('0')
            self.edit7.setValidator(valid_float)
            self.unit7 = QtWidgets.QLabel('eV')
            self.edit7.setObjectName(name + '_edit')
            self.unit7.setObjectName(name + '_units')

            row += 1
            self.layout.addWidget(self.label7, row, 0)
            self.layout.addWidget(self.edit7, row, 1)
            self.layout.addWidget(self.unit7, row, 2)

            name = 'excl_end'

            self.label8 = QtWidgets.QLabel('Excl. End:')
            self.edit8 = QtWidgets.QLineEdit('0')
            self.edit8.setValidator(valid_float)
            self.unit8 = QtWidgets.QLabel('eV')
            self.edit8.setObjectName(name + '_edit')
            self.unit8.setObjectName(name + '_units')

            row += 1
            self.layout.addWidget(self.label8, row, 0)
            self.layout.addWidget(self.edit8, row, 1)
            self.layout.addWidget(self.unit8, row, 2)

            name = 'multiplier'

            self.label9 = QtWidgets.QLabel('Multiplier:')
            self.edit9 = QtWidgets.QLineEdit('0')
            self.edit9.setValidator(valid_float)
            self.unit9 = QtWidgets.QLabel('a.u.')
            self.edit9.setObjectName(name + '_edit')
            self.unit9.setObjectName(name + '_units')

            row += 1
            self.layout.addWidget(self.label9, row, 0)
            self.layout.addWidget(self.edit9, row, 1)
            self.layout.addWidget(self.unit9, row, 2)

            self.separator3 = QtWidgets.QLabel(dialog)
            self.separator3.setAutoFillBackground(True)
            self.separator3.setAlignment(QtCore.Qt.AlignCenter)
            self.separator3.setMaximumHeight(50)
            self.separator3.setPalette(palette)
            ######################################################################
            self.separator3.setText("Other Fit Parameters")

            row += 1
            self.layout.addWidget(self.separator3, row, 0, 1, 4)
            ######################################################################

            self.check10 = QtWidgets.QCheckBox("Conv. LL")
            self.show_edges = QtWidgets.QCheckBox("Show Edges")
            self.select10 = QtWidgets.QCheckBox("Fit Comp.")

            row += 1
            self.layout.addWidget(self.check10, row, 0)
            self.layout.addWidget(self.show_edges, row, 1)
            self.layout.addWidget(self.select10, row, 2)

            self.check10.setObjectName('conv_ll')
            self.select10.setObjectName('fit_comp')

            row += 1
            self.check_probability = QtWidgets.QCheckBox("Probab.")
            self.check_probability.setObjectName('probability')
            self.layout.addWidget(self.check_probability, row, 0)

            row += 1
            self.progress = QtWidgets.QProgressBar()
            self.progress.setGeometry(0, 0, 300, 25)
            self.progress.setMaximum(100)

            self.do_all_button = QtWidgets.QPushButton('Do All', dialog)

            self.do_fit_button = QtWidgets.QPushButton('Fit Composition', dialog)
            self.do_fit_button.setCheckable(True)

            self.layout.addWidget(self.do_all_button, row, 0)
            self.layout.addWidget(self.progress, row, 1)
            self.layout.addWidget(self.do_fit_button, row, 2)

            dialog.setLayout(self.layout)
            dialog.setWindowTitle('Fit Composition Input')

            # self.buttonBox.accepted.connect(Dialog.accept)
            # self.buttonBox.rejected.connect(Dialog.reject)
            QtCore.QMetaObject.connectSlotsByName(dialog)
