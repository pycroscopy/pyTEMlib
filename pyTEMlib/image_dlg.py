"""
Gui for image_dialog

Author: Gerd Duscher

"""
# -*- coding: utf-8 -*-
Qt_available = True
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except:
    Qt_available = False
    print('Qt dialogs are not available')

from matplotlib.figure import Figure

from pyTEMlib.microscope import microscope

if Qt_available:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas

    class MySICanvas(Canvas):
        def __init__(self, parent, width=10, height=10, dpi=100):
            self.figure = Figure(figsize=(width, height), dpi=dpi)
            self.figure.subplots_adjust(bottom=.2)
            Canvas.__init__(self, self.figure)
            self.setParent(parent)

            Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

            Canvas.updateGeometry(self)


    class UiDialog(object):
        def __init__(self, dialog, parent=None):
            dialog.setObjectName('Image Info')
            dialog.resize(371, 184)

            valid_float = QtGui.QDoubleValidator()
            # valid_int = QtGui.QIntValidator()
            self.histogram = MySICanvas(parent, width=10, height=10, dpi=70)

            # Defining a plot instance (axes) and assigning a variable to it
            self.histogram.axes = self.histogram.figure.add_subplot(1, 1, 1)

            self.TEM = []
            self.TEM = microscope.get_available_microscope_names()

            plot_layout = QtWidgets.QGridLayout()

            # Adding histogram
            plot_layout.addWidget(self.histogram, 0, 0)

            # making a single widget out of histogram
            histogram_plot = QtWidgets.QWidget()
            histogram_plot.setLayout(plot_layout)

            layout = self.layout = QtWidgets.QGridLayout()
            self.layout.setVerticalSpacing(2)

            self.separator1 = QtWidgets.QLabel()
            self.separator1.setAutoFillBackground(True)
            palette = self.separator1.palette()
            palette.setColor(self.separator1.backgroundRole(), QtCore.Qt.blue)
            palette.setColor(self.separator1.foregroundRole(), QtCore.Qt.white)
            self.separator1.setAlignment(QtCore.Qt.AlignCenter)
            self.separator1.setMaximumHeight(50)

            self.separator1.setPalette(palette)
            ######################################################################
            self.separator1.setText("Microscope")
            layout.addWidget(self.separator1, 0, 0, 1, 3)
            row = 0
            layout.addWidget(self.separator1, row, 0, 1, 4)

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
            self.E0Label = QtWidgets.QLabel("Acc. Voltage")
            self.E0Edit = QtWidgets.QLineEdit(" 100.0")
            self.E0Edit.setValidator(valid_float)
            self.E0Unit = QtWidgets.QLabel("kV")
            self.layout.addWidget(self.E0Label, row, 0)
            self.layout.addWidget(self.E0Edit, row, 1)
            self.layout.addWidget(self.E0Unit, row, 2)

            self.separator2 = QtWidgets.QLabel()
            self.separator2.setAutoFillBackground(True)
            self.separator2.setAlignment(QtCore.Qt.AlignCenter)
            self.separator2.setMaximumHeight(50)
            self.separator2.setPalette(palette)

            row += 1
            ######################################################################
            self.separator2.setText("Image")
            layout.addWidget(self.separator2, row, 0, 1, 4)
            ######################################################################
            row += 1
            self.sizeLabel = QtWidgets.QLabel("Size")
            self.sizeEdit = QtWidgets.QLineEdit(" 1 x 1")
            self.sizeEdit.setValidator(valid_float)
            self.sizeUnit = QtWidgets.QLabel("px")

            self.layout.addWidget(self.sizeLabel, row, 0)
            self.layout.addWidget(self.sizeEdit, row, 1)
            self.layout.addWidget(self.sizeUnit, row, 2)

            row += 1
            self.timeLabel = QtWidgets.QLabel("Exp. Time")
            self.timeEdit = QtWidgets.QLineEdit(" 100.0")
            self.timeEdit.setValidator(valid_float)
            self.timeUnit = QtWidgets.QLabel("s")

            self.layout.addWidget(self.timeLabel, row, 0)
            self.layout.addWidget(self.timeEdit, row, 1)
            self.layout.addWidget(self.timeUnit, row, 2)

            self.separator3 = QtWidgets.QLabel(dialog)
            self.separator3.setAutoFillBackground(True)
            self.separator3.setAlignment(QtCore.Qt.AlignCenter)
            self.separator3.setMaximumHeight(50)
            self.separator3.setPalette(palette)

            row += 1
            ######################################################################
            self.separator3.setText("Histogram")
            self.layout.addWidget(self.separator3, row, 0, 1, 4)
            ######################################################################

            row += 1
            layout.addWidget(histogram_plot, row, 0, 1, 3)
            dialog.setLayout(layout)
