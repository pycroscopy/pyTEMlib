""" Interactive routines for EELS analysis

this file provides additional dialogs for EELS quantification

Author: Gerd Duscher
"""

Qt_available = True

try:
    from PyQt5 import QtCore, QtGui, QtWidgets

except:
    Qt_available = False
    print('Qt dialogs are not available')

if Qt_available:
    from pyTEMlib import eels_dialog

    from pyTEMlib import info_dialog
    from pyTEMlib import peak_dialog
    from pyTEMlib.eels_dialog_utilities import *

    CompositionDialog = eels_dialog.EELSDialog
    CurveVisualizer = eels_dialog.CurveVisualizer
    InfoDialog = info_dialog.InfoDialog
    PeakFitDialog = peak_dialog.PeakFitDialog
