"""Will move to sidpy"""
from PyQt5 import QtWidgets, QtCore


class ProgressDialog(QtWidgets.QDialog):
    """Simple dialog that consists of a Progress Bar and a Button.

    Clicking on the button results in the start of a timer and
    updates the progress bar.
    """

    def __init__(self, title='Progress', number_of_counts=100):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setGeometry(10, 10, 500, 50)
        self.progress.setMaximum(number_of_counts)
        self.show()

    def set_value(self, count):
        self.progress.setValue(count)
        QtWidgets.QApplication.processEvents()
