import os
import numpy as np
import h5py

Qt_available = True
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
except:
    print('Qt dialogs are not available')
    Qt_available = False

from PIL import Image, ImageQt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


if Qt_available:
    import pyTEMlib.file_tools as ft


    class FileIconDialog(QtWidgets.QDialog):
        """Qt5 Dialog to select directories or files from a list of Thumbnails

        The dialog converts the name of the nion file to the one in Nion's swift software,
        The dialog converts the name of the nion file to the one in Nion's swift software,
        because it is otherwise incomprehensible. Any Icon in a hf5 file will be displayed.

        Attributes
        ----------
        dir_name: str
            name of starting directory
        extension: list of str
            extensions of files to be listed  in widget

        Methods
        -------
        set_icon
        set_all_icons
        select
        get_directory
        update

        Example
        -------
        >>file_view = pyTEMlib.file_tools_qt.FileIconDialog('.')
        >>dataset = pyTEMlib.file_tools.open_file(file_view.file_name)
        >>view=dataset.plot()

        """

        def __init__(self, dir_name=None, extension=None):
            super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
            self.setModal(True)

            self.save_path = False
            self.dir_dictionary = {}
            self.dir_list = ['.', '..']
            self.display_list = ['.', '..']
            self.icon_size = 100
            self.file_name = None

            self.dir_name = '.'
            if dir_name is None:
                self.dir_name = ft.get_last_path()
                self.save_path = True
            elif os.path.isdir(dir_name):
                self.dir_name = dir_name

            self.get_directory()

            # setting geometry
            self.setGeometry(100, 100, 500, 400)

            # creating a QListWidget
            self.list_widget = QtWidgets.QListWidget(self)
            self.list_widget.setIconSize(QtCore.QSize(self.icon_size, self.icon_size))
            self.layout = QtWidgets.QVBoxLayout()
            self.layout.addWidget(self.list_widget)

            self.update()

            button_layout = QtWidgets.QHBoxLayout()

            button_select = QtWidgets.QPushButton('Select')
            button_layout.addWidget(button_select)
            button_get_icon = QtWidgets.QPushButton('Get Icon')
            button_layout.addWidget(button_get_icon)
            button_get_all_icons = QtWidgets.QPushButton('Get All Icons')
            button_layout.addWidget(button_get_all_icons)

            self.layout.addLayout(button_layout)
            self.setLayout(self.layout)

            self.list_widget.itemDoubleClicked.connect(self.select)
            button_select.clicked.connect(self.select)
            button_get_icon.clicked.connect(self.set_icon)
            button_get_all_icons.clicked.connect(self.set_all_icons)

            # showing all the widgets
            self.exec_()

        def set_icon(self):
            plt.ioff()
            figure = plt.figure(figsize=(1, 1))
            item = self.list_widget.currentItem().text()
            index = self.display_list.index(item)
            file_name = os.path.abspath(os.path.join(self.dir_name, self.dir_list[index]))
            dataset = ft.open_file(file_name)
            dataset.set_thumbnail(figure=figure)
            plt.close()
            plt.ion()
            self.update()

        def set_all_icons(self):

            plt.ioff()
            figure = plt.figure(figsize=(1, 1))
            for item in self.dir_list:
                file_name = os.path.join(self.dir_name, item)
                if os.path.isfile(file_name):
                    base_name, extension = os.path.splitext(file_name)
                    if extension in ['.hf5', '.dm3', '.dm4', '.ndata', '.hf5']:
                        try:
                            dataset = ft.open_file(file_name)
                            dataset.set_thumbnail(figure=figure)
                            dataset.h5_dataset.file.close()
                        except:
                            pass
            plt.close()
            plt.ion()
            self.update()

        def select(self):
            item = self.list_widget.currentItem().text()
            index = self.display_list.index(item)
            item = os.path.abspath(os.path.join(self.dir_name, self.dir_list[index]))
            self.setWindowTitle(" Chooser " + os.path.abspath(self.dir_name))
            if os.path.isdir(item):
                self.dir_name = item
                self.update()

            elif os.path.isfile(os.path.join(self.dir_name, item)):
                self.setWindowTitle(f" Selected File: {item}")
                self.file_name = item
                self.close()

        def get_directory(self):

            dir_list = os.listdir(self.dir_name)
            file_dict = ft.update_directory_list(self.dir_name)

            sort = np.argsort(file_dict['directory_list'])
            self.dir_list = ['.', '..']
            self.display_list = ['.', '..']
            for j in sort:
                self.display_list.append(f"{file_dict['directory_list'][j]}")
                self.dir_list.append(file_dict['directory_list'][j])

            sort = np.argsort(file_dict['display_file_list'])

            for i, j in enumerate(sort):
                if '--' in dir_list[j]:
                    self.display_list.append(f"{file_dict['display_file_list'][j]}")
                else:
                    self.display_list.append(f"{file_dict['display_file_list'][j]}")
                self.dir_list.append(file_dict['file_list'][j])

        def update(self):
            self.get_directory()
            self.setWindowTitle("File Chooser " + os.path.abspath(self.dir_name))
            # creating a QListWidget
            default_icons = QtWidgets.QFileIconProvider()
            self.list_widget.clear()

            item_list = []
            for index, item_text in enumerate(self.dir_list):
                if os.path.isdir(os.path.join(self.dir_name, item_text)):
                    icon = default_icons.icon(QtWidgets.QFileIconProvider.Folder)
                elif item_text[-4:] == '.hf5':
                    try:
                        f = h5py.File(os.path.join(self.dir_name, item_text), 'r')
                        if 'Thumbnail' in f:
                            picture = ImageQt.ImageQt(Image.fromarray(f['Thumbnail/Thumbnail'][()]))
                            icon = QtGui.QIcon(QtGui.QPixmap.fromImage(picture))
                        else:
                            icon = default_icons.icon(QtWidgets.QFileIconProvider.File)
                    except:
                        icon = default_icons.icon(QtWidgets.QFileIconProvider.File)
                else:
                    icon = default_icons.icon(QtWidgets.QFileIconProvider.File)
                item_list.append(QtWidgets.QListWidgetItem(icon, self.display_list[index]))
                self.list_widget.addItem(item_list[-1])
