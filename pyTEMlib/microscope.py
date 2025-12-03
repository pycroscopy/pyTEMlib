""" default microscope parameters from config file

Read microscope CSV file

for pyTEMLib by Gerd

copyright 2012, Gerd Duscher
updated 2021
"""
# -*- coding: utf-8 -*-

import csv
import os.path

from pyTEMlib.config_dir import config_path
microscopes_file = os.path.join(config_path, 'microscopes.csv')



class Microscope():
    """Class to read configuration file and provide microscope information"""
    microscopes = {}
    name = None
    E0 = None
    alpha = None
    beta = None
    pppc = None
    correlation_factor = None

    def __init__(self):
        self.load_microscopes()
        default_tem = self.microscopes[list(self.microscopes.keys())[0]]
        self.set_microscope(default_tem['Microscope'])

    def load_microscopes(self):
        """Load microscope parameters from CSV file."""
        with open(microscopes_file, 'r', encoding='utf-8') as f:
            labels = f.readline().strip().split(',')
            # print labels
            csv_read = csv.DictReader(f, labels, delimiter=",")

            for line in csv_read:
                tem = line['Microscope']
                self.microscopes[tem] = line
                for i in self.microscopes[tem]:
                    if i != 'Microscope':
                        self.microscopes[tem][i] = float(self.microscopes[tem][i])

    def get_available_microscope_names(self):
        """Return list of available microscope names."""
        tem = []
        for scope in self.microscopes:
            tem.append(scope)
        return tem

    def set_microscope(self, microscope_name):
        """Set current microscope by name."""
        if microscope_name in self.microscopes:
            self.name = microscope_name


microscope = Microscope()
