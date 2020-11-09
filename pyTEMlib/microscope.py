# -*- coding: cp1252 -*-
#################################################################
# Read microscope CSV file
# Original by EELSLab Author: Francisco Javier de la Peña
# Made more flexible for load microscopes with csv.DictReader
# for pyTEMLib by Gerd
# copyright 2012, Gerd Duscher
################################################################

import csv
import os.path

from pyTEMlib.config_dir import config_path
from pyTEMlib.defaults_parser import defaults

microscopes_file = os.path.join(config_path, 'microscopes.csv')


class Microscope(object):
    microscopes = {}
    name = None
    E0 = None
    alpha = None
    beta = None
    pppc = None
    correlation_factor = None

    def __init__(self):
        self.load_microscopes()
        
        defaults.microscope = defaults.microscope.replace('.', ' ')
        self.set_microscope(defaults.microscope)
    
    def load_microscopes(self):
        f = open(microscopes_file, 'r')

        labels = f.readline().strip().split(',')
#        print labels
        csv_read = csv.DictReader(f, labels, delimiter=",")
        
        for line in csv_read:
            tem = line['Microscope']
            self.microscopes[tem] = line
            for i in self.microscopes[tem]:
                if i != 'Microscope':
                    self.microscopes[tem][i] = float(self.microscopes[tem][i])
            
        f.close()
        
    def get_available_microscope_names(self):
        tem = []
        for scope in self.microscopes.keys():
            tem.append(scope)
        return tem
    
    def set_microscope(self, microscope_name):
        
        for key in self.microscopes[microscope_name]:
            exec('self.%s = self.microscopes[\'%s\'][\'%s\']' % (key, microscope_name, key))
        self.name = microscope_name
    
    def __repr__(self):
        info = '''
        Microscope parameters:
        -----------------------------
        
        Microscope: %s
        Convergence angle: %1.2f mrad
        Collection angle: %1.2f mrad
        Beam energy: %1.2E eV
        pppc: %1.2f
        Correlation factor: %1.2f
        ''' % (self.name, self.alpha, self.beta, self.E0,
               self.pppc, self.correlation_factor)
        return info


microscope = Microscope()
