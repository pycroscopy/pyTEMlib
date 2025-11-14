# -*- coding: utf-8 -*-
"""
Created on December 28 2021

@author: Gerd Duscher
"""
# -*- coding: utf-8 -*-
"""
Created on December 28 2021

@author: Gerd Duscher
"""
from tracemalloc import start
import unittest

import os
import numpy as np
import sidpy
print(sidpy.__version__)
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

import pyTEMlib
print(pyTEMlib.__version__)


def get_dataset():
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(file_path, '../example_data/EDS-STO.emd')
    datasets = pyTEMlib.file_tools.open_file(file_name)
    dataset = datasets['SuperX']
    return dataset
spectrum = get_dataset()

class TestFileFunctions(unittest.TestCase):
   
    def test_dataset(self):
        """Test loading dataset"""
        
        start = np.searchsorted(spectrum.energy_scale.values, 100)
        energy_scale = spectrum.energy_scale.values[start:]
        detector_Efficiency= pyTEMlib.eds_tools.detector_response(spectrum)  # tags, spectrum.energy_scale.values[start:])
        if 'start_energy' not in spectrum.metadata['EDS']['detector']:
            spectrum.metadata['EDS']['detector']['start_energy'] = 120
        
        self.assertIsInstance(spectrum, sidpy.Dataset)

    def test_find_elements(self):
        """Test finding elements in EDS spectrum"""     
        minor_peaks = pyTEMlib.eds_tools.detect_peaks(spectrum, minimum_number_of_peaks=10)

        keys = list(spectrum.metadata['EDS'].keys())
        for key in keys:
            if len(key) < 3:
                del spectrum.metadata['EDS'][key]

        elements = pyTEMlib.eds_tools.peaks_element_correlation(spectrum, minor_peaks)
        spectrum.metadata['EDS'].update(pyTEMlib.eds_tools.get_x_ray_lines(spectrum, elements))
        print(spectrum.metadata['EDS'].keys())
        self.assertIsInstance(elements, list)
        self.assertTrue('Ti' in elements)

    def test_fit_spectrum(self):
        """Test fitting EDS spectrum"""
        peaks, _ = pyTEMlib.eds_tools.fit_model(spectrum, use_detector_efficiency=True)
        model = pyTEMlib.eds_tools.get_model(spectrum)
        self.assertIsInstance(model, np.ndarray)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertTrue(np.sum(peaks) > .9)

    def test_quantify_xsection(self):
        """Test quantification using cross sections"""
        pyTEMlib.eds_tools.quantify_eds(spectrum, mask=['Cu'])
        self.assertIn('GUI', spectrum.metadata['EDS'])
        self.assertIn('Cu', spectrum.metadata['EDS']['GUI'])

    def test_quantify_kfactors(self):
        """Test quantification using k-factors"""
        q_dict = pyTEMlib.eds_tools.load_k_factors()
        pyTEMlib.eds_tools.quantify_eds(spectrum, q_dict, mask=['Cu'])
        self.assertIsInstance(q_dict, dict)
        self.assertIn('GUI', spectrum.metadata['EDS'])
        self.assertIn('Cu', spectrum.metadata['EDS']['GUI'])


    def test_r_absorption(self):
        """Test absorption correction"""
        pyTEMlib.eds_tools.apply_absorption_correction(spectrum, 30)
        self.assertIn('corrected-atom%', spectrum.metadata['EDS']['GUI']['Ti'])
