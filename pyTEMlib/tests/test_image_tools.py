# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np
import ase
import ase.build
import sys
import os
sys.path.insert(0, "../")

import pyTEMlib.image_tools as it
import pyTEMlib.file_tools as ft

class TestUtilityFunctions(unittest.TestCase):

    def test_get_wavelength(self):
        wavelength =  it.get_wavelength(200000)
        self.assertTrue(np.allclose(wavelength, 0.0025079340450548005, atol=1e-3))
    """
    def test_read_image_info(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.dm3')
        dataset = ft.open_file(file_name)
        metadata = it.read_image_info(dataset)
    """
if __name__ == '__main__':
    unittest.main()