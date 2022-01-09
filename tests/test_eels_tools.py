# -*- coding: utf-8 -*-
"""
Created on December 28 2021

@author: Gerd Duscher
"""
import unittest
import h5py

import sys
import os
import ase.build
import numpy as np

sys.path.insert(0, "../")
sys.path.insert(0, '../../sidpy/')
import sidpy
print(sidpy.__version__)
import pyTEMlib.file_tools as ft
import pyTEMlib.eels_tools as eels

class TestFileFunctions(unittest.TestCase):
    def test_dm3_eels_info(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        dataset = ft.open_file(file_name)
        dataset.h5_dataset.file.close()
        metadata = eels.read_dm3_eels_info(dataset.original_metadata)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['exposure_time'], 10.0)

    def test_set_previous_quantification(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        dataset = ft.open_file(file_name)
        eels.set_previous_quantification(dataset)
        dataset.h5_dataset.file.close()

    def test_fit_peaks(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        dataset = ft.open_file(file_name)
        start_channel = np.searchsorted(dataset.energy_loss, -2)
        end_channel = np.searchsorted(dataset.energy_loss, 2)
        p = eels.fit_peaks(dataset, dataset.energy_loss.values, [[0 ,dataset.max(),.6]], start_channel, end_channel)
        dataset.h5_dataset.file.close()
        self.assertIsInstance(p, list)
        print(p)

    def test_get_x_sections(self):
        x = eels.get_x_sections()
        self.assertIsInstance(x, dict)
        self.assertEqual(len(x), 82)
        x = eels.get_x_sections(14)

        self.assertIsInstance(x, dict)
        self.assertEqual(x['name'], 'Si')

    def test_get_z(self):
        z = eels.get_z(14)
        self.assertEqual(z, 14)
        z = eels.get_z('Si')
        self.assertEqual(z, 14)
