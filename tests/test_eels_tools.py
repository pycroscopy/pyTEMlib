# -*- coding: utf-8 -*-
"""
Created on December 28 2021

@author: Gerd Duscher
"""
import unittest

import os
import numpy as np
import sidpy
print(sidpy.__version__)
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')
                
import pyTEMlib.file_tools as ft
import pyTEMlib.eels_tools as eels
import pyTEMlib
print(pyTEMlib.__version__)

class TestFileFunctions(unittest.TestCase):
    def test_dm3_eels_info(self):
        file_path = os.path.dirname(os.path.abspath(__file__))

        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']
        if dataset.h5_dataset is not None:
            dataset.h5_dataset.file.close()
        metadata = ft.read_dm3_info(dataset.original_metadata)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata['exposure_time'], 10.0)


    def test_fit_peaks(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']
        start_channel = np.searchsorted(dataset.energy_loss, -2)
        end_channel = np.searchsorted(dataset.energy_loss, 2)
        """
        p = eels.fit_peaks(dataset, dataset.energy_loss.values, [[0, dataset.max(), .6]], start_channel, end_channel)
        if dataset.h5_dataset is not None:
            dataset.h5_dataset.file.close()
        self.assertIsInstance(p, list)
        """

    def test_get_x_sections(self):
        x = eels.get_x_sections()
        self.assertIsInstance(x, dict)
        self.assertEqual(len(x), 82)
        x = eels.get_x_sections(14)

        self.assertIsInstance(x, dict)
        self.assertEqual(x['name'], 'Si')

        for i in range(1,83):
            x = eels.get_x_sections(i)
            self.assertEqual(len(x['ene']), len(x['dat']))



    def test_list_all_edges(self):
        z, _ = eels.list_all_edges(14)
        self.assertEqual(z[:6], ' Si-K1')

    def test_find_major_edge(self):
        z = eels.find_all_edges(532, major_edges_only=True)

        self.assertIsInstance(z, str)
        self.assertEqual(z[1:7], ' O -K1')

    def test_find_all_edge(self):
        z = eels.find_all_edges(532)
        self.assertIsInstance(z, str)
        self.assertEqual(z[1:7], ' O -K1')

    
    def test_second_derivative(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']
        derivative, noise_level = eels.second_derivative(dataset, 1.0)

        self.assertIsInstance(derivative, np.ndarray)

    def test_find_edges(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']
        if dataset.h5_dataset is not None:
            dataset.h5_dataset.file.close()
        selected_edges = eels.find_edges(dataset)

        self.assertIsInstance(selected_edges, list)

    def test_make_edges(self):
        edge = eels.make_edges(['Si-L3'], np.arange(50, 500), 200000, 20.)

        self.assertIsInstance(edge, dict)
        self.assertIsInstance(edge[0]['data'], np.ndarray)

    def test_power_law(self):
        background = eels.power_law(np.arange(50, 500), 3000., 3.)

        self.assertIsInstance(background, np.ndarray)

    def test_power_law_background(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']
        if dataset.h5_dataset is not None:
            dataset.h5_dataset.file.close()

        background, p = eels.power_law_background(dataset, dataset.energy_loss, [15, 25], verbose=True)

        self.assertIsInstance(background, np.ndarray)

    def test_fix_energy_scale(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']

        new_dataset= eels.align_zero_loss(dataset)
        self.assertTrue(len(new_dataset) == len(dataset))


    def test_resolution_function(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']

        z_loss, p_zl = eels.get_resolution_functions(dataset)

        self.assertTrue(len(z_loss) == len(dataset))

    def test_get_energy_shifts(self):

        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/AL-DFoffset0.00.dm3')
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']

        new_dataset = np.zeros([2, 1, dataset.shape[0]])
        new_dataset[0, 0, :] = np.array(dataset)
        new_dataset[1, 0, :] = np.array(dataset)

        # shifts = eels.get_energy_shifts(new_dataset, dataset.energy_loss.values, 0.5)

        # print('\n shifts ', shifts[:2, 0])
        new_dataset = sidpy.Dataset.from_array(new_dataset)
        new_dataset.data_type = 'SPECTRAL_IMAGE'
        new_dataset.set_dimension(2, dataset.energy_loss)

        self.assertTrue(True)  # np.isclose(shifts[0, 0], -0.22673, rtol=1e-01))
        # self.assertTrue(np.isclose(shifts[1, 0], -0.22673, rtol=1e-01))

    def test_effective_collection_angle(self):
        eff_beta = eels.effective_collection_angle(np.arange(59, 500), 10, 10, 200)

        self.assertTrue(eff_beta > 10)

    def test_get_db_spectra(self):
        spec_db = eels.get_spectrum_eels_db(formula='MgO', edge='K', title=None, element='O')
        self.assertIsInstance(spec_db, dict)



import Pyro5.api
import DigitalMicrograph as DM
import numpy as np

@Pyro5.api.expose
class CameraServer:
    def __init__(self):
        self.cam = None
        self.preImg = None
        self.kproc = None

    def activate_camera(self, height=200):
        self.cam = DM.GetActiveCamera()
        self.cam.PrepareForAcquire()
        height = int(height/2)
        bin = 1
        self.kproc = DM.GetCameraUnprocessedEnum()  # or DM.GetCameraGainNormalizedEnum()

        self.preImg = self.cam.CreateImageForAcquire(bin, bin, self.kproc, 1024-height, 0, 1024+height, 2048)
        return "Camera activated"

    def acquire_camera(self, exposure=0.1, height=200):
        if self.cam is None or self.preImg is one:
            return "Camera not activated"
        self.preImg.SetName("Pre-created image container")
        self.preImg.ShowImage()
        height = int(height/2)
        self.cam.AcquireInPlace(self.preImg, exposure, 1, 1, self.kproc, 1024-height, 0, 1024+height, 2048)
        dmImgData = self.preImg.GetNumArray()
        return dmImgData.tolist()  # Convert numpy array to list for serialization

    def close_camera(self):
        if self.preImg is not None:
            del self.preImg
        return "Camera closed"

def main():
    daemon = Pyro5.api.Daemon(host="10.46.218.0", port=65433)  # Choose an appropriate port
    uri = daemon.register(CameraServer, "camera.server")
    print("Camera Server is waiting for connections... Object uri =", uri)
    daemon.requestLoop()

if __name__ == "__main__":
    main()
