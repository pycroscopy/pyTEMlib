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
sys.path.insert(0, '../../pyNSID/')
sys.path.insert(1, "../")

import sidpy
import pyNSID
print(sidpy.__version__)
print('pyNSID', pyNSID.__version__)
import pyTEMlib.file_tools as ft

class TestFileFunctions(unittest.TestCase):

    def test_open_file(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.hf5')  # GOLD-NP-DIFF.dm3')

        datasets = ft.open_file(file_name, write_hdf_file=True)
        dataset = datasets['Channel_000']

        self.assertIsInstance(dataset, sidpy.Dataset)
        self.assertIsInstance(dataset.h5_dataset, h5py.Dataset)
        dataset.h5_dataset.file.close()
        datasets = ft.open_file(file_name)
        dataset = datasets['Channel_000']

        self.assertIsInstance(dataset, sidpy.Dataset)
        # self.assertTrue(dataset.h5_dataset is None)

    def test_file_widget(self):     
        file_list = ft.FileWidget()
        self.assertTrue('.' in file_list.dir_list)
        
    def test_choose_dataset(self):     
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.hf5')
        datasets = ft.open_file(file_name, write_hdf_file=True)
        dataset = datasets['Channel_000']

        data_chooser = ft.ChooseDataset(dataset)
        for dset in data_chooser.datasets.values():
            self.assertIsInstance(dset, sidpy.Dataset)
        
    def test_update_directory_list(self):
        file_path = os.path.dirname(os.path.abspath(__file__)) 
        file_dictionary = ft.update_directory_list(os.path.join(file_path, '../example_data'))
        self.assertIsInstance(file_dictionary, dict) 
        
    def test_get_last_path(self):
        last_path = ft.get_last_path()
        self.assertIsInstance(last_path, str)
        
    def test_save_path(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.dm3')
        last_path = ft.save_path(file_name)
        self.assertIsInstance(last_path, str)
        
    def test_savefile_dialog_Qt(self):
        # g = ft.savefile_dialog_Qt()
        self.assertTrue(True)
    
    def test_read_hf5(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.hf5')
        datasets = ft.open_file(file_name, write_hdf_file=True)
        dataset = datasets['Channel_000']
        self.assertIsInstance(dataset, sidpy.Dataset)
        self.assertIsInstance(dataset.h5_dataset, h5py.Dataset)
        dataset.h5_dataset.file.close()
        
    def test_add_structure(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.hf5')
        datasets = ft.open_file(file_name, write_hdf_file=False)
        dataset = datasets['Channel_000']
        dataset.structures.update({'Al': ase.build.bulk('Al', 'fcc', a=4.05, cubic=True)})
        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF-2.hf5')
        h5_group = ft.save_dataset(datasets, file_name)

        self.assertTrue('Structure_000' in dataset.h5_dataset.parent)
        h5_group.file.close()

    def test_add_dataset(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        #file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.dm3')
        #dataset2 = ft.open_file(file_name, write_hdf_file=True)
        #dataset2.h5_dataset.file.close()

        file_name = os.path.join(file_path, '../example_data/GOLD-NP-DIFF.hf5')
        dataset = ft.open_file(file_name, write_hdf_file=True)

        new_dataset =  dataset.copy()
        atoms = ase.build.bulk('Al', 'fcc', a=4.05, cubic=True)
        new_dataset['Channel_000'].structures.update({'Al': atoms})
        atoms = ase.build.bulk('Cu', 'fcc', a=4.05, cubic=True)
        new_dataset['Channel_000'].structures.update({'Cu': atoms})

        #h5_dataset = ft.add_dataset(new_dataset['Channel_000'], dataset['Channel_000'])
        #self.assertIsInstance(h5_dataset, h5py.Dataset)
        #self.assertTrue('Structure_000' in h5_dataset.parent)
        #self.assertTrue('Structure_001' in h5_dataset.parent)
        #self.assertTrue('Structure_001/Cu' in h5_dataset.parent)
        #dataset['Channel_000'].h5_dataset.file.close()


    def test_log_results(self):
        pass
        #file_path = os.path.dirname(os.path.abspath(__file__))
        #file_name = os.path.join(file_path, '../example_data/GOLD_NP_DIFF.hf5')
        #dataset = ft.open_file(file_name, write_hdf_file=True)
        
        #new_dataset = dataset.copy()
        #atoms = ase.build.bulk('Al', 'fcc', a=4.05, cubic=True)
        #new_dataset.structures.update({'Al': atoms})
        #atoms = ase.build.bulk('Cu', 'fcc', a=4.05, cubic=True)
        #new_dataset.structures.update({'Cu': atoms})
        #new_dataset.title = 'with_structure'
        
        #log_group = ft.log_results(dataset, new_dataset)
        #self.assertIsInstance(log_group, h5py.Group)
        #self.assertTrue('Structure_000' in log_group)
             
        
if __name__ == '__main__':
    unittest.main()
