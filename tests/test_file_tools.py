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
sys.path.insert(0, "../")
sys.path.insert(0, '../../sidpy/')
import sidpy
print(sidpy.__version__)
import pyTEMlib.file_tools as ft


class TestFileFunctions(unittest.TestCase):
    
    def test_open_file(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD_NP_DIFF.hf5')  # GOLD-NP-DIFF.dm3')
        dataset = ft.open_file(file_name)
        
        self.assertIsInstance(dataset, sidpy.Dataset)
        self.assertIsInstance(dataset.h5_dataset, h5py.Dataset)
        dataset.h5_dataset.file.close()
        
    def test_file_widget(self):     
        file_list = ft.FileWidget()
        self.assertTrue('.' in file_list.dir_list)
        
    def test_choose_dataset(self):     
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD_NP_DIFF.hf5')
        dataset = ft.open_file(file_name)
        
        data_chooser = ft.ChooseDataset(dataset)
        self.assertIsInstance(data_chooser.dataset_list[0], sidpy.Dataset)
        
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
        file_name = os.path.join(file_path, '../example_data/GOLD_NP_DIFF.hf5')
        dataset = ft.open_file(file_name)
        self.assertIsInstance(dataset, sidpy.Dataset)
        self.assertIsInstance(dataset.h5_dataset, h5py.Dataset)
        dataset.h5_dataset.file.close()
        
    def test_add_structure(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD_NP_DIFF.hf5')
        dataset = ft.open_file(file_name)
        
        atoms = ase.build.bulk('Al', 'fcc', a=4.05, cubic=True)
        
        h5_group = dataset.h5_dataset.file
        structure_group = ft.h5_add_crystal_structure(h5_group, atoms)
        self.assertTrue(structure_group.name[:11] == '/Structure_')
        
        h5_group = dataset.h5_dataset.parent.parent
        structure_group = ft.h5_add_crystal_structure(h5_group, atoms)
        self.assertTrue('/Measurement_000/Channel_000/Structure_' == structure_group.name[:-3])
        dataset.h5_dataset.file.close()
        
    def test_add_dataset(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD_NP_DIFF.hf5')
        dataset = ft.open_file(file_name)

        new_dataset =  dataset.copy()
        atoms = ase.build.bulk('Al', 'fcc', a=4.05, cubic=True)
        new_dataset.structures = [atoms]
        atoms = ase.build.bulk('Cu', 'fcc', a=4.05, cubic=True)
        new_dataset.structures.append(atoms)
        
        h5_dataset = ft.add_dataset(new_dataset, dataset)
        self.assertIsInstance(h5_dataset, h5py.Dataset)
        self.assertTrue('Structure_000' in h5_dataset.parent.parent)
        self.assertTrue('Structure_001' in h5_dataset.parent.parent)
        dataset.h5_dataset.file.close()
        
    def test_log_results(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(file_path, '../example_data/GOLD_NP_DIFF.hf5')
        dataset = ft.open_file(file_name)
        
        new_dataset = dataset.copy()
        atoms = ase.build.bulk('Al', 'fcc', a=4.05, cubic=True)
        new_dataset.structures = [atoms]
        atoms = ase.build.bulk('Cu', 'fcc', a=4.05, cubic=True)
        new_dataset.structures.append(atoms)
        new_dataset.title = 'with_structure'
        
        log_group = ft.log_results(dataset, new_dataset)
        self.assertIsInstance(log_group, h5py.Group)
        self.assertTrue('Structure_000' in log_group)
             
        
if __name__ == '__main__':
    unittest.main()
