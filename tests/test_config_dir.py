# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2020

@author: Gerd Duscher
"""
import unittest
import sys
import os

sys.path.append("../pyTEMlib/")
import pyTEMlib.config_dir


class TestPackageImport(unittest.TestCase):

    def test_config_path(self):
        path = pyTEMlib.config_dir.config_path
        self.assertTrue(isinstance(path, str))
        filename = os.path.join(path, 'path.txt')
        # self.assertTrue(os.path.isfile(filename))

    def test_config_path2(self):
        path = pyTEMlib.config_dir.config_path
        self.assertTrue(isinstance(path, str))
        filename = os.path.join(path, 'path.txt')
        self.assertTrue(os.path.isfile(filename))


if __name__ == '__main__':
    unittest.main()
