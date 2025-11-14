# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2020

@author: Gerd Duscher
"""
import unittest

from pyTEMlib.microscope import Microscope

class TestPackageImport(unittest.TestCase):

    def test_microscope_init(self):
        tem = Microscope()
        self.assertTrue(isinstance(tem.name, str))

    def test_get_available_microscope_names(self):

        available_names = Microscope().get_available_microscope_names()
        print(available_names)
        self.assertTrue(isinstance(available_names, list))


if __name__ == '__main__':
    unittest.main()
