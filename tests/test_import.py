# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:07:16 2020

@author: Suhas Somnath
"""
import unittest
import sys

sys.path.append("../pyTEMlib/")


class TestPackageImport(unittest.TestCase):

    def test_package_import(self):
        import pyTEMlib
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
