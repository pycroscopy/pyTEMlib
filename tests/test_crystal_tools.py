# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np

import pyTEMlib.crystal_tools as cs

import sys
sys.path.append("../pyTEMlib/")

if sys.version_info.major == 3:
    unicode = str


class TestUtilityFunctions(unittest.TestCase):

    def test_structure_by_name(self):
        with self.assertRaises(TypeError):
            cs.structure_by_name(1)

        actual = cs.structure_by_name('Gerd')
        self.assertEqual(actual, {})

        actual = cs.structure_by_name('FCC Fe')
        self.assertIsInstance(actual, dict)
        self.assertAlmostEqual(actual['a'], 0.3571)

        actual = cs.structure_by_name('BCC Fe')
        self.assertIsInstance(actual, dict)
        self.assertAlmostEqual(actual['a'], 0.2866)

        actual = cs.structure_by_name('diamond')
        self.assertEqual(actual['symmetry'], 'zinc_blende')

        actual = cs.structure_by_name('GaN Wurzite')
        self.assertEqual(actual['symmetry'], 'wurzite')

        actual = cs.structure_by_name('MgO')
        self.assertEqual(actual['symmetry'], 'rocksalt')

        actual = cs.structure_by_name('MoS2')
        self.assertEqual(actual['symmetry'], 'dichalcogenide')


if __name__ == '__main__':
    unittest.main()
