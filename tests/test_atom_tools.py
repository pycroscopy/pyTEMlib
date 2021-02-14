
# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np

import sidpy
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("../pyTEMlib/")
import pyTEMlib.file_tools as ft
ft.QT_available = False
import pyTEMlib.atom_tools as atom_tools

if sys.version_info.major == 3:
    unicode = str


def make_test_data():
    im = np.zeros([64, 64])
    im[4::8, 4::8] = 1

    image = sidpy.Dataset.from_array(gaussian_filter(im, sigma=2))
    image.data_type = 'Image'
    image.dim_0.dimension_type = 'spatial'
    image.dim_1.dimension_type = 'spatial'

    atoms = []
    for i in range(8):
        for j in range(8):
            atoms.append([8 * i + 4, 8 * j + 4])

    return image, atoms


class TestUtilityFunctions(unittest.TestCase):
    def test_find_atoms(self):
        image, atoms_placed = make_test_data()

        with self.assertRaises(TypeError):
            atom_tools.find_atoms(np.array(image))

        found_atoms = atom_tools.find_atoms(image)

        matches = 0
        for i, pos in enumerate(atoms_placed):
            if list(found_atoms[i, :2]) in atoms_placed:
                matches += 1
        self.assertEqual(64, matches)

    def test_atom_refine(self):
        image, atoms_placed = make_test_data()
        image = np.array(image)
        found_atoms_dict = atom_tools.atom_refine(image, atoms_placed, radius=3)
        found_atoms = np.array(found_atoms_dict['atoms'])
        matches = 0
        for i, pos in enumerate(atoms_placed):
            aa = np.round(found_atoms[i, :2]+.5, decimals=0)
            if list(aa) in atoms_placed:
                matches += 1
        self.assertEqual(64, matches)

    def test_intensity_area(self):
        image, atoms_placed = make_test_data()
        areas = atom_tools.intensity_area(np.array(image), atoms_placed, radius=3)

        self.assertIsNone(np.testing.assert_allclose(areas, 0.636566, atol=1e-1))


if __name__ == '__main__':
    unittest.main()
