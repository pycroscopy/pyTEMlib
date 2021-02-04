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
    def test_cubic(self):
        a = 0.5
        desired = np.identity(3) * a
        actual = cs.cubic(a)
        self.assertIsNone(np.testing.assert_allclose(actual, desired))

        with self.assertRaises(TypeError):
            cs.cubic()
        with self.assertRaises(TypeError):
            cs.cubic('lattice_parameter')

    def test_from_parameters(self):
        a = b = 1
        c = 2
        alpha = beta = 90
        gamma = 120
        desired = [[1.0,  0.0,  0.0], [-0.5, 0.866, 0.0], [0.0,  0.0,  2.0]]
        actual = cs.from_parameters(a, b, c, alpha, beta, gamma)

        self.assertIsNone(np.testing.assert_allclose(actual, desired, atol=1e-4))
        with self.assertRaises(TypeError):
            cs.from_parameters(a, b, c, 'alpha', beta, gamma)
        with self.assertRaises(TypeError):
            cs.from_parameters(a, 'a', c, alpha, beta, gamma)

    def test_tetragonal(self):
        a = 1
        c = 2
        desired = [[1.0,  0.0,  0.0], [0.0,  1.0,  0.0], [0.0,  0.0,  2.0]]
        actual = cs.tetragonal(a, c)
        self.assertIsNone(np.testing.assert_allclose(actual, desired, atol=1e-4))
        with self.assertRaises(TypeError):
            cs.tetragonal(a, 'b')

    def test_bcc(self):
        actual_cell, actual_base, actual_atoms = cs.bcc(1, 'Fe')
        desired_cell = np.identity(3)
        desired_base = [(0., 0., 0.), (0.5, 0.5, 0.5)]
        desired_atoms = ['Fe', 'Fe']
        self.assertIsNone(np.testing.assert_allclose(actual_cell, desired_cell, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(actual_base, desired_base, atol=1e-4))
        self.assertEqual(actual_atoms, desired_atoms)

        with self.assertRaises(TypeError):
            cs.bcc('a', 'b')
        with self.assertRaises(TypeError):
            cs.bcc(1, 3)

    def test_fcc(self):
        actual_cell, actual_base, actual_atoms = cs.fcc(1, 'Fe')
        desired_cell = np.identity(3)
        desired_base = [(0., 0., 0.), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0., 0.5, 0.5)]
        desired_atoms = ['Fe']*4
        self.assertIsNone(np.testing.assert_allclose(actual_cell, desired_cell, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(actual_base, desired_base, atol=1e-4))
        self.assertEqual(actual_atoms, desired_atoms)

        with self.assertRaises(TypeError):
            cs.fcc('a', 'b')
        with self.assertRaises(TypeError):
            cs.fcc(1, 3)

    def test_dichalcogenide(self):
        u = 0.1
        actual_cell, actual_base, actual_atoms = cs.dichalcogenide(1, 2, u, 'Fe')
        desired_cell = [[1.0,  0.0,  0.0], [-0.5, 0.866, 0.0], [0.0,  0.0,  2.0]]
        desired_base = [(1 / 3., 2 / 3., 1 / 4.), (2 / 3., 1 / 3., 3 / 4.),
                        (2 / 3., 1 / 3., 1 / 4. + u), (2 / 3., 1 / 3., 1 / 4. - u),
                        (1 / 3., 2 / 3., 3 / 4. + u), (1 / 3., 2 / 3., 3 / 4. - u)]
        desired_atoms = ['Fe'] * 6

        self.assertIsNone(np.testing.assert_allclose(actual_cell, desired_cell, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(actual_base, desired_base, atol=1e-4))
        self.assertEqual(actual_atoms, desired_atoms)

        with self.assertRaises(TypeError):
            cs.dichalcogenide(1, 2, 'u', 'Fe')
        with self.assertRaises(TypeError):
            cs.dichalcogenide(1, 2, 0., 1)

    def test_wurzite(self):
        u = 0.1
        actual_cell, actual_base, actual_atoms = cs.wurzite(1, 2, u, 'Fe')
        desired_cell = [[1.0,  0.0,  0.0], [-0.5, 0.866, 0.0], [0.0,  0.0,  2.0]]
        desired_base =  [(2./3., 1./3., .500), (1./3., 2./3., 0.000), (2./3., 1./3., 0.5+u), (1./3., 2./3., u)]
        desired_atoms = ['Fe'] * 4

        self.assertIsNone(np.testing.assert_allclose(actual_cell, desired_cell, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(actual_base, desired_base, atol=1e-4))
        self.assertEqual(actual_atoms, desired_atoms)

        with self.assertRaises(TypeError):
            cs.wurzite(1, 2, 'u', 'Fe')
        with self.assertRaises(TypeError):
            cs.wurzite(1, 2, 0., 1)


    def test_rocksalt(self):
        actual_cell, actual_base, actual_atoms = cs.rocksalt(1, 'Fe')
        desired_cell = np.identity(3)
        desired_base =  [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5),
                         (0.5, 0.5, 0.5), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.0, 0.0)]
        desired_atoms = ['Fe'] * 8

        self.assertIsNone(np.testing.assert_allclose(actual_cell, desired_cell, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(actual_base, desired_base, atol=1e-4))
        self.assertEqual(actual_atoms, desired_atoms)

        with self.assertRaises(TypeError):
            cs.rocksalt('1', 'Fe')
        with self.assertRaises(TypeError):
            cs.rocksalt(1, 1)

    def test_zincblende(self):
        actual_cell, actual_base, actual_atoms = cs.zincblende(1, 'Fe')
        desired_cell = np.identity(3)
        desired_base =  [(0.00, 0.00, 0.00), (0.50, 0.00, 0.50), (0.50, 0.50, 0.00), (0.00, 0.50, 0.50),
                         (0.25, 0.25, 0.25), (0.75, 0.25, 0.75), (0.75, 0.75, 0.25), (0.25, 0.75, 0.75)]
        desired_atoms = ['Fe'] * 8

        self.assertIsNone(np.testing.assert_allclose(actual_cell, desired_cell, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(actual_base, desired_base, atol=1e-4))
        self.assertEqual(actual_atoms, desired_atoms)

        with self.assertRaises(TypeError):
            cs.zincblende('1', 'Fe')
        with self.assertRaises(TypeError):
            cs.zincblende(1, 1)


    def test_perovskite(self):
        actual_cell, actual_base, actual_atoms = cs.perovskite(1, 'Fe')
        desired_cell = np.identity(3)
        desired_base = [(0., 0., 0.), (0.5, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.), (0., 0.5, 0.5)]
        desired_atoms = ['Fe'] * 5

        self.assertIsNone(np.testing.assert_allclose(actual_cell, desired_cell, atol=1e-4))
        self.assertIsNone(np.testing.assert_allclose(actual_base, desired_base, atol=1e-4))
        self.assertEqual(actual_atoms, desired_atoms)

        with self.assertRaises(TypeError):
            cs.perovskite('1', 'Fe')
        with self.assertRaises(TypeError):
            cs.perovskite(1, 1)


if __name__ == '__main__':
    unittest.main()
