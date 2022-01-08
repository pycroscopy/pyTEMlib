# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np
import ase
import ase.build
import sys
sys.path.insert(0, "../")

import pyTEMlib.graph_tools as graph
from pyTEMlib import crystal_tools

class TestUtilityFunctions(unittest.TestCase):

    def test_circum_center_tetrahedron(self):
        # tetrahedron = np.random.random([4, 3])
        tetrahedron = np.asarray([[0, -1 / np.sqrt(3), 0], [0.5, 1 / (2 * np.sqrt(3)), 0],
                                  [-0.5, 1 / (2 * np.sqrt(3)), 0], [0, 0, np.sqrt(2 / 3)]])
        center, radius = graph.circum_center(tetrahedron)
        self.assertEqual(center.shape, (3,))
        self.assertTrue(np.allclose(center, (0., 0., 0.204), atol=1e-3))

    def test_circum_center_weighted(self):
        # tetrahedron = np.random.random([4, 3])
        tetrahedron = np.asarray([[0, -1 / np.sqrt(3), 0], [0.5, 1 / (2 * np.sqrt(3)), 0],
                                  [-0.5, 1 / (2 * np.sqrt(3)), 0], [0, 0, np.sqrt(2 / 3)]])
        atom_radii = np.array([0.1, 0.1, 0.1, 0.2])
        center, radius = graph.interstitial_sphere_center(tetrahedron, atom_radii)
        self.assertEqual(center.shape, (3,))
        self.assertTrue(np.allclose(center, (0., 0., 0.1256), atol=1e-3))
        self.assertTrue(np.allclose(np.linalg.norm(tetrahedron-center, axis=1), atom_radii+radius, atol=1e-3))

    def test_circum_center_radius(self):
        # tetrahedron = np.random.random([4, 3])
        tetrahedron = np.asarray([[0, -1 / np.sqrt(3), 0], [0.5, 1 / (2 * np.sqrt(3)), 0],
                                  [-0.5, 1 / (2 * np.sqrt(3)), 0], [0, 0, np.sqrt(2 / 3)]])
        atom_radii = np.array([0.1, 0.1, 0.1, 0.1])
        center, radius = graph.interstitial_sphere_center(tetrahedron, atom_radii)
        center2, radius2 = graph.circum_center(tetrahedron)

        self.assertEqual(radius, radius2-atom_radii[0])
        self.assertTrue(np.allclose(center, center2, atol=1e-3))

    def test_circum_center_triangle(self):
        triangle = np.asarray([[5., 7.], [6., 6.], [2., -2.]])

        center, radius = graph.circum_center(triangle)
        self.assertTrue(np.allclose(center, (2., 3.)))

    def test_circum_center_triangle_weighted(self):
        triangle = np.asarray([[5., 7.], [6., 6.], [2., -2.]])

        center, radius = graph.interstitial_sphere_center(triangle, [0.3, 0.3, 0.4])
        self.assertTrue(np.allclose(center, (2.04, 3.04), atol=1e-2))

    def test_circum_center_line(self):
        triangle = np.asarray([[1., 0.], [2., 0.], [3., 0.]])

        center, radius = graph.circum_center(triangle)
        self.assertTrue(np.allclose(center, (0., 0.)))


class TestPolyhedraFunctions(unittest.TestCase):

    def test_find_polyhedra(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True) * (2, 2, 2)
        polyhedra = graph.find_polyhedra(atoms)
        self.assertTrue(len(polyhedra)>=50)

    def test_sort_polyhedra(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True) * (2, 2, 2)
        atoms[7].symbol = 'Cs'
        polyhedra = graph.find_polyhedra(atoms)
        indices = graph.sort_polyhedra_by_vertices(polyhedra, visible=[6], verbose=True)
        self.assertEqual(len(indices), 3)

    def test_plot_super_cell(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True) * (2, 2, 2)
        super_cell = graph.plot_super_cell(atoms)
        self.assertEqual(atoms.cell.array[0, 0], super_cell.cell.array[0, 0])

    def test_plot_polyhedron(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True) * (2, 2, 2)
        polyhedra = graph.find_polyhedra(atoms)
        data = graph.plot_polyhedron(polyhedra, range(len(polyhedra)), center=False)
        self.assertTrue(len(data) >= 100)
        self.assertIsInstance(data[0], dict)

    def test_plot_bonds(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True) * (2, 2, 2)
        polyhedra = graph.find_polyhedra(atoms)

        data = graph.plot_bonds(polyhedra)
        self.assertTrue(len(atoms.info['graph']['connectivity_matrix']) > 10)
        self.assertIsInstance(data[0], dict)
        self.assertTrue(len(data) > 10)
        

    def test_get_boundary_polyhedra(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True) * (2, 2, 2)
        polyhedra = graph.find_polyhedra(atoms)
        indices = graph.get_boundary_polyhedra(polyhedra)
        self.assertTrue(len(indices) >= 1)


if __name__ == '__main__':
    unittest.main()