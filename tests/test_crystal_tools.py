# -*- coding: utf-8 -*-
"""
Created on January 23 2021

@author: Gerd Duscher
"""
import unittest
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import ase
import ase.build

import pyTEMlib.crystal_tools as cs

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')


class TestUtilityFunctions(unittest.TestCase):

    def test_ball_and_stick(self):
        atoms = ase.build.bulk('Fe', 'bcc', cubic=True)
        cell_2_plot = cs.ball_and_stick(atoms, extend=1)

        self.assertTrue(len(cell_2_plot.info['plot_cell']['corner_matrix']) == 12)

        balls_desired = [[0., 0., 0.], [0.5, 0.5, 0.5], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.],
                         [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]
        np.testing.assert_allclose(cell_2_plot.get_scaled_positions(wrap=False), balls_desired)

        self.assertTrue(len(cell_2_plot) == 9)

        bonds_desired = [[0, 1, 1, 1, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        np.testing.assert_allclose(cell_2_plot.info['plot_cell']['bond_matrix'].toarray(), bonds_desired)

    def test_from_dictionary(self):
        tags = {'unit_cell': np.array([[4.05, 0, 0], [0, 4.05, 0], [0, 0, 4.05]]),
                'elements': ['Al']*4,
                'base': np.array([[0., 0., 0.], [0., .5, .5], [.5, 0, .5], [.5, .5, 0.]])
                }
        crystal = cs.atoms_from_dictionary(tags)
        self.assertIsInstance(crystal, ase.Atoms)

    def test_get_dict(self):

        atoms = ase.build.bulk('Al', 'fcc', cubic=True)
        tags = cs.get_dictionary(atoms)
        self.assertTrue(atoms.get_chemical_formula() == tags['elements'])
        self.assertTrue(np.allclose(atoms.get_scaled_positions(), tags['base']))

        crystal2 = cs.atoms_from_dictionary(tags)
        self.assertTrue(crystal2.get_chemical_formula() == tags['elements'])

    def test_plot_unit_cell(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True)
        fig = cs.plot_unit_cell(atoms)
        x_y_z = fig.axes[0].lines[0].get_xydata()
        print(x_y_z)

    def test_get_symmetry(self):
        atoms = ase.build.bulk('Al', 'fcc', cubic=True)
        sym = cs.get_symmetry(atoms, verbose=True)
        self.assertTrue(sym)

    def test_structure_by_name(self):
        for key in cs.crystal_data_base.keys():
            a = cs.structure_by_name(key)
            self.assertIsInstance(a, ase.Atoms)

        a = cs.structure_by_name('Graphite')
        self.assertIsInstance(a, ase.Atoms)

        a = cs.structure_by_name('MoS2')
        self.assertIsInstance(a, ase.Atoms)


if __name__ == '__main__':
    unittest.main()
