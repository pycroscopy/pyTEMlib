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

    def test_ball_and_stick(self):
        in_tags = {'unit_cell': np.identity(3), 'base': [[0, 0, 0], [0.5, 0.5, 0.5]], 'elements': ['Fe', 'Fe']}
        corners, balls, atomic_number, bonds = cs.ball_and_stick(in_tags, extend=1, max_bond_length=1.)

        corners_desired = [[(0.0, 0.0), (0.0, 0.0), (0.0, 1.0)], [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
                           [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
                           [(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)], [(1.0, 1.0), (0.0, 0.0), (0.0, 1.0)],
                           [(1.0, 1.0), (0.0, 1.0), (1.0, 1.0)], [(1.0, 1.0), (1.0, 1.0), (0.0, 1.0)],
                           [(1.0, 1.0), (1.0, 0.0), (0.0, 0.0)], [(0.0, 1.0), (0.0, 0.0), (1.0, 1.0)],
                           [(0.0, 1.0), (1.0, 1.0), (0.0, 0.0)], [(0.0, 1.0), (1.0, 1.0), (1.0, 1.0)]]
        np.testing.assert_allclose(corners, corners_desired)

        balls_desired = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.], [1., 0., 1.],
                         [1., 1., 0.], [1., 1., 1.], [0.5, 0.5, 0.5]]
        np.testing.assert_allclose(balls, balls_desired)

        self.assertTrue(len(atomic_number) == 9)

        bonds_desired = [[(0.0, 0.5), (0.0, 0.5), (0.0, 0.5)], [(0.0, 0.5), (0.0, 0.5), (1.0, 0.5)],
                         [(0.0, 0.5), (1.0, 0.5), (0.0, 0.5)], [(0.0, 0.5), (1.0, 0.5), (1.0, 0.5)],
                         [(1.0, 0.5), (0.0, 0.5), (0.0, 0.5)], [(1.0, 0.5), (0.0, 0.5), (1.0, 0.5)],
                         [(1.0, 0.5), (1.0, 0.5), (0.0, 0.5)], [(1.0, 0.5), (1.0, 0.5), (1.0, 0.5)],
                         [(0.5, 0.0), (0.5, 1.0), (0.5, 0.0)], [(0.5, 0.0), (0.5, 1.0), (0.5, 1.0)],
                         [(0.5, 1.0), (0.5, 0.0), (0.5, 0.0)], [(0.5, 1.0), (0.5, 0.0), (0.5, 1.0)],
                         [(0.5, 1.0), (0.5, 1.0), (0.5, 0.0)], [(0.5, 0.0), (0.5, 0.0), (0.5, 1.0)],
                         [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]]
        # np.testing.assert_allclose(bonds, bonds_desired)  # does not work in python 3.6, why?

if __name__ == '__main__':
    unittest.main()
