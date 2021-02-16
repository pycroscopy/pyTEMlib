# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:07:16 2021

@author: Gerd Duscher
"""

import unittest
import numpy as np

import sys
sys.path.append("../pyTEMlib/")
import pyTEMlib.KinsCat as ks


class TestUtilityFunctions(unittest.TestCase):

    def test_Zuo_fig_3_18(self):
        tags = ks.Zuo_fig_3_18(verbose=False)
        self.assertIsInstance(tags, dict)
        self.assertEqual(tags['crystal_name'], 'Silicon')
        self.assertEqual(tags['lattice_parameter_nm'], 0.514)
        self.assertEqual(tags['acceleration_voltage_V'], 101.6*1000.0)
        self.assertEqual(tags['convergence_angle_mrad'], 7.1)
        np.testing.assert_allclose(tags['zone_hkl'], np.array([-2, 2, 1]))

    def test_example(self):
        tags = ks.example(verbose=False)
        self.assertEqual(tags['plot HOLZ'], 1)
        self.assertEqual(tags['plot HOLZ'], 1)

    def test_zone_mistilt(self):
        rotated_zone_axis = ks.zone_mistilt([1, 0, 0], [45, 0, 0])
        np.testing.assert_allclose(rotated_zone_axis, [1, 0, 0])

        rotated_zone_axis = ks.zone_mistilt([1, 0, 0], [0, 10, 0])
        np.testing.assert_allclose(rotated_zone_axis, [0.98480775, 0., 0.17364818])

        with self.assertRaises(TypeError):
            ks.zone_mistilt([1, 0, 0], [0,  0])

        with self.assertRaises(TypeError):
            ks.zone_mistilt([1j, 0, 0], [0,  0])

    def test_get_symmetry(self):
        # Todo: better test
        self.assertTrue(ks.get_symmetry(np.identity(3), [[0, 0, 0], [0.5, 0.5, 0.5]], ['Fe', 'Fe']))

    def test_ball_and_stick(self):
        in_tags = {'unit_cell': np.identity(3), 'base': [[0, 0, 0], [0.5, 0.5, 0.5]], 'elements': ['Fe', 'Fe']}
        corners, balls, atomic_number, bonds = ks.ball_and_stick(in_tags, extend=1, max_bond_length=1.)

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

    def test_metric_tensor(self):
        # Todo: better testing
        np.testing.assert_allclose(ks.metric_tensor(np.identity(3)), np.identity(3))

    def test_vector_norm(self):
        g = [[2, 3, 4], [4, 5, 6]]
        vector_length = ks.vector_norm(g)
        np.testing.assert_allclose(vector_length, np.linalg.norm(g, axis=1))

    def test_make_pretty_labels(self):
        labels = ks.make_pretty_labels(np.array([[1, 0, 0], [1, 1, -1]]))
        self.assertEqual(labels[0], '[$\\bar {1},0,0} $]')
        self.assertEqual(labels[1], '[$\\bar {1},1,\\bar {1} $]')

    def test_get_wavelength(self):
        wavelength = ks.get_wavelength(200000)
        self.assertEqual(np.round(wavelength * 1000, 3), 2.508)
        wavelength = ks.get_wavelength(60000)
        self.assertEqual(np.round(wavelength * 1000, 3), 4.866)

        with self.assertRaises(TypeError):
            ks.get_wavelength('lattice_parameter')

    def test_get_rotation_matrix(self):
        matrix, theta, phi = ks.get_rotation_matrix([1, 0, 0])
        self.assertEqual(theta, 90.)
        self.assertEqual(phi, 0.)

        matrix, theta, phi = ks.get_rotation_matrix([1, 1, 1])
        matrix_desired = [[0.40824829, -0.70710678, 0.57735027],
                          [0.40824829, 0.70710678, 0.57735027],
                          [-0.81649658,  0., 0.57735027]]
        np.testing.assert_allclose(matrix, matrix_desired)

    def test_check_sanity(self):
        self.assertFalse(ks.check_sanity({}))

        tags = ks.example(verbose=False)
        self.assertTrue(ks.check_sanity(tags))

    def test_ring_pattern_calculation(self):
        tags = ks.example(verbose=False)
        new_tags = ks.ring_pattern_calculation(tags)
        self.assertTrue('{2 6 4}' in new_tags)
        self.assertAlmostEqual(new_tags['{1 1 1}']['reciprocal_distance'], 3.369748652857738)
        self.assertAlmostEqual(new_tags['{1 1 1}']['real_distance'], 1/3.369748652857738)
        self.assertAlmostEqual(new_tags['{1 1 1}']['F'], 17.53103039316424)
        self.assertEqual(new_tags['{1 1 1}']['multiplicity'], 8)

    def test_kinematic_scattering(self):
        tags = ks.example(verbose=False)
        ks.kinematic_scattering(tags)
        self.assertIsInstance(tags['HOLZ'], dict)
        self.assertAlmostEqual(tags['wave_length_nm'], 0.0036695, delta=1e-6)

    def test_plotSAED(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(tags)
        # ks.plotSAED(tags)

    def test_plotKikuchi(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(tags)
        # ks.plotKikuchi(tags)

    def test_plotHOLZ(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(tags)
        # ks.plotHOLZ(tags)

    def test_plotCBED(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(tags)
        # ks.plotCBED(tags)

    def test_circles(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(tags)
        # ks.circles(tags)

    def test_plot_diffraction_pattern(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(tags)
        # ks.plot_diffraction_pattern(tags)

    def test_diffraction_pattern(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(tags)
        # ks.diffraction_pattern(tags)

    def test_feq(self):
        self.assertAlmostEqual(ks.feq('Au', 3.6), 7.43164303450277)
        self.assertAlmostEqual(ks.feq('Si', 12.6), 0.5398190143297035)
