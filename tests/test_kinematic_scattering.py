# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:07:16 2021

@author: Gerd Duscher
"""

import unittest
import numpy as np
import ase

import sys
sys.path.append("../pyTEMlib/")
import pyTEMlib.kinematic_scattering as ks
import pyTEMlib.crystal_tools as cs


class TestUtilityFunctions(unittest.TestCase):

    def test_Zuo_fig_3_18(self):
        atoms = ks.Zuo_fig_3_18(verbose=True)
        self.assertIsInstance(atoms.info, dict)
        self.assertEqual(atoms.symbols[0], 'Si')
        self.assertEqual(atoms.cell[0, 0], 5.14)
        self.assertEqual(atoms.info['experimental']['acceleration_voltage_V'], 99.2*1000.0)
        self.assertEqual(atoms.info['experimental']['convergence_angle_mrad'], 7.15)
        np.testing.assert_allclose(atoms.info['experimental']['zone_hkl'], np.array([-2, 2, 1]))

    def test_example(self):
        atoms = ks.example(verbose=False)
        self.assertEqual(atoms.info['output']['plot_HOLZ'], 1)

    def test_zone_mistilt(self):
        rotated_zone_axis = ks.zone_mistilt([1, 0, 0], [45, 0, 0])
        np.testing.assert_allclose(rotated_zone_axis, [1, 0, 0])

        rotated_zone_axis = ks.zone_mistilt([1, 0, 0], [0, 10, 0])
        np.testing.assert_allclose(rotated_zone_axis, [0.98480775, 0., 0.17364818])

        with self.assertRaises(TypeError):
            ks.zone_mistilt([1, 0, 0], [0,  0])

        with self.assertRaises(TypeError):
            ks.zone_mistilt([1j, 0, 0], [0,  0])

    def test_metric_tensor(self):
        # Todo: better testing
        np.testing.assert_allclose(ks.get_metric_tensor(np.identity(3)), np.identity(3))

    def test_make_pretty_labels(self):
        labels = ks.make_pretty_labels(np.array([[1, 0, 0], [1, 1, -1]]))
        self.assertEqual(labels[0], '[$\\bar {1},0,0} $]')
        self.assertEqual(labels[1], '[$\\bar {1},1,\\bar {1} $]')

    def test_get_wavelength(self):
        wavelength = ks.get_wavelength(200000)
        self.assertEqual(np.round(wavelength * 100, 3), 2.508)
        wavelength = ks.get_wavelength(60000.)
        self.assertEqual(np.round(wavelength * 100, 3), 4.866)

        with self.assertRaises(TypeError):
            ks.get_wavelength('lattice_parameter')

    def test_get_rotation_matrix(self):
        tags = {'zone_hkl': [1, 1, 1], 'mistilt_alpha': 0, 'mistilt_beta': 0, 'reciprocal_unit_cell': np.identity(3)}

        matrix = ks.get_zone_rotation(tags)
        matrix_desired = [[0.81649658,  0., 0.57735027],
                          [-0.40824829, 0.70710678, 0.5773502],
                          [-0.40824829, -0.70710678, 0.57735027]]
        np.testing.assert_allclose(matrix, matrix_desired, 1e-5)

    def test_check_sanity(self):
        atoms = ase.Atoms()
        self.assertFalse(atoms)

        atoms = ks.example(verbose=False)
        self.assertTrue(ks.check_sanity(atoms))


class TestScatteringFunctions(unittest.TestCase):

    def test_ring_pattern_calculation(self):
        atoms = ks.example(verbose=False)
        ks.ring_pattern_calculation(atoms, verbose=True)

        self.assertAlmostEqual(atoms.info['Ring_Pattern']['allowed']['hkl'][7][0], 4.)
        self.assertAlmostEqual(atoms.info['Ring_Pattern']['allowed']['g norm'][0], 0.33697)
        self.assertAlmostEqual(atoms.info['Ring_Pattern']['allowed']['structure factor'][0].real, 12.396310472193898)
        self.assertEqual(atoms.info['Ring_Pattern']['allowed']['multiplicity'][0], 8)

    def test_kinematic_scattering(self):
        atoms = ks.example(verbose=False)
        ks.kinematic_scattering(atoms, verbose=True)
        self.assertIsInstance(atoms.info['diffraction']['HOLZ'], dict)
        self.assertAlmostEqual(atoms.info['experimental']['wave_length'], 0.03717657397994318, delta=1e-6)

    def test_feq(self):
        self.assertAlmostEqual(ks.feq('Au', 0.36), 7.43164303450277)
        self.assertAlmostEqual(ks.feq('Si', 1.26), 0.5398190143297035)


class TestScatteringFunctions2(unittest.TestCase):

    def test_ring_pattern_plot(self):
        pass

    """
    def test_plotSAED(self):
        atoms, tags, output = ks.example(verbose=False)

        ks.kinematic_scattering(atoms, tags, output)
        ks.plotSAED(tags)
    
    def test_plotKikuchi(self):
        atoms, tags, output = ks.example(verbose=False)

        ks.kinematic_scattering(atoms, tags, output)
        # ks.plotKikuchi(tags)

    def test_plotHOLZ(self):
        tags = ks.example(verbose=False)

        ks.kinematic_scattering(atoms, tags, output)
        # ks.plotHOLZ(tags)

    def test_plotCBED(self):
        atoms, tags, output = ks.example(verbose=False)

        ks.kinematic_scattering(atoms, tags, output)
        # ks.plotCBED(tags)

    def test_circles(self):
        atoms, tags, output = ks.example(verbose=False)

        ks.kinematic_scattering(v)
        # ks.circles(tags)

    def test_plot_diffraction_pattern(self):
        atoms, tags, output = ks.example(verbose=False)

        ks.kinematic_scattering(atoms, tags, output)
        # ks.plot_diffraction_pattern(tags)

    def test_diffraction_pattern(self):
        atoms, tags, output = ks.example(verbose=False)

        ks.kinematic_scattering(atoms, tags, output)
        # ks.diffraction_pattern(tags)
    """

if __name__ == '__main__':
    unittest.main()