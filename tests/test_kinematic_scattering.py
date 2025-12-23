# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:07:16 2021

@author: Gerd Duscher
"""

import unittest
import matplotlib
matplotlib.use('Agg')
import numpy as np
import ase

import pyTEMlib

class TestUtilityFunctions(unittest.TestCase):
    """Unittest conversion of pytest-based utility function tests."""

    def test_zuo_fig_3_18(self):
        """Test loading of Zuo Fig. 3.18 example."""
        atoms = pyTEMlib.diffraction_tools.example(verbose=True)
        self.assertIsInstance(atoms.info, dict)
        self.assertEqual(atoms.symbols[0], 'Si')
        self.assertEqual(atoms.cell[0, 0], 5.14)
        self.assertEqual(atoms.info['experimental']['acceleration_voltage'], 99.2*1000.0)
        self.assertEqual(atoms.info['experimental']['convergence_angle_mrad'], 7.15)
        np.testing.assert_allclose(atoms.info['experimental']['zone_hkl'], np.array([-2, 2, 1]))

    def test_example(self):
        """Test loading of example structure."""
        atoms = pyTEMlib.diffraction_tools.example(verbose=False)
        self.assertEqual(atoms.info['output']['plot_HOLZ'], 1)

    def test_zone_mistilt(self):
        """Test zone axis rotation with mistilt angles."""
        rotated_zone_axis = pyTEMlib.diffraction_tools.zone_mistilt([1, 0, 0], [45, 0, 0])
        np.testing.assert_allclose(rotated_zone_axis, [1, 0, 0])

        rotated_zone_axis = pyTEMlib.diffraction_tools.zone_mistilt([1, 0, 0], [0, 10, 0])
        np.testing.assert_allclose(rotated_zone_axis, [0.98480775, 0., 0.17364818])

        with self.assertRaises(TypeError):
            pyTEMlib.diffraction_tools.zone_mistilt([1, 0, 0], [0,  0])

        with self.assertRaises(TypeError):
            pyTEMlib.diffraction_tools.zone_mistilt([1j, 0, 0], [0,  0])

    def test_metric_tensor(self):
        """Test metric tensor calculation."""
        # Todo: better testing
        np.testing.assert_allclose(pyTEMlib.diffraction_tools.get_metric_tensor(np.identity(3)),
                                   np.identity(3))

    def test_make_pretty_labels(self):
        """Test making pretty hkl labels."""
        labels = pyTEMlib.diffraction_tools.make_pretty_labels(np.array([[1, 0, 0], [1, 1, -1]]))
        self.assertEqual(labels[0], '[$\\bar {1},0,0} $]')
        self.assertEqual(labels[1], '[$\\bar {1},1,\\bar {1} $]')

    def test_get_wavelength(self):
        """Test electron wavelength calculation."""
        wavelength = pyTEMlib.diffraction_tools.get_wavelength(200000, unit='A')
        self.assertEqual(np.round(wavelength * 100, 3), 2.508)
        wavelength = pyTEMlib.diffraction_tools.get_wavelength(60000., unit='A')
        self.assertEqual(np.round(wavelength * 100, 3), 4.866)

        with self.assertRaises(TypeError):
            pyTEMlib.diffraction_tools.get_wavelength('lattice_parameter')

    def test_get_rotation_matrix(self):
        """Test zone axis rotation matrix calculation."""
        tags = {'zone_hkl': [1, 1, 1], 'mistilt_alpha': 0, 'mistilt_beta': 0,
                'reciprocal_unit_cell': np.identity(3)}

        matrix = pyTEMlib.diffraction_tools.get_zone_rotation(tags)
        matrix_desired = [[0.81649658,  0., 0.57735027],
                          [-0.40824829, 0.70710678, 0.5773502],
                          [-0.40824829, -0.70710678, 0.57735027]]
        np.testing.assert_allclose(matrix, matrix_desired, 1e-5)

    def test_check_sanity(self):
        """Test sanity check for ASE atoms object."""
        atoms = ase.Atoms()
        self.assertFalse(atoms)

        atoms = pyTEMlib.diffraction_tools.example(verbose=False)
        self.assertTrue(pyTEMlib.diffraction_tools.check_sanity(atoms))


class TestScatteringFunctions(unittest.TestCase):
    """Unittest conversion of pytest-based scattering function tests."""

    def test_ring_pattern_calculation(self):
        """Test ring pattern calculation."""
        atoms = pyTEMlib.diffraction_tools.example(verbose=False)
        pyTEMlib.diffraction_tools.ring_pattern_calculation(atoms, verbose=True)

        self.assertAlmostEqual(atoms.info['Ring_Pattern']['allowed']['hkl'][7].sum(), 8.)
        self.assertAlmostEqual(atoms.info['Ring_Pattern']['allowed']['g norm'][0], 0.33697)
        self.assertAlmostEqual(atoms.info['Ring_Pattern']['allowed']['structure_factor'][0].real,
                               12.396310472193898)
        self.assertEqual(atoms.info['Ring_Pattern']['allowed']['multiplicity'][0], 8)

    def test_kinematic_scattering(self):
        """Test kinematic scattering calculation."""
        atoms = pyTEMlib.diffraction_tools.example(verbose=False)
        diff_dict = pyTEMlib.diffraction_tools.get_bragg_reflections(atoms, atoms.info['experimental'], verbose=True)
        self.assertIsInstance(diff_dict['allowed'], dict)
        self.assertAlmostEqual(diff_dict['K_0'],
                               26.90502565,
                               delta=1e-4)

    def test_feq(self):
        """Test atomic scattering factor calculation."""
        self.assertAlmostEqual(pyTEMlib.diffraction_tools.form_factor('Au', 0.36), 
                               7.43164303450277)
        self.assertAlmostEqual(pyTEMlib.diffraction_tools.form_factor('Si', 1.26), 
                               0.5398190143297035)


class TestScatteringFunctions2(unittest.TestCase):
    """Unittest conversion of pytest-based scattering function tests."""

    def test_ring_pattern_plot(self):
        """Test ring pattern plotting function."""
        atoms = pyTEMlib.diffraction_tools.example(verbose=False)
        pyTEMlib.diffraction_tools.ring_pattern_calculation(atoms, verbose=False)
        figure = pyTEMlib.diffraction_tools.plot_ring_pattern(atoms)
        self.assertIsNotNone(figure)
        self.assertIsInstance(figure, matplotlib.figure.Figure)


    def test_plotSAED(self):
        atoms = pyTEMlib.diffraction_tools.example(verbose=False)

        diff_dict = pyTEMlib.diffraction_tools.get_bragg_reflections(atoms, atoms.info['experimental'], verbose=False)
        diff_dict.update(pyTEMlib.diffraction_tools.plot_saed_parameter())

        figure = pyTEMlib.diffraction_tools.plot_diffraction_pattern(diff_dict)
        self.assertIsInstance(figure, matplotlib.figure.Figure)
    
    def test_plotKikuchi(self):
        """Kikuchi plotting parameter integration with diffraction pattern."""
        atoms = pyTEMlib.diffraction_tools.example(verbose=False)
        diff_dict = pyTEMlib.diffraction_tools.get_bragg_reflections(atoms, atoms.info['experimental'], verbose=False)
        kikuchi_tags = pyTEMlib.diffraction_tools.plot_kikuchi()
        diff_dict.update(kikuchi_tags)
        fig = pyTEMlib.diffraction_tools.plot_diffraction_pattern(diff_dict)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_plot_diffraction_pattern(self):
        """Spot pattern plotting returns a matplotlib figure."""
        atoms = pyTEMlib.diffraction_tools.example(verbose=False)
        diff_dict = pyTEMlib.diffraction_tools.get_bragg_reflections(atoms, atoms.info['experimental'], verbose=False)
        saed_tags = pyTEMlib.diffraction_tools.plot_saed_parameter()
        diff_dict.update(saed_tags)
        fig = pyTEMlib.diffraction_tools.plot_diffraction_pattern(diff_dict)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_plotHOLZ_parameter_only(self):
        """HOLZ parameter function returns a dict of tags."""
        holz_tags = pyTEMlib.diffraction_tools.plot_holz_parameter()
        self.assertIsInstance(holz_tags, dict)
        self.assertIn('plot_HOLZ', holz_tags)

    def test_plotCBED_parameter_only(self):
        """CBED parameter function returns a dict of tags."""
        cbed_tags = pyTEMlib.diffraction_tools.plot_cbed_parameter()
        self.assertIsInstance(cbed_tags, dict)
        self.assertIn('plot_Kikuchi', cbed_tags)
    

if __name__ == '__main__':
    unittest.main()
