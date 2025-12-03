"""Unit tests for probe_tools module"""
import unittest
import numpy as np
from pyTEMlib import probe_tools
import pyTEMlib


class TestProbeShapes(unittest.TestCase):
    """Test basic probe shape generation functions."""

    def test_make_gauss_shape(self):
        """Test Gaussian probe generation."""
        probe = probe_tools.make_gauss(64, 64, width=2.0, intensity=100.0)
        self.assertEqual(probe.shape, (64, 64))
        self.assertAlmostEqual(probe.sum(), 100.0, places=1)
        # Peak should be at center
        max_idx = np.unravel_index(np.argmax(probe), probe.shape)
        self.assertAlmostEqual(max_idx[0], 32, delta=1)
        self.assertAlmostEqual(max_idx[1], 32, delta=1)

    def test_make_gauss_offset(self):
        """Test Gaussian probe with offset center."""
        probe = probe_tools.make_gauss(64, 64, width=2.0, x0=5, y0=-3)
        max_idx = np.unravel_index(np.argmax(probe), probe.shape)
        # Should be offset from center
        self.assertNotEqual(max_idx[0], 32)

    def test_make_lorentz_shape(self):
        """Test Lorentzian probe generation."""
        probe = probe_tools.make_lorentz(64, 64, gamma=1.0, intensity=100.0)
        self.assertEqual(probe.shape, (64, 64))
        self.assertAlmostEqual(probe.sum(), 100.0, places=1)

    def test_make_lorentz_offset(self):
        """Test Lorentzian probe with offset center."""
        probe = probe_tools.make_lorentz(64, 64, gamma=1.0, x0=3, y0=2)
        max_idx = np.unravel_index(np.argmax(probe), probe.shape)
        self.assertNotEqual(max_idx[0], 32)


class TestAberrations(unittest.TestCase):
    """Test aberration function calculations."""

    def setUp(self):
        """Set up basic aberration dictionary."""
        self.ab = {
            'C10': 0, 'C12a': 0, 'C12b': 0,
            'C21a': 0, 'C21b': 0, 'C23a': 0, 'C23b': 0,
            'C30': 0, 'C32a': 0, 'C32b': 0, 'C34a': 0, 'C34b': 0,
            'C41a': 0, 'C41b': 0, 'C43a': 0, 'C43b': 0, 'C45a': 0, 'C45b': 0,
            'C50': 0, 'C52a': 0, 'C52b': 0, 'C54a': 0, 'C54b': 0,
            'C56a': 0, 'C56b': 0,
            'acceleration_voltage': 200000,
            'convergence_angle': 30,
            'FOV': 10,
            'Cc': 1.3e6
        }

    def test_make_chi_zero_aberrations(self):
        """Test chi function with zero aberrations."""
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(0, 0.03, 100)
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        wavelength = pyTEMlib.image_tools.get_wavelength(200000)
        self.ab['wavelength'] = wavelength

        chi = probe_tools.make_chi(phi_grid, theta_grid, self.ab)
        # With zero aberrations, chi should be zero
        self.assertTrue(np.allclose(chi, 0))

    def test_make_chi_with_defocus(self):
        """Test chi function with defocus only."""
        self.ab['C10'] = 100  # 100 nm defocus
        phi = np.linspace(0, 2*np.pi, 50)
        theta = np.linspace(0, 0.03, 50)
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        wavelength = pyTEMlib.image_tools.get_wavelength(200000)
        self.ab['wavelength'] = wavelength

        chi = probe_tools.make_chi(phi_grid, theta_grid, self.ab)
        # Chi should be non-zero with defocus
        self.assertFalse(np.allclose(chi, 0))

    def test_get_chi_returns_tuple(self):
        """Test get_chi returns chi and aperture."""
        chi, aperture = probe_tools.get_chi(self.ab, 64, 64)
        self.assertEqual(chi.shape, (64, 64))
        self.assertEqual(aperture.shape, (64, 64))
        # Aperture should be 0 or 1
        self.assertTrue(np.all((aperture == 0) | (aperture == 1)))


class TestAberrationConversions(unittest.TestCase):
    """Test aberration format conversion functions."""

    def test_cart2pol(self):
        """Test cartesian to polar conversion."""
        x, y = 3.0, 4.0
        theta, rho = probe_tools.cart2pol(x, y)
        self.assertAlmostEqual(rho, 5.0)
        self.assertAlmostEqual(theta, np.arctan2(y, x))

    def test_ceos_to_nion_basic(self):
        """Test CEOS to Nion aberration conversion."""
        ceos_ab = {'C1': 10, 'C3': 1000, 'C5': 10000}
        nion_ab = probe_tools.ceos_to_nion(ceos_ab)
        self.assertIn('C10', nion_ab)
        self.assertIn('C30', nion_ab)
        self.assertIn('C50', nion_ab)
        self.assertEqual(nion_ab['C10'], 10)
        self.assertEqual(nion_ab['C30'], 1000)
        self.assertEqual(nion_ab['C50'], 10000)

    def test_nion_to_ceos_basic(self):
        """Test Nion to CEOS aberration conversion."""
        nion_ab = {'C10': 10, 'C30': 1000, 'C50': 10000}
        ceos_ab = probe_tools.nion_to_ceos(nion_ab)
        self.assertIn('C1', ceos_ab)
        self.assertIn('C3', ceos_ab)
        self.assertIn('C5', ceos_ab)


class TestProbeGeneration(unittest.TestCase):
    """Test electron probe generation."""

    def setUp(self):
        """Set up aberration parameters."""
        self.ab = {
            'C10': 0, 'C12a': 0, 'C12b': 0,
            'C21a': 0, 'C21b': 0, 'C23a': 0, 'C23b': 0,
            'C30': 0, 'C32a': 0, 'C32b': 0, 'C34a': 0, 'C34b': 0,
            'C41a': 0, 'C41b': 0, 'C43a': 0, 'C43b': 0, 'C45a': 0, 'C45b': 0,
            'C50': 0, 'C52a': 0, 'C52b': 0, 'C54a': 0, 'C54b': 0,
            'C56a': 0, 'C56b': 0,
            'acceleration_voltage': 200000,
            'convergence_angle': 30,
            'FOV': 10,
            'Cc': 1.3e6
        }

    def test_make_probe_shape(self):
        """Test probe generation returns correct shape."""
        chi = np.zeros((64, 64))
        aperture = np.ones((64, 64))
        probe = probe_tools.make_probe(chi, aperture)
        self.assertEqual(probe.shape, (64, 64))
        self.assertTrue(np.all(probe >= 0))  # Intensity should be positive

    def test_get_probe_returns_tuple(self):
        """Test get_probe returns probe, aperture, and chi."""
        probe, aperture, chi = probe_tools.get_probe(self.ab, 64, 64, verbose=False)
        self.assertEqual(probe.shape, (64, 64))
        self.assertEqual(aperture.shape, (64, 64))
        self.assertEqual(chi.shape, (64, 64))
        self.assertTrue(np.all(probe >= 0))


class TestTargetAberrations(unittest.TestCase):
    """Test retrieval of target aberrations for specific microscopes."""

    def test_get_target_aberrations_nion_us200_200kv(self):
        """Test getting NionUS200 target aberrations at 200kV."""
        ab = probe_tools.get_target_aberrations('NionUS200', 200000)
        self.assertIsNotNone(ab)
        self.assertEqual(ab['acceleration_voltage'], 200000)
        self.assertIn('C10', ab)
        self.assertIn('C30', ab)
        self.assertIn('tem_name', ab)
        self.assertEqual(ab['tem_name'], 'NionUS200')

    def test_get_target_aberrations_nion_us200_100kv(self):
        """Test getting NionUS200 target aberrations at 100kV."""
        ab = probe_tools.get_target_aberrations('NionUS200', 100000)
        self.assertEqual(ab['acceleration_voltage'], 100000)

    def test_get_target_aberrations_nion_us200_60kv(self):
        """Test getting NionUS200 target aberrations at 60kV."""
        ab = probe_tools.get_target_aberrations('NionUS200', 60000)
        self.assertEqual(ab['acceleration_voltage'], 60000)

    def test_get_target_aberrations_zeiss(self):
        """Test getting Zeiss MC200 target aberrations."""
        ab = probe_tools.get_target_aberrations('ZeissMC200', 200000)
        self.assertIsNotNone(ab)
        self.assertEqual(ab['tem_name'], 'ZeissMC200')


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_zero_loss_peak_weight(self):
        """Test zero loss peak weight function."""
        x, y = probe_tools.zero_loss_peak_weight()
        self.assertEqual(len(x), 29)
        self.assertEqual(len(y), 29)
        self.assertAlmostEqual(max(y), 1.0)

    def test_get_source_energy_spread(self):
        """Test source energy spread function."""
        x, y = probe_tools.get_source_energy_spread()
        self.assertEqual(len(x), 29)
        self.assertEqual(len(y), 29)


class TestChiDerivatives(unittest.TestCase):
    """Test chi second derivative calculations."""

    def setUp(self):
        """Set up aberrations with non-zero values."""
        self.ab = {
            'C10': 10, 'C12a': 5, 'C12b': 5,
            'C21a': 10, 'C21b': 10, 'C23a': 10, 'C23b': 10,
            'C30': 1000, 'C32a': 100, 'C32b': 100, 'C34a': 100, 'C34b': 100,
            'C41a': 1000, 'C41b': 1000, 'C43a': 1000, 'C43b': 1000,
            'C45a': 1000, 'C45b': 1000,
            'C50': 10000, 'C52a': 1000, 'C52b': 1000, 'C54a': 1000, 'C54b': 1000,
            'C56a': 1000, 'C56b': 1000,
            'wavelength': 0.00251
        }

    def test_get_d2chidu2_shape(self):
        """Test d2chi/du2 returns correct shape."""
        u = np.linspace(-1, 1, 50)
        v = np.linspace(-1, 1, 50)
        u_grid, v_grid = np.meshgrid(u, v)
        result = probe_tools.get_d2chidu2(self.ab, u_grid, v_grid)
        self.assertEqual(result.shape, (50, 50))

    def test_get_d2chidv2_shape(self):
        """Test d2chi/dv2 returns correct shape."""
        u = np.linspace(-1, 1, 50)
        v = np.linspace(-1, 1, 50)
        u_grid, v_grid = np.meshgrid(u, v)
        result = probe_tools.get_d2chidv2(self.ab, u_grid, v_grid)
        self.assertEqual(result.shape, (50, 50))

    def test_get_d2chidudv_shape(self):
        """Test d2chi/dudv returns correct shape."""
        u = np.linspace(-1, 1, 50)
        v = np.linspace(-1, 1, 50)
        u_grid, v_grid = np.meshgrid(u, v)
        result = probe_tools.get_d2chidudv(self.ab, u_grid, v_grid)
        self.assertEqual(result.shape, (50, 50))


class TestRonchigram(unittest.TestCase):
    """Test Ronchigram generation."""

    def setUp(self):
        """Set up aberrations for Ronchigram."""
        self.ab = {
            'C10': 0, 'C12a': 0, 'C12b': 0,
            'C21a': 0, 'C21b': 0, 'C23a': 0, 'C23b': 0,
            'C30': 1000, 'C32a': 0, 'C32b': 0, 'C34a': 0, 'C34b': 0,
            'C41a': 0, 'C41b': 0, 'C43a': 0, 'C43b': 0, 'C45a': 0, 'C45b': 0,
            'C50': 0, 'C52a': 0, 'C52b': 0, 'C54a': 0, 'C54b': 0,
            'C56a': 0, 'C56b': 0,
            'acceleration_voltage': 200000,
            'convergence_angle': 30,
            'FOV': 10,
            'Cc': 1.3e6
        }

    def test_get_ronchigram_shape(self):
        """Test Ronchigram generation returns correct shape."""
        ronchigram = probe_tools.get_ronchigram(64, self.ab, scale='mrad')
        self.assertEqual(ronchigram.shape, (64, 64))
        self.assertTrue(np.all(ronchigram >= 0))
        self.assertIn('chi', self.ab)
        self.assertIn('ronchi_extent', self.ab)

    def test_get_ronchigram_scale_1_per_nm(self):
        """Test Ronchigram with 1/nm scale."""
        _ = probe_tools.get_ronchigram(64, self.ab, scale='1/nm')
        self.assertEqual(self.ab['ronchi_label'], 'reciprocal distance [1/nm]')


if __name__ == '__main__':
    unittest.main()
