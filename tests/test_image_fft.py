import numpy as np
import unittest
from pyTEMlib.image import image_fft

try:
    import sidpy  # type: ignore
except ImportError:
    sidpy = None  # noqa: E305
try:
    import skimage.feature  # noqa: F401
except ImportError:
    skimage_feature = None  # noqa: F401


def _make_simple_image_dataset():
    """Create a synthetic sidpy image dataset with sinusoidal components."""
    N = 64
    dx = 0.5  # nm
    x_vals = np.linspace(0, (N - 1) * dx, N)
    y_vals = np.linspace(0, (N - 1) * dx, N)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')
    img = np.sin(2 * np.pi * 3 * x_grid / (N * dx)) + np.sin(2 * np.pi * 5 * y_grid / (N * dx))
    dset = sidpy.Dataset.from_array(img)
    dset.data_type = 'IMAGE'
    dset.quantity = 'intensity'
    dset.units = 'a.u.'
    dset.title = 'synthetic image'
    dset.set_dimension(0, sidpy.Dimension(x_vals, name='x', units='nm', quantity='length',
                                          dimension_type='SPATIAL'))
    dset.set_dimension(1, sidpy.Dimension(y_vals, name='y', units='nm', quantity='length',
                                          dimension_type='SPATIAL'))
    return dset


def _make_synthetic_diffractogram():
    """Create a synthetic diffractogram sidpy dataset with bright spots."""
    N = 64
    u_vals = np.linspace(-5, 5, N)
    v_vals = np.linspace(-5, 5, N)
    data = np.zeros((N, N))
    spot_positions = [(-3, 0), (3, 0), (0, -4), (0, 4)]
    for (u0, v0) in spot_positions:
        ui = np.argmin(np.abs(u_vals - u0))
        vi = np.argmin(np.abs(v_vals - v0))
        data[ui, vi] = 10.0
    dset = sidpy.Dataset.from_array(data)
    dset.data_type = 'IMAGE'
    dset.title = 'synthetic diffractogram'
    dset.set_dimension(0, sidpy.Dimension(u_vals, name='u', units='1/nm', quantity='reciprocal_length',
                                          dimension_type='RECIPROCAL'))
    dset.set_dimension(1, sidpy.Dimension(v_vals, name='v', units='1/nm', quantity='reciprocal_length',
                                          dimension_type='RECIPROCAL'))
    return dset, spot_positions


@unittest.skipIf(sidpy is None, "sidpy not installed")
class TestImageFFT(unittest.TestCase):
    """Unittest conversion of pytest-based image FFT tests."""

    def setUp(self):  # runs before each test needing dataset
        self.simple_image = _make_simple_image_dataset()

    def test_fourier_transform_basic(self):
        fft_dset = image_fft.fourier_transform(self.simple_image)
        self.assertIsInstance(fft_dset, sidpy.Dataset)
        self.assertEqual(fft_dset.shape, self.simple_image.shape)
        self.assertEqual(fft_dset.data_type.name, 'IMAGE')
        dims = fft_dset.get_image_dims(return_axis=True)
        self.assertEqual(dims[0].name, 'u')
        self.assertEqual(dims[1].name, 'v')
        self.assertEqual(dims[0].units, '1/nm')
        self.assertEqual(dims[1].units, '1/nm')
        self.assertEqual(fft_dset.quantity, self.simple_image.quantity)
        self.assertEqual(fft_dset.units, 'a.u.')

    def test_power_spectrum_metadata(self):
        ps = image_fft.power_spectrum(self.simple_image, smoothing=2)
        self.assertIn('fft', ps.metadata)
        self.assertEqual(ps.metadata['fft']['smoothing'], 2)
        self.assertLess(ps.metadata['fft']['minimum_intensity'], ps.metadata['fft']['maximum_intensity'])
        self.assertTrue(ps.title.startswith('power spectrum'))

    def test_rotational_symmetry_diffractogram(self):
        spots = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ])
        sym = image_fft.rotational_symmetry_diffractogram(spots)
        self.assertIn(2, sym)
        self.assertIn(4, sym)
        self.assertNotIn(3, sym)
        self.assertNotIn(6, sym)

    def test_diffractogram_spots_detection(self):
        dset, expected = _make_synthetic_diffractogram()
        spots, center = image_fft.diffractogram_spots(dset, spot_threshold=0.2, return_center=True)
        self.assertEqual(spots.shape[1], 3)
        self.assertLess(np.linalg.norm(center), 1.0)
        detected_xy = spots[:, :2]
        for (u0, v0) in expected:
            dist = np.min(np.linalg.norm(detected_xy - np.array([u0, v0]), axis=1))
            self.assertLess(dist, (dset.u.values[1] - dset.u.values[0]) * 2.5)
        self.assertTrue(np.all(spots[:, 2] <= np.pi))
        self.assertTrue(np.all(spots[:, 2] >= -np.pi))

    def test_adaptive_fourier_filter_preserves_selected_frequencies(self):
        fft_dset = image_fft.fourier_transform(self.simple_image)
        mag = np.abs(np.array(fft_dset))
        center_idx = (mag.shape[0] // 2, mag.shape[1] // 2)
        mag[center_idx] = 0.0
        threshold = 0.4 * mag.max()
        peak_indices = np.argwhere(mag > threshold)
        u_vals = fft_dset.u.values
        v_vals = fft_dset.v.values
        spots_list = []
        for (i, j) in peak_indices:
            u = u_vals[i]
            v = v_vals[j]
            angle = np.arctan2(u, v)
            spots_list.append([u, v, angle])
        spots = np.array(spots_list)
        filtered = image_fft.adaptive_fourier_filter(self.simple_image, spots, low_pass=0.15, reflection_radius=0.25)
        self.assertEqual(filtered.shape, self.simple_image.shape)
        self.assertIn('analysis', filtered.metadata)
        self.assertEqual(filtered.metadata['analysis'], 'adaptive fourier filtered')
        x, y = np.meshgrid(fft_dset.v.values, fft_dset.u.values)
        mask = np.zeros(fft_dset.shape)
        for spot in spots:
            mask_spot = (x - spot[1]) ** 2 + (y - spot[0]) ** 2 < 0.25 ** 2
            mask += mask_spot
        mask += (x ** 2 + y ** 2 < 0.15 ** 2)
        mask[mask > 1] = 1
        original_masked_mag = np.abs(np.array(fft_dset)) * mask
        filtered_fft = image_fft.fourier_transform(filtered)
        filtered_mag = np.abs(np.array(filtered_fft))
        outside_ratio = filtered_mag[mask == 0].sum() / np.abs(np.array(fft_dset))[mask == 0].sum()
        self.assertLess(outside_ratio, 1.)
        inside_ratio = filtered_mag[mask == 1].sum() / original_masked_mag[mask == 1].sum()
        self.assertGreater(inside_ratio, 0.5)


if __name__ == '__main__':  # Allow running this file directly
    unittest.main()
