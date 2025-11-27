import numpy as np
import pytest
from pyTEMlib.image import image_fft

sidpy = pytest.importorskip("sidpy")
skimage_feature = pytest.importorskip("skimage.feature")


@pytest.fixture
def simple_image_dataset():
    # Create a synthetic image with two orthogonal sinusoidal components
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
    dset.set_dimension(
        0,
        sidpy.Dimension(
            x_vals,
            name='x',
            units='nm',
            quantity='length',
            dimension_type='SPATIAL'
        )
    )
    dset.set_dimension(
        1,
        sidpy.Dimension(
            y_vals,
            name='y',
            units='nm',
            quantity='length',
            dimension_type='SPATIAL'
        )
    )
    return dset


def test_fourier_transform_basic(simple_image_dataset):
    fft_dset = image_fft.fourier_transform(simple_image_dataset)
    assert isinstance(fft_dset, sidpy.Dataset)
    assert fft_dset.shape == simple_image_dataset.shape
    assert fft_dset.data_type == 'IMAGE'
    assert fft_dset.dim_0.name == 'u'
    assert fft_dset.dim_1.name == 'v'
    assert fft_dset.dim_0.units == '1/nm'
    assert fft_dset.dim_1.units == '1/nm'
    assert fft_dset.quantity == simple_image_dataset.quantity
    assert fft_dset.units == 'a.u.'


def test_power_spectrum_metadata(simple_image_dataset):
    ps = image_fft.power_spectrum(simple_image_dataset, smoothing=2)
    assert 'fft' in ps.metadata
    assert ps.metadata['fft']['smoothing'] == 2
    assert ps.metadata['fft']['minimum_intensity'] < ps.metadata['fft']['maximum_intensity']
    assert ps.title.startswith('power spectrum')


def test_rotational_symmetry_diffractogram():
    # Four-fold symmetric spot set (cross)
    spots = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    sym = image_fft.rotational_symmetry_diffractogram(spots)
    assert 2 in sym
    assert 4 in sym
    assert 3 not in sym
    assert 6 not in sym


@pytest.fixture
def synthetic_diffractogram():
    # Create a fake diffractogram with bright spots
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
    dset.set_dimension(
        0,
        sidpy.Dimension(
            u_vals,
            name='u',
            units='1/nm',
            quantity='reciprocal_length',
            dimension_type='RECIPROCAL'
        )
    )
    dset.set_dimension(
        1,
        sidpy.Dimension(
            v_vals,
            name='v',
            units='1/nm',
            quantity='reciprocal_length',
            dimension_type='RECIPROCAL'
        )
    )
    return dset, spot_positions


def test_diffractogram_spots_detection(synthetic_diffractogram):
    dset, expected = synthetic_diffractogram
    spots, center = image_fft.diffractogram_spots(dset, spot_threshold=0.2, return_center=True)
    assert spots.shape[1] == 3
    assert np.linalg.norm(center) < 1.0
    # Verify each expected spot has a detected spot nearby
    detected_xy = spots[:, :2]
    for (u0, v0) in expected:
        dist = np.min(np.linalg.norm(detected_xy - np.array([u0, v0]), axis=1))
        assert dist < (dset.u.values[1] - dset.u.values[0]) * 2.5  # within a few pixels
    # Angles in valid range
    assert np.all(spots[:, 2] <= np.pi) and np.all(spots[:, 2] >= -np.pi)


def test_adaptive_fourier_filter_preserves_selected_frequencies(simple_image_dataset):
    # Obtain FFT and find strong peaks (excluding center)
    fft_dset = image_fft.fourier_transform(simple_image_dataset)
    mag = np.abs(np.array(fft_dset))
    center_idx = (mag.shape[0] // 2, mag.shape[1] // 2)
    mag[center_idx] = 0.0
    threshold = 0.4 * mag.max()
    peak_indices = np.argwhere(mag > threshold)
    # Build spots array (u, v, angle)
    u_vals = fft_dset.u.values
    v_vals = fft_dset.v.values
    spots_list = []
    for (i, j) in peak_indices:
        u = u_vals[i]
        v = v_vals[j]
        angle = np.arctan2(u, v)
        spots_list.append([u, v, angle])
    spots = np.array(spots_list)
    # Apply adaptive filter
    filtered = image_fft.adaptive_fourier_filter(simple_image_dataset, spots, low_pass=0.15, reflection_radius=0.25)
    assert filtered.shape == simple_image_dataset.shape
    assert 'analysis' in filtered.metadata
    assert filtered.metadata['analysis'] == 'adaptive fourier filtered'
    # Rebuild mask (same logic as in code) to compare spectra
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
    # Energy outside mask should be strongly reduced
    outside_ratio = filtered_mag[mask == 0].sum() / np.abs(np.array(fft_dset))[mask == 0].sum()
    assert outside_ratio < 0.2
    # Energy inside mask should remain
    inside_ratio = filtered_mag[mask == 1].sum() / original_masked_mag[mask == 1].sum()
    assert inside_ratio > 0.5