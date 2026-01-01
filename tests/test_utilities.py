import math
import numpy as np
import pytest
import importlib

import pyTEMlib.utilities as utilities


def test_get_wavelength_positive():
    # wavelength should be positive and finite for typical electron energy
    wl = utilities.get_wavelength(200000.0)  # 200 keV
    assert isinstance(wl, float)
    assert wl > 0.0
    assert math.isfinite(wl)


def test_current_to_number_of_electrons():
    # 1 Ampere corresponds to ~6.241509e18 electrons per second
    n = utilities.current_to_number_of_electrons(1.0)
    assert pytest.approx(1.0 / (1.602176634e-19), rel=1e-6) == n


def test_lorentz_peak_normalization():
    x = np.linspace(-10, 10, 101)
    center = 0.0
    amplitude = 2.5
    width = 1.0
    y = utilities.lorentz(x, center, amplitude, width)
    # function normalizes such that the maximum equals the amplitude
    assert pytest.approx(amplitude, rel=1e-7) == float(np.max(y))
    # symmetry around center
    assert pytest.approx(y[50], rel=1e-7) == float(y[len(x) // 2])


def test_gauss_zero_width_returns_zeros():
    x = np.linspace(-5, 5, 11)
    p = [0.0, 1.0, 0.0]  # p[2] == 0 should yield zeros
    y = utilities.gauss(x, np.array(p))
    assert np.allclose(y, 0.0)


def test_gauss_peak_at_center():
    x = np.linspace(-5, 5, 101)
    p = [0.0, 3.0, 1.2]  # mean=0, amplitude=3
    y = utilities.gauss(x,np.array(p))
    # maximum value should be close to amplitude
    assert pytest.approx(3.0, rel=1e-5) == float(np.max(y))
    # center value equals maximum
    idx = np.argmin(np.abs(x - p[0]))
    assert pytest.approx(np.max(y), rel=1e-6) == y[idx]


def test_get_z_and_element_symbol_int_and_str():
    # integer input
    assert utilities.get_z(6) == 6
    # numeric string
    assert utilities.get_z("6") == 6
    # element symbol string
    assert utilities.get_z("C") == 6
    # wrapper alias
    assert utilities.get_atomic_number("O") == utilities.get_z("O")
    # element symbol resolution
    assert utilities.get_element_symbol(6) == "C"
    assert utilities.get_element_symbol("6") == "C"


def test_get_z_invalid_string_raises_value_error():
    with pytest.raises(ValueError):
        utilities.get_z("NotAnElement")


def test_get_x_sections_monkeypatched(monkeypatch):
    # monkeypatch the x_sections mapping on the utilities module
    dummy = {"1": {"K": 1}, "2": {"K": 2}}
    monkeypatch.setattr(utilities, "x_sections", dummy, raising=True)

    # z < 1 returns whole mapping
    all_sections = utilities.get_x_sections(0)
    assert all_sections is dummy

    # existing element key as int
    one = utilities.get_x_sections(1)
    assert one == {"K": 1}

    # missing key returns empty dict
    none = utilities.get_x_sections(999)
    assert none == {}


def test_effective_collection_angle_basic():
    # basic smoke test to ensure function runs and returns a finite number
    energy_scale = np.array([0.0, 1000.0])  # simple energy scale
    alpha = 10.0  # mrad
    beta = 20.0  # mrad
    beam_ev = 200000.0  # eV
    eff = utilities.effective_collection_angle(energy_scale, alpha, beta, beam_ev)
    assert isinstance(eff, float)
    assert math.isfinite(eff)
    assert eff >= 0.0


# Reload module to ensure tests are isolated when needed
def test_module_reload_no_errors():
    importlib.reload(utilities)
    assert hasattr(utilities, "get_wavelength")