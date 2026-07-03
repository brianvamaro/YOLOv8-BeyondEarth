"""Characterization tests for the Test 5 spectral module — locking the FRAME CONVENTIONS.

The load-bearing subtleties are directional: power at wavevector azimuth theta means intensity
varies ALONG theta (structures along theta put power at theta+90; a blur along theta makes a power
DEFICIT at theta). These were validated empirically during Test 5; the tests here pin them so a
refactor cannot silently flip a sign/frame. All CPU, no data, fast.
"""
import numpy as np
import pytest

from YOLOv8BeyondEarth.regolith_spectrum import (
    angular_power_profile, ensemble_profiles, axis_contrast, profile_summary,
    directional_blur, isotropize, warp_rotate)


def _stripes(n=64, azimuth_deg=135.0, wavelength=4.0):
    """Sinusoid whose WAVEVECTOR azimuth (from North, cw) is ``azimuth_deg`` — i.e. intensity
    varies along that azimuth; the stripes themselves run along azimuth_deg + 90."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    th = np.radians(azimuth_deg)
    t = xx * np.sin(th) - yy * np.cos(th)                  # East*sin + North*cos, North = -row
    return 128.0 + 20.0 * np.sin(2 * np.pi * t / wavelength)


@pytest.mark.parametrize("az", [0.0, 45.0, 90.0, 135.0])
def test_wavevector_azimuth_convention(az):
    rng = np.random.default_rng(0)
    pats = np.array([_stripes(azimuth_deg=az) + rng.standard_normal((64, 64)) for _ in range(8)])
    c, profs = ensemble_profiles(pats)
    peak = float(c[np.nanargmax(np.nanmean(profs, 0))])
    d = abs(((peak - az + 90.0) % 180.0) - 90.0)           # axial distance
    assert d <= 10.0, f"stripe wavevector at {az} deg produced profile peak at {peak} deg"


def test_directional_blur_makes_deficit_at_blur_azimuth():
    rng = np.random.default_rng(1)
    pats = rng.standard_normal((30, 64, 64)) * 5.0 + 128.0
    blurred = np.array([directional_blur(p, sigma_px=1.0, angle_deg=135.0) for p in pats])
    c, profs = ensemble_profiles(blurred)
    mean = np.nanmean(profs, 0)
    # blur ALONG 135 suppresses wavevector power AT 135 and spares 45
    assert axis_contrast(c, mean, (135.0,)) < -0.1
    assert axis_contrast(c, mean, (45.0,)) > 0.1


def test_white_noise_is_flat():
    rng = np.random.default_rng(2)
    pats = rng.standard_normal((60, 64, 64)) * 5.0 + 128.0
    c, profs = ensemble_profiles(pats)
    mean = np.nanmean(profs, 0)
    for ax in (0.0, 45.0, 90.0, 135.0):
        assert abs(axis_contrast(c, mean, (ax,))) < 0.08


def test_isotropize_removes_directionality():
    rng = np.random.default_rng(3)
    pats = np.array([_stripes(azimuth_deg=135.0, wavelength=5.0)
                     + rng.standard_normal((64, 64)) * 2.0 for _ in range(12)])
    iso = np.array([isotropize(p, seed=i) for i, p in enumerate(pats)])
    c, p_orig = ensemble_profiles(pats)
    _, p_iso = ensemble_profiles(iso)
    c135_orig = abs(axis_contrast(c, np.nanmean(p_orig, 0), (135.0,)))
    c135_iso = abs(axis_contrast(c, np.nanmean(p_iso, 0), (135.0,)))
    assert c135_iso < c135_orig / 3.0
    assert iso.shape == pats.shape
    assert np.all(iso >= 0) and np.all(iso <= 255)         # quantized to uint8 range


def test_angular_power_profile_shape_and_normalisation():
    rng = np.random.default_rng(4)
    c, prof = angular_power_profile(rng.standard_normal((64, 64)), nbins=36)
    assert len(c) == 36 and len(prof) == 36
    assert np.isclose(np.nanmean(prof), 1.0, atol=1e-6)    # normalised to mean 1


def test_warp_rotate_center_crop():
    patch = np.random.default_rng(5).standard_normal((128, 128))
    out = warp_rotate(patch, 30.0, "bilinear")
    assert out.shape == (64, 64)


def test_profile_summary_keys():
    rng = np.random.default_rng(6)
    c, prof = angular_power_profile(rng.standard_normal((64, 64)) + 128)
    s = profile_summary(c, prof)
    for k in ("diag_contrast", "card_contrast", "peak_deg", "trough_deg", "modulation",
              "c0", "c45", "c90", "c135"):
        assert k in s
