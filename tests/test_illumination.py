"""Tests for the illumination-artifact exploration Component A
(``docs/illumination_artifact_exploration_plan.md``):

- the sun-relative orientation metrics ``signed_axial_deviation`` and ``half_plane_fraction_about``
  (generalisations of ``axial_distance`` / ``half_plane_fraction``);
- the incidence-driven cast shadow added to ``render_boulder_field`` (length grows with incidence;
  the rendered *truth* angle is the bright cap's own azimuth, independent of the shadow; the legacy
  ``inc_deg=None`` path is untouched);
- the rock-tracing proxy ``rock_trace_orientations`` (recovers the rock's true angle whether or not
  a long cast shadow is present — i.e. it genuinely excludes the shadow).
"""
import numpy as np

from YOLOv8BeyondEarth.orientation import (
    axial_distance,
    signed_axial_deviation,
    half_plane_fraction,
    half_plane_fraction_about,
    axis_alignment_fraction,
)
from YOLOv8BeyondEarth.transforms_test import render_boulder_field
from YOLOv8BeyondEarth.orientation_validation import rock_trace_orientations


# --- signed_axial_deviation --------------------------------------------------------------------

def test_signed_axial_deviation_sign_and_wrap():
    # positive = clockwise of ref (larger azimuth, short way round); negative = counter-clockwise
    assert signed_axial_deviation([10.0], 0.0)[0] == 10.0
    assert signed_axial_deviation([170.0], 0.0)[0] == -10.0        # 170 is 10 deg CCW of 0 axially
    assert signed_axial_deviation([100.0], 90.0)[0] == 10.0
    assert signed_axial_deviation([80.0], 90.0)[0] == -10.0


def test_signed_axial_deviation_matches_axial_distance_magnitude():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 180, 5000)
    for c in (0.0, 37.0, 90.0, 155.0):
        assert np.allclose(np.abs(signed_axial_deviation(a, c)), axial_distance(a, c), atol=1e-9)


def test_signed_axial_deviation_propagates_nan():
    out = signed_axial_deviation([10.0, np.nan, 100.0], 0.0)
    assert out[0] == 10.0 and np.isnan(out[1])


def test_signed_axial_deviation_array_ref_broadcasts():
    # ref may be a per-element array (e.g. each boulder's own sun axis)
    ang = np.array([10.0, 100.0, 50.0])
    ref = np.array([0.0, 90.0, 60.0])
    assert np.allclose(signed_axial_deviation(ang, ref), [10.0, 10.0, -10.0])


# --- half_plane_fraction_about -----------------------------------------------------------------

def test_half_plane_about_reduces_to_fixed_at_ref_zero():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 180, 20000)
    assert abs(half_plane_fraction_about(a, 0.0) - half_plane_fraction(a)) < 1e-12


def test_half_plane_about_symmetric_null_and_edges():
    ref = 40.0
    # a set reflection-symmetric about ref -> the two halves split 50/50 (null 0.5)
    d = np.array([5.0, 15.0, 35.0, 60.0, 80.0])
    sym = np.concatenate([(ref + d) % 180.0, (ref - d) % 180.0])
    assert abs(half_plane_fraction_about(sym, ref) - 0.5) < 1e-9
    # all mass one side of ref -> 1.0; all the other side -> 0.0
    assert half_plane_fraction_about(np.full(100, (ref + 45.0) % 180.0), ref) == 1.0
    assert half_plane_fraction_about(np.full(100, (ref - 45.0) % 180.0), ref) == 0.0


def test_half_plane_about_drops_nan_and_empty():
    assert np.isnan(half_plane_fraction_about(np.array([]), 30.0))
    assert np.isnan(half_plane_fraction_about(np.array([np.nan, np.nan]), 30.0))


def test_axis_alignment_fraction_null_and_concentration():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 180, 200_000)
    # uniform: half of axes fall within 45 deg of ANY reference axis (the "along" half); null 0.5
    assert abs(axis_alignment_fraction(a, 37.0, 45.0) - 0.5) < 0.01
    # tighter tol -> grid_fraction-style share about an arbitrary axis; null tol/90
    assert abs(axis_alignment_fraction(a, 37.0, 10.0) - 10.0 / 90.0) < 0.01
    # all axes exactly on ref -> 1.0; all perpendicular to ref -> 0.0
    assert axis_alignment_fraction(np.full(100, 37.0), 37.0, 45.0) == 1.0
    assert axis_alignment_fraction(np.full(100, 37.0 + 90.0), 37.0, 45.0) == 0.0
    assert np.isnan(axis_alignment_fraction(np.array([]), 0.0))


# --- incidence-driven cast shadow --------------------------------------------------------------

def _fixed_field(inc_deg, seed=3):
    # a fixed set of boulders (same seed => identical placement/size) rendered at one incidence
    angles = np.linspace(5.0, 175.0, 30)
    return render_boulder_field(size=512, angles=angles, diam_px=(10, 14), aspect=(1.6, 2.6),
                                bg=110, noise=6.0, cap=80.0, shadow=-55.0, sun_az_deg=90.0,
                                inc_deg=inc_deg, height_frac=0.6, seed=seed)


def test_incidence_shadow_length_grows_and_truth_is_shadow_invariant():
    img_lo, t_lo = _fixed_field(20.0)
    img_hi, t_hi = _fixed_field(70.0)
    # shadow value ~ bg+shadow = 55; count clearly-shadowed pixels (well below bg-3*noise)
    dark_lo = int((img_lo < 80).sum())
    dark_hi = int((img_hi < 80).sum())
    assert dark_hi > 1.5 * dark_lo                     # longer shadow at higher incidence
    # the rendered truth angle is the cap ellipse's own azimuth -> unchanged by the shadow
    assert np.allclose(t_lo["angle180"].to_numpy(), t_hi["angle180"].to_numpy())


def test_near_vertical_sun_has_negligible_shadow():
    img_hi, _ = _fixed_field(70.0)
    img_v, _ = _fixed_field(3.0)                        # sun ~overhead -> L -> 0, guarded off
    assert int((img_v < 80).sum()) < int((img_hi < 80).sum())


def test_legacy_shadow_path_unchanged_by_new_kwargs():
    # inc_deg=None keeps the original fixed-offset shadow and ignores the incidence-only kwargs
    kw = dict(size=256, angles=np.linspace(0, 170, 12), diam_px=(10, 14), aspect=(1.6, 2.6),
              noise=6.0, sun_az_deg=90.0, seed=7)
    img_a, _ = render_boulder_field(inc_deg=None, height_frac=0.5, shadow_width_frac=0.9, **kw)
    img_b, _ = render_boulder_field(inc_deg=None, height_frac=9.9, shadow_width_frac=0.1, **kw)
    assert np.array_equal(img_a, img_b)                 # incidence-only kwargs are inert in legacy
    # and the legacy shadow is actually present (shadow_frac default 0.6 draws dark pixels)
    assert int((img_a < 80).sum()) > 0


# --- rock-tracing proxy (the human-prior control) ----------------------------------------------

def _proxy_error(inc_deg):
    # one true angle so the recovered-vs-truth error is a clean scalar distribution
    angles = np.full(24, 35.0)
    img, truth = render_boulder_field(size=512, angles=angles, diam_px=(11, 15), aspect=(1.8, 2.8),
                                      bg=110, noise=5.0, cap=80.0, shadow=-55.0, sun_az_deg=90.0,
                                      inc_deg=inc_deg, height_frac=0.6, min_sep_frac=2.2, seed=5)
    proxy = rock_trace_orientations(img, truth, bg=110, cap=80.0)
    return np.nanmedian(axial_distance(proxy["angle180"].to_numpy(), 35.0))


def test_rock_proxy_recovers_truth_and_ignores_shadow():
    # the proxy thresholds the bright cap only, so it recovers the rock angle whether the shadow is
    # short (low incidence) or long (high incidence) -- that shadow-invariance is what makes it the
    # rock-only "human prior" control.
    assert _proxy_error(20.0) < 8.0
    assert _proxy_error(70.0) < 8.0
