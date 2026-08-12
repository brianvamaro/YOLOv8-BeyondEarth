"""The Goal-4 fabric gate (review item G4-07).

The two tests that matter are `test_the_gate_is_direction_aware` and
`test_fdr_stops_the_gate_manufacturing_fabric` — those are the defects the gate exists to fix, and
a gate that passes everything else while failing those is worthless.
"""
import numpy as np
import pytest

from YOLOv8BeyondEarth import fabric_gate as fg


def _rose(peak_deg, kappa, n, seed=0):
    """n axial angles concentrated on `peak_deg`. kappa=0 is uniform."""
    rng = np.random.default_rng(seed)
    if kappa == 0:
        return rng.uniform(0, 180, n)
    doubled = rng.vonmises(np.deg2rad(peak_deg) * 2, kappa, n)
    return np.rad2deg(doubled) / 2 % 180


def _tiles(n, k, seed=0):
    return np.random.default_rng(seed).integers(0, k, n)


#: kappa 0.63 gives R ~ 0.30 in doubled-angle space, matching the measured control-site median
#: R = 0.2965 (range 0.196-0.443). Using a realistically strong background matters: with a weak one
#: every candidate saturates the p-value floor and the direction test cannot see anything.
BG_KAPPA = 0.63


def _cardinal_background(n, seed):
    """Fabric-free ground as measured: not isotropic, leaning cardinal (grid 0.255-0.344)."""
    return _rose(0.0, BG_KAPPA, n, seed)


# --- axial machinery -------------------------------------------------------------------------

def test_uniform_has_no_resultant_and_a_tight_rose_has_one():
    assert fg.resultant_length(_rose(0, 0, 20000)) < 0.03
    assert fg.resultant_length(_rose(30, 50, 2000)) > 0.9


@pytest.mark.parametrize("peak,expect", [(0.0, "cardinal"), (90.0, "cardinal"),
                                         (45.0, "diagonal"), (135.0, "diagonal")])
def test_cardinal_and_diagonal_land_on_the_right_component(peak, expect):
    card, diag = fg.cardinal_diagonal_split(fg.axial_resultant(_rose(peak, 60, 4000)))
    assert (card > diag) if expect == "cardinal" else (diag > card)


def test_peak_recovery_is_the_doubled_angle_mean_not_a_smoothed_mode():
    """The review found `refine_peak`'s bandwidth drove a headline; this uses the stable estimator."""
    rows = fg.fabric_gate([("c", _rose(112.0, 40, 3000), _tiles(3000, 40))],
                          fg.build_control_null(_rose(0, 0, 5000), _tiles(5000, 40),
                                                cut={"a": 1}, n_tiles=40, n_boot=200),
                          cut={"a": 1}, min_n_eff=0)
    assert abs(rows[0]["peak_deg"] - 112.0) < 3.0


# --- defect 1: direction-awareness ------------------------------------------------------------

def test_the_gate_is_direction_aware():
    """Same R, different direction, against a cardinal-leaning background.

    A diagonal fabric must be easier to establish than a cardinal one of identical strength,
    because the artifact can manufacture the cardinal case and cannot manufacture the diagonal.
    """
    bg = _cardinal_background(6000, 1)
    # Candidate strength only modestly above the background's own lean: that is the regime where
    # direction decides the answer. Far above it, both directions are significant and the test
    # sees nothing; far below, neither is.
    # R ~ 0.33 against a background of ~0.30, at n = 600 where one sampling sd is ~0.029. So the
    # cardinal candidate is about 1 sigma above the background it is confusable with, and the
    # diagonal one is many sigma away from the background's diagonal content (~0).
    kappa, n = 0.70, 600
    cardinal = _rose(0.0, kappa, n, 7)
    diagonal = _rose(45.0, kappa, n, 7)
    r_c, r_d = fg.resultant_length(cardinal), fg.resultant_length(diagonal)
    assert abs(r_c - r_d) < 0.06, "test needs comparable R to isolate direction"

    null = fg.build_control_null(bg, _tiles(6000, 60, 1), cut={"a": 1.35}, n_tiles=30,
                                 n_target=n, n_boot=1500)
    p_c = fg.directional_pvalue(fg.axial_resultant(cardinal), null.draws)
    p_d = fg.directional_pvalue(fg.axial_resultant(diagonal), null.draws)
    assert p_d < p_c, (
        f"diagonal fabric (p={p_d}) must be easier to establish than cardinal (p={p_c}) "
        f"at equal R (R={r_c:.3f} vs {r_d:.3f}) — this is the whole point of the gate")
    assert p_c > 0.01, (
        f"cardinal p={p_c} is already at the floor, so this run does not actually demonstrate "
        "direction-awareness — the candidate is too strong relative to the background")


def test_a_scalar_R_bar_would_not_have_separated_them():
    """Negative control: the OLD gate form scores the two identically, by construction."""
    cardinal, diagonal = _rose(0.0, 0.75, 1200, 7), _rose(45.0, 0.75, 1200, 7)
    old_bar = 0.5152                                   # R_control_max_upper95, the shipped string
    assert (fg.resultant_length(cardinal) > old_bar) == (fg.resultant_length(diagonal) > old_bar)


# --- defect 2: multiplicity --------------------------------------------------------------------

def test_fdr_stops_the_gate_manufacturing_fabric():
    """1000 fabric-free cells. Uncorrected, ~alpha of them 'have fabric'; with BH, ~none do."""
    # A ControlPool, not a pre-built null: the null must be matched to each cell on n as well as
    # tile count. Passing a fixed null built from denser control ground is what made the first
    # version of this test pass ~19% of fabric-free cells even after BH.
    pool = fg.ControlPool(angles_deg=_cardinal_background(8000, 2),
                          tile_ids=_tiles(8000, 80, 2), cut={"a": 1.35})
    rng = np.random.default_rng(3)
    cells = [(f"c{i}", _cardinal_background(700, 100 + i), rng.integers(0, 25, 700))
             for i in range(1000)]

    uncorrected = fg.fabric_gate(cells, pool, cut={"a": 1.35}, alpha=0.05, fdr=False,
                                 design_effect=1.0)
    corrected = fg.fabric_gate(cells, pool, cut={"a": 1.35}, alpha=0.05, fdr=True,
                               design_effect=1.0)
    n_unc = sum(r["claimable"] for r in uncorrected)
    n_cor = sum(r["claimable"] for r in corrected)
    assert n_unc > 10, f"expected the uncorrected gate to manufacture fabric, got {n_unc}"
    assert n_cor <= max(1, n_unc // 10), f"BH should suppress nearly all of them, got {n_cor}"


def test_benjamini_hochberg_matches_a_hand_worked_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # Thresholds i*alpha/n are 0.00625, 0.0125, 0.01875, 0.025, ... so 0.001 and 0.008 pass and
    # 0.039 does not; the largest passing rank is 2, so BH rejects the first two. (Worked by hand
    # because the first version of this test asserted four and the implementation was right.)
    assert list(fg.benjamini_hochberg(p, 0.05)) == [True] * 2 + [False] * 6


def test_bh_rejects_the_whole_prefix_up_to_the_largest_passing_rank():
    """Step-UP, not step-down: a later rank passing pulls in every earlier one."""
    p = np.array([0.04, 0.001, 0.03, 0.02])       # unsorted on purpose
    assert list(fg.benjamini_hochberg(p, 0.20)) == [True, True, True, True]


def test_bh_is_never_more_conservative_than_bonferroni():
    p = np.random.default_rng(0).uniform(0, 1, 200)
    assert fg.benjamini_hochberg(p, 0.05).sum() >= (p <= 0.05 / 200).sum()


# --- defect 3: minimum effective n --------------------------------------------------------------

def test_a_tiny_cell_is_refused_however_strong_its_rose_looks():
    """The retracted R ~ 0.35 bar came from 8-22 boulder sites; R is huge there for free."""
    null = fg.build_control_null(_cardinal_background(4000, 4), _tiles(4000, 40, 4),
                                 cut={"a": 1.35}, n_tiles=10, n_boot=400)
    tiny = _rose(45.0, 5.0, 9, 11)
    assert fg.resultant_length(tiny) > 0.6            # looks spectacular
    row = fg.fabric_gate([("tiny", tiny, np.arange(9))], null, cut={"a": 1.35})[0]
    assert not row["claimable"] and "n_eff" in row["reason"]


def test_underpowered_cells_do_not_dilute_the_correction_for_the_rest():
    """Padding a map with unusable cells must not make a real fabric harder to establish."""
    null = fg.build_control_null(_cardinal_background(6000, 5), _tiles(6000, 60, 5),
                                 cut={"a": 1.35}, n_tiles=25, n_boot=800)
    real = ("real", _rose(45.0, 1.2, 900, 12), np.random.default_rng(1).integers(0, 25, 900))
    junk = [(f"j{i}", _rose(45.0, 5.0, 5, 200 + i), np.arange(5)) for i in range(500)]
    alone = fg.fabric_gate([real], null, cut={"a": 1.35}, design_effect=1.0)
    padded = fg.fabric_gate([real] + junk, null, cut={"a": 1.35}, design_effect=1.0)
    assert alone[0]["claimable"] == padded[0]["claimable"]


# --- defect 4: clustering ------------------------------------------------------------------------

def test_clustered_angles_report_a_design_effect_above_one():
    """Angles correlated within tiles must inflate the variance relative to i.i.d."""
    rng = np.random.default_rng(6)
    per_tile, n_tiles = 60, 40
    tile_mean = rng.uniform(0, 180, n_tiles)
    angles = np.concatenate([_rose(m, 8.0, per_tile, 300 + i) for i, m in enumerate(tile_mean)])
    tiles = np.repeat(np.arange(n_tiles), per_tile)
    assert fg.tile_design_effect(angles, tiles) > 1.5


def test_unclustered_angles_sit_near_a_design_effect_of_one():
    n = 2400
    angles = _rose(0.0, 0.35, n, 7)
    assert 0.6 < fg.tile_design_effect(angles, _tiles(n, 40, 7)) < 1.6


# --- the cut must travel with the number ----------------------------------------------------------

def test_comparing_across_cuts_is_refused():
    null = fg.build_control_null(_cardinal_background(2000, 8), _tiles(2000, 20, 8),
                                 cut={"aspect": 1.35}, n_tiles=20, n_boot=200)
    with pytest.raises(ValueError, match="cut"):
        fg.fabric_gate([("c", _rose(45, 2, 400), _tiles(400, 20))], null, cut={"aspect": 1.7})


def test_a_real_strong_fabric_still_gets_through():
    """A gate that refuses everything is not a gate."""
    null = fg.build_control_null(_cardinal_background(6000, 9), _tiles(6000, 60, 9),
                                 cut={"a": 1.35}, n_tiles=30, n_boot=1500)
    rows = fg.fabric_gate([("strong", _rose(45.0, 2.0, 1500, 13),
                            np.random.default_rng(2).integers(0, 30, 1500))],
                          null, cut={"a": 1.35})
    assert rows[0]["claimable"], rows[0]["reason"]
