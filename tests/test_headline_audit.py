"""Tests for `scripts/review/audit_headline_numbers.py` — the claim gate (re-run blocker A-3).

The audit shipped with two ways of passing while a citable document was wrong:

* its pattern wanted the literal ``lower than YOLO on every body``, so the notebook's
  "SAM's median |Δangle| is lower than YOLO's on **every** body" — a possessive and two asterisks
  away — went undetected while the same notebook's own table twenty lines later said the opposite;
* it returned **0** when the gitignored T35 table was absent, so on a fresh clone (or any CI without
  `data/`) it reported success having checked nothing.

These pin both, plus the retired-claim registry added on top.
"""
import sys
from pathlib import Path

import pytest

REVIEW = Path(__file__).resolve().parents[1] / "scripts" / "review"
sys.path.insert(0, str(REVIEW))

import audit_headline_numbers as A                                          # noqa: E402

_TABLE_ABSENT = {"n_matched": None, "n_detections": None, "sam_wins_every_body": False}


def _check(tmp_path, text, facts=None, name="doc.md"):
    """`name` matters: in a jupytext `.py` a leading '# ' is a comment marker, in a `.md` it is an
    H1 heading — and headings end a correction paragraph while comment markers do not."""
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return A.check_docs(facts or _TABLE_ABSENT, docs=[p])


# --- normalisation: a claim must not be able to hide behind formatting ---------------------------

@pytest.mark.parametrize("line", [
    "SAM's median |Δangle| is lower than YOLO's on **every** body.",
    "SAM is lower than YOLO on every body",
    "`SAM` beats YOLO *everywhere*",
    "SAM   beats   YOLO   on   every   body",
])
def test_markdown_emphasis_cannot_hide_the_g2_09_claim(tmp_path, line):
    problems = _check(tmp_path, line)
    assert len(problems) == 1 and "[G2-09]" in problems[0]


def test_the_exact_notebook_wording_that_slipped_through(tmp_path):
    """Verbatim from `Goal2_SAM2_angle_validation.py` before 2026-08-06. The audit exited 0 on it."""
    line = ("# - **Angle accuracy vs human:** SAM's median \\|Δangle\\| is lower than YOLO's on "
            "**every** body")
    assert _check(tmp_path, line), "the wording that defeated the pre-fix audit must now be caught"


def test_g2_09_is_only_a_defect_while_the_table_disagrees(tmp_path):
    """If Mars ever flips, the claim stops being false and the audit must stop failing on it."""
    line = "SAM beats YOLO on every body"
    assert _check(tmp_path, line, {"n_matched": 1, "n_detections": 1,
                                   "sam_wins_every_body": True}) == []


# --- the retired-claim registry -------------------------------------------------------------------

@pytest.mark.parametrize("cid,line", [
    ("GATE-R", "the roses still reach R = 0.35 with peaks there"),
    ("GATE-R", "a candidate must beat the control-site null (R ≈ 0.35)"),
    ("GATE-PEAKS", "the control sites give peaks at 43-76 degrees"),
    ("G2-01", "it is not something the re-masker introduces"),
    ("T3-01", "the near-circular pile-up is 12.9% on the treatment scene"),
    ("T2-RECALL", "counts collapse at high incidence (7-24 vs 211-711 on identical ground)"),
    ("T2-RECALL", "the detector finds 7–24 boulders where its partner finds 211–711"),
])
def test_retired_claims_are_caught(tmp_path, cid, line):
    problems = _check(tmp_path, line)
    assert any(f"[{cid}]" in p for p in problems), f"{cid} not caught in {line!r}"


def test_a_retraction_may_quote_the_wording_it_retracts(tmp_path):
    """Otherwise no document could ever record what it withdrew."""
    assert _check(tmp_path, "⚠ WITHDRAWN: the bar was R ≈ 0.35 with peaks at 43-76°.") == []


def test_the_exemption_spans_the_paragraph_not_just_the_line(tmp_path):
    """A retraction wraps. The retired wording usually lands on a continuation line that carries no
    marker of its own, so a line-scoped exemption produces false positives on correct documents."""
    text = ("*(Corrected 2026-08-06: this cell read\n"
            "\"lower than YOLO on **every** body\", contradicting the table below.)*\n")
    assert _check(tmp_path, text) == []


def test_the_exemption_spans_a_wrapped_python_comment(tmp_path):
    """Same, in a jupytext source, where every line carries a '# ' that is NOT a heading."""
    text = ("# *(Corrected 2026-08-06: this cell read\n"
            "#   \"lower than YOLO on **every** body\", contradicting the table below.)*\n")
    assert _check(tmp_path, text, name="doc.py") == []


def test_the_exemption_survives_a_blockquote_continuation(tmp_path):
    """A blockquote is a container, not an assertion: '> ' continuation lines inside a retraction
    quote must stay exempt, or a wrapped sentence resets on every single line. This is the shape the
    real T2 correction in orientation_results_summary.md uses."""
    text = ("> ⚠ **Withdrawn 2026-08-06.** The pooled form of this is a cross-pair mixture:\n"
            "> the ranges quoted as high-vs-low incidence, 7-24 vs 211-711, put two of the six\n"
            "> members on the wrong side.\n")
    assert _check(tmp_path, text) == []


def test_the_exemption_ends_at_a_blank_line(tmp_path):
    """...but it must not leak into the rest of the document."""
    text = ("⚠ Corrected: an earlier draft overstated this.\n"
            "\n"
            "SAM beats YOLO on every body.\n")
    assert len(_check(tmp_path, text)) == 1


def test_the_exemption_ends_at_the_next_list_item(tmp_path):
    """The blast-radius fix (pass 9, D1-02). Scoping the exemption to 'until the next blank line'
    silenced 11.4% of all tracked lines, because these documents put the disclaiming word in the
    FIRST bullet of long lists — so one caveat bullet exempted every claim after it."""
    text = ("- One caveat: densities from this sample are lower bounds.\n"
            "- The roses still reach R = 0.35 with peaks at 43-76 deg.\n"
            "- The near-circular pile-up is 12.9% on the treatment scene.\n")
    problems = _check(tmp_path, text)
    assert len(problems) == 3, f"a caveat bullet must not silence its siblings: {problems}"


def test_a_list_item_inside_a_quote_still_ends_the_exemption(tmp_path):
    """Stripping the '>' container must not also swallow the list marker underneath it."""
    text = ("> ⚠ Corrected: an earlier draft overstated this.\n"
            "> - SAM beats YOLO on every body.\n")
    assert len(_check(tmp_path, text)) == 1


def test_matched_count_check_does_not_crash_without_the_table(tmp_path):
    """`main()` runs the document scan with n_matched=None so it works on a fresh clone. Before the
    pass-9 fix, any document quoting a matched count made that pre-scan raise TypeError from the
    f-string — so the check named after the very defect this script exists for could never report."""
    problems = _check(tmp_path, "The study matched 20,896 objects matched to a human label.")
    assert problems == [], "no table means no matched-count verdict, but it must not raise"


def test_matched_count_check_fires_when_the_table_is_present(tmp_path):
    """...and with the table it must still catch a wrong count."""
    problems = _check(tmp_path, "26,830 objects matched to a human label.",
                      {"n_matched": 20896, "n_detections": 30756, "sam_wins_every_body": False})
    assert len(problems) == 1 and "26,830" in problems[0] and "20,896" in problems[0]


# --- an audit that cannot audit must not report success -------------------------------------------

def test_missing_table_is_not_a_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "T35", tmp_path / "absent.csv")
    monkeypatch.setattr(A, "TRACKED_DOCS", [])
    assert A.main([]) == 2, "a fresh clone has no data/, and vacuous success is the A-3 defect"
    assert A.main(["--allow-missing-table"]) == 0


def test_missing_table_still_runs_the_document_scan(monkeypatch, tmp_path):
    """The half that caught the citable notebook needs no data, so it must run regardless."""
    doc = tmp_path / "doc.md"
    doc.write_text("SAM beats YOLO on every body", encoding="utf-8")
    monkeypatch.setattr(A, "T35", tmp_path / "absent.csv")
    monkeypatch.setattr(A, "TRACKED_DOCS", [doc])
    assert A.main([]) == 1
    assert A.main(["--allow-missing-table"]) == 1, "--allow-missing-table must not mute real hits"


# --- and the shipped documents must pass ----------------------------------------------------------

def test_the_tracked_documents_are_clean():
    present = [p for p in A.TRACKED_DOCS if p.exists()]
    assert len(present) >= 4, f"tracked documents have moved: {A.TRACKED_DOCS}"
    facts = A.derive() if A.T35.exists() else _TABLE_ABSENT
    problems = A.check_docs(facts, docs=present)
    assert problems == [], "\n".join(problems)
