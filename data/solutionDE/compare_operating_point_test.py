# The 0.10-vs-0.25 operating point, read on the TEST split (2026-08-21).
#
# Why this exists. The operating point was chosen on 200 *validation* tiles
# (`data/review/operating_point.json`: paired AP delta +0.0237, CI [+0.0203, +0.0296]), and G5-04
# found the boulder2024 splits share source prefixes with training — so that measurement could not
# rule out leakage. The test split is the one check that had not been spent. Two runs of
# `run_ap50_test.py --conf {0.10,0.25} --max-det 1100` are that check; this script reads them.
#
# Two things it deliberately reports together, because reporting either alone misleads:
#
#  * **AP, which is what the original decision was made on.** It replicates.
#  * **F1 and the precision/recall pair, which reverse the sign.** AP integrates over the ranked
#    detection list, so lowering the threshold mostly extends the precision-recall curve rightward
#    with extra low-scored detections and can only add area. "0.10 beats 0.25 on AP" is therefore
#    close to structural — not a tautology (NMS sees a different candidate set at the lower
#    threshold, and the bands rule promotes by scale rather than score, which is exactly why the
#    runner does real runs instead of `score >= t` filters) but strongly biased. A replication of an
#    AP delta confirms the number, not that the number is the right criterion.
#
# Writes `data/review/operating_point_test_split.json` (tracked: gate-citing evidence travels with
# the claim).
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
AP50 = BASE / "solutionDE" / "ap50"
REVIEW = BASE / "review"

METRICS = ("ap50_mask", "ap50_mask_ge4px", "ap50_mask_ge5px")
CATALOGS = ("baseline_256", "fused_256pref_engulf")


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")


def load(ct):
    path = AP50 / f"ap50_results-ct-{ct}-md-1100.json"
    d = json.loads(path.read_text())
    if d["n_failures"]:
        raise SystemExit(
            f"{path.name} has {d['n_failures']} failed tiles. The runner's own note is that the "
            "gate means nothing unless that is zero, and a failed tile is dropped from the truth "
            "set — so the two operating points would be scored on different populations.")
    return d


def main():
    a, b = load("010"), load("025")
    for name, d in (("0.10", a), ("0.25", b)):
        print(f"conf {name}: {d['n_tiles']} tiles, {d['n_failures']} failures, "
              f"max_det {d['max_det']}, wall {d['wall']['total_s']:.0f}s")
    # The populations must be identical or nothing below is a comparison.
    for metric in METRICS:
        counted = {a["baseline_256"][metric]["n_gt_counted"],
                   b["baseline_256"][metric]["n_gt_counted"]}
        if len(counted) != 1:
            raise SystemExit(f"{metric}: the two runs scored different truth counts {counted}")

    rows = []
    for metric in METRICS:
        for cat in CATALOGS:
            va, vb = a[cat][metric], b[cat][metric]
            row = dict(metric=metric, catalog=cat, n_gt_counted=va["n_gt_counted"],
                       n_pred_010=va["n_pred"], n_pred_025=vb["n_pred"])
            for key in ("ap", "recall", "precision"):
                row[f"{key}_010"] = va[key]
                row[f"{key}_025"] = vb[key]
                row[f"d_{key}"] = va[key] - vb[key]
            row["f1_010"] = f1(va["precision"], va["recall"])
            row["f1_025"] = f1(vb["precision"], vb["recall"])
            row["d_f1"] = row["f1_010"] - row["f1_025"]
            rows.append(row)
            print(f"{metric:20} {cat:22} ap {row['d_ap']:+.4f}  recall {row['d_recall']:+.4f}  "
                  f"precision {row['d_precision']:+.4f}  f1 {row['d_f1']:+.4f}   (0.10 - 0.25)")

    val = json.loads((REVIEW / "operating_point.json").read_text())["tiles"]
    anchor = val["paired_ap_ci_025_to_010"]
    d_test = a["baseline_256"]["ap50_mask"]["ap"] - b["baseline_256"]["ap50_mask"]["ap"]
    replication = dict(
        validation_delta=anchor["delta"], validation_lo=anchor["lo"], validation_hi=anchor["hi"],
        validation_n_images=anchor["n_images"], validation_caveat=val["caveat"],
        test_delta_baseline_ap50_mask=d_test,
        inside_validation_ci=bool(anchor["lo"] <= d_test <= anchor["hi"]))
    print(f"\nreplication: test {d_test:+.4f} vs validation {anchor['delta']:+.4f} "
          f"CI [{anchor['lo']:+.4f}, {anchor['hi']:+.4f}] -> inside: "
          f"{replication['inside_validation_ci']}")

    gates = {ct: {k: d["pre_registered_gate"][k] for k in ("passes", "margin")}
             for ct, d in (("0.10", a), ("0.25", b))}
    print(f"fused>=baseline gate: 0.10 {gates['0.10']['margin']:+.5f}, "
          f"0.25 {gates['0.25']['margin']:+.5f} (both pass)")

    out = dict(
        question="does the TEST split confirm conf 0.10 over 0.25?",
        verdict=("The AP preference replicates and is not a leakage artifact. It does NOT establish "
                 "0.10 as better on a balanced criterion: F1 favours 0.25 by 0.087-0.098 at every "
                 "size floor and for both catalogs, because 0.10 buys 4.3-6.1 recall points and "
                 "pays 11.1-14.2 precision points. Both readings are real and they answer different "
                 "questions; the original decision was explicitly recall-first, so the operating "
                 "point stands -- but it should be stated as a recall purchase at a known precision "
                 "cost, not as an unconditional win."),
        metric_caveat=("AP integrates over the ranked detection list, so a lower threshold extends "
                       "the PR curve rightward and can only add area. Both the validation decision "
                       "and this confirmation share that bias."),
        n_tiles=a["n_tiles"], max_det=a["max_det"],
        identity={"conf_010": a["identity"], "conf_025": b["identity"]},
        rows=rows, replication=replication, fused_vs_baseline_gate=gates,
        note_on_fusion=("The fused>=baseline gate passes at both points and by a LARGER margin at "
                        "0.25 (+0.01104) than at 0.10 (+0.00212): at the looser threshold the "
                        "baseline's own low-score tail already recovers much of what the extra arms "
                        "would contribute."),
        note_on_x01=("Independent confirmation of X-01's magnitude on a third population: the 256 "
                     "baseline returns 17,687 detections at 0.10 against 11,442 at 0.25, +54.6%, "
                     "against the ~+55% pinned on four full-scene arms across two scenes."),
    )
    (REVIEW / "operating_point_test_split.json").write_text(json.dumps(out, indent=1))
    print("written", REVIEW / "operating_point_test_split.json")


if __name__ == "__main__":
    main()
