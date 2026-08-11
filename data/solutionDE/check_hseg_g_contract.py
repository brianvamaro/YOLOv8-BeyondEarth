"""Pre-flight for H-seg x G: does the path the ladder WRITES equal the path the analyzer READS?

That contract is the one that was broken. `analyze_hseg_acceptance.py` re-derived catalog filenames
with `ct-010` and an untagged mask hardcoded, so the scoring half could not read the output of its
own GPU half once conf was retuned or `mask_tag="sam2"` was passed. Both sides now go through
`predict.prediction_filenames`, but the *directory* halves are still chosen independently by the
ladder's `--dir-tag` and the analyzer's `--{hseg,prod}-rom-dir` / `--{hseg,prod}-dev-dir`, so they
can still be pointed at different places by a typo.

This is a pre-flight, not a unit test: it needs `data/` and therefore cannot live in `tests/`
(which must not depend on gitignored data). Run it before spending GPU on the arms.

    python data/solutionDE/check_hseg_g_contract.py --hseg-conf 0.15 --dir-tag _g

Exits non-zero on any mismatch. The naming itself is pinned by tests/test_prediction_filenames.py.

**What it does and does not catch.** It catches the two halves disagreeing about the *directory*
convention (e.g. someone renaming `hseg{tag}_{scene}` to `hseg_{scene}{tag}` on one side only) and
any divergence in the filename halves. It does **not** independently re-derive the naming — both
sides deliberately call the same `prediction_filenames`, which is the point — so it cannot catch an
error inside that function; `tests/test_prediction_filenames.py` covers that end-to-end instead.
Requires a tagged run: the untagged production dev-crop layout is per-arm and historical
(`{scene}_512`, `{scene}_baseline`, `{scene}_e2stream`) rather than a single ladder directory.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DE = ROOT / "data" / "solutionDE"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DE))

from YOLOv8BeyondEarth.predict import prediction_filenames  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def writer_path(arms, sub_match, ss, sr, conf, mask_tag):
    """Where `_ladder.run_ladder` would write this arm — the same derivation it resumes on."""
    for sub, raster, s, r in arms:
        if sub_match in sub.replace("\\", "/") and s == ss and r == sr:
            return (DE / "preds" / sub / prediction_filenames(
                Path(raster).stem, confidence_threshold=conf, slice_size=s, inference_size=1024,
                overlap_height_ratio=0.2, sr_factor=r, downscale_pred=True, mask_tag=mask_tag,
                output_format="gpkg")["nms"])
    raise AssertionError(f"no arm matching {sub_match!r} ss={ss} sr={sr}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hseg-conf", type=float, default=0.15,
                   help="H-seg's retuned operating point (step 0 of the plan)")
    p.add_argument("--prod-conf", type=float, default=0.10, help="production's operating point")
    p.add_argument("--dir-tag", default="_g", help="the --dir-tag both ladders will be given")
    p.add_argument("--no-remask", action="store_true", help="check a non-G run instead")
    a = p.parse_args()
    tag, mask_tag = a.dir_tag, None if a.no_remask else "sam2"
    if not tag:
        p.error("--dir-tag must be non-empty. This pre-flight checks a NEW tagged arm set; the "
                "untagged production dev crops live in per-arm historical directories "
                "({scene}_512 / _baseline / _e2stream), not one ladder directory, so there is no "
                "contract to check there.")

    hseg = _load("rhl", DE / "run_hseg_ladder.py")
    goal3 = _load("rgl", DE / "run_goal3_ladder.py")
    ana = _load("ahs", DE / "analyze_hseg_acceptance.py")

    ana.ARM_SPECS["hseg"].update(
        conf=a.hseg_conf, mask_tag=mask_tag, rom_dir=f"hseg{tag}_rom_ladder",
        dev_dirs={k: f"hseg{tag}_{{scene}}" for k in ana.ARM_SPECS["hseg"]["dev_dirs"]})
    ana.ARM_SPECS["prod"].update(
        conf=a.prod_conf, mask_tag=mask_tag, rom_dir=f"rom{tag}_ladder",
        dev_dirs={k: f"{{scene}}{tag}_ladder" for k in ana.ARM_SPECS["prod"]["dev_dirs"]})

    h_arms, p_arms = hseg.arms_for(tag), goal3.arms_for(tag)
    if not any("rom" in s for s, *_ in h_arms):
        print("WARNING: no hirise_rom??.tif found — the ROM half of the contract is unchecked.")
    bad = 0
    for ss, sr in ((512, 1), (256, 1), (256, 2)):
        cases = [
            ("hseg ROM", writer_path(h_arms, f"rom_ladder/hirise_rom00", ss, sr, a.hseg_conf, mask_tag),
             ana.rom_gpkg_for("hseg")("hirise_rom00", ss, sr)),
            ("prod ROM", writer_path(p_arms, f"rom{tag}_ladder/hirise_rom00", ss, sr, a.prod_conf, mask_tag),
             ana.rom_gpkg_for("prod")("hirise_rom00", ss, sr)),
            ("hseg dev", writer_path(h_arms, f"hseg{tag}_hirise", ss, sr, a.hseg_conf, mask_tag),
             ana.dev_gpkg("hseg", "hirise", "hirise_crop4k", ss, sr)),
            ("prod dev", writer_path(p_arms, f"hirise{tag}_ladder", ss, sr, a.prod_conf, mask_tag),
             ana.dev_gpkg("prod", "hirise", "hirise_crop4k", ss, sr)),
        ]
        for label, written, read in cases:
            ok = written == read
            bad += not ok
            print(f"  {'OK      ' if ok else 'MISMATCH'} {label}  {written.relative_to(DE)}")
            if not ok:
                print(f"      analyzer expects: {read.relative_to(DE)}")

    n = 12
    print(f"\n{n - bad}/{n} producer/consumer paths agree")
    if a.hseg_conf != a.prod_conf:
        print(f"per-arm conf confirmed distinct: hseg ct-{int(a.hseg_conf * 100):03d} vs "
              f"prod ct-{int(a.prod_conf * 100):03d} — the comparison step 0 requires")
    else:
        print("NOTE both arms at the same conf; step 0 asks for H-seg to be retuned first.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
