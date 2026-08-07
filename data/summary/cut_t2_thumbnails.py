"""Matched image thumbnails for the same-ground / different-sun figure.

The repeat-pair result — on fixed ground, the boulder rose does **not** rotate with the sun — is a
negative result, and negative results need the reader to trust that the two panels really are the same
patch of ground under different illumination. A rose pair alone cannot show that; a thumbnail pair can
(the same craters and boulders are recognisable, and the shadows visibly point differently).

So this cuts, for each genuine repeat pair, **the same geographic window** from both acquisitions,
using the identical numeric-bounds overlap the SAM run used
(`data/sam2_validation/run_t2_hirise_sam.py`: no reprojection, which sidesteps the known lat-of-origin
projection bug — the WKTs can differ but the coordinates coincide for repeat-target acquisitions).

Window choice is not cosmetic: it is picked to be the most boulder-populated sub-window *common to both
members*, scored on the low-incidence member (the one that actually detects boulders) using a simple
local-contrast proxy, so the thumbnails show rocks rather than flat regolith. Contrast is stretched
per-thumbnail to its own percentiles, and the stretch bounds are recorded — the two members have very
different DN ranges, and a shared stretch would render the high-incidence member black.

Also emitted: each member's sun axis and incidence (from its `.LBL`), so the figure can draw the sun
line without re-reading labels, and the scale bar length in pixels.

No GPU. Run from inside YOLOv8-BeyondEarth/:
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run -n bouldernet python data/summary/cut_t2_thumbnails.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
ROOT = BASE / "YOLOv8-BeyondEarth"
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "summary"

# Same pair list as the SAM T2 run; '_2260'/'_2270' is a latitude band, not one footprint.
PAIRS = [("2270a", "ESP_067214_2270", "ESP_069088_2270"),
         ("2260b", "ESP_076235_2260", "ESP_055833_2260"),
         ("2270b", "ESP_059499_2270", "ESP_055714_2270")]
THUMB = 768          # thumbnail side, native px
SEARCH_STEP = 512    # stride when hunting for a boulder-rich common window
MAX_PROBES = 240


def tif(o):
    p = ROOT / f"data/test2_pairs/{o}/{o}_RED.tif"
    return p if p.exists() else ROOT / f"data/mars_hirise/raster/{o}_RED.tif"


def sun_of(o):
    t = tif(o).with_suffix(".LBL").read_text(errors="replace")

    def g(k):
        return float(re.search(rf"{k}\s*=\s*([-\d.]+)", t).group(1))
    return (g("SUB_SOLAR_AZIMUTH") - g("NORTH_AZIMUTH")) % 180.0, g("INCIDENCE_ANGLE")


def texture(a):
    """Cheap boulder-richness proxy: local standard deviation of the 3x3-differenced field."""
    a = a.astype(np.float32)
    if a.size == 0 or a.max() == a.min():
        return 0.0
    gy, gx = np.gradient(a)
    return float(np.hypot(gx, gy).std())


def main():
    import rasterio
    from rasterio.windows import Window, from_bounds
    OUT.mkdir(parents=True, exist_ok=True)
    thumbs, meta = {}, {}
    for pid, a, b in PAIRS:
        with rasterio.open(tif(a)) as sa, rasterio.open(tif(b)) as sb:
            L, B = max(sa.bounds.left, sb.bounds.left), max(sa.bounds.bottom, sb.bounds.bottom)
            R, T = min(sa.bounds.right, sb.bounds.right), min(sa.bounds.top, sb.bounds.top)
            if R <= L or T <= B:
                print(f"SKIP {pid}: no numeric-bounds overlap")
                continue
            suns = {o: sun_of(o) for o in (a, b)}
            lead = min((a, b), key=lambda o: suns[o][1])       # lowest incidence = best detections
            others = [o for o in (a, b) if o != lead]
            srcs = {a: sa, b: sb}

            # hunt for a boulder-rich window on the lead member, inside the shared overlap
            w = from_bounds(L, B, R, T, transform=srcs[lead].transform)
            c0, r0 = int(round(w.col_off)), int(round(w.row_off))
            ww = min(int(round(w.width)), srcs[lead].width - c0)
            hh = min(int(round(w.height)), srcs[lead].height - r0)
            best, probes = None, 0
            for rr in range(r0, r0 + hh - THUMB, SEARCH_STEP):
                for cc in range(c0, c0 + ww - THUMB, SEARCH_STEP):
                    if probes >= MAX_PROBES:
                        break
                    probes += 1
                    arr = srcs[lead].read(1, window=Window(cc, rr, THUMB, THUMB))
                    if arr.shape != (THUMB, THUMB) or (arr > 0).mean() < 0.999:
                        continue
                    s = texture(arr)
                    if best is None or s > best[0]:
                        best = (s, cc, rr)
                if probes >= MAX_PROBES:
                    break
            if best is None:
                print(f"SKIP {pid}: no fully-valid {THUMB}px window in the overlap")
                continue
            _, cc, rr = best
            bx = rasterio.windows.bounds(Window(cc, rr, THUMB, THUMB), srcs[lead].transform)

            pair_entry = {}
            for o in (lead, *others):
                src = srcs[o]
                w2 = from_bounds(*bx, transform=src.transform)
                arr = src.read(1, window=w2, out_shape=(THUMB, THUMB),
                               resampling=rasterio.enums.Resampling.bilinear)
                valid = arr[arr > 0]
                lo, hi = (np.percentile(valid, [1, 99]) if valid.size else (0, 1))
                sun, inc = suns[o]
                pair_entry[o] = dict(img=arr, vmin=float(lo), vmax=float(hi), sun_axis=float(sun),
                                     incidence=float(inc), res_m=float(src.res[0]),
                                     is_lead=(o == lead))
                meta.setdefault(pid, {})[o] = dict(
                    sun_axis=round(sun, 1), incidence=round(inc, 1), res_m=round(src.res[0], 3),
                    stretch=[round(float(lo), 1), round(float(hi), 1)], is_lead=(o == lead))
            thumbs[pid] = pair_entry
            meta[pid]["window_bounds"] = [round(v, 2) for v in bx]
            meta[pid]["thumb_px"] = THUMB
            meta[pid]["overlap_m"] = [round(R - L, 1), round(T - B, 1)]
            meta[pid]["d_sun_deg"] = round(abs(((suns[b][0] - suns[a][0] + 90) % 180) - 90), 1)
            meta[pid]["d_inc_deg"] = round(abs(suns[b][1] - suns[a][1]), 1)
            print(f"{pid}: window {THUMB}px at ({cc},{rr}) on {lead} (lead, inc "
                  f"{suns[lead][1]:.0f}deg) | dsun {meta[pid]['d_sun_deg']:.0f}deg "
                  f"dinc {meta[pid]['d_inc_deg']:.0f}deg", flush=True)

    np.savez_compressed(OUT / "t2_thumbnails.npz", **{
        f"{pid}__{o}": e["img"] for pid, pe in thumbs.items() for o, e in pe.items()})
    (OUT / "t2_thumbnails_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT / 't2_thumbnails.npz'} "
          f"({sum(len(v) for v in thumbs.values())} thumbnails, {len(thumbs)} pairs)")
    print("Note the per-thumbnail contrast stretch in the meta: the two members of a pair have very "
          "different DN ranges, so a shared stretch would render the high-incidence member black.")


if __name__ == "__main__":
    main()
