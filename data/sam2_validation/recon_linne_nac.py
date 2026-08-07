"""Component-B recon (2026-07-23): LRO NAC illumination lever over Linne crater (27.7N, 11.8E).

Queries ODE REST for LROC NAC CDR products over a small box around Linne, then reads NORTH_AZIMUTH +
SUB_SOLAR_AZIMUTH out of each product's embedded CUMINDEX record (ODE has no sub-solar-azimuth field,
but it ships the raw index row inside ODE_notes). Geographic sun axis = (SUB_SOLAR - NORTH) mod 180
(same convention as mars_sun.json). Prints a table sorted by incidence and reports the sun-axis span
within constant-incidence bands.

Finding: a near-nadir (emi<8), inc 30-37 deg, 0.5-1.1 m/px subset spans ~112 deg of sun axis (two
levels ~35 and ~147, from ascending vs descending passes) -> lever confirmed >= 60 deg. Full Component
B (download + co-register flipped/oppositely-lit scenes + detect->SAM->match->dAngle-vs-dSun) is
DEFERRED (see docs/illumination_artifact_exploration_plan.md). No ISIS locally -> co-registration is
the real cost. Run: conda run -n bouldernet python recon_linne_nac.py
"""
import json, urllib.request, urllib.parse
import numpy as np

BASE = "https://oderest.rsl.wustl.edu/live2/"
BOX = dict(minlat=27.60, maxlat=27.85, westernlon=11.65, easternlon=11.95)  # ~Linne


def fetch(params):
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "boulder-recon/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def index_record(prod):
    notes = prod.get("ODE_notes", {}).get("ODE_note", [])
    notes = notes if isinstance(notes, list) else [notes]
    cand = [n for n in notes if n.count(",") > 20]
    return max(cand, key=lambda s: s.count(",")) if cand else None


def main():
    d = fetch(dict(query="product", target="moon", ihid="LRO", iid="LROC",
                   pt="CDRNAC4", results="mp", output="JSON", **BOX))
    plist = d["ODEResults"]["Products"]["Product"]
    plist = plist if isinstance(plist, list) else [plist]

    rows = []
    for p in plist:
        rec = index_record(p)
        if not rec:
            continue
        nums = []
        for t in rec.split(","):
            try:
                nums.append(float(t.strip().strip('"')))
            except ValueError:
                nums.append(None)
        emi = float(p["Emission_angle"]); inc = float(p["Incidence_angle"]); pha = float(p["Phase_angle"])
        north = subsolar = None
        for i in range(len(nums) - 4):                     # locate (emi, inc, pha); next 2 = N_az, subsol_az
            w = nums[i:i + 3]
            if None in w:
                continue
            if abs(w[0] - emi) < 0.02 and abs(w[1] - inc) < 0.02 and abs(w[2] - pha) < 0.02:
                north, subsolar = nums[i + 3], nums[i + 4]
                break
        if north is None:
            continue
        pid = p.get("pdsid") or p.get("Product_name", "?")
        rows.append(dict(pdsid=str(pid).split(".")[-1].upper().replace(".IMG", ""),
                         inc=inc, emi=emi, res=float(p["Map_resolution"]), north=north,
                         subsolar=subsolar, sun_axis=(subsolar - north) % 180.0,
                         clat=float(p["Center_latitude"]), utc=p["UTC_start_time"][:10]))

    rows.sort(key=lambda r: r["inc"])
    print(f"parsed {len(rows)}/{len(plist)} products\n")
    hdr = f"{'pdsid':14s} {'inc':>6s} {'emi':>6s} {'res':>6s} {'sun_axis':>8s} {'clat':>7s} {'date':>11s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['pdsid']:14s} {r['inc']:6.1f} {r['emi']:6.1f} {r['res']:6.2f} "
              f"{r['sun_axis']:8.1f} {r['clat']:7.2f} {r['utc']:>11s}")

    good = [r for r in rows if r["emi"] < 8 and 30 <= r["inc"] < 37 and r["res"] < 1.15]
    ax = np.array(sorted(r["sun_axis"] for r in good))
    gaps = np.diff(np.r_[ax, ax[0] + 180.0])
    print(f"\nnear-nadir inc[30,37) res<1.15: n={len(good)}  "
          f"sun-axis levels ~{ax.min():.0f} and ~{ax.max():.0f}; "
          f"axial span (180 - largest gap) = {180 - gaps.max():.0f} deg")


if __name__ == "__main__":
    main()
