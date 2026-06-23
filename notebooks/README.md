# notebooks/

**New** notebooks for ongoing BoulderNet work (analysis, diagnostics, experiments).

> Legacy tutorial notebooks live in [`../resources/nb/`](../resources/nb/) and are
> **hand-maintained from the original fork — do not edit them.** Everything new goes here.

## Conventions

- **Logic belongs in `src/`.** Keep `YOLOv8BeyondEarth` (under [`../src/`](../src/)) as the
  source of truth; notebooks should *import and call* it, not redefine pipeline logic. If a
  notebook grows real logic, promote it into `src/` and call it back.
- **Document aggressively.** Lead with a markdown cell stating the notebook's purpose, inputs,
  outputs, and the env it expects. Narrate each step in markdown — a reader should follow the
  reasoning without running it. Record assumptions, data paths, and conclusions.
- **One at a time.** Never run two notebooks (or two context-heavy jobs) concurrently.
- **Hyperlink every citation** to its canonical DOI.
- **Reproducibility.** Note the `bouldernet` env, set random seeds where relevant, and prefer
  relative/parameterized paths over hard-coded absolute ones.

## Running

Conda is not on PATH. Launch Jupyter (or execute a notebook) via the absolute `conda run`:

```bash
C:\Users\brian\anaconda3\Scripts\conda.exe run -n bouldernet jupyter lab
```
