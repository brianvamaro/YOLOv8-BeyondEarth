# Sherlock training runbook — lessons learned (prep for the next run)

Everything we hit getting Solution H training onto Sherlock (mlapotre group, OnDemand), with the
fix for each. Read this **before** launching a new training run so you don't rediscover them.
`README.md` is the step-by-step workflow; this is the "why / what bit us / known-good config".

## TL;DR — the config that actually works (2026-07-07)
- **Env:** venv on `$GROUP_HOME` (`/home/groups/mlapotre/bamaro/envs/bouldernet26`), `ml python/3.12.1`,
  `pip install --only-binary=:all: ultralytics==8.4.75 shapely scikit-image pandas opencv-python-headless`.
- **GPU:** `gpu` partition, 24 GB card (RTX 3090) is plentiful; 32/48/80 GB also requestable.
- **obb:** `batch 8, imgsz 1024`, ~6 min/epoch, **~19 h for 200 ep**. Fits 24 GB (~16 G used).
- **seg:** `mask_ratio=4, batch 8` (NOT `mask_ratio=1` — see below). Fits 24 GB.
- **Use `--no-amp` (`AMP=0`) for definitive runs** — AMP diverged to NaN (see below).
- **Model converged by ~epoch 45** (obb); 200 epochs was overkill. Consider ~80 for reruns.
- Result: obb `best.pt` = OBB mAP50 **0.819** / mAP50-95 **0.534** (vs stock 0.762 / 0.350).

## Prep checklist for a new run
1. **Build the two upload tarballs on the laptop** (browser uploads choke on many small files):
   `bnet_code.tgz` (scripts, sherlock, src, `data/weights/yolo26m*.pt`) + `boulder2024.tar` (dataset).
   Commands in `README.md` → "OnDemand quickstart".
2. **Upload** via OnDemand **Files**; **extract** in the OnDemand **Shell** (login node) into `~/BoulderNet`.
3. **Build the env** (`setup_sherlock_env.sh`) — it now bakes in all the pip fixes below.
4. **Pre-stage weights** — the pretrained `.pt` + `yolo26n.pt` (AMP check) must already be in
   `data/weights/` (batch nodes are offline; see gotcha #3).
5. **Create the log dir**: `mkdir -p ~/BoulderNet/logs` (gotcha #7).
6. **Probe** (`EPOCHS=3`) → read per-epoch time → set `--time` = per-epoch × epochs × 1.3.
7. **Submit** seg + obb as **separate jobs** (one failing doesn't take both down).

## Gotchas & fixes (each cost us a failed job)
| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `ImportError: libGL.so.1` on first `import cv2` | ultralytics pulls `opencv-python`, needs system OpenGL — absent on headless nodes | `opencv-python-headless` (setup script does the swap) |
| 2 | `ModuleNotFoundError: No module named 'pandas'` | ultralytics 8.4 dropped pandas (uses polars); `orientation.py` imports it | `pip install pandas` (in setup script) |
| 3 | AMP-check download fails / `[Errno 2] ... yolo26n.pt` | `gpu` **batch** nodes have no outbound internet | pre-stage `yolo26m-*.pt` + `yolo26n.pt` in `data/weights/`; AMP check falls back gracefully |
| 4 | `Dataset ... images not found, missing '.../datasets/c:/Users/...'` | `boulder2024_local.yaml` has a Windows absolute `path:` | sbatch rewrites `path:` to the cluster location at submit (or `sed` it manually) |
| 5 | `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` every step; ~15 min/epoch | `batch=16` at imgsz 1024 saturates the 24 GB 3090 (~23 G); assigner spills to CPU | `batch=8` (leaves headroom; assigner stays on GPU) |
| 6 | seg OOMs even at batch 1 / **even on a 48 GB card**; `single_mask_loss` tries to alloc ~8 GB | `mask_ratio=1` builds full-res `[n_inst,1024,1024]` mask tensors; dense tiles have 100s–1000s of instances | `mask_ratio=4` (16× less mask-loss memory); fits 24 GB at batch 8 |
| 7 | Job dies in ~15 s, exit 1, log path "not found" | SLURM opens `--output=logs/...` (relative to submit dir) **before** the script's `mkdir -p logs` | `mkdir -p ~/BoulderNet/logs` **before** `sbatch`; submit from `~/BoulderNet` |
| 8 | `cls_loss → ~250`, val mAP → ~0, `EMA contains NaN/Inf`, run diverges **late** (slow erosion over ~150 ep) | `amp=True` + skipped offline AMP self-check = numerically unstable for this model | `--no-amp` (`AMP=0`) for definitive runs; `best.pt` still captures the pre-collapse peak |
| 9 | **Sudden single-epoch NaN** (all losses `nan` in one epoch) then a hard OOM crash, on the `DEGREES=180` R-A obb arm; `TaskAlignedAssigner ... using CPU` spamming beforehand (job 33330690, ep37→38) | `batch=8` + rotation aug builds very dense tiles (a 1649-instance batch seen); the assigner OOMs → CPU-spills → a corrupt/degenerate rotated-OBB target yields a NaN grad in one step, then OOM-crashes the next. **Not AMP** — verified `amp: false` in `args.yaml`; this is memory, distinct from #8's slow AMP erosion | Give it headroom so the assigner never spills: `--constraint='GPU_MEM:48GB'` (keeps batch 8, comparable to the degrees=0 arm) **or** `BATCH=4`; add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. If it still NaNs with *no* assigner-CPU warnings → the cause is rotation-aug OBB target degeneracy, not memory (address separately) |

## Config decisions & their trade-offs
- **`mask_ratio=1` (Prieur, faithful) is not viable here.** It OOMs even on 48 GB because the densest
  boulder tiles + full-res masks blow up the loss, and where it fits (batch 2) it's ~60 min/epoch →
  ~200 h. **Use `mask_ratio=4`** and note the deviation on the seg card. A perfectly faithful seg
  comparison would need an 80 GB card *and* accepting the batch-2/200 h cost (or a data subset).
- **AMP off for the definitive run.** The obb run peaked at epoch 45 (0.819/0.534) then eroded into a
  NaN collapse by epoch 200 — the AMP instability, not the LR schedule. `best.pt` is safe (saved at
  the peak, and ultralytics refuses to overwrite it once EMA is NaN), so a diverged run still yields a
  usable model — but for a clean end-to-end curve use `--no-amp`.
- **Epochs.** obb converged by ~epoch 45; the remaining 155 added nothing. 200 is faithful to Prieur
  but wasteful — ~80 epochs (with the LR schedule set to match) is likely plenty for reruns.
- **degrees=0** (no rotation aug) is intentional — controlled comparison vs the stock baseline;
  rotation aug is a separate deferred arm (R-A).

## SLURM quick reference (mlapotre / OnDemand)
- Submit from a login-node **Shell** — **no env activation needed** (the sbatch does `ml` + `source`).
- `sbatch --export=ALL,VARIANT=obb,BATCH=8,MASK_RATIO=4,AMP=0,NAME=... train_h.sbatch` — env vars override header defaults.
- `sbatch --constraint='GPU_MEM:48GB' ...` — command-line overrides the header's `GPU_MEM:24GB`.
- Available `GPU_MEM` features: `11GB 12GB 16GB 24GB 32GB 48GB 80GB` (`sinfo -p gpu -o "%f" | tr ',' '\n' | grep GPU_MEM | sort -u`).
- `squeue -u $USER` (`PD`=queued, `R`=running) · `scancel <JOBID>` · `scancel -u $USER` (all).
- Which GPU a job got: `head -5 ~/BoulderNet/logs/h-retrain-<JOBID>.out` (sbatch logs `nvidia-smi`).
- **Verify a flag actually took** (don't assume from the `--export`): `grep -i '^amp:' $SCRATCH/bouldernet/h_training/<name>_<variant>/args.yaml` — ultralytics writes the *resolved* args there. The sbatch also now echoes `resolved: AMP=... amp_flag=...` near the top of the log. An `AMP: running ... checks` line means AMP was **on** (only runs when `amp=True`).
- Live GPU util: `srun --jobid=<JOBID> --overlap --pty nvidia-smi`.
- Logs land in `~/BoulderNet/logs/` (relative to the submit dir), **not** `YOLOv8-BeyondEarth/logs/`.
- Outputs → `$SCRATCH/bouldernet/h_training/<name>_<variant>/weights/best.pt`; download via OnDemand Files.

## Sync note
The Sherlock copy is a **one-time tarball snapshot** — local `git commit`s do NOT propagate. Either
re-upload changed files, or (cleaner long-term) `git push` + `git pull` on Sherlock. The `--export`
flags override stale sbatch defaults, so you can usually avoid re-uploading for a config tweak.
