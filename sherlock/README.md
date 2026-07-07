# Solution H on Sherlock — training the yolo26m seg + OBB retrain

The 8 GB RTX 5070 (laptop) can't train `yolo26m` at the production `imgsz=1024` (OOM; the seg
`mask_ratio=1` full-res masks OOM even at 640). The full 200-epoch run therefore goes to Sherlock's
`gpu` partition. This mirrors the `hirise2ctx` setup (venv on `$GROUP_HOME`, `mlapotre` group, no
conda). Everything here reuses `scripts/train_h.py` unchanged — only `--imgsz/--batch/--project`
differ from the local smoke.

## Layout on Sherlock
Replicate the two-level tree so `train_h.py`'s relative paths resolve (`REPO/data_prieur`):
```
$HOME/BoulderNet/
├── YOLOv8-BeyondEarth/      # this repo (git clone or rsync)
│   ├── scripts/train_h.py   # the entry point (same one used locally)
│   ├── sherlock/            # this dir
│   └── data/weights/        # PRE-STAGE the pretrained .pt here (see step 3)
└── data_prieur/
    └── boulder2024/         # seg dataset (images + labels) — rsync this
    # boulder2024_obb/  is REBUILT on Sherlock (step 4); do NOT rsync it (Windows junctions)
```

## Transfer — two options
**A) OnDemand (browser Files app + web Shell)** — the recommended path here. Browser uploads choke
on thousands of small files, so upload **two tarballs** (built by the laptop) and extract them in
the OnDemand Shell. This bundles the code + all three weights + the dataset, so steps 1–4 below
collapse into "upload 2 files, run 2 `tar` commands". See **"OnDemand quickstart"** at the bottom.

**B) rsync/scp from a laptop terminal** — steps 1–4 below. Use if you have `rsync`/`ssh` set up.

## Step 1 — get the code up (option B)
```bash
# from the laptop, in C:/Users/brian/Documents/PhD/BoulderNet
rsync -avz --exclude data/ YOLOv8-BeyondEarth/ bamaro@dtn.sherlock.stanford.edu:BoulderNet/YOLOv8-BeyondEarth/
```
(`--exclude data/` keeps the gitignored local weights/runs off the transfer; we stage weights
explicitly in step 3.)

## Step 2 — build the env (in a GPU dev session, which has internet + CUDA)
```bash
sh_dev -p gpu -G 1 -t 1:00:00
cd ~/BoulderNet
bash YOLOv8-BeyondEarth/sherlock/setup_sherlock_env.sh   # venv + ultralytics==8.4.75, prints CUDA + both YOLO26 heads
```

## Step 3 — pre-stage weights (Sherlock batch nodes may lack outbound internet)
Upload the pretrained checkpoints we already fetched locally, so training never depends on a
download mid-job. Also upload `yolo26n.pt` — ultralytics' AMP self-check downloads it, which fails
on an offline node:
```bash
# from the laptop (all three live in the gitignored data/weights/ locally)
scp YOLOv8-BeyondEarth/data/weights/yolo26m-seg.pt \
    YOLOv8-BeyondEarth/data/weights/yolo26m-obb.pt \
    YOLOv8-BeyondEarth/data/weights/yolo26n.pt \
    bamaro@dtn.sherlock.stanford.edu:BoulderNet/YOLOv8-BeyondEarth/data/weights/
# then on Sherlock: put yolo26n.pt where the AMP self-check looks (the job's cwd)
# ln -s data/weights/yolo26n.pt ~/BoulderNet/YOLOv8-BeyondEarth/yolo26n.pt
```
(If the `gpu` batch nodes *do* have proxy internet, you can skip this — ultralytics will fetch them.
Confirm with a 1-tile probe before the 36 h run.)

## Step 4 — stage the dataset
```bash
rsync -avz data_prieur/boulder2024/ bamaro@dtn.sherlock.stanford.edu:BoulderNet/data_prieur/boulder2024/
```
The OBB label set (`boulder2024_obb/`) is **rebuilt on Sherlock** — `train_h.sbatch` runs
`build_obb_labels.py` if it's missing, and on Linux that creates real `images/` **symlinks**
(the Windows junctions don't transfer). ~10 s, CPU.

## Step 5 — train
```bash
cd ~/BoulderNet
sbatch YOLOv8-BeyondEarth/sherlock/train_h.sbatch                          # both variants, imgsz 1024, 200 ep
# or a fast throughput/scaling probe first:
sbatch --export=ALL,EPOCHS=3,NAME=probe,VARIANT=obb YOLOv8-BeyondEarth/sherlock/train_h.sbatch
```
Tune `BATCH` to the GPU: `train_h.sbatch` requests `GPU_MEM:24GB` and defaults `BATCH=16`. On a
16 GB card drop to `--export=ALL,BATCH=8` and relax the constraint to `GPU_MEM:16GB`. If seg still
OOMs, the `mask_ratio` comparison arm (4 vs 1) is the lever (see the plan).

## Step 6 — retrieve
```bash
# from the laptop
scp -r bamaro@dtn.sherlock.stanford.edu:'$SCRATCH/bouldernet/h_training/h_full_*/weights/best.pt' ./
```
Then run the validation card locally (detection parity vs the stock baseline, OBB §4c-null,
synthetic-truth |Δangle|) per `docs/solutions/H_yolo26_obb_retrain.md`.

## Notes
- **No rotation aug** (`degrees=0`) — controlled comparison; architecture is the only variable vs
  the stock `yolov8m-seg` baseline (0.762 box / 0.467 mask mAP50). Rotation aug and higher imgsz
  are deferred recipe-change arms.
- **Checkpointing:** ultralytics writes `last.pt` each epoch; a pre-empted/timed-out job resumes
  with `YOLO('.../last.pt').train(resume=True)` — add a resume path if 36 h isn't enough (set the
  real `--time` from the `EPOCHS=3` probe: per-epoch × 200 + val, ~30% headroom).
- **Detection parity must be measured in-env** (same ultralytics version); never compare
  `model.val()` numbers across ultralytics versions.
- **Headless OpenCV:** ultralytics pulls `opencv-python`, which needs `libGL.so.1` (absent on
  Sherlock's headless nodes → `ImportError: libGL.so.1`). `setup_sherlock_env.sh` swaps in
  `opencv-python-headless`. If you hit it on an existing env: `pip uninstall -y opencv-python &&
  pip install --only-binary=:all: opencv-python-headless`.

## OnDemand quickstart (browser Files app + web Shell)
The laptop builds two tarballs (in `BoulderNet/sherlock_upload/`): `bnet_code.tgz` (code + the three
`yolo26m*` weights) and `boulder2024.tar` (the seg dataset). Rebuild them anytime with:
```bash
# laptop, from C:/Users/brian/Documents/PhD/BoulderNet
mkdir -p sherlock_upload
tar czf sherlock_upload/bnet_code.tgz \
  YOLOv8-BeyondEarth/scripts YOLOv8-BeyondEarth/sherlock YOLOv8-BeyondEarth/src \
  YOLOv8-BeyondEarth/data/weights/yolo26m-seg.pt YOLOv8-BeyondEarth/data/weights/yolo26m-obb.pt \
  YOLOv8-BeyondEarth/data/weights/yolo26n.pt
tar cf  sherlock_upload/boulder2024.tar data_prieur/boulder2024
```
1. **Upload** both files via OnDemand **Files** (Dashboard → Files → Home Directory → Upload).
2. **Shell** (Dashboard → Clusters → Sherlock Shell Access) — a login node; extract + build env + submit:
```bash
mkdir -p ~/BoulderNet && cd ~/BoulderNet
tar xzf ~/bnet_code.tgz          # -> YOLOv8-BeyondEarth/{scripts,sherlock,src,data/weights}
tar xf  ~/boulder2024.tar        # -> data_prieur/boulder2024
ln -sf YOLOv8-BeyondEarth/data/weights/yolo26n.pt YOLOv8-BeyondEarth/yolo26n.pt   # AMP self-check
bash YOLOv8-BeyondEarth/sherlock/setup_sherlock_env.sh    # pip on the login node; prints cuda_available False here (fine)
sbatch --export=ALL,EPOCHS=3,VARIANT=obb,NAME=probe YOLOv8-BeyondEarth/sherlock/train_h.sbatch  # time it
sbatch YOLOv8-BeyondEarth/sherlock/train_h.sbatch         # full run once the probe's per-epoch × 200 fits --time
```
3. **Monitor:** OnDemand **Jobs → Active Jobs**, or `squeue -u $USER` / `tail -f YOLOv8-BeyondEarth/logs/*.out` in the Shell.
4. **Download** results via **Files**: browse to `$SCRATCH/bouldernet/h_training/h_full_{seg,obb}/weights/best.pt` and Download.
Note: env build runs on the login node, so its smoke prints `cuda_available False` — expected; the
real GPU check is in the sbatch job log (`nvidia-smi` at the top).
