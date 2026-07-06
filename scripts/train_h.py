"""Solution H training entry point — YOLO26-m seg / OBB retrain on boulder2024.

Replicates the production recipe (recovered from src/YOLOv8BeyondEarth/training.py) at **m scale**
with the architecture as the only variable vs the stock yolov8m-seg baseline: imgsz 1024, 200 ep,
Adam lr0=1e-3, mask_ratio=1, **degrees=0** (R-A: rotation aug deferred to a follow-up arm), fliplr
0.5, mosaic 1.0, scale 0.5, translate 0.1, max_det 2000, workers=0 (Windows-safe). Same recipe for
seg and obb so the two H variants are directly comparable.

Runs the same way locally (smoke) and on a cluster (full) — only --epochs/--batch/--fraction change.

Examples:
    # local smoke test (fast, validates the full train->val->checkpoint path on a data fraction):
    python scripts/train_h.py --variant both --epochs 3 --fraction 0.05 --batch 4 --name smoke
    # full production run (e.g. on Sherlock):
    python scripts/train_h.py --variant seg --epochs 200 --batch 4 --name h_seg_full
"""
import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = {
    "seg": REPO / "data_prieur" / "boulder2024" / "boulder2024_local.yaml",
    "obb": REPO / "data_prieur" / "boulder2024_obb" / "boulder2024_obb.yaml",
}
WEIGHTS = {
    "seg": REPO / "YOLOv8-BeyondEarth" / "data" / "weights" / "yolo26m-seg.pt",
    "obb": REPO / "YOLOv8-BeyondEarth" / "data" / "weights" / "yolo26m-obb.pt",
}
PROJECT = REPO / "YOLOv8-BeyondEarth" / "data" / "h_training"   # gitignored (data/)


def train_one(variant, args):
    from ultralytics import YOLO
    task = "segment" if variant == "seg" else "obb"
    weights = str(WEIGHTS[variant]) if WEIGHTS[variant].exists() else f"yolo26m-{variant}.pt"
    model = YOLO(weights, task=task)
    name = f"{args.name}_{variant}"
    project = args.project or str(PROJECT)     # override to $SCRATCH on a cluster
    print(f"\n=== training H-{variant} ({task}) | weights={weights} | data={DATA[variant].name} "
          f"| epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} fraction={args.fraction} "
          f"| project={project} name={name} ===")
    return model.train(
        data=str(DATA[variant]),
        task=task,
        project=project,
        name=name,
        exist_ok=True,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fraction,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,                 # Windows spawn-safe; matches the original recipe
        optimizer="Adam",
        lr0=0.001,
        patience=0,                # disable early stopping (match original)
        save=True,
        val=True,
        plots=True,
        seed=0,
        amp=True,                  # 8 GB VRAM headroom at imgsz 1024
        close_mosaic=10,
        mask_ratio=1,              # seg-only knob; ignored by the obb task
        max_det=2000,
        # augmentation (recipe): NO rotation (R-A deferred)
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.4,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["seg", "obb", "both"], default="both")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of the train set (smoke: <1)")
    p.add_argument("--imgsz", type=int, default=1024, help="1024 = production; drop to 640 for an 8GB smoke")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=None, help="output dir (default local data/h_training; set to $SCRATCH on a cluster)")
    p.add_argument("--name", default="h")
    args = p.parse_args()
    variants = ["seg", "obb"] if args.variant == "both" else [args.variant]
    for v in variants:
        train_one(v, args)


if __name__ == "__main__":
    main()
