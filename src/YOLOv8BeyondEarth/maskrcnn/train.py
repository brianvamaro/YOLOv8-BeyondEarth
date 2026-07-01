"""Train a torchvision Mask R-CNN on the boulder2024 YOLO-seg data (Test 3, step 2).

A pure-PyTorch alternative segmenter (no compiled ops → installs cleanly on cu128/Blackwell) used
to test whether the HiRISE ~135 deg orientation lock is YOLO-specific (H_seg) or in the pixels
(``docs/hirise_tests/test3_different_model.md``). Also the reusable retraining scaffold for the
gated model-robustness goal (``docs/model_retraining_plan.md``).

Design notes
------------
- **Same data, different architecture.** Trains on the exact ``boulder2024`` split our YOLOv8-seg
  used (``BoulderYOLODataset``); only the model changes.
- **AMP** (autocast + GradScaler) keeps activations in fp16 — the deciding factor on the 5070's
  8 GB. Disabled automatically on CPU.
- **Augmentation** defaults to random horizontal flip (p=0.5), mirroring YOLO's ``fliplr=0.5`` so
  the recipe difference is smaller. Flips are orientation-symmetrising, so they do not bias the
  eventual 45/135 measurement.
- **Checkpointing** on validation loss (torchvision detection models only return losses in
  ``train()`` mode, so val loss is computed under ``train()`` + ``no_grad``).
- **Sleep-robust.** Checkpoints carry optimizer/scheduler/scaler/epoch/history so ``fit(resume=...)``
  restarts mid-run losing only the current epoch; and the loop asks Windows not to idle-sleep
  (auto-released on exit). Neither fully prevents a hard lid-close sleep — resume is the safety net.
"""
import contextlib
import ctypes
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from YOLOv8BeyondEarth.maskrcnn.dataset import (
    NUM_CLASSES,
    build_splits,
    detection_collate_fn,
)


# ---------------------------------------------------------------------- keep-awake
@contextlib.contextmanager
def keep_system_awake(enabled=True):
    """Ask Windows not to idle-sleep (or spin down) while a long job runs; auto-released on exit.

    Uses ``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` — prevents *automatic idle*
    sleep (the unattended case) but NOT a manual lid-close/sleep. The display is left free to sleep.
    No-op off Windows or if the call is unavailable. Not a substitute for ``fit(resume=...)``.
    """
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    active = False
    if enabled:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            active = True
        except (AttributeError, OSError):
            active = False
    try:
        yield active
    finally:
        if active:
            with contextlib.suppress(AttributeError, OSError):
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


# --------------------------------------------------------------------------- model
def build_maskrcnn(num_classes=NUM_CLASSES, weights="DEFAULT", trainable_backbone_layers=3,
                   mask_hidden=256):
    """Mask R-CNN R50-FPN with COCO-pretrained backbone, box + mask heads re-sized to
    ``num_classes`` (background + boulder = 2). ``weights=None`` gives random init (offline/testing).
    """
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_feats_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_mask, mask_hidden, num_classes)
    return model


# ----------------------------------------------------------------------- transforms
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Flip image, boxes and masks together (p). Works on the dataset's tensor format:
    image [3,H,W] float, target masks [N,H,W] uint8, boxes [N,4] xyxy."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if float(torch.rand(1)) >= self.p:
            return image, target
        w = image.shape[-1]
        image = image.flip(-1)
        if target["masks"].numel():
            target["masks"] = target["masks"].flip(-1)
        boxes = target["boxes"]
        if boxes.numel():
            x1 = boxes[:, 0].clone()
            boxes[:, 0] = w - boxes[:, 2]
            boxes[:, 2] = w - x1
            target["boxes"] = boxes
        return image, target


def default_train_transforms():
    return Compose([RandomHorizontalFlip(0.5)])


# --------------------------------------------------------------------------- engine
def _to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(model, optimizer, loader, device, scaler, epoch, log_every=50, warmup=True):
    model.train()
    amp = scaler.is_enabled()
    lr_sched = None
    if warmup and epoch == 0:  # linear LR warmup over the first epoch (torchvision reference)
        warmup_iters = min(1000, len(loader) - 1)
        if warmup_iters > 0:
            lr_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, total_iters=warmup_iters)
    running = 0.0
    t0 = time.time()
    for i, (images, targets) in enumerate(loader):
        images, targets = _to_device(images, targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if lr_sched is not None:
            lr_sched.step()
        running += float(loss)
        if log_every and (i % log_every == 0):
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch} it {i}/{len(loader)} loss {float(loss):.4f} lr {lr:.2e} "
                  f"({(time.time()-t0)/(i+1):.2f}s/it)")
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Mean validation loss. torchvision detection returns losses only in train() mode, so we
    keep train() but disable grad — no BN/label leakage of concern for checkpoint selection."""
    model.train()
    total, n = 0.0, 0
    for images, targets in loader:
        images, targets = _to_device(images, targets, device)
        loss_dict = model(images, targets)
        total += float(sum(loss_dict.values()))
        n += 1
    return total / max(1, n)


def fit(dataset_root, out_dir, epochs=20, batch_size=2, lr=0.005, momentum=0.9,
        weight_decay=5e-4, num_workers=4, device=None, weights="DEFAULT",
        trainable_backbone_layers=3, aug=True, subset=None, seed=0, log_every=50,
        resume=False, prevent_sleep=True):
    """Train Mask R-CNN on ``dataset_root`` (a boulder2024-style root with train/validation/test).

    Saves ``last.pt`` and ``best.pt`` (min val loss) to ``out_dir`` after every epoch. Returns a
    history dict. ``subset`` (int) trims the train/val splits for a quick smoke run.

    Sleep-robustness
    ----------------
    - ``resume``: ``True`` (or a checkpoint path) continues from ``last.pt`` — restores
      model/optimizer/scheduler/scaler/epoch/best-val/history and runs the remaining epochs. Use
      this after a laptop sleep, crash, or reboot interrupts a run; only the in-progress epoch is
      lost. ``resume=True`` with no checkpoint present starts fresh.
    - ``prevent_sleep``: ask Windows not to *idle*-sleep during the run (released on return). Does
      not stop a manual lid-close sleep — ``resume`` is the safety net for that.
    """
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = default_train_transforms() if aug else None
    train_ds, val_ds, _ = build_splits(dataset_root, train_transforms=train_tf)
    if train_ds is None:
        raise FileNotFoundError(f"no train/ split under {dataset_root}")
    if subset:
        train_ds = Subset(train_ds, range(min(subset, len(train_ds))))
        if val_ds is not None:
            val_ds = Subset(val_ds, range(min(subset, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=detection_collate_fn,
                              pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=detection_collate_fn,
                                pin_memory=(device.type == "cuda"),
                                persistent_workers=num_workers > 0)

    model = build_maskrcnn(weights=weights, trainable_backbone_layers=trainable_backbone_layers)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    milestones = [int(0.7 * epochs), int(0.9 * epochs)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    config = dict(dataset_root=str(dataset_root), epochs=epochs, batch_size=batch_size, lr=lr,
                  weight_decay=weight_decay, weights=str(weights),
                  trainable_backbone_layers=trainable_backbone_layers, aug=aug, subset=subset,
                  num_classes=NUM_CLASSES)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    start_epoch = 0

    # --- resume from an interrupted run (sleep / crash / reboot) -------------------
    resume_path = Path(resume) if isinstance(resume, (str, Path)) else (out_dir / "last.pt")
    if resume and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        history = ckpt.get("history", history)
        best_val = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {resume_path} at epoch {start_epoch} (best_val {best_val:.4f})")
    elif resume:
        print(f"resume requested but no checkpoint at {resume_path}; starting fresh")

    with keep_system_awake(prevent_sleep and device.type == "cuda"):
        for epoch in range(start_epoch, epochs):
            tl = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch, log_every)
            scheduler.step()
            vl = evaluate_loss(model, val_loader, device) if val_loader is not None else float("nan")
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            print(f"[epoch {epoch}] train_loss {tl:.4f}  val_loss {vl:.4f}")
            ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "val_loss": vl, "best_val": min(best_val, vl) if vl == vl else best_val,
                    "history": history, "config": config}
            torch.save(ckpt, out_dir / "last.pt")
            if val_loader is None or vl < best_val:
                best_val = vl
                ckpt["best_val"] = best_val
                torch.save(ckpt, out_dir / "best.pt")
    history["best_val_loss"] = best_val
    history["out_dir"] = str(out_dir)
    return history
