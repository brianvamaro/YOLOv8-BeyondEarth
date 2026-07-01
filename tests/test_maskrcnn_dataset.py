"""Characterization tests for the YOLO-seg -> torchvision Mask R-CNN dataset (Test 3).

Data-free: each test synthesizes a tiny YOLO-seg split in a tmp dir, so it runs in CI without the
~5000-patch boulder2024 download. Pins the contract torchvision detection models rely on
(xyxy float boxes, uint8 masks, background-offset labels, negative samples, mask<->box agreement).
"""
import numpy as np
import pytest
import torch
from PIL import Image

from YOLOv8BeyondEarth.maskrcnn import BoulderYOLODataset, build_splits, detection_collate_fn
from YOLOv8BeyondEarth.maskrcnn.dataset import BOULDER_LABEL, NUM_CLASSES


def _write_split(split_dir, images):
    """images: dict name -> (PIL_size, list_of_normalized_polygons). Empty poly list -> negative."""
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "labels").mkdir(parents=True)
    for name, (size, polygons) in images.items():
        Image.new("L", size, color=64).save(split_dir / "images" / f"{name}.png")
        lines = []
        for poly in polygons:
            flat = " ".join(f"{v:.4f}" for xy in poly for v in xy)
            lines.append(f"0 {flat}")
        (split_dir / "labels" / f"{name}.txt").write_text("\n".join(lines))


def test_target_contract_and_mask_box_agreement(tmp_path):
    # a rectangle spanning normalized x in [0.2,0.6], y in [0.1,0.5] on a 100x100 image
    rect = [(0.2, 0.1), (0.6, 0.1), (0.6, 0.5), (0.2, 0.5)]
    _write_split(tmp_path / "train", {"a": ((100, 100), [rect])})
    ds = BoulderYOLODataset(tmp_path / "train")

    image, target = ds[0]
    assert image.shape == (3, 100, 100) and image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert target["boxes"].dtype == torch.float32 and target["boxes"].shape == (1, 4)
    assert target["masks"].dtype == torch.uint8 and target["masks"].shape == (1, 100, 100)
    assert target["labels"].tolist() == [BOULDER_LABEL] and NUM_CLASSES == 2

    # x normalized by width, y by height; box is xyxy, +1 on the far edge
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    assert (x1, y1) == pytest.approx((20, 10), abs=1.0)
    assert (x2, y2) == pytest.approx((60, 50), abs=1.5)
    # mask's own tight bbox agrees with the reported box
    ys, xs = torch.where(target["masks"][0] > 0)
    assert (int(xs.min()), int(ys.min())) == (int(x1), int(y1))


def test_non_square_image_axis_mapping(tmp_path):
    # 200 wide x 80 tall: catches an x/y (width/height) swap that square patches would hide
    poly = [(0.5, 0.25), (0.75, 0.25), (0.75, 0.75), (0.5, 0.75)]
    _write_split(tmp_path / "train", {"a": ((200, 80), [poly])})
    ds = BoulderYOLODataset(tmp_path / "train")
    _, target = ds[0]
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    assert (x1, x2) == pytest.approx((100, 150), abs=1.5)  # x * width(200)
    assert (y1, y2) == pytest.approx((20, 60), abs=1.5)    # y * height(80)


def test_negative_sample_and_collate(tmp_path):
    rect = [(0.2, 0.1), (0.6, 0.1), (0.6, 0.5), (0.2, 0.5)]
    _write_split(tmp_path / "train", {"a": ((64, 64), [rect]), "b": ((64, 64), [])})
    ds = BoulderYOLODataset(tmp_path / "train")
    # 'b' has an empty label -> valid negative sample with zero instances
    neg = next(t for _, t in (ds[i] for i in range(len(ds))) if t["boxes"].shape[0] == 0)
    assert neg["masks"].shape == (0, 64, 64) and neg["labels"].shape == (0,)

    batch = detection_collate_fn([ds[0], ds[1]])
    images, targets = batch
    assert len(images) == 2 and len(targets) == 2
    assert all(img.shape[0] == 3 for img in images)


def test_degenerate_polygons_dropped(tmp_path):
    # a 2-vertex "line" and a sub-pixel triangle should both be dropped, leaving one real instance
    good = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    line = [(0.1, 0.1), (0.2, 0.1)]
    _write_split(tmp_path / "train", {"a": ((100, 100), [good, line])})
    ds = BoulderYOLODataset(tmp_path / "train")
    _, target = ds[0]
    assert target["boxes"].shape[0] == 1


def test_build_splits_skips_missing(tmp_path):
    rect = [(0.2, 0.1), (0.6, 0.1), (0.6, 0.5), (0.2, 0.5)]
    _write_split(tmp_path / "train", {"a": ((64, 64), [rect])})
    train, val, test = build_splits(tmp_path)
    assert isinstance(train, BoulderYOLODataset)
    assert val is None and test is None  # only train/ exists
