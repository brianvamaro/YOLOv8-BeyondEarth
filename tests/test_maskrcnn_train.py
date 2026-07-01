"""Tests for the Mask R-CNN model builder + augmentation (Test 3, step 2).

Fast tests use ``weights=None`` (random init, no 170 MB COCO download). The actual
forward/backward step is marked ``slow`` (builds the full R50-FPN and runs a train step on CPU).
"""
import pytest
import torch
from PIL import Image

from YOLOv8BeyondEarth.maskrcnn.dataset import NUM_CLASSES
from YOLOv8BeyondEarth.maskrcnn.train import (
    RandomHorizontalFlip,
    build_maskrcnn,
    default_train_transforms,
    fit,
    keep_system_awake,
)


def _synthetic_sample(h=64, w=96, n=2):
    """One (image, target) pair with n axis-aligned boxes/masks — no external data."""
    image = torch.rand(3, h, w)
    boxes, masks = [], []
    for k in range(n):
        x1, y1 = 5 + 10 * k, 5 + 8 * k
        x2, y2 = x1 + 20, y1 + 15
        m = torch.zeros(h, w, dtype=torch.uint8)
        m[y1:y2, x1:x2] = 1
        masks.append(m)
        boxes.append([x1, y1, x2, y2])
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.ones(n, dtype=torch.int64),
        "masks": torch.stack(masks),
        "image_id": torch.tensor(0),
        "area": torch.ones(n),
        "iscrowd": torch.zeros(n, dtype=torch.int64),
    }
    return image, target


def test_build_maskrcnn_head_sizes():
    model = build_maskrcnn(weights=None)
    assert model.roi_heads.box_predictor.cls_score.out_features == NUM_CLASSES
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == NUM_CLASSES


def test_flip_is_involution_and_valid():
    image, target = _synthetic_sample()
    flip = RandomHorizontalFlip(p=1.0)
    orig_boxes = target["boxes"].clone()
    fimg, ftgt = flip(image.clone(), {k: v.clone() for k, v in target.items()})
    # boxes stay valid (x2 > x1) after flipping
    assert bool((ftgt["boxes"][:, 2] > ftgt["boxes"][:, 0]).all())
    # masks flipped horizontally
    assert torch.equal(ftgt["masks"], target["masks"].flip(-1))
    # flipping twice restores the original boxes (involution)
    _, dtgt = flip(fimg, ftgt)
    assert torch.allclose(dtgt["boxes"], orig_boxes, atol=1e-4)


def test_flip_p_zero_is_noop():
    image, target = _synthetic_sample()
    out_img, out_tgt = RandomHorizontalFlip(p=0.0)(image.clone(), target)
    assert torch.equal(out_img, image)
    assert torch.equal(out_tgt["boxes"], target["boxes"])


def test_default_train_transforms_runs():
    image, target = _synthetic_sample()
    tf = default_train_transforms()
    out_img, out_tgt = tf(image, target)
    assert out_img.shape == image.shape
    assert out_tgt["boxes"].shape == target["boxes"].shape


def test_keep_system_awake_is_safe():
    # must not raise and must always release, whatever the platform returns
    with keep_system_awake(enabled=True) as active:
        assert isinstance(active, bool)
    with keep_system_awake(enabled=False) as active:
        assert active is False


def _write_tiny_root(root, n_train=4, n_val=2, size=(64, 64)):
    rect = [(0.2, 0.2), (0.7, 0.2), (0.7, 0.7), (0.2, 0.7)]
    for split, n in (("train", n_train), ("validation", n_val)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            Image.new("L", size, color=100).save(root / split / "images" / f"{i}.png")
            flat = " ".join(f"{v:.4f}" for xy in rect for v in xy)
            (root / split / "labels" / f"{i}.txt").write_text(f"0 {flat}")


@pytest.mark.slow
def test_fit_resume_continues_from_checkpoint(tmp_path):
    root = tmp_path / "ds"
    _write_tiny_root(root)
    out = tmp_path / "run"
    # initial run: 1 epoch (weights=None so no download; CPU)
    h1 = fit(root, out, epochs=1, batch_size=2, num_workers=0, weights=None,
             device="cpu", log_every=0, prevent_sleep=False)
    assert (out / "last.pt").is_file() and len(h1["train_loss"]) == 1
    # resume with a higher epoch budget: should run only the remaining epoch
    h2 = fit(root, out, epochs=2, batch_size=2, num_workers=0, weights=None,
             device="cpu", log_every=0, resume=True, prevent_sleep=False)
    assert len(h2["train_loss"]) == 2  # history carried over + one more epoch


@pytest.mark.slow
def test_train_step_forward_backward():
    torch.manual_seed(0)
    model = build_maskrcnn(weights=None)
    model.train()
    images = [_synthetic_sample()[0], _synthetic_sample()[0]]
    targets = [_synthetic_sample()[1], _synthetic_sample()[1]]
    loss_dict = model(images, targets)
    expected = {"loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness",
                "loss_rpn_box_reg"}
    assert expected <= set(loss_dict)
    loss = sum(loss_dict.values())
    assert torch.isfinite(loss)
    loss.backward()  # gradients flow
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
