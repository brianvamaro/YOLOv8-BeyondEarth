"""The mask downscale moved to the device — it must stay bit-identical to the CPU it replaced.

``_downscale_masks_on_device`` exists purely for speed (a third of the production run: see its
docstring), so the only thing that matters is that it changes nothing. These pin both halves of
that: the values equal ``cv2.INTER_AREA`` exactly, and every configuration outside the exact
integer-ratio guarantee falls through untouched so the CPU path still handles it.
"""
import cv2
import numpy as np
import pytest
import torch

from YOLOv8BeyondEarth.polygon import binary_mask_to_polygon
from YOLOv8BeyondEarth.predict import _downscale_masks_on_device


def _hard_masks(n, size, seed=0):
    """Thresholded {0, 1} float32 masks with blob-like structure, as the model path produces."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    out = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.integers(0, size, 2)
            r = rng.integers(size // 64 or 1, size // 6)
            out[i][(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1.0
    return out


@pytest.mark.parametrize("imgsz,slice_size", [(1024, 256), (1024, 512), (640, 320), (512, 128)])
def test_device_pooling_equals_cv2_inter_area_exactly(imgsz, slice_size):
    masks = _hard_masks(6, imgsz, seed=slice_size)
    pooled = _downscale_masks_on_device(torch.from_numpy(masks), slice_size, True).numpy()
    expected = np.stack([cv2.resize(m, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                         for m in masks])
    assert pooled.shape == expected.shape
    # not allclose: the box mean of a {0, 1} mask over a power-of-two block is a dyadic rational,
    # so both implementations land on the same float32 and any drift is a real behaviour change
    assert np.array_equal(pooled, expected)


def test_traced_polygons_are_unchanged():
    """Values are the contract, but geometry is what the catalogs are made of."""
    masks = _hard_masks(8, 1024, seed=7)
    pooled = _downscale_masks_on_device(torch.from_numpy(masks), 256, True).numpy()
    expected = np.stack([cv2.resize(m, (256, 256), interpolation=cv2.INTER_AREA) for m in masks])
    for a, b in zip(pooled, expected):
        pa, pb = binary_mask_to_polygon(a), binary_mask_to_polygon(b)
        assert (pa is None) == (pb is None)
        if pa is not None:
            assert np.array_equal(pa, pb)


@pytest.mark.parametrize("masks_shape,slice_size,downscale_pred", [
    ((3, 1024, 1024), 256, False),   # downscale_pred=False keeps the model frame on purpose (G3-10)
    ((3, 1000, 1000), 256, True),    # not an integer ratio
    ((3, 256, 256), 256, True),      # already at slice size
    ((3, 128, 128), 256, True),      # smaller than slice size (an upscale, not ours to do)
    ((3, 1024, 512), 256, True),     # non-square
])
def test_ineligible_inputs_pass_through_to_the_cpu_path(masks_shape, slice_size, downscale_pred):
    masks = torch.zeros(masks_shape)
    out = _downscale_masks_on_device(masks, slice_size, downscale_pred)
    assert out.shape == masks.shape
    assert out is masks


def test_a_fractional_mask_is_left_alone_only_by_shape_not_by_value():
    """The exactness argument assumes hard {0, 1} input; the guard is shape-based, so document it.

    A caller that hands in an already-fractional mask still gets a correct box mean — averaging is
    linear — it simply no longer has the dyadic-exactness guarantee. Pinned so the distinction is
    not lost if someone later reuses this helper on re-masker output.
    """
    rng = np.random.default_rng(3)
    masks = rng.random((2, 1024, 1024), dtype=np.float32)
    pooled = _downscale_masks_on_device(torch.from_numpy(masks), 256, True).numpy()
    expected = np.stack([cv2.resize(m, (256, 256), interpolation=cv2.INTER_AREA) for m in masks])
    assert pooled.shape == expected.shape
    np.testing.assert_allclose(pooled, expected, atol=1e-6)
