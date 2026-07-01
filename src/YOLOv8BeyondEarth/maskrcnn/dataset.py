"""YOLO-seg -> torchvision Mask R-CNN dataset (Test 3).

Reads the ``boulder2024`` dataset in Ultralytics YOLO **segmentation** format — one ``.txt`` per
image under ``labels/`` mirroring ``images/``, each line::

    <class> xn1 yn1 xn2 yn2 ... xnk ynk        # normalized [0,1] polygon ring, one instance/line

(class 0 = boulder) — and yields ``(image, target)`` pairs in the format torchvision detection
models expect (https://pytorch.org/vision/stable/models/mask_rcnn.html):

    image  : FloatTensor [3, H, W] in [0, 1]
    target : dict(boxes [N,4] xyxy float32, labels [N] int64, masks [N,H,W] uint8,
                  image_id int64, area [N] float32, iscrowd [N] int64)

**Class indexing:** torchvision reserves label 0 for background, so YOLO class 0 (boulder) maps to
label **1**; a model built for this dataset needs ``num_classes = 2``.

**Why this exists:** Test 3 (``docs/hirise_tests/test3_different_model.md``) trains a *different
architecture* (Mask R-CNN) on the *same data* our YOLOv8-seg used, to decide whether the ~135 deg
orientation lock is YOLO-specific (H_seg) or in the pixels. It also seeds the gated retraining goal
(``docs/model_retraining_plan.md``).

**GT rasterization note:** ground-truth masks are rasterized from the polygons for *training only*.
The orientation test measures Mask R-CNN's *predicted* masks at inference — never these GT
rasterizations — so an ordinary polygon fill is fine here; the never-MRR orientation rule applies to
the measurement step (``orientation.py``), not to ground-truth generation.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.draw import polygon as sk_polygon
from torch.utils.data import Dataset

# YOLO class id -> torchvision label (0 is background in torchvision detection).
BOULDER_LABEL = 1
NUM_CLASSES = 2  # background + boulder


def _parse_yolo_seg_label(txt, width, height):
    """Parse a YOLO-seg label file's text into a list of (N,2) float polygon arrays in *pixel*
    coordinates (col=x, row=y). Skips malformed / degenerate (<3-vertex) lines."""
    polygons = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 (x,y) pairs
            continue
        coords = np.asarray(parts[1:], dtype=np.float64)
        coords = coords[: (coords.size // 2) * 2].reshape(-1, 2)  # drop any stray odd value
        if coords.shape[0] < 3:
            continue
        xy = np.empty_like(coords)
        xy[:, 0] = coords[:, 0] * width   # x <- normalized by width
        xy[:, 1] = coords[:, 1] * height  # y <- normalized by height
        polygons.append(xy)
    return polygons


def _polygons_to_target(polygons, width, height, image_id):
    """Rasterize polygons to a torchvision target dict. Instances whose filled mask is empty
    (tiny polygons that round away) or whose box is degenerate are dropped."""
    masks, boxes = [], []
    for xy in polygons:
        rr, cc = sk_polygon(xy[:, 1], xy[:, 0], shape=(height, width))  # (row=y, col=x)
        if rr.size == 0:
            continue
        m = np.zeros((height, width), dtype=np.uint8)
        m[rr, cc] = 1
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        if x2 <= x1 or y2 <= y1:  # need a positive-area box (torchvision asserts w,h > 0)
            continue
        masks.append(m)
        boxes.append([x1, y1, x2 + 1, y2 + 1])  # xyxy, +1 so a 1-px extent is width 1, not 0

    n = len(masks)
    if n == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "masks": torch.zeros((0, height, width), dtype=torch.uint8),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }
    boxes_t = torch.as_tensor(np.asarray(boxes), dtype=torch.float32)
    return {
        "boxes": boxes_t,
        "labels": torch.full((n,), BOULDER_LABEL, dtype=torch.int64),
        "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
        "iscrowd": torch.zeros((n,), dtype=torch.int64),
    }


class BoulderYOLODataset(Dataset):
    """A single YOLO-seg split (``images/`` + ``labels/``) as a torchvision detection dataset.

    Parameters
    ----------
    split_dir : path to a split folder containing ``images/`` and ``labels/`` (e.g.
        ``data_prieur/boulder2024/train``). The yaml's ``D:/BOULDERING/...`` paths are ignored;
        point directly at the local split folder.
    transforms : optional callable ``(image, target) -> (image, target)`` applied after loading
        (e.g. augmentation). Must keep ``image`` a [3,H,W] float tensor and ``target`` boxes/masks
        consistent. ``None`` = no augmentation.
    image_glob : image filename pattern (default ``*.png``).

    Notes
    -----
    - Grayscale ('L') patches are broadcast to 3 channels (Mask R-CNN's ResNet backbone is 3-ch).
    - Images with an empty / missing label file become valid *negative* samples (0 instances),
      which recent torchvision detection models accept.
    """

    def __init__(self, split_dir, transforms=None, image_glob="*.png"):
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"no images/ under {self.split_dir}")
        self.image_paths = sorted(self.images_dir.glob(image_glob))
        if not self.image_paths:
            raise FileNotFoundError(f"no images matching {image_glob!r} in {self.images_dir}")
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def _label_path(self, image_path):
        return self.labels_dir / (image_path.stem + ".txt")

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)  # H,W,3
        height, width = arr.shape[:2]
        image = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0

        label_path = self._label_path(image_path)
        txt = label_path.read_text().strip() if label_path.is_file() else ""
        polygons = _parse_yolo_seg_label(txt, width, height) if txt else []
        target = _polygons_to_target(polygons, width, height, image_id=idx)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def detection_collate_fn(batch):
    """Collate for torchvision detection: keep per-sample (image, target) as tuples (images have
    the same size here, but targets have variable instance counts, so we do not stack)."""
    return tuple(zip(*batch))


def build_splits(dataset_root, transforms=None, train_transforms=None):
    """Build the ``train`` / ``validation`` / ``test`` datasets from a ``boulder2024``-style root.

    ``train_transforms`` (augmentation) is applied to the train split only; ``transforms`` applies
    to val/test (usually ``None``). Missing split folders are skipped and returned as ``None``.
    """
    root = Path(dataset_root)
    out = {}
    for split in ("train", "validation", "test"):
        split_dir = root / split
        if not (split_dir / "images").is_dir():
            out[split] = None
            continue
        tf = train_transforms if split == "train" else transforms
        out[split] = BoulderYOLODataset(split_dir, transforms=tf)
    return out["train"], out["validation"], out["test"]
