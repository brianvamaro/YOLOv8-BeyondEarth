"""COCO AP evaluation for the Mask R-CNN sanity check (Test 3, step 2).

A learning gate: before we trust Mask R-CNN's *masks* for the orientation comparison, confirm it
actually detects boulders. Reports standard COCO bbox + segm AP on a held-out split. Uses
``pycocotools`` (already in the env). This is a detection-quality check, NOT the orientation test —
the orientation test lives in ``orientation.py`` and runs on full-scene inference.
"""
import contextlib
import io

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from YOLOv8BeyondEarth.maskrcnn.dataset import BOULDER_LABEL


def _mask_to_rle(mask_uint8):
    rle = mask_util.encode(np.asfortranarray(mask_uint8))
    rle["counts"] = rle["counts"].decode("ascii")  # JSON-serializable
    return rle


def _build_coco_gt(dataset):
    """Assemble a COCO ground-truth object from a BoulderYOLODataset(-like) dataset."""
    images, annotations = [], []
    ann_id = 1
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        img_id = int(target["image_id"])
        masks = target["masks"].numpy()
        h, w = (masks.shape[1], masks.shape[2]) if masks.shape[0] else _probe_hw(dataset, idx)
        images.append({"id": img_id, "height": int(h), "width": int(w)})
        boxes = target["boxes"].numpy()
        for j in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[j]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": BOULDER_LABEL,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float((x2 - x1) * (y2 - y1)),
                "segmentation": _mask_to_rle(masks[j].astype(np.uint8)),
                "iscrowd": 0,
            })
            ann_id += 1
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations,
                    "categories": [{"id": BOULDER_LABEL, "name": "boulder"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _probe_hw(dataset, idx):
    img, _ = dataset[idx]
    return img.shape[-2], img.shape[-1]


@torch.no_grad()
def _predict_coco(model, dataset, device, batch_size, num_workers, score_thresh, mask_thresh):
    from torch.utils.data import DataLoader
    from YOLOv8BeyondEarth.maskrcnn.dataset import detection_collate_fn

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=detection_collate_fn)
    results = []
    seen = 0
    for images, targets in loader:
        image_ids = [int(t["image_id"]) for t in targets]
        images = [img.to(device) for img in images]
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
        for img_id, out in zip(image_ids, outputs):
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            masks = out["masks"].cpu().numpy()  # [N,1,H,W] float prob
            for k in range(boxes.shape[0]):
                if scores[k] < score_thresh:
                    continue
                x1, y1, x2, y2 = boxes[k]
                m = (masks[k, 0] >= mask_thresh).astype(np.uint8)
                results.append({
                    "image_id": img_id, "category_id": BOULDER_LABEL, "score": float(scores[k]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "segmentation": _mask_to_rle(m),
                })
        seen += len(images)
    return results


def evaluate_coco_ap(model, dataset, device=None, batch_size=2, num_workers=4,
                     score_thresh=0.05, mask_thresh=0.5, iou_types=("bbox", "segm")):
    """Run COCO evaluation; return {iou_type: {AP, AP50, AP75}}. Prints the full COCO summary."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    coco_gt = _build_coco_gt(dataset)
    results = _predict_coco(model, dataset, device, batch_size, num_workers,
                            score_thresh, mask_thresh)
    out = {}
    if not results:
        print("evaluate_coco_ap: model produced 0 detections above threshold.")
        return {t: {"AP": 0.0, "AP50": 0.0, "AP75": 0.0} for t in iou_types}
    coco_dt = coco_gt.loadRes(results)
    for iou_type in iou_types:
        ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        ev.evaluate()
        ev.accumulate()
        print(f"\n== COCO {iou_type} ==")
        ev.summarize()
        out[iou_type] = {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]),
                         "AP75": float(ev.stats[2])}
    return out
