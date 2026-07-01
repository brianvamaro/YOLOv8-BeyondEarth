"""torchvision Mask R-CNN train -> infer pipeline (Test 3: different architecture, same data).

A pure-PyTorch (no compiled ops) alternative segmenter used to test whether the HiRISE ~135 deg
long-axis orientation peak is YOLO-specific (H_seg) or lives in the pixels — see
``docs/hirise_tests/test3_different_model.md``. Also the reusable retraining scaffold for the gated
model-robustness goal (``docs/model_retraining_plan.md``).
"""
from YOLOv8BeyondEarth.maskrcnn.dataset import (
    BoulderYOLODataset,
    build_splits,
    detection_collate_fn,
)
from YOLOv8BeyondEarth.maskrcnn.train import build_maskrcnn, fit
from YOLOv8BeyondEarth.maskrcnn.coco_eval import evaluate_coco_ap
from YOLOv8BeyondEarth.maskrcnn.infer import sliced_predict, predict_scene

__all__ = [
    "BoulderYOLODataset",
    "build_splits",
    "detection_collate_fn",
    "build_maskrcnn",
    "fit",
    "evaluate_coco_ap",
    "sliced_predict",
    "predict_scene",
]
