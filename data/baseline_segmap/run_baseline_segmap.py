# Baseline seg-mAP row for the Goal-2 solution benchmark cards (docs/solutions/README.md).
# Stock yolov8-m-boulder weights, boulder2024 val split, benchmark.seg_map_eval defaults.
# __main__ guard required: ultralytics dataloader workers use Windows spawn and re-import this file.
import json, sys, time

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

WEIGHTS = r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\weights\yolov8-m-boulder-detection-tmp.pt"
DATA = r"c:\Users\brian\Documents\PhD\BoulderNet\data_prieur\boulder2024\boulder2024_local.yaml"
OUT = r"C:\Users\brian\AppData\Local\Temp\claude\c--Users-brian-Documents-PhD-BoulderNet\acb87663-1934-4270-a4b4-abb08b8fa6ad\scratchpad\baseline_segmap.json"

if __name__ == "__main__":
    from YOLOv8BeyondEarth.benchmark import seg_map_eval

    t0 = time.time()
    res = seg_map_eval(WEIGHTS, DATA, split="val", imgsz=1024, device=0)
    res["wall_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
