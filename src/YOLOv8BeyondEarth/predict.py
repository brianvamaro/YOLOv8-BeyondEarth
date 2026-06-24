import os
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from concurrent.futures import ThreadPoolExecutor

from YOLOv8BeyondEarth.polygon import (binary_mask_to_polygon, binary_mask_to_polygon_cv,
                                       is_within_slice, shift_polygon,
                                       add_geometries, bboxes_to_shp, outlines_to_shp)
from lsnms import nms, wbc

from sahi.slicing import slice_image
from tqdm import tqdm
from pathlib import Path

from rastertools_BOULDERING import raster, convert as raster_convert, metadata as raster_metadata
from shptools_BOULDERING import shp

#from torchvision.ops import (nms as nms_torch, batched_nms as batched_nms_torch)

def _result_to_df(prediction_result, detection_model, has_mask, shift_amount, slice_size,
                  min_area_threshold, downscale_pred, contour_method="skimage", executor=None):
    """Convert a single ultralytics Result (one slice) into the detections DataFrame.

    This is the per-slice core extracted from ``YOLOv8`` so the model can be called on a
    *batch* of slices (see ``get_sliced_prediction``) instead of one at a time. Masks are
    thresholded at 0.5, optionally resized to ``slice_size`` with ``INTER_AREA``, and converted
    to polygons.

    ``contour_method`` selects the mask->polygon extractor:
    - ``"skimage"`` (default): ``binary_mask_to_polygon`` (skimage ``find_contours``) — the
      original behaviour, bit-identical geometry.
    - ``"cv2"``: ``binary_mask_to_polygon_cv`` (OpenCV) — much faster, sparser vertices;
      negligible effect on orientation (see Goal-2 bridge notebook), opt-in.

    ``executor``: an optional ``ThreadPoolExecutor``. The per-detection work (resize, contour
    extraction, geometry) is independent and dominated by GIL-releasing C code
    (``find_contours``, ``cv2``, numpy), so mapping it across threads gives a large speedup on
    dense slices. Results are collected in input order, so output is identical to the serial
    path.
    """
    extract_polygon = binary_mask_to_polygon_cv if contour_method == "cv2" else binary_mask_to_polygon
    shift_x = shift_amount[0]
    shift_y = shift_amount[1]

    # if no predictions
    if prediction_result.boxes.data.size()[0] == 0:
        return pd.DataFrame(columns=['score', 'polygon', 'category_id', 'category_name', 'is_within_slice'])

    conf_mask = prediction_result.boxes.data[:, 4] >= detection_model.confidence_threshold
    result_boxes = prediction_result.boxes.data[conf_mask]
    if has_mask:
        result_masks = prediction_result.masks.data[conf_mask]
    else:
        result_masks = torch.tensor([[] for _ in range(result_boxes.size()[0])])

    # Threshold the whole mask batch in one GPU op (equivalent to the old per-mask
    # ``bool_mask[bool_mask>=0.5]=1; [<0.5]=0``) before moving to CPU. Kept as float32 {0.,1.}
    # so the later INTER_AREA downscale yields the same fractional mask the contour extractor
    # sees -> geometry-neutral, just without two full-array writes per detection.
    masks_np = (result_masks >= 0.5).to(torch.float32).cpu().detach().numpy()
    boxes_np = result_boxes.cpu().detach().numpy()

    min_edge_distance = 0.05 * slice_size
    max_edge_distance = 0.95 * slice_size

    def _process_detection(item):
        """Mask -> (score, polygon, category_id, category_name, is_within_slice) or None.
        Pure and free of shared mutable state, so it is safe to run across threads."""
        prediction, bool_mask = item
        category_id = int(prediction[5])

        if downscale_pred and bool_mask.shape[0] != slice_size:
            bool_mask = cv2.resize(bool_mask, (slice_size, slice_size), interpolation=cv2.INTER_AREA)

        # number of pixels
        area = np.count_nonzero(bool_mask == 1)
        if area <= min_area_threshold:
            return None

        try:
            polygon = extract_polygon(bool_mask)
            if polygon is None or len(polygon) < 3:
                return None
            if downscale_pred:
                polygon_slice = polygon
            else:
                polygon_slice = np.stack([(polygon[:, 0] / bool_mask.shape[0]) * slice_size,
                                          (polygon[:, 1] / bool_mask.shape[0]) * slice_size], axis=-1)
            is_polygon_within_slice = (np.logical_and(polygon_slice[:, 0].min() >= min_edge_distance,
                                                      polygon_slice[:, 0].max() <= max_edge_distance) and
                                       np.logical_and(polygon_slice[:, 1].min() >= min_edge_distance,
                                                      polygon_slice[:, 1].max() <= max_edge_distance))

            # if at edge set score to a low value
            score = prediction[4] if is_polygon_within_slice else 0.10
            shifted_polygon = shift_polygon(polygon_slice, shift_x, shift_y)
            category_name = detection_model.category_mapping[str(category_id)]
            return (score, shifted_polygon, category_id, category_name, is_polygon_within_slice)
        except Exception:
            return None

    items = list(zip(boxes_np, masks_np))
    if executor is not None and len(items) > 1:
        results = executor.map(_process_detection, items)
    else:
        results = map(_process_detection, items)

    scores = []
    polygons = []
    category_ids = []
    category_names = []
    is_polygon_within_slice_list = []
    for res in results:
        if res is None:
            continue
        scores.append(res[0])
        polygons.append(res[1])  # polygon in absolute (shifted) coordinates
        category_ids.append(res[2])
        category_names.append(res[3])
        is_polygon_within_slice_list.append(res[4])

    data = {'score': scores, 'polygon': polygons,
            'category_id': category_ids, 'category_name': category_names,
            'is_within_slice': is_polygon_within_slice_list}

    return pd.DataFrame(data)


def YOLOv8(detection_model, image, has_mask, shift_amount, slice_size, min_area_threshold,
           downscale_pred, contour_method="skimage"):
    """
    Single-image (single-slice) prediction. Kept for backward compatibility; the batched
    pipeline in ``get_sliced_prediction`` calls ``_result_to_df`` directly.

    YOLOv8 expects numpy arrays to have BGR (height, width, 3).

    1. Let's say you want to detect very very small objects, the slice height and width should be
    pretty small, and detection_model.image_size should be increased:
    - slice_height, slice_width = 256
    - detection_model.image_size = 1024.

    2. If you are in the opposite situation, where you realized that most of the large boulders are missed out. You can
    increase the slice height and width.
    - slice_height, slice_width = 1024
    - detection_model.image_size = 512, 1024.

    You can get the best of both worlds by combining predictions (1) and (2) with NMS. Obviously the larger the
    slices height and width, and the larger the detection_model.image_size, the more time it takes to run this
    script.

    If the predictions is starting to be very large compare to the size of the slice, WBF can be advantageous as it
    will merge the overlapping boulders. However, WBF and NMS works better in less dense area (or at least is less
    sensitive to the iou_threshold selected).

    Test Time Augmentation could be included too, but it takes lot of time to run it.. so not sure about that too.
    WBF for instance seg: "https://www.kaggle.com/code/mistag/sartorius-tta-with-weighted-segments-fusion"

    Note that the bboxes (in absolute coordinates) are calculated from the bounds of the polygons after the
    predictions are computed.
    """

    prediction_results = detection_model.model(image, imgsz=detection_model.image_size, verbose=False,
                                               device=detection_model.device)
    return _result_to_df(prediction_results[0], detection_model, has_mask, shift_amount, slice_size,
                         min_area_threshold, downscale_pred, contour_method)

def get_sliced_prediction(in_raster,
                          detection_model=None,
                          confidence_threshold: float = 0.1,
                          has_mask=True,
                          output_dir=None,
                          interim_file_name=None,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
                          interim_dir=None,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
                          slice_size: int = 256,
                          inference_size: int = 1024,
                          overlap_height_ratio: float = 0.2,
                          overlap_width_ratio: float = 0.2,
                          min_area_threshold: int = None,
                          downscale_pred: bool = False,
                          postprocess: bool = True,
                          postprocess_match_threshold: float = 0.5,
                          postprocess_class_agnostic: bool = False,
                          batch_size: int = 8,
                          contour_method: str = "skimage",
                          save_bbox: bool = True,
                          save_prenms: bool = True,
                          output_format: str = "shp",
                          n_workers: int = None):
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    The time to run the script is dependent on the number of predictions over the whole image. This is because we need
    to loop through each prediction and transform the bool_mask to polygon. 

    Args:
        in_raster: str or Path()
            Path to raster tif file.
        detection_model: model.DetectionModel
        confidence_threshold: float
            minimum confidence threshold, values below will be automatically filtered away.
        has_mask: bool
        interim_dir: str or Path()
        slice_sze: int
            Height and width of each slice.  Defaults to ``None``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        postprocess: bool
            Include postprocessing or not.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        save_bbox: bool
            Write the bounding-box layers (pre- and post-NMS). Default ``True``. Set ``False``
            to skip them when only the mask outlines are needed.
        save_prenms: bool
            Write the pre-NMS (un-suppressed) layers. Default ``True``. Set ``False`` to write
            only the final post-NMS layer(s) — useful for dense full-image runs.
        output_format: str
            ``"shp"`` (default) or ``"gpkg"``. GeoPackage avoids the shapefile 2 GB / field
            limits and is faster to write for very large (~1e6-feature) outputs.
        n_workers: int
            Threads for the per-detection mask->polygon post-processing. Default
            ``min(8, os.cpu_count())``; pass ``1`` to force the serial path. Output is identical
            regardless of worker count (order-preserving).

    Returns:
        A pd.DataFrame.
    """

    # convert in_raster tif file to png file
    in_raster = Path(in_raster)
    output_dir = Path(output_dir)
    out_png = in_raster.with_name(in_raster.stem + ".png")
    raster_convert.tiff_to_png(in_raster, out_png) # only work with 8bit

    # create temporary directory
    tmp_dir = (Path.home() / "tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # set model's confidence_threshold and inference size
    detection_model.image_size = inference_size
    detection_model.confidence_threshold = confidence_threshold

    slice_image_result = slice_image(
        image=out_png.as_posix(),  # need to be path to
        output_file_name=interim_file_name,  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir=interim_dir,  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        out_ext=".png",  # FORMAT OF (OPTIONALLY) SAVED SLICES
    )

    num_slices = len(slice_image_result)
    shift_amounts = slice_image_result.starting_pixels
    slice_images = slice_image_result.images
    frames = []
    # The mask->polygon post-processing is CPU-bound and embarrassingly parallel per detection;
    # run it across a thread pool (find_contours / cv2 / numpy release the GIL). Output order is
    # preserved, so results are identical to the serial path. Default to a modest worker count
    # (returns diminish past ~8 and we don't want to oversubscribe alongside the GPU thread).
    workers = n_workers if n_workers is not None else min(8, os.cpu_count() or 1)
    executor = ThreadPoolExecutor(max_workers=workers) if workers and workers > 1 else None
    try:
        # perform sliced prediction in batches (batched GPU inference instead of one slice at a
        # time). Per-slice geometry is unchanged; only the model call is grouped.
        for start in tqdm(range(0, num_slices, batch_size), total=(num_slices + batch_size - 1) // batch_size):
            batch_images = list(slice_images[start:start + batch_size])
            prediction_results = detection_model.model(batch_images, imgsz=detection_model.image_size,
                                                       verbose=False, device=detection_model.device)
            for j, prediction_result in enumerate(prediction_results):
                df = _result_to_df(prediction_result, detection_model, has_mask, shift_amounts[start + j],
                                   slice_size, min_area_threshold, downscale_pred, contour_method,
                                   executor=executor)
                if df.shape[0] > 0:
                    frames.append(df)
    finally:
        if executor is not None:
            executor.shutdown()

    if len(frames) == 0:
        df_all = pd.DataFrame(columns=['score', 'polygon', 'category_id', 'category_name', 'is_within_slice'])
    else:
        df_all = pd.concat(frames, ignore_index=True)
    gdf = add_geometries(in_raster, df_all)

    # keep edge predictions (within 10% of slice size from the true footprint edge)

    # extract true footprint
    gdf_true_footprint = raster.true_footprint(in_raster, tmp_dir / "true-footprint.shp")
    in_res = raster_metadata.get_resolution(in_raster)[0]
    in_meta = raster_metadata.get_profile(in_raster)
    gpd.GeoDataFrame(geometry=gdf_true_footprint.geometry.boundary.values, crs=in_meta["crs"].to_wkt()).to_file(
        tmp_dir / "true-footprint-as-a-line.shp")
    gdf_line_buffer = shp.buffer(tmp_dir / "true-footprint-as-a-line.shp", slice_size * 0.10 * in_res,
                                 (tmp_dir / "footprint-buffer.shp"))

    # Flag boulders touching the footprint-edge buffer. We only need the boolean flag, not the
    # materialized intersection geometries, so use a spatial join (intersects predicate)
    # instead of gpd.overlay(how="intersection") — far cheaper and lighter on memory at ~1e6
    # boulders. The buffer was round-tripped through a shapefile whose .prj uses the ESRI-WKT
    # dialect of the same projection; normalize its CRS to the boulders' (identical grid) to
    # avoid a spurious CRS-mismatch and keep the join on one coordinate system.
    gdf_line_buffer = gdf_line_buffer.set_crs(gdf.crs, allow_override=True)
    edge_idx = gpd.sjoin(gdf[["geometry"]], gdf_line_buffer[["geometry"]],
                         predicate="intersects", how="inner").index.unique()
    gdf["is_at_edge"] = False
    gdf.loc[edge_idx, "is_at_edge"] = True

    # keep edge predictions close to the edge of the footprint of the raster, but otherwise remove edge predictions
    gdf = gdf.loc[np.logical_or(gdf.is_at_edge == True, gdf.is_within_slice == True)]

    # remove duplicates
    gdf = gdf.drop_duplicates(subset="geometry", ignore_index=True)
    gdf["id"] = gdf.index

    # save predictions before post-processing (include if downscaling is done or not...)
    ext = ".gpkg" if output_format == "gpkg" else ".shp"
    bbox_filename = in_raster.stem + "-predictions-ct-" + str(int(confidence_threshold * 100)).zfill(3) + "-ss-" + str(
        slice_size) + "-is-" + str(inference_size) + "-ov-" + str(int(overlap_height_ratio * 100)).zfill(3) + "-bbox" + ext
    mask_filename = bbox_filename.replace("-bbox" + ext, "-mask" + ext)

    if downscale_pred:
        bbox_filename = bbox_filename.replace("-bbox" + ext, "-downscaled-bbox" + ext)
        mask_filename = mask_filename.replace("-mask" + ext, "-downscaled-mask" + ext)

    out_bbox_shp = output_dir / bbox_filename
    out_mask_shp = output_dir / mask_filename
    # The pre-NMS files are intermediate, and bbox files aren't needed by many workflows.
    # Gate both so a dense full-image run can avoid writing ~1e6-feature layers it won't use.
    if save_prenms:
        if save_bbox:
            bboxes_to_shp(gdf, out_bbox_shp)
        outlines_to_shp(gdf, out_mask_shp)


    if postprocess:
        # Non-maximum suppression (NMS)
        # regardless of the classes ids (right now is a class agnoistic not supported)
        if postprocess_class_agnostic:
            keep = nms(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
                       iou_threshold=postprocess_match_threshold, class_ids=None, rtree_leaf_size=32)
        # or taking into account the classes ids
        else:
            keep = nms(boxes=np.stack(gdf.bbox.values), scores=gdf.score.values,
                       iou_threshold=postprocess_match_threshold, class_ids=gdf.category_id.values, rtree_leaf_size=32)

        # saving post-processed predictions (the NMS mask layer is the primary deliverable)
        gdf_nms = gdf.loc[keep]
        if save_bbox:
            bboxes_to_shp(gdf_nms, out_bbox_shp.with_name(out_bbox_shp.stem + "-nms" + ext))
        outlines_to_shp(gdf_nms, out_mask_shp.with_name(out_mask_shp.stem + "-nms" + ext))
        return gdf, gdf_nms
    else:
        return gdf, None