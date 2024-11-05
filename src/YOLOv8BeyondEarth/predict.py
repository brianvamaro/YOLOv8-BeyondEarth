import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import torch

from YOLOv8BeyondEarth.polygon import (binary_mask_to_polygon, is_within_slice, shift_polygon,
                                       add_geometries, bboxes_to_shp, outlines_to_shp, row_bbox)
from lsnms import nms, wbc
from PIL import Image
from sahi.slicing import slice_image
from tqdm import tqdm
from pathlib import Path

from rastertools_BOULDERING import raster, convert as raster_convert, metadata as raster_metadata
from shptools_BOULDERING import shp
from shapely.geometry import (box, Polygon)
from scipy.ndimage import rotate

#from torchvision.ops import (nms as nms_torch, batched_nms as batched_nms_torch)

def YOLOv8(detection_model, image, has_mask, shift_amount, slice_size, min_area_threshold, downscale_pred):
    """
    Process image with YOLOv8 model to detect objects and generate masks.

    Parameters
    ----------
    detection_model : object
        YOLOv8 model instance with loaded weights
    image : ndarray
        Input image in BGR format (height, width, 3)
    has_mask : bool
        Whether to generate segmentation masks
    shift_amount : tuple of int
        (x, y) coordinates to shift predictions to absolute position
    slice_size : int
        Size of image slice in pixels
    min_area_threshold : int
        Minimum pixel area for valid predictions
    downscale_pred : bool
        Whether to downscale predictions to match slice size

    Returns
    -------
    pandas.DataFrame
        DataFrame containing predictions with columns:
        - score: confidence score
        - polygon: coordinates of segmentation mask
        - category_id: class ID
        - category_name: class name
        - is_within_slice: bool indicating if prediction is fully within slice

    Notes
    -----
    The function processes image slices through YOLOv8 model and converts
    mask predictions to polygon format. Edge predictions have reduced confidence
    scores. Predictions smaller than min_area_threshold are filtered out.
    """

    shift_x = shift_amount[0]
    shift_y = shift_amount[1]

    prediction_results = detection_model.model(image, imgsz=detection_model.image_size, verbose=False,
                                               device=detection_model.device)

    # if no predictions
    if prediction_results[0].boxes.data.size()[0] == 0:
        df = pd.DataFrame(columns=['score', 'polygon', 'category_id', 'category_name', 'is_within_slice'])
    else:
        if has_mask:
            predictions = [
                (result.boxes.data[result.boxes.data[:, 4] >= detection_model.confidence_threshold],
                 result.masks.data[result.boxes.data[:, 4] >= detection_model.confidence_threshold],)
                for result in prediction_results]

        else:
            predictions = []
            for result in prediction_results:
                result_boxes = result.boxes.data[result.boxes.data[:, 4] >= detection_model.confidence_threshold]
                result_masks = torch.tensor([[] for _ in range(result_boxes.size()[0])])
                predictions.append((result_boxes, result_masks))

        # for one image
        # bboxes = [], dropping it as I am calculating it later on.
        scores = []
        polygons = []
        category_ids = []
        category_names = []
        is_polygon_within_slice_list = []

        # names are very confusing I should fix that
        for image_ind, image_predictions in enumerate(predictions):
            image_predictions_in_xyxy_format = image_predictions[0]
            image_predictions_masks = image_predictions[1]

            for prediction, bool_mask in zip(
                    image_predictions_in_xyxy_format.cpu().detach().numpy(),
                    image_predictions_masks.cpu().detach().numpy()
            ):

                score = prediction[4]
                category_id = int(prediction[5])
                category_name = detection_model.category_mapping[str(category_id)]

                # more accurate to have this operation before the eventual resizing
                # takes a little bit of extra computational time
                bool_mask[bool_mask >= 0.5] = 1
                bool_mask[bool_mask < 0.5] = 0

                if downscale_pred:
                    if bool_mask.shape[0] == slice_size:
                        None
                    else:
                        bool_mask = cv2.resize(bool_mask, (slice_size, slice_size), interpolation=cv2.INTER_AREA)

                # number of pixels
                area = len(bool_mask[bool_mask == 1])

                if area > min_area_threshold:
                    try:
                        polygon = binary_mask_to_polygon(bool_mask)
                        if downscale_pred:
                            polygon_slice = polygon
                        else:
                            polygon_slice = np.stack([(polygon[:, 0] / bool_mask.shape[0]) * slice_size,
                                                      (polygon[:, 1] / bool_mask.shape[0]) * slice_size], axis=-1)
                        min_edge_distance = 0.05 * slice_size
                        max_edge_distance = 0.95 * slice_size
                        is_polygon_within_slice = (np.logical_and(polygon_slice[:, 0].min() >= min_edge_distance,
                                                                  polygon_slice[:, 0].max() <= max_edge_distance) and
                                                   np.logical_and(polygon_slice[:, 1].min() >= min_edge_distance,
                                                                  polygon_slice[:, 1].max() <= max_edge_distance))

                        if not is_polygon_within_slice:
                            score = 0.10  # if at edge set score to a low value

                        # conversion to absolute coordinates,
                        shifted_polygon = shift_polygon(polygon_slice, shift_x, shift_y)

                        scores.append(score)
                        polygons.append(shifted_polygon)  # conversion of polygon to absolute coordinates
                        category_ids.append(category_id)
                        category_names.append(category_name)
                        is_polygon_within_slice_list.append(is_polygon_within_slice)
                    except:
                        None

        dict = {'score': scores, 'polygon': polygons,
                'category_id': category_ids, 'category_name': category_names,
                'is_within_slice': is_polygon_within_slice_list}

        df = pd.DataFrame(dict)
    return df

def get_sliced_prediction(in_raster, detection_model=None, confidence_threshold=0.1,
                         has_mask=True, output_dir=None, interim_file_name=None,
                         interim_dir=None, slice_size=None, inference_size=None,
                         overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                         min_area_threshold=None, downscale_pred=False,
                         rotation_angle=None, postprocess=True,
                         postprocess_match_threshold=0.5,
                         postprocess_class_agnostic=False):
    """
    Generate predictions on sliced raster image using YOLOv8 model.

    Parameters
    ----------
    in_raster : str or Path
        Path to input raster file
    detection_model : object, optional
        YOLOv8 model instance
    confidence_threshold : float, default=0.1
        Minimum confidence threshold for predictions
    has_mask : bool, default=True
        Whether to generate segmentation masks
    output_dir : str or Path, optional
        Directory to save output files
    interim_file_name : str, optional
        Name pattern for saved image slices
    interim_dir : str or Path, optional
        Directory to save intermediate slices
    slice_size : int, optional
        Size of image slices in pixels
    inference_size : int, optional
        Input size for model inference
    overlap_height_ratio : float, default=0.2
        Overlap ratio between slices in height
    overlap_width_ratio : float, default=0.2
        Overlap ratio between slices in width
    min_area_threshold : int, optional
        Minimum pixel area for valid predictions
    downscale_pred : bool, default=False
        Whether to downscale predictions
    rotation_angle : int, optional
        Angle to rotate image before processing
    postprocess : bool, default=True
        Whether to apply non-maximum suppression
    postprocess_match_threshold : float, default=0.5
        IoU threshold for NMS
    postprocess_class_agnostic : bool, default=False
        Whether to apply class-agnostic NMS

    Returns
    -------
    tuple of geopandas.GeoDataFrame
        Two GeoDataFrames containing predictions:
        - First contains all predictions
        - Second contains NMS-filtered predictions (if postprocess=True)
        - Returns (predictions, None) if postprocess=False

    Notes
    -----
    The function:
    1. Converts input raster to PNG
    2. Optionally rotates image
    3. Slices image with overlap
    4. Runs YOLOv8 on each slice
    5. Combines predictions and converts to geographic coordinates
    6. Optionally applies NMS
    7. Saves results as shapefiles
    """

    # convert in_raster tif file to png file
    in_raster = Path(in_raster)
    output_dir = Path(output_dir)
    in_png = in_raster.with_name(in_raster.stem + ".png")
    out_png = in_png
    raster_convert.tiff_to_png(in_raster, out_png)  # only work with 8bit

    # apply rotation angle and change out_png filename if needed
    if rotation_angle and rotation_angle != 0:
        out_png = in_raster.with_name(in_raster.stem + "_rotated.png")
        rotate_png(in_png, out_png, rotation_angle)

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

    frames = []
    for i, image in tqdm(enumerate(slice_image_result.images), total=num_slices):
        df = YOLOv8(detection_model, image, has_mask, shift_amounts[i], slice_size, min_area_threshold, downscale_pred)
        if df.shape[0] > 0:
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    if rotation_angle and rotation_angle != 0:
        ## calculate centroid of image
        in_raster = Path(in_raster)
        in_crs = raster_metadata.get_crs(in_raster).to_wkt()
        bbox = raster_metadata.get_bounds(in_raster)
        bbox = box(*bbox)
        gs = gpd.GeoSeries(bbox, crs=in_crs)
        footprint_centroid = gs.centroid

        boulder_geometry = []
        for polygon in df_all.polygon.values:
            xs, ys = (polygon[:, 0], polygon[:, 1])
            p0 = Polygon(np.stack([xs, ys], axis=-1))
            boulder_geometry.append(p0)

        gdf_pixel = gpd.GeoDataFrame(df_all, geometry=boulder_geometry, crs=in_crs)

        boulder_geometry = []
        with rio.open(in_raster) as src:
            for polygon in gdf_pixel.geometry:  # gdf_rotated
                xt = np.array(list(polygon.exterior.coords.xy[1]))
                yt = np.array(list(polygon.exterior.coords.xy[0]))
                x_proj, y_proj = rio.transform.xy(src.transform, xt, yt)
                boulder_geometry.append(Polygon(np.stack([x_proj, y_proj], axis=-1)))

        gdf_world = gpd.GeoDataFrame(df_all, geometry=boulder_geometry, crs=in_crs)

        rotated_geom = gdf_world.rotate(rotation_angle, origin=footprint_centroid.values[0])
        gdf = gpd.GeoDataFrame(df_all, geometry=rotated_geom, crs=in_crs)

        raster_height, raster_width = raster_metadata.get_shape(in_raster)
        in_res = raster_metadata.get_resolution(in_raster)[0]
        shift_v = ((raster_height / 2.0) - (raster_width / 2.0)) * in_res

        # the rotation of the matrix generate shifts in the shapefiles
        # the shifts only occur for angles equal to +-90 or +-270
        if rotation_angle == 90 or rotation_angle == -270:
            geom_shifted = gdf.geometry.translate(xoff=shift_v, yoff=-shift_v)
            gdf["geometry"] = geom_shifted
            gdf["bbox"] = gdf.apply(row_bbox, axis=1)
        elif rotation_angle == -90 or rotation_angle == 270:
            geom_shifted = gdf.geometry.translate(xoff=-shift_v, yoff=shift_v)
            gdf["geometry"] = geom_shifted
            gdf["bbox"] = gdf.apply(row_bbox, axis=1)
        elif rotation_angle == -180 or rotation_angle == 180:
            gdf["bbox"] = gdf.apply(row_bbox, axis=1)
    else:
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

    gdf_boulders = gdf.copy()
    gdf_boulders["id"] = gdf_boulders.index
    gdf_intersected = gpd.overlay(gdf_boulders, gdf_line_buffer, how="intersection", keep_geom_type=True)

    gdf["is_at_edge"] = False
    gdf.loc[gdf_intersected.id.values, "is_at_edge"] = True

    # keep edge predictions close to the edge of the footprint of the raster, but otherwise remove edge predictions
    gdf = gdf.loc[np.logical_or(gdf.is_at_edge == True, gdf.is_within_slice == True)]

    # remove duplicates
    gdf = gdf.drop_duplicates(subset="geometry", ignore_index=True)
    gdf["id"] = gdf.index

    # save shapefile before post-processing (include if downscaling is done or not...)
    bbox_filename = in_raster.stem + "-predictions-ct-" + str(int(confidence_threshold * 100)).zfill(3) + "-ss-" + str(
        slice_size) + "-is-" + str(inference_size) + "-ov-" + str(int(overlap_height_ratio * 100)).zfill(3) + "-bbox.shp"
    mask_filename = bbox_filename.replace("-bbox.shp", "-mask.shp")

    if downscale_pred:
        bbox_filename = bbox_filename.replace("-bbox.shp", "-downscaled-bbox.shp")
        mask_filename = mask_filename.replace("-mask.shp", "-downscaled-mask.shp")

    out_bbox_shp = output_dir / bbox_filename
    out_mask_shp = output_dir / mask_filename
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

        # saving post-processed shapefiles
        gdf_nms = gdf.loc[keep]
        bboxes_to_shp(gdf_nms, out_bbox_shp.with_name(out_bbox_shp.stem + "-nms.shp"))
        outlines_to_shp(gdf_nms, out_mask_shp.with_name(out_mask_shp.stem + "-nms.shp"))
        return gdf, gdf_nms
    else:
        return gdf, None

def rotate_png(in_png, out_png, rotation_angle):
    """
    Rotate PNG image by specified angle.

    Parameters
    ----------
    in_png : str or Path
        Path to input PNG file
    out_png : str or Path
        Path to save rotated image
    rotation_angle : int
        Angle to rotate image in degrees (counter-clockwise)

    Notes
    -----
    Uses scipy.ndimage.rotate to perform the rotation.
    The rotation_angle is negated since scipy rotates counter-clockwise
    while the input angle is specified clockwise.
    """
    in_png = Path(in_png)
    out_png = Path(out_png)
    in_img = Image.open(in_png)
    array = np.array(in_img)
    # the rotation_angle for scipy rotate is counter clockwise
    # while the calculated rotation angle is clockwise
    # need to take the negative of the rotation angle value.
    array_rotated = rotate(array, -rotation_angle)
    out_img= Image.fromarray(array_rotated)
    out_img.save(out_png)