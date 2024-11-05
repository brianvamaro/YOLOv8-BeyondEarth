import numpy as np
import skimage
import rasterio as rio
import geopandas as gpd

from shapely.geometry import (box, Polygon)

def remvove_edge_predictions():
    """
    Remove predictions that touch the edge of the true footprint.

    Parameters
    ----------
    in_raster : str
        Path to the input raster file.

    Returns
    -------
    bool
        True if prediction is within footprint, False if touching edge.

    Notes
    -----
    This function computes the true footprint and checks if predictions 
    are touching it, marking those that do with a column value.
    """
    None

def is_within_slice(polygon, slice_height, slice_width):
    """
    Check if a polygon touches the edge of an image slice.

    Parameters
    ----------
    polygon : ndarray
        Array of shape (N, 2) containing polygon coordinates.
    slice_height : int
        Height of the image slice.
    slice_width : int
        Width of the image slice.

    Returns
    -------
    bool
        False if polygon touches slice edge, True otherwise.
    """
    at_edge12 = np.any(np.any(polygon == -0.5, axis=0) == True)
    at_edge3 = np.any(polygon[:, 0] == slice_width - 0.5, axis=0)
    at_edge4 = np.any(polygon[:, 1] == slice_height - 0.5, axis=0)
    is_intersecting_edge = np.any(np.array([at_edge12, at_edge3, at_edge4]) == True)
    return (False if is_intersecting_edge else True)

def shift_polygon(polygon, shift_x, shift_y):
    """
    Shift polygon coordinates by given x and y offsets.

    Parameters
    ----------
    polygon : ndarray
        Array of shape (N, 2) containing polygon coordinates.
    shift_x : float
        Amount to shift in x direction.
    shift_y : float
        Amount to shift in y direction.

    Returns
    -------
    ndarray
        Shifted polygon coordinates array of same shape as input.
    """
    return (np.stack([polygon[:, 0] + shift_x, polygon[:, 1] + shift_y], axis=-1))

def get_bbox_index(row, src):
    """
    Get bounding box indices from row coordinates.

    Parameters
    ----------
    row : pandas.Series
        Row containing bbox coordinates.
    src : rasterio.DatasetReader
        Opened raster dataset.

    Returns
    -------
    list
        [col_min, row_min, col_max, row_max] indices.
    """
    row_min, col_min = src.index(x=row.bbox[0], y=row.bbox[1])
    row_max, col_max = src.index(x=row.bbox[2], y=row.bbox[3])
    return [col_min, row_min, col_max, row_max]

def get_bbox_xy_shapely(bbox, src):
    """
    Convert bbox indices to shapely polygon coordinates.

    Parameters
    ----------
    bbox : list
        [col_min, row_min, col_max, row_max] indices.
    src : rasterio.DatasetReader
        Opened raster dataset.

    Returns
    -------
    list
        List of (x,y) coordinate tuples forming a polygon.

    Notes
    -----
    Returns coordinates in order: [xmin,ymin], [xmin,ymax], [xmax,ymax], 
    [xmax,ymin], [xmin,ymin].
    """
    return [src.xy(bbox[1],bbox[0]),
            src.xy(bbox[3],bbox[0]),
            src.xy(bbox[3],bbox[2]),
            src.xy(bbox[1],bbox[2]),
            src.xy(bbox[1],bbox[0])]

def row_bbox(row):
    """
    Extract bbox coordinates from geometry row.

    Parameters
    ----------
    row : pandas.Series
        Row containing geometry column.

    Returns
    -------
    list
        [xmin, ymin, xmax, ymax] coordinates.
    """
    return(list(row.geometry.bounds))

def row_bbox_to_shapely(row):
    """
    Convert bbox coordinates to shapely box geometry.

    Parameters
    ----------
    row : pandas.Series
        Row containing bbox coordinates.

    Returns
    -------
    shapely.geometry.box
        Box geometry from bbox coordinates.
    """
    return(box(*row.bbox))

def add_geometries(in_raster, df):
    """
    Convert polygon coordinates to shapely geometries with CRS.

    Parameters
    ----------
    in_raster : str
        Path to input raster file.
    df : pandas.DataFrame
        DataFrame containing polygon coordinates.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with shapely geometries and CRS from raster.
    """
    with rio.open(in_raster) as src:
        in_crs = src.meta["crs"]
        boulder_geometry = []
        for polygon in df.polygon.values:
            xs, ys = rio.transform.xy(src.transform, polygon[:, 1], polygon[:, 0])
            boulder_geometry.append(Polygon(np.stack([xs, ys], axis=-1)))
        gdf = gpd.GeoDataFrame(df, geometry=boulder_geometry, crs=in_crs.to_wkt())
        gdf["bbox"] = gdf.apply(row_bbox, axis=1)
    return gdf

def bboxes_to_shp(gdf, out_shp):
    """
    Save bounding box geometries to shapefile.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing bbox coordinates.
    out_shp : str
        Output shapefile path.
    """
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    gdf_copy["geometry"] = gdf_copy.apply(row_bbox_to_shapely, axis=1)
    gdf_copy.drop(columns=['bbox','polygon']).to_file(out_shp)

def outlines_to_shp(gdf, out_shp):
    """
    Save polygon outlines to shapefile.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries.
    out_shp : str
        Output shapefile path.
    """
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    gdf_copy.drop(columns=['bbox','polygon']).to_file(out_shp)

def close_contour(contour):
    """
    Ensure contour is closed by adding first point to end if needed.

    Parameters
    ----------
    contour : ndarray
        Array of contour coordinates.

    Returns
    -------
    ndarray
        Closed contour with matching first and last points.
    """
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask):
    """
    Convert binary mask to polygon representation.

    Parameters
    ----------
    binary_mask : ndarray
        2D binary numpy array where 1s represent the object.

    Returns
    -------
    ndarray
        Array of polygon coordinates.

    Notes
    -----
    Pads mask to close contours at edges and finds contours using
    skimage.measure.find_contours.
    """
    polygon = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = skimage.measure.find_contours(padded_binary_mask, 0.5)

    # yolo can produce a mask where pixels are not interconnected
    # in this case the following line does not work
    contours = np.subtract(contours, 1)
    contour = np.flip(contours[0], axis=1) # should be interconnected
    return contour

def check_mask_validity(binary_mask, min_area_threshold=4):
    """
    Check if binary mask meets validity criteria.

    Parameters
    ----------
    binary_mask : ndarray
        2D binary numpy array where 1s represent the object.
    min_area_threshold : int, optional
        Minimum number of pixels required, by default 4.

    Returns
    -------
    bool
        True if mask meets all criteria, False otherwise.

    Notes
    -----
    Checks:
    - Minimum area threshold
    - At least 2 pixels in width/height
    - Single connected component
    - No holes
    """
    # I need to be careful if I mix height, width...
    # at least two cells in height or width
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    # is the mask at least a pixel in width or height?
    if not np.any(rows) or not np.any(cols):
        wh_criteria = False
    else:
        wh_criteria = True

    # number of pixels
    area = len(binary_mask[binary_mask == 1])

    # number of blobs
    n_blobs = skimage.measure.label(binary_mask).max()

    # is there any holes in the mask?
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    n_contours = len(skimage.measure.find_contours(padded_binary_mask, 0.5))
    if n_contours == 1:
        contour_criteria = True
    else:
        contour_criteria = False

    # we want at least 4 pixels, a width/height of at least two pixels
    # and only masks that have pixels that are interconnected, i.e., no multipolygons
    if area >= min_area_threshold and n_blobs == 1 and wh_criteria and contour_criteria:
        return True
    else:
        return False