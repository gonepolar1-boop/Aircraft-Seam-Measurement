import numpy as np

from seam_geometry_3d.Common.validate_data import is_pixel_in_bounds, validate_point_map, validate_pixel_xy, validate_point3d
from seam_geometry_3d.Common.utils import normalize_vector


def pixel_to_point3d(pixel_xy, point_map):
    validate_pixel_xy(pixel_xy)
    x, y = int(round(float(pixel_xy[0]))), int(round(float(pixel_xy[1])))
    if not is_pixel_in_bounds(pixel_xy, point_map.shape[:2]):
        return None
    point3d = point_map[y, x]
    return point3d if validate_point3d(point3d) else None


def pixels_to_points3d(pixels_xy, point_map):
    points3d = np.full((len(pixels_xy), 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(len(pixels_xy), dtype=bool)
    for index, pixel_xy in enumerate(pixels_xy):
        point3d = pixel_to_point3d(pixel_xy, point_map)
        if point3d is None:
            continue
        points3d[index] = point3d
        valid_mask[index] = True
    return points3d, valid_mask


def mask_to_points3d(binary_mask, point_map):
    validate_point_map(point_map)
    binary_mask = np.asarray(binary_mask)
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must have shape (H, W).")
    if binary_mask.shape != point_map.shape[:2]:
        raise ValueError(
            f"binary_mask shape {binary_mask.shape} does not match point_map shape {point_map.shape[:2]}."
        )

    ys, xs = np.where(binary_mask > 0)
    pixels_xy = np.column_stack([xs, ys]).astype(np.float32)
    points3d, valid_mask = pixels_to_points3d(pixels_xy, point_map)
    return {
        "pixels_xy": pixels_xy,
        "points3d": points3d,
        "valid_mask": valid_mask,
        "valid_pixels_xy": pixels_xy[valid_mask],
        "valid_points3d": points3d[valid_mask],
    }
