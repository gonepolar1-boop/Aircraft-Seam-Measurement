import numpy as np


def validate_point_map(point_map):
    """Validate that the point map has shape [H, W, 3]."""
    if not isinstance(point_map, np.ndarray):
        raise TypeError("point_map must be a numpy.ndarray.")
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        raise ValueError(f"point_map must have shape (H, W, 3), got {point_map.shape}.")
    if point_map.shape[0] <= 0 or point_map.shape[1] <= 0:
        raise ValueError("point_map height and width must be positive.")
    return True


def validate_mask(mask, point_map):
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask.shape}.")
    if mask.shape != point_map.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match point_map shape {point_map.shape[:2]}.")


def validate_image(image, mask):
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray.")
    if image is not None:
        if image.shape[:2] != mask.shape:
            raise ValueError(f"image shape {image.shape[:2]} does not match mask shape {mask.shape}.")


def validate_pixel_xy(pixel_xy):
    if pixel_xy.size != 2:
        raise ValueError("pixel_xy must contain exactly 2 values.")


def is_pixel_in_bounds(pixel_xy, shape_hw):
    validate_pixel_xy(pixel_xy)
    if len(shape_hw) != 2:
        raise ValueError("shape_hw must contain exactly 2 values: (H, W).")

    x = int(round(float(pixel_xy[0])))
    y = int(round(float(pixel_xy[1])))
    height, width = int(shape_hw[0]), int(shape_hw[1])
    return 0 <= x < width and 0 <= y < height


def validate_point3d(point3d):
    if point3d is None:
        return False
    return point3d.size == 3 and np.all(np.isfinite(point3d))
