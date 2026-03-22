import numpy as np

from seam_segmentation_2d.Common.util import ANALYZE_SEAM_CONFIG
from seam_segmentation_2d.analyze_seam_mask import extract_seam_geometry

from seam_geometry_3d.Common.utils import normalize_vector
from seam_geometry_3d.Common.validate_data import validate_image, validate_mask, validate_point_map
from seam_geometry_3d.map_2d_to_3d import pixels_to_points3d


def estimate_section_direction_2d(centerline_2d, index, left_edge_xy, right_edge_xy, window=2):
    """Estimate a stable left-to-right section direction for one seam row."""
    num_points = len(centerline_2d)
    # Stage 1: estimate the local seam tangent from neighboring centerline points.
    if num_points < 2:
        tangent_2d = np.array([0.0, 1.0], dtype=np.float32)
    else:
        left_index = max(0, index - window)
        right_index = min(num_points - 1, index + window)
        tangent_2d = normalize_vector(centerline_2d[right_index] - centerline_2d[left_index])
        if tangent_2d is None:
            tangent_2d = np.array([0.0, 1.0], dtype=np.float32)
    # Stage 2: rotate the tangent by 90 degrees to get the cross-seam direction.
    section_dir_2d = normalize_vector(np.array([tangent_2d[1], -tangent_2d[0]], dtype=np.float32))
    if section_dir_2d is None:
        section_dir_2d = np.array([1.0, 0.0], dtype=np.float32)
    # Stage 3: align the direction with the actual left-to-right edge direction.
    edge_dir_2d = normalize_vector(right_edge_xy - left_edge_xy)
    if edge_dir_2d is not None and float(np.dot(section_dir_2d, edge_dir_2d)) < 0.0:
        section_dir_2d = -section_dir_2d
    return None, section_dir_2d.astype(np.float32)


def build_surface_sample_pixels(left_edge_xy, right_edge_xy, section_dir_2d, side_sample_count=8, sample_step_px=1.0):
    """Build left/right 2D neighbor pixels from a precomputed section direction."""
    offsets = np.arange(1, side_sample_count + 1, dtype=np.float32) * float(sample_step_px)
    left_surface_pixels = left_edge_xy - offsets[:, None] * section_dir_2d
    right_surface_pixels = right_edge_xy + offsets[:, None] * section_dir_2d
    return left_surface_pixels.astype(np.float32), right_surface_pixels.astype(np.float32)


def estimate_local_section_axes_3d(
    left_edge_3d,
    right_edge_3d,
    centerline_3d,
    left_valid_mask,
    right_valid_mask,
    centerline_valid_mask,
    index,
    window=2,
):
    """Estimate the local 3D tangent and section direction for one seam row."""
    num_points = len(centerline_3d)
    # Stage 1: estimate the local seam tangent from neighboring valid centerline points.
    tangent_3d = None
    candidate_indices = list(range(index - window, index + window + 1))
    valid_indices = [idx for idx in candidate_indices if 0 <= idx < num_points and centerline_valid_mask[idx]]
    if len(valid_indices) >= 2:
        tangent_3d = normalize_vector(centerline_3d[valid_indices[-1]] - centerline_3d[valid_indices[0]])
    # Stage 2: estimate the local cross-seam direction from the left/right edge pair.
    section_dir_3d = None
    if left_valid_mask[index] and right_valid_mask[index]:
        raw_section = right_edge_3d[index] - left_edge_3d[index]
        if tangent_3d is not None:
            raw_section = raw_section - float(np.dot(raw_section, tangent_3d)) * tangent_3d
        section_dir_3d = normalize_vector(raw_section)
    return tangent_3d, section_dir_3d


def build_local_frame(
    left_edge_3d,
    right_edge_3d,
    centerline_3d,
    left_valid_mask,
    right_valid_mask,
    centerline_valid_mask,
    index,
):
    tangent_3d, section_dir_3d = estimate_local_section_axes_3d(
        left_edge_3d=left_edge_3d,
        right_edge_3d=right_edge_3d,
        centerline_3d=centerline_3d,
        left_valid_mask=left_valid_mask,
        right_valid_mask=right_valid_mask,
        centerline_valid_mask=centerline_valid_mask,
        index=index,
    )

    section_normal_3d = None
    if tangent_3d is not None and section_dir_3d is not None:
        section_normal_3d = normalize_vector(np.cross(tangent_3d, section_dir_3d))

    return {
        "row_index": index,
        "tangent_3d": tangent_3d,
        "section_dir_3d": section_dir_3d,
        "section_normal_3d": section_normal_3d,
        "center_valid": bool(centerline_valid_mask[index]),
        "left_valid": bool(left_valid_mask[index]),
        "right_valid": bool(right_valid_mask[index]),
    }


def extract_3d_seam_geometry(
    mask,
    point_map,
    image=None,
    analyze_cfg=ANALYZE_SEAM_CONFIG,
    side_sample_count=8,
    sample_step_px=1.0,
):
    """Extract 2D seam geometry first, then map it to aligned 3D points."""
    validate_point_map(point_map)
    validate_mask(mask, point_map)
    if image is not None:
        validate_image(image, mask)

    geometry_2d = extract_seam_geometry(
        image=image,
        mask=mask,
        min_width=analyze_cfg["min_width"],
        threshold=analyze_cfg["threshold"],
        kernel_size=analyze_cfg["kernel_size"],
        keep_largest_component=analyze_cfg["keep_largest_component"],
    )
    left_edge_2d, right_edge_2d = geometry_2d["left_edge"], geometry_2d["right_edge"]
    centerline_2d, rows = geometry_2d["centerline"], geometry_2d["rows"]
    width_2d = geometry_2d["width_profile"][:, 1] if len(geometry_2d["width_profile"]) else np.empty((0,), dtype=np.float32)

    left_edge_3d, left_valid_mask = pixels_to_points3d(left_edge_2d, point_map)
    right_edge_3d, right_valid_mask = pixels_to_points3d(right_edge_2d, point_map)
    centerline_3d, centerline_valid_mask = pixels_to_points3d(centerline_2d, point_map)

    surface_row_samples = []
    local_frames = []

    for index in range(len(rows)):
        # Stage 1: estimate the per-row 2D cross-seam direction.
        _, section_dir_2d = estimate_section_direction_2d(
            centerline_2d=centerline_2d,
            index=index,
            left_edge_xy=left_edge_2d[index],
            right_edge_xy=right_edge_2d[index],
        )

        # Stage 2: sample neighbor pixels on both sides of the seam.
        left_surface_pixels, right_surface_pixels = build_surface_sample_pixels(
            left_edge_xy=left_edge_2d[index],
            right_edge_xy=right_edge_2d[index],
            section_dir_2d=section_dir_2d,
            side_sample_count=side_sample_count,
            sample_step_px=sample_step_px,
        )

        # Stage 3: map the sampled pixels into 3D points.
        left_surface_points, left_surface_valid = pixels_to_points3d(left_surface_pixels, point_map)
        right_surface_points, right_surface_valid = pixels_to_points3d(right_surface_pixels, point_map)

        local_frame = build_local_frame(
            left_edge_3d=left_edge_3d,
            right_edge_3d=right_edge_3d,
            centerline_3d=centerline_3d,
            left_valid_mask=left_valid_mask,
            right_valid_mask=right_valid_mask,
            centerline_valid_mask=centerline_valid_mask,
            index=index,
        )

        surface_row_samples.append(
            {
                "left_surface_pixels_2d": left_surface_pixels,
                "right_surface_pixels_2d": right_surface_pixels,
                "left_surface_points_3d": left_surface_points,
                "right_surface_points_3d": right_surface_points,
                "left_surface_valid_mask": left_surface_valid,
                "right_surface_valid_mask": right_surface_valid,
                "section_dir_2d": section_dir_2d,
            }
        )
        local_frames.append(local_frame)

    return {
        "rows": rows,
        "left_edge_2d": left_edge_2d,
        "right_edge_2d": right_edge_2d,
        "centerline_2d": centerline_2d,
        "left_edge_3d": left_edge_3d,
        "right_edge_3d": right_edge_3d,
        "centerline_3d": centerline_3d,
        "left_edge_3d_valid_mask": left_valid_mask,
        "right_edge_3d_valid_mask": right_valid_mask,
        "centerline_3d_valid_mask": centerline_valid_mask,
        "surface_row_samples": surface_row_samples,
        "left_surface_pixels_2d": [sample["left_surface_pixels_2d"] for sample in surface_row_samples],
        "right_surface_pixels_2d": [sample["right_surface_pixels_2d"] for sample in surface_row_samples],
        "left_surface_points_3d": [sample["left_surface_points_3d"] for sample in surface_row_samples],
        "right_surface_points_3d": [sample["right_surface_points_3d"] for sample in surface_row_samples],
        "left_surface_valid_masks": [sample["left_surface_valid_mask"] for sample in surface_row_samples],
        "right_surface_valid_masks": [sample["right_surface_valid_mask"] for sample in surface_row_samples],
        "section_dirs_2d": np.asarray(
            [sample["section_dir_2d"] for sample in surface_row_samples], dtype=np.float32
        ) if surface_row_samples else np.empty((0, 2), dtype=np.float32),
        "local_frames": local_frames,
        "width_2d": width_2d.astype(np.float32),
        "geometry_2d": geometry_2d,
    }
