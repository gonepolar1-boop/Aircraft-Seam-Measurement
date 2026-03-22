import numpy as np

from seam_geometry_3d.Common.utils import normalize_vector


def fit_plane_least_squares(points_3d):
    """Fit a plane to 3D points using least squares via SVD."""
    points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 3)

    # Stage 1: keep only finite 3D points for plane fitting.
    valid_mask = np.all(np.isfinite(points_3d), axis=1)
    valid_points = points_3d[valid_mask]
    if len(valid_points) < 3:
        return None

    # Stage 2: fit the plane normal from centered points via SVD.
    centroid = np.mean(valid_points, axis=0)
    centered_points = valid_points - centroid
    _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
    normal = normalize_vector(vh[-1])
    if normal is None:
        return None

    return {
        "centroid": centroid.astype(np.float32),
        "normal": normal.astype(np.float32),
        "d": float(-np.dot(normal, centroid)),
        "num_points": int(len(valid_points)),
    }


def point_to_plane_distance(point, plane):
    """Compute signed distance from a 3D point to a plane."""
    point = np.asarray(point, dtype=np.float32).reshape(3)
    return float(np.dot(plane["normal"], point) + plane["d"])


def project_vector_to_plane(vector, plane_normal):
    """Project a vector onto a plane defined by its normal."""
    vector = np.asarray(vector, dtype=np.float32).reshape(3)
    plane_normal = normalize_vector(plane_normal)
    if plane_normal is None:
        return None
    return (vector - float(np.dot(vector, plane_normal)) * plane_normal).astype(np.float32)


def compute_local_gap(left_point, right_point, tangent_3d, reference_normal=None):
    """Compute local gap as transverse distance in the section plane."""
    left_point = np.asarray(left_point, dtype=np.float32).reshape(3)
    right_point = np.asarray(right_point, dtype=np.float32).reshape(3)

    # Stage 1: project the left-to-right edge vector into the local section plane.
    tangent_3d = normalize_vector(tangent_3d)
    if tangent_3d is None:
        return np.nan, {"gap_vector_projected": None, "section_dir_3d": None}

    gap_vector = right_point - left_point
    gap_vector_projected = project_vector_to_plane(gap_vector, tangent_3d)
    if gap_vector_projected is None:
        return np.nan, {"gap_vector_projected": None, "section_dir_3d": None}

    # Stage 2: estimate the local section direction for measuring the transverse gap.
    if reference_normal is not None:
        reference_normal = normalize_vector(reference_normal)
        if reference_normal is not None:
            section_dir_3d = normalize_vector(np.cross(reference_normal, tangent_3d))
        else:
            section_dir_3d = normalize_vector(gap_vector_projected)
    else:
        section_dir_3d = normalize_vector(gap_vector_projected)

    if section_dir_3d is None:
        section_dir_3d = normalize_vector(gap_vector_projected)
    if section_dir_3d is None:
        return np.nan, {"gap_vector_projected": gap_vector_projected, "section_dir_3d": None}

    # Stage 3: measure the projected edge distance along the section direction.
    gap = float(abs(np.dot(gap_vector_projected, section_dir_3d)))
    return gap, {
        "gap_vector_projected": gap_vector_projected,
        "section_dir_3d": section_dir_3d,
    }


def compute_local_flush(left_surface_points, right_surface_points, query_point=None):
    """Fit local planes on both sides and compute their normal height difference."""
    # Stage 1: fit one local support plane on each side of the seam.
    left_plane = fit_plane_least_squares(left_surface_points)
    right_plane = fit_plane_least_squares(right_surface_points)
    if left_plane is None or right_plane is None:
        return np.nan, {
            "left_plane": left_plane,
            "right_plane": right_plane,
            "reference_normal": None,
            "query_point": None,
        }

    # Stage 2: build a shared reference normal from the two plane normals.
    left_normal = left_plane["normal"]
    right_normal = right_plane["normal"]
    if float(np.dot(left_normal, right_normal)) < 0.0:
        right_normal = -right_normal

    reference_normal = normalize_vector(left_normal + right_normal)
    if reference_normal is None:
        reference_normal = left_normal

    # Stage 3: evaluate both planes at the query point along the shared normal.
    if query_point is None:
        query_point = 0.5 * (left_plane["centroid"] + right_plane["centroid"])
    else:
        query_point = np.asarray(query_point, dtype=np.float32).reshape(3)

    left_reference_plane = {
        **left_plane,
        "normal": reference_normal,
        "d": float(-np.dot(reference_normal, left_plane["centroid"])),
    }
    right_reference_plane = {
        **right_plane,
        "normal": reference_normal,
        "d": float(-np.dot(reference_normal, right_plane["centroid"])),
    }
    left_distance = point_to_plane_distance(query_point, left_reference_plane)
    right_distance = point_to_plane_distance(query_point, right_reference_plane)
    flush = float(right_distance - left_distance)
    return flush, {
        "left_plane": left_plane,
        "right_plane": right_plane,
        "reference_normal": reference_normal,
        "query_point": query_point,
        "left_distance": left_distance,
        "right_distance": right_distance,
    }


def compute_gap_flush_from_geometry(geometry_3d, point_map=None, mask=None):
    """Compute gap/flush profiles from mapped 3D seam geometry."""
    gap_profile, flush_profile, row_debug = [], [], []

    for index, row_y in enumerate(geometry_3d["rows"]):
        # Stage 1: gather the per-row geometry and validity state.
        left_valid = bool(geometry_3d["left_edge_3d_valid_mask"][index])
        right_valid = bool(geometry_3d["right_edge_3d_valid_mask"][index])
        center_valid = bool(geometry_3d["centerline_3d_valid_mask"][index])
        frame = geometry_3d["local_frames"][index]
        tangent_3d = frame["tangent_3d"]

        gap_value = np.nan
        gap_debug = {"gap_vector_projected": None, "section_dir_3d": None}
        flush_value = np.nan
        flush_debug = {"left_plane": None, "right_plane": None, "reference_normal": None, "query_point": None}

        # Stage 2: compute flush first, then reuse its reference normal for gap.
        if left_valid and right_valid and center_valid and tangent_3d is not None:
            flush_value, flush_debug = compute_local_flush(
                left_surface_points=geometry_3d["left_surface_points_3d"][index],
                right_surface_points=geometry_3d["right_surface_points_3d"][index],
                query_point=geometry_3d["centerline_3d"][index],
            )

            reference_normal = flush_debug["reference_normal"]
            if reference_normal is None:
                reference_normal = frame["section_normal_3d"]

            gap_value, gap_debug = compute_local_gap(
                left_point=geometry_3d["left_edge_3d"][index],
                right_point=geometry_3d["right_edge_3d"][index],
                tangent_3d=tangent_3d,
                reference_normal=reference_normal,
            )

        # Stage 3: append the row-wise profiles and debug payload.
        gap_profile.append([float(row_y), float(gap_value)])
        flush_profile.append([float(row_y), float(flush_value)])
        row_debug.append(
            {
                "row_y": int(row_y),
                "gap": float(gap_value),
                "flush": float(flush_value),
                "left_valid": left_valid,
                "right_valid": right_valid,
                "center_valid": center_valid,
                "tangent_3d": tangent_3d,
                "section_dir_3d": gap_debug["section_dir_3d"],
                "reference_normal": flush_debug["reference_normal"],
                "left_plane": flush_debug["left_plane"],
                "right_plane": flush_debug["right_plane"],
            }
        )

    gap_profile = np.asarray(gap_profile, dtype=np.float32).reshape(-1, 2)
    flush_profile = np.asarray(flush_profile, dtype=np.float32).reshape(-1, 2)

    # Stage 4: summarize the valid gap and flush measurements.
    gap_values = gap_profile[:, 1] if len(gap_profile) else np.empty((0,), dtype=np.float32)
    flush_values = flush_profile[:, 1] if len(flush_profile) else np.empty((0,), dtype=np.float32)
    valid_gap_values = gap_values[np.isfinite(gap_values)]
    valid_flush_values = flush_values[np.isfinite(flush_values)]

    summary = {
        "valid_gap_rows": int(len(valid_gap_values)),
        "valid_flush_rows": int(len(valid_flush_values)),
        "mean_gap": float(np.mean(valid_gap_values)) if len(valid_gap_values) else np.nan,
        "max_gap": float(np.max(valid_gap_values)) if len(valid_gap_values) else np.nan,
        "min_gap": float(np.min(valid_gap_values)) if len(valid_gap_values) else np.nan,
        "mean_flush": float(np.mean(valid_flush_values)) if len(valid_flush_values) else np.nan,
        "mean_abs_flush": float(np.mean(np.abs(valid_flush_values))) if len(valid_flush_values) else np.nan,
        "max_flush": float(np.max(valid_flush_values)) if len(valid_flush_values) else np.nan,
        "min_flush": float(np.min(valid_flush_values)) if len(valid_flush_values) else np.nan,
        "max_abs_flush": float(np.max(np.abs(valid_flush_values))) if len(valid_flush_values) else np.nan,
    }

    return {
        "gap_profile": gap_profile,
        "flush_profile": flush_profile,
        "summary": summary,
        "debug": {
            "row_debug": row_debug,
            "point_map_shape": None if point_map is None else tuple(point_map.shape),
            "mask_shape": None if mask is None else tuple(mask.shape),
        },
    }
