import numpy as np

from seam_geometry_3d.compute_gap_flush import compute_gap_flush_from_geometry
from seam_geometry_3d.extract_3d_seam_geometry import extract_3d_seam_geometry
from seam_geometry_3d.Demo.generate_3d_data import build_demo_image, build_demo_mask, build_demo_point_map


def main():
    mask = build_demo_mask(height=128, width=128, seam_center_x=64, seam_width_px=8)
    image = build_demo_image(mask)
    point_map = build_demo_point_map(mask, base_flush=0.6)

    geometry_3d = extract_3d_seam_geometry(
        image=image,
        mask=mask,
        point_map=point_map,
        side_sample_count=8,
        sample_step_px=1.0,
    )
    measurement = compute_gap_flush_from_geometry(geometry_3d, point_map=point_map, mask=mask)

    print("3D seam demo finished.")
    print(f"Rows: {len(geometry_3d['rows'])}")
    print("Summary:")
    for key, value in measurement["summary"].items():
        print(f"  {key}: {value}")

    if len(geometry_3d["width_2d"]):
        print(f"Mean 2D width (inclusive pixel count): {float(np.mean(geometry_3d['width_2d'])):.4f}")
        print(
            "Mean gap here is edge-point center distance, so it is expected to be about "
            "1 pixel smaller than the inclusive 2D width for this demo point_map definition."
        )

    print("First 5 rows:")
    for row in measurement["debug"]["row_debug"][:5]:
        print(
            f"  row={row['row_y']}, gap={row['gap']:.4f}, flush={row['flush']:.4f}, "
            f"left_valid={row['left_valid']}, right_valid={row['right_valid']}"
        )


if __name__ == "__main__":
    main()
