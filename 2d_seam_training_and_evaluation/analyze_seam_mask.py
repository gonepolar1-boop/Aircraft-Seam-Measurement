import csv
import logging
import os

import cv2
import numpy as np

from Common.plot import plot_seam_measurements, plot_seam_overlay
from Common.util import ANALYZE_SEAM_CONFIG, ensure_dir, GEOMETRY_RESULTS_DIR_PATH, GEOMETRY_SUMMARY_CSV_PATH, IMAGE_DIR_PATH, PRED_MASKS_DIR_PATH


ensure_dir(GEOMETRY_RESULTS_DIR_PATH)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_seam_geometry(
    image,
    mask,
    min_width,
    threshold,
    kernel_size,
    keep_largest_component,
):
    binary_mask_raw = (mask > threshold).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binary_mask_opened = cv2.morphologyEx(binary_mask_raw, cv2.MORPH_OPEN, kernel)

    if keep_largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_opened, connectivity=8)
        binary_mask = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) if num_labels > 1 else binary_mask_opened
    else:
        binary_mask = binary_mask_opened

    rows, left_edge, right_edge, centerline, width_profile = [], [], [], [], []
    for y in range(binary_mask.shape[0]):
        xs = np.where(binary_mask[y] > 0)[0]
        if xs.size == 0:
            continue
        x_left, x_right = int(xs[0]), int(xs[-1])
        width = float(x_right - x_left + 1)
        if width < min_width:
            continue
        center_x = float((x_left + x_right) / 2.0)
        rows.append(y)
        left_edge.append([float(x_left), float(y)])
        right_edge.append([float(x_right), float(y)])
        centerline.append([center_x, float(y)])
        width_profile.append([float(y), width])

    rows = np.asarray(rows, dtype=np.int32)
    centerline = np.asarray(centerline, dtype=np.float32).reshape(-1, 2)
    left_edge = np.asarray(left_edge, dtype=np.float32).reshape(-1, 2)
    right_edge = np.asarray(right_edge, dtype=np.float32).reshape(-1, 2)
    width_profile = np.asarray(width_profile, dtype=np.float32).reshape(-1, 2)

    if len(rows) < 2:
        centerline_fit = centerline.copy()
        center_residual = np.zeros(len(rows), dtype=np.float32)
        summary = {
            "valid_rows": int(len(rows)),
            "mean_width": float(np.mean(width_profile[:, 1])) if len(rows) else 0.0,
            "max_width": float(np.max(width_profile[:, 1])) if len(rows) else 0.0,
            "min_width": float(np.min(width_profile[:, 1])) if len(rows) else 0.0,
            "std_width": float(np.std(width_profile[:, 1])) if len(rows) else 0.0,
            "centerline_fit_slope": 0.0,
            "centerline_fit_bias": float(centerline[0, 0]) if len(rows) else 0.0,
            "mean_abs_center_residual": 0.0,
            "max_abs_center_residual": 0.0,
        }
    else:
        fit_coeff = np.polyfit(centerline[:, 1], centerline[:, 0], deg=1)
        fitted_center_x = fit_coeff[0] * centerline[:, 1] + fit_coeff[1]
        centerline_fit = np.column_stack([fitted_center_x, centerline[:, 1]]).astype(np.float32)
        center_residual = (centerline[:, 0] - centerline_fit[:, 0]).astype(np.float32)
        summary = {
            "valid_rows": int(len(rows)),
            "mean_width": float(np.mean(width_profile[:, 1])),
            "max_width": float(np.max(width_profile[:, 1])),
            "min_width": float(np.min(width_profile[:, 1])),
            "std_width": float(np.std(width_profile[:, 1])),
            "centerline_fit_slope": float(fit_coeff[0]),
            "centerline_fit_bias": float(fit_coeff[1]),
            "mean_abs_center_residual": float(np.mean(np.abs(center_residual))),
            "max_abs_center_residual": float(np.max(np.abs(center_residual))),
        }

    return {
        "binary_mask_raw": binary_mask_raw,
        "binary_mask_opened": binary_mask_opened,
        "binary_mask": binary_mask,
        "image": image,
        "rows": rows,
        "centerline": centerline,
        "centerline_fit": centerline_fit,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "width_profile": width_profile,
        "center_residual": center_residual,
        "summary": summary,
    }


def save_geometry_results(geometry, image_name, mask_name, prefix, save_dir):
    ensure_dir(save_dir)
    np.save(os.path.join(save_dir, f"{prefix}_centerline.npy"), geometry["centerline"])
    np.save(os.path.join(save_dir, f"{prefix}_left_edge.npy"), geometry["left_edge"])
    np.save(os.path.join(save_dir, f"{prefix}_right_edge.npy"), geometry["right_edge"])
    np.save(os.path.join(save_dir, f"{prefix}_width_profile.npy"), geometry["width_profile"])
    np.save(os.path.join(save_dir, f"{prefix}_binary_mask_raw.npy"), geometry["binary_mask_raw"])
    np.save(os.path.join(save_dir, f"{prefix}_binary_mask_opened.npy"), geometry["binary_mask_opened"])
    np.save(os.path.join(save_dir, f"{prefix}_binary_mask_processed.npy"), geometry["binary_mask"])

    with open(os.path.join(save_dir, f"{prefix}_geometry.csv"), "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["row_y", "left_x", "right_x", "center_x", "fitted_center_x", "width", "center_residual"])
        for i in range(len(geometry["rows"])):
            writer.writerow(
                [
                    int(geometry["rows"][i]),
                    float(geometry["left_edge"][i, 0]),
                    float(geometry["right_edge"][i, 0]),
                    float(geometry["centerline"][i, 0]),
                    float(geometry["centerline_fit"][i, 0]),
                    float(geometry["width_profile"][i, 1]),
                    float(geometry["center_residual"][i]),
                ]
            )

    with open(os.path.join(save_dir, f"{prefix}_summary.txt"), "w", encoding="utf-8") as file:
        file.write(f"image_name: {image_name}\n")
        file.write(f"mask_name: {mask_name}\n")
        for key, value in geometry["summary"].items():
            file.write(f"{key}: {value}\n")

    cv2.imwrite(os.path.join(save_dir, f"{prefix}_binary_mask_raw.png"), (geometry["binary_mask_raw"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_binary_mask_opened.png"), (geometry["binary_mask_opened"] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_binary_mask_processed.png"), (geometry["binary_mask"] * 255).astype(np.uint8))


def save_geometry_summary_results(analysis_rows):
    with open(GEOMETRY_SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "mask_name",
                "valid_rows",
                "mean_width",
                "max_width",
                "min_width",
                "std_width",
                "mean_abs_center_residual",
                "max_abs_center_residual",
            ],
        )
        writer.writeheader()
        writer.writerows(analysis_rows)


def analyze_seam_masks(cfg=ANALYZE_SEAM_CONFIG, mask_root=PRED_MASKS_DIR_PATH, output_root=GEOMETRY_RESULTS_DIR_PATH):
    analysis_rows = []
    pred_mask_names = sorted(
        name for name in os.listdir(mask_root)
        if name.lower().endswith(".png") and name.lower().endswith("_pred_mask.png")
    )
    for pred_mask_name in pred_mask_names:
        image_name = pred_mask_name.replace("_pred_mask.png", ".png")
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        mask_path = os.path.join(mask_root, pred_mask_name)
        if not os.path.exists(image_path):
            continue

        geometry = extract_seam_geometry(
            image=cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
            mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE),
            min_width=cfg["min_width"],
            threshold=cfg["threshold"],
            kernel_size=cfg["kernel_size"],
            keep_largest_component=cfg["keep_largest_component"],
        )
        prefix = os.path.splitext(image_name)[0]
        save_dir = os.path.join(output_root, prefix)
        save_geometry_results(geometry, image_name, pred_mask_name, prefix, save_dir)
        plot_seam_overlay(geometry, os.path.join(save_dir, f"{prefix}_overlay.png"))
        plot_seam_measurements(geometry, save_dir, prefix)
        analysis_rows.append(
            {
                "image_name": image_name,
                "mask_name": pred_mask_name,
                "valid_rows": int(geometry["summary"]["valid_rows"]),
                "mean_width": float(geometry["summary"]["mean_width"]),
                "max_width": float(geometry["summary"]["max_width"]),
                "min_width": float(geometry["summary"]["min_width"]),
                "std_width": float(geometry["summary"]["std_width"]),
                "mean_abs_center_residual": float(geometry["summary"]["mean_abs_center_residual"]),
                "max_abs_center_residual": float(geometry["summary"]["max_abs_center_residual"]),
            }
        )

    save_geometry_summary_results(analysis_rows)
    return analysis_rows


if __name__ == "__main__":
    results = analyze_seam_masks(ANALYZE_SEAM_CONFIG)
    logger.info("Seam geometry analysis finished. Processed %d samples.", len(results))
