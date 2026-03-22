import csv
import logging
import os
import cv2
import numpy as np

from seam_segmentation_2d.analyze_seam_mask import extract_seam_geometry
from seam_segmentation_2d.Common.plot import plot_width_eval_figures
from seam_segmentation_2d.Common.util import (
    EVALUATE_WIDTH_CONFIG,
    ensure_dir,
    LABELS_CSV_PATH,
    PRED_MASKS_DIR_PATH,
    WIDTH_EVAL_CSV_PATH,
    WIDTH_EVAL_DIR_PATH,
    WIDTH_EVAL_GROUPED_ERROR_CSV_PATH,
    WIDTH_EVAL_SUMMARY_PATH,
)


ensure_dir(WIDTH_EVAL_DIR_PATH)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_evaluate_width_results(eval_rows, summary, group_metrics):
    with open(WIDTH_EVAL_CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "mask_name",
                "gt_width",
                "pred_mean_width",
                "pred_max_width",
                "pred_min_width",
                "pred_std_width",
                "abs_error",
                "error",
                "valid_rows",
                "mean_abs_center_residual",
                "max_abs_center_residual",
            ],
        )
        writer.writeheader()
        writer.writerows(eval_rows)

    with open(WIDTH_EVAL_SUMMARY_PATH, "w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")
        file.write("\nGrouped Metrics\n")
        for group in group_metrics:
            file.write(
                f"gt_width={group['gt_width_group']}, num_samples={group['num_samples']}, "
                f"MAE={group['MAE']:.6f}, RMSE={group['RMSE']:.6f}, Max Error={group['Max Error']:.6f}\n"
            )

    with open(WIDTH_EVAL_GROUPED_ERROR_CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["gt_width_group", "num_samples", "MAE", "RMSE", "Max Error"],
        )
        writer.writeheader()
        writer.writerows(group_metrics)

    plot_width_eval_figures(eval_rows)


def evaluate_width(cfg=EVALUATE_WIDTH_CONFIG):
    with open(LABELS_CSV_PATH, "r", encoding="utf-8-sig") as file:
        labels = list(csv.DictReader(file))
    labels_by_image_name = {label["image_name"]: label for label in labels}

    eval_rows = []
    pred_mask_names = sorted(
        name for name in os.listdir(PRED_MASKS_DIR_PATH)
        if name.lower().endswith(".png") and name.lower().endswith("_pred_mask.png")
    )
    for mask_name in pred_mask_names:
        image_name = mask_name.replace("_pred_mask.png", ".png")
        label = labels_by_image_name.get(image_name)
        if label is None:
            continue
        mask_path = os.path.join(PRED_MASKS_DIR_PATH, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        geometry = extract_seam_geometry(
            image=None,
            mask=mask,
            min_width=cfg["min_width"],
            threshold=cfg["threshold"],
            kernel_size=cfg["kernel_size"],
            keep_largest_component=cfg["keep_largest_component"],
        )
        pred_mean_width = float(geometry["summary"]["mean_width"])
        gt_width = float(label["gt_width"])
        error = pred_mean_width - gt_width
        eval_rows.append(
            {
                "image_name": image_name,
                "mask_name": mask_name,
                "gt_width": gt_width,
                "pred_mean_width": pred_mean_width,
                "pred_max_width": float(geometry["summary"]["max_width"]),
                "pred_min_width": float(geometry["summary"]["min_width"]),
                "pred_std_width": float(geometry["summary"]["std_width"]),
                "abs_error": abs(error),
                "error": error,
                "valid_rows": int(geometry["summary"]["valid_rows"]),
                "mean_abs_center_residual": float(geometry["summary"]["mean_abs_center_residual"]),
                "max_abs_center_residual": float(geometry["summary"]["max_abs_center_residual"]),
            }
        )

    abs_errors = np.array([float(row["abs_error"]) for row in eval_rows], dtype=np.float32)
    errors = np.array([float(row["error"]) for row in eval_rows], dtype=np.float32)
    gt_widths = np.array([float(row["gt_width"]) for row in eval_rows], dtype=np.float32)
    center_residuals = np.array([float(row["mean_abs_center_residual"]) for row in eval_rows], dtype=np.float32)
    summary = {
        "num_samples": int(len(eval_rows)),
        "MAE": float(np.mean(abs_errors)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Max Error": float(np.max(abs_errors)),
        "Mean Relative Error": float(np.mean(abs_errors / np.maximum(gt_widths, 1e-6))),
        "Mean Center Residual": float(np.mean(center_residuals)),
    }

    groups = {}
    for row in eval_rows:
        key = int(round(float(row["gt_width"])))
        groups.setdefault(key, []).append(row)

    group_metrics = []
    for key in sorted(groups):
        group = groups[key]
        group_abs_errors = np.array([float(item["abs_error"]) for item in group], dtype=np.float32)
        group_errors = np.array([float(item["error"]) for item in group], dtype=np.float32)
        group_metrics.append(
            {
                "gt_width_group": key,
                "num_samples": len(group),
                "MAE": float(np.mean(group_abs_errors)),
                "RMSE": float(np.sqrt(np.mean(group_errors ** 2))),
                "Max Error": float(np.max(group_abs_errors)),
            }
        )

    save_evaluate_width_results(eval_rows, summary, group_metrics)
    return eval_rows, summary, group_metrics


if __name__ == "__main__":
    rows, summary, group_metrics = evaluate_width(EVALUATE_WIDTH_CONFIG)
    logger.info("Width evaluation finished.")
    logger.info("Num samples: %d", summary["num_samples"])
    logger.info("MAE: %.6f", summary["MAE"])
    logger.info("RMSE: %.6f", summary["RMSE"])
