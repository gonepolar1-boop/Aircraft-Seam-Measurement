import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from seam_segmentation_2d.Common.util import (
    ensure_parent_dir,
    THRESHOLD_SENSITIVITY_CURVE_PATH,
    TRAIN_METRICS_CURVE_PATH,
    WIDTH_EVAL_ERROR_BOXPLOT_PATH,
    WIDTH_EVAL_ERROR_HIST_PATH,
    WIDTH_EVAL_ERROR_VS_GT_WIDTH_PATH,
    WIDTH_EVAL_GT_VS_PRED_PATH,
)
ensure_parent_dir(THRESHOLD_SENSITIVITY_CURVE_PATH)
ensure_parent_dir(TRAIN_METRICS_CURVE_PATH)

logger = logging.getLogger(__name__)


def plot_training_metrics(history=None, save_path=TRAIN_METRICS_CURVE_PATH):
    if history is None:
        from seam_segmentation_2d.train_model import load_history
        history = load_history()
    epochs = range(1, len(history["train_losses"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plots = [
        ("train_losses", "Train Loss"),
        ("val_losses", "Validation Loss"),
        ("val_dices", "Validation Dice"),
        ("val_ious", "Validation IoU"),
        ("val_precisions", "Validation Precision"),
        ("val_recalls", "Validation Recall"),
    ]

    for ax, (key, title) in zip(axes.flat, plots):
        ax.plot(epochs, history[key], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)

    axes[0, 0].set_ylabel("Loss")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 2].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    axes[1, 1].set_ylabel("Score")
    axes[1, 2].set_ylabel("Score")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_threshold_sensitivity(thresholds, metrics_by_threshold, save_path=THRESHOLD_SENSITIVITY_CURVE_PATH):
    dice_scores = metrics_by_threshold["dice"]
    best_idx = int(np.argmax(dice_scores))
    best_threshold, best_dice = thresholds[best_idx], dice_scores[best_idx]
    logger.info("Best threshold = %.1f, best val dice = %.4f", best_threshold, best_dice)

    plt.figure(figsize=(8, 5))
    curve_specs = [
        ("dice", "Dice"),
        ("iou", "IoU"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    for key, label in curve_specs:
        plt.plot(thresholds, metrics_by_threshold[key], marker="o", label=label)

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sensitivity Analysis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Threshold sensitivity curve saved to: %s", save_path)


def plot_seam_overlay(geometry, save_path):
    overlay = cv2.cvtColor(geometry["image"].astype(np.uint8), cv2.COLOR_GRAY2BGR) if geometry["image"] is not None else cv2.cvtColor((geometry["binary_mask"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_tint = np.zeros_like(overlay)
    mask_tint[:, :, 1] = (geometry["binary_mask"] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.8, mask_tint, 0.2, 0)

    for x, y in geometry["left_edge"]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
    for x, y in geometry["right_edge"]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
    for x, y in geometry["centerline"]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)
    for x, y in geometry["centerline_fit"]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (255, 0, 0), -1)
    cv2.imwrite(save_path, overlay)


def plot_seam_measurements(geometry, save_dir, prefix):
    plt.figure(figsize=(8, 4))
    plt.plot(geometry["width_profile"][:, 0], geometry["width_profile"][:, 1], marker="o", markersize=2)
    plt.xlabel("Row (y)")
    plt.ylabel("Width (pixel)")
    plt.title("Seam Width Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_width_profile.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(geometry["rows"], geometry["center_residual"], marker="o", markersize=2)
    plt.xlabel("Row (y)")
    plt.ylabel("Center Residual (pixel)")
    plt.title("Centerline Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_center_residual.png"), dpi=300)
    plt.close()


def plot_width_eval_figures(rows):
    gt_widths = np.array([float(row["gt_width"]) for row in rows], dtype=np.float32)
    pred_widths = np.array([float(row["pred_mean_width"]) for row in rows], dtype=np.float32)
    errors = np.array([float(row["error"]) for row in rows], dtype=np.float32)
    min_val = float(min(np.min(gt_widths), np.min(pred_widths)))
    max_val = float(max(np.max(gt_widths), np.max(pred_widths)))

    plt.figure(figsize=(6, 6))
    plt.scatter(gt_widths, pred_widths, alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Ground Truth Width")
    plt.ylabel("Predicted Mean Width")
    plt.title("GT vs Predicted Width")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(WIDTH_EVAL_GT_VS_PRED_PATH, dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=20, edgecolor="black")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Width Error Histogram")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(WIDTH_EVAL_ERROR_HIST_PATH, dpi=300)
    plt.close()

    grouped_errors = {}
    for row in rows:
        key = int(round(float(row["gt_width"])))
        grouped_errors.setdefault(key, []).append(float(row["error"]))
    labels = [str(key) for key in sorted(grouped_errors)]
    values = [grouped_errors[key] for key in sorted(grouped_errors)]

    plt.figure(figsize=(7, 4))
    plt.boxplot(values, labels=labels)
    plt.xlabel("Ground Truth Width Group")
    plt.ylabel("Prediction Error")
    plt.title("Width Error Boxplot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(WIDTH_EVAL_ERROR_BOXPLOT_PATH, dpi=300)
    plt.close()

    sorted_pairs = sorted(zip(gt_widths, errors), key=lambda item: item[0])
    sorted_gt_widths = [pair[0] for pair in sorted_pairs]
    sorted_errors = [pair[1] for pair in sorted_pairs]
    plt.figure(figsize=(8, 4))
    plt.plot(sorted_gt_widths, sorted_errors, marker="o", linestyle="-")
    plt.axhline(0.0, color="r", linestyle="--")
    plt.xlabel("Ground Truth Width")
    plt.ylabel("Prediction Error")
    plt.title("Width Error vs Ground Truth Width")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(WIDTH_EVAL_ERROR_VS_GT_WIDTH_PATH, dpi=300)
    plt.close()
