import csv
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from seam_segmentation_2d.Wrapper.dataset_wrapper import SeamDataset
from seam_segmentation_2d.Wrapper.model_wrapper import UNet, segmentation_metrics
from seam_segmentation_2d.Common.plot import plot_threshold_sensitivity, plot_training_metrics
from seam_segmentation_2d.Common.util import (
    BASELINE_EXPERIMENT_CONFIG,
    BEST_CHECKPOINT_PATH,
    BEST_THRESHOLD_SUMMARY_PATH,
    FINAL_SEGMENTATION_METRICS_PATH,
    THRESHOLD_SENSITIVITY_METRICS_CSV_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_threshold(model, val_dataset, device, threshold, batch_size, checkpoint_path=BEST_CHECKPOINT_PATH):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    metric_lists = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred_prob = torch.sigmoid(model(img))
            pred = (pred_prob > threshold).float()
            for i in range(img.size(0)):
                metrics = segmentation_metrics(pred[i], mask[i])
                for key in metric_lists:
                    metric_lists[key].append(metrics[key])
    return {key: float(np.mean(values)) for key, values in metric_lists.items()}


def save_threshold_sensitivity_results(rows):
    with open(THRESHOLD_SENSITIVITY_METRICS_CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["threshold", "dice", "iou", "precision", "recall"])
        writer.writeheader()
        writer.writerows(rows)

    best_row = max(rows, key=lambda row: row["dice"])
    with open(BEST_THRESHOLD_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(f"best_threshold: {best_row['threshold']}\n")
        file.write(f"dice: {best_row['dice']}\n")
        file.write(f"iou: {best_row['iou']}\n")
        file.write(f"precision: {best_row['precision']}\n")
        file.write(f"recall: {best_row['recall']}\n")


def threshold_sensitivity_analysis(batch_size, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    dataset = SeamDataset()
    _, val_dataset = dataset.split_dataset()

    metrics_by_threshold = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }
    rows = []
    for threshold in thresholds:
        metrics = evaluate_threshold(model, val_dataset, device, threshold=threshold, batch_size=batch_size)
        for key in metrics_by_threshold:
            metrics_by_threshold[key].append(metrics[key])
        rows.append({"threshold": threshold, **metrics})
        logger.info(
            "threshold=%.1f, val_dice=%.4f, val_iou=%.4f, val_precision=%.4f, val_recall=%.4f",
            threshold,
            metrics["dice"],
            metrics["iou"],
            metrics["precision"],
            metrics["recall"],
        )
    save_threshold_sensitivity_results(rows)
    plot_threshold_sensitivity(thresholds, metrics_by_threshold)
    return rows


def save_final_segmentation_metrics():
    from seam_segmentation_2d.train_model import load_history
    history = load_history()
    if not history["val_dices"]:
        return None

    final_metrics = {
        "epoch": len(history["val_dices"]),
        "dice": float(history["val_dices"][-1]),
        "iou": float(history["val_ious"][-1]),
        "precision": float(history["val_precisions"][-1]),
        "recall": float(history["val_recalls"][-1]),
        "val_loss": float(history["val_losses"][-1]),
        "train_loss": float(history["train_losses"][-1]),
    }
    with open(FINAL_SEGMENTATION_METRICS_PATH, "w", encoding="utf-8") as file:
        for key, value in final_metrics.items():
            file.write(f"{key}: {value}\n")

    if final_metrics is not None:
        logger.info(
            "Final metrics: Dice=%.4f, IoU=%.4f, Precision=%.4f, Recall=%.4f",
            final_metrics["dice"],
            final_metrics["iou"],
            final_metrics["precision"],
            final_metrics["recall"],
        )


if __name__ == "__main__":
    plot_training_metrics()
    save_final_segmentation_metrics()
    threshold_sensitivity_analysis(batch_size=BASELINE_EXPERIMENT_CONFIG["batch_size"])
