import csv
import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from seam_geometry_3d.compute_gap_flush import compute_gap_flush_from_geometry
from seam_geometry_3d.extract_3d_seam_geometry import extract_3d_seam_geometry
from seam_segmentation_2d.Common.util import ANALYZE_SEAM_CONFIG, BASELINE_EXPERIMENT_CONFIG, BEST_CHECKPOINT_PATH
from seam_segmentation_2d.Wrapper.model_wrapper import UNet
from seam_segmentation_2d.analyze_seam_mask import extract_seam_geometry


@dataclass
class MeasurementConfig:
    segmentation_threshold: float = float(BASELINE_EXPERIMENT_CONFIG["threshold"])
    mask_threshold: int = int(ANALYZE_SEAM_CONFIG["threshold"])
    min_width: int = int(ANALYZE_SEAM_CONFIG["min_width"])
    kernel_size: int = int(ANALYZE_SEAM_CONFIG["kernel_size"])
    keep_largest_component: bool = bool(ANALYZE_SEAM_CONFIG["keep_largest_component"])
    side_sample_count: int = 5
    sample_step_px: float = 1.0
    gap_limit: float = 2.0
    flush_limit: float = 1.0
    checkpoint_path: str = BEST_CHECKPOINT_PATH

    def analyze_cfg(self) -> dict[str, Any]:
        return {
            "threshold": int(self.mask_threshold),
            "min_width": int(self.min_width),
            "kernel_size": int(self.kernel_size),
            "keep_largest_component": bool(self.keep_largest_component),
        }


def load_grayscale_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return image


def load_point_map(point_map_path: str) -> np.ndarray:
    extension = os.path.splitext(point_map_path)[1].lower()
    if extension == ".npy":
        point_map = np.load(point_map_path)
    elif extension == ".npz":
        data = np.load(point_map_path)
        if not data.files:
            raise ValueError(f"No arrays found in point map file: {point_map_path}")
        point_map = data[data.files[0]]
    else:
        raise ValueError("Point map must be a .npy or .npz file.")

    point_map = np.asarray(point_map, dtype=np.float32)
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        raise ValueError(f"Point map must have shape [H, W, 3], got {point_map.shape}.")
    return point_map


def _load_model(checkpoint_path: str, device: torch.device) -> UNet:
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_mask(image: np.ndarray, checkpoint_path: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)

    img_size = int(BASELINE_EXPERIMENT_CONFIG["img_size"])
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(resized.astype(np.float32)[None, None, ...] / 255.0).to(device)

    with torch.no_grad():
        pred_logits = model(image_tensor)
        pred_prob = torch.sigmoid(pred_logits).cpu().numpy().squeeze().astype(np.float32)

    pred_prob = cv2.resize(pred_prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred_mask = (pred_prob >= float(threshold)).astype(np.uint8) * 255
    return pred_prob, pred_mask


def build_overlay(image: np.ndarray, mask: np.ndarray, geometry: dict[str, Any] | None = None) -> np.ndarray:
    overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    mask_bool = mask > 0
    overlay[mask_bool] = (0.4 * overlay[mask_bool] + np.array([20, 180, 20])).astype(np.uint8)

    if geometry is not None:
        for points, color in (
            (geometry.get("left_edge"), (0, 255, 255)),
            (geometry.get("right_edge"), (0, 255, 255)),
            (geometry.get("centerline"), (255, 80, 80)),
        ):
            if points is None:
                continue
            for x, y in np.asarray(points):
                cv2.circle(overlay, (int(round(float(x))), int(round(float(y)))), 1, color, -1)
    return overlay


def evaluate_quality(summary: dict[str, Any], cfg: MeasurementConfig) -> dict[str, Any]:
    mean_gap = summary.get("mean_gap")
    max_abs_flush = summary.get("max_abs_flush")
    gap_ok = bool(np.isfinite(mean_gap)) and float(mean_gap) <= float(cfg.gap_limit)
    flush_ok = bool(np.isfinite(max_abs_flush)) and float(max_abs_flush) <= float(cfg.flush_limit)
    passed = gap_ok and flush_ok
    return {
        "gap_limit": float(cfg.gap_limit),
        "flush_limit": float(cfg.flush_limit),
        "gap_ok": gap_ok,
        "flush_ok": flush_ok,
        "passed": passed,
        "status_text": "鍚堟牸" if passed else "瓒呭樊",
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_profile_csv(save_path: str, header: tuple[str, str], values: np.ndarray) -> None:
    with open(save_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row_y, metric in np.asarray(values, dtype=np.float32):
            writer.writerow([float(row_y), None if not np.isfinite(metric) else float(metric)])


def create_report_figure(result: dict[str, Any], save_path: str) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(result["image"], cmap="gray")
    ax0.set_title("Original")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(result["mask"], cmap="gray")
    ax1.set_title("Mask")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(cv2.cvtColor(result["overlay"], cv2.COLOR_BGR2RGB))
    ax2.set_title("Overlay")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    width_profile = result["geometry_2d"]["width_profile"]
    if len(width_profile):
        ax3.plot(width_profile[:, 0], width_profile[:, 1], color="#0F766E")
    ax3.set_title("Width Profile")
    ax3.set_xlabel("row")
    ax3.set_ylabel("px")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    gap_profile = result["measurement_3d"]["gap_profile"]
    if len(gap_profile):
        ax4.plot(gap_profile[:, 0], gap_profile[:, 1], color="#1D4ED8")
    ax4.set_title("Gap Profile")
    ax4.set_xlabel("row")
    ax4.set_ylabel("gap")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    flush_profile = result["measurement_3d"]["flush_profile"]
    if len(flush_profile):
        ax5.plot(flush_profile[:, 0], flush_profile[:, 1], color="#DC2626")
    ax5.set_title("Flush Profile")
    ax5.set_xlabel("row")
    ax5.set_ylabel("flush")
    ax5.grid(True, alpha=0.3)

    summary_lines = [
        f"image: {os.path.basename(result['inputs']['image_path'])}",
        f"status: {result['quality']['status_text']}",
        f"mean gap: {result['measurement_3d']['summary'].get('mean_gap'):.4f}" if np.isfinite(result["measurement_3d"]["summary"].get("mean_gap", np.nan)) else "mean gap: N/A",
        f"mean flush: {result['measurement_3d']['summary'].get('mean_flush'):.4f}" if np.isfinite(result["measurement_3d"]["summary"].get("mean_flush", np.nan)) else "mean flush: N/A",
        f"max abs flush: {result['measurement_3d']['summary'].get('max_abs_flush'):.4f}" if np.isfinite(result["measurement_3d"]["summary"].get("max_abs_flush", np.nan)) else "max abs flush: N/A",
    ]
    fig.suptitle("Aircraft Seam Measurement Summary\n" + " | ".join(summary_lines), fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def export_measurement_result(result: dict[str, Any], output_dir: str) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    gap_csv_path = os.path.join(output_dir, "gap_profile.csv")
    flush_csv_path = os.path.join(output_dir, "flush_profile.csv")
    summary_json_path = os.path.join(output_dir, "summary.json")
    report_png_path = os.path.join(output_dir, "report.png")
    overlay_png_path = os.path.join(output_dir, "overlay.png")
    mask_png_path = os.path.join(output_dir, "mask.png")

    _write_profile_csv(gap_csv_path, ("row_y", "gap"), result["measurement_3d"]["gap_profile"])
    _write_profile_csv(flush_csv_path, ("row_y", "flush"), result["measurement_3d"]["flush_profile"])

    cv2.imwrite(overlay_png_path, result["overlay"])
    cv2.imwrite(mask_png_path, result["mask"])
    create_report_figure(result, report_png_path)

    summary_payload = {
        "inputs": result["inputs"],
        "config": _json_ready(result["config"]),
        "geometry_2d_summary": _json_ready(result["geometry_2d"]["summary"]),
        "measurement_3d_summary": _json_ready(result["measurement_3d"]["summary"]),
        "quality": _json_ready(result["quality"]),
        "exports": {
            "gap_profile_csv": gap_csv_path,
            "flush_profile_csv": flush_csv_path,
            "overlay_png": overlay_png_path,
            "mask_png": mask_png_path,
            "report_png": report_png_path,
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, ensure_ascii=False, indent=2)

    return {
        "gap_profile_csv": gap_csv_path,
        "flush_profile_csv": flush_csv_path,
        "summary_json": summary_json_path,
        "overlay_png": overlay_png_path,
        "mask_png": mask_png_path,
        "report_png": report_png_path,
    }


def run_measurement_once(
    image_path: str,
    point_map_path: str,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    config: MeasurementConfig | None = None,
    mask_path: str | None = None,
) -> dict[str, Any]:
    cfg = config or MeasurementConfig()
    if checkpoint_path:
        cfg.checkpoint_path = checkpoint_path

    image = load_grayscale_image(image_path)
    point_map = load_point_map(point_map_path)
    if image.shape[:2] != point_map.shape[:2]:
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match point map shape {point_map.shape[:2]}."
        )

    if mask_path:
        mask = load_grayscale_image(mask_path)
        if mask.shape != image.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}.")
        pred_prob = mask.astype(np.float32) / 255.0
    else:
        pred_prob, mask = predict_mask(image, cfg.checkpoint_path, cfg.segmentation_threshold)

    geometry_2d = extract_seam_geometry(
        image=image,
        mask=mask,
        min_width=cfg.min_width,
        threshold=cfg.mask_threshold,
        kernel_size=cfg.kernel_size,
        keep_largest_component=cfg.keep_largest_component,
    )
    geometry_3d = extract_3d_seam_geometry(
        mask=mask,
        point_map=point_map,
        image=image,
        analyze_cfg=cfg.analyze_cfg(),
        side_sample_count=cfg.side_sample_count,
        sample_step_px=cfg.sample_step_px,
    )
    measurement_3d = compute_gap_flush_from_geometry(geometry_3d, point_map=point_map, mask=mask)
    overlay = build_overlay(image, mask, geometry_2d)
    quality = evaluate_quality(measurement_3d["summary"], cfg)

    result = {
        "inputs": {
            "image_path": os.path.abspath(image_path),
            "point_map_path": os.path.abspath(point_map_path),
            "checkpoint_path": os.path.abspath(cfg.checkpoint_path) if cfg.checkpoint_path else None,
            "mask_path": os.path.abspath(mask_path) if mask_path else None,
        },
        "config": dict(cfg.__dict__),
        "image": image,
        "probability": pred_prob,
        "mask": mask,
        "overlay": overlay,
        "geometry_2d": geometry_2d,
        "geometry_3d": geometry_3d,
        "measurement_3d": measurement_3d,
        "quality": quality,
        "exports": {},
    }

    if output_dir:
        result["exports"] = export_measurement_result(result, output_dir)

    return result
