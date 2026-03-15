# 2D Seam Training and Evaluation

This directory contains the current working implementation of the 2D seam pipeline. It generates synthetic seam data, trains a U-Net segmentation model, predicts masks on the validation split, extracts seam geometry, and evaluates width-estimation accuracy.

## Workflow

The standard pipeline is:

1. `generate_data.py`
2. `train_model.py`
3. `predict_masks.py`
4. `analyze_seam_mask.py`
5. `evaluate_width.py`

Optional post-training analysis:

6. `train_analysis.py`

## Directory Layout

```text
2d_seam_training_and_evaluation/
+-- Common/                    # Shared configuration, paths, and plotting helpers
+-- Wrapper/                   # Dataset wrapper and U-Net model wrapper
+-- analyze_seam_mask.py       # Seam geometry extraction from predicted masks
+-- evaluate_width.py          # Width evaluation against generated labels
+-- generate_data.py           # Synthetic image and mask generation
+-- predict_masks.py           # Validation-set inference and visualization
+-- train_analysis.py          # Training curves and threshold sweep analysis
+-- train_model.py             # U-Net training entrypoint
+-- run_2d_seam_pipeline.sh    # End-to-end shell runner
```

Generated directories are created automatically during execution:

- `dataset/`
- `model/`
- `results/`

## Environment

Recommended dependencies:

- Python 3.10+
- PyTorch
- torchvision
- OpenCV
- NumPy
- matplotlib
- tqdm

Example install:

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm
```

## Default Configuration

Configuration is defined in [Common/util.py](/d:/MyProgram/aircraft_seam_measurement/2d_seam_training_and_evaluation/Common/util.py).

Key defaults:

- synthetic samples: `300`
- image size: `512 x 512`
- train / validation split: `0.8 / 0.2`
- batch size: `4`
- learning rate: `1e-3`
- epochs: `3`
- inference threshold: `0.8`
- geometry / width-analysis binary threshold: `127`

## Scripts

### `generate_data.py`

Generates grayscale seam images, binary masks, and `labels.csv`.

Outputs:

- `dataset/images/`
- `dataset/masks/`
- `dataset/labels.csv`

### `train_model.py`

Trains the U-Net segmentation model with `BCEWithLogitsLoss + Dice loss`.

Outputs:

- `model/checkpoints/best_checkpoint.pth`
- `model/checkpoints/latest_checkpoint.pth`
- `model/logs/train.log`
- `model/metrics/npy/*.npy`

### `predict_masks.py`

Loads `best_checkpoint.pth`, runs inference on the validation split, saves predicted masks, and writes visualization panels.

Outputs:

- `results/pred_masks/*_pred_mask.png`
- `results/pred_masks/visualizations/*_pred_vis.png`

### `analyze_seam_mask.py`

Post-processes predicted masks with thresholding, morphological opening, and optional largest-component filtering, then extracts:

- left edge
- right edge
- centerline
- fitted centerline
- width profile
- centerline residuals

Outputs:

- `results/geometry/<sample>/`
- `results/geometry/geometry_summary.csv`

### `evaluate_width.py`

Compares predicted mean width with generated ground-truth width labels and computes summary statistics.

Outputs:

- `results/width_eval/width_eval.csv`
- `results/width_eval/grouped_error_by_gt_width.csv`
- `results/width_eval/summary.txt`
- `results/width_eval/gt_vs_pred.png`
- `results/width_eval/error_hist.png`
- `results/width_eval/error_boxplot.png`
- `results/width_eval/error_vs_gt_width.png`

### `train_analysis.py`

Reads saved training history and creates training / threshold-analysis reports.

Outputs:

- `model/figures/training_metrics_curve.png`
- `model/figures/threshold_sensitivity_curve.png`
- `model/figures/threshold_sensitivity_metrics.csv`
- `model/figures/best_threshold_summary.txt`
- `model/figures/final_segmentation_metrics.txt`

## How to Run

Run from this directory:

```bash
python generate_data.py
python train_model.py
python predict_masks.py
python analyze_seam_mask.py
python evaluate_width.py
```

Run the optional analysis step:

```bash
python train_analysis.py
```

If your environment supports shell scripts:

```bash
bash run_2d_seam_pipeline.sh
```

## Methods Summary

- Model: U-Net for single-channel binary segmentation
- Loss: BCE with logits plus Dice loss
- Validation metrics: Dice, IoU, Precision, Recall
- Geometry extraction: row-wise edge search after mask cleanup
- Width evaluation: MAE, RMSE, max error, relative error, centerline residual statistics

## Notes

- The current dataset is synthetic and generated locally.
- `predict_masks.py` uses the validation split returned by `SeamDataset.split_dataset()`.
- `train_model.py` can resume from `latest_checkpoint.pth` if `resume=True` in `BASELINE_EXPERIMENT_CONFIG`.
- Root-level project overview is in [../README.md](/d:/MyProgram/aircraft_seam_measurement/README.md).
