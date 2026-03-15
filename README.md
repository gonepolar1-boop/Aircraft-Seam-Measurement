# Aircraft Seam Measurement

Research code for a staged aircraft skin seam inspection workflow. The repository currently focuses on a complete 2D pipeline for synthetic data generation, seam segmentation, seam geometry extraction, and seam width evaluation.

## Current Scope

The active implementation lives in `2d_seam_training_and_evaluation/` and supports:

- synthetic grayscale seam image and mask generation
- U-Net based binary seam segmentation
- validation-set mask prediction and visualization
- seam centerline, edge, and width-profile extraction
- width-error analysis with summary tables and plots

Later stages such as 2D-3D mapping, gap/flush evaluation, and point-cloud processing are not yet implemented in this repository.

## Repository Layout

```text
aircraft_seam_measurement/
+-- 2d_seam_training_and_evaluation/
|   +-- Common/                    # Shared paths, config, and plotting helpers
|   +-- Wrapper/                   # Dataset and model wrappers
|   +-- analyze_seam_mask.py       # Geometry extraction from predicted masks
|   +-- evaluate_width.py          # Width evaluation against generated labels
|   +-- generate_data.py           # Synthetic dataset generation
|   +-- predict_masks.py           # Inference on the validation split
|   +-- train_analysis.py          # Training and threshold analysis helpers
|   +-- train_model.py             # U-Net training entrypoint
|   +-- run_2d_seam_pipeline.sh    # End-to-end pipeline runner
|   +-- README.md                  # Stage-specific notes
+-- .gitignore
+-- LICENSE
+-- README.md
```

## Environment

Recommended:

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

The main defaults are defined in [2d_seam_training_and_evaluation/Common/util.py](/d:/MyProgram/aircraft_seam_measurement/2d_seam_training_and_evaluation/Common/util.py):

- generated samples: `300`
- image size: `512 x 512`
- train / validation split: `0.8 / 0.2`
- batch size: `4`
- learning rate: `1e-3`
- epochs: `3`
- prediction threshold: `0.8`

## Typical Workflow

Run from `2d_seam_training_and_evaluation/`:

```bash
python generate_data.py
python train_model.py
python predict_masks.py
python analyze_seam_mask.py
python evaluate_width.py
```

If your environment supports shell scripts, you can also run:

```bash
bash run_2d_seam_pipeline.sh
```

## Generated Outputs

The 2D pipeline writes generated artifacts under:

- `2d_seam_training_and_evaluation/dataset/`
- `2d_seam_training_and_evaluation/model/`
- `2d_seam_training_and_evaluation/results/`

This includes images, masks, labels, checkpoints, metric histories, logs, predicted masks, geometry exports, CSV summaries, and evaluation plots. These outputs are treated as generated artifacts and are ignored by Git.

## Notes

- The dataset used by the current code is synthetic and produced locally by `generate_data.py`.
- Prediction loads `model/checkpoints/best_checkpoint.pth` by default.
- Root-level documentation is a project overview; the stage-specific details are in [2d_seam_training_and_evaluation/README.md](/d:/MyProgram/aircraft_seam_measurement/2d_seam_training_and_evaluation/README.md).

## License

Released under the MIT License. See [LICENSE](/d:/MyProgram/aircraft_seam_measurement/LICENSE).
