import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from Common.util import (
    BASELINE_EXPERIMENT_CONFIG,
    BEST_CHECKPOINT_PATH,
    ensure_dir,
    IMAGE_DIR_PATH,
    PRED_MASKS_DIR_PATH,
    PRED_MASKS_VIS_DIR_PATH,
)
from Wrapper.dataset_wrapper import SeamDataset
from Wrapper.model_wrapper import UNet

ensure_dir(PRED_MASKS_DIR_PATH)


def save_prediction_visualization(image, gt_mask, pred_mask, pred_prob, save_path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt_overlay = image_bgr.copy()
    gt_overlay[gt_mask > 127] = (0, 255, 0)
    pred_overlay = image_bgr.copy()
    pred_overlay[pred_mask > 127] = (0, 0, 255)

    probability_map = cv2.normalize(pred_prob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    probability_map = cv2.applyColorMap(probability_map, cv2.COLORMAP_JET)

    top_row = np.hstack([image_bgr, gt_overlay])
    bottom_row = np.hstack([pred_overlay, probability_map])
    canvas = np.vstack([top_row, bottom_row])
    cv2.imwrite(save_path, canvas)


def predict_masks(threshold, checkpoint_path=BEST_CHECKPOINT_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = SeamDataset()
    _, val_dataset = dataset.split_dataset()

    for idx in tqdm(val_dataset.indices, desc="Predicting masks"):
        image_name = dataset.image_names[idx]
        image = cv2.imread(os.path.join(IMAGE_DIR_PATH, image_name), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(os.path.join(dataset.mask_directory_path, dataset.mask_names[idx]), cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(
            image,
            (BASELINE_EXPERIMENT_CONFIG["img_size"], BASELINE_EXPERIMENT_CONFIG["img_size"]),
            interpolation=cv2.INTER_LINEAR,
        )
        image_tensor = torch.tensor(
            resized_image.astype(np.float32)[None, None, ...] / 255.0,
            dtype=torch.float32,
        )
        with torch.no_grad():
            pred_prob = torch.sigmoid(model(image_tensor.to(device)))
            pred_prob_np = pred_prob.cpu().squeeze().numpy().astype(np.float32)
            pred_mask = (pred_prob_np > threshold).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(PRED_MASKS_DIR_PATH, f"{os.path.splitext(image_name)[0]}_pred_mask.png"), pred_mask)
        save_prediction_visualization(
            image=resized_image,
            gt_mask=cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST),
            pred_mask=pred_mask,
            pred_prob=pred_prob_np,
            save_path=os.path.join(PRED_MASKS_VIS_DIR_PATH, f"{os.path.splitext(image_name)[0]}_pred_vis.png"),
        )


if __name__ == "__main__":
    predict_masks(BASELINE_EXPERIMENT_CONFIG["threshold"])
