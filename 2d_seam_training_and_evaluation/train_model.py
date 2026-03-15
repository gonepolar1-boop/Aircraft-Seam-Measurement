import os
import random
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from Wrapper.model_wrapper import UNet, SegmentationCriterion, segmentation_metrics
from Wrapper.dataset_wrapper import SeamDataset
from Common.util import (
    BASELINE_EXPERIMENT_CONFIG,
    BEST_CHECKPOINT_PATH,
    LATEST_CHECKPOINT_PATH,
    TRAIN_LOG_PATH,
    ensure_training_dirs,
    TRAIN_LOSSES_PATH,
    VAL_DICES_PATH,
    VAL_IOUS_PATH,
    VAL_LOSSES_PATH,
    VAL_PRECISIONS_PATH,
    VAL_RECALLS_PATH,
)


ensure_training_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TRAIN_LOG_PATH, encoding="utf-8")
    ]
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, best_val_dice, save_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_dice = checkpoint.get("best_val_dice", 0.0)
    return model, optimizer, start_epoch, best_val_dice


def save_history(history):
    np.save(TRAIN_LOSSES_PATH, np.array(history["train_losses"]))
    np.save(VAL_LOSSES_PATH, np.array(history["val_losses"]))
    np.save(VAL_DICES_PATH, np.array(history["val_dices"]))
    np.save(VAL_IOUS_PATH, np.array(history["val_ious"]))
    np.save(VAL_PRECISIONS_PATH, np.array(history["val_precisions"]))
    np.save(VAL_RECALLS_PATH, np.array(history["val_recalls"]))


def load_history():
    history_files = {
        "train_losses": TRAIN_LOSSES_PATH,
        "val_losses": VAL_LOSSES_PATH,
        "val_dices": VAL_DICES_PATH,
        "val_ious": VAL_IOUS_PATH,
        "val_precisions": VAL_PRECISIONS_PATH,
        "val_recalls": VAL_RECALLS_PATH,
    }
    history = {}
    for key, path in history_files.items():
        history[key] = np.load(path).tolist() if os.path.exists(path) else []
    return history


def evaluate(model, val_loader, criterion, device, history, threshold):
    model.eval()
    val_loss_sum, batch_count, sample_count = 0.0, 0, 0
    metric_sums = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = criterion(output, mask)
            pred_prob = torch.sigmoid(output)
            pred = (pred_prob > threshold).float()

            val_loss_sum += loss.item()
            batch_count += 1

            for i in range(img.size(0)):
                metrics = segmentation_metrics(pred[i], mask[i])
                for key in metric_sums:
                    metric_sums[key] += metrics[key]
                sample_count += 1

        history["val_losses"].append(val_loss_sum / max(batch_count, 1))
        history["val_dices"].append(metric_sums["dice"] / max(sample_count, 1))
        history["val_ious"].append(metric_sums["iou"] / max(sample_count, 1))
        history["val_precisions"].append(metric_sums["precision"] / max(sample_count, 1))
        history["val_recalls"].append(metric_sums["recall"] / max(sample_count, 1))
    return history


def train(model, train_dataset, val_dataset, device, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = SegmentationCriterion()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    start_epoch, best_val_dice = 0, 0.0
    history = {
        "train_losses": [],
        "val_losses": [],
        "val_dices": [],
        "val_ious": [],
        "val_precisions": [],
        "val_recalls": [],
    }
    if cfg["resume"]:
        if os.path.exists(LATEST_CHECKPOINT_PATH):
            model, optimizer, start_epoch, best_val_dice = load_checkpoint(model, optimizer, LATEST_CHECKPOINT_PATH, device)
            logging.info(f"Resume training from checkpoint: start_epoch={start_epoch}, best_val_dice={best_val_dice:.4f}")
        history = load_history()

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss, batch_count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output = model(img)
                loss = criterion(output, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=loss.item())
        history["train_losses"].append(epoch_loss / max(batch_count, 1))
        history = evaluate(model, val_loader, criterion, device, history, cfg["threshold"])
        logging.info(
            "Epoch %d/%d, Train Loss: %.4f, Val Loss: %.4f, Val Dice: %.4f, Val IoU: %.4f, Val Precision: %.4f, Val Recall: %.4f",
            epoch + 1, cfg["epochs"],
            history["train_losses"][-1],
            history["val_losses"][-1],
            history["val_dices"][-1],
            history["val_ious"][-1],
            history["val_precisions"][-1],
            history["val_recalls"][-1],
        )
        save_history(history)

        if history["val_dices"][-1] > best_val_dice:
            best_val_dice = history["val_dices"][-1]
            save_checkpoint(model, optimizer, epoch, best_val_dice, BEST_CHECKPOINT_PATH)
        save_checkpoint(model, optimizer, epoch, best_val_dice, LATEST_CHECKPOINT_PATH)


def train_model(cfg=BASELINE_EXPERIMENT_CONFIG):
    set_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    dataset = SeamDataset()
    train_dataset, val_dataset = dataset.split_dataset()
    train(model, train_dataset, val_dataset, device, cfg)


if __name__ == "__main__":
    train_model(BASELINE_EXPERIMENT_CONFIG)
