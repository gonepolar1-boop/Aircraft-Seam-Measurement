import os
import random
import csv
import cv2
import numpy as np

from seam_segmentation_2d.Common.util import (
    ANALYZE_SEAM_CONFIG,
    GENERATE_IMG_CONFIG,
    IMAGE_DIR_PATH,
    LABELS_CSV_PATH,
    MASK_DIR_PATH,
    ensure_dirs,
    ensure_parent_dir,
)
from seam_segmentation_2d.analyze_seam_mask import extract_seam_geometry


ensure_dirs(IMAGE_DIR_PATH, MASK_DIR_PATH)
ensure_parent_dir(LABELS_CSV_PATH)


class Augmenter:
    def __init__(self):
        self.augmentations = {
            "rotate": self._rotate,
            "noise": self._noise,
            "blur": self._blur,
            "brightness": self._brightness,
            "illumination": self._illumination,
            "shadow": self._shadow,
            "highlight": self._highlight,
            "scratch": self._scratch,
            "spot": self._spot,
            "fake_seam": self._fake_seam,
            "short_fake_seam": self._short_fake_seam,
            "occlude_on_seam": self._occlude_on_seam,
            "rivet_occlusion": self._rivet_occlusion,
            "affine": self._affine,
            "break": self._break
        }

    def apply_random_augmentations(self, img, mask, max_augmentations=5, augmentation_names=None, augmentation_params=None):
        if augmentation_names is None:
            augmentation_names = list(self.augmentations.keys())
        if augmentation_params is None:
            augmentation_params = {}
        selected = random.sample(
            augmentation_names,
            random.randint(1, min(max_augmentations, len(augmentation_names))),
        )
        self.img_shape = img.shape

        for aug_name in selected:
            augmentation_func = self.augmentations[aug_name]
            img, mask = augmentation_func(img, mask, augmentation_params.get(aug_name, None))
        return img, mask

    def _rotate(self, img, mask, params=None):
        center = (self.img_shape[1] // 2, self.img_shape[0] // 2)
        angle = random.uniform(-180, 180) if params is None else params.get("angle", random.uniform(-180, 180))
        transform_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        img_out = cv2.warpAffine(
            img, transform_matrix, (self.img_shape[1], self.img_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        mask_out = cv2.warpAffine(
            mask, transform_matrix, (self.img_shape[1], self.img_shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return img_out, mask_out

    def _noise(self, img, mask, params=None):
        sigma = random.uniform(3, 10) if params is None else params.get("sigma", random.uniform(3, 10))
        noise = np.random.normal(0, sigma, self.img_shape)

        img_out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img_out, mask

    def _blur(self, img, mask, params=None):
        ksize = random.choice([3, 5]) if params is None else params.get("ksize", random.choice([3, 5]))

        img_out = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img_out, mask

    def _brightness(self, img, mask, params=None):
        beta = random.randint(-20, 20) if params is None else params.get("beta", random.randint(-20, 20))

        img_out = np.clip(img.astype(np.int16) + beta, 0, 255).astype(np.uint8)
        return img_out, mask

    def _illumination(self, img, mask, params=None):
        h, w = img.shape
        img_f = img.astype(np.float32)
        strength = random.uniform(18, 45) if params is None else params.get("strength", random.uniform(18, 45))
        mode = random.choice(["x", "y", "xy"]) if params is None else params.get("mode", random.choice(["x", "y", "xy"]))

        if mode == "x":
            grad = np.tile(np.linspace(-strength, strength, w, dtype=np.float32), (h, 1))
        elif mode == "y":
            grad = np.tile(np.linspace(-strength, strength, h, dtype=np.float32).reshape(h, 1), (1, w))
        else:
            gx = np.tile(np.linspace(-strength, strength, w, dtype=np.float32), (h, 1))
            gy = np.tile(np.linspace(strength, -strength, h, dtype=np.float32).reshape(h, 1), (1, w))
            grad = 0.5 * gx + 0.5 * gy

        img_out = np.clip(img_f + grad, 0, 255).astype(np.uint8)
        return img_out, mask

    def _shadow(self, img, mask, params=None):
        h, w = img.shape
        img_f = img.astype(np.float32)
        cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
        rx, ry = random.randint(50, 160), random.randint(50, 160)
        yy, xx = np.mgrid[0:h, 0:w]
        strength = random.uniform(20, 60) if params is None else params.get("strength", random.uniform(20, 60))
        shadow = np.exp(-(((xx - cx) ** 2) / (2 * rx ** 2) + ((yy - cy) ** 2) / (2 * ry ** 2)))

        img_out = np.clip(img_f - strength * shadow, 0, 255).astype(np.uint8)
        return img_out, mask

    def _highlight(self, img, mask, params=None):
        h, w = img.shape
        img_f = img.astype(np.float32)
        cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
        rx, ry = random.randint(25, 90), random.randint(25, 90)
        yy, xx = np.mgrid[0:h, 0:w]
        strength = random.uniform(25, 80) if params is None else params.get("strength", random.uniform(25, 80))
        highlight = np.exp(-(((xx - cx) ** 2) / (2 * rx ** 2) + ((yy - cy) ** 2) / (2 * ry ** 2)))

        img_out = np.clip(img_f + strength * highlight, 0, 255).astype(np.uint8)
        return img_out, mask

    def _scratch(self, img, mask, params=None):
        h, w = img.shape
        img_out = img.copy()
        num = random.randint(2, 6) if params is None else params.get("num", random.randint(2, 6))

        for _ in range(num):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            length, angle = random.randint(30, 140), random.uniform(-np.pi, np.pi)
            x2, y2 = int(x1 + length * np.cos(angle)), int(y1 + length * np.sin(angle))
            cv2.line(
                img_out, (x1, y1), (x2, y2),
                color=random.randint(140, 235),
                thickness=random.randint(1, 2),
                lineType=cv2.LINE_AA
            )
        return img_out, mask

    def _spot(self, img, mask, params=None):
        img_out = img.copy()
        h, w = img.shape
        num = random.randint(6, 18) if params is None else params.get("num", random.randint(6, 18))

        for _ in range(num):
            x, y, r = random.randint(0, w - 1), random.randint(0, h - 1), random.randint(2, 7)
            color = random.randint(220, 255) if random.random() < 0.5 else random.randint(80, 150)
            cv2.circle(img_out, (x, y), r, color, -1, lineType=cv2.LINE_AA)
        return img_out, mask

    def _short_fake_seam(self, img, mask, params=None):
        img_out = img.astype(np.float32).copy()
        h, w = img.shape
        num = random.randint(1, 4) if params is None else params.get("num", random.randint(1, 4))

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for _ in range(num):
            cx, cy = random.randint(w // 8, 7 * w // 8), random.randint(h // 8, 7 * h // 8)
            theta = random.uniform(0, np.pi)
            s = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
            d = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)

            fake_width, fake_len = random.uniform(2.5, 7.0), random.uniform(25, 120)
            region = np.abs(s) <= (fake_len / 2.0)
            sigma_valley = max(fake_width / 2.3, 1.0)
            valley = np.exp(-(d ** 2) / (2 * sigma_valley ** 2)) * region.astype(np.float32)

            edge_sigma = random.uniform(0.8, 1.4)
            edge_offset = fake_width / 2.0 + random.uniform(0.5, 1.2)
            edge1 = np.exp(-((d - edge_offset) ** 2) / (2 * edge_sigma ** 2)) * region.astype(np.float32)
            edge2 = np.exp(-((d + edge_offset) ** 2) / (2 * edge_sigma ** 2)) * region.astype(np.float32)

            img_out = img_out - valley * random.uniform(18, 40) + (edge1 + edge2) * random.uniform(2, 8)
        img_out = np.clip(img_out, 0, 255).astype(np.uint8)
        return img_out, mask

    def _fake_seam(self, img, mask, params=None):
        img_out = img.astype(np.float32).copy()
        h, w = img.shape
        cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
        theta = random.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
        fake_width = random.uniform(1.5, 4.0)
        fake = np.exp(-(d ** 2) / (2 * (fake_width / 2.2) ** 2))
        img_out -= fake * random.uniform(20, 45)

        img_out = np.clip(img_out, 0, 255).astype(np.uint8)
        return img_out, mask

    def _rivet_occlusion(self, img, mask, params=None):
        img_out = img.copy()
        h, w = img.shape
        num = random.randint(2, 8) if params is None else params.get("num", random.randint(2, 8))

        for _ in range(num):
            x, y, r = random.randint(0, w - 1), random.randint(0, h - 1), random.randint(3, 7)
            cv2.circle(img_out, (x, y), r, color=random.randint(220, 255), thickness=-1, lineType=cv2.LINE_AA)
        return img_out, mask

    def _affine(self, img, mask, params=None):
        h, w = img.shape
        dx, dy, shear = random.uniform(-12, 12), random.uniform(-12, 12), random.uniform(-0.03, 0.03)
        transform_matrix = np.float32([
            [1, shear, dx],
            [shear, 1, dy]
        ])

        img_out = cv2.warpAffine(
            img, transform_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        mask_out = cv2.warpAffine(
            mask, transform_matrix, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return img_out, mask_out

    def _break(self, img, mask, params=None):
        img_out = img.copy()
        h, w = img.shape
        num = random.randint(1, 3) if params is None else params.get("num", random.randint(1, 3))

        for _ in range(num):
            x1, y1 = random.randint(0, w - 40), random.randint(0, h - 40)
            x2, y2 = min(w, x1 + random.randint(20, 80)), min(h, y1 + random.randint(8, 30))
            patch = np.random.normal(190, 8, (y2 - y1, x2 - x1)).astype(np.uint8)
            img_out[y1:y2, x1:x2] = patch
        return img_out, mask

    def _occlude_on_seam(self, img, mask, params=None):
        img_out = img.copy()
        h, w = img.shape
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return img_out, mask

        num = random.randint(1, 3) if params is None else params.get("num", random.randint(1, 3))
        for _ in range(num):
            idx = random.randint(0, len(xs) - 1)
            cx, cy = xs[idx], ys[idx]
            bw = random.randint(18, 55) if params is None else params.get("bw", random.randint(18, 55))
            bh = random.randint(10, 28) if params is None else params.get("bh", random.randint(10, 28))

            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w, x1 + bw), min(h, y1 + bh)
            patch_mean, patch_std = random.randint(175, 220), random.randint(3, 10)
            patch = np.random.normal(patch_mean, patch_std, (y2 - y1, x2 - x1)).astype(np.uint8)
            if random.random() < 0.5:
                patch[:] = np.clip(
                    np.random.normal(patch_mean, max(2, patch_std // 2), (y2 - y1, x2 - x1)),
                    0, 255
                ).astype(np.uint8)
            img_out[y1:y2, x1:x2] = patch
        return img_out, mask


def generate_base_sample(image_height, image_width):
    base = np.random.normal(205, 6, (image_height, image_width)).astype(np.float32)
    grad = np.tile(np.linspace(-18, 18, image_width, dtype=np.float32), (image_height, 1)) if random.random() < 0.5 else np.tile(np.linspace(-18, 18, image_height, dtype=np.float32).reshape(image_height, 1), (1, image_width))
    img = base + grad
    cx, cy = random.randint(image_width // 4, 3 * image_width // 4), random.randint(image_height // 4, 3 * image_height // 4)
    theta = random.uniform(0, np.pi)
    yy, xx = np.mgrid[0:image_height, 0:image_width].astype(np.float32)
    s = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    d = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    curve_amp = random.uniform(2.0, 8.0)
    curve_period = random.uniform(180, 420)
    curve_phase = random.uniform(0, 2 * np.pi)

    quad = random.uniform(-1e-4, 1e-4) * (s ** 2)
    center_offset = curve_amp * np.sin(2 * np.pi * s / curve_period + curve_phase) + quad
    base_width = random.uniform(3.0, 8.0)
    width_amp = random.uniform(0.4, 1.5)
    width_period = random.uniform(100, 240)
    width_phase = random.uniform(0, 2 * np.pi)

    width_map = base_width + width_amp * np.sin(2 * np.pi * s / width_period + width_phase)
    width_map = np.clip(width_map, 2.0, 10.0)

    asym = random.uniform(-0.6, 0.6)
    jitter = np.random.normal(0, 0.35, (image_height, image_width)).astype(np.float32)
    jitter = cv2.GaussianBlur(jitter, (0, 0), 1.2)

    d_eff = d - center_offset + jitter
    left_bound = -(width_map / 2.0) + asym
    right_bound = (width_map / 2.0) + asym

    seam_region = (d_eff >= left_bound) & (d_eff <= right_bound)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    mask[seam_region] = 255
    sigma_valley = np.maximum(width_map / 2.3, 1.0)
    valley = np.exp(-((d_eff - asym) ** 2) / (2 * sigma_valley ** 2))
    img = img - valley * random.uniform(85, 120)
    edge_sigma = random.uniform(0.8, 1.6)
    edge_offset = width_map / 2.0 + random.uniform(0.6, 1.5)

    edge1 = np.exp(-((d_eff - edge_offset) ** 2) / (2 * edge_sigma ** 2))
    edge2 = np.exp(-((d_eff + edge_offset) ** 2) / (2 * edge_sigma ** 2))
    img = img + (edge1 + edge2) * random.uniform(5, 14)

    img += np.random.normal(0, 2.5, (image_height, image_width)).astype(np.float32)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask


def generate_dataset(cfg=GENERATE_IMG_CONFIG):
    augmenter = Augmenter()
    label_rows = []
    for i in range(cfg["num_samples"]):
        img, mask = generate_base_sample(cfg["image_height"], cfg["image_width"])
        img, mask = augmenter.apply_random_augmentations(img, mask)
        image_name, mask_name = f"img_{i:04d}.png", f"mask_{i:04d}.png"
        cv2.imwrite(os.path.join(IMAGE_DIR_PATH, image_name), img)
        cv2.imwrite(os.path.join(MASK_DIR_PATH, mask_name), mask)
        gt_geometry = extract_seam_geometry(
            image=None,
            mask=mask,
            min_width=ANALYZE_SEAM_CONFIG["min_width"],
            threshold=ANALYZE_SEAM_CONFIG["threshold"],
            kernel_size=ANALYZE_SEAM_CONFIG["kernel_size"],
            keep_largest_component=ANALYZE_SEAM_CONFIG["keep_largest_component"],
        )
        label_rows.append(
            {
                "image_name": image_name,
                "mask_name": mask_name,
                "gt_width": gt_geometry["summary"]["mean_width"],
                "gt_mean_width": gt_geometry["summary"]["mean_width"],
                "gt_max_width": gt_geometry["summary"]["max_width"],
                "gt_min_width": gt_geometry["summary"]["min_width"],
                "gt_std_width": gt_geometry["summary"]["std_width"],
            }
        )

    with open(LABELS_CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "mask_name",
                "gt_width",
                "gt_mean_width",
                "gt_max_width",
                "gt_min_width",
                "gt_std_width",
            ],
        )
        writer.writeheader()
        writer.writerows(label_rows)
    return label_rows

if __name__ == "__main__":
    generate_dataset(GENERATE_IMG_CONFIG)
