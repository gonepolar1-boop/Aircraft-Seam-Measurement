import numpy as np


def build_demo_mask(height=128, width=128, seam_center_x=64, seam_width_px=8):
    mask = np.zeros((height, width), dtype=np.uint8)
    left_x, right_x = max(0, seam_center_x - seam_width_px // 2), min(width, seam_center_x + seam_width_px // 2)
    mask[:, left_x:right_x] = 255
    return mask


def build_demo_image(mask):
    ys, xs = np.indices((mask.shape), dtype=np.float32)
    image = (180.0 + 5.0 * np.sin(xs / 15.0) + 3.0 * np.cos(ys / 21.0)).astype(np.float32)  # 增加背景噪声 
    image[mask > 0] -= 60.0
    return np.clip(image, 0, 255).astype(np.uint8)


def build_demo_point_map(mask, base_flush=0.6, left_tilt_x=0.002, right_tilt_x=-0.001, tilt_y=0.001):
    ys, xs = np.indices((mask.shape), dtype=np.float32)
    point_map = np.zeros((*mask.shape, 3), dtype=np.float32)
    point_map[..., 0], point_map[..., 1] = xs, ys

    seam_columns = np.where(mask.any(axis=0))[0]  # [60, 61, ..., 67]
    seam_center_x = float(0.5 * (seam_columns[0] + seam_columns[-1]))

    left_surface, right_surface = 0.0 + left_tilt_x * xs + tilt_y * ys, base_flush + right_tilt_x * xs + tilt_y * ys
    point_map[..., 2] = np.where(xs < seam_center_x, left_surface, right_surface)
    return point_map
