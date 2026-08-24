#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_train_list(train_txt: Path):
    return [line.strip() for line in train_txt.read_text().splitlines() if line.strip()]


def augment_brightness(img):
    alpha = np.random.uniform(0.7, 1.4)
    beta = np.random.uniform(-35, 35)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def add_gaussian_noise(img):
    noise_sigma = np.random.uniform(10, 30)
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_delta = np.random.randint(-12, 13)
    s_scale = np.random.uniform(0.75, 1.25)
    v_scale = np.random.uniform(0.75, 1.25)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_gamma(img):
    gamma = np.random.uniform(0.5, 1.5)
    img_f = img.astype(np.float32) / 255.0
    out = np.power(img_f, gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.squeeze(np.asarray(depth))
    if depth.ndim == 0:
        depth = depth.reshape(1, 1)
    elif depth.ndim == 1:
        depth = depth.reshape(1, -1)
    if depth.ndim == 3:
        if depth.shape[-1] in (1, 3, 4):
            depth = depth[..., 0]
        elif depth.shape[0] in (1, 3, 4):
            depth = depth[0, ...]
        else:
            raise ValueError(f"Unsupported depth shape: {depth.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D, got {depth.shape}")
    depth = np.array(depth, dtype=np.float32, order="C", copy=True)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def read_openexr(path: Path):
    import Imath
    import OpenEXR

    input_file = OpenEXR.InputFile(str(path))
    try:
        header = input_file.header()
        channels = list(header["channels"].keys())
        if not channels:
            raise RuntimeError(f"No channels found: {path}")
        channel_name = next(
            (name for name in ("Z", "Y", "R") if name in channels), channels[0]
        )
        window = header["dataWindow"]
        width = int(window.max.x - window.min.x + 1)
        height = int(window.max.y - window.min.y + 1)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = input_file.channel(channel_name, pixel_type)
    finally:
        input_file.close()

    values = np.frombuffer(raw, dtype=np.float32)
    expected = width * height
    if values.size != expected:
        raise RuntimeError(f"Unexpected pixel count: {values.size} != {expected}")
    return values.reshape(height, width)


def read_openimageio(path: Path):
    import OpenImageIO as oiio

    input_file = oiio.ImageInput.open(str(path))
    if input_file is None:
        raise RuntimeError(f"Could not open {path}")
    try:
        spec = input_file.spec()
        raw = input_file.read_image(oiio.FLOAT)
    finally:
        input_file.close()

    channels = max(1, int(spec.nchannels))
    values = np.asarray(raw, dtype=np.float32)
    expected = int(spec.width) * int(spec.height) * channels
    if values.size != expected:
        raise RuntimeError(f"Unexpected pixel count: {values.size} != {expected}")
    values = values.reshape(int(spec.height), int(spec.width), channels)
    return values[..., 0]


def read_depth_image(path: Path):
    path = Path(path)
    errors = []
    if path.suffix.lower() == ".exr":
        for reader in (read_openexr, read_openimageio):
            try:
                return sanitize_depth(reader(path))
            except (ImportError, ModuleNotFoundError):
                continue
            except Exception as exc:
                errors.append(f"{reader.__name__}: {exc}")

    try:
        import imageio.v3 as iio

        return sanitize_depth(iio.imread(path))
    except Exception as exc:
        detail = "; ".join(errors)
        detail = f"{detail}; imageio: {exc}" if detail else str(exc)
        raise RuntimeError(f"Could not read depth {path}: {detail}") from exc


class DepthNoiseAugmenter:
    def __init__(
        self,
        focal_length: float = 399.0,
        baseline: float = 0.09,
        max_depth: float = 10.0,
        seed: Optional[int] = None,
    ):
        if focal_length <= 0 or baseline <= 0 or max_depth <= 0:
            raise ValueError("focal_length, baseline, and max_depth must be positive")
        self.f = float(focal_length)
        self.b = float(baseline)
        self.max_depth = float(max_depth)
        self._fb = self.f * self.b
        self.rng = np.random.default_rng(seed)

    def _valid(self, depth):
        return np.isfinite(depth) & (depth > 0.0) & (depth <= self.max_depth)

    def apply_quantization(self, depth, subpixel_bits=4):
        depth = sanitize_depth(depth)
        valid = self._valid(depth)
        out = np.zeros_like(depth, dtype=np.float32)
        if not np.any(valid):
            return out

        disparity = self._fb / depth[valid]
        disparity_step = 1.0 / float(2**subpixel_bits)
        quantized = np.rint(disparity / disparity_step) * disparity_step
        representable = quantized > 0

        quantized_depth = np.zeros_like(disparity, dtype=np.float32)
        quantized_depth[representable] = self._fb / quantized[representable]
        out[valid] = quantized_depth
        return out

    def apply_quadratic_noise(self, depth, noise_multiplier=0.0015):
        depth = sanitize_depth(depth)
        valid = self._valid(depth)
        out = np.zeros_like(depth, dtype=np.float32)
        if not np.any(valid):
            return out

        sigma = noise_multiplier * np.square(depth[valid])
        noisy = depth[valid] + self.rng.normal(0, sigma).astype(np.float32)
        noisy = np.where(np.isfinite(noisy) & (noisy > 0), noisy, 0)
        out[valid] = np.clip(noisy, 0, self.max_depth)
        return out

    def apply_flying_pixels(
        self,
        depth,
        edge_threshold=0.5,
        blur_kernel_size=15,
        dilation_kernel_size=5,
        reference_depth=None,
    ):
        depth = sanitize_depth(depth)
        reference = sanitize_depth(
            reference_depth if reference_depth is not None else depth
        )
        valid_ref = self._valid(reference)

        edge = np.zeros_like(valid_ref, dtype=np.uint8)
        horizontal = (
            valid_ref[:, :-1]
            & valid_ref[:, 1:]
            & (np.abs(reference[:, 1:] - reference[:, :-1]) > edge_threshold)
        )
        vertical = (
            valid_ref[:-1, :]
            & valid_ref[1:, :]
            & (np.abs(reference[1:, :] - reference[:-1, :]) > edge_threshold)
        )

        edge[:, :-1] |= horizontal
        edge[:, 1:] |= horizontal
        edge[:-1, :] |= vertical
        edge[1:, :] |= vertical

        dilation_kernel = np.ones(
            (dilation_kernel_size, dilation_kernel_size), dtype=np.uint8
        )
        smear_zone = cv2.dilate(edge, dilation_kernel, iterations=1) > 0

        valid = self._valid(depth)
        disparity = np.zeros_like(depth, dtype=np.float32)
        disparity[valid] = self._fb / depth[valid]

        weights = valid.astype(np.float32)
        blurred_disparity = cv2.GaussianBlur(
            disparity, (blur_kernel_size, blur_kernel_size), 0
        )
        blurred_weights = cv2.GaussianBlur(
            weights, (blur_kernel_size, blur_kernel_size), 0
        )
        blurred_disparity /= np.maximum(blurred_weights, 1e-6)

        out = depth.copy()
        replace = smear_zone & valid & (blurred_disparity > 0)
        out[replace] = self._fb / blurred_disparity[replace]
        out[~valid] = 0

        return np.clip(out, 0, self.max_depth).astype(np.float32)

    def apply_occlusion_shadows(
        self,
        depth,
        shadow_threshold=0.3,
        max_shadow_width=64,
        direction=1,
        reference_depth=None,
    ):
        depth = sanitize_depth(depth)
        reference = sanitize_depth(
            reference_depth if reference_depth is not None else depth
        )
        valid = self._valid(reference)

        h, w = depth.shape
        shadow_mask = np.zeros((h, w), dtype=bool)

        if direction == 1:
            near, far = reference[:, :-1], reference[:, 1:]
            transitions = valid[:, :-1] & valid[:, 1:] & (far > near + shadow_threshold)

            for row in range(h):
                starts = np.flatnonzero(transitions[row])
                if starts.size == 0:
                    continue

                widths = np.ceil(
                    self._fb * (1.0 / near[row, starts] - 1.0 / far[row, starts])
                ).astype(np.int32)
                widths = np.clip(widths, 1, max_shadow_width)

                difference = np.zeros(w + 1, dtype=np.int32)
                starts = starts + 1
                ends = np.minimum(w - 1, starts + widths - 1)

                difference[starts] += 1
                difference[ends + 1] -= 1
                shadow_mask[row] = np.cumsum(difference[:-1]) > 0
        else:
            near, far = reference[:, 1:], reference[:, :-1]
            transitions = valid[:, :-1] & valid[:, 1:] & (far > near + shadow_threshold)

            for row in range(h):
                starts = np.flatnonzero(transitions[row])
                if starts.size == 0:
                    continue

                widths = np.ceil(
                    self._fb * (1.0 / near[row, starts] - 1.0 / far[row, starts])
                ).astype(np.int32)
                widths = np.clip(widths, 1, max_shadow_width)

                difference = np.zeros(w + 1, dtype=np.int32)
                ends = starts
                starts = np.maximum(0, starts - widths)

                difference[starts] += 1
                difference[ends] -= 1
                shadow_mask[row] = np.cumsum(difference[:-1]) > 0

        out = depth.copy()
        out[shadow_mask] = 0
        return out.astype(np.float32)

    def process(
        self,
        perfect_depth,
        subpixel_bits=4,
        noise_multiplier=0.0015,
        edge_threshold=0.5,
        blur_kernel_size=15,
        dilation_kernel_size=5,
        shadow_threshold=0.3,
        max_shadow_width=64,
        occlusion_direction=1,
    ):
        source = sanitize_depth(perfect_depth)
        depth = self.apply_quantization(source, subpixel_bits)
        depth = self.apply_quadratic_noise(depth, noise_multiplier)
        depth = self.apply_flying_pixels(
            depth,
            edge_threshold,
            blur_kernel_size,
            dilation_kernel_size,
            reference_depth=source,
        )
        depth = self.apply_occlusion_shadows(
            depth,
            shadow_threshold,
            max_shadow_width,
            occlusion_direction,
            reference_depth=source,
        )
        return depth


def load_original_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Cannot read depth: {path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype == np.uint16:
        depth_m = depth.astype(np.float32) / 1000.0
    elif np.issubdtype(depth.dtype, np.floating):
        depth_m = depth.astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported depth dtype: {depth.dtype}")

    return sanitize_depth(depth_m)


def save_dttd_depth(path: Path, depth_m: np.ndarray):
    depth_m = sanitize_depth(depth_m)
    depth_mm = np.rint(depth_m * 1000.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)


def augment_frame(src_data_dir, dst_data_dir, stem, rgb_aug_fn, depth_augmenter):
    ensure_dir(dst_data_dir)

    color_suffixes = ["_color.jpg", "_color.jpeg", "_color.png"]
    color_src = None
    for suffix in color_suffixes:
        candidate = src_data_dir / f"{stem}{suffix}"
        if candidate.exists():
            color_src = candidate
            break

    if color_src is None:
        print(f"[WARNING] RGB missing: {stem}")
        return False

    img = cv2.imread(str(color_src), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARNING] Cannot read RGB: {color_src}")
        return False

    cv2.imwrite(str(dst_data_dir / color_src.name), rgb_aug_fn(img))

    depth_src = src_data_dir / f"{stem}_depth.png"
    if not depth_src.exists():
        print(f"[WARNING] Depth missing: {depth_src}")
        return False

    try:
        depth_m = load_original_depth(depth_src)
        aug_depth_m = depth_augmenter.process(depth_m)
        save_dttd_depth(dst_data_dir / f"{stem}_depth.png", aug_depth_m)
    except Exception as e:
        print(f"[WARNING] Depth augmentation failed: {depth_src}\n          {e}")
        return False

    label_src = src_data_dir / f"{stem}_label.png"
    if label_src.exists():
        shutil.copy2(label_src, dst_data_dir / label_src.name)

    meta_src = src_data_dir / f"{stem}_meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, dst_data_dir / meta_src.name)

    return True


def augment_dataset(dataset_dir, focal_length=399.0, baseline=0.09, max_depth=10.0):
    root = Path(dataset_dir)
    data_root = root / "data"
    config_dir = root / "dataset_config"

    if not data_root.exists():
        raise RuntimeError(f"Missing data folder: {data_root}")

    train_txt = config_dir / "train_data_list.txt"
    test_txt = config_dir / "test_data_list.txt"

    if not train_txt.exists():
        raise RuntimeError(f"Missing: {train_txt}")

    train_lines = read_train_list(train_txt)
    test_lines = read_train_list(test_txt) if test_txt.exists() else []

    depth_augmenter = DepthNoiseAugmenter(
        focal_length=focal_length, baseline=baseline, max_depth=max_depth
    )

    augmenters = [
        ("bright", augment_brightness),
        ("noise", add_gaussian_noise),
        ("hsv", adjust_hsv),
        ("gamma", adjust_gamma),
    ]

    new_train_lines = list(train_lines)
    print("[INFO] Generate augmented train data...")
    print(f"[INFO] Original train frames: {len(train_lines)}")

    for aug_name, rgb_aug_fn in augmenters:
        print(f"\n[INFO] augmentation: {aug_name} + depth_realistic")
        for line in tqdm(train_lines, desc=aug_name):
            parts = line.split("/")
            if len(parts) < 3:
                continue

            scene_name, stem = parts[0], parts[-1]
            aug_scene_name = f"{scene_name}_{aug_name}"
            src_data_dir = data_root / scene_name / "data"
            dst_data_dir = data_root / aug_scene_name / "data"

            success = augment_frame(
                src_data_dir, dst_data_dir, stem, rgb_aug_fn, depth_augmenter
            )
            if success:
                new_train_lines.append(f"{aug_scene_name}/data/{stem}")

    print("\n[INFO] Write dataset_config...")
    (config_dir / "train_data_list.txt").write_text(
        "\n".join(new_train_lines) + ("\n" if new_train_lines else "")
    )
    (config_dir / "test_data_list.txt").write_text(
        "\n".join(test_lines) + ("\n" if test_lines else "")
    )

    print("\n[DONE]")
    print(f"Original train frames: {len(train_lines)}")
    print(f"Augmented train frames: {len(new_train_lines) - len(train_lines)}")
    print(f"Final train frames: {len(new_train_lines)}")
    print(f"Output: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="DTTD dataset folder")
    parser.add_argument(
        "--focal-length",
        type=float,
        default=399.0,
        help="Stereo focal length in pixels",
    )
    parser.add_argument(
        "--baseline", type=float, default=0.09, help="Stereo baseline in meters"
    )
    parser.add_argument(
        "--max-depth", type=float, default=10.0, help="Maximum valid depth in meters"
    )
    args = parser.parse_args()

    augment_dataset(
        args.dataset_dir,
        focal_length=args.focal_length,
        baseline=args.baseline,
        max_depth=args.max_depth,
    )
