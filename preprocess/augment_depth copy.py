#!/usr/bin/env python3
"""Depth Noise Augmentation Pipeline for DTTD Datasets."""

import argparse
from pathlib import Path
import shutil
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm


# ==============================================================================
# I/O & Utility Helper Functions
# ==============================================================================

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def read_train_list(train_txt: Path) -> List[str]:
    """Read non-empty lines from dataset split text file."""
    return [
        line.strip()
        for line in train_txt.read_text().splitlines()
        if line.strip()
    ]


# ==============================================================================
# Depth Processing Utilities
# ==============================================================================

def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert input array into contiguous 2D float32 depth map with NaN/Inf zeroed out."""
    depth = np.squeeze(np.asarray(depth))

    if depth.ndim == 0:
        depth = depth.reshape(1, 1)
    elif depth.ndim == 1:
        depth = depth.reshape(1, -1)
    elif depth.ndim == 3:
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


# ==============================================================================
# Low-Level Depth File Readers
# ==============================================================================

def read_openexr(path: Path) -> np.ndarray:
    """Read float32 depth map using OpenEXR bindings."""
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
        raise RuntimeError(
            f"Unexpected pixel count: {values.size} != {expected}"
        )

    return values.reshape(height, width)


def read_openimageio(path: Path) -> np.ndarray:
    """Read float32 depth map using OpenImageIO."""
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
        raise RuntimeError(
            f"Unexpected pixel count: {values.size} != {expected}"
        )

    values = values.reshape(int(spec.height), int(spec.width), channels)
    return values[..., 0]


def read_depth_image(path: Path) -> np.ndarray:
    """Universal depth reader supporting EXR, PNG, and imageio formats."""
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


# ==============================================================================
# Depth Noise Augmenter Engine
# ==============================================================================

class DepthNoiseAugmenter:
    """Applies realistic sensor noise models to clean depth maps."""

    def __init__(
        self,
        focal_length: float = 399.0,
        baseline: float = 0.09,
        max_depth: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        if focal_length <= 0 or baseline <= 0 or max_depth <= 0:
            raise ValueError(
                "focal_length, baseline, and max_depth must be strictly positive"
            )

        self.f = float(focal_length)
        self.b = float(baseline)
        self.max_depth = float(max_depth)
        self._fb = self.f * self.b
        self.rng = np.random.default_rng(seed)

    def _valid(self, depth: np.ndarray) -> np.ndarray:
        return np.isfinite(depth) & (depth > 0.0) & (depth <= self.max_depth)

    def apply_quantization(
        self, depth: np.ndarray, subpixel_bits: int = 4
    ) -> np.ndarray:
        """Apply disparity-space quantization based on subpixel precision."""
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

    def apply_quadratic_noise(
        self, depth: np.ndarray, noise_multiplier: float = 0.0015
    ) -> np.ndarray:
        """Apply distance-dependent Gaussian noise using stereo model (sigma = k * Z^2)."""
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
        depth: np.ndarray,
        edge_threshold: float = 0.5,
        blur_kernel_size: int = 15,
        dilation_kernel_size: int = 5,
        reference_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulate disparity smearing artifacts around foreground edges."""
        depth = sanitize_depth(depth)
        reference = sanitize_depth(
            reference_depth if reference_depth is not None else depth
        )

        valid_ref = self._valid(reference)
        edge = np.zeros_like(valid_ref, dtype=np.uint8)

        # Detect depth discontinuities
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

        kernel_dim = (blur_kernel_size, blur_kernel_size)
        blurred_disparity = cv2.GaussianBlur(disparity, kernel_dim, 0)
        blurred_weights = cv2.GaussianBlur(weights, kernel_dim, 0)
        blurred_disparity /= np.maximum(blurred_weights, 1e-6)

        out = depth.copy()
        replace = smear_zone & valid & (blurred_disparity > 0)
        out[replace] = self._fb / blurred_disparity[replace]
        out[~valid] = 0

        return np.clip(out, 0, self.max_depth).astype(np.float32)

    def apply_occlusion_shadows(
        self,
        depth: np.ndarray,
        shadow_threshold: float = 0.3,
        max_shadow_width: int = 64,
        direction: int = 1,
        reference_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Simulate stereo occlusion shadow zones."""
        depth = sanitize_depth(depth)
        reference = sanitize_depth(
            reference_depth if reference_depth is not None else depth
        )

        valid = self._valid(reference)
        h, w = depth.shape
        shadow_mask = np.zeros((h, w), dtype=bool)

        if direction == 1:
            near, far = reference[:, :-1], reference[:, 1:]
            transitions = (
                valid[:, :-1] & valid[:, 1:] & (far > near + shadow_threshold)
            )

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
            transitions = (
                valid[:, :-1] & valid[:, 1:] & (far > near + shadow_threshold)
            )

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
        perfect_depth: np.ndarray,
        subpixel_bits: int = 4,
        noise_multiplier: float = 0.0015,
        edge_threshold: float = 0.5,
        blur_kernel_size: int = 15,
        dilation_kernel_size: int = 5,
        shadow_threshold: float = 0.3,
        max_shadow_width: int = 64,
        occlusion_direction: int = 1,
    ) -> np.ndarray:
        """Process clean depth through the complete noise pipeline."""
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


# ==============================================================================
# Pipeline Save / Load Helpers
# ==============================================================================

def load_original_depth(path: Path) -> np.ndarray:
    """Load depth map from disk and convert to float32 meters."""
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


def save_dttd_depth(path: Path, depth_m: np.ndarray) -> None:
    """Save depth map in millimeter uint16 PNG format."""
    depth_m = sanitize_depth(depth_m)
    depth_mm = np.rint(depth_m * 1000.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"Failed to save depth to {path}")


# ==============================================================================
# Frame Augmentation Execution
# ==============================================================================

def augment_frame(
    src_data_dir: Path,
    dst_data_dir: Path,
    stem: str,
    depth_augmenter: DepthNoiseAugmenter,
) -> bool:
    """Augment single dataset frame and sync color/label/meta files."""
    ensure_dir(dst_data_dir)

    # 1. RGB (Copy without modification)
    color_suffixes = ["_color.jpg", "_color.jpeg", "_color.png"]
    color_src = next(
        (
            src_data_dir / f"{stem}{suffix}"
            for suffix in color_suffixes
            if (src_data_dir / f"{stem}{suffix}").exists()
        ),
        None,
    )

    if color_src is None:
        print(f"[WARNING] RGB missing: {stem}")
        return False

    shutil.copy2(color_src, dst_data_dir / color_src.name)

    # 2. Depth Augmentation
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

    # 3. Copy Annotations & Metadata Files (_label.png, _meta.json)
    for suffix in ("_label.png", "_meta.json"):
        extra_file = src_data_dir / f"{stem}{suffix}"
        if extra_file.exists():
            shutil.copy2(extra_file, dst_data_dir / extra_file.name)

    return True


# ==============================================================================
# Global Dataset Handler
# ==============================================================================

def augment_dataset(
    dataset_dir: str,
    focal_length: float = 399.0,
    baseline: float = 0.09,
    max_depth: float = 10.0,
) -> None:
    """Augment dataset train split using depth noise pipeline."""
    root = Path(dataset_dir)
    data_root = root / "data"
    config_dir = root / "dataset_config"

    if not data_root.exists():
        raise RuntimeError(f"Missing data folder: {data_root}")

    train_txt = config_dir / "train_data_list.txt"
    test_txt = config_dir / "test_data_list.txt"

    if not train_txt.exists():
        raise RuntimeError(f"Missing config file: {train_txt}")

    train_lines = read_train_list(train_txt)
    test_lines = read_train_list(test_txt) if test_txt.exists() else []

    depth_augmenter = DepthNoiseAugmenter(
        focal_length=focal_length,
        baseline=baseline,
        max_depth=max_depth,
    )

    augmented_train_lines = []
    print("[INFO] Generate depth-only augmented train data...")
    print(f"[INFO] Original train frames: {len(train_lines)}")

    for line in tqdm(train_lines, desc="Depth augmentation"):
        parts = line.split("/")
        if len(parts) < 3:
            print(f"[WARNING] Invalid train entry: {line}")
            continue

        scene_name = parts[0]
        stem = parts[-1]

        aug_scene_name = f"{scene_name}_depth"
        src_data_dir = data_root / scene_name / "data"
        dst_data_dir = data_root / aug_scene_name / "data"

        if augment_frame(src_data_dir, dst_data_dir, stem, depth_augmenter):
            augmented_train_lines.append(f"{aug_scene_name}/data/{stem}")

    # Write dataset config split lists
    print("\n[INFO] Saving dataset configuration files...")
    train_content = "\n".join(augmented_train_lines) + (
        "\n" if augmented_train_lines else ""
    )
    test_content = "\n".join(test_lines) + ("\n" if test_lines else "")

    (config_dir / "train_data_list.txt").write_text(train_content)
    (config_dir / "test_data_list.txt").write_text(test_content)

    # Print Processing Summary
    print("\n[DONE]")
    print(f"Original train frames: {len(train_lines)}")
    print(f"Depth augmented frames: {len(augmented_train_lines)}")
    print(f"Final train frames:     {len(augmented_train_lines)}")
    print(f"Test frames:            {len(test_lines)}")
    print(f"Output directory:       {root}")


# ==============================================================================
# Entry Point Execution
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth map noise augmentation tool."
    )
    parser.add_argument(
        "--dataset_dir", required=True, help="Path to DTTD dataset directory"
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=399.0,
        help="Stereo focal length in pixels",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.09,
        help="Stereo baseline in meters",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid depth boundary in meters",
    )

    args = parser.parse_args()

    augment_dataset(
        args.dataset_dir,
        focal_length=args.focal_length,
        baseline=args.baseline,
        max_depth=args.max_depth,
    )