#!/usr/bin/env python3
"""
OAK4 StereoDepth + Grayscale Augmentation Pipeline for DTTD Dataset.

Important:
    Every frame is processed independently.

    There is NO temporal state between frames.
    Frame N does not affect Frame N+1.

RGB:
    RGB -> grayscale -> 3-channel BGR JPEG

Depth:
    Clean depth
        -> valid range clipping
        -> disparity subpixel quantization
        -> quadratic stereo noise
        -> flying pixels
        -> occlusion shadows
        -> median filtering
        -> spatial filtering
        -> speckle filtering
        -> save as uint16 millimeter PNG

Train and test are processed independently.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm


# ==============================================================================
# OAK4 StereoDepth Configuration
# ==============================================================================

# These values follow the direction of the actual OAK4 configuration.
#
# They are augmentation parameters, not a replacement for the actual device
# calibration.

DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 5000

DEPTH_CONFIDENCE = 200

LR_CHECK = True
LR_CHECK_THRESHOLD = 10

EXTENDED_DISPARITY = True
DISPARITY_SHIFT = 0

SUBPIXEL = True
SUBPIXEL_FRACTIONAL_BITS = 3

MEDIAN_FILTER = "KERNEL_5x5"

SPECKLE_FILTER = True
SPECKLE_RANGE = 48
SPECKLE_DIFFERENCE_THRESHOLD = 2

SPATIAL_FILTER = True
SPATIAL_ALPHA = 0.5
SPATIAL_DELTA = 8
SPATIAL_HOLE_FILLING_RADIUS = 2
SPATIAL_ITERATIONS = 1

TEMPORAL_FILTER = False

DECIMATION_FACTOR = 1


# ==============================================================================
# I/O
# ==============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_train_list(train_txt: Path) -> List[str]:
    return [
        line.strip()
        for line in train_txt.read_text().splitlines()
        if line.strip()
    ]


# ==============================================================================
# Grayscale
# ==============================================================================

def convert_to_grayscale_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB/BGR image to grayscale while keeping 3 channels.

    Output:
        H x W x 3 uint8

    B == G == R
    """

    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY,
        )

    else:
        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )


# ==============================================================================
# Depth Utility
# ==============================================================================

def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Convert depth to contiguous float32 2D array.
    """

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
            raise ValueError(
                f"Unsupported depth shape: {depth.shape}"
            )

    if depth.ndim != 2:
        raise ValueError(
            f"Depth must be 2D, got {depth.shape}"
        )

    depth = np.array(
        depth,
        dtype=np.float32,
        order="C",
        copy=True,
    )

    depth[~np.isfinite(depth)] = 0.0

    return depth


# ==============================================================================
# Depth Reader
# ==============================================================================

def read_openexr(path: Path) -> np.ndarray:

    import Imath
    import OpenEXR

    input_file = OpenEXR.InputFile(str(path))

    try:

        header = input_file.header()

        channels = list(
            header["channels"].keys()
        )

        if not channels:
            raise RuntimeError(
                f"No channels found: {path}"
            )

        channel_name = next(
            (
                name
                for name in ("Z", "Y", "R")
                if name in channels
            ),
            channels[0],
        )

        window = header["dataWindow"]

        width = int(
            window.max.x -
            window.min.x +
            1
        )

        height = int(
            window.max.y -
            window.min.y +
            1
        )

        pixel_type = Imath.PixelType(
            Imath.PixelType.FLOAT
        )

        raw = input_file.channel(
            channel_name,
            pixel_type,
        )

    finally:
        input_file.close()

    values = np.frombuffer(
        raw,
        dtype=np.float32,
    )

    expected = width * height

    if values.size != expected:
        raise RuntimeError(
            f"Unexpected pixel count: "
            f"{values.size} != {expected}"
        )

    return values.reshape(
        height,
        width,
    )


def read_openimageio(path: Path) -> np.ndarray:

    import OpenImageIO as oiio

    input_file = oiio.ImageInput.open(
        str(path)
    )

    if input_file is None:
        raise RuntimeError(
            f"Could not open {path}"
        )

    try:

        spec = input_file.spec()

        raw = input_file.read_image(
            oiio.FLOAT
        )

    finally:
        input_file.close()

    channels = max(
        1,
        int(spec.nchannels),
    )

    values = np.asarray(
        raw,
        dtype=np.float32,
    )

    expected = (
        int(spec.width)
        * int(spec.height)
        * channels
    )

    if values.size != expected:
        raise RuntimeError(
            f"Unexpected pixel count: "
            f"{values.size} != {expected}"
        )

    values = values.reshape(
        int(spec.height),
        int(spec.width),
        channels,
    )

    return values[..., 0]


def read_depth_image(path: Path) -> np.ndarray:

    path = Path(path)

    errors = []

    if path.suffix.lower() == ".exr":

        for reader in (
            read_openexr,
            read_openimageio,
        ):

            try:
                return sanitize_depth(
                    reader(path)
                )

            except (
                ImportError,
                ModuleNotFoundError,
            ):
                continue

            except Exception as exc:
                errors.append(
                    f"{reader.__name__}: {exc}"
                )

    try:

        import imageio.v3 as iio

        return sanitize_depth(
            iio.imread(path)
        )

    except Exception as exc:

        detail = "; ".join(errors)

        if detail:
            detail = (
                f"{detail}; "
                f"imageio: {exc}"
            )

        else:
            detail = str(exc)

        raise RuntimeError(
            f"Could not read depth "
            f"{path}: {detail}"
        ) from exc


# ==============================================================================
# OAK4 Depth Noise Augmenter
# ==============================================================================

class DepthNoiseAugmenter:
    """
    Simulate OAK4 StereoDepth behavior.

    IMPORTANT:
        This class contains NO temporal state.

        Every call to process() is independent.
    """

    def __init__(
        self,
        focal_length: float = 399.0,
        baseline: float = 0.09,
        max_depth: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:

        if focal_length <= 0:
            raise ValueError(
                "focal_length must be > 0"
            )

        if baseline <= 0:
            raise ValueError(
                "baseline must be > 0"
            )

        if max_depth <= 0:
            raise ValueError(
                "max_depth must be > 0"
            )

        self.f = float(
            focal_length
        )

        self.b = float(
            baseline
        )

        self.max_depth = float(
            max_depth
        )

        self._fb = (
            self.f * self.b
        )

        # ==========================================================
        # IMPORTANT
        #
        # This RNG is only used for this augmenter.
        # No previous depth frame is stored.
        #
        # For strict reproducibility each frame can optionally
        # provide its own seed to process().
        # ==========================================================

        self.rng = np.random.default_rng(
            seed
        )

    # ------------------------------------------------------------------
    # Valid depth
    # ------------------------------------------------------------------

    def _valid(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:

        return (
            np.isfinite(depth)
            & (depth > 0.0)
            & (
                depth
                >= DEPTH_MIN_MM / 1000.0
            )
            & (
                depth
                <= self.max_depth
            )
        )

    # ------------------------------------------------------------------
    # Depth range
    # ------------------------------------------------------------------

    def apply_depth_range(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:

        depth = sanitize_depth(
            depth
        )

        min_depth = (
            DEPTH_MIN_MM / 1000.0
        )

        max_depth = (
            DEPTH_MAX_MM / 1000.0
        )

        valid = (
            np.isfinite(depth)
            & (depth >= min_depth)
            & (depth <= max_depth)
        )

        out = np.zeros_like(
            depth,
            dtype=np.float32,
        )

        out[valid] = depth[valid]

        return out

    # ------------------------------------------------------------------
    # Subpixel disparity quantization
    # ------------------------------------------------------------------

    def apply_quantization(
        self,
        depth: np.ndarray,
        subpixel_bits: int = SUBPIXEL_FRACTIONAL_BITS,
    ) -> np.ndarray:

        depth = sanitize_depth(
            depth
        )

        valid = self._valid(
            depth
        )

        out = np.zeros_like(
            depth,
            dtype=np.float32,
        )

        if not np.any(valid):
            return out

        disparity = (
            self._fb
            / depth[valid]
        )

        # OAK4:
        #
        # SUBPIXEL_FRACTIONAL_BITS = 3
        #
        # Therefore disparity resolution:
        #
        # 1 / 8 disparity pixel

        disparity_step = (
            1.0
            / float(
                2 ** subpixel_bits
            )
        )

        quantized = (
            np.rint(
                disparity
                / disparity_step
            )
            * disparity_step
        )

        representable = (
            quantized > 0
        )

        quantized_depth = np.zeros_like(
            disparity,
            dtype=np.float32,
        )

        quantized_depth[
            representable
        ] = (
            self._fb
            / quantized[
                representable
            ]
        )

        out[valid] = (
            quantized_depth
        )

        return out

    # ------------------------------------------------------------------
    # Stereo measurement noise
    # ------------------------------------------------------------------

    def apply_quadratic_noise(
        self,
        depth: np.ndarray,
        noise_multiplier: float = 0.0012,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:

        depth = sanitize_depth(
            depth
        )

        valid = self._valid(
            depth
        )

        out = np.zeros_like(
            depth,
            dtype=np.float32,
        )

        if not np.any(valid):
            return out

        if rng is None:
            rng = self.rng

        sigma = (
            noise_multiplier
            * np.square(
                depth[valid]
            )
        )

        noisy = (
            depth[valid]
            + rng.normal(
                0.0,
                sigma,
            ).astype(
                np.float32
            )
        )

        noisy = np.where(
            np.isfinite(noisy)
            & (noisy > 0),
            noisy,
            0,
        )

        out[valid] = np.clip(
            noisy,
            0,
            self.max_depth,
        )

        return out

    # ------------------------------------------------------------------
    # Flying pixels
    # ------------------------------------------------------------------

    def apply_flying_pixels(
        self,
        depth: np.ndarray,
        edge_threshold: float = 0.5,
        blur_kernel_size: int = 5,
        dilation_kernel_size: int = 3,
        reference_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        depth = sanitize_depth(
            depth
        )

        reference = sanitize_depth(
            reference_depth
            if reference_depth is not None
            else depth
        )

        valid_ref = self._valid(
            reference
        )

        edge = np.zeros_like(
            valid_ref,
            dtype=np.uint8,
        )

        horizontal = (
            valid_ref[:, :-1]
            & valid_ref[:, 1:]
            & (
                np.abs(
                    reference[:, 1:]
                    - reference[:, :-1]
                )
                > edge_threshold
            )
        )

        vertical = (
            valid_ref[:-1, :]
            & valid_ref[1:, :]
            & (
                np.abs(
                    reference[1:, :]
                    - reference[:-1, :]
                )
                > edge_threshold
            )
        )

        edge[:, :-1] |= horizontal
        edge[:, 1:] |= horizontal

        edge[:-1, :] |= vertical
        edge[1:, :] |= vertical

        dilation_kernel = np.ones(
            (
                dilation_kernel_size,
                dilation_kernel_size,
            ),
            dtype=np.uint8,
        )

        smear_zone = (
            cv2.dilate(
                edge,
                dilation_kernel,
                iterations=1,
            )
            > 0
        )

        valid = self._valid(
            depth
        )

        disparity = np.zeros_like(
            depth,
            dtype=np.float32,
        )

        disparity[valid] = (
            self._fb
            / depth[valid]
        )

        weights = valid.astype(
            np.float32
        )

        blurred_disparity = (
            cv2.GaussianBlur(
                disparity,
                (
                    blur_kernel_size,
                    blur_kernel_size,
                ),
                0,
            )
        )

        blurred_weights = (
            cv2.GaussianBlur(
                weights,
                (
                    blur_kernel_size,
                    blur_kernel_size,
                ),
                0,
            )
        )

        blurred_disparity /= np.maximum(
            blurred_weights,
            1e-6,
        )

        out = depth.copy()

        replace = (
            smear_zone
            & valid
            & (
                blurred_disparity
                > 0
            )
        )

        out[replace] = (
            self._fb
            / blurred_disparity[
                replace
            ]
        )

        out[~valid] = 0

        return np.clip(
            out,
            0,
            self.max_depth,
        ).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Occlusion shadow
    # ------------------------------------------------------------------

    def apply_occlusion_shadows(
        self,
        depth: np.ndarray,
        shadow_threshold: float = 0.3,
        max_shadow_width: int = 48,
        direction: int = 1,
        reference_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        depth = sanitize_depth(
            depth
        )

        reference = sanitize_depth(
            reference_depth
            if reference_depth is not None
            else depth
        )

        valid = self._valid(
            reference
        )

        h, w = depth.shape

        shadow_mask = np.zeros(
            (h, w),
            dtype=bool,
        )

        if direction == 1:

            near = reference[:, :-1]
            far = reference[:, 1:]

            transitions = (
                valid[:, :-1]
                & valid[:, 1:]
                & (
                    far
                    > near
                    + shadow_threshold
                )
            )

            for row in range(h):

                starts = np.flatnonzero(
                    transitions[row]
                )

                if starts.size == 0:
                    continue

                widths = np.ceil(
                    self._fb
                    * (
                        1.0
                        / near[row, starts]
                        - 1.0
                        / far[row, starts]
                    )
                ).astype(
                    np.int32
                )

                widths = np.clip(
                    widths,
                    1,
                    max_shadow_width,
                )

                difference = np.zeros(
                    w + 1,
                    dtype=np.int32,
                )

                starts = starts + 1

                ends = np.minimum(
                    w - 1,
                    starts + widths - 1,
                )

                difference[
                    starts
                ] += 1

                difference[
                    ends + 1
                ] -= 1

                shadow_mask[row] = (
                    np.cumsum(
                        difference[:-1]
                    )
                    > 0
                )

        else:

            near = reference[:, 1:]
            far = reference[:, :-1]

            transitions = (
                valid[:, :-1]
                & valid[:, 1:]
                & (
                    far
                    > near
                    + shadow_threshold
                )
            )

            for row in range(h):

                starts = np.flatnonzero(
                    transitions[row]
                )

                if starts.size == 0:
                    continue

                widths = np.ceil(
                    self._fb
                    * (
                        1.0
                        / near[row, starts]
                        - 1.0
                        / far[row, starts]
                    )
                ).astype(
                    np.int32
                )

                widths = np.clip(
                    widths,
                    1,
                    max_shadow_width,
                )

                difference = np.zeros(
                    w + 1,
                    dtype=np.int32,
                )

                ends = starts

                starts = np.maximum(
                    0,
                    starts - widths,
                )

                difference[
                    starts
                ] += 1

                difference[
                    ends
                ] -= 1

                shadow_mask[row] = (
                    np.cumsum(
                        difference[:-1]
                    )
                    > 0
                )

        out = depth.copy()

        out[shadow_mask] = 0

        return out.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Median filter
    # ------------------------------------------------------------------

    def apply_median_filter(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:

        if MEDIAN_FILTER != "KERNEL_5x5":
            return depth

        depth = sanitize_depth(
            depth
        )

        valid = depth > 0

        # Median filtering must not treat invalid 0 as a real depth.
        #
        # Temporarily replace zero with nearest valid value.
        #
        # This is intentionally simple and deterministic.

        if not np.any(valid):
            return depth

        filled = depth.copy()

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        valid_u8 = valid.astype(
            np.uint8
        )

        weighted = cv2.blur(
            np.where(
                valid,
                filled,
                0,
            ),
            (5, 5),
        )

        count = cv2.blur(
            valid_u8.astype(
                np.float32
            ),
            (5, 5),
        )

        local_mean = (
            weighted
            / np.maximum(
                count,
                1e-6,
            )
        )

        filled[~valid] = (
            local_mean[~valid]
        )

        filtered = cv2.medianBlur(
            filled.astype(
                np.float32
            ),
            5,
        )

        filtered[~valid] = 0

        return filtered.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Spatial filter
    # ------------------------------------------------------------------

    def apply_spatial_filter(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:

        if not SPATIAL_FILTER:
            return depth

        depth = sanitize_depth(
            depth
        )

        result = depth.copy()

        for _ in range(
            SPATIAL_ITERATIONS
        ):

            valid = result > 0

            if not np.any(valid):
                break

            # ------------------------------------------------------
            # Alpha controls smoothing strength.
            #
            # alpha = 0.5
            # approximately follows the OAK4 setting.
            # ------------------------------------------------------

            blurred = cv2.bilateralFilter(
                result.astype(
                    np.float32
                ),
                d=5,
                sigmaColor=float(
                    SPATIAL_DELTA
                ) / 1000.0,
                sigmaSpace=2.0,
            )

            # Difference gating.
            difference = np.abs(
                blurred - result
            )

            allowed = (
                valid
                & (
                    difference
                    <= (
                        SPATIAL_DELTA
                        / 1000.0
                    )
                )
            )

            result[allowed] = (
                (
                    SPATIAL_ALPHA
                    * blurred[allowed]
                )
                + (
                    (
                        1.0
                        - SPATIAL_ALPHA
                    )
                    * result[allowed]
                )
            )

            # ------------------------------------------------------
            # Hole filling
            # ------------------------------------------------------

            if SPATIAL_HOLE_FILLING_RADIUS > 0:

                invalid = (
                    result <= 0
                )

                local = cv2.blur(
                    result,
                    (
                        SPATIAL_HOLE_FILLING_RADIUS
                        * 2
                        + 1,
                        SPATIAL_HOLE_FILLING_RADIUS
                        * 2
                        + 1,
                    ),
                )

                result[
                    invalid
                    & (local > 0)
                ] = local[
                    invalid
                    & (local > 0)
                ]

        return np.clip(
            result,
            0,
            self.max_depth,
        ).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Speckle filter
    # ------------------------------------------------------------------

    def apply_speckle_filter(
        self,
        depth: np.ndarray,
    ) -> np.ndarray:

        if not SPECKLE_FILTER:
            return depth

        depth = sanitize_depth(
            depth
        )

        valid = depth > 0

        if not np.any(valid):
            return depth

        # Convert depth to a compact disparity-like representation.
        disparity = np.zeros_like(
            depth,
            dtype=np.float32,
        )

        disparity[valid] = (
            self._fb
            / depth[valid]
        )

        # Identify small isolated disparity components.
        #
        # This is an approximation of StereoDepth speckle filtering.
        disp_norm = cv2.normalize(
            disparity,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        binary = (
            valid.astype(
                np.uint8
            )
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        output = depth.copy()

        # OAK4 SPECKLE_RANGE = 48.
        #
        # Instead of aggressively deleting every small component,
        # only remove extremely small isolated regions.
        #
        # This avoids destroying legitimate thin objects.

        min_component_area = max(
            2,
            int(
                SPECKLE_RANGE
                / 16
            ),
        )

        for label_id in range(
            1,
            num_labels,
        ):

            area = stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]

            if area <= min_component_area:

                output[
                    labels == label_id
                ] = 0

        return output.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Complete pipeline
    # ------------------------------------------------------------------

    def process(
        self,
        perfect_depth: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Process ONE frame.

        No information from another frame is used.

        seed:
            Optional per-frame deterministic seed.
        """

        # ==========================================================
        # IMPORTANT:
        #
        # Create a local RNG for this frame.
        #
        # This guarantees that this call does not depend on
        # previous frame state.
        # ==========================================================

        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(
                seed
            )

        # ==========================================================
        # 1. Source
        # ==========================================================

        source = sanitize_depth(
            perfect_depth
        )

        # ==========================================================
        # 2. OAK4 depth range
        # ==========================================================

        depth = self.apply_depth_range(
            source
        )

        # ==========================================================
        # 3. Subpixel disparity
        #
        # SUBPIXEL = True
        # SUBPIXEL_FRACTIONAL_BITS = 3
        # ==========================================================

        if SUBPIXEL:

            depth = self.apply_quantization(
                depth,
                SUBPIXEL_FRACTIONAL_BITS,
            )

        # ==========================================================
        # 4. Stereo noise
        # ==========================================================

        depth = self.apply_quadratic_noise(
            depth,
            noise_multiplier=0.0012,
            rng=rng,
        )

        # ==========================================================
        # 5. Flying pixels
        # ==========================================================

        depth = self.apply_flying_pixels(
            depth,
            edge_threshold=0.5,
            blur_kernel_size=5,
            dilation_kernel_size=3,
            reference_depth=source,
        )

        # ==========================================================
        # 6. Occlusion shadows
        #
        # LR_CHECK = True
        # ==========================================================

        if LR_CHECK:

            depth = self.apply_occlusion_shadows(
                depth,
                shadow_threshold=0.3,
                max_shadow_width=48,
                direction=1,
                reference_depth=source,
            )

        # ==========================================================
        # 7. Median 5x5
        # ==========================================================

        depth = self.apply_median_filter(
            depth
        )

        # ==========================================================
        # 8. Spatial filter
        # ==========================================================

        depth = self.apply_spatial_filter(
            depth
        )

        # ==========================================================
        # 9. Speckle filter
        # ==========================================================

        depth = self.apply_speckle_filter(
            depth
        )

        # ==========================================================
        # 10. Final depth range
        # ==========================================================

        depth = np.where(
            (
                depth >=
                DEPTH_MIN_MM / 1000.0
            )
            & (
                depth <=
                DEPTH_MAX_MM / 1000.0
            ),
            depth,
            0.0,
        )

        return depth.astype(
            np.float32
        )


# ==============================================================================
# Original Depth Loader
# ==============================================================================

def load_original_depth(
    path: Path,
) -> np.ndarray:

    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        raise RuntimeError(
            f"Cannot read depth: {path}"
        )

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype == np.uint16:

        depth_m = (
            depth.astype(
                np.float32
            )
            / 1000.0
        )

    elif np.issubdtype(
        depth.dtype,
        np.floating,
    ):

        depth_m = depth.astype(
            np.float32
        )

    else:

        raise RuntimeError(
            f"Unsupported depth dtype: "
            f"{depth.dtype}"
        )

    return sanitize_depth(
        depth_m
    )


# ==============================================================================
# Save Depth
# ==============================================================================

def save_dttd_depth(
    path: Path,
    depth_m: np.ndarray,
) -> None:

    depth_m = sanitize_depth(
        depth_m
    )

    depth_mm = np.rint(
        depth_m * 1000.0
    )

    depth_mm = np.clip(
        depth_mm,
        0,
        65535,
    ).astype(
        np.uint16
    )

    if not cv2.imwrite(
        str(path),
        depth_mm,
    ):
        raise RuntimeError(
            f"Failed to save depth "
            f"to {path}"
        )


# ==============================================================================
# Frame Augmentation
# ==============================================================================

def augment_frame(
    src_data_dir: Path,
    dst_data_dir: Path,
    stem: str,
    depth_augmenter: DepthNoiseAugmenter,
    frame_seed: Optional[int] = None,
) -> bool:

    ensure_dir(
        dst_data_dir
    )

    # ==========================================================
    # 1. RGB -> GRAYSCALE
    # ==========================================================

    color_suffixes = [
        "_color.jpg",
        "_color.jpeg",
        "_color.png",
    ]

    color_src = next(
        (
            src_data_dir
            / f"{stem}{suffix}"
            for suffix in color_suffixes
            if (
                src_data_dir
                / f"{stem}{suffix}"
            ).exists()
        ),
        None,
    )

    if color_src is None:

        print(
            f"[WARNING] RGB missing: "
            f"{stem}"
        )

        return False

    try:

        color = cv2.imread(
            str(color_src),
            cv2.IMREAD_COLOR,
        )

        if color is None:
            raise RuntimeError(
                f"Failed to read RGB: "
                f"{color_src}"
            )

        gray_color = (
            convert_to_grayscale_rgb(
                color
            )
        )

        color_dst = (
            dst_data_dir
            / f"{stem}_color.jpg"
        )

        if not cv2.imwrite(
            str(color_dst),
            gray_color,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        ):
            raise RuntimeError(
                f"Failed to save grayscale RGB: "
                f"{color_dst}"
            )

    except Exception as e:

        print(
            f"[WARNING] Grayscale conversion failed:\n"
            f"  Source: {color_src}\n"
            f"  Error : {e}"
        )

        return False

    # ==========================================================
    # 2. DEPTH
    # ==========================================================

    depth_src = (
        src_data_dir
        / f"{stem}_depth.png"
    )

    if not depth_src.exists():

        print(
            f"[WARNING] Depth missing: "
            f"{depth_src}"
        )

        return False

    try:

        depth_m = load_original_depth(
            depth_src
        )

        # ------------------------------------------------------
        # IMPORTANT:
        #
        # frame_seed is unique to this frame.
        #
        # No previous frame data is passed into process().
        # ------------------------------------------------------

        aug_depth_m = (
            depth_augmenter.process(
                depth_m,
                seed=frame_seed,
            )
        )

        save_dttd_depth(
            dst_data_dir
            / f"{stem}_depth.png",
            aug_depth_m,
        )

    except Exception as e:

        print(
            f"[WARNING] Depth augmentation failed:\n"
            f"  Source: {depth_src}\n"
            f"  Error : {e}"
        )

        return False

    # ==========================================================
    # 3. LABEL / META
    # ==========================================================

    for suffix in (
        "_label.png",
        "_meta.json",
    ):

        extra_file = (
            src_data_dir
            / f"{stem}{suffix}"
        )

        if extra_file.exists():

            shutil.copy2(
                extra_file,
                dst_data_dir
                / extra_file.name,
            )

    return True


# ==============================================================================
# Split Processing
# ==============================================================================

def process_split(
    lines: List[str],
    split_name: str,
    data_root: Path,
    depth_augmenter: DepthNoiseAugmenter,
) -> List[str]:

    output_lines = []

    print()
    print("=" * 70)
    print(
        f"[INFO] Processing {split_name}"
    )
    print(
        f"[INFO] Original "
        f"{split_name} frames: "
        f"{len(lines)}"
    )
    print("=" * 70)

    for index, line in enumerate(
        tqdm(
            lines,
            desc=f"{split_name} augmentation",
        )
    ):

        parts = line.split("/")

        if len(parts) < 3:

            print(
                f"[WARNING] Invalid "
                f"{split_name} entry: "
                f"{line}"
            )

            continue

        scene_name = parts[0]
        stem = parts[-1]

        aug_scene_name = (
            f"{scene_name}_gray"
        )

        src_data_dir = (
            data_root
            / scene_name
            / "data"
        )

        dst_data_dir = (
            data_root
            / aug_scene_name
            / "data"
        )

        if not src_data_dir.exists():

            print(
                f"[WARNING] Source "
                f"directory missing: "
                f"{src_data_dir}"
            )

            continue

        # ======================================================
        # IMPORTANT:
        #
        # Generate a deterministic but independent seed for
        # every frame.
        #
        # Frame N does NOT inherit RNG state from frame N-1.
        # ======================================================

        frame_seed = (
            hash(
                (
                    split_name,
                    scene_name,
                    stem,
                )
            )
            & 0xFFFFFFFF
        )

        success = augment_frame(
            src_data_dir=src_data_dir,
            dst_data_dir=dst_data_dir,
            stem=stem,
            depth_augmenter=depth_augmenter,
            frame_seed=frame_seed,
        )

        if success:

            output_lines.append(
                f"{aug_scene_name}/data/{stem}"
            )

    return output_lines


# ==============================================================================
# Dataset
# ==============================================================================

def augment_dataset(
    dataset_dir: str,
    focal_length: float = 399.0,
    baseline: float = 0.09,
    max_depth: float = 5.0,
) -> None:

    root = Path(
        dataset_dir
    )

    data_root = (
        root / "data"
    )

    config_dir = (
        root / "dataset_config"
    )

    if not data_root.exists():

        raise RuntimeError(
            f"Missing data folder: "
            f"{data_root}"
        )

    train_txt = (
        config_dir
        / "train_data_list.txt"
    )

    test_txt = (
        config_dir
        / "test_data_list.txt"
    )

    if not train_txt.exists():

        raise RuntimeError(
            f"Missing config file: "
            f"{train_txt}"
        )

    train_lines = read_train_list(
        train_txt
    )

    test_lines = (
        read_train_list(
            test_txt
        )
        if test_txt.exists()
        else []
    )

    # ==========================================================
    # Depth augmenter
    #
    # The object itself contains NO temporal state.
    # ==========================================================

    depth_augmenter = (
        DepthNoiseAugmenter(
            focal_length=focal_length,
            baseline=baseline,
            max_depth=max_depth,
        )
    )

    # ==========================================================
    # TRAIN
    # ==========================================================

    augmented_train_lines = (
        process_split(
            lines=train_lines,
            split_name="train",
            data_root=data_root,
            depth_augmenter=depth_augmenter,
        )
    )

    # ==========================================================
    # TEST
    # ==========================================================

    augmented_test_lines = (
        process_split(
            lines=test_lines,
            split_name="test",
            data_root=data_root,
            depth_augmenter=depth_augmenter,
        )
    )

    # ==========================================================
    # Save split lists
    # ==========================================================

    print()
    print("=" * 70)
    print(
        "[INFO] Saving dataset configuration files..."
    )
    print("=" * 70)

    train_content = (
        "\n".join(
            augmented_train_lines
        )
    )

    if train_content:
        train_content += "\n"

    test_content = (
        "\n".join(
            augmented_test_lines
        )
    )

    if test_content:
        test_content += "\n"

    (
        config_dir
        / "train_data_list.txt"
    ).write_text(
        train_content
    )

    (
        config_dir
        / "test_data_list.txt"
    ).write_text(
        test_content
    )

    # ==========================================================
    # Summary
    # ==========================================================

    print()
    print("=" * 70)
    print("[DONE]")
    print("=" * 70)

    print(
        f"Original train frames : "
        f"{len(train_lines)}"
    )

    print(
        f"Gray train frames     : "
        f"{len(augmented_train_lines)}"
    )

    print(
        f"Original test frames  : "
        f"{len(test_lines)}"
    )

    print(
        f"Gray test frames      : "
        f"{len(augmented_test_lines)}"
    )

    print()
    print(
        "[Depth Configuration]"
    )

    print(
        f"Depth range           : "
        f"{DEPTH_MIN_MM} - "
        f"{DEPTH_MAX_MM} mm"
    )

    print(
        f"Subpixel              : "
        f"{SUBPIXEL}"
    )

    print(
        f"Subpixel bits         : "
        f"{SUBPIXEL_FRACTIONAL_BITS}"
    )

    print(
        f"Median                : "
        f"{MEDIAN_FILTER}"
    )

    print(
        f"Speckle filter        : "
        f"{SPECKLE_FILTER}"
    )

    print(
        f"Spatial filter        : "
        f"{SPATIAL_FILTER}"
    )

    print(
        f"Temporal filter       : "
        f"{TEMPORAL_FILTER}"
    )

    print()
    print(
        "[IMPORTANT] "
        "Every frame is processed independently."
    )

    print(
        "[IMPORTANT] "
        "No temporal depth state is used."
    )

    print(
        f"Output directory      : "
        f"{root}"
    )

    print("=" * 70)


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "OAK4 StereoDepth noise + "
            "grayscale augmentation tool."
        )
    )

    # ==========================================================
    # KEEP ORIGINAL ARGUMENT NAMES
    # ==========================================================

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help=(
            "Path to DTTD dataset directory"
        ),
    )

    parser.add_argument(
        "--focal-length",
        type=float,
        default=399.0,
        help=(
            "Stereo focal length in pixels"
        ),
    )

    parser.add_argument(
        "--baseline",
        type=float,
        default=0.09,
        help=(
            "Stereo baseline in meters"
        ),
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=5.0,
        help=(
            "Maximum valid depth "
            "boundary in meters"
        ),
    )

    args = parser.parse_args()

    augment_dataset(
        args.dataset_dir,
        focal_length=args.focal_length,
        baseline=args.baseline,
        max_depth=args.max_depth,
    )