#!/usr/bin/env python3

"""
Extract the first RGB and Depth frame from a recording folder.

Input:
    folder/
        *.mp4
        *.mkv
        *.json

Output:
    folder/
        color.jpg
        depth.png

RGB:
    MP4 -> first frame -> JPEG

Depth:
    MKV
        FFV1
        gray16le
        uint16
        millimetres
        -> first frame
        -> uint16 PNG

Important:
    Depth is NOT normalized.
    Depth is NOT scaled.
    Depth values remain in millimetres.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Find Files
# =============================================================================

def find_video_file(folder: Path, extensions):
    """
    Find the first video file matching the given extensions.
    """

    files = []

    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))

    files = sorted(set(files))

    if not files:
        return None

    if len(files) > 1:
        print()
        print(f"[WARN] Multiple files found for {extensions}:")

        for f in files:
            print(f"       {f.name}")

        print()
        print(f"[INFO] Using: {files[0].name}")

    return files[0]


# =============================================================================
# Extract RGB
# =============================================================================

def extract_color(
    mp4_path: Path,
    output_path: Path,
):
    """
    Extract first frame from MP4 and save as JPEG.
    """

    print()
    print("=" * 70)
    print("RGB")
    print("=" * 70)

    print(f"Input : {mp4_path}")

    cap = cv2.VideoCapture(str(mp4_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Failed to open RGB video:\n{mp4_path}"
        )

    ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        raise RuntimeError(
            f"Failed to read first RGB frame:\n{mp4_path}"
        )

    print()
    print("RGB frame:")
    print(f"  shape : {frame.shape}")
    print(f"  dtype : {frame.dtype}")

    height, width = frame.shape[:2]

    print(f"  size  : {width} x {height}")

    # =========================================================================
    # Save JPEG
    # =========================================================================

    success = cv2.imwrite(
        str(output_path),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100,
        ],
    )

    if not success:
        raise RuntimeError(
            f"Failed to write RGB image:\n{output_path}"
        )

    # =========================================================================
    # Verify
    # =========================================================================

    check = cv2.imread(
        str(output_path),
        cv2.IMREAD_COLOR,
    )

    if check is None:
        raise RuntimeError(
            f"RGB image was written but cannot be read:\n{output_path}"
        )

    print()
    print("Saved RGB verification:")
    print(f"  shape : {check.shape}")
    print(f"  dtype : {check.dtype}")

    print()
    print(f"Output: {output_path}")


# =============================================================================
# Probe Depth Video
# =============================================================================

def probe_depth_video(
    mkv_path: Path,
):
    """
    Read depth video metadata using ffprobe.

    Expected format:

        codec_name = ffv1
        pix_fmt    = gray16le
        width      = 1920
        height     = 1080
    """

    print()
    print("=" * 70)
    print("Depth Video Metadata")
    print("=" * 70)

    print(f"Input: {mkv_path}")

    command = [
        "ffprobe",

        "-v",
        "error",

        "-select_streams",
        "v:0",

        "-show_entries",
        "stream=codec_name,pix_fmt,width,height",

        "-of",
        "default=noprint_wrappers=1",

        str(mkv_path),
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    except FileNotFoundError:

        raise RuntimeError(
            "ffprobe was not found.\n\n"
            "Please install FFmpeg:\n"
            "    sudo apt install ffmpeg"
        )

    except subprocess.CalledProcessError as e:

        raise RuntimeError(
            "ffprobe failed:\n"
            f"{e.stderr}"
        )

    metadata = {}

    for line in result.stdout.splitlines():

        if "=" in line:

            key, value = line.split(
                "=",
                1,
            )

            metadata[key] = value

    if "width" not in metadata:
        raise RuntimeError(
            "Could not determine depth video width."
        )

    if "height" not in metadata:
        raise RuntimeError(
            "Could not determine depth video height."
        )

    width = int(metadata["width"])
    height = int(metadata["height"])

    codec = metadata.get(
        "codec_name",
        "unknown",
    )

    pix_fmt = metadata.get(
        "pix_fmt",
        "unknown",
    )

    print()
    print("Metadata:")
    print(f"  codec     : {codec}")
    print(f"  pixel fmt : {pix_fmt}")
    print(f"  width     : {width}")
    print(f"  height    : {height}")

    # =========================================================================
    # Validation
    # =========================================================================

    if codec.lower() != "ffv1":

        print()
        print(
            f"[WARN] Expected FFV1 codec, got: {codec}"
        )

    if pix_fmt.lower() != "gray16le":

        print()
        print(
            f"[WARN] Expected gray16le pixel format, got: {pix_fmt}"
        )

    return (
        width,
        height,
        codec,
        pix_fmt,
    )


# =============================================================================
# Extract Depth
# =============================================================================

def extract_depth(
    mkv_path: Path,
    output_path: Path,
):
    """
    Extract first depth frame from FFV1 + gray16le MKV.

    The raw depth data is decoded directly using FFmpeg.

    Output:
        uint16 PNG

    Depth values:
        millimetres
    """

    print()
    print("=" * 70)
    print("DEPTH")
    print("=" * 70)

    print(f"Input : {mkv_path}")

    # =========================================================================
    # Probe video
    # =========================================================================

    (
        width,
        height,
        codec,
        pix_fmt,
    ) = probe_depth_video(
        mkv_path
    )

    # =========================================================================
    # FFmpeg extraction
    # =========================================================================

    print()
    print("Extracting first depth frame using FFmpeg...")

    command = [
        "ffmpeg",

        "-v",
        "error",

        "-i",
        str(mkv_path),

        # Only first frame
        "-frames:v",
        "1",

        # Force original depth representation
        "-pix_fmt",
        "gray16le",

        # Raw video output
        "-f",
        "rawvideo",

        "pipe:1",
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    except FileNotFoundError:

        raise RuntimeError(
            "ffmpeg was not found.\n\n"
            "Please install FFmpeg:\n"
            "    sudo apt install ffmpeg"
        )

    except subprocess.CalledProcessError as e:

        error_message = e.stderr.decode(
            errors="replace"
        )

        raise RuntimeError(
            "FFmpeg failed:\n"
            f"{error_message}"
        )

    raw = result.stdout

    # =========================================================================
    # Validate raw frame size
    # =========================================================================

    expected_bytes = (
        width
        * height
        * 2
    )

    actual_bytes = len(raw)

    print()
    print("Raw frame:")
    print(f"  expected bytes : {expected_bytes}")
    print(f"  received bytes : {actual_bytes}")

    if actual_bytes != expected_bytes:

        raise RuntimeError(
            "Unexpected raw depth frame size.\n"
            f"Expected: {expected_bytes} bytes\n"
            f"Received: {actual_bytes} bytes"
        )

    # =========================================================================
    # Raw bytes -> uint16
    # =========================================================================

    depth = np.frombuffer(
        raw,
        dtype="<u2",
    )

    depth = depth.reshape(
        height,
        width,
    )

    # Make writable
    depth = depth.copy()

    # =========================================================================
    # Depth Statistics
    # =========================================================================

    print()
    print("Depth statistics:")
    print(f"  shape : {depth.shape}")
    print(f"  dtype : {depth.dtype}")
    print(f"  min   : {depth.min()}")
    print(f"  max   : {depth.max()}")
    print(f"  mean  : {depth.mean():.3f}")

    valid = depth > 0

    valid_count = int(
        valid.sum()
    )

    total_count = int(
        valid.size
    )

    print()
    print(
        f"Valid depth: "
        f"{valid_count} / {total_count} "
        f"({100.0 * valid_count / total_count:.2f}%)"
    )

    if valid_count == 0:

        raise RuntimeError(
            "Depth frame contains no valid pixels."
        )

    print()
    print("Valid depth range:")

    print(
        f"  min : {depth[valid].min()} mm"
    )

    print(
        f"  max : {depth[valid].max()} mm"
    )

    print(
        f"  mean: {depth[valid].mean():.3f} mm"
    )

    # =========================================================================
    # Save uint16 PNG
    # =========================================================================

    print()
    print("Saving uint16 PNG...")

    success = cv2.imwrite(
        str(output_path),
        depth,
    )

    if not success:

        raise RuntimeError(
            f"Failed to write depth image:\n"
            f"{output_path}"
        )

    # =========================================================================
    # Verify saved PNG
    # =========================================================================

    check = cv2.imread(
        str(output_path),
        cv2.IMREAD_UNCHANGED,
    )

    if check is None:

        raise RuntimeError(
            "Depth PNG was written "
            "but cannot be read:\n"
            f"{output_path}"
        )

    print()
    print("Saved depth verification:")
    print(f"  shape : {check.shape}")
    print(f"  dtype : {check.dtype}")
    print(f"  min   : {check.min()}")
    print(f"  max   : {check.max()}")
    print(f"  mean  : {check.mean():.3f}")

    # =========================================================================
    # Verify exact format
    # =========================================================================

    if check.dtype != np.uint16:

        raise RuntimeError(
            "Depth PNG is not uint16.\n"
            f"Got: {check.dtype}"
        )

    if check.shape != (
        height,
        width,
    ):

        raise RuntimeError(
            "Depth PNG shape mismatch.\n"
            f"Expected: {(height, width)}\n"
            f"Got: {check.shape}"
        )

    # =========================================================================
    # Verify values were not changed
    # =========================================================================

    if not np.array_equal(
        depth,
        check,
    ):

        raise RuntimeError(
            "Depth values changed "
            "during PNG saving/loading."
        )

    print()
    print("Depth verification PASSED:")
    print("  Format : uint16")
    print("  Unit   : millimetres")
    print("  Scale  : 1:1")
    print("  Values : unchanged")

    print()
    print(f"Output: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract first RGB frame from MP4 "
            "and first uint16 depth frame from "
            "FFV1/gray16le MKV."
        )
    )

    parser.add_argument(
        "folder",
        type=str,
        help=(
            "Recording folder containing MP4 "
            "and MKV files."
        ),
    )

    args = parser.parse_args()

    folder = Path(
        args.folder
    )

    # =========================================================================
    # Validate folder
    # =========================================================================

    if not folder.exists():

        raise FileNotFoundError(
            f"Folder does not exist:\n{folder}"
        )

    if not folder.is_dir():

        raise NotADirectoryError(
            f"Not a directory:\n{folder}"
        )

    # =========================================================================
    # Find MP4
    # =========================================================================

    mp4_path = find_video_file(
        folder,
        [".mp4"],
    )

    if mp4_path is None:

        raise FileNotFoundError(
            "No MP4 file found in:\n"
            f"{folder}"
        )

    # =========================================================================
    # Find MKV
    # =========================================================================

    mkv_path = find_video_file(
        folder,
        [".mkv"],
    )

    if mkv_path is None:

        raise FileNotFoundError(
            "No MKV file found in:\n"
            f"{folder}"
        )

    # =========================================================================
    # Output
    # =========================================================================

    color_output = (
        folder / "color.jpg"
    )

    depth_output = (
        folder / "depth.png"
    )

    # =========================================================================
    # Input Summary
    # =========================================================================

    print()
    print("=" * 70)
    print("INPUT")
    print("=" * 70)

    print(f"Folder : {folder}")
    print(f"MP4    : {mp4_path.name}")
    print(f"MKV    : {mkv_path.name}")

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(f"Color  : {color_output}")
    print(f"Depth  : {depth_output}")

    # =========================================================================
    # Extract RGB
    # =========================================================================

    extract_color(
        mp4_path,
        color_output,
    )

    # =========================================================================
    # Extract Depth
    # =========================================================================

    extract_depth(
        mkv_path,
        depth_output,
    )

    # =========================================================================
    # Final
    # =========================================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        f"  RGB   : {color_output}"
    )

    print(
        f"  Depth : {depth_output}"
    )

    print()
    print("Depth format:")
    print("  dtype = uint16")
    print("  unit  = millimetres")
    print("  scale = 1"
          " (no conversion)")


if __name__ == "__main__":
    main()