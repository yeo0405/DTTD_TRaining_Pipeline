#!/usr/bin/env python3

import argparse
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize uint16 instance mask."
    )

    parser.add_argument(
        "mask",
        type=str,
        help="Path to mask image"
    )

    return parser.parse_args()


def create_colors(num_colors):
    """
    Generate visually distinct colors in BGR format.
    """

    colors = []

    for i in range(num_colors):

        hue = int(
            180 * i / max(num_colors, 1)
        )

        hsv = np.uint8(
            [[[hue, 220, 255]]]
        )

        bgr = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )[0, 0]

        colors.append(
            tuple(
                int(x)
                for x in bgr
            )
        )

    return colors


def main():

    args = parse_args()

    mask_path = os.path.abspath(
        args.mask
    )

    if not os.path.isfile(mask_path):

        raise FileNotFoundError(
            f"Mask file does not exist:\n"
            f"{mask_path}"
        )

    # ========================================================
    # Load mask
    # ========================================================

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_UNCHANGED
    )

    if mask is None:

        raise RuntimeError(
            f"Failed to read mask:\n"
            f"{mask_path}"
        )

    print("=" * 60)
    print("MASK VISUALIZATION")
    print("=" * 60)

    print(
        f"Input : {mask_path}"
    )

    print(
        f"Shape : {mask.shape}"
    )

    print(
        f"Dtype : {mask.dtype}"
    )

    # ========================================================
    # Handle multi-channel mask
    # ========================================================

    if mask.ndim == 3:

        print(
            f"Channels: {mask.shape[2]}"
        )

        # The instance ID is expected to be
        # identical across channels.
        #
        # Use the first channel.
        mask = mask[:, :, 0]

        print(
            "Using channel 0 for instance IDs."
        )

    elif mask.ndim != 2:

        raise ValueError(
            f"Unsupported mask shape: "
            f"{mask.shape}"
        )

    # ========================================================
    # Convert to integer
    # ========================================================

    mask = mask.astype(
        np.uint32
    )

    # ========================================================
    # Find instance IDs
    # ========================================================

    instance_ids = np.unique(
        mask
    )

    instance_ids = [
        int(x)
        for x in instance_ids
        if int(x) > 0
    ]

    print()
    print(
        f"Instance IDs: {instance_ids}"
    )

    print(
        f"Number of objects: "
        f"{len(instance_ids)}"
    )

    # ========================================================
    # Create visualization
    # ========================================================

    height, width = mask.shape

    vis = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    colors = create_colors(
        len(instance_ids)
    )

    # ========================================================
    # Draw each instance
    # ========================================================

    for idx, instance_id in enumerate(
        instance_ids
    ):

        color = colors[idx]

        object_mask = (
            mask == instance_id
        )

        # ----------------------------------------------------
        # Fill object
        # ----------------------------------------------------

        vis[
            object_mask
        ] = color

        # ----------------------------------------------------
        # Find bounding box
        # ----------------------------------------------------

        ys, xs = np.where(
            object_mask
        )

        if len(xs) == 0:
            continue

        x_min = int(
            xs.min()
        )

        x_max = int(
            xs.max()
        )

        y_min = int(
            ys.min()
        )

        y_max = int(
            ys.max()
        )

        # ----------------------------------------------------
        # Draw contour
        # ----------------------------------------------------

        binary_mask = (
            object_mask.astype(
                np.uint8
            ) * 255
        )

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            vis,
            contours,
            -1,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Calculate center
        # ----------------------------------------------------

        moments = cv2.moments(
            binary_mask
        )

        if moments["m00"] != 0:

            center_x = int(
                moments["m10"]
                / moments["m00"]
            )

            center_y = int(
                moments["m01"]
                / moments["m00"]
            )

        else:

            center_x = (
                x_min + x_max
            ) // 2

            center_y = (
                y_min + y_max
            ) // 2

        # ----------------------------------------------------
        # Draw ID
        # ----------------------------------------------------

        text = (
            f"ID {instance_id}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 1.0

        thickness = 2

        text_size, _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        text_x = (
            center_x
            - text_size[0] // 2
        )

        text_y = (
            center_y
            + text_size[1] // 2
        )

        # Black outline
        cv2.putText(
            vis,
            text,
            (
                text_x,
                text_y
            ),
            font,
            font_scale,
            (0, 0, 0),
            5,
            cv2.LINE_AA
        )

        # White text
        cv2.putText(
            vis,
            text,
            (
                text_x,
                text_y
            ),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        print(
            f"  ID {instance_id}: "
            f"pixels={len(xs):,} "
            f"bbox=({x_min}, {y_min}) "
            f"- ({x_max}, {y_max})"
        )

    # ========================================================
    # Save
    #
    # IMPORTANT:
    # Save in the directory where the Python command
    # is executed, NOT beside the input mask.
    # ========================================================

    output_dir = os.getcwd()

    output_path = os.path.join(
        output_dir,
        "mask_visualized.png"
    )

    if not cv2.imwrite(
        output_path,
        vis
    ):

        raise RuntimeError(
            f"Failed to save:\n"
            f"{output_path}"
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Objects : "
        f"{len(instance_ids)}"
    )

    print(
        f"Output  : "
        f"{output_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
