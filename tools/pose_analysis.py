#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import yaml


def summarize(values):
    if not values:
        return {}

    a = np.asarray(values, dtype=np.float64)

    return {
        "mean": round(float(np.mean(a)), 4),
        "median": round(float(np.median(a)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "p99": round(float(np.percentile(a, 99)), 4),
        "max": round(float(np.max(a)), 4),
    }


def success_rate(values, thresholds):
    a = np.asarray(values, dtype=np.float64)

    return {
        str(t): round(float(np.mean(a < t) * 100), 2)
        for t in thresholds
    }


def correlation(x, y):
    if len(x) < 2:
        return None

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def analyze(path, worst_n):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    frames = data.get("frames", [])

    rotations = []
    translations = []
    samples = []

    confidences = []
    weights = []

    for frame in frames:
        frame_id = frame.get("frame")
        frame_name = frame.get("frame_name")

        for obj in frame.get("objects", []):
            error = obj.get("error", {})

            rot = error.get("rotation_deg")
            trans = error.get("translation_mm")

            if rot is None or trans is None:
                continue

            rot = float(rot)
            trans = float(trans)

            rotations.append(rot)
            translations.append(trans)

            samples.append({
                "frame": frame_id,
                "name": frame_name,
                "rot": rot,
                "trans": trans,
            })

            top_k = obj.get("top_k", {})

            confidences.extend(
                float(x)
                for x in top_k.get("confidence", [])
            )

            weights.extend(
                float(x)
                for x in top_k.get("weights", [])
            )

    if not samples:
        raise RuntimeError("No valid pose results found.")

    rot = np.asarray(rotations)
    trans = np.asarray(translations)

    worst_rotation = sorted(
        samples,
        key=lambda x: x["rot"],
        reverse=True,
    )[:worst_n]

    worst_translation = sorted(
        samples,
        key=lambda x: x["trans"],
        reverse=True,
    )[:worst_n]

    result = {
        "source": str(path),

        "count": {
            "frames": len(frames),
            "objects": len(samples),
        },

        "rotation_deg": summarize(rotations),

        "translation_mm": summarize(translations),

        "success_rate_rotation_percent": success_rate(
            rotations,
            [5, 10, 20, 30, 45, 60, 90],
        ),

        "success_rate_translation_percent": success_rate(
            translations,
            [2, 5, 10, 20, 30, 50],
        ),

        "failure_count": {
            "rotation_gt_30": int(np.sum(rot > 30)),
            "rotation_gt_60": int(np.sum(rot > 60)),
            "rotation_gt_90": int(np.sum(rot > 90)),
            "translation_gt_10": int(np.sum(trans > 10)),
            "translation_gt_20": int(np.sum(trans > 20)),
            "translation_gt_50": int(np.sum(trans > 50)),
            "both_rot_gt_30_trans_gt_10": int(
                np.sum((rot > 30) & (trans > 10))
            ),
        },

        "correlation": {
            "rotation_translation": correlation(
                rotations,
                translations,
            ),
        },
    }

    if confidences:
        c = np.asarray(confidences)

        result["top_k_confidence"] = {
            "mean": round(float(np.mean(c)), 6),
            "std": round(float(np.std(c)), 6),
            "min": round(float(np.min(c)), 6),
            "max": round(float(np.max(c)), 6),
        }

    if weights:
        w = np.asarray(weights)

        result["top_k_weight"] = {
            "mean": round(float(np.mean(w)), 6),
            "std": round(float(np.std(w)), 6),
            "min": round(float(np.min(w)), 6),
            "max": round(float(np.max(w)), 6),
        }

    result["worst_rotation"] = [
        {
            "frame": x["frame"],
            "name": x["name"],
            "rot_deg": round(x["rot"], 3),
            "trans_mm": round(x["trans"], 3),
        }
        for x in worst_rotation
    ]

    result["worst_translation"] = [
        {
            "frame": x["frame"],
            "name": x["name"],
            "rot_deg": round(x["rot"], 3),
            "trans_mm": round(x["trans"], 3),
        }
        for x in worst_translation
    ]

    return result


def print_summary(result):
    print("\n" + "=" * 60)
    print("DTTD POSE ANALYSIS")
    print("=" * 60)

    c = result["count"]
    print(f"Frames / Objects : {c['frames']} / {c['objects']}")

    print("\nRotation (deg)")
    print(result["rotation_deg"])

    print("\nTranslation (mm)")
    print(result["translation_mm"])

    print("\nRotation Success (%)")
    print(result["success_rate_rotation_percent"])

    print("\nTranslation Success (%)")
    print(result["success_rate_translation_percent"])

    print("\nFailure Count")
    for k, v in result["failure_count"].items():
        print(f"  {k}: {v}")

    print("\nCorrelation")
    print(result["correlation"])

    if "top_k_confidence" in result:
        print("\nTop-K Confidence")
        print(result["top_k_confidence"])

    if "top_k_weight" in result:
        print("\nTop-K Weight")
        print(result["top_k_weight"])

    print("\nWorst Rotation")
    for x in result["worst_rotation"]:
        print(
            f"  {x['name']} | "
            f"rot={x['rot_deg']}° | "
            f"trans={x['trans_mm']}mm"
        )

    print("\nWorst Translation")
    for x in result["worst_translation"]:
        print(
            f"  {x['name']} | "
            f"rot={x['rot_deg']}° | "
            f"trans={x['trans_mm']}mm"
        )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "yaml",
        type=Path,
        help="pose_result.yaml",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of worst samples",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not args.yaml.is_file():
        raise FileNotFoundError(args.yaml)

    result = analyze(
        args.yaml,
        args.top,
    )

    print_summary(result)

    output = args.output or (
        args.yaml.parent
        / f"{args.yaml.stem}_analysis.yaml"
    )

    with open(output, "w") as f:
        yaml.safe_dump(
            result,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()