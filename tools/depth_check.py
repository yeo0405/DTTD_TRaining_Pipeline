#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# Depth definition
# ============================================================
#
# depth PNG:
#   dtype : uint16
#   unit  : millimeter (mm)
#
# Special values:
#   0       -> invalid depth
#   65535   -> infinity / no return / invalid depth
#
# Therefore:
#   valid depth = 0 < depth < 65535
#
# ============================================================

INVALID_MIN = 0
INVALID_MAX = 65535


def find_depth_files(root: Path):
    """Find all migrated depth PNG files."""
    return sorted(root.rglob("*_depth.png"))


def inspect_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    result = {
        "path": path,
        "ok": True,
        "warnings": [],
        "errors": [],
    }

    # ========================================================
    # Read
    # ========================================================

    if depth is None:
        result["ok"] = False
        result["errors"].append("cannot read depth image")
        return result

    # ========================================================
    # Shape
    # ========================================================

    if depth.ndim != 2:
        result["ok"] = False
        result["errors"].append(
            f"depth is not single-channel: shape={depth.shape}"
        )
        return result

    result["shape"] = depth.shape
    result["dtype"] = str(depth.dtype)

    # ========================================================
    # Dtype
    # ========================================================

    if depth.dtype != np.uint16:
        result["ok"] = False
        result["errors"].append(
            f"wrong dtype: {depth.dtype}, expected uint16"
        )
        return result

    # ========================================================
    # Pixel statistics
    # ========================================================

    total_pixels = depth.size

    invalid_zero = np.count_nonzero(depth == 0)
    invalid_inf = np.count_nonzero(depth == INVALID_MAX)

    valid_mask = (
        (depth > INVALID_MIN)
        & (depth < INVALID_MAX)
    )

    valid = depth[valid_mask]

    result["total_pixels"] = int(total_pixels)
    result["zero_pixels"] = int(invalid_zero)
    result["infinity_pixels"] = int(invalid_inf)
    result["valid_pixels"] = int(valid.size)

    result["zero_ratio"] = float(invalid_zero / total_pixels)
    result["infinity_ratio"] = float(invalid_inf / total_pixels)
    result["valid_ratio"] = float(valid.size / total_pixels)

    # ========================================================
    # No valid depth
    # ========================================================

    if valid.size == 0:
        result["ok"] = False
        result["errors"].append(
            "no valid depth pixels "
            "(all pixels are 0 or 65535)"
        )
        return result

    # ========================================================
    # Convert to float for statistics
    # ========================================================

    values = valid.astype(np.float64)

    result["min"] = float(values.min())
    result["max"] = float(values.max())
    result["mean"] = float(values.mean())
    result["median"] = float(np.median(values))

    result["p01"] = float(np.percentile(values, 1))
    result["p05"] = float(np.percentile(values, 5))
    result["p10"] = float(np.percentile(values, 10))
    result["p25"] = float(np.percentile(values, 25))
    result["p75"] = float(np.percentile(values, 75))
    result["p90"] = float(np.percentile(values, 90))
    result["p95"] = float(np.percentile(values, 95))
    result["p99"] = float(np.percentile(values, 99))

    # ========================================================
    # Unit sanity check
    # ========================================================

    # --------------------------------------------------------
    # Suspiciously small
    #
    # If depth is actually stored in meters:
    #
    #   0.8 m -> uint16 -> usually 0 or 1
    #
    # Therefore valid values mostly below 50 are suspicious.
    # --------------------------------------------------------

    if result["median"] < 50:
        result["warnings"].append(
            "median depth < 50 mm; "
            "possible meter-unit data or wrong scaling"
        )

    # --------------------------------------------------------
    # Very small range
    # --------------------------------------------------------

    if result["max"] < 50:
        result["warnings"].append(
            "maximum valid depth < 50 mm; "
            "possible meter-unit data or wrong scaling"
        )

    # --------------------------------------------------------
    # Extremely large values
    #
    # 65535 has already been removed, so this checks actual
    # valid depth values only.
    # --------------------------------------------------------

    if result["p99"] > 10000:
        result["warnings"].append(
            "P99 > 10000 mm; "
            "possible incorrect depth scaling"
        )

    # --------------------------------------------------------
    # Check whether values are suspiciously concentrated
    # --------------------------------------------------------

    if result["max"] < 100 and result["valid_pixels"] > 100:
        unique_count = len(np.unique(valid))

        if unique_count < 20:
            result["warnings"].append(
                "very low depth value diversity; "
                "possible wrong unit conversion"
            )

    # ========================================================
    # 65535 information
    # ========================================================

    if result["infinity_ratio"] > 0:
        result["has_infinity"] = True
    else:
        result["has_infinity"] = False

    return result


def print_result(result):
    path = result["path"]

    if not result["ok"]:
        status = "FAIL"
    elif result["warnings"]:
        status = "WARN"
    else:
        status = "PASS"

    print()
    print(f"[{status}] {path}")

    if "dtype" not in result:
        for error in result["errors"]:
            print(f"  ERROR        : {error}")
        return

    print(f"  dtype        : {result['dtype']}")
    print(f"  shape        : {result['shape']}")

    print(f"  valid ratio  : {result['valid_ratio']:.4f}")
    print(f"  zero ratio   : {result['zero_ratio']:.4f}")
    print(f"  65535 ratio  : {result['infinity_ratio']:.4f}")

    print()
    print("  Valid depth (mm)")
    print(f"    min        : {result['min']:.2f}")
    print(f"    max        : {result['max']:.2f}")
    print(f"    mean       : {result['mean']:.2f}")
    print(f"    median     : {result['median']:.2f}")
    print(f"    P01        : {result['p01']:.2f}")
    print(f"    P05        : {result['p05']:.2f}")
    print(f"    P10        : {result['p10']:.2f}")
    print(f"    P25        : {result['p25']:.2f}")
    print(f"    P75        : {result['p75']:.2f}")
    print(f"    P90        : {result['p90']:.2f}")
    print(f"    P95        : {result['p95']:.2f}")
    print(f"    P99        : {result['p99']:.2f}")

    for error in result["errors"]:
        print(f"  ERROR        : {error}")

    for warning in result["warnings"]:
        print(f"  WARNING      : {warning}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check migrated depth PNG files for "
            "unit consistency."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Migrated dataset root directory",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Maximum files to check. 0 = all",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print result for every file",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset)

    if not dataset_root.exists():
        print(f"ERROR: dataset does not exist: {dataset_root}")
        return 1

    # ========================================================
    # Find files
    # ========================================================

    depth_files = find_depth_files(dataset_root)

    if not depth_files:
        print("ERROR: no *_depth.png files found")
        return 1

    if args.max_files > 0:
        depth_files = depth_files[:args.max_files]

    # ========================================================
    # Header
    # ========================================================

    print("=" * 70)
    print("Depth Dataset Unit Check")
    print("=" * 70)

    print(f"Dataset : {dataset_root}")
    print(f"Files   : {len(depth_files)}")

    print()
    print("Expected depth format:")
    print("  dtype : uint16")
    print("  unit  : millimeter (mm)")
    print()
    print("Special values:")
    print("  0     : invalid depth")
    print("  65535 : infinity / invalid depth")
    print()
    print("Valid depth:")
    print("  0 < depth < 65535")
    print()

    # ========================================================
    # Process
    # ========================================================

    results = []

    for path in tqdm(depth_files, desc="Checking depth"):
        result = inspect_depth(path)
        results.append(result)

        if args.verbose:
            print_result(result)

    # ========================================================
    # Classification
    # ========================================================

    passed = [
        r for r in results
        if r["ok"] and not r["warnings"]
    ]

    warned = [
        r for r in results
        if r["ok"] and r["warnings"]
    ]

    failed = [
        r for r in results
        if not r["ok"]
    ]

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total  : {len(results)}")
    print(f"PASS   : {len(passed)}")
    print(f"WARN   : {len(warned)}")
    print(f"FAIL   : {len(failed)}")

    # ========================================================
    # Global valid-depth statistics
    # ========================================================

    medians = []
    mins = []
    maxs = []
    p01s = []
    p99s = []

    total_valid = 0
    total_zero = 0
    total_infinity = 0
    total_pixels = 0

    for r in results:
        if not r["ok"]:
            continue

        medians.append(r["median"])
        mins.append(r["min"])
        maxs.append(r["max"])
        p01s.append(r["p01"])
        p99s.append(r["p99"])

        total_valid += r["valid_pixels"]
        total_zero += r["zero_pixels"]
        total_infinity += r["infinity_pixels"]
        total_pixels += r["total_pixels"]

    if medians:

        print()
        print("=" * 70)
        print("GLOBAL VALID DEPTH STATISTICS")
        print("=" * 70)

        print("Note: 0 and 65535 are excluded.")

        print()
        print(
            f"Median of medians : "
            f"{np.median(medians):.2f} mm"
        )

        print(
            f"Minimum           : "
            f"{np.min(mins):.2f} mm"
        )

        print(
            f"Maximum           : "
            f"{np.max(maxs):.2f} mm"
        )

        print(
            f"Median P01        : "
            f"{np.median(p01s):.2f} mm"
        )

        print(
            f"Median P99        : "
            f"{np.median(p99s):.2f} mm"
        )

        print()
        print("Global pixel statistics")

        print(
            f"Valid depth       : "
            f"{total_valid / total_pixels:.4f}"
        )

        print(
            f"Zero              : "
            f"{total_zero / total_pixels:.4f}"
        )

        print(
            f"65535             : "
            f"{total_infinity / total_pixels:.4f}"
        )

    # ========================================================
    # Warning reason statistics
    # ========================================================

    warning_counter = {}

    for r in warned:
        for warning in r["warnings"]:
            warning_counter[warning] = (
                warning_counter.get(warning, 0) + 1
            )

    if warning_counter:

        print()
        print("=" * 70)
        print("WARNING REASONS")
        print("=" * 70)

        for warning, count in sorted(
            warning_counter.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"{count:5d} : {warning}")

    # ========================================================
    # Warning files
    # ========================================================

    if warned:

        print()
        print("=" * 70)
        print("WARNING FILES")
        print("=" * 70)

        for r in warned[:20]:

            print()
            print(r["path"])

            print(
                f"  median={r['median']:.2f} mm, "
                f"min={r['min']:.2f} mm, "
                f"max={r['max']:.2f} mm, "
                f"P99={r['p99']:.2f} mm"
            )

            print(
                f"  valid={r['valid_ratio']:.4f}, "
                f"65535={r['infinity_ratio']:.4f}"
            )

            for warning in r["warnings"]:
                print(f"  - {warning}")

        if len(warned) > 20:
            print()
            print(
                f"... and {len(warned) - 20} more warning files"
            )

    # ========================================================
    # Failed files
    # ========================================================

    if failed:

        print()
        print("=" * 70)
        print("FAILED FILES")
        print("=" * 70)

        for r in failed[:20]:

            print()
            print(r["path"])

            for error in r["errors"]:
                print(f"  - {error}")

        if len(failed) > 20:
            print(
                f"\n... and {len(failed) - 20} more failed files"
            )

    # ========================================================
    # Final result
    # ========================================================

    print()
    print("=" * 70)

    if failed:
        print("RESULT: FAIL")
        print("Please fix the depth dataset.")
        return 1

    if warned:
        print("RESULT: WARN")
        print(
            "Depth format is readable, "
            "but some files have suspicious values."
        )
        return 2

    print("RESULT: PASS")
    print(
        "All depth files are consistent with "
        "uint16 millimeter depth."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())