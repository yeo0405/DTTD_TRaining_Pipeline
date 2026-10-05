#!/usr/bin/env python3
"""
Remove selected scenes from an existing dataset.

Dataset format:

dataset/
├── data/
│   ├── 000000/
│   ├── 000001/
│   ├── 000002/
│   └── ...
├── objects/
└── dataset_config/
    ├── objectids.csv
    ├── train_data_list.txt
    └── test_data_list.txt

The selected scenes will be permanently removed.

The following files will also be updated:

    dataset_config/train_data_list.txt
    dataset_config/test_data_list.txt

Scene IDs are NOT renumbered.

Example:

    python remove_scenes.py \
        --dataset /path/to/dataset \
        --scenes 000001 000003 000005

Or:

    python remove_scenes.py \
        --dataset /path/to/dataset \
        --scene-list scenes.txt

Use --dry-run to preview what will be removed without modifying anything.
"""


import argparse
import shutil
from pathlib import Path
from typing import List, Set


# ============================================================
# Utilities
# ============================================================

def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def write_list(
    path: Path,
    lines: List[str],
) -> None:

    text = "\n".join(lines)

    if lines:
        text += "\n"

    path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Get scenes
# ============================================================

def get_scenes(
    dataset_dir: Path,
) -> List[Path]:

    data_dir = dataset_dir / "data"

    if not data_dir.exists():

        raise RuntimeError(
            f"Missing data directory:\n{data_dir}"
        )

    scenes = [
        p
        for p in data_dir.iterdir()
        if p.is_dir()
    ]

    def sort_key(p: Path):

        try:
            return (0, int(p.name))

        except ValueError:
            return (1, p.name)

    scenes.sort(key=sort_key)

    return scenes


# ============================================================
# Read scene list
# ============================================================

def read_scene_list_file(
    path: Path,
) -> List[str]:

    if not path.exists():

        raise RuntimeError(
            f"Scene list file does not exist:\n{path}"
        )

    scenes = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        scenes.append(line)

    return scenes


# ============================================================
# Validate selected scenes
# ============================================================

def validate_scene_names(
    all_scenes: List[Path],
    selected_names: List[str],
) -> List[str]:

    available = {
        scene.name
        for scene in all_scenes
    }

    # Remove duplicates
    selected = []
    seen: Set[str] = set()

    for name in selected_names:

        name = str(name).strip()

        if not name:
            continue

        if name in seen:
            continue

        seen.add(name)
        selected.append(name)

    missing = [
        name
        for name in selected
        if name not in available
    ]

    if missing:

        raise RuntimeError(
            "\n[ERROR] The following scenes do not exist:\n"
            + "\n".join(
                f"    {name}"
                for name in missing
            )
            + "\n"
        )

    return selected


# ============================================================
# Backup list file
# ============================================================

def backup_file(
    path: Path,
) -> Path:

    backup = path.with_suffix(
        path.suffix + ".bak"
    )

    shutil.copy2(
        path,
        backup,
    )

    return backup


# ============================================================
# Remove scenes
# ============================================================

def remove_scenes(
    dataset_path: str,
    scene_names: List[str],
    dry_run: bool = False,
) -> None:

    dataset = Path(dataset_path)

    print("\n==============================================")
    print("REMOVE DATASET SCENES")
    print("==============================================")
    print(f"Dataset : {dataset}")
    print("==============================================\n")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not dataset.exists():

        raise RuntimeError(
            f"Dataset does not exist:\n{dataset}"
        )

    data_dir = dataset / "data"

    if not data_dir.exists():

        raise RuntimeError(
            f"Missing data directory:\n{data_dir}"
        )

    # --------------------------------------------------------
    # Get all scenes
    # --------------------------------------------------------

    all_scenes = get_scenes(dataset)

    print(
        f"[INFO] Total scenes: "
        f"{len(all_scenes)}"
    )

    # --------------------------------------------------------
    # Validate scenes
    # --------------------------------------------------------

    selected_names = validate_scene_names(
        all_scenes,
        scene_names,
    )

    if not selected_names:

        raise RuntimeError(
            "No scenes were selected."
        )

    # --------------------------------------------------------
    # Print selected scenes
    # --------------------------------------------------------

    print(
        f"[INFO] Scenes to remove: "
        f"{len(selected_names)}"
    )

    for name in selected_names:

        print(
            f"    {name}"
        )

    remaining_count = (
        len(all_scenes)
        - len(selected_names)
    )

    print(
        f"\n[INFO] Remaining scenes: "
        f"{remaining_count}"
    )

    # --------------------------------------------------------
    # Safety confirmation
    # --------------------------------------------------------

    if not dry_run:

        print(
            "\n[WARNING] The selected scene directories "
            "will be permanently deleted."
        )

        answer = input(
            "Continue? [y/N]: "
        ).strip().lower()

        if answer not in ("y", "yes"):

            print(
                "[INFO] Cancelled."
            )

            return

    # --------------------------------------------------------
    # Remove scene directories
    # --------------------------------------------------------

    if dry_run:

        print(
            "\n[DRY RUN] No files will be modified."
        )

    else:

        print(
            "\n[INFO] Removing scene directories..."
        )

        for scene_name in selected_names:

            scene_path = (
                data_dir / scene_name
            )

            print(
                f"    Removing: "
                f"{scene_path}"
            )

            shutil.rmtree(
                scene_path
            )

    # --------------------------------------------------------
    # Update train/test lists
    # --------------------------------------------------------

    for list_name in [
        "train_data_list.txt",
        "test_data_list.txt",
    ]:

        list_path = (
            dataset
            / "dataset_config"
            / list_name
        )

        if not list_path.exists():

            print(
                f"\n[INFO] {list_name} does not exist."
            )

            continue

        lines = read_lines(
            list_path
        )

        original_count = len(lines)

        remaining_lines = []

        removed_lines = []

        selected_set = set(
            selected_names
        )

        for line in lines:

            parts = line.split("/")

            if not parts:
                continue

            scene_name = parts[0]

            if scene_name in selected_set:

                removed_lines.append(
                    line
                )

            else:

                remaining_lines.append(
                    line
                )

        removed_count = len(
            removed_lines
        )

        print(
            f"\n[INFO] {list_name}"
        )

        print(
            f"    Original entries : "
            f"{original_count}"
        )

        print(
            f"    Removed entries  : "
            f"{removed_count}"
        )

        print(
            f"    Remaining entries: "
            f"{len(remaining_lines)}"
        )

        if dry_run:

            print(
                "    [DRY RUN] List file "
                "will not be modified."
            )

        else:

            # Backup original list
            backup = backup_file(
                list_path
            )

            print(
                f"    Backup           : "
                f"{backup}"
            )

            write_list(
                list_path,
                remaining_lines,
            )

            print(
                f"    Updated          : "
                f"{list_path}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==============================================")

    if dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("REMOVE COMPLETE")

    print("==============================================")

    print(
        f"Original scenes : "
        f"{len(all_scenes)}"
    )

    print(
        f"Removed scenes  : "
        f"{len(selected_names)}"
    )

    print(
        f"Remaining scenes: "
        f"{remaining_count}"
    )

    print(
        f"Dataset         : "
        f"{dataset}"
    )

    print("==============================================")


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Remove selected scenes from an existing dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory",
    )

    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            "Scene IDs to remove, "
            "e.g. 000001 000003 000005"
        ),
    )

    parser.add_argument(
        "--scene-list",
        default=None,
        help=(
            "Text file containing scene IDs, "
            "one scene per line"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what will be removed without "
            "modifying the dataset"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if (
        args.scenes is None
        and args.scene_list is None
    ):

        parser.error(
            "Please provide either "
            "--scenes or --scene-list."
        )

    if (
        args.scenes is not None
        and args.scene_list is not None
    ):

        parser.error(
            "Use only one of "
            "--scenes or --scene-list."
        )

    # --------------------------------------------------------
    # Get scenes
    # --------------------------------------------------------

    if args.scenes is not None:

        scene_names = args.scenes

    else:

        scene_names = read_scene_list_file(
            Path(args.scene_list)
        )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    remove_scenes(
        dataset_path=args.dataset,
        scene_names=scene_names,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()