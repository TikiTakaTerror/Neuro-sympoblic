"""Phase 4 MNLogic inspection entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neurosymbolic_benchmark.data.mnlogic_inspection import (
    create_demo_dataset,
    default_rsbench_root,
    inspect_dataset,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Inspect the upstream MNLogic task.")
    parser.add_argument(
        "--rsbench-root",
        type=Path,
        default=default_rsbench_root(),
        help="Path to the rsbench checkout.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("results/inspection/mnlogic_demo"),
        help="Path to an MNLogic-style dataset root.",
    )
    parser.add_argument(
        "--create-demo-data",
        action="store_true",
        help="Create a tiny demo dataset before inspection.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Phase 4 inspection."""

    args = parse_args()

    print("MNLogic inspection")
    print("------------------")
    print(f"rsbench root: {args.rsbench_root}")
    print(f"dataset root: {args.dataset_root}")
    print()
    print("Upstream mapping")
    print("----------------")
    print("Task name in thesis language: MNLogic")
    print("Dataset name in rsseval CLI: xor")
    print("Primary upstream loader: rsseval/rss/datasets/xor.py")
    print()

    if args.create_demo_data:
        create_demo_dataset(args.dataset_root)
        print("Created a small demo dataset with MNLogic-compatible file layout.")
        print()

    if not args.dataset_root.exists():
        print("Dataset root does not exist.")
        print("Run again with --create-demo-data to generate a tiny inspection dataset.")
        return

    result = inspect_dataset(args.dataset_root, args.rsbench_root)

    print("Split counts")
    print("------------")
    for split in result["split_summaries"]:
        print(f"{split.name}: {split.count}")

    print()
    print("Example sample")
    print("--------------")
    sample = result["sample"]
    print(f"image shape: {sample.image_shape}")
    print(f"label: {sample.label}")
    print(f"concepts: {sample.concepts}")

    print()
    print("Example batch shapes")
    print("--------------------")
    batch_shapes = result["batch_shapes"]
    print(f"images: {batch_shapes['images']}")
    print(f"labels: {batch_shapes['labels']}")
    print(f"concepts: {batch_shapes['concepts']}")

    print()
    print("Interpretation")
    print("--------------")
    print("The upstream task loader reads one grayscale strip image plus one joblib")
    print("metadata file per sample and returns (image, label, concepts).")
    print("In the rsseval XOR path, the concepts are four binary values and the")
    print("logic-based model uses a parity-style binary target.")


if __name__ == "__main__":
    main()
