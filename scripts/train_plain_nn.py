"""Train the Phase 5 plain neural baseline on MNLogic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neurosymbolic_benchmark.data.mnlogic_inspection import create_demo_dataset
from neurosymbolic_benchmark.training.plain_nn_runner import (
    ensure_dir,
    train_plain_mnlogic_baseline,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Train the plain MNLogic neural baseline.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("results/inspection/mnlogic_demo"),
        help="MNLogic dataset root.",
    )
    parser.add_argument(
        "--create-demo-data",
        action="store_true",
        help="Create a tiny demo dataset before training.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="mnlogic_plain_nn_demo",
        help="Run name used for output folders.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the baseline and report the saved outputs."""

    args = parse_args()

    if args.create_demo_data:
        create_demo_dataset(args.dataset_root)

    run_root = Path(args.run_name)
    checkpoint_dir = REPO_ROOT / "results" / "checkpoints" / "plain_nn" / run_root
    log_dir = REPO_ROOT / "results" / "logs" / "plain_nn" / run_root
    metrics_dir = REPO_ROOT / "results" / "metrics" / "plain_nn" / run_root
    predictions_dir = REPO_ROOT / "results" / "predictions" / "plain_nn" / run_root

    for directory in [checkpoint_dir, log_dir, metrics_dir, predictions_dir]:
        ensure_dir(directory)

    outputs = train_plain_mnlogic_baseline(
        {
            "dataset_root": args.dataset_root,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "num_workers": args.num_workers,
            "device": args.device,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "metrics_dir": metrics_dir,
            "predictions_dir": predictions_dir,
        }
    )

    print("Training complete")
    print("-----------------")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

