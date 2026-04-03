"""Evaluate the Phase 5 plain neural baseline on MNLogic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from neurosymbolic_benchmark.data.mnlogic import create_mnlogic_dataloaders
from neurosymbolic_benchmark.data.mnlogic_inspection import create_demo_dataset
from neurosymbolic_benchmark.evaluation.mnlogic_eval import evaluate_mnlogic_classifier
from neurosymbolic_benchmark.models.plain_mnlogic_cnn import PlainMNLogicCNN
from neurosymbolic_benchmark.training.plain_nn_runner import (
    ensure_dir,
    save_json,
    select_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the plain MNLogic neural baseline.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Checkpoint produced by train_plain_nn.py.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("results/inspection/mnlogic_demo"),
        help="MNLogic dataset root.",
    )
    parser.add_argument(
        "--create-demo-data",
        action="store_true",
        help="Create a tiny demo dataset before evaluation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "ood"],
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and evaluate one split."""

    args = parse_args()

    if args.create_demo_data:
        create_demo_dataset(args.dataset_root)

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    model = PlainMNLogicCNN(
        hidden_dim=int(checkpoint["model_config"]["hidden_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataloaders = create_mnlogic_dataloaders(args.dataset_root, batch_size=4, num_workers=0)
    metrics, predictions = evaluate_mnlogic_classifier(model, dataloaders[args.split], device)

    run_name = args.checkpoint_path.parent.name
    metrics_dir = REPO_ROOT / "results" / "metrics" / "plain_nn" / run_name
    predictions_dir = REPO_ROOT / "results" / "predictions" / "plain_nn" / run_name
    ensure_dir(metrics_dir)
    ensure_dir(predictions_dir)

    metrics_path = metrics_dir / f"{args.split}_metrics_eval.json"
    predictions_path = predictions_dir / f"{args.split}_predictions_eval.csv"

    save_json(metrics_path, metrics)
    predictions.to_csv(predictions_path, index=False)

    print("Evaluation complete")
    print("-------------------")
    print(f"split: {args.split}")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"num_examples: {metrics['num_examples']}")
    print(f"metrics_path: {metrics_path}")
    print(f"predictions_path: {predictions_path}")


if __name__ == "__main__":
    main()
