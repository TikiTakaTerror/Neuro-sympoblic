"""Training runner for the plain MNLogic neural baseline."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

from neurosymbolic_benchmark.data.mnlogic import create_mnlogic_dataloaders
from neurosymbolic_benchmark.evaluation.mnlogic_eval import evaluate_mnlogic_classifier
from neurosymbolic_benchmark.models.plain_mnlogic_cnn import PlainMNLogicCNN


def set_seed(seed: int) -> None:
    """Set all main random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Resolve the requested device."""

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""

    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize_config(config: Dict[str, object]) -> Dict[str, object]:
    """Convert config values into checkpoint-safe primitives."""

    clean_config: Dict[str, object] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            clean_config[key] = str(value)
        else:
            clean_config[key] = value
    return clean_config


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch."""

    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_accuracy: float,
    config: Dict[str, object],
) -> None:
    """Save a training checkpoint."""

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "model_config": {"hidden_dim": config["hidden_dim"]},
            "run_config": sanitize_config(config),
        },
        path,
    )


def train_plain_mnlogic_baseline(config: Dict[str, object]) -> Dict[str, Path]:
    """Train the thesis-owned plain neural baseline."""

    set_seed(int(config["seed"]))
    device = select_device(str(config["device"]))
    dataloaders = create_mnlogic_dataloaders(
        dataset_root=config["dataset_root"],
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
    )

    model = PlainMNLogicCNN(hidden_dim=int(config["hidden_dim"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config["checkpoint_dir"])
    log_dir = Path(config["log_dir"])
    metrics_dir = Path(config["metrics_dir"])
    predictions_dir = Path(config["predictions_dir"])

    for directory in [checkpoint_dir, log_dir, metrics_dir, predictions_dir]:
        ensure_dir(directory)

    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    history = []
    best_val_accuracy = -1.0
    start_time = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            criterion,
            device,
        )
        val_metrics, _ = evaluate_mnlogic_classifier(model, dataloaders["val"], device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            best_val_accuracy,
            config,
        )

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val_accuracy,
                config,
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / "history.csv", index=False)

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_predictions = evaluate_mnlogic_classifier(
        model, dataloaders["test"], device
    )
    ood_metrics, ood_predictions = evaluate_mnlogic_classifier(
        model, dataloaders["ood"], device
    )

    test_predictions.to_csv(predictions_dir / "test_predictions.csv", index=False)
    ood_predictions.to_csv(predictions_dir / "ood_predictions.csv", index=False)

    save_json(metrics_dir / "test_metrics.json", test_metrics)
    save_json(metrics_dir / "ood_metrics.json", ood_metrics)
    save_json(
        log_dir / "train_summary.json",
        {
            "device": str(device),
            "seed": int(config["seed"]),
            "epochs": int(config["epochs"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "hidden_dim": int(config["hidden_dim"]),
            "best_val_accuracy": best_val_accuracy,
            "training_seconds": time.time() - start_time,
            "best_checkpoint": str(best_checkpoint_path),
            "last_checkpoint": str(last_checkpoint_path),
            "test_metrics_path": str(metrics_dir / "test_metrics.json"),
            "ood_metrics_path": str(metrics_dir / "ood_metrics.json"),
            "test_predictions_path": str(predictions_dir / "test_predictions.csv"),
            "ood_predictions_path": str(predictions_dir / "ood_predictions.csv"),
        },
    )

    return {
        "best_checkpoint": best_checkpoint_path,
        "last_checkpoint": last_checkpoint_path,
        "history_csv": log_dir / "history.csv",
        "train_summary": log_dir / "train_summary.json",
        "test_metrics": metrics_dir / "test_metrics.json",
        "ood_metrics": metrics_dir / "ood_metrics.json",
        "test_predictions": predictions_dir / "test_predictions.csv",
        "ood_predictions": predictions_dir / "ood_predictions.csv",
    }
