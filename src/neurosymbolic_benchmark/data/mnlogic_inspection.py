"""Helpers for inspecting the upstream MNLogic task."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


@dataclass
class SplitInspection:
    """Small summary of one split."""

    name: str
    count: int


@dataclass
class SampleInspection:
    """Summary of one loaded sample."""

    image_shape: List[int]
    label: int
    concepts: List[int]


DEMO_SPLITS: Dict[str, List[List[int]]] = {
    "train": [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ],
    "val": [
        [0, 1, 0, 1],
        [1, 1, 1, 0],
    ],
    "test": [
        [1, 1, 1, 1],
        [1, 0, 1, 0],
    ],
    "ood": [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ],
}


def repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[3]


def default_rsbench_root() -> Path:
    """Return the default sparse rsbench checkout path."""

    return repo_root() / "external" / "rsbench-code"


def load_upstream_xor_dataset_class(rsbench_root: Path):
    """Load the upstream XORDataset class directly from its file."""

    dataset_path = (
        rsbench_root / "rsseval" / "rss" / "datasets" / "utils" / "xor_creation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rsbench_xor_creation", dataset_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.XORDataset


def parity_label(concepts: List[int]) -> int:
    """Match the hard-coded rsseval parity-style label semantics."""

    return int(sum(concepts) % 2 == 0)


def make_demo_image(concepts: List[int]) -> np.ndarray:
    """Create a simple 4-panel grayscale strip with MNLogic-compatible shape."""

    panel_size = 28
    image = np.zeros((panel_size, panel_size * len(concepts)), dtype=np.uint8)

    for idx, concept in enumerate(concepts):
        start = idx * panel_size
        end = start + panel_size
        fill = 225 if concept == 1 else 30
        image[:, start:end] = fill

        # Add a border so panels are visible when visually inspected.
        image[:, start] = 128
        image[:, end - 1] = 128
        image[0, start:end] = 128
        image[-1, start:end] = 128

    return image


def create_demo_dataset(dataset_root: Path) -> None:
    """Create a tiny MNLogic demo dataset with the upstream file layout."""

    for split, worlds in DEMO_SPLITS.items():
        split_root = dataset_root / split
        split_root.mkdir(parents=True, exist_ok=True)

        for old_file in split_root.glob("*.png"):
            old_file.unlink()
        for old_file in split_root.glob("*.joblib"):
            old_file.unlink()

        for idx, concepts in enumerate(worlds):
            image = make_demo_image(concepts)
            image_path = split_root / f"{idx}.png"
            meta_path = split_root / f"{idx}.joblib"

            Image.fromarray(image, mode="L").save(image_path)
            joblib.dump(
                {
                    "label": bool(parity_label(concepts)),
                    "meta": {"concepts": concepts},
                },
                meta_path,
            )


def inspect_dataset(dataset_root: Path, rsbench_root: Path) -> Dict[str, object]:
    """Load a dataset through the upstream XORDataset reader and summarize it."""

    XORDataset = load_upstream_xor_dataset_class(rsbench_root)

    split_summaries: List[SplitInspection] = []
    loaded_splits = {}
    for split in ["train", "val", "test", "ood"]:
        dataset = XORDataset(base_path=str(dataset_root), split=split)
        loaded_splits[split] = dataset
        split_summaries.append(SplitInspection(name=split, count=len(dataset)))

    train_dataset = loaded_splits["train"]
    sample_image, sample_label, sample_concepts = train_dataset[0]
    sample = SampleInspection(
        image_shape=list(sample_image.shape),
        label=int(sample_label),
        concepts=[int(value) for value in sample_concepts.tolist()],
    )

    batch_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=False)
    batch_images, batch_labels, batch_concepts = next(iter(batch_loader))

    return {
        "split_summaries": split_summaries,
        "sample": sample,
        "batch_shapes": {
            "images": list(batch_images.shape),
            "labels": list(batch_labels.shape),
            "concepts": list(batch_concepts.shape),
        },
    }
