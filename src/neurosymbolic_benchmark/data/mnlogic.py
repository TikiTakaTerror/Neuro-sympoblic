"""Local MNLogic dataset utilities for thesis-owned experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


MNLOGIC_SPLITS = ("train", "val", "test", "ood")


@dataclass
class MNLogicRecord:
    """Single MNLogic sample on disk."""

    sample_id: int
    image_path: Path
    label: int
    concepts: List[int]


class MNLogicDataset(Dataset):
    """Read MNLogic samples from a thesis-owned or upstream-compatible folder."""

    def __init__(self, dataset_root: Path | str, split: str) -> None:
        dataset_root = Path(dataset_root)
        if split not in MNLOGIC_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        self.dataset_root = dataset_root
        self.split = split
        self.split_root = dataset_root / split
        if not self.split_root.exists():
            raise FileNotFoundError(f"Missing split directory: {self.split_root}")

        self.transform = ToTensor()
        self.records = self._load_records()

    def _load_records(self) -> List[MNLogicRecord]:
        records: List[MNLogicRecord] = []
        for image_path in sorted(self.split_root.glob("*.png"), key=lambda path: int(path.stem)):
            metadata_path = self.split_root / f"{image_path.stem}.joblib"
            if not metadata_path.exists():
                continue

            metadata = joblib.load(metadata_path)
            label = int(bool(metadata["label"]))
            concepts = [int(value) for value in metadata["meta"]["concepts"]]

            records.append(
                MNLogicRecord(
                    sample_id=int(image_path.stem),
                    image_path=image_path,
                    label=label,
                    concepts=concepts,
                )
            )

        if not records:
            raise ValueError(f"No MNLogic samples found in {self.split_root}")

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("L")

        return {
            "image": self.transform(image),
            "label": torch.tensor(record.label, dtype=torch.long),
            "concepts": torch.tensor(record.concepts, dtype=torch.long),
            "sample_id": torch.tensor(record.sample_id, dtype=torch.long),
        }


def create_mnlogic_dataloaders(
    dataset_root: Path | str,
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Build dataloaders for all available MNLogic splits."""

    loaders: Dict[str, DataLoader] = {}
    for split in MNLOGIC_SPLITS:
        dataset = MNLogicDataset(dataset_root, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders

