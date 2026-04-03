"""Simple Phase 2 environment sanity check."""

from __future__ import annotations

import importlib
from importlib import metadata

import torch


def package_version(module_name: str, package_name: str) -> str:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    return metadata.version(package_name)


def main() -> None:
    packages = [
        ("numpy", "numpy", "numpy"),
        ("pandas", "pandas", "pandas"),
        ("scikit-learn", "sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib", "matplotlib"),
        ("jupyter", "jupyter", "jupyter"),
        ("torch", "torch", "torch"),
        ("torchvision", "torchvision", "torchvision"),
    ]

    print("Environment sanity check")
    print("------------------------")
    for label, module_name, package_name in packages:
        print(f"{label}: {package_version(module_name, package_name)}")

    print()
    print("PyTorch backend")
    print("---------------")
    print(f"torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")
    print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
