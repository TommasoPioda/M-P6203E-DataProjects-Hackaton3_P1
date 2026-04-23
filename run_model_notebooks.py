#!/usr/bin/env python3
"""Run all model notebooks under the Models directory.

This script executes notebooks recursively from `Models/` and writes
executed copies under `executed_notebooks/Models/...`.

Usage:
    python run_model_notebooks.py
    python run_model_notebooks.py --dry-run
    python run_model_notebooks.py --notebooks Models/Transformer.ipynb Models/embedding_based/notebooks/KNN_baseline.ipynb
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError as exc:
    raise SystemExit(
        "Missing required packages. Install with: pip install nbformat nbconvert"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NOTEBOOK_ROOT = REPO_ROOT / "Models"
OUTPUT_ROOT = REPO_ROOT / "executed_notebooks"


def find_model_notebooks(root: Path) -> list[Path]:
    notebooks = []
    for path in sorted(root.rglob("*.ipynb")):
        if "__pycache__" in path.parts:
            continue
        notebooks.append(path)
    return notebooks


def execute_notebook(notebook_path: Path, output_path: Path, timeout: int = 3600) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    resources = {"metadata": {"path": str(REPO_ROOT)}}

    try:
        ep.preprocess(nb, resources=resources)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR executing {notebook_path}: {exc}")
        with output_path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        raise

    with output_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute model notebooks and save executed copies."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the notebooks that would be executed without running them.",
    )
    parser.add_argument(
        "--notebooks",
        nargs="*",
        default=None,
        help="Optional explicit list of notebooks to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory to write executed notebooks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Execution timeout per notebook in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.notebooks:
        notebooks = [Path(nb).resolve() for nb in args.notebooks]
    else:
        notebooks = find_model_notebooks(DEFAULT_NOTEBOOK_ROOT)

    if not notebooks:
        print("No notebooks found to execute.")
        return 1

    print(f"Found {len(notebooks)} notebook(s) to execute:")
    for path in notebooks:
        print(f" - {path}")

    if args.dry_run:
        print("Dry run complete. No notebooks were executed.")
        return 0

    failed = []
    for notebook_path in notebooks:
        rel_path = notebook_path.relative_to(REPO_ROOT)
        output_path = args.output_dir / rel_path
        print(f"\nExecuting notebook: {rel_path}")
        try:
            execute_notebook(notebook_path, output_path, timeout=args.timeout)
            print(f"Saved executed notebook to: {output_path}")
        except Exception:
            failed.append(notebook_path)
            print(f"Failed: {notebook_path}")
            continue

    if failed:
        print("\nThe following notebooks failed:")
        for path in failed:
            print(f" - {path}")
        return 1

    print("\nAll notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
