#!/usr/bin/env python3
"""Combine multiple Jupyter notebooks into a single notebook (.ipynb).

Features:
- Preserves original cells (markdown & code) including existing metadata.id values.
- Adds a top instructional markdown cell and a dataset path code cell.
- Inserts a markdown section header before each source notebook.
- Keeps magics intact in code cells.

Usage:
    python3 code/combine_notebooks_to_ipynb.py \
        --out code/final_project_combined.ipynb \
        LABS-03_Systems.ipynb LABS-06_Design-2.ipynb Fairness_lab.ipynb LABS_09_Analytics(2).ipynb
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List

DEFAULT_ORDER = [
    Path("LABS-03_Systems.ipynb"),
    Path("LABS-06_Design-2.ipynb"),
    Path("Fairness_lab.ipynb"),
    Path("LABS_09_Analytics(2).ipynb"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine notebooks into one .ipynb")
    p.add_argument("notebooks", nargs="*", type=Path, help="Notebooks in desired order")
    p.add_argument("--out", type=Path, default=Path("code/final_project_combined.ipynb"))
    return p.parse_args()


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    if args.notebooks:
        nb_paths = [p for p in args.notebooks if p.exists()]
    else:
        nb_paths = [p for p in DEFAULT_ORDER if p.exists()]
    if not nb_paths:
        raise SystemExit("No notebooks found to combine.")

    first_nb = load_notebook(nb_paths[0])
    combined_cells = []

    # Intro markdown cell
    combined_cells.append({
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": [
            f"# Final Project Combined Notebook\n",
            f"Generated on {datetime.now().isoformat(timespec='seconds')}\n",
            "\n",
            "This notebook merges several lab notebooks into a single workflow for students.\n",
            "Markdown content preserved; execute sequentially.\n",
            "Magics require Jupyter/IPython kernel.\n",
            "Update the dataset path below before starting.\n",
        ],
    })

    # Dataset path code cell
    combined_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": [
            "# Set dataset path used throughout the notebook\n",
            "DATASET_PATH = 'data/your_dataset.csv'  # TODO: replace with actual file path\n",
            "\n",
            "# Example:\n",
            "# import pandas as pd\n",
            "# df = pd.read_csv(DATASET_PATH)\n",
            "# df.head()\n",
        ],
    })

    # Append each notebook's cells with a section header
    for path in nb_paths:
        nb = load_notebook(path)
        section_cell = {
            "cell_type": "markdown",
            "metadata": {"language": "markdown"},
            "source": [
                f"## Section: {path.name}\n",
                "---\n",
            ],
        }
        combined_cells.append(section_cell)
        for cell in nb.get("cells", []):
            # Preserve cell exactly; ensure metadata.language present
            meta = cell.get("metadata", {})
            if "language" not in meta:
                # Infer language for code cells
                if cell.get("cell_type") == "markdown":
                    meta["language"] = "markdown"
                else:
                    meta["language"] = "python"
            cell["metadata"] = meta
            combined_cells.append(cell)

    # Build final notebook structure
    final_nb = {
        "cells": combined_cells,
        "metadata": first_nb.get("metadata", {}),
        "nbformat": first_nb.get("nbformat", 4),
        "nbformat_minor": first_nb.get("nbformat_minor", 5),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(final_nb, f, indent=2)
    print(f"Wrote combined notebook: {args.out}")


if __name__ == "__main__":
    main()
