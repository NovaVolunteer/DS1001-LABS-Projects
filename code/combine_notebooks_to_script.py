#!/usr/bin/env python3
"""
Combine multiple Jupyter notebooks into a single Python script.

- Preserves markdown as commented blocks.
- Keeps IPython/Jupyter magics (%, %% ) as-is.
- Inserts clear section headers between notebooks and cells.

Note: The resulting script is best executed in IPython/Jupyter so that
magics run correctly. For standard `python` execution, magics will raise
SyntaxError.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Union
from datetime import datetime

CellSource = Union[str, List[str]]


def _to_lines(src: CellSource) -> List[str]:
    if isinstance(src, list):
        return src
    # Ensure we split while preserving newlines as in list form
    return [line for line in src.splitlines(keepends=True)]


def _write_markdown_as_comments(lines: Iterable[str]) -> str:
    out: List[str] = []
    for ln in lines:
        # Strip only trailing "\n" to avoid doubling newlines when re-adding
        if ln.endswith("\n"):
            ln_no_nl = ln[:-1]
            out.append(f"# {ln_no_nl}\n")
        else:
            out.append(f"# {ln}\n")
    # Ensure a trailing newline between cells
    if not out or not out[-1].endswith("\n"):
        out.append("\n")
    return "".join(out)


essential_header = (
    "# ================================================\n"
    "# NOTEBOOK SECTION\n"
    "# ================================================\n"
)


def combine_notebooks(notebooks: List[Path], output: Path) -> None:
    chunks: List[str] = []

    # File header
    header = (
        "# Combined notebook script\n"
        f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n"
        "# This script aggregates multiple notebooks.\n"
        "# Markdown cells are included as comments.\n"
        "# IPython/Jupyter magics are preserved and require IPython to run.\n"
        "\n"
        "# Optional: set the dataset path students should use:\n"
        "DATASET_PATH = 'data/your_dataset.csv'  # Update as needed\n\n"
    )
    chunks.append(header)

    for nb_path in notebooks:
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {nb_path}")

        with nb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)

        nb_name = nb_path.name
        chunks.append(essential_header)
        chunks.append(f"# Notebook: {nb_name}\n\n")

        cells = nb.get("cells", [])
        for idx, cell in enumerate(cells, start=1):
            ctype = cell.get("cell_type", "")
            source = _to_lines(cell.get("source", []))

            # Cell banner
            chunks.append("# --------------------------------\n")
            chunks.append(f"# Cell {idx} - {ctype}\n")
            chunks.append("# --------------------------------\n")

            if ctype == "markdown":
                chunks.append(_write_markdown_as_comments(source))
                chunks.append("\n")
            elif ctype == "code":
                # Write code as-is to preserve magics
                chunks.append("".join(source))
                # Ensure exactly one blank line after each code cell
                if not (len(source) and source[-1].endswith("\n")):
                    chunks.append("\n")
                chunks.append("\n")
            else:
                # Unknown cell types treated as commented text
                chunks.append(_write_markdown_as_comments(["Unsupported cell type\n"]))

        # Spacer between notebooks
        chunks.append("\n\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(chunks), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine Jupyter notebooks into a single .py script")
    p.add_argument(
        "--out",
        dest="out",
        type=Path,
        default=Path("code/final_project_combined.py"),
        help="Output .py file path",
    )
    p.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths in desired order. If omitted, uses a default set if present.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.notebooks:
        notebooks = [Path(p) for p in args.notebooks]
    else:
        # Default order (adjust as needed later)
        candidates = [
            Path("LABS-03_Systems.ipynb"),
            Path("LABS-06_Design-2.ipynb"),
            Path("Fairness_lab.ipynb"),
            Path("LABS_09_Analytics(2).ipynb"),
        ]
        notebooks = [p for p in candidates if p.exists()]
        if not notebooks:
            raise SystemExit("No default notebooks found. Provide paths explicitly.")

    combine_notebooks(notebooks, args.out)
    print(f"Wrote combined script to: {args.out}")


if __name__ == "__main__":
    main()
