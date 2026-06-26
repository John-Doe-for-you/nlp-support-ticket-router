"""Strip outputs from 03_model_eval.ipynb so the source-controlled copy
is small and reviewable. Mirrors `scripts/strip_nb02_outputs.py`.

Run from repo root:  .venv/Scripts/python.exe scripts/strip_nb03_outputs.py
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "03_model_eval.ipynb"


def strip_outputs(nb_path: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(nb, nb_path)
    print(f"Stripped outputs from {nb_path}")


if __name__ == "__main__":
    if os.environ.get("KEEP_OUTPUTS") == "1":
        print("KEEP_OUTPUTS=1, leaving notebook as-is.")
    else:
        strip_outputs(NB_PATH)