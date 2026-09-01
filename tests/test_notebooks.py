"""The example notebooks must run, and must ship without stored outputs.

Example notebooks rot silently: an API change breaks them and nobody notices
until a reader tries one. Executing them here turns that into a test failure.

Marked ``slow`` -- run the fast suite with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path("notebooks")


def _notebooks() -> list[Path]:
    if not NOTEBOOK_DIR.exists():  # pragma: no cover - only when run from elsewhere
        return []
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


NOTEBOOKS = _notebooks()


@pytest.mark.skipif(not NOTEBOOKS, reason="run from the repository root")
def test_the_expected_notebooks_are_present():
    names = {path.name for path in NOTEBOOKS}
    assert "01_quickstart_standardisation.ipynb" in names
    assert len(names) >= 4


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_carry_no_stored_outputs(path):
    """Stored outputs bloat diffs and have leaked credentials before now."""
    content = json.loads(path.read_text(encoding="utf-8"))
    offenders = [
        index
        for index, cell in enumerate(content["cells"])
        if cell.get("outputs") or cell.get("execution_count")
    ]
    assert not offenders, f"{path.name}: cells {offenders} carry outputs"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_contain_no_credentials(path):
    """A JWT in a notebook output is how DESP tokens have escaped before."""
    text = path.read_text(encoding="utf-8")
    for marker in ("Bearer eyJ", "access_token", "password"):
        assert marker not in text, f"{path.name} contains {marker!r}"


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_execute(path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": "."}}
    )
    client.execute()
