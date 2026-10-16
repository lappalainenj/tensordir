"""Execute every example notebook in docs/examples, top to bottom, in a fresh kernel."""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "docs" / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))
REQUIRED = ["nbclient", "nbformat", "ipykernel", "xarray", "pandas", "pyarrow", "joblib", "jaxtyping", "h5py", "yaml", "deltalake", "duckdb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_example_notebook_runs(path, tmp_path):
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}})
    client.execute()  # raises CellExecutionError with the failing cell and its traceback


def test_examples_are_stored_without_outputs():
    """The docs build executes the notebooks; stored outputs would only go stale."""
    import json

    for path in NOTEBOOKS:
        cells = json.loads(path.read_text())["cells"]
        assert not any(c.get("outputs") or c.get("execution_count") for c in cells if c["cell_type"] == "code"), path.name
