"""Run every Python example in the docs pages that promise runnable scripts."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"
PAGES = ["extending.md"]


def _examples():
    for page in PAGES:
        text = (DOCS / page).read_text()
        for i, match in enumerate(re.finditer(r"```python\n(.*?)```", text, flags=re.S)):
            code = match.group(1)
            line = text[: match.start()].count("\n") + 2
            yield pytest.param(code, id=f"{page}:{line}")


@pytest.mark.parametrize("code", list(_examples()))
def test_docs_example_runs(code, tmp_path):
    required = re.match(r"# requires: (.*)\n", code)
    for module in (required.group(1).split() if required else []):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"needs {module}")
    script = tmp_path / "example.py"
    script.write_text(code)
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
