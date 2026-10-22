import re
import subprocess
import sys
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_readme_quickstart_runs(tmp_path):
    (code,) = re.findall(r"## Quickstart\n\n```python\n(.*?)```", README.read_text(), flags=re.S)
    script = tmp_path / "quickstart.py"
    script.write_text(code)
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "done (8, 2)" in result.stdout
    assert (tmp_path / "results" / "Fit").is_dir()
