import pytest

import tensordir as td


@pytest.fixture
def root(tmp_path):
    with td.use_root(tmp_path / "store") as path:
        yield path
