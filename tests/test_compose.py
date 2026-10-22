"""Records composed from other records: navigable lineage, records in specs, checked inputs."""

import shutil

import numpy as np
import pytest

import tensordir as td


class Source(td.Record):
    __record_name__ = "ComposeSource"

    x: np.ndarray

    def __init__(self, /, *, seed: int = 0):
        self.x = np.random.default_rng(seed).normal(size=5)


class Scaled(td.Record):
    __record_name__ = "ComposeScaled"

    y: np.ndarray

    def __init__(self, source: Source, /, *, scale: float = 2.0):
        self.y = source.x * scale


class Summed(td.Record):
    __record_name__ = "ComposeSummed"

    z: np.ndarray

    def __init__(self, scaled: Scaled, source: Source, weights: np.ndarray, sources: list, /, *, shift: float = 1.0):
        self.z = scaled.y + source.x * weights + shift + sum(s.x for s in sources)


class Referencing(td.Record):
    __record_name__ = "ComposeReferencing"

    w: np.ndarray

    def __init__(self, /, *, scaled: Scaled, offset: float = 0.0):
        self.w = scaled.y + offset


class Nested(td.Record):
    __record_name__ = "ComposeNested"

    y: np.ndarray

    def __init__(self, /, *, source: Source.Spec = Source.Spec(), optional: Source.Spec | None = None, scale: float = 2.0):
        self.y = Source(**source).x * scale


class Workspace(td.Record):
    __record_name__ = "ComposeWorkspace"
    mutable = True

    def __init__(self, /, *, name: str = "w"):
        pass


class Total(td.Record):
    __record_name__ = "ComposeTotal"

    total: float

    def __init__(self, workspace: Workspace, /):
        self.total = float(workspace.load("v").sum())


@pytest.fixture
def chain(root):
    source = Source()
    scaled = Scaled(source)
    extra = Source(seed=1)
    summed = Summed(scaled, source, np.ones(5), [extra])
    return source, scaled, extra, summed


def test_inputs_reopen_records_and_keep_other_values(chain):
    source, scaled, extra, summed = chain
    inputs = summed.inputs
    assert inputs[0] == scaled and inputs[1] == source
    assert isinstance(inputs[0], Scaled)
    assert inputs[3] == [extra]
    assert inputs[2] == td.Digest.of(np.ones(5))
    np.testing.assert_array_equal(inputs[0].y, scaled.y)
    assert summed.meta["upstream"][scaled.key] == str(scaled.path)


def test_lineage_walks_the_dag_upstream_first(chain):
    source, scaled, extra, summed = chain
    lineage = summed.lineage()
    assert sorted(r.key for r in lineage) == sorted({source.key, scaled.key, extra.key})
    assert len(lineage) == 3  # source reached twice, listed once
    assert lineage.index(source) < lineage.index(scaled)
    assert source.lineage() == []


def test_upstream_change_rekeys_downstream(chain):
    _, _, extra, summed = chain
    other = Source(seed=2)
    rebuilt = Summed(Scaled(other), other, np.ones(5), [extra])
    assert rebuilt.key != summed.key


def test_inputs_survive_a_moved_root(root, chain, tmp_path):
    _, scaled, _, summed = chain
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    with td.use_root(moved):
        inputs = td.Record.at(moved / "ComposeSummed" / summed.key[:16]).inputs
        assert inputs[0].path == moved / "ComposeScaled" / scaled.key[:16]
        assert inputs[0].status == "done"


def test_deleted_input_reopens_as_absent(chain):
    _, scaled, _, summed = chain
    scaled.delete()
    assert summed.inputs[0].status == "absent"
    assert {r.key: r.status for r in summed.lineage()}[scaled.key] == "absent"


def test_record_in_spec_field_is_revived(chain):
    _, scaled, _, _ = chain
    built = Referencing(scaled=scaled)
    reopened = td.Record.at(built.path)
    assert isinstance(reopened.spec.scaled, Scaled)
    assert reopened.spec.scaled == scaled
    assert reopened.spec.hashed_dict() == built.spec.hashed_dict()
    assert reopened.key == built.key
    assert built.meta["spec"]["scaled"]["path"] == str(scaled.path)
    assert scaled in built.lineage()
    assert len(Referencing.query(scaled=scaled)) == 1


def test_unbuilt_input_is_refused_before_the_build(root):
    pending = Source.locate(seed=9)
    with pytest.raises(ValueError, match=r"input 'source'.*absent.*build it first"):
        Scaled(pending)
    assert Scaled.locate(pending).status == "absent"  # nothing recorded as failed
    with pytest.raises(ValueError, match="settings hold"):
        Referencing(scaled=Scaled.locate(pending))
    assert Scaled.lookup(pending) is None


def test_done_record_reopens_without_checking_inputs(chain):
    source, scaled, _, _ = chain
    source.delete()
    assert Scaled(Source.locate()) == scaled


def test_workspace_input_edit_rekeys_downstream(root):
    workspace = Workspace()
    workspace["v"] = np.zeros(3)
    first = Total(workspace)
    assert first.total == 0.0
    assert Total(workspace) == first
    workspace["v"] = np.ones(3)
    second = Total(workspace)
    assert second.key != first.key and second.total == 3.0
    assert first.meta["inputs"][0]["modified"] != second.meta["inputs"][0]["modified"]


def test_dict_given_to_nested_spec_field_becomes_a_spec(root):
    spec = Nested.Spec(source={"seed": 3}, optional={"seed": 4})
    assert isinstance(spec.source, Source.Spec) and spec.source.seed == 3
    assert isinstance(spec.optional, Source.Spec)
    assert spec == Nested.Spec(source=Source.Spec(seed=3), optional=Source.Spec(seed=4))
    assert Nested(source={"seed": 3}).spec.source.seed == 3
    assert len(Nested.query(source__seed=3)) == 1 and len(Nested.query(source__seed=4)) == 0
