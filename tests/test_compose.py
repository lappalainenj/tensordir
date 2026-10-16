"""Records composed from other records: navigable lineage, records in specs, checked inputs."""

import shutil

import numpy as np
import pytest

import tensordir as td


class Source(td.Record):
    __record_name__ = "ComposeSource"

    class Spec(td.Spec):
        seed: int = 0

    x: np.ndarray

    def build(self, spec):
        self.x = np.random.default_rng(spec.seed).normal(size=5)


class Scaled(td.Record):
    __record_name__ = "ComposeScaled"

    class Spec(td.Spec):
        scale: float = 2.0

    y: np.ndarray

    def build(self, spec, source: Source):
        self.y = source.x * spec.scale


class Summed(td.Record):
    __record_name__ = "ComposeSummed"

    class Spec(td.Spec):
        shift: float = 1.0

    z: np.ndarray

    def build(self, spec, scaled: Scaled, source: Source, weights: np.ndarray, sources: list):
        self.z = scaled.y + source.x * weights + spec.shift + sum(s.x for s in sources)


class Referencing(td.Record):
    __record_name__ = "ComposeReferencing"

    class Spec(td.Spec):
        scaled: Scaled
        offset: float = 0.0

    w: np.ndarray

    def build(self, spec):
        self.w = spec.scaled.y + spec.offset


class Nested(td.Record):
    __record_name__ = "ComposeNested"

    class Spec(td.Spec):
        source: Source.Spec = Source.Spec()
        optional: Source.Spec | None = None
        scale: float = 2.0

    y: np.ndarray

    def build(self, spec):
        self.y = Source.get(spec.source).x * spec.scale


class Workspace(td.Record):
    __record_name__ = "ComposeWorkspace"
    mutable = True

    class Spec(td.Spec):
        name: str = "w"


class Total(td.Record):
    __record_name__ = "ComposeTotal"

    total: float

    def build(self, spec, workspace: Workspace):
        self.total = float(workspace.load("v").sum())


@pytest.fixture
def chain(root):
    source = Source.get()
    scaled = Scaled.get(source=source)
    extra = Source.get({"seed": 1})
    summed = Summed.get(scaled=scaled, source=source, weights=np.ones(5), sources=[extra])
    return source, scaled, extra, summed


def test_inputs_reopen_records_and_keep_other_values(chain):
    source, scaled, extra, summed = chain
    inputs = summed.inputs
    assert inputs["scaled"] == scaled and inputs["source"] == source
    assert isinstance(inputs["scaled"], Scaled)
    assert inputs["sources"] == [extra]
    assert inputs["weights"] == td.Digest.of(np.ones(5))
    np.testing.assert_array_equal(inputs["scaled"].y, scaled.y)
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
    other = Source.get({"seed": 2})
    rebuilt = Summed.get(scaled=Scaled.get(source=other), source=other, weights=np.ones(5), sources=[extra])
    assert rebuilt.key != summed.key


def test_inputs_survive_a_moved_root(root, chain, tmp_path):
    _, scaled, _, summed = chain
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    with td.use_root(moved):
        inputs = Summed.at(moved / "ComposeSummed" / summed.key[:16]).inputs
        assert inputs["scaled"].path == moved / "ComposeScaled" / scaled.key[:16]
        assert inputs["scaled"].status == "done"


def test_deleted_input_reopens_as_absent(chain):
    _, scaled, _, summed = chain
    scaled.delete()
    assert summed.inputs["scaled"].status == "absent"
    assert {r.key: r.status for r in summed.lineage()}[scaled.key] == "absent"


def test_record_in_spec_field_is_revived(chain):
    _, scaled, _, _ = chain
    built = Referencing.get(Referencing.Spec(scaled=scaled))
    reopened = Referencing.at(built.path)
    assert isinstance(reopened.spec.scaled, Scaled)
    assert reopened.spec.scaled == scaled
    assert reopened.spec.hashed_dict() == built.spec.hashed_dict()
    assert reopened.key == built.key
    assert built.meta["spec"]["scaled"]["path"] == str(scaled.path)
    assert scaled in built.lineage()
    assert len(Referencing.query(scaled=scaled)) == 1


def test_unbuilt_input_is_refused_before_the_build(root):
    pending = Source.locate({"seed": 9})
    with pytest.raises(ValueError, match=r"input 'source'.*absent.*Source\.get"):
        Scaled.get(source=pending)
    assert Scaled.locate(source=pending).status == "absent"  # nothing recorded as failed
    with pytest.raises(ValueError, match="spec holds"):
        Referencing.get(Referencing.Spec(scaled=Scaled.locate(source=pending)))
    assert Scaled.lookup(source=pending) is None


def test_done_record_reopens_without_checking_inputs(chain):
    source, scaled, _, _ = chain
    source.delete()
    assert Scaled.get(source=Source.locate()) == scaled


def test_workspace_input_edit_rekeys_downstream(root):
    workspace = Workspace.get()
    workspace["v"] = np.zeros(3)
    first = Total.get(workspace=workspace)
    assert first.total == 0.0
    assert Total.get(workspace=workspace) == first
    workspace["v"] = np.ones(3)
    second = Total.get(workspace=workspace)
    assert second.key != first.key and second.total == 3.0
    assert first.meta["inputs"]["workspace"]["modified"] != second.meta["inputs"]["workspace"]["modified"]


def test_dict_given_to_nested_spec_field_becomes_a_spec(root):
    spec = Nested.Spec(source={"seed": 3}, optional={"seed": 4})
    assert isinstance(spec.source, Source.Spec) and spec.source.seed == 3
    assert isinstance(spec.optional, Source.Spec)
    assert spec == Nested.Spec(source=Source.Spec(seed=3), optional=Source.Spec(seed=4))
    assert Nested.get({"source": {"seed": 3}}).spec.source.seed == 3
    assert len(Nested.query(source__seed=3)) == 1 and len(Nested.query(source__seed=4)) == 0
