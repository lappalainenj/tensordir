"""Packs: many small records of one type stored as one record, read back as members."""

import pickle

import numpy as np
import pytest

import tensordir as td
from tensordir import pack as pack_module
from tensordir._record import _REGISTRY

BUILDS: list[int] = []


class Segment(td.Record):
    __record_name__ = "PackSegment"

    def __init__(self, seg: int, /, *, scale: float = 1.0):
        BUILDS.append(seg)
        n = seg % 4 + 1
        self.write("pos", (np.arange(n * 3, dtype="i2").reshape(n, 3) + seg) * int(scale))
        self.write("area", np.array(seg * scale, dtype="f4"))
        if seg % 2:
            self.write("odd", np.full(seg, seg, dtype="u1"))  # only some members have it
        self.write("summary", {"seg": seg})


class Failing(td.Record):
    __record_name__ = "PackFailing"

    def __init__(self, seg: int, /):
        if seg == 3:
            raise RuntimeError("bad segment")
        self.write("x", np.arange(seg))


class Ragged(td.Record):
    __record_name__ = "PackRagged"

    def __init__(self, seg: int, /):
        self.write("x", np.zeros((2, seg)))


@pytest.fixture(autouse=True)
def _clear_builds():
    BUILDS.clear()


def _pack(segs, **settings):
    return td.Pack(Segment)([Segment.call(s, **settings) for s in segs])


def test_members_keep_their_standalone_keys_and_fields(root):
    pack = _pack([5, 2, 9])
    for seg in (5, 2, 9):
        member = pack.member(Segment.call(seg))
        standalone = Segment.get(seg, root=root / "standalone")
        assert member.key == standalone.key == Segment.compute_key(seg)
        assert isinstance(member, Segment) and member.status == "done" and member.spec == standalone.spec
        assert member.keys() == standalone.keys()
        for name in standalone.keys():
            np.testing.assert_array_equal(member.load(name), standalone.load(name))
        np.testing.assert_array_equal(member.pos, standalone.pos)


def test_the_pack_key_ignores_order_and_duplicates(root):
    a = _pack([5, 2, 9])
    b = _pack([9, 5, 2, 5])
    assert a.path == b.path and a.member_keys() == sorted(Segment.compute_key(s) for s in (2, 5, 9))
    assert _pack([5, 2]).key != a.key


def test_reopening_runs_no_build(root):
    pack = _pack([1, 2])
    assert sorted(BUILDS) == [1, 2]
    BUILDS.clear()
    again = _pack([2, 1])
    reopened = td.Record.at(pack.path)
    assert BUILDS == [] and again == pack and type(reopened) is td.Pack(Segment)
    assert reopened.member(Segment.call(1)).load("summary") == {"seg": 1}


def test_finished_standalone_members_are_reused(root):
    Segment.get(4)
    BUILDS.clear()
    _pack([4, 6])
    assert BUILDS == [6]


def test_layout_of_the_pack(root):
    pack = _pack([1, 2, 3])
    assert pack.keys() == ["area", "area.offsets", "members", "odd", "odd.offsets", "pos", "pos.offsets"]
    order = np.argsort([Segment.compute_key(s) for s in (1, 2, 3)])  # members are in key order
    assert np.diff(pack.load("pos.offsets")).tolist() == [[2, 3, 4][i] for i in order]
    assert np.diff(pack.load("odd.offsets")).tolist() == [[1, 0, 3][i] for i in order]
    assert "odd" not in pack.member(Segment.call(2)).keys()


def test_member_reads(root):
    pack = _pack([7, 3])
    member = pack.member(Segment.call(7))
    pos = member.pos
    out = member.read({"pos": [3, 0, -1], "odd": [(1, 3), (5, 7)]})
    np.testing.assert_array_equal(out["pos"], pos[[3, 0, -1]])
    np.testing.assert_array_equal(out["odd"], [7, 7, 7, 7])
    np.testing.assert_array_equal(member.read({"pos": slice(1, None)})["pos"], pos[1:])
    with pytest.raises(IndexError):
        member.read({"pos": [4]})  # 4 rows: the next member's rows are out of reach
    assert member.area.shape == () and float(member.area) == 7.0


def test_members_are_read_only(root):
    member = _pack([1]).member(Segment.call(1))
    with pytest.raises(td.ReadOnlyError):
        member.write("x", np.arange(2))
    with pytest.raises(td.ReadOnlyError):
        member.pos = np.arange(2)
    with pytest.raises(TypeError, match="publish the pack"):
        member.publish(root / "elsewhere")


def test_members_pickle_and_compare(root):
    pack = _pack([1, 2])
    member = pack.member(Segment.call(2))
    back = pickle.loads(pickle.dumps(member))
    assert back == member and back.key == member.key
    np.testing.assert_array_equal(back.pos, member.pos)


def test_members_table_matches_standalone_records(root):
    calls = [Segment.call(s, scale=2.0) for s in (1, 2, 3)]
    pack = td.Pack(Segment)(calls)
    standalone = td.Collection([Segment.get(s, scale=2.0, root=root / "standalone") for s in (1, 2, 3)])
    packed = pack.members.table(sizes=False)
    alone = standalone.table(sizes=False)
    assert sorted(packed["key"]) == sorted(alone["key"])
    assert set(packed["spec.scale"]) == {2.0} and list(packed.columns) == list(alone.columns)


def test_a_failing_member_fails_the_pack(root):
    calls = [Failing.call(s) for s in (1, 3)]
    with pytest.raises(td.BuildFailed, match=Failing.compute_key(3)[:16]):
        td.Pack(Failing)(calls)


def test_members_must_agree_on_trailing_shapes(root):
    with pytest.raises(td.BuildFailed, match="trailing shape"):
        td.Pack(Ragged)([Ragged.call(1), Ragged.call(2)])


def test_calls_must_be_of_the_member_type(root):
    with pytest.raises(TypeError, match="packs calls of"):
        td.Pack(Segment)([Failing.call(1)])
    with pytest.raises(TypeError):
        td.Pack(td.Pack(Segment))


def test_a_pack_reopens_without_calls_but_cannot_build_without_them(root):
    pack_type = td.Pack(Segment)
    keys = pack_type.keys_of([Segment.call(1)])
    with pytest.raises(ValueError, match="needs its calls"):
        pack_type.get(keys)
    assert pack_type.locate(keys).status == "absent"  # nothing was recorded as failed
    pack = _pack([1])
    assert pack_type.get(pack_type.keys_of([Segment.call(1)])).path == pack.path


def test_pack_types_are_found_by_type_id(root, monkeypatch):
    pack = _pack([1])
    type_id = td.Pack(Segment).type_id()
    assert type_id == "Pack[PackSegment]"
    monkeypatch.delitem(_REGISTRY, type_id)
    monkeypatch.setattr(pack_module, "_PACKS", {})
    reopened = td.Record.at(pack.path)
    assert reopened.type_id() == type_id and reopened.member(Segment.call(1)).load("summary") == {"seg": 1}


def test_calls_bind_without_building(root):
    call = Segment.call(3, scale=2.0)
    assert call.key == Segment.compute_key(3, scale=2.0) and BUILDS == []
    assert call.settings() == {"scale": 2.0} and call.inputs == (3,)
    assert Segment.locate(3, scale=2.0).status == "absent"


def test_packs_publish(root, tmp_path):
    pack = _pack([1, 2])
    copy = pack.publish(tmp_path / "published")
    np.testing.assert_array_equal(copy.member(Segment.call(2)).pos, pack.member(Segment.call(2)).pos)
