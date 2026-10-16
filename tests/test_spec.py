import dataclasses
import enum
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

import tensordir as td


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


class Inner(td.Spec):
    width: int = 4
    scale: float = 1.0


class Outer(td.Spec):
    n: int
    lr: float = 0.1
    name: str = "a"
    path: Path = Path("data/x")
    shape: tuple[int, ...] = (2, 3)
    mode: Mode = Mode.FAST
    inner: Inner = td.field(default_factory=Inner)
    cutoff: float = math.inf
    label: str | None = None
    weights: td.Digest | None = None
    device: str = td.field("cpu", hashed=False)


class V1(td.Spec):
    n: int


class V2(td.Spec):
    n: int
    prior: float = td.field(1.0, since=2)


def test_frozen_keyword_only_dataclass():
    s = Outer(n=1)
    assert dataclasses.is_dataclass(s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.n = 2
    with pytest.raises(TypeError):
        Outer(1)
    assert s.replace(n=2).n == 2 and s.n == 1
    assert hash(s) == hash(Outer(n=1))


def test_round_trip_through_dict():
    s = Outer(
        n=3, lr=0.25, path=Path("/abs/p"), shape=(4,), mode=Mode.SLOW, inner=Inner(width=8),
        cutoff=-math.inf, label="x", weights=td.Digest.of(np.arange(3)), device="cuda",
    )
    back = Outer.from_dict(s.to_dict())
    assert back == s
    assert isinstance(back.path, Path) and isinstance(back.shape, tuple) and back.mode is Mode.SLOW
    assert isinstance(back.inner, Inner) and back.cutoff == -math.inf


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="no fields"):
        V1.from_dict({"n": 1, "extra": 2})


def test_unhashed_fields_do_not_change_the_key():
    a, b = Outer(n=1, device="cpu"), Outer(n=1, device="cuda")
    assert a != b
    assert a.key() == b.key()
    assert "device" not in a.hashed_dict() and a.to_dict()["device"] == "cpu"


def test_since_keeps_old_keys():
    assert V1(n=5).key() == V2(n=5).key()
    assert V2(n=5, prior=np.float64(1.0)).key() == V1(n=5).key()
    assert V2(n=5, prior=2.0).key() != V1(n=5).key()
    assert "prior" in V2(n=5).to_dict()


def test_field_option_validation():
    with pytest.raises(TypeError, match="needs a default"):
        td.field(since=2)
    with pytest.raises(TypeError, match="hashed"):
        td.field(1, hashed=False, since=2)


def test_int_given_for_float_field_hashes_like_float():
    assert Outer(n=1, lr=1).key() == Outer(n=1, lr=1.0).key()
    assert isinstance(Outer(n=1, lr=1).lr, float)
    assert Outer(n=1, lr=np.float32(0.5)).key() == Outer(n=1, lr=0.5).key()


def test_dict_given_for_nested_spec_field_becomes_the_spec():
    outer = Outer(n=1, inner={"width": 8})
    assert isinstance(outer.inner, Inner) and outer.inner.width == 8 and outer.inner.scale == 1.0
    assert outer == Outer(n=1, inner=Inner(width=8))
    assert Outer(n=1).replace(inner={"width": 8}) == outer


def test_invalid_values_fail_at_construction():
    with pytest.raises(td.CanonicalizationError, match="NaN"):
        Outer(n=1, lr=float("nan"))
    with pytest.raises(td.CanonicalizationError, match="Digest"):
        V1(n=np.arange(3))


def test_nested_spec_hashing_respects_inner_options():
    class Sub(td.Spec):
        a: int = 1
        dev: str = td.field("cpu", hashed=False)

    class Top(td.Spec):
        sub: Sub = td.field(default_factory=Sub)

    assert Top(sub=Sub(dev="gpu")).key() == Top().key()
    assert Top().to_dict() == {"sub": {"a": 1, "dev": "cpu"}}


def test_pickle_is_canonical_and_lossless():
    s = Outer(n=np.int64(2), device="cuda")
    back = pickle.loads(pickle.dumps(s))
    assert back == s and type(back.n) is int


# Keys pinned across releases: a change here orphans every stored record.
GOLDEN = {
    "v1": "11d0a8967009cbcdf468f09e5b09e73e7119b528c35a0e0b23f2ae052786b8fa",  # sha256(b'{"n":5}')
    "outer": "232b6a2ba1e3b65fdcb933e975a178c52c7e4cf42565ff068e7f11bd74007aec",
}


def test_golden_spec_keys():
    keys = {
        "v1": V1(n=5).key(),
        "outer": Outer(n=3, weights=td.Digest.of(np.arange(6, dtype="<i8").reshape(2, 3))).key(),
    }
    assert keys == GOLDEN
