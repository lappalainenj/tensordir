import enum
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tensordir import CanonicalizationError, Digest
from tensordir._canonical import canonical, dumps, sha256, untag


class Color(enum.Enum):
    RED = "red"


def test_scalars():
    assert canonical(np.int64(3)) == 3 and type(canonical(np.int64(3))) is int
    assert canonical(np.float32(0.5)) == 0.5 and type(canonical(np.float32(0.5))) is float
    assert canonical(np.bool_(True)) is True
    assert canonical(True) is True and canonical(1) == 1 and dumps(canonical(True)) != dumps(canonical(1))
    assert canonical(np.str_("a")) == "a"
    assert canonical(None) is None


def test_negative_zero_and_infinities():
    assert dumps(canonical(-0.0)) == dumps(canonical(0.0)) == "0.0"
    assert canonical(math.inf) == {"__float__": "inf"}
    assert canonical(-np.inf) == {"__float__": "-inf"}
    assert untag(canonical([math.inf, -math.inf])) == [math.inf, -math.inf]


def test_nan_rejected():
    with pytest.raises(CanonicalizationError, match="NaN"):
        canonical({"a": [1.0, float("nan")]})
    with pytest.raises(CanonicalizationError, match=r"value\.a\[1\]"):
        canonical({"a": [1.0, np.nan]})


def test_paths_enums_sets_tuples():
    assert canonical(Path("a") / "b") == "a/b"
    assert canonical(PurePosixPath("/x/y")) == "/x/y"
    assert canonical(Color.RED) == "red"
    assert canonical((1, 2)) == canonical([1, 2]) == [1, 2]
    assert canonical({3, 1, 2}) == [1, 2, 3]
    assert canonical(frozenset({"b", "a"})) == ["a", "b"]


def test_mapping_rules():
    assert list(canonical({"b": 1, "a": 2})) == ["a", "b"]
    with pytest.raises(CanonicalizationError, match="keys must be str"):
        canonical({1: "a"})
    with pytest.raises(CanonicalizationError, match="reserved"):
        canonical({"__digest__": "x"})


def test_arrays_need_digest():
    a = np.arange(3)
    with pytest.raises(CanonicalizationError, match="Digest"):
        canonical({"a": a})
    assert canonical({"a": Digest.of(a)}) == {"a": {"__digest__": Digest.of(a).sha256}}
    assert canonical(a, digest_arrays=True) == canonical(Digest.of(a))


def test_unknown_types_rejected():
    with pytest.raises(CanonicalizationError, match="object"):
        canonical(object())
    with pytest.raises(CanonicalizationError):
        canonical(1 + 2j)


def test_digest_semantics():
    a = np.arange(6, dtype="<f8").reshape(2, 3)
    assert Digest.of(a) == Digest.of(a.astype(">f8"))
    assert Digest.of(a) == Digest.of(np.asfortranarray(a))
    assert Digest.of(a) != Digest.of(a.reshape(3, 2))
    assert Digest.of(a) != Digest.of(a.astype("f4"))
    assert Digest.of(a).shape == (2, 3) and Digest.of(a).dtype == "<f8"
    assert Digest(Digest.of(a).sha256) == Digest.of(a)
    with pytest.raises(CanonicalizationError):
        Digest.of(np.array([object()]))


json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text().filter(lambda k: not (k.startswith("__") and k.endswith("__"))), children, max_size=4),
    max_leaves=20,
)


@given(st.dictionaries(st.text(max_size=5).filter(lambda k: not k.startswith("__")), json_values, max_size=6))
def test_key_independent_of_insertion_order(d):
    reordered = dict(reversed(list(d.items())))
    assert sha256(canonical(d)) == sha256(canonical(reordered))


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_floats_round_trip_exactly(x):
    c = canonical(x)
    assert json.loads(dumps(c)) == x
    assert canonical(np.float64(x)) == c


@given(st.integers(min_value=-(2**63), max_value=2**63 - 1))
def test_numpy_and_python_ints_agree(i):
    assert dumps(canonical(np.int64(i))) == dumps(canonical(i))


@given(json_values)
def test_canonical_is_idempotent(v):
    c = canonical(v)
    assert canonical(untag(c)) == c
