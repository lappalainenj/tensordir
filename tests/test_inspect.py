import numpy as np
import pytest
from jaxtyping import Float

import tensordir as td


class Optim(td.Spec):
    lr: float = 0.1
    betas: tuple[float, ...] = (0.9, 0.999)


class Net(td.Spec):
    width: int = 8
    optim: Optim = td.field(default_factory=Optim)


class Train(td.Record):
    __record_name__ = "InspectTrain"

    class Spec(td.Spec):
        seed: int
        net: Net = td.field(default_factory=Net)
        device: str = td.field("cpu", hashed=False)
        prior: float = td.field(1.0, since=2)

    weights: Float[np.ndarray, "n_layers n_units"]

    def build(self, spec):
        self.weights = np.zeros((2, spec.net.width))
        self.append("loss", np.array([1.0, 0.5]))
        self["summary"] = {"seed": spec.seed}


def test_replace_with_dotted_names():
    spec = Train.Spec(seed=0)
    changed = spec.replace(seed=1, net__optim__lr=0.01, **{"net.width": 16})
    assert (changed.seed, changed.net.width, changed.net.optim.lr) == (1, 16, 0.01)
    assert spec.net.optim.lr == 0.1  # frozen: the original is untouched
    both = spec.replace(net=Net(width=4), net__optim__lr=0.5)
    assert (both.net.width, both.net.optim.lr) == (4, 0.5)
    with pytest.raises(TypeError, match="not a nested spec"):
        spec.replace(seed__x=1)
    with pytest.raises(TypeError):
        spec.replace(nope=1)


def test_flat():
    flat = Train.Spec(seed=3).flat()
    assert flat == {
        "device": "cpu", "net.optim.betas": [0.9, 0.999], "net.optim.lr": 0.1, "net.width": 8, "prior": 1.0, "seed": 3,
    }


def test_diff():
    a = Train.Spec(seed=0)
    b = a.replace(seed=1, net__optim__lr=0.01, device="cuda")
    d = td.diff(a, b)
    assert d.changed == {"device": ("cpu", "cuda"), "net.optim.lr": (0.1, 0.01), "seed": (0, 1)}
    assert not d.added and not d.removed and bool(d)
    assert a.diff(b, hashed=True).names() == ["net.optim.lr", "seed"]  # device is not hashed
    assert not td.diff(a, Train.Spec(seed=0)) and repr(td.diff(a, a)) == "SpecDiff(no differences)"
    assert repr(d) == "SpecDiff(\n  ~ device: 'cpu' -> 'cuda'\n  ~ net.optim.lr: 0.1 -> 0.01\n  ~ seed: 0 -> 1\n)"
    page = d._repr_html_()
    assert "<table" in page and "net.optim.lr" in page and "&#x27;cuda&#x27;" in page
    loose = td.diff({"a": 1, "b": {"c": [1, 2]}}, {"b": {"c": [1, 3]}, "d": None})
    assert loose.removed == {"a": 1} and loose.added == {"d": None}
    assert loose.changed == {"b.c": ("[1, 2]", "[1, 3]")}
    assert loose.lines() == ["- a: 1", "~ b.c: '[1, 2]' -> '[1, 3]'", "+ d: None"]
    with pytest.raises(TypeError):
        td.diff(a, 3)


def test_diff_records_and_views(root):
    r0, r1 = Train.get({"seed": 0}), Train.get({"seed": 1, "device": "cuda"})
    assert td.diff(r0, r1).names() == ["device", "seed"]
    assert td.diff(td.View(r0), r1, hashed=True).names() == ["seed"]
    assert td.diff(r0, r0.spec.to_dict()).names() == []
    untyped = td.Record.at(r0.path)
    assert type(untyped) is Train  # registered class
    with pytest.raises(TypeError, match="typed"):
        td.diff(td.Record._new(r0.path, key=r0.key, managed_root=None), r1, hashed=True)


def test_spec_html():
    page = Train.Spec(seed=2)._repr_html_()
    assert "<table" in page and "net.optim.lr" in page and "not hashed" in page and "since 2" in page


def test_record_tree_and_html(root, capsys):
    record = Train.get({"seed": 0})
    record.memory.cache(lambda x: x)(1)
    capsys.readouterr()
    record.tree()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"{record!r} {record.path}"
    assert lines[1].split() == ["├──", "loss", "ndarray", "float64", "(2,)", "appendable"]
    assert lines[3].split() == ["├──", "weights", "ndarray", "float64", "(2,", "8)", "[layer,", "unit]"]
    assert lines[-1].split() == ["└──", "__cache__/", "joblib"]
    page = record._repr_html_()
    assert "<b>Train</b>" in page and "<i>done</i>" in page
    assert "net.optim.lr" in page and "[layer, unit]" in page and "summary" in page
    assert "<b>View</b> of" in td.View(record)._repr_html_()
    absent = Train.locate({"seed": 5})
    assert "<i>absent</i>" in absent._repr_html_()
    absent.tree()
    assert capsys.readouterr().out.strip() == f"{absent!r} {absent.path}"


def test_failed_record_html(root):
    class Broken(td.Record):
        __record_name__ = "InspectBroken"

        def build(self, spec):
            raise ValueError("bad input")

    with pytest.raises(td.BuildFailed):
        Broken.get()
    assert "ValueError: bad input" in Broken.locate()._repr_html_()
